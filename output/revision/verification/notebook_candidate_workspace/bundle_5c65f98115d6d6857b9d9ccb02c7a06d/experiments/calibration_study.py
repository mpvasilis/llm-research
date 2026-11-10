"""Development-only checkpoint calibration companion for the revision study.

The calibration manifest is derived from a frozen full-study protocol.  It uses
only development prompt families, never consults or relaxes the evaluation
human-review gate, and requires an explicit 256- or 512-token cap.  Planning is
read-only; a manifest must be frozen separately before an explicitly authorized
generation run can start.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import full_revision_study as core
except ImportError:
    import full_revision_study as core


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_RUNNER_VERSION = "revision-study-calibration-runner-v1"
MANIFEST_SCHEMA_VERSION = "revision-study-calibration-manifest-v1"
RUN_SCHEMA_VERSION = "revision-study-calibration-run-v1"
DEFAULT_PARENT_PROTOCOL = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "calibration_study"
CHECKPOINT_ALIASES = (
    "1b_sft", "1b_dpo", "1b_instruct",
    "7b_sft", "7b_dpo", "7b_instruct",
)
STAGES = ("sft", "dpo", "instruct")
DOMAINS = ("health", "everyday")
STAGE_INDICES = {
    "sft": (0, 1),
    "dpo": (2, 3),
    "instruct": (4, 0),
}
ALLOWED_CAPS = (256, 512)
EXPECTED_RESPONSES = 96
STOP_REQUESTED = False


def _source_hashes() -> dict[str, str]:
    return {
        "experiments/calibration_study.py": core.sha256_file(Path(__file__)),
        "experiments/full_revision_study.py": core.sha256_file(Path(core.__file__)),
    }


def _load_parent_and_prompts(parent_protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    parent = core.load_protocol(parent_protocol_path)
    corpus, prompts, corpus_hash = core.load_corpus()
    declared = parent.get("prompt_corpus", {}).get("sha256")
    if declared != corpus_hash:
        raise ValueError("Parent protocol prompt-corpus hash does not match the frozen corpus file")
    if parent.get("study_version") != corpus.get("study_version"):
        raise ValueError("Parent protocol and prompt corpus study versions differ")
    if set(parent.get("checkpoints", {})) != set(CHECKPOINT_ALIASES):
        raise ValueError("Parent protocol must contain exactly the six retained checkpoints")
    required_checkpoint_fields = {
        "checkpoint_alias", "model_id", "model_revision", "tokenizer_revision",
        "model_hash", "serialization_id", "chat_template_sha256",
    }
    for alias in CHECKPOINT_ALIASES:
        checkpoint = parent["checkpoints"][alias]
        missing = sorted(field for field in required_checkpoint_fields if not checkpoint.get(field))
        if checkpoint.get("checkpoint_alias") != alias or missing:
            raise ValueError(f"Incomplete pinned checkpoint identity for {alias}: {missing}")
    return parent, prompts, corpus_hash


def _development_selection(prompts: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    evaluation = [row for row in prompts if row.get("split") == "evaluation"]
    development = [row for row in prompts if row.get("split") == "development"]
    if len(evaluation) != 160 or len(development) != 40:
        raise ValueError("Frozen corpus must contain 160 evaluation and 40 development prompts")
    family_order: dict[str, list[str]] = {}
    for domain in DOMAINS:
        family_ids = sorted({
            row["scenario_family_id"] for row in development
            if row.get("condition", {}).get("domain") == domain
        })
        if len(family_ids) != 5:
            raise ValueError(f"Expected five development families in {domain}, found {len(family_ids)}")
        family_order[domain] = family_ids
    allocations = {
        stage: {
            domain: [family_order[domain][index] for index in STAGE_INDICES[stage]]
            for domain in DOMAINS
        }
        for stage in STAGES
    }
    covered = {
        family_id
        for stage in STAGES
        for domain in DOMAINS
        for family_id in allocations[stage][domain]
    }
    all_development = {row["scenario_family_id"] for row in development}
    if covered != all_development:
        raise AssertionError("Stage allocation must cover all ten development families")
    return family_order, allocations


def _manifest_core(parent_protocol_path: Path, max_new_tokens: int) -> dict[str, Any]:
    if max_new_tokens not in ALLOWED_CAPS:
        raise ValueError("max_new_tokens must be explicitly set to 256 or 512")
    parent, prompts, corpus_hash = _load_parent_and_prompts(parent_protocol_path)
    family_order, allocations = _development_selection(prompts)
    generation_config = {**core.PRIMARY_GENERATION, "max_new_tokens": max_new_tokens}
    development_prompts = [row for row in prompts if row["split"] == "development"]
    grid: list[dict[str, Any]] = []
    for alias in CHECKPOINT_ALIASES:
        stage = alias.split("_", 1)[1]
        selected_families = {
            family_id for domain in DOMAINS for family_id in allocations[stage][domain]
        }
        selected_prompts = [
            row for row in development_prompts
            if row["scenario_family_id"] in selected_families
        ]
        entries = core.grid_entries(
            selected_prompts,
            parent["checkpoints"],
            (alias,),
            (0,),
            generation_config,
            corpus_hash,
        )
        grid.extend(entries)

    _validate_grid_shape(grid, allocations, max_new_tokens)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "calibration_runner_version": CALIBRATION_RUNNER_VERSION,
        "classification": "exploratory_development_only_not_confirmatory_analysis",
        "purpose": "guide refinement and runtime/timing calibration only",
        "parent_protocol": {
            "protocol_id": parent["protocol_id"],
            "sha256": core.sha256_file(parent_protocol_path),
            "protocol_schema_version": parent["protocol_schema_version"],
            "study_version": parent["study_version"],
            "runner_version": parent["runner_version"],
        },
        "prompt_corpus": {
            "sha256": corpus_hash,
            "corpus_version": parent["prompt_corpus"]["corpus_version"],
        },
        "checkpoints": {alias: parent["checkpoints"][alias] for alias in CHECKPOINT_ALIASES},
        "selection_rule": {
            "domain_order": list(DOMAINS),
            "within_domain_order": "lexicographic sort of the five development scenario_family_id values",
            "stage_indices": {stage: list(STAGE_INDICES[stage]) for stage in STAGES},
            "same_stage_families_shared_across_model_sizes": True,
            "logical_seeds": [0],
        },
        "development_family_order": family_order,
        "stage_allocations": allocations,
        "generation_config": generation_config,
        "max_new_tokens_decision": max_new_tokens,
        "parent_primary_max_new_tokens": parent["primary_generation"]["max_new_tokens"],
        "grid": grid,
        "expected_responses": EXPECTED_RESPONSES,
    }


def _validate_grid_shape(
    grid: list[dict[str, Any]],
    allocations: dict[str, dict[str, list[str]]],
    max_new_tokens: int,
) -> None:
    if len(grid) != EXPECTED_RESPONSES:
        raise ValueError(f"Calibration grid must contain 96 responses, found {len(grid)}")
    ids = [entry.get("response_id") for entry in grid]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Calibration response IDs must be nonblank and unique")
    if any(entry.get("split") != "development" for entry in grid):
        raise ValueError("Evaluation-family leakage into calibration grid")
    if any(entry.get("logical_seed") != 0 for entry in grid):
        raise ValueError("Calibration grid must use logical seed 0 only")
    if any(entry.get("response_identity", {}).get("generation_config", {}).get("max_new_tokens") != max_new_tokens for entry in grid):
        raise ValueError("Calibration grid contains a response with the wrong token cap")
    aliases = Counter(entry.get("checkpoint_alias") for entry in grid)
    if aliases != Counter({alias: 16 for alias in CHECKPOINT_ALIASES}):
        raise ValueError(f"Each checkpoint must have 16 calibration responses: {dict(aliases)}")
    for alias in CHECKPOINT_ALIASES:
        stage = alias.split("_", 1)[1]
        rows = [entry for entry in grid if entry["checkpoint_alias"] == alias]
        expected_families = {
            family_id for domain in DOMAINS for family_id in allocations[stage][domain]
        }
        if {entry["scenario_family_id"] for entry in rows} != expected_families:
            raise ValueError(f"Wrong family allocation for {alias}")
        family_counts = Counter(entry["scenario_family_id"] for entry in rows)
        if family_counts != Counter({family_id: 4 for family_id in expected_families}):
            raise ValueError(f"Every selected family must contribute all four variants for {alias}")
        cells = Counter(
            (entry["condition"]["domain"], entry["condition"]["emotional_framing"], entry["condition"]["request_type"])
            for entry in rows
        )
        if any(cells[(domain, emotion, request)] != 2 for domain in DOMAINS for emotion in (False, True) for request in ("information", "advice")):
            raise ValueError(f"Unbalanced 2x2 development cells for {alias}")


def build_manifest(parent_protocol_path: Path, max_new_tokens: int) -> dict[str, Any]:
    core_value = _manifest_core(parent_protocol_path, max_new_tokens)
    return {
        **core_value,
        "calibration_manifest_id": core.stable_id("calibration_manifest", core_value, 32),
    }


def freeze_manifest(parent_protocol_path: Path, max_new_tokens: int, output: Path) -> dict[str, Any]:
    manifest = build_manifest(parent_protocol_path, max_new_tokens)
    if manifest["max_new_tokens_decision"] != manifest["parent_primary_max_new_tokens"]:
        raise ValueError("Calibration cap must match the parent protocol primary cap before manifest freezing")
    core.atomic_write_json(output, manifest, immutable=True)
    return manifest


def load_manifest(path: Path, parent_protocol_path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration manifest schema")
    claimed_id = manifest.get("calibration_manifest_id")
    manifest_core = {key: value for key, value in manifest.items() if key != "calibration_manifest_id"}
    if claimed_id != core.stable_id("calibration_manifest", manifest_core, 32):
        raise ValueError("Calibration manifest content does not match its identity")
    _validate_grid_shape(
        manifest.get("grid", []),
        manifest.get("stage_allocations", {}),
        manifest.get("max_new_tokens_decision"),
    )
    if manifest.get("max_new_tokens_decision") != manifest.get("parent_primary_max_new_tokens"):
        raise ValueError("Frozen calibration cap does not match the parent protocol primary cap")
    expected = build_manifest(parent_protocol_path, manifest.get("max_new_tokens_decision"))
    if core.canonical_json(manifest) != core.canonical_json(expected):
        raise ValueError("Calibration manifest does not match the pinned parent protocol and corpus")
    return manifest


def plan(parent_protocol_path: Path, max_new_tokens: int) -> dict[str, Any]:
    manifest = build_manifest(parent_protocol_path, max_new_tokens)
    return {
        "status": "candidate_plan_not_frozen",
        "execute_required": True,
        "max_new_tokens": max_new_tokens,
        "responses": len(manifest["grid"]),
        "checkpoints": len(manifest["checkpoints"]),
        "development_families_covered": len({entry["scenario_family_id"] for entry in manifest["grid"]}),
        "evaluation_responses": sum(entry["split"] == "evaluation" for entry in manifest["grid"]),
        "candidate_manifest_id": manifest["calibration_manifest_id"],
        "stage_allocations": manifest["stage_allocations"],
    }


def _response_with_calibration_provenance(
    entry: dict[str, Any],
    checkpoint: dict[str, Any],
    model: Any,
    tokenizer: Any,
    execution: dict[str, Any],
    dtype: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    row = core.entry_to_response(
        entry, checkpoint, model, tokenizer, execution, dtype,
        manifest["calibration_manifest_id"],
    )
    row.update({
        "calibration_manifest_id": manifest["calibration_manifest_id"],
        "calibration_runner_version": CALIBRATION_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "calibration_classification": manifest["classification"],
        "checkpoint_alias": entry["checkpoint_alias"],
    })
    return row


def validate_response(
    row: dict[str, Any],
    expected: dict[str, Any],
    manifest: dict[str, Any],
    execution: dict[str, Any],
) -> None:
    core.validate_response(row, expected, manifest["calibration_manifest_id"], execution)
    required = {
        "calibration_manifest_id": manifest["calibration_manifest_id"],
        "calibration_runner_version": CALIBRATION_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "calibration_classification": manifest["classification"],
        "checkpoint_alias": expected["checkpoint_alias"],
    }
    for field, value in required.items():
        if row.get(field) != value:
            raise ValueError(f"Calibration response {field} mismatch")
    if row.get("split") != "development":
        raise ValueError("Calibration response is not development-only")
    checkpoint = manifest["checkpoints"][expected["checkpoint_alias"]]
    for field in ("model_id", "model_revision", "tokenizer_revision", "model_hash", "serialization_id"):
        if row.get(field) != checkpoint[field]:
            raise ValueError(f"Calibration response pinned {field} mismatch")


def _run_config(
    manifest_path: Path,
    parent_protocol_path: Path,
    manifest: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    config_core = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "calibration_runner_version": CALIBRATION_RUNNER_VERSION,
        "calibration_manifest_id": manifest["calibration_manifest_id"],
        "calibration_manifest_sha256": core.sha256_file(manifest_path),
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": core.sha256_file(parent_protocol_path),
        "classification": manifest["classification"],
        "source_sha256": _source_hashes(),
        "execution_profile": execution,
        "response_ids": [entry["response_id"] for entry in manifest["grid"]],
        "expected_responses": EXPECTED_RESPONSES,
    }
    return {**config_core, "run_id": core.stable_id("calibration_run", config_core, 32)}


def prepare_run(
    manifest_path: Path,
    parent_protocol_path: Path,
    output_root: Path,
    execution: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    manifest = load_manifest(manifest_path, parent_protocol_path)
    config = _run_config(manifest_path, parent_protocol_path, manifest, execution)
    run_dir = output_root / "runs" / config["run_id"]
    core.atomic_write_json(run_dir / "run_config.json", config, immutable=True)
    core.atomic_copy_bytes(manifest_path, run_dir / "calibration_manifest.json", immutable=True)
    core.atomic_copy_bytes(parent_protocol_path, run_dir / "parent_protocol.json", immutable=True)
    return manifest, run_dir


def _signal_handler(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_generation(
    manifest_path: Path,
    parent_protocol_path: Path,
    output_root: Path,
    device_request: str,
    local_files_only: bool,
    torch_threads: int | None,
) -> dict[str, Any]:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    execution = core.execution_profile(device_request, torch_threads)
    manifest, run_dir = prepare_run(manifest_path, parent_protocol_path, output_root, execution)
    grid = manifest["grid"]
    expected = {entry["response_id"]: entry for entry in grid}
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    paths = list(response_dir.glob("*.json"))
    unexpected = {path.stem for path in paths} - set(expected)
    if unexpected:
        raise ValueError(f"Cache contains unexpected response IDs: {sorted(unexpected)[:5]}")
    completed: set[str] = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if path.stem in completed:
            raise ValueError(f"Duplicate cached response ID: {path.stem}")
        validate_response(row, expected[path.stem], manifest, execution)
        completed.add(path.stem)
    pending = [entry for entry in grid if entry["response_id"] not in completed]
    print(f"run={run_dir.name} complete={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        return {"run_dir": str(run_dir), "completed": EXPECTED_RESPONSES, "new_responses": 0, "remaining": 0, "status": "complete"}

    attempt_id = f"attempt_{time.time_ns()}"
    progress_path = run_dir / "progress.jsonl"
    core.append_jsonl_durable(progress_path, {
        "event": "attempt_started", "attempt_id": attempt_id, "at": core.utc_now(),
        "complete_before_attempt": len(completed), "pending_before_attempt": len(pending),
        "execution_profile_id": execution["execution_profile_id"],
    })
    prior_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            prior_handlers[signum] = signal.signal(signum, _signal_handler)
        except (ValueError, OSError):
            pass
    new_count = 0
    try:
        for alias in CHECKPOINT_ALIASES:
            checkpoint_pending = [entry for entry in pending if entry["checkpoint_alias"] == alias]
            if not checkpoint_pending or STOP_REQUESTED:
                continue
            checkpoint = manifest["checkpoints"][alias]
            load_started = time.monotonic()
            model, tokenizer, dtype = core.load_checkpoint(checkpoint, execution, local_files_only)
            load_seconds = time.monotonic() - load_started
            core.append_jsonl_durable(progress_path, {
                "event": "checkpoint_loaded", "attempt_id": attempt_id, "at": core.utc_now(),
                "checkpoint_alias": alias, "model_load_seconds": load_seconds,
                "pending_for_checkpoint": len(checkpoint_pending), "dtype": dtype,
            })
            for entry in checkpoint_pending:
                row = _response_with_calibration_provenance(
                    entry, checkpoint, model, tokenizer, execution, dtype, manifest,
                )
                validate_response(row, entry, manifest, execution)
                core.atomic_write_json(response_dir / f"{entry['response_id']}.json", row, immutable=True)
                new_count += 1
                core.append_jsonl_durable(progress_path, {
                    "event": "response_checkpointed", "attempt_id": attempt_id, "at": core.utc_now(),
                    "response_id": entry["response_id"], "checkpoint_alias": alias,
                    "completed": len(completed) + new_count, "expected": EXPECTED_RESPONSES,
                    "generation_seconds": row["generation"]["timing"]["generation_seconds"],
                    "output_tokens": row["generation"]["output_tokens"],
                })
                if STOP_REQUESTED:
                    break
            del model
            if execution["device_class"] == "cuda":
                import torch
                torch.cuda.empty_cache()
            if STOP_REQUESTED:
                break
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
    complete = len(completed) + new_count
    result = {
        "run_dir": str(run_dir),
        "completed": complete,
        "new_responses": new_count,
        "remaining": EXPECTED_RESPONSES - complete,
        "status": "complete" if complete == EXPECTED_RESPONSES else "interrupted_resumable",
        "attempt_id": attempt_id,
    }
    core.append_jsonl_durable(progress_path, {"event": "attempt_finished", "at": core.utc_now(), **result})
    return result


def _validated_run(
    manifest_path: Path,
    parent_protocol_path: Path,
    run_dir: Path,
    require_complete: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest(manifest_path, parent_protocol_path)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    execution = config.get("execution_profile")
    if not isinstance(execution, dict):
        raise ValueError("Calibration run lacks an execution profile")
    execution_core = {key: value for key, value in execution.items() if key != "execution_profile_id"}
    if execution.get("execution_profile_id") != core.stable_id("execution", execution_core, 32):
        raise ValueError("Calibration execution profile does not match its identity")
    expected_config = _run_config(manifest_path, parent_protocol_path, manifest, execution)
    if core.canonical_json(config) != core.canonical_json(expected_config):
        raise ValueError("Calibration run config or pinned provenance has changed")
    if run_dir.name != config["run_id"]:
        raise ValueError("Calibration run directory does not match run identity")
    if core.sha256_file(run_dir / "calibration_manifest.json") != config["calibration_manifest_sha256"]:
        raise ValueError("Calibration manifest snapshot hash mismatch")
    if core.sha256_file(run_dir / "parent_protocol.json") != config["parent_protocol_sha256"]:
        raise ValueError("Parent protocol snapshot hash mismatch")

    expected = {entry["response_id"]: entry for entry in manifest["grid"]}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for path in (run_dir / "responses").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            response_id = row.get("response_id")
            if response_id in seen:
                raise ValueError(f"duplicate response_id {response_id}")
            seen.add(response_id)
            if path.stem != response_id:
                raise ValueError(f"filename does not match response_id {response_id}")
            if response_id not in expected:
                raise ValueError(f"unexpected response_id {response_id}")
            validate_response(row, expected[response_id], manifest, execution)
            rows.append(row)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    missing = sorted(set(expected) - seen)
    if require_complete and missing:
        errors.append(f"missing {len(missing)} expected responses")
    if errors:
        raise ValueError("Calibration run verification failed:\n- " + "\n- ".join(errors[:20]))
    rows.sort(key=lambda row: row["response_id"])
    return manifest, rows, config


def verify_run(manifest_path: Path, parent_protocol_path: Path, run_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    manifest, rows, _config = _validated_run(manifest_path, parent_protocol_path, run_dir, require_complete)
    counts = Counter(row["generation"]["stopping"]["reason"] for row in rows)
    return {
        "status": "complete" if len(rows) == EXPECTED_RESPONSES else "partial_valid",
        "classification": manifest["classification"],
        "expected_responses": EXPECTED_RESPONSES,
        "verified_responses": len(rows),
        "missing_responses": EXPECTED_RESPONSES - len(rows),
        "unique_response_ids": len({row["response_id"] for row in rows}),
        "stopping_reasons": dict(sorted(counts.items())),
        "cap_rate": sum(row["generation"]["stopping"]["cap_reached"] for row in rows) / len(rows) if rows else None,
    }


def export_run(manifest_path: Path, parent_protocol_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    manifest, rows, config = _validated_run(manifest_path, parent_protocol_path, run_dir, True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temp, output)
    return {
        "status": "complete_calibration_export",
        "classification": manifest["classification"],
        "run_id": config["run_id"],
        "responses": len(rows),
        "output": str(output),
        "output_sha256": core.sha256_file(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    plan_parser.add_argument("--max-new-tokens", type=int, choices=ALLOWED_CAPS, required=True)
    freeze = sub.add_parser("freeze-manifest")
    freeze.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    freeze.add_argument("--max-new-tokens", type=int, choices=ALLOWED_CAPS, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--torch-threads", type=int)
    run.add_argument("--execute", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    verify.add_argument("--run-dir", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = plan(args.parent_protocol, args.max_new_tokens)
    elif args.command == "freeze-manifest":
        manifest = freeze_manifest(args.parent_protocol, args.max_new_tokens, args.output)
        result = {
            "status": "frozen_exploratory_calibration_manifest",
            "output": str(args.output),
            "calibration_manifest_id": manifest["calibration_manifest_id"],
            "responses": EXPECTED_RESPONSES,
            "max_new_tokens": manifest["max_new_tokens_decision"],
        }
    elif args.command == "run":
        if not args.execute:
            manifest = load_manifest(args.manifest, args.parent_protocol)
            result = {
                "status": "dry_run",
                "execute_required": True,
                "calibration_manifest_id": manifest["calibration_manifest_id"],
                "responses": EXPECTED_RESPONSES,
                "max_new_tokens": manifest["max_new_tokens_decision"],
            }
        else:
            result = run_generation(
                args.manifest, args.parent_protocol, args.output_root,
                args.device, args.local_files_only, args.torch_threads,
            )
    elif args.command == "verify":
        result = verify_run(args.manifest, args.parent_protocol, args.run_dir)
    elif args.command == "export":
        result = export_run(args.manifest, args.parent_protocol, args.run_dir, args.output)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

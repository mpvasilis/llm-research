"""96-response development calibration for the extended revision study."""
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
    from . import extended_revision_study as core
except ImportError:  # pragma: no cover
    import extended_revision_study as core


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_RUNNER_VERSION = "extended-revision-study-calibration-runner-v1"
MANIFEST_SCHEMA_VERSION = "extended-revision-study-calibration-manifest-v1"
RUN_SCHEMA_VERSION = "extended-revision-study-calibration-run-v1"
DEFAULT_PARENT_PROTOCOL = ROOT / "output" / "revision" / "extended_study" / "full_study" / "protocol_v1.json"
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "extended_study" / "calibration_study"
CHECKPOINT_ALIASES = (
    "1b_sft", "1b_dpo", "1b_instruct",
    "7b_sft", "7b_dpo", "7b_instruct",
)
STAGES = ("sft", "dpo", "instruct")
DOMAINS = ("health", "everyday")
STAGE_INDICES = {"sft": (0, 1), "dpo": (2, 3), "instruct": (4, 0)}
EXPECTED_RESPONSES = 96
STOP_REQUESTED = False


def _source_hashes() -> dict[str, str]:
    return {
        "experiments/extended_calibration_study.py": core.sha256_file(Path(__file__)),
        **core.source_hashes(),
    }


def _load_parent_and_prompts(parent_protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    parent_protocol_path = core.fs_path(parent_protocol_path)
    parent = core.load_protocol(parent_protocol_path)
    _corpus, prompts, corpus_hash = core.load_corpus()
    if parent["prompt_corpus"]["sha256"] != corpus_hash:
        raise ValueError("Parent protocol and prompt corpus differ")
    if parent.get("primary_cap", 0) <= 512:
        raise ValueError("Calibration requires an explicit higher-budget successor parent")
    if set(parent.get("checkpoints", {})) != set(CHECKPOINT_ALIASES):
        raise ValueError("Parent must contain exactly the six retained checkpoints")
    return parent, prompts, corpus_hash


def _development_selection(prompts: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    evaluation = [row for row in prompts if row["split"] == "evaluation"]
    development = [row for row in prompts if row["split"] == "development"]
    if len(evaluation) != 160 or len(development) != 40:
        raise ValueError("Frozen corpus must contain the exact 160/40 prompt split")
    family_order = {
        domain: sorted({row["scenario_family_id"] for row in development if row["condition"]["domain"] == domain})
        for domain in DOMAINS
    }
    if any(len(values) != 5 for values in family_order.values()):
        raise ValueError("Each domain must have five development families")
    allocations = {
        stage: {domain: [family_order[domain][index] for index in STAGE_INDICES[stage]] for domain in DOMAINS}
        for stage in STAGES
    }
    covered = {family for stage in STAGES for domain in DOMAINS for family in allocations[stage][domain]}
    if covered != {row["scenario_family_id"] for row in development}:
        raise ValueError("Stage rotation does not cover all development families")
    return family_order, allocations


def _validate_grid_shape(grid: list[dict[str, Any]], allocations: dict[str, dict[str, list[str]]], cap: int) -> None:
    if len(grid) != EXPECTED_RESPONSES or len({row.get("response_id") for row in grid}) != EXPECTED_RESPONSES:
        raise ValueError("Calibration grid must contain exactly 96 unique responses")
    if any(row.get("split") != "development" or row.get("logical_seed") != 0 for row in grid):
        raise ValueError("Calibration grid must be development-only at logical seed zero")
    if any(row.get("response_identity", {}).get("generation_config", {}).get("max_new_tokens") != cap for row in grid):
        raise ValueError("Calibration grid generation cap mismatch")
    for alias in CHECKPOINT_ALIASES:
        rows = [row for row in grid if row.get("checkpoint_alias") == alias]
        stage = alias.split("_", 1)[1]
        expected_families = {family for domain in DOMAINS for family in allocations[stage][domain]}
        if len(rows) != 16 or Counter(row["scenario_family_id"] for row in rows) != Counter({family: 4 for family in expected_families}):
            raise ValueError(f"Calibration allocation mismatch for {alias}")
        cells = Counter((row["condition"]["domain"], row["condition"]["emotional_framing"], row["condition"]["request_type"]) for row in rows)
        if any(cells[(domain, emotion, request)] != 2 for domain in DOMAINS for emotion in (False, True) for request in ("information", "advice")):
            raise ValueError(f"Unbalanced calibration cells for {alias}")


def build_manifest(parent_protocol_path: Path, max_new_tokens: int) -> dict[str, Any]:
    parent_protocol_path = core.fs_path(parent_protocol_path)
    parent, prompts, corpus_hash = _load_parent_and_prompts(parent_protocol_path)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens != parent["primary_cap"]:
        raise ValueError("max_new_tokens must equal the explicit higher cap in the parent protocol")
    family_order, allocations = _development_selection(prompts)
    generation = dict(parent["primary_generation"])
    development = [row for row in prompts if row["split"] == "development"]
    grid: list[dict[str, Any]] = []
    for alias in CHECKPOINT_ALIASES:
        stage = alias.split("_", 1)[1]
        selected = {family for domain in DOMAINS for family in allocations[stage][domain]}
        selected_prompts = [row for row in development if row["scenario_family_id"] in selected]
        grid.extend(core.grid_entries(selected_prompts, parent["checkpoints"], (alias,), (0,), generation, corpus_hash))
    _validate_grid_shape(grid, allocations, max_new_tokens)
    manifest_core = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "calibration_runner_version": CALIBRATION_RUNNER_VERSION,
        "classification": "exploratory_development_only_not_confirmatory_analysis",
        "purpose": "guide refinement and runtime/timing calibration only",
        "parent_protocol": {
            "protocol_id": parent["protocol_id"], "sha256": core.sha256_file(parent_protocol_path),
            "protocol_schema_version": parent["protocol_schema_version"],
            "orchestration_version": parent["orchestration_version"],
            "study_version": parent["study_version"], "runner_version": parent["runner_version"],
        },
        "source_sha256": _source_hashes(),
        "prompt_corpus": {"sha256": corpus_hash, "corpus_version": parent["prompt_corpus"]["corpus_version"]},
        "checkpoints": {alias: parent["checkpoints"][alias] for alias in CHECKPOINT_ALIASES},
        "selection_rule": {"domain_order": list(DOMAINS), "within_domain_order": "lexicographic sort of five development scenario_family_id values", "stage_indices": {stage: list(STAGE_INDICES[stage]) for stage in STAGES}, "same_stage_families_shared_across_model_sizes": True, "logical_seeds": [0]},
        "development_family_order": family_order,
        "stage_allocations": allocations,
        "generation_config": generation,
        "max_new_tokens_decision": max_new_tokens,
        "parent_primary_max_new_tokens": parent["primary_cap"],
        "parent_sensitivity_max_new_tokens": parent["sensitivity_cap"],
        "grid": grid,
        "expected_responses": EXPECTED_RESPONSES,
    }
    return {**manifest_core, "calibration_manifest_id": core.stable_id("extended_calibration_manifest", manifest_core, 32)}


def freeze_manifest(parent_protocol_path: Path, max_new_tokens: int, output: Path) -> dict[str, Any]:
    parent_protocol_path = core.fs_path(parent_protocol_path)
    output = core.fs_path(output)
    manifest = build_manifest(parent_protocol_path, max_new_tokens)
    core.atomic_write_json(output, manifest, immutable=True)
    return manifest


def load_manifest(path: Path, parent_protocol_path: Path) -> dict[str, Any]:
    path = core.fs_path(path)
    parent_protocol_path = core.fs_path(parent_protocol_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported extended calibration manifest schema")
    claimed = manifest.get("calibration_manifest_id")
    body = {key: value for key, value in manifest.items() if key != "calibration_manifest_id"}
    if claimed != core.stable_id("extended_calibration_manifest", body, 32):
        raise ValueError("Calibration manifest content does not match its identity")
    _validate_grid_shape(manifest.get("grid", []), manifest.get("stage_allocations", {}), manifest.get("max_new_tokens_decision"))
    expected = build_manifest(parent_protocol_path, manifest.get("max_new_tokens_decision"))
    if core.canonical_json(manifest) != core.canonical_json(expected):
        raise ValueError("Calibration manifest does not reconstruct from pinned dependencies")
    return manifest


def plan(parent_protocol_path: Path, max_new_tokens: int) -> dict[str, Any]:
    parent_protocol_path = core.fs_path(parent_protocol_path)
    manifest = build_manifest(parent_protocol_path, max_new_tokens)
    return {"status": "candidate_plan_not_frozen", "execute_required": True, "max_new_tokens": max_new_tokens, "responses": EXPECTED_RESPONSES, "checkpoints": 6, "development_families_covered": 10, "evaluation_responses": 0, "candidate_manifest_id": manifest["calibration_manifest_id"], "stage_allocations": manifest["stage_allocations"]}


def _response_with_calibration_provenance(entry: dict[str, Any], checkpoint: dict[str, Any], model: Any, tokenizer: Any, execution: dict[str, Any], dtype: str, manifest: dict[str, Any]) -> dict[str, Any]:
    row = core.entry_to_response(entry, checkpoint, model, tokenizer, execution, dtype, manifest["calibration_manifest_id"])
    row.update({"calibration_manifest_id": manifest["calibration_manifest_id"], "calibration_runner_version": CALIBRATION_RUNNER_VERSION, "parent_protocol_id": manifest["parent_protocol"]["protocol_id"], "parent_protocol_sha256": manifest["parent_protocol"]["sha256"], "calibration_classification": manifest["classification"], "checkpoint_alias": entry["checkpoint_alias"]})
    return row


def validate_response(row: dict[str, Any], expected: dict[str, Any], manifest: dict[str, Any], execution: dict[str, Any]) -> None:
    core.validate_response(row, expected, manifest["calibration_manifest_id"], execution)
    required = {"calibration_manifest_id": manifest["calibration_manifest_id"], "calibration_runner_version": CALIBRATION_RUNNER_VERSION, "parent_protocol_id": manifest["parent_protocol"]["protocol_id"], "parent_protocol_sha256": manifest["parent_protocol"]["sha256"], "calibration_classification": manifest["classification"], "checkpoint_alias": expected["checkpoint_alias"]}
    for field, value in required.items():
        if row.get(field) != value:
            raise ValueError(f"Calibration response {field} mismatch")
    if row.get("split") != "development":
        raise ValueError("Calibration response is not development-only")
    checkpoint = manifest["checkpoints"][expected["checkpoint_alias"]]
    for field in ("model_id", "model_revision", "tokenizer_revision", "model_hash", "serialization_id"):
        if row.get(field) != checkpoint[field]:
            raise ValueError(f"Calibration response pinned {field} mismatch")


def _run_config(manifest_path: Path, parent_protocol_path: Path, manifest: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    manifest_path = core.fs_path(manifest_path)
    parent_protocol_path = core.fs_path(parent_protocol_path)
    value = {"run_schema_version": RUN_SCHEMA_VERSION, "calibration_runner_version": CALIBRATION_RUNNER_VERSION, "calibration_manifest_id": manifest["calibration_manifest_id"], "calibration_manifest_sha256": core.sha256_file(manifest_path), "parent_protocol_id": manifest["parent_protocol"]["protocol_id"], "parent_protocol_sha256": core.sha256_file(parent_protocol_path), "classification": manifest["classification"], "source_sha256": _source_hashes(), "execution_profile": execution, "response_ids": [entry["response_id"] for entry in manifest["grid"]], "expected_responses": EXPECTED_RESPONSES}
    return {**value, "run_id": core.stable_id("extended_calibration_run", value, 32)}


def prepare_run(manifest_path: Path, parent_protocol_path: Path, output_root: Path, execution: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    manifest_path = core.fs_path(manifest_path)
    parent_protocol_path = core.fs_path(parent_protocol_path)
    output_root = core.fs_path(output_root)
    manifest = load_manifest(manifest_path, parent_protocol_path)
    config = _run_config(manifest_path, parent_protocol_path, manifest, execution)
    run_dir = output_root / "runs" / config["run_id"]
    core.atomic_write_json(run_dir / "run_config.json", config, immutable=True)
    core.atomic_copy_bytes(manifest_path, run_dir / "calibration_manifest.json", immutable=True)
    core.atomic_copy_bytes(parent_protocol_path, run_dir / "parent_protocol.json", immutable=True)
    source_evidence = core._materialize_evidence_root(parent_protocol_path.parent / core.load_protocol(parent_protocol_path)["freeze_evidence"]["relative_root"])
    relative_root = core.load_protocol(parent_protocol_path)["freeze_evidence"]["relative_root"]
    core._write_evidence_zip(source_evidence, run_dir / f"{relative_root}.zip")
    return manifest, run_dir


def _signal_handler(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_generation(manifest_path: Path, parent_protocol_path: Path, output_root: Path, device_request: str, local_files_only: bool, torch_threads: int | None) -> dict[str, Any]:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    manifest_path = core.fs_path(manifest_path)
    parent_protocol_path = core.fs_path(parent_protocol_path)
    output_root = core.fs_path(output_root)
    execution = core.execution_profile(device_request, torch_threads)
    manifest, run_dir = prepare_run(manifest_path, parent_protocol_path, output_root, execution)
    parent = core.load_protocol(parent_protocol_path)
    expected = {entry["response_id"]: entry for entry in manifest["grid"]}
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    for path in response_dir.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if path.stem in completed or path.stem not in expected:
            raise ValueError(f"Unexpected or duplicate cached calibration response: {path.name}")
        validate_response(row, expected[path.stem], manifest, execution)
        completed.add(path.stem)
    pending = [entry for entry in manifest["grid"] if entry["response_id"] not in completed]
    if not pending:
        return {"run_dir": core.display_path(run_dir), "completed": EXPECTED_RESPONSES, "new_responses": 0, "remaining": 0, "status": "complete"}
    attempt_id = f"attempt_{time.time_ns()}"
    started = core.utc_now()
    clock = time.monotonic()
    progress = run_dir / "progress.jsonl"
    core.append_jsonl_durable(progress, {"event": "attempt_started", "attempt_id": attempt_id, "at": started, "complete_before_attempt": len(completed), "pending_before_attempt": len(pending), "execution_profile_id": execution["execution_profile_id"]})
    handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            handlers[signum] = signal.signal(signum, _signal_handler)
        except (ValueError, OSError):
            pass
    new_count = 0
    try:
        for alias in CHECKPOINT_ALIASES:
            checkpoint_pending = [entry for entry in pending if entry["checkpoint_alias"] == alias]
            if not checkpoint_pending or STOP_REQUESTED:
                continue
            load_clock = time.monotonic()
            model, tokenizer, dtype = core.load_checkpoint(manifest["checkpoints"][alias], execution, local_files_only)
            load_seconds = time.monotonic() - load_clock
            preflight_clock = time.monotonic()
            preflight = core.context_preflight(parent, checkpoint_pending, model, tokenizer)
            preflight_seconds = time.monotonic() - preflight_clock
            core.append_jsonl_durable(progress, {"event": "checkpoint_loaded_and_context_validated", "attempt_id": attempt_id, "at": core.utc_now(), "checkpoint_alias": alias, "model_load_seconds": load_seconds, "context_preflight_seconds": preflight_seconds, "dtype": dtype, "context_preflight": preflight})
            for entry in checkpoint_pending:
                row = _response_with_calibration_provenance(entry, manifest["checkpoints"][alias], model, tokenizer, execution, dtype, manifest)
                validate_response(row, entry, manifest, execution)
                core.atomic_write_json(response_dir / f"{entry['response_id']}.json", row, immutable=True)
                new_count += 1
                core.append_jsonl_durable(progress, {"event": "response_checkpointed", "attempt_id": attempt_id, "at": core.utc_now(), "response_id": entry["response_id"], "checkpoint_alias": alias, "completed": len(completed) + new_count, "expected": EXPECTED_RESPONSES, "generation_seconds": row["generation"]["timing"]["generation_seconds"], "output_tokens": row["generation"]["output_tokens"], "max_new_tokens": row["generation"]["max_new_tokens"], "stop_reason": row["generation"]["stopping"]["reason"], "eos_reached": row["generation"]["stopping"]["eos_reached"], "cap_reached": row["generation"]["stopping"]["cap_reached"]})
                if STOP_REQUESTED:
                    break
            del model
            if execution["device_class"] == "cuda":
                import torch
                torch.cuda.empty_cache()
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    complete = len(completed) + new_count
    result = {"run_dir": core.display_path(run_dir), "completed": complete, "new_responses": new_count, "remaining": EXPECTED_RESPONSES - complete, "status": "complete" if complete == EXPECTED_RESPONSES else "interrupted_resumable", "attempt_id": attempt_id, "attempt_started_at": started, "attempt_finished_at": core.utc_now(), "attempt_wall_seconds": time.monotonic() - clock}
    core.atomic_write_json(run_dir / "attempts" / f"{attempt_id}.json", result, immutable=True)
    core.append_jsonl_durable(progress, {"event": "attempt_finished", **result})
    return result


def _validated_run(manifest_path: Path, parent_protocol_path: Path, run_dir: Path, require_complete: bool) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = core.fs_path(manifest_path)
    parent_protocol_path = core.fs_path(parent_protocol_path)
    run_dir = core.fs_path(run_dir)
    manifest = load_manifest(manifest_path, parent_protocol_path)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    execution = config.get("execution_profile")
    body = {key: value for key, value in execution.items() if key != "execution_profile_id"} if isinstance(execution, dict) else {}
    if not isinstance(execution, dict) or execution.get("execution_profile_id") != core.stable_id("execution", body, 32):
        raise ValueError("Calibration execution profile identity mismatch")
    if core.canonical_json(config) != core.canonical_json(_run_config(manifest_path, parent_protocol_path, manifest, execution)) or run_dir.name != config.get("run_id"):
        raise ValueError("Calibration run config or pinned provenance changed")
    if core.sha256_file(run_dir / "calibration_manifest.json") != config["calibration_manifest_sha256"] or core.sha256_file(run_dir / "parent_protocol.json") != config["parent_protocol_sha256"]:
        raise ValueError("Calibration snapshot hash mismatch")
    core.load_protocol(run_dir / "parent_protocol.json")
    expected = {entry["response_id"]: entry for entry in manifest["grid"]}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    for path in (run_dir / "responses").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            response_id = row.get("response_id")
            if response_id in seen or path.stem != response_id or response_id not in expected:
                raise ValueError("duplicate, misnamed, or unexpected response")
            seen.add(response_id)
            validate_response(row, expected[response_id], manifest, execution)
            rows.append(row)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    missing = set(expected) - seen
    if require_complete and missing:
        errors.append(f"missing {len(missing)} expected responses")
    if errors:
        raise ValueError("Calibration verification failed:\n- " + "\n- ".join(errors[:20]))
    rows.sort(key=lambda row: row["response_id"])
    return manifest, rows, config


def verify_run(manifest_path: Path, parent_protocol_path: Path, run_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    manifest_path = core.fs_path(manifest_path)
    parent_protocol_path = core.fs_path(parent_protocol_path)
    run_dir = core.fs_path(run_dir)
    manifest, rows, _config = _validated_run(manifest_path, parent_protocol_path, run_dir, require_complete)
    counts = Counter(row["generation"]["stopping"]["reason"] for row in rows)
    return {"status": "complete" if len(rows) == EXPECTED_RESPONSES else "partial_valid", "classification": manifest["classification"], "expected_responses": EXPECTED_RESPONSES, "verified_responses": len(rows), "missing_responses": EXPECTED_RESPONSES - len(rows), "unique_response_ids": len({row["response_id"] for row in rows}), "stopping_reasons": dict(sorted(counts.items())), "cap_rate": sum(row["generation"]["stopping"]["cap_reached"] for row in rows) / len(rows) if rows else None}


def export_run(manifest_path: Path, parent_protocol_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    manifest_path = core.fs_path(manifest_path)
    parent_protocol_path = core.fs_path(parent_protocol_path)
    run_dir = core.fs_path(run_dir)
    output = core.fs_path(output)
    manifest, rows, config = _validated_run(manifest_path, parent_protocol_path, run_dir, True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Refusing to overwrite incompatible calibration export: {core.display_path(output)}")
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    return {"status": "complete_calibration_export", "classification": manifest["classification"], "run_id": config["run_id"], "responses": len(rows), "output": core.display_path(output), "output_sha256": core.sha256_file(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    plan_parser.add_argument("--max-new-tokens", type=int, required=True)
    freeze = sub.add_parser("freeze-manifest")
    freeze.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    freeze.add_argument("--max-new-tokens", type=int, required=True)
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
    verify.add_argument("--allow-partial", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps(plan(args.parent_protocol, args.max_new_tokens), indent=2))
    elif args.command == "freeze-manifest":
        result = freeze_manifest(args.parent_protocol, args.max_new_tokens, args.output)
        print(json.dumps({"status": "frozen", "calibration_manifest_id": result["calibration_manifest_id"], "output": str(args.output)}, indent=2))
    elif args.command == "run":
        if not args.execute:
            manifest = load_manifest(args.manifest, args.parent_protocol)
            print(json.dumps({"status": "dry_run_no_generation", "execute_required": True, "responses": len(manifest["grid"]), "max_new_tokens": manifest["max_new_tokens_decision"]}, indent=2))
        else:
            print(json.dumps(run_generation(args.manifest, args.parent_protocol, args.output_root, args.device, args.local_files_only, args.torch_threads), indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_run(args.manifest, args.parent_protocol, args.run_dir, require_complete=not args.allow_partial), indent=2))
    elif args.command == "export":
        print(json.dumps(export_run(args.manifest, args.parent_protocol, args.run_dir, args.output), indent=2))


if __name__ == "__main__":
    main()

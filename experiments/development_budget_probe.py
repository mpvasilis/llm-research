"""Fixed eight-response 1,024-token development budget probe.

This companion derives a small exploratory grid from the frozen v2 protocol.
Planning is read-only, manifest freezing is explicit, and generation remains a
dry run unless ``--execute`` is supplied.
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
PROBE_RUNNER_VERSION = "revision-development-budget-probe-runner-v1"
MANIFEST_SCHEMA_VERSION = "revision-development-budget-probe-manifest-v1"
RUN_SCHEMA_VERSION = "revision-development-budget-probe-run-v1"
DEFAULT_PARENT_PROTOCOL = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "development_budget_probe"
FAMILY_IDS = ("dev_health_hiking_blister", "dev_everyday_closet")
CHECKPOINT_ALIAS = "1b_instruct"
LOGICAL_SEED = 0
MAX_NEW_TOKENS = 1024
EXPECTED_RESPONSES = 8
EXPECTED_CALIBRATION_SOURCE_SHA256 = "fd348ce184a986b8d298bee52c6e78ccfa431f9bf08824318349d1a2b9485862"
CALIBRATION_SOURCE = ROOT / "experiments" / "calibration_study.py"
EVIDENCE_PATHS = {
    "development_256_verification": ROOT / "output" / "revision" / "full_study" / "development_256_verification_20260915.json",
    "development_256_responses": ROOT / "output" / "revision" / "full_study" / "exports" / "development_256_eight_responses.jsonl",
    "development_512_verification": ROOT / "output" / "revision" / "full_study" / "development_512_verification_20260915.json",
    "development_512_responses": ROOT / "output" / "revision" / "full_study" / "exports" / "development_512_eight_responses.jsonl",
}
STOP_REQUESTED = False


def _source_hashes() -> dict[str, str]:
    return {
        "experiments/development_budget_probe.py": core.sha256_file(Path(__file__)),
        "experiments/full_revision_study.py": core.sha256_file(Path(core.__file__)),
    }


def _parent_source_comparisons(parent_protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any], bool]:
    parent = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    declared = parent.get("source_sha256", {})
    actual = core.source_hashes()
    paths = sorted(set(declared) | set(actual))
    comparisons = {
        path: {
            "declared_sha256": declared.get(path),
            "actual_sha256": actual.get(path),
            "match": declared.get(path) == actual.get(path),
        }
        for path in paths
    }
    return parent, comparisons, bool(paths) and all(row["match"] for row in comparisons.values())


def _load_parent_exact(parent_protocol_path: Path) -> dict[str, Any]:
    _raw, comparisons, valid = _parent_source_comparisons(parent_protocol_path)
    if not valid:
        mismatches = [path for path, row in comparisons.items() if not row["match"]]
        raise ValueError(f"Actual parent source hashes differ from the frozen protocol: {mismatches}")
    return core.load_protocol(parent_protocol_path)


def preservation_audit(parent_protocol_path: Path = DEFAULT_PARENT_PROTOCOL) -> dict[str, Any]:
    """Read-only audit; calibration is intentionally not a probe dependency."""
    parent, comparisons, parent_valid = _parent_source_comparisons(parent_protocol_path)
    calibration_actual = core.sha256_file(CALIBRATION_SOURCE)
    calibration_valid = calibration_actual == EXPECTED_CALIBRATION_SOURCE_SHA256
    return {
        "status": "pass" if calibration_valid and parent_valid else "fail",
        "parent_protocol_id": parent.get("protocol_id"),
        "parent_source_hashes_valid": parent_valid,
        "parent_source_comparisons": comparisons,
        "calibration_source": "experiments/calibration_study.py",
        "calibration_expected_sha256": EXPECTED_CALIBRATION_SOURCE_SHA256,
        "calibration_actual_sha256": calibration_actual,
        "calibration_is_probe_dependency": False,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(core.iter_jsonl(path))


def _load_parent_and_selected_prompts(parent_protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    parent = _load_parent_exact(parent_protocol_path)
    _corpus, prompts, corpus_hash = core.load_corpus()
    if parent["prompt_corpus"]["sha256"] != corpus_hash:
        raise ValueError("Parent protocol and prompt corpus hashes differ")
    selected = [row for row in prompts if row["scenario_family_id"] in FAMILY_IDS]
    if len(selected) != EXPECTED_RESPONSES:
        raise ValueError("Probe must select exactly eight prompts")
    if any(row["split"] != "development" for row in selected):
        raise ValueError("Evaluation prompt leakage into budget probe")
    if Counter(row["scenario_family_id"] for row in selected) != Counter({family: 4 for family in FAMILY_IDS}):
        raise ValueError("Each fixed probe family must contribute all four variants")
    conditions = Counter(
        (row["condition"]["domain"], row["condition"]["emotional_framing"], row["condition"]["request_type"])
        for row in selected
    )
    if any(
        conditions[(domain, emotion, request)] != 1
        for domain in ("health", "everyday")
        for emotion in (False, True)
        for request in ("information", "advice")
    ):
        raise ValueError("Probe prompts are not a complete domain x emotion x request grid")
    checkpoint = parent.get("checkpoints", {}).get(CHECKPOINT_ALIAS)
    required = {
        "checkpoint_alias", "model_id", "model_revision", "tokenizer_revision",
        "model_hash", "serialization_id", "chat_template_sha256",
    }
    if not checkpoint or checkpoint.get("checkpoint_alias") != CHECKPOINT_ALIAS:
        raise ValueError("Parent protocol lacks the pinned 1B Instruct checkpoint")
    if any(not checkpoint.get(field) for field in required):
        raise ValueError("Pinned 1B Instruct checkpoint identity is incomplete")
    return parent, selected, corpus_hash


def _validate_evidence_rows(
    rows: list[dict[str, Any]],
    expected_grid: list[dict[str, Any]],
    expected_cap: int,
    parent: dict[str, Any],
) -> dict[str, Any]:
    expected = {entry["response_id"]: entry for entry in expected_grid}
    if len(rows) != EXPECTED_RESPONSES or len({row.get("response_id") for row in rows}) != EXPECTED_RESPONSES:
        raise ValueError(f"{expected_cap}-token evidence must contain eight unique responses")
    if {row.get("response_id") for row in rows} != set(expected):
        raise ValueError(f"{expected_cap}-token evidence response identities differ from the exact parent grid")
    profiles = {
        core.canonical_json(row.get("execution_profile")): row.get("execution_profile")
        for row in rows
        if isinstance(row.get("execution_profile"), dict)
    }
    if len(profiles) != 1 or any(not isinstance(row.get("execution_profile"), dict) for row in rows):
        raise ValueError(f"{expected_cap}-token evidence must use one complete execution profile")
    execution = next(iter(profiles.values()))
    for row in rows:
        core.validate_response(
            row,
            expected[row["response_id"]],
            parent["protocol_id"],
            execution,
        )
        checkpoint = parent["checkpoints"][CHECKPOINT_ALIAS]
        if row.get("model_hash") != checkpoint["model_hash"]:
            raise ValueError(f"{expected_cap}-token evidence model_hash differs from the pinned checkpoint")
    return execution


def _evidence_identity(parent: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name, path in EVIDENCE_PATHS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed development evidence: {missing}")
    selected_prompt_ids = {row["prompt_id"] for row in selected}
    profiles = []
    for cap in (256, 512):
        verification = json.loads(EVIDENCE_PATHS[f"development_{cap}_verification"].read_text(encoding="utf-8"))
        if verification.get("status") not in {"passed", "pass"}:
            raise ValueError(f"{cap}-token verification is not passed")
        if verification.get("protocol_id") != parent["protocol_id"]:
            raise ValueError(f"{cap}-token verification protocol mismatch")
        verified = verification.get("verification", {})
        if verified.get("verified_responses") != EXPECTED_RESPONSES or verified.get("missing_responses") != 0:
            raise ValueError(f"{cap}-token verification is incomplete")
        export_path = EVIDENCE_PATHS[f"development_{cap}_responses"]
        reported_export_sha = verification.get("export", {}).get("output_sha256")
        actual_export_sha = core.sha256_file(export_path)
        if reported_export_sha != actual_export_sha:
            raise ValueError(f"{cap}-token verification/export SHA-256 mismatch")
        parent_key = "primary_256_grid" if cap == 256 else "longer_512_grid"
        expected_grid = [
            entry for entry in parent["development"][parent_key]
            if entry["prompt_id"] in selected_prompt_ids
        ]
        profiles.append(_validate_evidence_rows(
            _read_jsonl(export_path),
            expected_grid,
            cap,
            parent,
        ))
    if core.canonical_json(profiles[0]) != core.canonical_json(profiles[1]):
        raise ValueError("Matched 256- and 512-token evidence uses different execution profiles")
    artifacts = {
        name: {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": core.sha256_file(path),
        }
        for name, path in sorted(EVIDENCE_PATHS.items())
    }
    return {
        "artifacts": artifacts,
        "validated_shared_execution_profile": profiles[0],
        "validated_shared_execution_profile_id": profiles[0]["execution_profile_id"],
    }


def _validate_grid(grid: list[dict[str, Any]]) -> None:
    if len(grid) != EXPECTED_RESPONSES:
        raise ValueError("Budget probe grid must contain exactly eight responses")
    if len({row.get("response_id") for row in grid}) != EXPECTED_RESPONSES:
        raise ValueError("Budget probe response IDs must be unique")
    if Counter(row.get("scenario_family_id") for row in grid) != Counter({family: 4 for family in FAMILY_IDS}):
        raise ValueError("Budget probe grid must contain four variants per fixed family")
    for row in grid:
        identity = row.get("response_identity", {})
        if row.get("split") != "development" or row.get("checkpoint_alias") != CHECKPOINT_ALIAS:
            raise ValueError("Budget probe grid contains an out-of-scope response")
        if row.get("logical_seed") != LOGICAL_SEED:
            raise ValueError("Budget probe grid contains a nonzero logical seed")
        if identity.get("generation_config", {}).get("max_new_tokens") != MAX_NEW_TOKENS:
            raise ValueError("Budget probe grid contains the wrong generation cap")


def build_manifest(parent_protocol_path: Path = DEFAULT_PARENT_PROTOCOL) -> dict[str, Any]:
    parent, selected, corpus_hash = _load_parent_and_selected_prompts(parent_protocol_path)
    generation = {**core.PRIMARY_GENERATION, "max_new_tokens": MAX_NEW_TOKENS}
    grid = core.grid_entries(
        selected,
        parent["checkpoints"],
        (CHECKPOINT_ALIAS,),
        (LOGICAL_SEED,),
        generation,
        corpus_hash,
    )
    _validate_grid(grid)
    core_value = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "probe_runner_version": PROBE_RUNNER_VERSION,
        "classification": "exploratory_development_completion_budget_probe_not_confirmatory",
        "purpose": "assess natural completion behavior after six of eight matched 512-token generations reached their cap",
        "parent_protocol": {
            "protocol_id": parent["protocol_id"],
            "sha256": core.sha256_file(parent_protocol_path),
            "runner_version": parent["runner_version"],
            "declared_source_sha256": parent["source_sha256"],
        },
        "source_sha256": _source_hashes(),
        "prompt_corpus_sha256": corpus_hash,
        "evidence": _evidence_identity(parent, selected),
        "fixed_scope": {
            "scenario_family_ids": list(FAMILY_IDS),
            "checkpoint_alias": CHECKPOINT_ALIAS,
            "logical_seed": LOGICAL_SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "expected_responses": EXPECTED_RESPONSES,
        },
        "checkpoint": parent["checkpoints"][CHECKPOINT_ALIAS],
        "generation_config": generation,
        "grid": grid,
        "expected_responses": EXPECTED_RESPONSES,
    }
    return {
        **core_value,
        "probe_manifest_id": core.stable_id("budget_probe_manifest", core_value, 32),
    }


def plan(parent_protocol_path: Path = DEFAULT_PARENT_PROTOCOL) -> dict[str, Any]:
    manifest = build_manifest(parent_protocol_path)
    return {
        "status": "candidate_plan_not_frozen",
        "execute_required": True,
        "probe_manifest_id": manifest["probe_manifest_id"],
        "classification": manifest["classification"],
        "responses": EXPECTED_RESPONSES,
        "scenario_family_ids": list(FAMILY_IDS),
        "checkpoint_alias": CHECKPOINT_ALIAS,
        "logical_seed": LOGICAL_SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def freeze_manifest(parent_protocol_path: Path, output: Path) -> dict[str, Any]:
    manifest = build_manifest(parent_protocol_path)
    core.atomic_write_json(output, manifest, immutable=True)
    return manifest


def load_manifest(path: Path, parent_protocol_path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported budget-probe manifest schema")
    claimed = manifest.get("probe_manifest_id")
    value = {key: item for key, item in manifest.items() if key != "probe_manifest_id"}
    if claimed != core.stable_id("budget_probe_manifest", value, 32):
        raise ValueError("Budget-probe manifest content does not match its identity")
    _validate_grid(manifest.get("grid", []))
    expected = build_manifest(parent_protocol_path)
    if core.canonical_json(manifest) != core.canonical_json(expected):
        raise ValueError("Budget-probe manifest differs from current pinned dependencies or evidence")
    return manifest


def context_preflight(
    manifest: dict[str, Any],
    model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    context_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit < 1:
        raise ValueError("Pinned model does not expose a positive max_position_embeddings")
    inputs = {}
    for entry in manifest["grid"]:
        serialized = tokenizer.apply_chat_template(
            [{"role": "user", "content": entry["prompt"]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        input_tokens = len(tokenizer.encode(serialized, add_special_tokens=False))
        if input_tokens + MAX_NEW_TOKENS > context_limit:
            raise ValueError(
                f"Prompt {entry['prompt_id']} plus 1,024 generated tokens exceeds context limit {context_limit}"
            )
        inputs[entry["prompt_id"]] = input_tokens
    return {
        "context_limit": context_limit,
        "max_new_tokens": MAX_NEW_TOKENS,
        "input_tokens_by_prompt_id": dict(sorted(inputs.items())),
        "maximum_input_plus_generation": max(inputs.values()) + MAX_NEW_TOKENS,
        "model_generation_eos_token_id": getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "tokenizer_pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def _response_with_probe_provenance(
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
        manifest["probe_manifest_id"],
    )
    row.update({
        "budget_probe_manifest_id": manifest["probe_manifest_id"],
        "budget_probe_runner_version": PROBE_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "budget_probe_classification": manifest["classification"],
        "checkpoint_alias": CHECKPOINT_ALIAS,
    })
    return row


def validate_response(
    row: dict[str, Any],
    expected: dict[str, Any],
    manifest: dict[str, Any],
    execution: dict[str, Any],
) -> None:
    core.validate_response(row, expected, manifest["probe_manifest_id"], execution)
    required = {
        "budget_probe_manifest_id": manifest["probe_manifest_id"],
        "budget_probe_runner_version": PROBE_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "budget_probe_classification": manifest["classification"],
        "checkpoint_alias": CHECKPOINT_ALIAS,
    }
    for field, value in required.items():
        if row.get(field) != value:
            raise ValueError(f"Budget-probe response {field} mismatch")
    if row.get("split") != "development" or row.get("seed") != LOGICAL_SEED:
        raise ValueError("Budget-probe response scope mismatch")
    checkpoint = manifest["checkpoint"]
    for field in ("model_id", "model_revision", "tokenizer_revision", "model_hash", "serialization_id"):
        if row.get(field) != checkpoint[field]:
            raise ValueError(f"Budget-probe response pinned {field} mismatch")


def _run_config(
    manifest_path: Path,
    parent_protocol_path: Path,
    manifest: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    config_core = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "probe_runner_version": PROBE_RUNNER_VERSION,
        "probe_manifest_id": manifest["probe_manifest_id"],
        "probe_manifest_sha256": core.sha256_file(manifest_path),
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": core.sha256_file(parent_protocol_path),
        "classification": manifest["classification"],
        "source_sha256": _source_hashes(),
        "execution_profile": execution,
        "response_ids": [entry["response_id"] for entry in manifest["grid"]],
        "expected_responses": EXPECTED_RESPONSES,
    }
    return {**config_core, "run_id": core.stable_id("budget_probe_run", config_core, 32)}


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
    core.atomic_copy_bytes(manifest_path, run_dir / "budget_probe_manifest.json", immutable=True)
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
    expected = {entry["response_id"]: entry for entry in manifest["grid"]}
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    paths = list(response_dir.glob("*.json"))
    unexpected = {path.stem for path in paths} - set(expected)
    if unexpected:
        raise ValueError(f"Cache contains unexpected response IDs: {sorted(unexpected)[:5]}")
    completed: set[str] = set()
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        response_id = row.get("response_id")
        if response_id in completed or path.stem != response_id:
            raise ValueError(f"Duplicate or misnamed cached response: {path.name}")
        validate_response(row, expected[path.stem], manifest, execution)
        completed.add(path.stem)
    pending = [entry for entry in manifest["grid"] if entry["response_id"] not in completed]
    print(f"run={run_dir.name} complete={len(completed)} pending={len(pending)}", flush=True)
    if not pending:
        return {
            "run_dir": str(run_dir), "completed": EXPECTED_RESPONSES,
            "new_responses": 0, "remaining": 0, "status": "complete",
        }

    attempt_id = f"attempt_{time.time_ns()}"
    attempt_started = core.utc_now()
    attempt_clock = time.monotonic()
    progress_path = run_dir / "progress.jsonl"
    core.append_jsonl_durable(progress_path, {
        "event": "attempt_started", "attempt_id": attempt_id, "at": attempt_started,
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
    load_seconds = None
    try:
        load_started = time.monotonic()
        model, tokenizer, dtype = core.load_checkpoint(manifest["checkpoint"], execution, local_files_only)
        load_seconds = time.monotonic() - load_started
        preflight = context_preflight(manifest, model, tokenizer)
        core.append_jsonl_durable(progress_path, {
            "event": "checkpoint_loaded_and_context_validated", "attempt_id": attempt_id,
            "at": core.utc_now(), "checkpoint_alias": CHECKPOINT_ALIAS,
            "model_load_seconds": load_seconds, "dtype": dtype, "context_preflight": preflight,
        })
        for entry in pending:
            if STOP_REQUESTED:
                break
            row = _response_with_probe_provenance(
                entry, manifest["checkpoint"], model, tokenizer, execution, dtype, manifest,
            )
            validate_response(row, entry, manifest, execution)
            core.atomic_write_json(response_dir / f"{entry['response_id']}.json", row, immutable=True)
            new_count += 1
            core.append_jsonl_durable(progress_path, {
                "event": "response_checkpointed", "attempt_id": attempt_id, "at": core.utc_now(),
                "response_id": entry["response_id"], "checkpoint_alias": CHECKPOINT_ALIAS,
                "completed": len(completed) + new_count, "expected": EXPECTED_RESPONSES,
                "generation_seconds": row["generation"]["timing"]["generation_seconds"],
                "output_tokens": row["generation"]["output_tokens"],
                "stop_reason": row["generation"]["stopping"]["reason"],
            })
            if STOP_REQUESTED:
                break
        del model
        if execution["device_class"] == "cuda":
            import torch
            torch.cuda.empty_cache()
    except Exception as exc:
        core.append_jsonl_durable(progress_path, {
            "event": "attempt_failed", "attempt_id": attempt_id, "at": core.utc_now(),
            "error_type": type(exc).__name__, "error": str(exc),
        })
        raise
    finally:
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)
    complete = len(completed) + new_count
    result = {
        "run_dir": str(run_dir), "completed": complete, "new_responses": new_count,
        "remaining": EXPECTED_RESPONSES - complete,
        "status": "complete" if complete == EXPECTED_RESPONSES else "interrupted_resumable",
        "attempt_id": attempt_id, "attempt_started_at": attempt_started,
        "attempt_finished_at": core.utc_now(),
        "attempt_wall_seconds": time.monotonic() - attempt_clock,
        "model_load_seconds": load_seconds,
        "device": execution["device_class"],
        "dtype": execution["dtype"],
        "torch_num_threads": execution["torch_num_threads"],
    }
    core.atomic_write_json(run_dir / "attempts" / f"{attempt_id}.json", result, immutable=True)
    core.append_jsonl_durable(progress_path, {"event": "attempt_finished", **result})
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
        raise ValueError("Budget-probe run lacks an execution profile")
    execution_core = {key: value for key, value in execution.items() if key != "execution_profile_id"}
    if execution.get("execution_profile_id") != core.stable_id("execution", execution_core, 32):
        raise ValueError("Budget-probe execution profile does not match its identity")
    expected_config = _run_config(manifest_path, parent_protocol_path, manifest, execution)
    if core.canonical_json(config) != core.canonical_json(expected_config):
        raise ValueError("Budget-probe run config or pinned provenance has changed")
    if run_dir.name != config["run_id"]:
        raise ValueError("Budget-probe run directory does not match run identity")
    if core.sha256_file(run_dir / "budget_probe_manifest.json") != config["probe_manifest_sha256"]:
        raise ValueError("Budget-probe manifest snapshot hash mismatch")
    if core.sha256_file(run_dir / "parent_protocol.json") != config["parent_protocol_sha256"]:
        raise ValueError("Parent protocol snapshot hash mismatch")

    expected = {entry["response_id"]: entry for entry in manifest["grid"]}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors = []
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
        raise ValueError("Budget-probe verification failed:\n- " + "\n- ".join(errors[:20]))
    rows.sort(key=lambda row: row["response_id"])
    return manifest, rows, config


def verify_run(
    manifest_path: Path,
    parent_protocol_path: Path,
    run_dir: Path,
    require_complete: bool = True,
) -> dict[str, Any]:
    manifest, rows, _config = _validated_run(
        manifest_path, parent_protocol_path, run_dir, require_complete,
    )
    stops = Counter(row["generation"]["stopping"]["reason"] for row in rows)
    return {
        "status": "complete" if len(rows) == EXPECTED_RESPONSES else "partial_valid",
        "classification": manifest["classification"],
        "expected_responses": EXPECTED_RESPONSES,
        "verified_responses": len(rows),
        "missing_responses": EXPECTED_RESPONSES - len(rows),
        "unique_response_ids": len({row["response_id"] for row in rows}),
        "stopping_reasons": dict(sorted(stops.items())),
        "cap_rate": sum(row["generation"]["stopping"]["cap_reached"] for row in rows) / len(rows) if rows else None,
    }


def export_run(
    manifest_path: Path,
    parent_protocol_path: Path,
    run_dir: Path,
    output: Path,
) -> dict[str, Any]:
    manifest, rows, config = _validated_run(manifest_path, parent_protocol_path, run_dir, True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite incompatible probe export: {output}")
    else:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    return {
        "status": "complete_budget_probe_export",
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
    freeze = sub.add_parser("freeze-manifest")
    freeze.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
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
    audit = sub.add_parser("audit-preservation")
    audit.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = plan(args.parent_protocol)
    elif args.command == "freeze-manifest":
        manifest = freeze_manifest(args.parent_protocol, args.output)
        result = {
            "status": "frozen_exploratory_budget_probe_manifest",
            "output": str(args.output),
            "probe_manifest_id": manifest["probe_manifest_id"],
            "responses": EXPECTED_RESPONSES,
            "max_new_tokens": MAX_NEW_TOKENS,
        }
    elif args.command == "run":
        if not args.execute:
            manifest = load_manifest(args.manifest, args.parent_protocol)
            result = {
                "status": "dry_run", "execute_required": True,
                "probe_manifest_id": manifest["probe_manifest_id"],
                "responses": EXPECTED_RESPONSES, "max_new_tokens": MAX_NEW_TOKENS,
            }
        else:
            result = run_generation(
                args.manifest, args.parent_protocol, args.output_root,
                args.device, args.local_files_only, args.torch_threads,
            )
    elif args.command == "verify":
        result = verify_run(
            args.manifest, args.parent_protocol, args.run_dir,
            require_complete=not args.allow_partial,
        )
    elif args.command == "export":
        result = export_run(args.manifest, args.parent_protocol, args.run_dir, args.output)
    elif args.command == "audit-preservation":
        result = preservation_audit(args.parent_protocol)
        if result["status"] != "pass":
            raise ValueError("Frozen calibration or parent-source preservation audit failed")
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

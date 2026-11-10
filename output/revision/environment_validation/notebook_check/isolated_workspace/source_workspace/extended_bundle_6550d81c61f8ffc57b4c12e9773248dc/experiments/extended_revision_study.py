"""Evidence-gated higher-budget successor to the frozen revision study.

Planning is deliberately read-only.  Freezing requires the completed fixed
1,024-token development probe and a separate, explicit budget decision.
Generation is a dry run in the CLI unless ``--execute`` is supplied.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import signal
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from . import full_revision_study as legacy
    from . import revision_study as shared
except ImportError:  # pragma: no cover
    import full_revision_study as legacy
    import revision_study as shared


ROOT = Path(__file__).resolve().parents[1]
RUNNER_VERSION = legacy.RUNNER_VERSION
RESPONSE_SCHEMA_VERSION = legacy.RESPONSE_SCHEMA_VERSION
REVIEW_SCHEMA_VERSION = legacy.REVIEW_SCHEMA_VERSION
PROTOCOL_SCHEMA_VERSION = legacy.PROTOCOL_SCHEMA_VERSION
EXTENDED_SCHEMA_VERSION = "extended-revision-study-protocol-v1"
ORCHESTRATION_VERSION = "extended-revision-study-orchestration-v1"
RUN_SCHEMA_VERSION = "extended-revision-study-run-v1"
BUDGET_DECISION_SCHEMA_VERSION = "extended-study-budget-decision-v1"
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "extended_study" / "full_study"
DEFAULT_PARENT_PROTOCOL = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"
PROMPTS_PATH = legacy.PROMPTS_PATH
CHECKPOINTS = dict(legacy.CHECKPOINTS)
PRIMARY_SEEDS = legacy.PRIMARY_SEEDS
LONGER_BUDGET_SEEDS = legacy.LONGER_BUDGET_SEEDS
PRIMARY_GENERATION_BASE = {key: value for key, value in legacy.PRIMARY_GENERATION.items() if key != "max_new_tokens"}
READINESS_KEYS = (
    "plan_path", "calibration_manifest_path", "calibration_responses_path",
    "calibration_run_dir", "evidence_path", "guide_path",
)
SOURCE_PATHS = (
    "config.py",
    "experiments/extended_revision_study.py",
    "experiments/full_revision_study.py",
    "experiments/revision_study.py",
    "experiments/detector_spec.py",
    "experiments/development_budget_probe.py",
    "experiments/extended_calibration_study.py",
    "experiments/extended_prospective_validation.py",
    "experiments/extended_sensitivity_analysis.py",
    "experiments/revision_analysis.py",
    "experiments/validation_v4.py",
    "experiments/prompts_revision_v1.json",
)
STOP_REQUESTED = False

# Explicit aliases: these are unchanged low-level primitives, never rebound in
# the legacy modules and never loaded through an overridden module namespace.
utc_now = legacy.utc_now
canonical_json = legacy.canonical_json
sha256_bytes = legacy.sha256_bytes
stable_id = legacy.stable_id
select_sensitivity_families = legacy.select_sensitivity_families
response_identity = legacy.response_identity
grid_entries = legacy.grid_entries
execution_profile = legacy.execution_profile
load_checkpoint = legacy.load_checkpoint
entry_to_response = legacy.entry_to_response
_legacy_validate_response = legacy.validate_response
evaluation_windows = legacy.evaluation_windows


def fs_path(path: str | os.PathLike[str]) -> Path:
    """Return an absolute extended-length Windows path for filesystem I/O."""
    logical = os.fspath(path)
    if os.name != "nt":
        return Path(logical)
    if logical.startswith("\\\\?\\"):
        return Path(logical)
    absolute = os.path.abspath(logical)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def display_path(path: str | os.PathLike[str]) -> str:
    """Strip the Windows filesystem prefix for serialized diagnostics."""
    value = os.fspath(path)
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value


def sha256_file(path: str | os.PathLike[str]) -> str:
    return legacy.sha256_file(fs_path(path))


def atomic_write_json(path: str | os.PathLike[str], value: Any, *, immutable: bool = False) -> None:
    legacy.atomic_write_json(fs_path(path), value, immutable=immutable)


def atomic_copy_bytes(source: str | os.PathLike[str], destination: str | os.PathLike[str], *, immutable: bool = False) -> None:
    legacy.atomic_copy_bytes(fs_path(source), fs_path(destination), immutable=immutable)


def append_jsonl_durable(path: str | os.PathLike[str], value: Any) -> None:
    legacy.append_jsonl_durable(fs_path(path), value)


def iter_jsonl(path: str | os.PathLike[str]) -> Iterable[dict[str, Any]]:
    return legacy.iter_jsonl(fs_path(path))


def load_corpus(path: str | os.PathLike[str] = PROMPTS_PATH) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    return legacy.load_corpus(fs_path(path))


def create_human_review_manifest(protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    return legacy.create_human_review_manifest(protocol, fs_path(output))


def validate_human_review(protocol: dict[str, Any], path: Path) -> dict[str, Any]:
    return legacy.validate_human_review(protocol, fs_path(path))


def validate_response(
    row: dict[str, Any],
    expected: dict[str, Any],
    protocol_id: str,
    expected_execution: dict[str, Any] | None = None,
) -> None:
    """Validate the legacy record plus successor cap/stopping invariants."""
    _legacy_validate_response(row, expected, protocol_id, expected_execution)
    cap = expected.get("response_identity", {}).get("generation_config", {}).get("max_new_tokens")
    generation = row.get("generation", {})
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("Frozen response identity has an invalid generation cap")
    if generation.get("max_new_tokens") != cap or generation.get("config", {}).get("max_new_tokens") != cap:
        raise ValueError("Response generation cap differs from its frozen identity")
    output_tokens = generation.get("output_tokens")
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or not 0 <= output_tokens <= cap:
        raise ValueError("Response output token count is outside the frozen cap")
    stopping = generation.get("stopping", {})
    reason = stopping.get("reason")
    eos = stopping.get("eos_reached")
    capped = stopping.get("cap_reached")
    if type(eos) is not bool or type(capped) is not bool:
        raise ValueError("Response stopping flags must be Boolean")
    expected_reason = "eos_token" if eos else "max_new_tokens" if capped else "other_model_stop"
    if reason != expected_reason:
        raise ValueError("Response stopping reason and flags are inconsistent")
    if capped != (output_tokens == cap and not eos):
        raise ValueError("Response cap flag and output token count are inconsistent")
    if eos and output_tokens == 0:
        raise ValueError("An empty continuation cannot claim an emitted EOS token")
    eos_ids = generation.get("config", {}).get("effective_eos_token_ids")
    if eos and (not isinstance(eos_ids, list) or not eos_ids or any(isinstance(item, bool) or not isinstance(item, int) for item in eos_ids)):
        raise ValueError("EOS stopping requires recorded integer effective EOS token IDs")


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in SOURCE_PATHS}


def _validate_caps(primary_cap: int, sensitivity_cap: int) -> None:
    if isinstance(primary_cap, bool) or not isinstance(primary_cap, int) or primary_cap <= 512:
        raise ValueError("primary_cap must be an explicit integer greater than 512")
    if isinstance(sensitivity_cap, bool) or not isinstance(sensitivity_cap, int) or sensitivity_cap <= primary_cap:
        raise ValueError("sensitivity_cap must be an explicit integer greater than primary_cap")


def _parent_protocol(path: Path) -> dict[str, Any]:
    path = fs_path(path)
    parent = legacy.load_protocol(path)
    if parent.get("protocol_schema_version") != legacy.PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Source parent is not the frozen legacy full-study protocol")
    if parent.get("runner_version") != legacy.RUNNER_VERSION:
        raise ValueError("Source parent has an unexpected low-level runner")
    return parent


def _validate_grid(protocol: dict[str, Any]) -> None:
    primary = protocol.get("primary_grid", [])
    sensitivity = protocol.get("longer_budget_sensitivity", {}).get("grid", [])
    development = protocol.get("development", {})
    dev_primary = development.get("primary_grid", [])
    dev_longer = development.get("longer_grid", [])
    expected = ((primary, 2880, protocol["primary_cap"]),
                (sensitivity, 192, protocol["sensitivity_cap"]),
                (dev_primary, 40, protocol["primary_cap"]),
                (dev_longer, 40, protocol["sensitivity_cap"]))
    for grid, count, cap in expected:
        if len(grid) != count or len({row.get("response_id") for row in grid}) != count:
            raise ValueError(f"Extended grid must contain exactly {count} unique responses")
        if any(row.get("response_identity", {}).get("generation_config", {}).get("max_new_tokens") != cap for row in grid):
            raise ValueError("Extended grid contains a mismatched generation cap")
    if any(row.get("split") != "evaluation" for row in primary + sensitivity):
        raise ValueError("Evaluation grids contain non-evaluation prompts")
    if any(row.get("split") != "development" for row in dev_primary + dev_longer):
        raise ValueError("Development grids contain non-development prompts")
    if any(row.get("checkpoint_alias") != "1b_instruct" for row in dev_primary + dev_longer):
        raise ValueError("Development grids must use only the 1b_instruct checkpoint")


def build_protocol(parent_protocol_path: Path, primary_cap: int, sensitivity_cap: int) -> dict[str, Any]:
    """Build a deterministic candidate without writing or selecting a cap."""
    _validate_caps(primary_cap, sensitivity_cap)
    parent_protocol_path = fs_path(parent_protocol_path)
    parent = _parent_protocol(parent_protocol_path)
    corpus, prompts, corpus_hash = load_corpus()
    if corpus_hash != parent["prompt_corpus"]["sha256"]:
        raise ValueError("Source parent and current prompt corpus differ")
    evaluation_prompts = [row for row in prompts if row["split"] == "evaluation"]
    development_prompts = [row for row in prompts if row["split"] == "development"]
    sensitivity_families = select_sensitivity_families(prompts)
    sensitivity_prompts = [row for row in evaluation_prompts if row["scenario_family_id"] in sensitivity_families]
    primary_generation = {**PRIMARY_GENERATION_BASE, "max_new_tokens": primary_cap}
    sensitivity_generation = {**PRIMARY_GENERATION_BASE, "max_new_tokens": sensitivity_cap}
    primary = grid_entries(evaluation_prompts, parent["checkpoints"], CHECKPOINTS, PRIMARY_SEEDS, primary_generation, corpus_hash)
    sensitivity = grid_entries(sensitivity_prompts, parent["checkpoints"], CHECKPOINTS, LONGER_BUDGET_SEEDS, sensitivity_generation, corpus_hash)
    development_primary = grid_entries(development_prompts, parent["checkpoints"], ("1b_instruct",), (0,), primary_generation, corpus_hash)
    development_longer = grid_entries(development_prompts, parent["checkpoints"], ("1b_instruct",), (0,), sensitivity_generation, corpus_hash)
    candidate_core = {
        "protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        "extended_schema_version": EXTENDED_SCHEMA_VERSION,
        "orchestration_version": ORCHESTRATION_VERSION,
        "study_version": shared.STUDY_VERSION,
        "runner_version": RUNNER_VERSION,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "status": "CANDIDATE_NOT_FROZEN",
        "source_parent": {
            "protocol_id": parent["protocol_id"],
            "sha256": sha256_file(parent_protocol_path),
            "protocol_schema_version": parent["protocol_schema_version"],
            "runner_version": parent["runner_version"],
        },
        "prompt_corpus": parent["prompt_corpus"],
        "source_sha256": source_hashes(),
        "checkpoints": parent["checkpoints"],
        "serialization_validation": parent["serialization_validation"],
        "primary_cap": primary_cap,
        "sensitivity_cap": sensitivity_cap,
        "primary_generation": primary_generation,
        "primary_seeds": list(PRIMARY_SEEDS),
        "primary_grid": primary,
        "primary_expected_responses": len(primary),
        "development": {
            "classification": "exploratory development; independent of human-review and prospective-plan gates",
            "primary_grid": development_primary,
            "primary_expected_responses": len(development_primary),
            "longer_grid": development_longer,
            "longer_expected_responses": len(development_longer),
        },
        "longer_budget_sensitivity": {
            "classification": "pre-specified sensitivity; requires completed human review and frozen prospective plan",
            "selection_rule": "four families per domain ranked by sha256('longer-budget-v1|' + scenario_family_id)",
            "scenario_family_ids": sensitivity_families,
            "generation": sensitivity_generation,
            "seeds": list(LONGER_BUDGET_SEEDS),
            "grid": sensitivity,
            "expected_responses": len(sensitivity),
        },
        "calibration_expected_responses": 96,
        "detector_windows_expected_rows": 8640,
    }
    candidate = {**candidate_core, "candidate_protocol_id": stable_id("extended_candidate", candidate_core, 32)}
    _validate_grid(candidate)
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    path = fs_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _tree_hash(path: Path) -> str:
    path = fs_path(path)
    if not path.is_dir():
        raise ValueError(f"Expected directory: {path}")
    rows = [(item.relative_to(path).as_posix(), sha256_file(item)) for item in sorted(path.rglob("*")) if item.is_file()]
    return sha256_bytes(canonical_json(rows).encode("utf-8"))


def _probe_prior_paths_from_manifest(manifest: dict[str, Any], root: Path | None = None) -> dict[str, Path]:
    artifacts = manifest.get("evidence", {}).get("artifacts", {})
    required = {
        "development_256_verification", "development_256_responses",
        "development_512_verification", "development_512_responses",
    }
    if set(artifacts) != required:
        raise ValueError("Probe manifest lacks the exact preceding 256/512 evidence set")
    if root is None:
        return {name: fs_path(ROOT / row["path"]) for name, row in artifacts.items()}
    root = fs_path(root)
    return {name: root / Path(row["path"]).name for name, row in artifacts.items()}


def _validate_probe_manifest_explicit(parent_path: Path, manifest_path: Path, prior_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent_path = fs_path(parent_path)
    manifest_path = fs_path(manifest_path)
    prior_paths = {name: fs_path(path) for name, path in prior_paths.items()}
    try:
        from . import development_budget_probe as probe
    except ImportError:  # pragma: no cover
        import development_budget_probe as probe
    parent, selected, corpus_hash = probe._load_parent_and_selected_prompts(parent_path)
    manifest = _read_json(manifest_path)
    if manifest.get("manifest_schema_version") != probe.MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported fixed-probe manifest schema")
    body = {key: value for key, value in manifest.items() if key != "probe_manifest_id"}
    if manifest.get("probe_manifest_id") != stable_id("budget_probe_manifest", body, 32):
        raise ValueError("Fixed-probe manifest content does not match its identity")
    probe._validate_grid(manifest.get("grid", []))
    generation = {**legacy.PRIMARY_GENERATION, "max_new_tokens": probe.MAX_NEW_TOKENS}
    rebuilt_grid = grid_entries(selected, parent["checkpoints"], (probe.CHECKPOINT_ALIAS,), (probe.LOGICAL_SEED,), generation, corpus_hash)
    if canonical_json(manifest["grid"]) != canonical_json(rebuilt_grid):
        raise ValueError("Fixed-probe grid does not reconstruct from its source parent")
    for name, artifact in manifest["evidence"]["artifacts"].items():
        path = prior_paths[name]
        if sha256_file(path) != artifact.get("sha256"):
            raise ValueError(f"Copied preceding probe evidence hash mismatch: {name}")
    selected_prompt_ids = {row["prompt_id"] for row in selected}
    profiles = []
    for cap in (256, 512):
        verification = _read_json(prior_paths[f"development_{cap}_verification"])
        if verification.get("status") not in {"passed", "pass"} or verification.get("protocol_id") != parent["protocol_id"]:
            raise ValueError(f"Copied {cap}-token verification is not passed for the source parent")
        verified = verification.get("verification", {})
        if verified.get("verified_responses") != 8 or verified.get("missing_responses") != 0:
            raise ValueError(f"Copied {cap}-token verification is incomplete")
        responses_path = prior_paths[f"development_{cap}_responses"]
        if verification.get("export", {}).get("output_sha256") != sha256_file(responses_path):
            raise ValueError(f"Copied {cap}-token verification/export hash mismatch")
        parent_key = "primary_256_grid" if cap == 256 else "longer_512_grid"
        expected_grid = [entry for entry in parent["development"][parent_key] if entry["prompt_id"] in selected_prompt_ids]
        evidence_rows = list(iter_jsonl(responses_path))
        profile = probe._validate_evidence_rows(evidence_rows, expected_grid, cap, parent)
        expected_by_id = {entry["response_id"]: entry for entry in expected_grid}
        for row in evidence_rows:
            validate_response(row, expected_by_id[row["response_id"]], parent["protocol_id"], profile)
        profiles.append(profile)
    if canonical_json(profiles[0]) != canonical_json(profiles[1]):
        raise ValueError("Copied preceding 256/512 evidence uses different profiles")
    expected_evidence = {
        "artifacts": manifest["evidence"]["artifacts"],
        "validated_shared_execution_profile": profiles[0],
        "validated_shared_execution_profile_id": profiles[0]["execution_profile_id"],
    }
    if canonical_json(manifest["evidence"]) != canonical_json(expected_evidence):
        raise ValueError("Fixed-probe evidence identity does not reconstruct")
    expected_core = {
        "manifest_schema_version": probe.MANIFEST_SCHEMA_VERSION,
        "probe_runner_version": probe.PROBE_RUNNER_VERSION,
        "classification": "exploratory_development_completion_budget_probe_not_confirmatory",
        "purpose": "assess natural completion behavior after six of eight matched 512-token generations reached their cap",
        "parent_protocol": {
            "protocol_id": parent["protocol_id"],
            "sha256": sha256_file(parent_path),
            "runner_version": parent["runner_version"],
            "declared_source_sha256": parent["source_sha256"],
        },
        "source_sha256": probe._source_hashes(),
        "prompt_corpus_sha256": corpus_hash,
        "evidence": expected_evidence,
        "fixed_scope": {
            "scenario_family_ids": list(probe.FAMILY_IDS),
            "checkpoint_alias": probe.CHECKPOINT_ALIAS,
            "logical_seed": probe.LOGICAL_SEED,
            "max_new_tokens": probe.MAX_NEW_TOKENS,
            "expected_responses": probe.EXPECTED_RESPONSES,
        },
        "checkpoint": parent["checkpoints"][probe.CHECKPOINT_ALIAS],
        "generation_config": generation,
        "grid": rebuilt_grid,
        "expected_responses": probe.EXPECTED_RESPONSES,
    }
    expected_manifest = {**expected_core, "probe_manifest_id": stable_id("budget_probe_manifest", expected_core, 32)}
    if canonical_json(manifest) != canonical_json(expected_manifest):
        raise ValueError("Fixed-probe manifest does not fully reconstruct from copied evidence")
    return manifest, parent


def _validate_probe_package(parent_path: Path, manifest_path: Path, run_dir: Path, export_path: Path, prior_paths: dict[str, Path]) -> dict[str, Any]:
    parent_path = fs_path(parent_path)
    manifest_path = fs_path(manifest_path)
    run_dir = fs_path(run_dir)
    export_path = fs_path(export_path)
    prior_paths = {name: fs_path(path) for name, path in prior_paths.items()}
    try:
        from . import development_budget_probe as probe
    except ImportError:  # pragma: no cover
        import development_budget_probe as probe
    manifest, _parent = _validate_probe_manifest_explicit(parent_path, manifest_path, prior_paths)
    config = _read_json(run_dir / "run_config.json")
    execution = config.get("execution_profile")
    execution_body = {key: value for key, value in execution.items() if key != "execution_profile_id"} if isinstance(execution, dict) else {}
    if not isinstance(execution, dict) or execution.get("execution_profile_id") != stable_id("execution", execution_body, 32):
        raise ValueError("Fixed-probe run execution profile is invalid")
    if canonical_json(execution) != canonical_json(manifest["evidence"]["validated_shared_execution_profile"]):
        raise ValueError("Fixed-probe run execution profile differs from its matched 256/512 evidence")
    expected_config = probe._run_config(manifest_path, parent_path, manifest, execution)
    # Path-dependent byte hashes are allowed in this evidence run config; after
    # relocation, reconstruct all scientific fields and separately validate the
    # copied snapshot hashes instead of rebuilding those two byte-hash fields.
    for key, value in expected_config.items():
        if key not in {"probe_manifest_sha256", "parent_protocol_sha256", "run_id"} and config.get(key) != value:
            raise ValueError(f"Fixed-probe run config field changed: {key}")
    if config.get("probe_manifest_sha256") != sha256_file(manifest_path) or config.get("parent_protocol_sha256") != sha256_file(parent_path):
        raise ValueError("Fixed-probe run snapshot hashes mismatch")
    config_body = {key: value for key, value in config.items() if key != "run_id"}
    if config.get("run_id") != stable_id("budget_probe_run", config_body, 32) or run_dir.name != config.get("run_id"):
        raise ValueError("Fixed-probe run identity or directory mismatch")
    if sha256_file(run_dir / "budget_probe_manifest.json") != config["probe_manifest_sha256"] or sha256_file(run_dir / "parent_protocol.json") != config["parent_protocol_sha256"]:
        raise ValueError("Fixed-probe embedded snapshots mismatch")
    expected = {entry["response_id"]: entry for entry in manifest["grid"]}
    response_rows = []
    for path in (run_dir / "responses").glob("*.json"):
        row = _read_json(path)
        if path.stem != row.get("response_id") or row.get("response_id") not in expected:
            raise ValueError("Fixed-probe response cache contains an unexpected row")
        probe.validate_response(row, expected[row["response_id"]], manifest, execution)
        validate_response(row, expected[row["response_id"]], manifest["probe_manifest_id"], execution)
        response_rows.append(row)
    if len(response_rows) != probe.EXPECTED_RESPONSES or len({row["response_id"] for row in response_rows}) != probe.EXPECTED_RESPONSES:
        raise ValueError("Fixed 1,024-token probe is not complete and verified 8/8")
    rows = list(iter_jsonl(export_path))
    rows.sort(key=lambda row: row.get("response_id", ""))
    response_rows.sort(key=lambda row: row.get("response_id", ""))
    if len(rows) != probe.EXPECTED_RESPONSES or canonical_json(rows) != canonical_json(response_rows):
        raise ValueError("Probe export is not the exact complete verified response set")
    return {
        "probe_manifest_id": manifest["probe_manifest_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "run_id": _read_json(run_dir / "run_config.json")["run_id"],
        "run_tree_sha256": _tree_hash(run_dir),
        "export_sha256": sha256_file(export_path),
        "verified_responses": len(response_rows),
        "fixed_max_new_tokens": manifest["fixed_scope"]["max_new_tokens"],
    }


def _validate_budget_decision(path: Path, parent: dict[str, Any], probe_evidence: dict[str, Any], primary_cap: int, sensitivity_cap: int) -> dict[str, Any]:
    path = fs_path(path)
    decision = _read_json(path)
    required = {
        "decision_schema_version": BUDGET_DECISION_SCHEMA_VERSION,
        "status": "COMPLETE_DEVELOPMENT_BUDGET_DECISION",
        "source_parent_protocol_id": parent["protocol_id"],
        "probe_manifest_id": probe_evidence["probe_manifest_id"],
        "probe_export_sha256": probe_evidence["export_sha256"],
        "primary_cap": primary_cap,
        "sensitivity_cap": sensitivity_cap,
    }
    for key, expected in required.items():
        if decision.get(key) != expected:
            raise ValueError(f"Budget decision {key} must equal {expected!r}")
    for field in ("decided_by", "decided_at", "rationale"):
        if not str(decision.get(field, "")).strip():
            raise ValueError(f"Budget decision requires non-empty {field}")
    claimed = decision.get("budget_decision_id")
    body = {key: value for key, value in decision.items() if key != "budget_decision_id"}
    if claimed != stable_id("budget_decision", body, 32):
        raise ValueError("Budget decision content does not match budget_decision_id")
    return decision


def _copy_tree_immutable(source: Path, destination: Path) -> None:
    source = fs_path(source)
    destination = fs_path(destination)
    for item in sorted(source.rglob("*")):
        if item.is_file():
            target = destination / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_copy_bytes(item, target, immutable=True)


def _write_evidence_zip(source: Path, destination: Path) -> None:
    source = fs_path(source)
    destination = fs_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                info = zipfile.ZipInfo(item.relative_to(source).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, item.read_bytes())
    if destination.exists():
        if destination.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise FileExistsError(f"Refusing to overwrite incompatible evidence bundle: {destination}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)


def _materialize_evidence_root(expected_root: Path) -> Path:
    expected_root = fs_path(expected_root)
    if expected_root.is_dir():
        return expected_root
    bundle = expected_root.with_suffix(".zip")
    if not bundle.is_file():
        raise FileNotFoundError(f"Missing evidence directory or bundle beside protocol: {expected_root}")
    temporary = fs_path(tempfile.mkdtemp(prefix="extended_evidence_"))
    atexit.register(shutil.rmtree, temporary, ignore_errors=True)
    with zipfile.ZipFile(bundle) as archive:
        names = [Path(name) for name in archive.namelist()]
        if any(name.is_absolute() or ".." in name.parts for name in names):
            raise ValueError("Unsafe path in evidence bundle")
        archive.extractall(temporary)
    return temporary


def freeze_protocol(
    parent_protocol_path: Path,
    primary_cap: int,
    sensitivity_cap: int,
    output: Path,
    *,
    probe_manifest_path: Path,
    probe_run_dir: Path,
    probe_export_path: Path,
    budget_decision_path: Path,
) -> dict[str, Any]:
    parent_protocol_path = fs_path(parent_protocol_path)
    output = fs_path(output)
    probe_manifest_path = fs_path(probe_manifest_path)
    probe_run_dir = fs_path(probe_run_dir)
    probe_export_path = fs_path(probe_export_path)
    budget_decision_path = fs_path(budget_decision_path)
    candidate = build_protocol(parent_protocol_path, primary_cap, sensitivity_cap)
    parent = _parent_protocol(parent_protocol_path)
    probe_manifest = _read_json(probe_manifest_path)
    prior_paths = _probe_prior_paths_from_manifest(probe_manifest)
    probe_evidence = _validate_probe_package(parent_protocol_path, probe_manifest_path, probe_run_dir, probe_export_path, prior_paths)
    decision = _validate_budget_decision(budget_decision_path, parent, probe_evidence, primary_cap, sensitivity_cap)
    package = output.parent / "e"
    atomic_copy_bytes(parent_protocol_path, package / "parent.json", immutable=True)
    atomic_copy_bytes(probe_manifest_path, package / "probe.json", immutable=True)
    _copy_tree_immutable(probe_run_dir, package / "p" / probe_evidence["run_id"])
    atomic_copy_bytes(probe_export_path, package / "export.jsonl", immutable=True)
    atomic_copy_bytes(budget_decision_path, package / "decision.json", immutable=True)
    for name, source in prior_paths.items():
        atomic_copy_bytes(source, package / "q" / source.name, immutable=True)
    evidence = {
        "package_schema_version": "extended-study-freeze-evidence-v1",
        "relative_root": "e",
        "source_parent": {"relative_path": "parent.json", "sha256": sha256_file(parent_protocol_path)},
        "probe_manifest": {"relative_path": "probe.json", "sha256": sha256_file(probe_manifest_path)},
        "probe_run": {"relative_path": f"p/{probe_evidence['run_id']}", "tree_sha256": probe_evidence["run_tree_sha256"], "run_id": probe_evidence["run_id"]},
        "probe_export": {"relative_path": "export.jsonl", "sha256": probe_evidence["export_sha256"]},
        "budget_decision": {"relative_path": "decision.json", "sha256": sha256_file(budget_decision_path), "budget_decision_id": decision["budget_decision_id"]},
        "prior_probe_evidence": {
            name: {"relative_path": f"q/{source.name}", "sha256": sha256_file(source)}
            for name, source in sorted(prior_paths.items())
        },
    }
    frozen_core = {key: value for key, value in candidate.items() if key != "candidate_protocol_id"}
    frozen_core.update({"status": "FROZEN_PROTOCOL_PROMPTS_PENDING_HUMAN_AND_PROSPECTIVE_REVIEW", "freeze_evidence": evidence})
    # Preserve the legacy protocol-ID envelope so revision_analysis can consume
    # the archive unchanged; the extended fields make the body successor-unique.
    frozen = {**frozen_core, "protocol_id": stable_id("protocol", frozen_core, 32)}
    atomic_write_json(output, frozen, immutable=True)
    return frozen


def _evidence_paths(protocol_path: Path, protocol: dict[str, Any], evidence_root: Path | None) -> dict[str, Path]:
    protocol_path = fs_path(protocol_path)
    evidence = protocol["freeze_evidence"]
    expected_root = fs_path(evidence_root) if evidence_root is not None else protocol_path.parent / evidence["relative_root"]
    root = _materialize_evidence_root(expected_root)
    return {key: root / evidence[key]["relative_path"] for key in ("source_parent", "probe_manifest", "probe_run", "probe_export", "budget_decision")}


def load_protocol(path: Path, evidence_root: Path | None = None) -> dict[str, Any]:
    path = fs_path(path)
    protocol = _read_json(path)
    if (protocol.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION
            or protocol.get("extended_schema_version") != EXTENDED_SCHEMA_VERSION
            or protocol.get("orchestration_version") != ORCHESTRATION_VERSION):
        raise ValueError("Unsupported or non-successor protocol schema")
    if "candidate_protocol_id" in protocol or protocol.get("status") == "CANDIDATE_NOT_FROZEN":
        raise ValueError("Candidate planning object is not a frozen successor protocol")
    claimed = protocol.get("protocol_id")
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if claimed != stable_id("protocol", body, 32):
        raise ValueError("Successor protocol content does not match protocol_id")
    if protocol.get("source_sha256") != source_hashes():
        raise ValueError("Current scientific sources differ from the frozen successor protocol")
    paths = _evidence_paths(path, protocol, evidence_root)
    evidence = protocol["freeze_evidence"]
    for key in ("source_parent", "probe_manifest", "probe_export", "budget_decision"):
        if sha256_file(paths[key]) != evidence[key]["sha256"]:
            raise ValueError(f"Frozen {key} evidence snapshot hash mismatch")
    if _tree_hash(paths["probe_run"]) != evidence["probe_run"]["tree_sha256"]:
        raise ValueError("Frozen probe-run evidence snapshot hash mismatch")
    materialized_root = paths["source_parent"].parent
    prior_paths = {
        name: materialized_root / row["relative_path"]
        for name, row in evidence.get("prior_probe_evidence", {}).items()
    }
    if set(prior_paths) != {"development_256_verification", "development_256_responses", "development_512_verification", "development_512_responses"}:
        raise ValueError("Frozen evidence package lacks preceding 256/512 probe evidence")
    for name, prior_path in prior_paths.items():
        if sha256_file(prior_path) != evidence["prior_probe_evidence"][name]["sha256"]:
            raise ValueError(f"Frozen preceding evidence snapshot hash mismatch: {name}")
    probe_evidence = _validate_probe_package(paths["source_parent"], paths["probe_manifest"], paths["probe_run"], paths["probe_export"], prior_paths)
    parent = _parent_protocol(paths["source_parent"])
    decision = _validate_budget_decision(paths["budget_decision"], parent, probe_evidence, protocol["primary_cap"], protocol["sensitivity_cap"])
    if decision["budget_decision_id"] != evidence["budget_decision"]["budget_decision_id"]:
        raise ValueError("Budget-decision identity mismatch")
    expected = build_protocol(paths["source_parent"], protocol["primary_cap"], protocol["sensitivity_cap"])
    actual_candidate = {key: value for key, value in protocol.items() if key not in {"protocol_id", "freeze_evidence"}}
    actual_candidate["status"] = "CANDIDATE_NOT_FROZEN"
    if canonical_json(actual_candidate) != canonical_json({key: value for key, value in expected.items() if key != "candidate_protocol_id"}):
        raise ValueError("Successor protocol grids or pinned identities do not reconstruct")
    _validate_grid(protocol)
    return protocol


def _mode_grid(protocol: dict[str, Any], mode: str, family_ids: list[str] | None) -> tuple[list[dict[str, Any]], str]:
    if mode == "evaluation":
        if family_ids:
            raise ValueError("Evaluation mode cannot subset the 2,880-response grid")
        return protocol["primary_grid"], "confirmatory_evaluation_after_all_readiness_gates"
    if mode == "sensitivity":
        if family_ids:
            raise ValueError("Sensitivity mode cannot subset the pre-specified grid")
        return protocol["longer_budget_sensitivity"]["grid"], "pre_specified_higher_budget_sensitivity"
    if mode in {"development-primary", "development-longer"}:
        grid = protocol["development"]["primary_grid" if mode.endswith("primary") else "longer_grid"]
        if family_ids:
            requested = set(family_ids)
            known = {row["scenario_family_id"] for row in grid}
            if not requested <= known:
                raise ValueError(f"Unknown development families: {sorted(requested - known)}")
            grid = [row for row in grid if row["scenario_family_id"] in requested]
        return grid, "exploratory_development_only"
    raise ValueError("mode must be evaluation, sensitivity, development-primary, or development-longer")


def _readiness_values(readiness: Mapping[str, Path] | None) -> dict[str, Path]:
    values = dict(readiness or {})
    if set(values) != set(READINESS_KEYS):
        raise ValueError(f"Readiness mapping must contain exactly: {', '.join(READINESS_KEYS)}")
    return {key: fs_path(values[key]) for key in READINESS_KEYS}


def _validate_readiness(protocol_path: Path, readiness: Mapping[str, Path] | None) -> tuple[dict[str, Path], dict[str, str]]:
    protocol_path = fs_path(protocol_path)
    values = _readiness_values(readiness)
    try:
        from .extended_prospective_validation import validate_plan
    except ImportError:  # pragma: no cover
        from extended_prospective_validation import validate_plan
    validate_plan(
        values["plan_path"], protocol_path, values["calibration_manifest_path"],
        values["calibration_responses_path"], values["calibration_run_dir"],
        values["evidence_path"], values["guide_path"],
    )
    hashes = {key: (_tree_hash(value) if key == "calibration_run_dir" else sha256_file(value)) for key, value in values.items()}
    return values, hashes


def _snapshot_readiness(values: dict[str, Path], run_dir: Path) -> None:
    run_dir = fs_path(run_dir)
    values = {key: fs_path(value) for key, value in values.items()}
    root = run_dir / "readiness"
    for key, source in values.items():
        if key == "calibration_run_dir":
            _copy_tree_immutable(source, root / key / source.name)
        else:
            atomic_copy_bytes(source, root / f"{key}{source.suffix}", immutable=True)


def _snapshot_readiness_mapping(run_dir: Path) -> dict[str, Path]:
    run_dir = fs_path(run_dir)
    root = run_dir / "readiness"
    run_roots = [item for item in (root / "calibration_run_dir").iterdir() if item.is_dir()]
    if len(run_roots) != 1:
        raise ValueError("Run readiness calibration snapshot is missing or ambiguous")
    values: dict[str, Path] = {"calibration_run_dir": run_roots[0]}
    for key in READINESS_KEYS:
        if key == "calibration_run_dir":
            continue
        matches = list(root.glob(f"{key}.*"))
        if len(matches) != 1:
            raise ValueError(f"Run readiness snapshot for {key} is missing or ambiguous")
        values[key] = matches[0]
    return values


def prepare_run(
    protocol_path: Path,
    mode: str,
    output_root: Path,
    family_ids: list[str] | None,
    human_review_path: Path | None,
    execution: dict[str, Any],
    *,
    readiness: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    protocol_path = fs_path(protocol_path)
    output_root = fs_path(output_root)
    human_review_path = fs_path(human_review_path) if human_review_path is not None else None
    protocol = load_protocol(protocol_path)
    grid, classification = _mode_grid(protocol, mode, family_ids)
    review_hash = None
    readiness_values: dict[str, Path] = {}
    readiness_hashes = None
    if mode in {"evaluation", "sensitivity"}:
        if human_review_path is None:
            raise ValueError(f"{mode} requires a genuine completed human-review manifest")
        validate_human_review(protocol, human_review_path)
        review_hash = sha256_file(human_review_path)
        readiness_values, readiness_hashes = _validate_readiness(protocol_path, readiness)
    elif human_review_path is not None or readiness is not None:
        raise ValueError("Development runs must not claim evaluation readiness gates")
    config_core = {
        "run_schema_version": RUN_SCHEMA_VERSION,
        "orchestration_version": ORCHESTRATION_VERSION,
        "runner_version": RUNNER_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "mode": mode,
        "classification": classification,
        "human_review_manifest_sha256": review_hash,
        "readiness_sha256": readiness_hashes,
        "scenario_family_ids": sorted({entry["scenario_family_id"] for entry in grid}),
        "execution_profile": execution,
        "response_ids": [entry["response_id"] for entry in grid],
        "expected_responses": len(grid),
    }
    config = {**config_core, "run_id": stable_id("extended_run", config_core, 32)}
    run_dir = output_root / "runs" / config["run_id"]
    atomic_write_json(run_dir / "run_config.json", config, immutable=True)
    atomic_copy_bytes(protocol_path, run_dir / "protocol.json", immutable=True)
    source_evidence = _materialize_evidence_root(protocol_path.parent / protocol["freeze_evidence"]["relative_root"])
    _write_evidence_zip(source_evidence, run_dir / f"{protocol['freeze_evidence']['relative_root']}.zip")
    if mode in {"evaluation", "sensitivity"}:
        atomic_copy_bytes(human_review_path, run_dir / "human_review_manifest.json", immutable=True)
        _snapshot_readiness(readiness_values, run_dir)
    return protocol, grid, run_dir


def context_preflight(protocol: dict[str, Any], entries: list[dict[str, Any]], model: Any, tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if not template or sha256_bytes(template.encode("utf-8")) != protocol["checkpoints"][entries[0]["checkpoint_alias"]]["chat_template_sha256"]:
        raise ValueError("Loaded tokenizer official chat template differs from the frozen checkpoint")
    context_limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not isinstance(context_limit, int) or context_limit < 1:
        raise ValueError("Pinned model does not expose a positive context limit")
    counts: dict[str, int] = {}
    for entry in entries:
        serialized = tokenizer.apply_chat_template([{"role": "user", "content": entry["prompt"]}], add_generation_prompt=True, tokenize=False)
        count = len(tokenizer.encode(serialized, add_special_tokens=False))
        counts[entry["prompt_id"]] = count
        for cap in (protocol["primary_cap"], protocol["sensitivity_cap"]):
            if count + cap > context_limit:
                raise ValueError(f"Prompt {entry['prompt_id']} plus cap {cap} exceeds context limit {context_limit}")
    return {
        "official_template_sha256": sha256_bytes(template.encode("utf-8")),
        "context_limit": context_limit,
        "primary_cap": protocol["primary_cap"],
        "sensitivity_cap": protocol["sensitivity_cap"],
        "input_tokens_by_prompt_id": dict(sorted(counts.items())),
        "maximum_input_plus_primary": max(counts.values()) + protocol["primary_cap"],
        "maximum_input_plus_sensitivity": max(counts.values()) + protocol["sensitivity_cap"],
        "model_generation_eos_token_id": getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "tokenizer_pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def _signal_handler(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def run_generation(
    protocol_path: Path,
    mode: str,
    output_root: Path,
    family_ids: list[str] | None,
    human_review_path: Path | None,
    device_request: str,
    local_files_only: bool,
    torch_threads: int | None,
    *,
    readiness: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    protocol_path = fs_path(protocol_path)
    output_root = fs_path(output_root)
    human_review_path = fs_path(human_review_path) if human_review_path is not None else None
    execution = execution_profile(device_request, torch_threads)
    protocol, grid, run_dir = prepare_run(protocol_path, mode, output_root, family_ids, human_review_path, execution, readiness=readiness)
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    expected = {entry["response_id"]: entry for entry in grid}
    completed: set[str] = set()
    for path in response_dir.glob("*.json"):
        if path.stem not in expected or path.stem in completed:
            raise ValueError(f"Unexpected or duplicate cached response: {path.name}")
        row = _read_json(path)
        validate_response(row, expected[path.stem], protocol["protocol_id"], execution)
        completed.add(path.stem)
    pending = [entry for entry in grid if entry["response_id"] not in completed]
    if not pending:
        return {"run_dir": display_path(run_dir), "completed": len(grid), "new_responses": 0, "remaining": 0, "status": "complete"}
    attempt_id = f"attempt_{time.time_ns()}"
    started = utc_now()
    clock = time.monotonic()
    progress = run_dir / "progress.jsonl"
    append_jsonl_durable(progress, {"event": "attempt_started", "attempt_id": attempt_id, "at": started, "complete_before_attempt": len(completed), "pending_before_attempt": len(pending)})
    handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            handlers[signum] = signal.signal(signum, _signal_handler)
        except (ValueError, OSError):
            pass
    new_count = 0
    try:
        for alias in CHECKPOINTS:
            checkpoint_pending = [entry for entry in pending if entry["checkpoint_alias"] == alias]
            if not checkpoint_pending or STOP_REQUESTED:
                continue
            load_clock = time.monotonic()
            model, tokenizer, dtype = load_checkpoint(protocol["checkpoints"][alias], execution, local_files_only)
            load_seconds = time.monotonic() - load_clock
            context_clock = time.monotonic()
            preflight = context_preflight(protocol, checkpoint_pending, model, tokenizer)
            context_seconds = time.monotonic() - context_clock
            append_jsonl_durable(progress, {"event": "checkpoint_loaded_and_context_validated", "attempt_id": attempt_id, "at": utc_now(), "checkpoint_alias": alias, "model_load_seconds": load_seconds, "context_preflight_seconds": context_seconds, "dtype": dtype, "context_preflight": preflight})
            for entry in checkpoint_pending:
                row = entry_to_response(entry, protocol["checkpoints"][alias], model, tokenizer, execution, dtype, protocol["protocol_id"])
                validate_response(row, entry, protocol["protocol_id"], execution)
                atomic_write_json(response_dir / f"{entry['response_id']}.json", row, immutable=True)
                new_count += 1
                append_jsonl_durable(progress, {"event": "response_checkpointed", "attempt_id": attempt_id, "at": utc_now(), "response_id": entry["response_id"], "checkpoint_alias": alias, "completed": len(completed) + new_count, "expected": len(grid), "generation_seconds": row["generation"]["timing"]["generation_seconds"], "output_tokens": row["generation"]["output_tokens"], "max_new_tokens": row["generation"]["max_new_tokens"], "stop_reason": row["generation"]["stopping"]["reason"], "eos_reached": row["generation"]["stopping"]["eos_reached"], "cap_reached": row["generation"]["stopping"]["cap_reached"]})
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
    result = {"run_dir": display_path(run_dir), "completed": complete, "new_responses": new_count, "remaining": len(grid) - complete, "status": "complete" if complete == len(grid) else "interrupted_resumable", "attempt_id": attempt_id, "attempt_started_at": started, "attempt_finished_at": utc_now(), "attempt_wall_seconds": time.monotonic() - clock}
    atomic_write_json(run_dir / "attempts" / f"{attempt_id}.json", result, immutable=True)
    append_jsonl_durable(progress, {"event": "attempt_finished", **result})
    return result


def _validate_run_config(protocol_path: Path, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol_path = fs_path(protocol_path)
    run_dir = fs_path(run_dir)
    protocol = load_protocol(protocol_path)
    config = _read_json(run_dir / "run_config.json")
    if config.get("run_schema_version") != RUN_SCHEMA_VERSION or config.get("orchestration_version") != ORCHESTRATION_VERSION:
        raise ValueError("Run is not an extended-study run")
    grid, classification = _mode_grid(protocol, config.get("mode"), None)
    selected_families = set(config.get("scenario_family_ids", []))
    if config["mode"].startswith("development-"):
        grid = [entry for entry in grid if entry["scenario_family_id"] in selected_families]
    execution = config.get("execution_profile")
    if not isinstance(execution, dict):
        raise ValueError("Run lacks an execution profile")
    execution_body = {key: value for key, value in execution.items() if key != "execution_profile_id"}
    if execution.get("execution_profile_id") != stable_id("execution", execution_body, 32):
        raise ValueError("Run execution profile identity mismatch")
    expected_core = {
        "run_schema_version": RUN_SCHEMA_VERSION, "orchestration_version": ORCHESTRATION_VERSION,
        "runner_version": RUNNER_VERSION, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path), "mode": config["mode"],
        "classification": classification,
        "human_review_manifest_sha256": config.get("human_review_manifest_sha256"),
        "readiness_sha256": config.get("readiness_sha256"),
        "scenario_family_ids": sorted({entry["scenario_family_id"] for entry in grid}),
        "execution_profile": execution, "response_ids": [entry["response_id"] for entry in grid],
        "expected_responses": len(grid),
    }
    expected_config = {**expected_core, "run_id": stable_id("extended_run", expected_core, 32)}
    if canonical_json(config) != canonical_json(expected_config) or run_dir.name != config.get("run_id"):
        raise ValueError("Run config, grid, provenance, or directory identity mismatch")
    if sha256_file(run_dir / "protocol.json") != config["protocol_sha256"]:
        raise ValueError("Run protocol snapshot hash mismatch")
    load_protocol(run_dir / "protocol.json")
    if config["mode"] in {"evaluation", "sensitivity"}:
        review = run_dir / "human_review_manifest.json"
        if sha256_file(review) != config["human_review_manifest_sha256"]:
            raise ValueError("Human-review snapshot hash mismatch")
        validate_human_review(protocol, review)
        snapshots = _snapshot_readiness_mapping(run_dir)
        _validate_readiness(run_dir / "protocol.json", snapshots)
        observed = {key: (_tree_hash(path) if key == "calibration_run_dir" else sha256_file(path)) for key, path in snapshots.items()}
        if observed != config["readiness_sha256"]:
            raise ValueError("Readiness snapshot hash mismatch")
    elif config.get("human_review_manifest_sha256") is not None or config.get("readiness_sha256") is not None:
        raise ValueError("Development run claims evaluation readiness")
    return protocol, config, grid


def verify_run(protocol_path: Path, run_dir: Path, require_complete: bool = True) -> dict[str, Any]:
    protocol_path = fs_path(protocol_path)
    run_dir = fs_path(run_dir)
    protocol, config, grid = _validate_run_config(protocol_path, run_dir)
    execution = config["execution_profile"]
    expected = {entry["response_id"]: entry for entry in grid}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    errors: list[str] = []
    response_dir = run_dir / "responses"
    for path in response_dir.glob("*.json"):
        try:
            row = _read_json(path)
            response_id = row.get("response_id")
            if response_id in seen or path.stem != response_id or response_id not in expected:
                raise ValueError("duplicate, misnamed, or unexpected response")
            seen.add(response_id)
            validate_response(row, expected[response_id], protocol["protocol_id"], execution)
            rows.append(row)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    missing = sorted(set(expected) - seen)
    if require_complete and missing:
        errors.append(f"missing {len(missing)} expected responses")
    if errors:
        raise ValueError("Extended run verification failed:\n- " + "\n- ".join(errors[:20]))
    counts = Counter(row["generation"]["stopping"]["reason"] for row in rows)
    return {"status": "complete" if len(rows) == len(expected) else "partial_valid", "mode": config["mode"], "classification": config["classification"], "expected_responses": len(expected), "verified_responses": len(rows), "missing_responses": len(missing), "unique_response_ids": len(seen), "stopping_reasons": dict(sorted(counts.items())), "cap_rate": sum(row["generation"]["stopping"]["cap_reached"] for row in rows) / len(rows) if rows else None}


def export_run(protocol_path: Path, run_dir: Path, output: Path, require_complete: bool = True) -> dict[str, Any]:
    protocol_path = fs_path(protocol_path)
    run_dir = fs_path(run_dir)
    output = fs_path(output)
    report = verify_run(protocol_path, run_dir, require_complete=require_complete)
    rows = [_read_json(path) for path in (run_dir / "responses").glob("*.json")]
    rows.sort(key=lambda row: row["response_id"])
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Refusing to overwrite incompatible extended export: {output}")
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    return {**report, "output": display_path(output), "output_sha256": sha256_file(output)}


def predict_complete_evaluation(protocol_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    """Use the unchanged pinned detector on a verified 2,880-response archive."""
    protocol_path = fs_path(protocol_path)
    run_dir = fs_path(run_dir)
    output = fs_path(output)
    report = verify_run(protocol_path, run_dir, require_complete=True)
    if report["mode"] != "evaluation" or report["verified_responses"] != 2880:
        raise ValueError("Prediction export requires the complete 2,880-response evaluation run")
    # The old scorer is compatible because response/study/runner schemas remain
    # the truthful unchanged low-level generator identifiers.  Temporarily
    # copying is unnecessary: score explicitly with the same implementation.
    protocol = load_protocol(protocol_path)
    config = _read_json(run_dir / "run_config.json")
    rows = [_read_json(path) for path in (run_dir / "responses").glob("*.json")]
    rows.sort(key=lambda row: row["response_id"])
    dependencies = shared.prediction_dependencies(include_embedding=True, include_prefix=True)
    tokenizer_dependency = dependencies["prefix_window_tokenizer"]
    from transformers import AutoTokenizer
    window_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dependency["model_id"], revision=tokenizer_dependency["resolved_revision"], local_files_only=True)
    prepared = [(row, unit, text, metadata) for row in rows for unit, text, metadata in evaluation_windows(row["generation"]["response"], window_tokenizer, tokenizer_dependency)]
    model, exemplar_vectors = shared.load_embedding_detector(local_files_only=True, revision=dependencies["embedding_model"]["resolved_revision"])
    embedding, maxima = shared.embedding_flags(model, exemplar_vectors, [text for _row, _unit, text, _metadata in prepared])
    by_id = {entry["response_id"]: entry for entry in protocol["primary_grid"]}
    predictions = []
    variant = "raw_output_robustness_four_category_union_provenance_v2"
    for index, (row, unit, text, metadata) in enumerate(prepared):
        exact = shared.exact_flags(text)
        components = {category: {"exact": exact[category], "embedding": embedding[index][category], "union": exact[category] or embedding[index][category], "intersection": exact[category] and embedding[index][category], "embedding_max_similarity": maxima[index][category]} for category in shared.CATEGORIES}
        entry = by_id[row["response_id"]]
        alias = entry["checkpoint_alias"]
        size, raw_stage = alias.split("_", 1)
        identity = {"response_id": row["response_id"], "detector_version": shared.DETECTOR_VERSION, "prediction_variant": variant, "evaluation_unit": unit, "dependency_identity": dependencies}
        prediction = {"prediction_id": shared.stable_id("prediction", identity, 32), "prediction_schema_version": shared.PREDICTION_SCHEMA_VERSION, "study_version": row["study_version"], "protocol_id": protocol["protocol_id"], "run_id": config["run_id"], "execution_profile_id": config["execution_profile"]["execution_profile_id"], "response_id": row["response_id"], "scenario_family_id": row["scenario_family_id"], "split": row["split"], "condition": row["condition"], "model_id": row["model_id"], "checkpoint_alias": alias, "model_size": size, "stage": "final" if raw_stage == "instruct" else raw_stage, "seed": row["seed"], "generation_output_tokens": row["generation"]["output_tokens"], "generation_stop_reason": row["generation"]["stopping"]["reason"], "detector_version": shared.DETECTOR_VERSION, "prediction_variant": variant, "evaluation_unit": unit, "decision_field": "union", "dependency_identity": dependencies, "window_metadata": metadata, "threshold": shared.EMBEDDING_THRESHOLD, "predictions": components, "unavailable_categories": {"crisis_referral": "not_applicable: absent from raw_output_robustness four-category baseline"}}
        shared.validate_prediction_record(prediction)
        predictions.append(prediction)
    if len(predictions) != 8640 or len({row["prediction_id"] for row in predictions}) != 8640:
        raise ValueError("Prediction export must contain exactly 8,640 correctly identified rows")
    shared.write_jsonl_compatible(output, predictions, "prediction_id")
    return {"status": "complete", "responses": 2880, "prediction_rows": 8640, "evaluation_units": list(shared.EVALUATION_UNITS), "dependency_identity": dependencies, "output": display_path(output), "output_sha256": sha256_file(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    plan.add_argument("--primary-cap", type=int, required=True)
    plan.add_argument("--sensitivity-cap", type=int, required=True)
    freeze = sub.add_parser("freeze-protocol")
    freeze.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT_PROTOCOL)
    freeze.add_argument("--primary-cap", type=int, required=True)
    freeze.add_argument("--sensitivity-cap", type=int, required=True)
    freeze.add_argument("--probe-manifest", type=Path, required=True)
    freeze.add_argument("--probe-run-dir", type=Path, required=True)
    freeze.add_argument("--probe-export", type=Path, required=True)
    freeze.add_argument("--budget-decision", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("create-human-review")
    review.add_argument("--protocol", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--mode", choices=("evaluation", "sensitivity", "development-primary", "development-longer"), required=True)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--family-id", action="append")
    run.add_argument("--human-review", type=Path)
    run.add_argument("--plan", type=Path)
    run.add_argument("--calibration-manifest", type=Path)
    run.add_argument("--calibration-responses", type=Path)
    run.add_argument("--calibration-run-dir", type=Path)
    run.add_argument("--development-evidence", type=Path)
    run.add_argument("--annotation-guide", type=Path)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--torch-threads", type=int)
    run.add_argument("--execute", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--allow-partial", action="store_true")
    export = sub.add_parser("export")
    export.add_argument("--protocol", type=Path, required=True)
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    predict = sub.add_parser("predict")
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        candidate = build_protocol(args.parent_protocol, args.primary_cap, args.sensitivity_cap)
        print(json.dumps({"status": candidate["status"], "candidate_protocol_id": candidate["candidate_protocol_id"], "primary_cap": candidate["primary_cap"], "sensitivity_cap": candidate["sensitivity_cap"], "primary_expected_responses": candidate["primary_expected_responses"], "sensitivity_expected_responses": candidate["longer_budget_sensitivity"]["expected_responses"]}, indent=2))
    elif args.command == "freeze-protocol":
        frozen = freeze_protocol(args.parent_protocol, args.primary_cap, args.sensitivity_cap, args.output, probe_manifest_path=args.probe_manifest, probe_run_dir=args.probe_run_dir, probe_export_path=args.probe_export, budget_decision_path=args.budget_decision)
        print(json.dumps({"status": frozen["status"], "protocol_id": frozen["protocol_id"], "output": str(args.output)}, indent=2))
    elif args.command == "create-human-review":
        protocol = load_protocol(args.protocol)
        result = create_human_review_manifest(protocol, args.output)
        print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
    elif args.command == "run":
        readiness_args = {
            "plan_path": args.plan, "calibration_manifest_path": args.calibration_manifest,
            "calibration_responses_path": args.calibration_responses,
            "calibration_run_dir": args.calibration_run_dir,
            "evidence_path": args.development_evidence, "guide_path": args.annotation_guide,
        }
        readiness = readiness_args if any(value is not None for value in readiness_args.values()) else None
        if not args.execute:
            protocol = load_protocol(args.protocol)
            grid, classification = _mode_grid(protocol, args.mode, args.family_id)
            print(json.dumps({"status": "dry_run_no_generation", "execute_required": True, "mode": args.mode, "classification": classification, "responses": len(grid)}, indent=2))
        else:
            print(json.dumps(run_generation(args.protocol, args.mode, args.output_root, args.family_id, args.human_review, args.device, args.local_files_only, args.torch_threads, readiness=readiness), indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_run(args.protocol, args.run_dir, require_complete=not args.allow_partial), indent=2))
    elif args.command == "export":
        print(json.dumps(export_run(args.protocol, args.run_dir, args.output, require_complete=not args.allow_partial), indent=2))
    elif args.command == "predict":
        print(json.dumps(predict_complete_evaluation(args.protocol, args.run_dir, args.output), indent=2))


if __name__ == "__main__":
    main()

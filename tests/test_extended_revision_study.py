from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import development_budget_probe as probe  # noqa: E402
from experiments import extended_revision_study as extended  # noqa: E402
from experiments import revision_analysis  # noqa: E402

PARENT = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"


def response_for(entry: dict, protocol_id: str, checkpoint: dict, execution: dict) -> dict:
    identity = entry["response_identity"]
    return {
        "response_schema_version": extended.RESPONSE_SCHEMA_VERSION,
        "study_version": extended.shared.STUDY_VERSION,
        "runner_version": extended.RUNNER_VERSION,
        "protocol_id": protocol_id,
        "execution_profile_id": execution["execution_profile_id"],
        "execution_profile": execution,
        "response_id": entry["response_id"], "response_identity": identity,
        "prompt_id": entry["prompt_id"],
        "prompt_corpus_sha256": identity["prompt_corpus_sha256"],
        "prompt_text_sha256": identity["prompt_text_sha256"],
        "scenario_family_id": entry["scenario_family_id"], "split": entry["split"],
        "condition": entry["condition"], "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"], "tokenizer_revision": checkpoint["tokenizer_revision"],
        "model_hash": checkpoint["model_hash"], "serialization_id": checkpoint["serialization_id"],
        "seed": entry["logical_seed"], "rng_seed": identity["rng_seed"], "rng_scheme": identity["rng_scheme"],
        "generation": {
            "prompt": entry["prompt"], "serialized_prompt": "<official>synthetic</official>",
            "response": "Clearly synthetic response.", "input_tokens": 12, "output_tokens": 4,
            "max_new_tokens": identity["generation_config"]["max_new_tokens"],
            "stopping": {"reason": "eos_token", "eos_reached": True, "cap_reached": False},
            "config": {**identity["generation_config"], "effective_eos_token_ids": [2], "device": execution["device_class"], "dtype": f"torch.{execution['dtype']}", "execution_profile_id": execution["execution_profile_id"]},
            "timing": {"generation_seconds": 0.01, "generated_tokens_per_second": 400.0},
        },
    }


def make_frozen_successor(tmp_path: Path, primary_cap: int = 1024, sensitivity_cap: int = 1536) -> tuple[Path, dict]:
    manifest_path = tmp_path / "probe_manifest.json"
    manifest = probe.freeze_manifest(PARENT, manifest_path)
    execution = manifest["evidence"]["validated_shared_execution_profile"]
    run_config = probe._run_config(manifest_path, PARENT, manifest, execution)
    run_dir = tmp_path / "probe_runs" / run_config["run_id"]
    probe.core.atomic_write_json(run_dir / "run_config.json", run_config)
    probe.core.atomic_copy_bytes(manifest_path, run_dir / "budget_probe_manifest.json")
    probe.core.atomic_copy_bytes(PARENT, run_dir / "parent_protocol.json")
    rows = []
    for entry in manifest["grid"]:
        row = response_for(entry, manifest["probe_manifest_id"], manifest["checkpoint"], execution)
        row.update({
            "budget_probe_manifest_id": manifest["probe_manifest_id"],
            "budget_probe_runner_version": probe.PROBE_RUNNER_VERSION,
            "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
            "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
            "budget_probe_classification": manifest["classification"],
            "checkpoint_alias": probe.CHECKPOINT_ALIAS,
        })
        probe.validate_response(row, entry, manifest, execution)
        probe.core.atomic_write_json(run_dir / "responses" / f"{entry['response_id']}.json", row)
        rows.append(row)
    rows.sort(key=lambda row: row["response_id"])
    export_path = tmp_path / "probe_export.jsonl"
    export_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    decision_core = {
        "decision_schema_version": extended.BUDGET_DECISION_SCHEMA_VERSION,
        "status": "COMPLETE_DEVELOPMENT_BUDGET_DECISION",
        "source_parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "probe_manifest_id": manifest["probe_manifest_id"],
        "probe_export_sha256": extended.sha256_file(export_path),
        "primary_cap": primary_cap, "sensitivity_cap": sensitivity_cap,
        "decided_by": "synthetic-test-decision-author",
        "decided_at": "2026-09-15T00:00:00+00:00",
        "rationale": "Synthetic fixture exercising the evidence contract.",
    }
    decision = {**decision_core, "budget_decision_id": extended.stable_id("budget_decision", decision_core, 32)}
    decision_path = tmp_path / "budget_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    protocol_path = tmp_path / "successor" / "protocol.json"
    protocol = extended.freeze_protocol(PARENT, primary_cap, sensitivity_cap, protocol_path, probe_manifest_path=manifest_path, probe_run_dir=run_dir, probe_export_path=export_path, budget_decision_path=decision_path)
    return protocol_path, protocol


def test_candidate_is_read_only_complete_and_cap_dependent(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    candidate = extended.build_protocol(PARENT, 1024, 1536)
    assert list(tmp_path.iterdir()) == before
    assert candidate["protocol_schema_version"] == extended.legacy.PROTOCOL_SCHEMA_VERSION
    assert candidate["extended_schema_version"] == extended.EXTENDED_SCHEMA_VERSION
    assert len(candidate["primary_grid"]) == 2880
    assert len(candidate["longer_budget_sensitivity"]["grid"]) == 192
    assert len(candidate["development"]["primary_grid"]) == 40
    assert len(candidate["development"]["longer_grid"]) == 40
    matched = [row for row in candidate["primary_grid"] if row["prompt_id"] == candidate["primary_grid"][0]["prompt_id"] and row["logical_seed"] == 0]
    assert len({row["response_identity"]["rng_seed"] for row in matched}) == 1
    changed = extended.build_protocol(PARENT, 1025, 1536)
    assert candidate["primary_grid"][0]["response_id"] != changed["primary_grid"][0]["response_id"]
    assert candidate["primary_grid"][0]["response_identity"]["rng_seed"] == changed["primary_grid"][0]["response_identity"]["rng_seed"]
    prefixed_candidate = extended.build_protocol(extended.fs_path(PARENT), 1024, 1536)
    assert prefixed_candidate["candidate_protocol_id"] == candidate["candidate_protocol_id"]
    assert extended.canonical_json(prefixed_candidate) == extended.canonical_json(candidate)
    for bad in ((512, 1024), (1024, 1024), (1024, 900), (True, 1536)):
        with pytest.raises(ValueError):
            extended.build_protocol(PARENT, *bad)


def test_nested_tree_copy_creates_parents_and_is_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source
    while len(str(nested / "artifact.json.tmp")) <= 320:
        nested /= "nested_calibration_readiness_evidence"
    nested /= "artifact.json"
    extended.atomic_write_json(nested, {"synthetic": True}, immutable=True)
    destination = tmp_path / "missing" / "deep" / "copy"
    extended._copy_tree_immutable(source, destination)
    copied = destination / nested.relative_to(source)
    assert len(str(nested)) > 300
    assert extended.fs_path(copied).read_bytes() == extended.fs_path(nested).read_bytes()
    assert extended.display_path(extended.fs_path(copied)) == os.path.abspath(copied)
    if os.name == "nt":
        assert str(extended.fs_path(copied)).startswith("\\\\?\\")
        assert str(extended.fs_path(extended.fs_path(copied))) == str(extended.fs_path(copied))
        assert str(extended.fs_path(r"\\server\share\artifact.json")) == r"\\?\UNC\server\share\artifact.json"
    extended._copy_tree_immutable(source, destination)
    extended.atomic_write_json(nested, {"synthetic": False})
    with pytest.raises(FileExistsError, match="immutable artifact"):
        extended._copy_tree_immutable(source, destination)


def test_freeze_load_reconstructs_after_complete_package_relocation(tmp_path: Path) -> None:
    protocol_path, protocol = make_frozen_successor(tmp_path)
    assert extended.load_protocol(protocol_path)["protocol_id"] == protocol["protocol_id"]
    relocated = tmp_path / "relocated"
    shutil.move(str(protocol_path.parent), relocated)
    loaded = extended.load_protocol(relocated / "protocol.json")
    assert loaded["protocol_id"] == protocol["protocol_id"]
    assert loaded["primary_expected_responses"] == 2880


def test_rehashed_protocol_grid_and_dependency_tamper_are_rejected(tmp_path: Path) -> None:
    protocol_path, protocol = make_frozen_successor(tmp_path)
    changed = json.loads(json.dumps(protocol))
    changed["primary_grid"][0]["prompt"] += " tamper"
    body = {key: value for key, value in changed.items() if key != "protocol_id"}
    changed["protocol_id"] = extended.stable_id("protocol", body, 32)
    protocol_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="reconstruct"):
        extended.load_protocol(protocol_path)

    protocol_path, protocol = make_frozen_successor(tmp_path / "second")
    changed = json.loads(json.dumps(protocol))
    changed["source_sha256"]["experiments/revision_study.py"] = "0" * 64
    body = {key: value for key, value in changed.items() if key != "protocol_id"}
    changed["protocol_id"] = extended.stable_id("protocol", body, 32)
    protocol_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific sources"):
        extended.load_protocol(protocol_path)


def test_rehashed_probe_fixed_scope_and_profile_mismatch_are_rejected(tmp_path: Path) -> None:
    protocol_path, protocol = make_frozen_successor(tmp_path)
    evidence = protocol_path.parent / protocol["freeze_evidence"]["relative_root"]
    manifest_path = evidence / protocol["freeze_evidence"]["probe_manifest"]["relative_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixed_scope"]["expected_responses"] = 7
    body = {key: value for key, value in manifest.items() if key != "probe_manifest_id"}
    manifest["probe_manifest_id"] = extended.stable_id("budget_probe_manifest", body, 32)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    protocol["freeze_evidence"]["probe_manifest"]["sha256"] = extended.sha256_file(manifest_path)
    body = {key: value for key, value in protocol.items() if key != "protocol_id"}
    protocol["protocol_id"] = extended.stable_id("protocol", body, 32)
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="fully reconstruct"):
        extended.load_protocol(protocol_path)


def test_rehashed_preceding_evidence_cap_metadata_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "probe.json"
    manifest = probe.freeze_manifest(PARENT, manifest_path)
    original_paths = extended._probe_prior_paths_from_manifest(manifest)
    prior_paths = {}
    for name, source in original_paths.items():
        destination = tmp_path / "prior" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        prior_paths[name] = destination
    responses_path = prior_paths["development_256_responses"]
    rows = list(extended.iter_jsonl(responses_path))
    rows[0]["generation"]["max_new_tokens"] += 1
    responses_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    verification_path = prior_paths["development_256_verification"]
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["export"]["output_sha256"] = extended.sha256_file(responses_path)
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    for name in ("development_256_responses", "development_256_verification"):
        manifest["evidence"]["artifacts"][name]["sha256"] = extended.sha256_file(prior_paths[name])
    body = {key: value for key, value in manifest.items() if key != "probe_manifest_id"}
    manifest["probe_manifest_id"] = extended.stable_id("budget_probe_manifest", body, 32)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="generation cap"):
        extended._validate_probe_manifest_explicit(PARENT, manifest_path, prior_paths)


def test_probe_run_must_use_declared_matched_execution_profile(tmp_path: Path) -> None:
    protocol_path, protocol = make_frozen_successor(tmp_path)
    root = protocol_path.parent / protocol["freeze_evidence"]["relative_root"]
    evidence = protocol["freeze_evidence"]
    manifest_path = root / evidence["probe_manifest"]["relative_path"]
    run_dir = root / evidence["probe_run"]["relative_path"]
    export_path = root / evidence["probe_export"]["relative_path"]
    prior_paths = {name: root / row["relative_path"] for name, row in evidence["prior_probe_evidence"].items()}
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    profile = dict(config["execution_profile"])
    profile["torch_num_threads"] += 1
    profile_body = {key: value for key, value in profile.items() if key != "execution_profile_id"}
    profile["execution_profile_id"] = extended.stable_id("execution", profile_body, 32)
    config["execution_profile"] = profile
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its matched"):
        extended._validate_probe_package(root / evidence["source_parent"]["relative_path"], manifest_path, run_dir, export_path, prior_paths)


def test_context_preflight_checks_official_template_and_both_caps() -> None:
    protocol = extended.build_protocol(PARENT, 1024, 1536)
    entry = protocol["development"]["primary_grid"][0]
    alias = entry["checkpoint_alias"]

    class Model:
        class Config:
            max_position_embeddings = 4096
        class Generation:
            eos_token_id = 2
        config = Config()
        generation_config = Generation()

    class Tokenizer:
        chat_template = "template"
        eos_token_id = 2
        pad_token_id = 2
        def apply_chat_template(self, *_args, **_kwargs): return "serialized"
        def encode(self, *_args, **_kwargs): return list(range(40))

    protocol["checkpoints"][alias]["chat_template_sha256"] = extended.sha256_bytes(b"template")
    report = extended.context_preflight(protocol, [entry], Model(), Tokenizer())
    assert report["maximum_input_plus_primary"] == 1064
    assert report["maximum_input_plus_sensitivity"] == 1576
    Model.Config.max_position_embeddings = 1500
    with pytest.raises(ValueError, match="cap 1536"):
        extended.context_preflight(protocol, [entry], Model(), Tokenizer())


def test_successor_response_validator_rejects_cap_and_stop_tamper() -> None:
    protocol = extended.build_protocol(PARENT, 1024, 1536)
    entry = protocol["development"]["primary_grid"][0]
    execution_core = {"device_class": "cpu", "dtype": "float32"}
    execution = {**execution_core, "execution_profile_id": extended.stable_id("execution", execution_core, 32)}
    row = response_for(entry, "synthetic_protocol", protocol["checkpoints"][entry["checkpoint_alias"]], execution)
    extended.validate_response(row, entry, "synthetic_protocol", execution)
    changed = json.loads(json.dumps(row))
    changed["generation"]["max_new_tokens"] = 999
    with pytest.raises(ValueError, match="generation cap"):
        extended.validate_response(changed, entry, "synthetic_protocol", execution)
    changed = json.loads(json.dumps(row))
    changed["generation"]["output_tokens"] = 1024
    changed["generation"]["stopping"] = {"reason": "other_model_stop", "eos_reached": False, "cap_reached": False}
    with pytest.raises(ValueError, match="cannot both|cap flag"):
        extended.validate_response(changed, entry, "synthetic_protocol", execution)
    changed = json.loads(json.dumps(row))
    changed["generation"]["stopping"] = {"reason": "max_new_tokens", "eos_reached": True, "cap_reached": False}
    with pytest.raises(ValueError, match="reason and flags"):
        extended.validate_response(changed, entry, "synthetic_protocol", execution)
    changed = json.loads(json.dumps(row))
    changed["generation"]["output_tokens"] = 1024
    changed["generation"]["stopping"] = {"reason": "eos_token", "eos_reached": True, "cap_reached": True}
    with pytest.raises(ValueError, match="cannot both|cap flag"):
        extended.validate_response(changed, entry, "synthetic_protocol", execution)
    changed["generation"]["stopping"]["cap_reached"] = False
    extended.validate_response(changed, entry, "synthetic_protocol", execution)


def test_evaluation_requires_human_and_complete_readiness(tmp_path: Path) -> None:
    protocol_path, protocol = make_frozen_successor(tmp_path)
    execution = {"device_class": "cpu"}
    with pytest.raises(ValueError, match="human-review"):
        extended.prepare_run(protocol_path, "evaluation", tmp_path / "runs", None, None, execution)
    pending = tmp_path / "pending.json"
    extended.create_human_review_manifest(protocol, pending)
    with pytest.raises(ValueError, match="Human review"):
        extended.prepare_run(protocol_path, "evaluation", tmp_path / "runs", None, pending, execution, readiness={})


def test_unchanged_analyzer_accepts_synthetic_full_2880_by_8640_archive(tmp_path: Path) -> None:
    protocol_path, protocol = make_frozen_successor(tmp_path)
    execution_core = {
        "profile_schema_version": "revision-study-execution-profile-v1", "device_class": "cpu",
        "dtype": "float32", "torch_num_threads": 2, "single_response_generation": True,
        "python_version": "synthetic", "torch_version": "synthetic", "transformers_version": "synthetic",
        "cuda_runtime": None, "accelerator": None,
    }
    execution = {**execution_core, "execution_profile_id": extended.stable_id("execution", execution_core, 32)}
    responses = [response_for(entry, protocol["protocol_id"], protocol["checkpoints"][entry["checkpoint_alias"]], execution) for entry in protocol["primary_grid"]]
    response_path = tmp_path / "synthetic_responses.jsonl"
    response_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in responses), encoding="utf-8")
    dependencies = {
        "embedding_model": {"model_id": "synthetic/embedding", "resolved_revision": "1" * 40, "applied": True},
        "prefix_window_tokenizer": {"model_id": "synthetic/tokenizer", "resolved_revision": "2" * 40, "applied": True},
    }
    categories = revision_analysis.CATEGORIES
    predictions = []
    by_id = {entry["response_id"]: entry for entry in protocol["primary_grid"]}
    for row in responses:
        entry = by_id[row["response_id"]]
        alias = entry["checkpoint_alias"]
        size, raw_stage = alias.split("_", 1)
        for unit, limit in (("full_response", None), ("prefix_64_tokens", 64), ("prefix_128_tokens", 128)):
            identity = {"response_id": row["response_id"], "unit": unit}
            predictions.append({
                "prediction_id": extended.stable_id("prediction", identity, 32),
                "prediction_schema_version": revision_analysis.PREDICTION_SCHEMA_VERSION,
                "study_version": row["study_version"], "response_id": row["response_id"],
                "detector_version": revision_analysis.EXPECTED_DETECTOR_VERSION,
                "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
                "evaluation_unit": unit, "decision_field": "union", "dependency_identity": dependencies,
                "predictions": {category: {"exact": False, "embedding": False, "union": False, "intersection": False} for category in categories},
                "scenario_family_id": row["scenario_family_id"], "split": "evaluation", "condition": row["condition"],
                "model_id": row["model_id"], "seed": row["seed"], "checkpoint_alias": alias,
                "model_size": size, "stage": "final" if raw_stage == "instruct" else raw_stage,
                "window_metadata": {"tokenizer_model_id": "synthetic/tokenizer", "tokenizer_revision": "2" * 40, "full_response_tokens": 4, "window_token_limit": limit, "evaluated_tokens": 4, "shorter_than_window": None if limit is None else True, "truncated": False},
                "protocol_id": protocol["protocol_id"], "run_id": "synthetic_run",
                "execution_profile_id": execution["execution_profile_id"],
                "generation_output_tokens": 4, "generation_stop_reason": "eos_token",
            })
    prediction_path = tmp_path / "synthetic_predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
    validated = revision_analysis.load_and_validate(protocol_path, response_path, prediction_path)
    assert len(validated.responses) == 2880
    assert len(validated.predictions) == 8640

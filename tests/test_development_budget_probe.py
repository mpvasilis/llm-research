from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import development_budget_probe as probe  # noqa: E402


PARENT = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"


def execution_profile() -> dict:
    value = {
        "profile_schema_version": "revision-study-execution-profile-v1",
        "device_class": "cpu",
        "dtype": "float32",
        "torch_num_threads": 4,
        "single_response_generation": True,
        "python_version": "test",
        "torch_version": "test",
        "transformers_version": "test",
        "cuda_runtime": None,
        "accelerator": None,
    }
    return {**value, "execution_profile_id": probe.core.stable_id("execution", value, 32)}


def response_for(entry: dict, manifest: dict, execution: dict) -> dict:
    identity = entry["response_identity"]
    checkpoint = manifest["checkpoint"]
    row = {
        "response_schema_version": probe.core.RESPONSE_SCHEMA_VERSION,
        "study_version": probe.core.shared.STUDY_VERSION,
        "runner_version": probe.core.RUNNER_VERSION,
        "protocol_id": manifest["probe_manifest_id"],
        "execution_profile_id": execution["execution_profile_id"],
        "execution_profile": execution,
        "response_id": entry["response_id"],
        "response_identity": identity,
        "prompt_id": entry["prompt_id"],
        "prompt_corpus_sha256": identity["prompt_corpus_sha256"],
        "prompt_text_sha256": identity["prompt_text_sha256"],
        "scenario_family_id": entry["scenario_family_id"],
        "split": entry["split"],
        "condition": entry["condition"],
        "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "model_hash": checkpoint["model_hash"],
        "serialization_id": checkpoint["serialization_id"],
        "seed": entry["logical_seed"],
        "rng_seed": identity["rng_seed"],
        "rng_scheme": identity["rng_scheme"],
        "generation": {
            "prompt": entry["prompt"],
            "serialized_prompt": f"<user>{entry['prompt']}</user><assistant>",
            "response": "Mocked 1,024-token probe response.",
            "input_tokens": 40,
            "output_tokens": 10,
            "max_new_tokens": probe.MAX_NEW_TOKENS,
            "stopping": {"reason": "eos_token", "eos_reached": True, "cap_reached": False},
            "config": {
                **identity["generation_config"],
                "effective_eos_token_ids": [2],
                "device": "cpu",
                "dtype": "torch.float32",
                "execution_profile_id": execution["execution_profile_id"],
            },
            "timing": {"generation_seconds": 0.01, "generated_tokens_per_second": 1000.0},
        },
        "budget_probe_manifest_id": manifest["probe_manifest_id"],
        "budget_probe_runner_version": probe.PROBE_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "budget_probe_classification": manifest["classification"],
        "checkpoint_alias": probe.CHECKPOINT_ALIAS,
    }
    return row


def test_fixed_plan_is_read_only_balanced_and_bound_to_evidence(tmp_path: Path) -> None:
    before = list(tmp_path.iterdir())
    plan = probe.plan(PARENT)
    assert list(tmp_path.iterdir()) == before
    assert plan == {
        "status": "candidate_plan_not_frozen",
        "execute_required": True,
        "probe_manifest_id": plan["probe_manifest_id"],
        "classification": "exploratory_development_completion_budget_probe_not_confirmatory",
        "responses": 8,
        "scenario_family_ids": list(probe.FAMILY_IDS),
        "checkpoint_alias": "1b_instruct",
        "logical_seed": 0,
        "max_new_tokens": 1024,
    }
    manifest = probe.build_manifest(PARENT)
    assert len(manifest["grid"]) == 8
    assert Counter(row["scenario_family_id"] for row in manifest["grid"]) == Counter({family: 4 for family in probe.FAMILY_IDS})
    assert {row["split"] for row in manifest["grid"]} == {"development"}
    assert len(manifest["evidence"]["artifacts"]) == 4
    assert all(len(item["sha256"]) == 64 for item in manifest["evidence"]["artifacts"].values())
    assert manifest["evidence"]["validated_shared_execution_profile_id"].startswith("execution_")


def test_1024_ids_are_distinct_but_rng_matches_512() -> None:
    manifest = probe.build_manifest(PARENT)
    parent = probe.core.load_protocol(PARENT)
    prior = {
        row["prompt_id"]: row
        for row in parent["development"]["longer_512_grid"]
        if row["scenario_family_id"] in probe.FAMILY_IDS
    }
    assert len(prior) == 8
    for row in manifest["grid"]:
        older = prior[row["prompt_id"]]
        assert row["response_id"] != older["response_id"]
        assert row["response_identity"]["rng_seed"] == older["response_identity"]["rng_seed"]
        assert row["response_identity"]["generation_config"]["max_new_tokens"] == 1024


def test_calibration_preservation_is_a_separate_non_dependency_audit() -> None:
    audit = probe.preservation_audit(PARENT)
    assert audit["status"] == "pass"
    assert audit["calibration_actual_sha256"] == probe.EXPECTED_CALIBRATION_SOURCE_SHA256
    assert audit["calibration_is_probe_dependency"] is False
    manifest = probe.build_manifest(PARENT)
    assert "calibration" not in json.dumps(manifest).lower()


def test_parent_source_substitution_is_reported_and_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    actual = probe.core.source_hashes()
    substituted = dict(actual)
    changed_path = sorted(substituted)[0]
    substituted[changed_path] = "0" * 64
    monkeypatch.setattr(probe.core, "source_hashes", lambda: substituted)
    audit = probe.preservation_audit(PARENT)
    assert audit["status"] == "fail"
    assert audit["parent_source_hashes_valid"] is False
    assert audit["parent_source_comparisons"][changed_path]["match"] is False
    with pytest.raises(ValueError, match="Actual parent source hashes differ"):
        probe.build_manifest(PARENT)


def test_context_preflight_accepts_current_shape_and_rejects_overflow() -> None:
    manifest = probe.build_manifest(PARENT)

    class Model:
        class Config:
            max_position_embeddings = 4096
        config = Config()

    class Tokenizer:
        def apply_chat_template(self, _messages, add_generation_prompt: bool, tokenize: bool):
            assert add_generation_prompt is True and tokenize is False
            return "serialized"

        def encode(self, _text, add_special_tokens: bool):
            assert add_special_tokens is False
            return list(range(44))

    report = probe.context_preflight(manifest, Model(), Tokenizer())
    assert report["context_limit"] == 4096
    assert report["maximum_input_plus_generation"] == 1068

    Model.Config.max_position_embeddings = 1000
    with pytest.raises(ValueError, match="exceeds context limit"):
        probe.context_preflight(manifest, Model(), Tokenizer())


def test_mocked_generation_resume_verify_export_and_tamper_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "probe_manifest.json"
    manifest = probe.freeze_manifest(PARENT, manifest_path)
    execution = execution_profile()
    calls = []

    class Model:
        class Config:
            max_position_embeddings = 4096
        config = Config()

    class Tokenizer:
        def apply_chat_template(self, _messages, add_generation_prompt: bool, tokenize: bool):
            return "serialized"

        def encode(self, _text, add_special_tokens: bool):
            return list(range(40))

    monkeypatch.setattr(probe.core, "execution_profile", lambda *_args: execution)
    monkeypatch.setattr(probe.core, "load_checkpoint", lambda *_args: (Model(), Tokenizer(), "torch.float32"))

    def fake_response(entry, _checkpoint, _model, _tokenizer, observed_execution, _dtype, observed_manifest):
        calls.append(entry["response_id"])
        return response_for(entry, observed_manifest, observed_execution)

    monkeypatch.setattr(probe, "_response_with_probe_provenance", fake_response)
    output_root = tmp_path / "output"
    first = probe.run_generation(manifest_path, PARENT, output_root, "cpu", True, 4)
    assert first["new_responses"] == 8 and first["remaining"] == 0
    run_dir = Path(first["run_dir"])
    response_dir = run_dir / "responses"
    before = {path.name: probe.core.sha256_file(path) for path in response_dir.glob("*.json")}
    second = probe.run_generation(manifest_path, PARENT, output_root, "cpu", True, 4)
    after = {path.name: probe.core.sha256_file(path) for path in response_dir.glob("*.json")}
    assert second == {
        "run_dir": str(run_dir), "completed": 8, "new_responses": 0,
        "remaining": 0, "status": "complete",
    }
    assert before == after and len(calls) == 8
    verification = probe.verify_run(manifest_path, PARENT, run_dir)
    assert verification["status"] == "complete" and verification["verified_responses"] == 8
    exported = probe.export_run(manifest_path, PARENT, run_dir, tmp_path / "responses.jsonl")
    assert exported["responses"] == 8

    missing_path = next(response_dir.glob("*.json"))
    payload = missing_path.read_bytes()
    missing_path.unlink()
    with pytest.raises(ValueError, match="missing 1 expected responses"):
        probe.verify_run(manifest_path, PARENT, run_dir)
    missing_path.write_bytes(payload)

    row = json.loads(missing_path.read_text(encoding="utf-8"))
    row["execution_profile"] = {**execution, "torch_num_threads": 8}
    missing_path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="execution profile"):
        probe.verify_run(manifest_path, PARENT, run_dir)


def test_manifest_rejects_changed_evidence_identity(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = probe.freeze_manifest(PARENT, path)
    manifest["evidence"]["artifacts"]["development_512_verification"]["sha256"] = "0" * 64
    core_value = {key: value for key, value in manifest.items() if key != "probe_manifest_id"}
    manifest["probe_manifest_id"] = probe.core.stable_id("budget_probe_manifest", core_value, 32)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from current pinned dependencies or evidence"):
        probe.load_manifest(path, PARENT)


def test_evidence_checkpoint_metadata_and_profile_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reader = probe._read_jsonl

    def changed_model_hash(path: Path):
        rows = json.loads(json.dumps(original_reader(path)))
        if "256" in path.name:
            rows[0]["model_hash"] = "tampered-model-hash"
        return rows

    monkeypatch.setattr(probe, "_read_jsonl", changed_model_hash)
    with pytest.raises(ValueError, match="model_hash differs"):
        probe.build_manifest(PARENT)

    def changed_profile(path: Path):
        rows = json.loads(json.dumps(original_reader(path)))
        if "512" in path.name:
            rows[0]["execution_profile"]["torch_num_threads"] = 99
        return rows

    monkeypatch.setattr(probe, "_read_jsonl", changed_profile)
    with pytest.raises(ValueError, match="one complete execution profile"):
        probe.build_manifest(PARENT)


def test_verification_must_report_the_actual_export_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = {}
    for name, source in probe.EVIDENCE_PATHS.items():
        destination = tmp_path / source.name
        destination.write_bytes(source.read_bytes())
        copied[name] = destination
    verification_path = copied["development_512_verification"]
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["export"]["output_sha256"] = "0" * 64
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(probe, "EVIDENCE_PATHS", copied)
    with pytest.raises(ValueError, match="verification/export SHA-256 mismatch"):
        probe.build_manifest(PARENT)

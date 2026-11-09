from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import calibration_study as calibration  # noqa: E402


PARENT_PROTOCOL = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"


def cpu_profile() -> dict:
    value = {
        "profile_schema_version": "revision-study-execution-profile-v1",
        "device_class": "cpu",
        "dtype": "float32",
        "torch_num_threads": 2,
        "single_response_generation": True,
        "python_version": "test",
        "torch_version": "test",
        "transformers_version": "test",
        "cuda_runtime": None,
        "accelerator": None,
    }
    return {**value, "execution_profile_id": calibration.core.stable_id("execution", value, 32)}


def base_response(entry: dict, manifest: dict, execution: dict) -> dict:
    identity = entry["response_identity"]
    checkpoint = manifest["checkpoints"][entry["checkpoint_alias"]]
    return {
        "response_schema_version": calibration.core.RESPONSE_SCHEMA_VERSION,
        "study_version": calibration.core.shared.STUDY_VERSION,
        "runner_version": calibration.core.RUNNER_VERSION,
        "protocol_id": manifest["calibration_manifest_id"],
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
            "response": "Mocked calibration response.",
            "input_tokens": 12,
            "output_tokens": 5,
            "max_new_tokens": identity["generation_config"]["max_new_tokens"],
            "stopping": {"reason": "eos_token", "eos_reached": True, "cap_reached": False},
            "config": {
                **identity["generation_config"],
                "effective_eos_token_ids": [2],
                "device": "cpu",
                "dtype": "torch.float32",
                "execution_profile_id": execution["execution_profile_id"],
            },
            "timing": {
                "generation_seconds": 0.01,
                "generated_tokens_per_second": 500.0,
                "scope": "mocked single generation call",
            },
        },
    }


@pytest.fixture()
def frozen_manifest(tmp_path: Path) -> tuple[Path, dict]:
    path = tmp_path / "calibration_manifest.json"
    manifest = calibration.freeze_manifest(PARENT_PROTOCOL, 256, path)
    return path, manifest


def test_complete_grid_has_required_rotation_and_no_evaluation_rows(frozen_manifest: tuple[Path, dict]) -> None:
    _path, manifest = frozen_manifest
    grid = manifest["grid"]
    assert len(grid) == 96
    assert len({row["response_id"] for row in grid}) == 96
    assert {row["split"] for row in grid} == {"development"}
    assert {row["logical_seed"] for row in grid} == {0}
    assert Counter(row["checkpoint_alias"] for row in grid) == Counter({alias: 16 for alias in calibration.CHECKPOINT_ALIASES})
    assert len({row["scenario_family_id"] for row in grid}) == 10
    for stage in calibration.STAGES:
        aliases = [f"1b_{stage}", f"7b_{stage}"]
        family_sets = [
            {row["scenario_family_id"] for row in grid if row["checkpoint_alias"] == alias}
            for alias in aliases
        ]
        assert family_sets[0] == family_sets[1]
        assert Counter(row["condition"]["domain"] for row in grid if row["checkpoint_alias"] == aliases[0]) == {"health": 8, "everyday": 8}


def test_recalculated_manifest_identity_cannot_hide_evaluation_leakage(
    frozen_manifest: tuple[Path, dict], tmp_path: Path,
) -> None:
    _path, manifest = frozen_manifest
    broken = copy.deepcopy(manifest)
    broken["grid"][0]["split"] = "evaluation"
    core_value = {key: value for key, value in broken.items() if key != "calibration_manifest_id"}
    broken["calibration_manifest_id"] = calibration.core.stable_id("calibration_manifest", core_value, 32)
    path = tmp_path / "leaking_manifest.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="Evaluation-family leakage"):
        calibration.load_manifest(path, PARENT_PROTOCOL)


def test_plan_is_read_only_and_freeze_refuses_cap_different_from_parent(
    tmp_path: Path,
) -> None:
    before = list(tmp_path.iterdir())
    candidate = calibration.plan(PARENT_PROTOCOL, 512)
    assert candidate["status"] == "candidate_plan_not_frozen"
    assert candidate["max_new_tokens"] == 512
    assert list(tmp_path.iterdir()) == before
    output = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="must match the parent protocol primary cap"):
        calibration.freeze_manifest(PARENT_PROTOCOL, 512, output)
    assert not output.exists()


def test_mocked_complete_run_export_missing_and_provenance_tamper(
    frozen_manifest: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = frozen_manifest
    execution = cpu_profile()
    monkeypatch.setattr(calibration.core, "execution_profile", lambda *_args: execution)
    monkeypatch.setattr(calibration.core, "load_checkpoint", lambda *_args: (object(), object(), "torch.float32"))
    calls: list[str] = []

    def fake_generate(entry, _checkpoint, _model, _tokenizer, observed_execution, _dtype, _protocol_id):
        calls.append(entry["response_id"])
        return base_response(entry, manifest, observed_execution)

    monkeypatch.setattr(calibration.core, "entry_to_response", fake_generate)
    first = calibration.run_generation(manifest_path, PARENT_PROTOCOL, tmp_path / "output", "cpu", True, 2)
    assert first["new_responses"] == 96
    assert len(calls) == 96
    run_dir = Path(first["run_dir"])
    progress = list(calibration.core.iter_jsonl(run_dir / "progress.jsonl"))
    loaded = [event for event in progress if event["event"] == "checkpoint_loaded"]
    saved = [event for event in progress if event["event"] == "response_checkpointed"]
    assert len(loaded) == 6 and all("model_load_seconds" in event for event in loaded)
    assert len(saved) == 96 and all("generation_seconds" in event and "output_tokens" in event for event in saved)
    report = calibration.verify_run(manifest_path, PARENT_PROTOCOL, run_dir)
    assert report["status"] == "complete"
    assert report["unique_response_ids"] == 96
    exported = calibration.export_run(manifest_path, PARENT_PROTOCOL, run_dir, tmp_path / "responses.jsonl")
    assert exported["responses"] == 96

    response_path = next((run_dir / "responses").glob("*.json"))
    original = response_path.read_text(encoding="utf-8")
    tampered = json.loads(original)
    tampered["parent_protocol_id"] = "protocol_tampered"
    response_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="parent_protocol_id mismatch"):
        calibration.verify_run(manifest_path, PARENT_PROTOCOL, run_dir)
    response_path.write_text(original, encoding="utf-8")
    response_path.unlink()
    with pytest.raises(ValueError, match="missing 1 expected responses"):
        calibration.verify_run(manifest_path, PARENT_PROTOCOL, run_dir)


def test_complete_resume_generates_zero_new_responses(
    frozen_manifest: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = frozen_manifest
    execution = cpu_profile()
    monkeypatch.setattr(calibration.core, "execution_profile", lambda *_args: execution)
    monkeypatch.setattr(calibration.core, "load_checkpoint", lambda *_args: (object(), object(), "torch.float32"))
    calls = 0

    def fake_generate(entry, _checkpoint, _model, _tokenizer, observed_execution, _dtype, _protocol_id):
        nonlocal calls
        calls += 1
        return base_response(entry, manifest, observed_execution)

    monkeypatch.setattr(calibration.core, "entry_to_response", fake_generate)
    first = calibration.run_generation(manifest_path, PARENT_PROTOCOL, tmp_path / "output", "cpu", True, 2)
    run_dir = Path(first["run_dir"])
    config_path = run_dir / "run_config.json"
    config_before = config_path.read_bytes()
    response_hashes_before = {
        path.name: calibration.core.sha256_file(path)
        for path in (run_dir / "responses").glob("*.json")
    }
    config = json.loads(config_before)
    assert config["source_sha256"] == calibration._source_hashes()
    second = calibration.run_generation(manifest_path, PARENT_PROTOCOL, tmp_path / "output", "cpu", True, 2)
    assert first["new_responses"] == 96
    assert second == {
        "run_dir": first["run_dir"],
        "completed": 96,
        "new_responses": 0,
        "remaining": 0,
        "status": "complete",
    }
    assert calls == 96
    assert config_path.read_bytes() == config_before
    assert {
        path.name: calibration.core.sha256_file(path)
        for path in (run_dir / "responses").glob("*.json")
    } == response_hashes_before

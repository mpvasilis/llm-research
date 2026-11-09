from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from experiments import extended_calibration_study as calibration  # noqa: E402
from test_extended_revision_study import make_frozen_successor, response_for  # noqa: E402


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, Path, dict]:
    root = tmp_path_factory.mktemp("extended-calibration")
    protocol_path, protocol = make_frozen_successor(root)
    manifest_path = root / "calibration_manifest.json"
    manifest = calibration.freeze_manifest(protocol_path, protocol["primary_cap"], manifest_path)
    return protocol_path, protocol, manifest_path, manifest


def calibration_response(entry: dict, manifest: dict, execution: dict) -> dict:
    checkpoint = manifest["checkpoints"][entry["checkpoint_alias"]]
    row = response_for(entry, manifest["calibration_manifest_id"], checkpoint, execution)
    row.update({
        "calibration_manifest_id": manifest["calibration_manifest_id"],
        "calibration_runner_version": calibration.CALIBRATION_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "calibration_classification": manifest["classification"],
        "checkpoint_alias": entry["checkpoint_alias"],
    })
    return row


def test_manifest_is_exact_96_response_rotation_and_strict_cap(frozen: tuple[Path, dict, Path, dict], tmp_path: Path) -> None:
    protocol_path, protocol, manifest_path, manifest = frozen
    assert calibration.plan(protocol_path, protocol["primary_cap"])["responses"] == 96
    assert len(manifest["grid"]) == 96
    assert {row["split"] for row in manifest["grid"]} == {"development"}
    assert {row["logical_seed"] for row in manifest["grid"]} == {0}
    assert all(sum(row["checkpoint_alias"] == alias for row in manifest["grid"]) == 16 for alias in calibration.CHECKPOINT_ALIASES)
    with pytest.raises(ValueError, match="equal the explicit higher cap"):
        calibration.build_manifest(protocol_path, protocol["primary_cap"] + 1)
    changed = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed["grid"][0]["scenario_family_id"] = "tampered"
    body = {key: value for key, value in changed.items() if key != "calibration_manifest_id"}
    changed["calibration_manifest_id"] = calibration.core.stable_id("extended_calibration_manifest", body, 32)
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError):
        calibration.load_manifest(changed_path, protocol_path)


def test_fake_complete_generation_resume_zero_and_strict_export(
    frozen: tuple[Path, dict, Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path, _protocol, manifest_path, manifest = frozen
    execution_core = {
        "profile_schema_version": "revision-study-execution-profile-v1", "device_class": "cpu",
        "dtype": "float32", "torch_num_threads": 2, "single_response_generation": True,
        "python_version": "test", "torch_version": "test", "transformers_version": "test",
        "cuda_runtime": None, "accelerator": None,
    }
    execution = {**execution_core, "execution_profile_id": calibration.core.stable_id("execution", execution_core, 32)}
    calls: list[str] = []

    class Model:
        pass

    class Tokenizer:
        pass

    monkeypatch.setattr(calibration.core, "execution_profile", lambda *_args: execution)
    monkeypatch.setattr(calibration.core, "load_checkpoint", lambda *_args: (Model(), Tokenizer(), "torch.float32"))
    monkeypatch.setattr(calibration.core, "context_preflight", lambda parent, entries, model, tokenizer: {"context_limit": 4096, "primary_cap": parent["primary_cap"], "sensitivity_cap": parent["sensitivity_cap"], "tested_prompts": len(entries)})

    def fake_generate(entry, _checkpoint, _model, _tokenizer, observed_execution, _dtype, observed_manifest):
        calls.append(entry["response_id"])
        return calibration_response(entry, observed_manifest, observed_execution)

    monkeypatch.setattr(calibration, "_response_with_calibration_provenance", fake_generate)
    first = calibration.run_generation(manifest_path, protocol_path, tmp_path / "output", "cpu", True, 2)
    assert first["new_responses"] == 96 and first["remaining"] == 0
    run_dir = Path(first["run_dir"])
    before = {path.name: calibration.core.sha256_file(path) for path in (run_dir / "responses").glob("*.json")}
    second = calibration.run_generation(manifest_path, protocol_path, tmp_path / "output", "cpu", True, 2)
    after = {path.name: calibration.core.sha256_file(path) for path in (run_dir / "responses").glob("*.json")}
    assert second["new_responses"] == 0 and second["status"] == "complete"
    assert before == after and len(calls) == 96
    assert calibration.verify_run(manifest_path, protocol_path, run_dir)["verified_responses"] == 96
    exported = calibration.export_run(manifest_path, protocol_path, run_dir, tmp_path / "calibration.jsonl")
    assert exported["responses"] == 96
    missing = next((run_dir / "responses").glob("*.json"))
    saved = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValueError, match="missing 1"):
        calibration.verify_run(manifest_path, protocol_path, run_dir)
    missing.write_bytes(saved)
    row = json.loads(missing.read_text(encoding="utf-8"))
    row["generation"]["max_new_tokens"] += 1
    missing.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="generation cap"):
        calibration.verify_run(manifest_path, protocol_path, run_dir)
    missing.write_bytes(saved)

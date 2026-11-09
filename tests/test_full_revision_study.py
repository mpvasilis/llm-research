from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import full_revision_study as full  # noqa: E402


def fake_tokenizer_identity(model_id: str, revision: str, _local: bool) -> dict:
    size = "1b" if "1B" in model_id else "7b"
    return {
        "tokenizer_revision": revision,
        "chat_template_sha256": ("1" if size == "1b" else "7") * 64,
        "rendered_probe_sha256": ("2" if size == "1b" else "8") * 64,
        "serialization_id": f"serialization_{size}",
    }


@pytest.fixture()
def frozen_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    monkeypatch.setattr(full, "resolve_revision", lambda model_id, _local: full.sha256_bytes(model_id.encode())[:40])
    monkeypatch.setattr(full, "tokenizer_identity", fake_tokenizer_identity)
    path = tmp_path / "protocol.json"
    protocol = full.freeze_protocol(path)
    return path, protocol


def cpu_profile() -> dict:
    core = {
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
    return {**core, "execution_profile_id": full.stable_id("execution", core, 32)}


def complete_review(protocol: dict, path: Path) -> Path:
    pending_path = path.with_name("pending.json")
    review = full.create_human_review_manifest(protocol, pending_path)
    review.update({
        "status": "COMPLETE_HUMAN_REVIEW",
        "reviewer_id": "external-reviewer-test",
        "reviewed_at": "2026-09-15T00:00:00+00:00",
        "review_method": "Manual paired prompt review",
    })
    for family in review["families"]:
        family["checks"] = {key: True for key in family["checks"]}
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def response_for(entry: dict, protocol: dict, execution: dict) -> dict:
    identity = entry["response_identity"]
    checkpoint = protocol["checkpoints"][entry["checkpoint_alias"]]
    return {
        "response_schema_version": full.RESPONSE_SCHEMA_VERSION,
        "study_version": full.shared.STUDY_VERSION,
        "runner_version": full.RUNNER_VERSION,
        "protocol_id": protocol["protocol_id"],
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
            "response": "Mocked integration response.",
            "input_tokens": 10,
            "output_tokens": 4,
            "max_new_tokens": identity["generation_config"]["max_new_tokens"],
            "stopping": {"reason": "eos_token", "eos_reached": True, "cap_reached": False},
            "config": {
                **identity["generation_config"], "effective_eos_token_ids": [2],
                "device": "cpu", "dtype": "torch.float32",
                "execution_profile_id": execution["execution_profile_id"],
            },
            "timing": {"generation_seconds": 0.01, "generated_tokens_per_second": 400.0},
        },
    }


def test_protocol_freezes_exact_grids_and_matched_rng(frozen_protocol: tuple[Path, dict]) -> None:
    _path, protocol = frozen_protocol
    assert protocol["primary_expected_responses"] == 2880
    assert len(protocol["development"]["primary_256_grid"]) == 40
    assert len(protocol["development"]["longer_512_grid"]) == 40
    assert protocol["longer_budget_sensitivity"]["expected_responses"] == 192
    assert Counter(row["logical_seed"] for row in protocol["primary_grid"]) == {0: 960, 1: 960, 2: 960}
    same_prompt_seed = [
        row for row in protocol["primary_grid"]
        if row["prompt_id"] == protocol["primary_grid"][0]["prompt_id"] and row["logical_seed"] == 0
    ]
    assert len(same_prompt_seed) == 6
    assert len({row["response_identity"]["rng_seed"] for row in same_prompt_seed}) == 1
    assert len({row["response_id"] for row in protocol["primary_grid"]}) == 2880


def test_human_review_is_blank_and_evaluation_gate_rejects_it(
    frozen_protocol: tuple[Path, dict], tmp_path: Path,
) -> None:
    protocol_path, protocol = frozen_protocol
    pending = tmp_path / "review.json"
    review = full.create_human_review_manifest(protocol, pending)
    assert review["status"] == "PENDING_HUMAN_REVIEW"
    assert len(review["families"]) == 40
    assert all(value is None for row in review["families"] for value in row["checks"].values())
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        full.prepare_run(protocol_path, "evaluation", tmp_path, None, pending, cpu_profile())


def test_development_subsets_only_complete_four_variant_families(
    frozen_protocol: tuple[Path, dict], tmp_path: Path,
) -> None:
    protocol_path, protocol = frozen_protocol
    family_ids = sorted({row["scenario_family_id"] for row in protocol["development"]["primary_256_grid"]})[:2]
    _protocol, grid, run_dir = full.prepare_run(
        protocol_path, "development-256", tmp_path, family_ids, None, cpu_profile(),
    )
    assert len(grid) == 8
    assert Counter(row["scenario_family_id"] for row in grid) == {family_ids[0]: 4, family_ids[1]: 4}
    report = full.verify_run(protocol_path, run_dir, require_complete=False)
    assert report["status"] == "partial_valid"
    assert report["expected_responses"] == 8


def test_mocked_complete_2880_grid_and_tamper_rejection(
    frozen_protocol: tuple[Path, dict], tmp_path: Path,
) -> None:
    protocol_path, protocol = frozen_protocol
    review_path = complete_review(protocol, tmp_path / "review_complete.json")
    execution = cpu_profile()
    _protocol, grid, run_dir = full.prepare_run(
        protocol_path, "evaluation", tmp_path, None, review_path, execution,
    )
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True)
    for entry in grid:
        row = response_for(entry, protocol, execution)
        (response_dir / f"{entry['response_id']}.json").write_text(
            json.dumps(row, ensure_ascii=False), encoding="utf-8",
        )
    report = full.verify_run(protocol_path, run_dir)
    assert report["status"] == "complete"
    assert report["verified_responses"] == 2880
    assert report["unique_response_ids"] == 2880

    missing_path = response_dir / f"{grid[-1]['response_id']}.json"
    missing_payload = missing_path.read_bytes()
    missing_path.unlink()
    with pytest.raises(ValueError, match="missing 1 expected responses"):
        full.verify_run(protocol_path, run_dir)
    missing_path.write_bytes(missing_payload)

    tampered = tmp_path / "tampered"
    shutil.copytree(run_dir, tampered)
    config_path = tampered / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["response_ids"] = config["response_ids"][:1]
    config["expected_responses"] = 1
    config["scenario_family_ids"] = [grid[0]["scenario_family_id"]]
    core = {key: value for key, value in config.items() if key != "run_id"}
    config["run_id"] = full.stable_id("run", core, 32)
    renamed = tmp_path / config["run_id"]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tampered.rename(renamed)
    with pytest.raises(ValueError, match="complete frozen grids"):
        full.verify_run(protocol_path, renamed, require_complete=False)


def test_mixed_execution_profile_is_rejected(
    frozen_protocol: tuple[Path, dict], tmp_path: Path,
) -> None:
    protocol_path, protocol = frozen_protocol
    family_id = protocol["development"]["primary_256_grid"][0]["scenario_family_id"]
    execution = cpu_profile()
    _protocol, grid, run_dir = full.prepare_run(
        protocol_path, "development-256", tmp_path, [family_id], None, execution,
    )
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True)
    for entry in grid:
        row = response_for(entry, protocol, execution)
        if entry is grid[-1]:
            row["execution_profile"] = {**execution, "torch_num_threads": 8}
        (response_dir / f"{entry['response_id']}.json").write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="execution profile"):
        full.verify_run(protocol_path, run_dir)


def test_generation_resume_skips_each_valid_checkpointed_response(
    frozen_protocol: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path, protocol = frozen_protocol
    family_id = protocol["development"]["primary_256_grid"][0]["scenario_family_id"]
    load_calls = []

    def fake_load(checkpoint: dict, execution: dict, _local: bool):
        load_calls.append((checkpoint["checkpoint_alias"], execution["execution_profile_id"]))
        return object(), object(), "torch.float32"

    def fake_generate(entry: dict, checkpoint: dict, _model, _tokenizer, _execution, _dtype, _protocol_id):
        return response_for(entry, protocol, _execution)

    monkeypatch.setattr(full, "load_checkpoint", fake_load)
    monkeypatch.setattr(full, "entry_to_response", fake_generate)
    first = full.run_generation(
        protocol_path, "development-256", tmp_path, [family_id], None,
        "cpu", True, 4,
    )
    assert first["new_responses"] == 4
    assert first["remaining"] == 0
    assert len(load_calls) == 1
    second = full.run_generation(
        protocol_path, "development-256", tmp_path, [family_id], None,
        "cpu", True, 4,
    )
    assert second["new_responses"] == 0
    assert second["completed"] == 4
    assert len(load_calls) == 1


def test_window_metadata_uses_pinned_detector_tokenizer_counts() -> None:
    class Tokenizer:
        def encode(self, _text: str, add_special_tokens: bool):
            assert add_special_tokens is False
            return list(range(100))

        def decode(self, ids: list[int], skip_special_tokens: bool):
            assert skip_special_tokens is True
            return " ".join(map(str, ids))

    dependency = {"model_id": "pinned/window-tokenizer", "resolved_revision": "a" * 40}
    windows = full.evaluation_windows("source text", Tokenizer(), dependency)
    assert [unit for unit, _text, _metadata in windows] == list(full.shared.EVALUATION_UNITS)
    by_unit = {unit: metadata for unit, _text, metadata in windows}
    assert by_unit["full_response"] == {
        "tokenizer_model_id": "pinned/window-tokenizer",
        "tokenizer_revision": "a" * 40,
        "full_response_tokens": 100,
        "window_token_limit": None,
        "evaluated_tokens": 100,
        "shorter_than_window": None,
        "truncated": False,
    }
    assert by_unit["prefix_64_tokens"]["evaluated_tokens"] == 64
    assert by_unit["prefix_64_tokens"]["shorter_than_window"] is False
    assert by_unit["prefix_64_tokens"]["truncated"] is True
    assert by_unit["prefix_128_tokens"]["evaluated_tokens"] == 100
    assert by_unit["prefix_128_tokens"]["shorter_than_window"] is True
    assert by_unit["prefix_128_tokens"]["truncated"] is False

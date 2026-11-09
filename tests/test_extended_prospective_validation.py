from __future__ import annotations

import copy
import inspect
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import extended_calibration_study as calibration  # noqa: E402
from experiments import extended_revision_study as full  # noqa: E402
from experiments import extended_prospective_validation as prospective  # noqa: E402
from experiments import development_budget_probe as probe  # noqa: E402
from experiments import revision_analysis  # noqa: E402
from experiments import validation_v4  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fake_tokenizer_identity(model_id: str, revision: str, _local: bool) -> dict:
    size = "1b" if "1B" in model_id else "7b"
    return {
        "tokenizer_revision": revision,
        "chat_template_sha256": ("1" if size == "1b" else "7") * 64,
        "rendered_probe_sha256": ("2" if size == "1b" else "8") * 64,
        "serialization_id": f"serialization_{size}",
    }


def _execution() -> dict:
    value = {
        "profile_schema_version": "revision-study-execution-profile-v1",
        "device_class": "cpu",
        "dtype": "float32",
        "torch_num_threads": 2,
        "single_response_generation": True,
        "python_version": "synthetic-test",
        "torch_version": "synthetic-test",
        "transformers_version": "synthetic-test",
        "cuda_runtime": None,
        "accelerator": None,
    }
    return {**value, "execution_profile_id": full.stable_id("execution", value, 32)}


def _calibration_response(entry: dict, manifest: dict, length: int, execution: dict) -> dict:
    checkpoint = manifest["checkpoints"][entry["checkpoint_alias"]]
    identity = entry["response_identity"]
    generation = {
        "prompt": entry["prompt"],
        "serialized_prompt": "<synthetic>",
        "response": f"Clearly synthetic calibration response {entry['response_id']}",
        "input_tokens": 4,
        "output_tokens": length,
        "max_new_tokens": identity["generation_config"]["max_new_tokens"],
        "stopping": {
            "reason": "max_new_tokens" if length == identity["generation_config"]["max_new_tokens"] else "eos_token",
            "eos_reached": length != identity["generation_config"]["max_new_tokens"],
            "cap_reached": length == identity["generation_config"]["max_new_tokens"],
        },
        "config": {
            **identity["generation_config"],
            "effective_eos_token_ids": [2],
            "device": "cpu",
            "dtype": "torch.float32",
            "execution_profile_id": execution["execution_profile_id"],
        },
        "timing": {"generation_seconds": 1.0, "generated_tokens_per_second": float(length)},
    }
    row = {
        "response_schema_version": full.RESPONSE_SCHEMA_VERSION,
        "study_version": full.shared.STUDY_VERSION,
        "runner_version": full.RUNNER_VERSION,
        "protocol_id": manifest["calibration_manifest_id"],
        "execution_profile_id": execution["execution_profile_id"],
        "execution_profile": execution,
        "response_id": entry["response_id"],
        "response_identity": identity,
        "prompt_id": entry["prompt_id"],
        "prompt_corpus_sha256": identity["prompt_corpus_sha256"],
        "prompt_text_sha256": identity["prompt_text_sha256"],
        "scenario_family_id": entry["scenario_family_id"],
        "split": "development",
        "condition": entry["condition"],
        "model_id": checkpoint["model_id"],
        "model_revision": checkpoint["model_revision"],
        "tokenizer_revision": checkpoint["tokenizer_revision"],
        "model_hash": checkpoint["model_hash"],
        "serialization_id": checkpoint["serialization_id"],
        "seed": entry["logical_seed"],
        "rng_seed": identity["rng_seed"],
        "rng_scheme": identity["rng_scheme"],
        "generation": generation,
    }
    row.update({
        "calibration_manifest_id": manifest["calibration_manifest_id"],
        "calibration_runner_version": calibration.CALIBRATION_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "calibration_classification": manifest["classification"],
        "checkpoint_alias": entry["checkpoint_alias"],
    })
    return row


def _probe_response(entry: dict, manifest: dict, execution: dict) -> dict:
    checkpoint = manifest["checkpoint"]
    identity = entry["response_identity"]
    row = {
        "response_schema_version": full.RESPONSE_SCHEMA_VERSION,
        "study_version": full.shared.STUDY_VERSION,
        "runner_version": full.RUNNER_VERSION,
        "protocol_id": manifest["probe_manifest_id"],
        "execution_profile_id": execution["execution_profile_id"],
        "execution_profile": execution,
        "response_id": entry["response_id"],
        "response_identity": identity,
        "prompt_id": entry["prompt_id"],
        "prompt_corpus_sha256": identity["prompt_corpus_sha256"],
        "prompt_text_sha256": identity["prompt_text_sha256"],
        "scenario_family_id": entry["scenario_family_id"],
        "split": "development",
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
            "serialized_prompt": "<synthetic-probe>",
            "response": f"Synthetic probe response {entry['response_id']}",
            "input_tokens": 4,
            "output_tokens": 700,
            "max_new_tokens": 1024,
            "stopping": {"reason": "eos_token", "eos_reached": True, "cap_reached": False},
            "config": {
                **identity["generation_config"],
                "effective_eos_token_ids": [2],
                "device": "cpu",
                "dtype": "torch.float32",
                "execution_profile_id": execution["execution_profile_id"],
            },
            "timing": {"generation_seconds": 1.0, "generated_tokens_per_second": 700.0},
        },
        "budget_probe_manifest_id": manifest["probe_manifest_id"],
        "budget_probe_runner_version": probe.PROBE_RUNNER_VERSION,
        "parent_protocol_id": manifest["parent_protocol"]["protocol_id"],
        "parent_protocol_sha256": manifest["parent_protocol"]["sha256"],
        "budget_probe_classification": manifest["classification"],
        "checkpoint_alias": probe.CHECKPOINT_ALIAS,
    }
    probe.validate_response(row, entry, manifest, execution)
    return row


def _freeze_successor_protocol(tmp_path: Path) -> tuple[Path, dict]:
    parent_path = full.DEFAULT_PARENT_PROTOCOL
    probe_manifest_path = tmp_path / "probe_manifest.json"
    probe_manifest = probe.freeze_manifest(parent_path, probe_manifest_path)
    execution = probe_manifest["evidence"]["validated_shared_execution_profile"]
    _manifest, probe_run_dir = probe.prepare_run(
        probe_manifest_path, parent_path, tmp_path / "probe_output", execution,
    )
    response_dir = probe_run_dir / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    for entry in probe_manifest["grid"]:
        row = _probe_response(entry, probe_manifest, execution)
        (response_dir / f"{row['response_id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True), encoding="utf-8",
        )
    probe_export = tmp_path / "probe_export.jsonl"
    probe.export_run(probe_manifest_path, parent_path, probe_run_dir, probe_export)
    parent = full.legacy.load_protocol(parent_path)
    decision_body = {
        "decision_schema_version": full.BUDGET_DECISION_SCHEMA_VERSION,
        "status": "COMPLETE_DEVELOPMENT_BUDGET_DECISION",
        "source_parent_protocol_id": parent["protocol_id"],
        "probe_manifest_id": probe_manifest["probe_manifest_id"],
        "probe_export_sha256": full.sha256_file(probe_export),
        "primary_cap": 1024,
        "sensitivity_cap": 1536,
        "decided_by": "synthetic-test-fixture",
        "decided_at": "2026-09-15T16:00:00Z",
        "rationale": "Synthetic fixture for successor contract tests only.",
    }
    decision = {
        **decision_body,
        "budget_decision_id": full.stable_id("budget_decision", decision_body, 32),
    }
    decision_path = tmp_path / "budget_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    protocol_path = tmp_path / "successor" / "protocol.json"
    protocol = full.freeze_protocol(
        parent_path, 1024, 1536, protocol_path,
        probe_manifest_path=probe_manifest_path,
        probe_run_dir=probe_run_dir,
        probe_export_path=probe_export,
        budget_decision_path=decision_path,
    )
    return protocol_path, protocol


def _build_frozen_inputs(work: Path) -> dict:
    execution = _execution()
    protocol_path, protocol = _freeze_successor_protocol(work)
    manifest_path = work / "cm.json"
    manifest = calibration.freeze_manifest(protocol_path, protocol["primary_cap"], manifest_path)
    _manifest, run_dir = calibration.prepare_run(
        manifest_path, protocol_path, work / "c", execution
    )
    pattern = [10] + [50] * 31 + [150] * 32 + [200] * 32
    responses = [
        _calibration_response(entry, manifest, pattern[index], execution)
        for index, entry in enumerate(manifest["grid"])
    ]
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True)
    for row in responses:
        (response_dir / f"{row['response_id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    responses_path = work / "cr.jsonl"
    calibration.export_run(manifest_path, protocol_path, run_dir, responses_path)
    run_id = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["run_id"]
    guide_path = work / "guide.md"
    guide_path.write_text("Synthetic human annotation guide.\n", encoding="utf-8")
    evidence_path = work / "human.json"
    reviewed = [row["response_id"] for row in responses[:60]]
    evidence = {
        "schema_version": prospective.EVIDENCE_SCHEMA_VERSION,
        "status": "complete_human_guide_calibration_and_timing",
        "certify_human": True,
        "labels_created_by_ai": False,
        "reviewer_ids": ["synthetic-human-a"],
        "reviewed_at": "2026-09-15T12:00:00Z",
        "reviewed_response_ids": reviewed,
        "calibration_manifest_id": manifest["calibration_manifest_id"],
        "calibration_responses_sha256": prospective.sha256_file(responses_path),
        "calibration_run_id": run_id,
        "primary_max_new_tokens": protocol["primary_cap"],
        "protocol_id": protocol["protocol_id"],
        "orchestration_version": protocol["orchestration_version"],
        "annotation_guide_sha256": prospective.sha256_file(guide_path),
        "timing": {
            "response_ids": reviewed[:4],
            "per_item_seconds": {response_id: 75.0 for response_id in reviewed[:4]},
            "denominator_distinct_responses": 4,
        },
    }
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    plan_path = work / "plan.json"
    return {
        "root": work,
        "protocol_path": protocol_path,
        "protocol": protocol,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "responses_path": responses_path,
        "responses": responses,
        "run_dir": run_dir,
        "guide_path": guide_path,
        "evidence_path": evidence_path,
        "plan_path": plan_path,
    }


@pytest.fixture(scope="module")
def frozen_inputs_base(tmp_path_factory: pytest.TempPathFactory) -> dict:
    work = full.fs_path(
        tmp_path_factory.getbasetemp()
        / "extended_prospective_validation_default_depth_nested_package"
    )
    work.mkdir()
    return _build_frozen_inputs(work)


@pytest.fixture()
def frozen_inputs(tmp_path: Path, frozen_inputs_base: dict) -> dict:
    source_root = frozen_inputs_base["root"]
    work = full.fs_path(
        tmp_path / "copied_successor_validation_package_at_default_depth"
    )
    shutil.copytree(source_root, work)
    copied = dict(frozen_inputs_base)
    copied["root"] = work
    for key in (
        "protocol_path", "manifest_path", "responses_path", "run_dir",
        "guide_path", "evidence_path", "plan_path",
    ):
        copied[key] = work / frozen_inputs_base[key].relative_to(source_root)
    return copied


def _freeze(values: dict) -> dict:
    return prospective.freeze_plan(
        values["protocol_path"],
        values["manifest_path"],
        values["responses_path"],
        values["run_dir"],
        values["evidence_path"],
        values["plan_path"],
        values["guide_path"],
    )


def _fake_study(protocol: dict) -> revision_analysis.ValidatedStudy:
    responses = {}
    aliases = {}
    families = {}
    predictions = {}
    by_cell: dict[str, list[str]] = defaultdict(list)
    for entry in protocol["primary_grid"]:
        response_id = entry["response_id"]
        alias = entry["checkpoint_alias"]
        checkpoint = protocol["checkpoints"][alias]
        condition = entry["condition"]
        cell = prospective.canonical_json({
            "checkpoint_alias": alias,
            "domain": condition["domain"],
            "emotional_framing": condition["emotional_framing"],
            "request_type": condition["request_type"],
        })
        by_cell[cell].append(response_id)
        aliases[response_id] = alias
        families[entry["scenario_family_id"]] = condition["domain"]
        responses[response_id] = {
            "response_id": response_id,
            "study_version": revision_analysis.STUDY_VERSION,
            "scenario_family_id": entry["scenario_family_id"],
            "split": "evaluation",
            "condition": condition,
            "model_id": checkpoint["model_id"],
            "model_revision": checkpoint["model_revision"],
            "tokenizer_revision": checkpoint["tokenizer_revision"],
            "model_hash": checkpoint["model_hash"],
            "serialization_id": checkpoint["serialization_id"],
            "seed": entry["logical_seed"],
            "generation": {
                "prompt": entry["prompt"],
                "response": f"Clearly synthetic evaluation response {response_id}",
                "output_tokens": 0,
            },
        }
    for ids in by_cell.values():
        for index, response_id in enumerate(sorted(ids)):
            responses[response_id]["generation"]["output_tokens"] = (
                20 if index < 10 else 100 if index < 30 else 200
            )
    ordered = sorted(responses)
    special = ordered[:31]
    for index, response_id in enumerate(ordered):
        flags = {category: False for category in prospective.CATEGORIES}
        # One response is guaranteed selected for empathy and is also eligible in
        # a 31-item validation pool, so its unselected-category probability must survive.
        if index == 0:
            flags["empathy_opener"] = True
        if response_id in special:
            flags["validation"] = True
        offset = 31
        for category in ("disclaimer", "structure"):
            if offset <= index < offset + 30:
                flags[category] = True
            offset += 30
        predictions[(response_id, "full_response")] = {
            "detector_version": revision_analysis.EXPECTED_DETECTOR_VERSION,
            "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
            "evaluation_unit": "full_response",
            "dependency_identity": {"synthetic": True},
            "run_id": "synthetic-run",
            "predictions": {
                category: {"union": value} for category, value in flags.items()
            },
        }
        identity = {
            "response_id": response_id,
            "detector_version": revision_analysis.EXPECTED_DETECTOR_VERSION,
            "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
            "evaluation_unit": "full_response",
            "dependency_identity": {"synthetic": True},
        }
        predictions[(response_id, "full_response")]["prediction_id"] = revision_analysis.stable_id(
            "prediction", identity
        )
    return revision_analysis.ValidatedStudy(
        protocol=protocol,
        responses=responses,
        predictions=predictions,
        grid_by_response_id={entry["response_id"]: entry for entry in protocol["primary_grid"]},
        alias_by_response_id=aliases,
        family_domains=families,
        dependency_identity={"synthetic": True},
    )


def _complete_prompt_review(protocol: dict, path: Path) -> Path:
    pending = path.with_name("pending_review.json")
    review = full.create_human_review_manifest(protocol, pending)
    review.update({
        "status": "COMPLETE_HUMAN_REVIEW",
        "reviewer_id": "synthetic-human-prompt-reviewer",
        "reviewed_at": "2026-09-15T10:00:00Z",
        "review_method": "Clearly synthetic manual-review fixture",
    })
    for family in review["families"]:
        family["checks"] = {key: True for key in family["checks"]}
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _strict_evaluation_archives(values: dict, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    protocol = values["protocol"]
    execution = _execution()
    review_path = _complete_prompt_review(protocol, tmp_path / "complete_review.json")
    _protocol, grid, run_dir = full.prepare_run(
        values["protocol_path"], "evaluation", tmp_path / "full_output", None,
        review_path, execution,
        readiness={
            "plan_path": values["plan_path"],
            "calibration_manifest_path": values["manifest_path"],
            "calibration_responses_path": values["responses_path"],
            "calibration_run_dir": values["run_dir"],
            "evidence_path": values["evidence_path"],
            "guide_path": values["guide_path"],
        },
    )
    response_dir = run_dir / "responses"
    response_dir.mkdir(parents=True)
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for entry in grid:
        condition = entry["condition"]
        by_cell[(
            entry["checkpoint_alias"], condition["domain"],
            condition["emotional_framing"], condition["request_type"],
        )].append(entry)
    lengths = {}
    for entries in by_cell.values():
        for index, entry in enumerate(sorted(entries, key=lambda row: row["response_id"])):
            lengths[entry["response_id"]] = 20 if index < 10 else 100 if index < 30 else 200
    responses = []
    for entry in grid:
        identity = entry["response_identity"]
        checkpoint = protocol["checkpoints"][entry["checkpoint_alias"]]
        length = lengths[entry["response_id"]]
        row = {
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
            "split": "evaluation",
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
                "response": f"Clearly synthetic evaluation response {entry['response_id']}.",
                "input_tokens": 10,
                "output_tokens": length,
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
                    "generation_seconds": 1.0,
                    "generated_tokens_per_second": float(length),
                    "scope": "clearly synthetic test fixture",
                },
            },
        }
        responses.append(row)
        (response_dir / f"{entry['response_id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    response_export = tmp_path / "strict_evaluation_responses.jsonl"
    full.export_run(values["protocol_path"], run_dir, response_export)
    run_id = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["run_id"]
    dependencies = {
        "embedding_model": {
            "model_id": "synthetic/embedding", "resolved_revision": "e" * 40, "applied": True,
        },
        "prefix_window_tokenizer": {
            "model_id": "synthetic/tokenizer", "resolved_revision": "f" * 40, "applied": True,
        },
    }
    alias_by_id = {entry["response_id"]: entry["checkpoint_alias"] for entry in grid}
    predictions = []
    for index, row in enumerate(sorted(responses, key=lambda item: item["response_id"])):
        alias = alias_by_id[row["response_id"]]
        flags = {
            category: (index + category_index) % 17 == 0
            for category_index, category in enumerate(revision_analysis.CATEGORIES)
        }
        for unit in revision_analysis.WINDOWS:
            limit = None if unit == "full_response" else 64 if unit == "prefix_64_tokens" else 128
            length = row["generation"]["output_tokens"]
            prediction_identity = {
                "response_id": row["response_id"],
                "detector_version": revision_analysis.EXPECTED_DETECTOR_VERSION,
                "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
                "evaluation_unit": unit,
                "dependency_identity": dependencies,
            }
            predictions.append({
                "prediction_id": revision_analysis.stable_id("prediction", prediction_identity),
                "prediction_schema_version": revision_analysis.PREDICTION_SCHEMA_VERSION,
                "study_version": revision_analysis.STUDY_VERSION,
                "protocol_id": protocol["protocol_id"],
                "run_id": run_id,
                "execution_profile_id": execution["execution_profile_id"],
                "response_id": row["response_id"],
                "scenario_family_id": row["scenario_family_id"],
                "split": "evaluation",
                "condition": row["condition"],
                "model_id": row["model_id"],
                "seed": row["seed"],
                "checkpoint_alias": alias,
                "model_size": alias.split("_", 1)[0],
                "stage": "final" if alias.endswith("instruct") else alias.split("_", 1)[1],
                "generation_output_tokens": length,
                "generation_stop_reason": "eos_token",
                "detector_version": revision_analysis.EXPECTED_DETECTOR_VERSION,
                "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
                "evaluation_unit": unit,
                "decision_field": "union",
                "dependency_identity": dependencies,
                "threshold": 0.82,
                "unavailable_categories": {
                    "crisis_referral": "not_applicable: absent from raw_output_robustness four-category baseline",
                },
                "window_metadata": {
                    "tokenizer_model_id": dependencies["prefix_window_tokenizer"]["model_id"],
                    "tokenizer_revision": dependencies["prefix_window_tokenizer"]["resolved_revision"],
                    "full_response_tokens": length,
                    "window_token_limit": limit,
                    "evaluated_tokens": length if limit is None else min(length, limit),
                    "shorter_than_window": None if limit is None else length < limit,
                    "truncated": False if limit is None else length > limit,
                },
                "predictions": {
                    category: {
                        "exact": flag,
                        "embedding": False,
                        "union": flag,
                        "intersection": False,
                        "embedding_max_similarity": 0.0,
                    }
                    for category, flag in flags.items()
                },
            })
    prediction_export = tmp_path / "strict_evaluation_predictions.jsonl"
    _write_jsonl(prediction_export, predictions)
    return run_dir, response_export, prediction_export, review_path


def _complete_annotations(path: Path, positive_first: bool = False) -> None:
    rows = validation_v4.read_csv(path)
    for index, row in enumerate(rows):
        for target in validation_v4.HUMAN_TARGETS:
            row[target] = "0"
        if positive_first and index == 0:
            row["validation"] = "1"
        row["empathy_expression_location"] = "none"
        row["completed_at"] = "2026-09-15T13:00:00Z"
    validation_v4.write_csv(path, rows, validation_v4.ANNOTATION_FIELDS)


def test_preview_freeze_and_validate_require_complete_human_bound_evidence(
    frozen_inputs: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = prospective.preview_plan(
        frozen_inputs["protocol_path"], frozen_inputs["manifest_path"],
        frozen_inputs["responses_path"], frozen_inputs["run_dir"],
        frozen_inputs["guide_path"],
    )
    assert preview["status"] == "candidate_not_frozen_without_human_development_evidence"
    assert preview["orchestration_version"] == full.ORCHESTRATION_VERSION
    assert preview["protocol"]["extended_schema_version"] == full.EXTENDED_SCHEMA_VERSION
    assert preview["protocol"]["primary_max_new_tokens"] == 1024
    assert preview["protocol"]["sensitivity_max_new_tokens"] == 1536
    assert set(preview["source_sha256"]) == {
        "experiments/extended_prospective_validation.py",
        "experiments/extended_calibration_study.py",
        "experiments/extended_revision_study.py",
        "experiments/revision_analysis.py",
        "experiments/validation_v4.py",
    }
    assert preview["length_bins"]["raw_tertile_values"] == [50, 150]
    assert preview["length_bins"]["effective_cutpoints"] == [50, 150]
    plan = _freeze(frozen_inputs)
    checked = prospective.validate_plan(
        frozen_inputs["plan_path"], frozen_inputs["protocol_path"],
        frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
        frozen_inputs["run_dir"], frozen_inputs["evidence_path"],
        frozen_inputs["guide_path"]
    )
    assert checked["plan_id"] == plan["plan_id"]
    assert checked["planned_probability_sample"] == 624
    with monkeypatch.context() as changed_source:
        changed_source.setattr(
            prospective,
            "_source_hashes",
            lambda: {**plan["source_sha256"], "experiments/extended_prospective_validation.py": "0" * 64},
        )
        with pytest.raises(ValueError, match="verified frozen inputs"):
            prospective.validate_plan(
                frozen_inputs["plan_path"], frozen_inputs["protocol_path"],
                frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
                frozen_inputs["run_dir"], frozen_inputs["evidence_path"],
                frozen_inputs["guide_path"],
            )

    altered = copy.deepcopy(plan)
    altered["probability_sample"]["sample_per_outer_cell"] = 12
    frozen_inputs["plan_path"].write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="plan content"):
        prospective.validate_plan(
            frozen_inputs["plan_path"], frozen_inputs["protocol_path"],
            frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
            frozen_inputs["run_dir"], frozen_inputs["evidence_path"],
            frozen_inputs["guide_path"]
        )


def test_freeze_rejects_partial_review_and_changed_guide(frozen_inputs: dict) -> None:
    legacy_manifest = copy.deepcopy(frozen_inputs["manifest"])
    legacy_manifest["manifest_schema_version"] = "revision-study-calibration-manifest-v1"
    legacy_body = {key: value for key, value in legacy_manifest.items() if key != "calibration_manifest_id"}
    legacy_manifest["calibration_manifest_id"] = full.stable_id(
        "extended_calibration_manifest", legacy_body, 32,
    )
    legacy_path = frozen_inputs["root"] / "legacy_masquerade.json"
    legacy_path.write_text(json.dumps(legacy_manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported extended calibration"):
        prospective.preview_plan(
            frozen_inputs["protocol_path"], legacy_path, frozen_inputs["responses_path"],
            frozen_inputs["run_dir"], frozen_inputs["guide_path"],
        )

    original_export = frozen_inputs["responses_path"].read_bytes()
    changed_rows = [json.loads(line) for line in original_export.decode("utf-8").splitlines()]
    changed_rows[0]["generation"]["response"] += " changed"
    _write_jsonl(frozen_inputs["responses_path"], changed_rows)
    with pytest.raises(ValueError, match="does not match the verified run directory"):
        _freeze(frozen_inputs)
    frozen_inputs["responses_path"].write_bytes(original_export)

    evidence = json.loads(frozen_inputs["evidence_path"].read_text(encoding="utf-8"))
    evidence["reviewed_response_ids"] = evidence["reviewed_response_ids"][:59]
    frozen_inputs["evidence_path"].write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="At least 60"):
        _freeze(frozen_inputs)
    evidence["reviewed_response_ids"] = [row["response_id"] for row in frozen_inputs["responses"][:60]]
    frozen_inputs["evidence_path"].write_text(json.dumps(evidence), encoding="utf-8")
    frozen_inputs["guide_path"].write_text("Changed guide.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation guide"):
        _freeze(frozen_inputs)


def test_length_allocation_edge_cases_and_exact_probabilities() -> None:
    assert list(inspect.signature(prospective.validate_plan).parameters) == [
        "plan_path", "protocol_path", "calibration_manifest_path",
        "calibration_responses_path", "calibration_run_dir", "evidence_path",
        "guide_path",
    ]
    assert prospective.allocate_length_bins({0: 10, 1: 20, 2: 30}) == {0: 3, 1: 4, 2: 6}
    assert prospective.allocate_length_bins({0: 0, 1: 1, 2: 59}) == {1: 1, 2: 12}
    assert prospective.allocate_length_bins({0: 60}) == {0: 13}
    with pytest.raises(ValueError, match="cannot cover"):
        prospective.allocate_length_bins({index: 1 for index in range(14)})
    tied_minimum = [
        {"generation": {"output_tokens": 100}} for _ in range(64)
    ] + [
        {"generation": {"output_tokens": 256}} for _ in range(32)
    ]
    assert prospective.derive_length_bins(tied_minimum)["effective_cutpoints"] == [100]


def test_default_depth_extended_path_atomic_and_read_boundaries(tmp_path: Path) -> None:
    deep = tmp_path / "default_depth_successor_validation_artifacts"
    while len(str(deep.resolve())) <= 320:
        deep /= "nested_readiness_and_annotation_package_segment"
    target = deep / "immutable_plan_fragment.json"
    prospective._atomic_json(target, {"status": "synthetic-long-path-check"})
    normalized = full.fs_path(target)
    assert len(full.display_path(normalized)) > 300
    assert json.loads(normalized.read_text(encoding="utf-8"))["status"] == "synthetic-long-path-check"
    assert not full.display_path(normalized).startswith("\\\\?\\")


def test_packet_is_deterministic_balanced_capped_and_records_all_probabilities(
    frozen_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _freeze(frozen_inputs)
    study = _fake_study(frozen_inputs["protocol"])
    monkeypatch.setattr(prospective, "_load_and_validate_successor_study", lambda *_args: study)
    monkeypatch.setattr(
        prospective, "_validate_evaluation_run_export",
        lambda *_args: {"run_id": "synthetic-run", "run_config_sha256": "0" * 64, "readiness_sha256": {}},
    )
    response_export = tmp_path / "evaluation_responses.jsonl"
    prediction_export = tmp_path / "evaluation_predictions.jsonl"
    response_export.write_text("strict loader is mocked with a complete synthetic study\n", encoding="utf-8")
    prediction_export.write_text("strict loader is mocked with complete synthetic predictions\n", encoding="utf-8")
    first_dir = tmp_path / "packet_first"
    second_dir = tmp_path / "packet_second"
    first = prospective.prepare_packet(
        frozen_inputs["plan_path"], frozen_inputs["protocol_path"], response_export,
        prediction_export, frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
        frozen_inputs["run_dir"], frozen_inputs["evidence_path"], tmp_path / "evaluation_run",
        first_dir,
        frozen_inputs["guide_path"],
    )
    prospective.prepare_packet(
        frozen_inputs["plan_path"], frozen_inputs["protocol_path"], response_export,
        prediction_export, frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
        frozen_inputs["run_dir"], frozen_inputs["evidence_path"], tmp_path / "evaluation_run",
        second_dir,
        frozen_inputs["guide_path"],
    )
    assert first["n_probability_sample"] == 624
    assert first["n_enriched_union"] <= 120
    assert 624 <= first["n_items"] <= 744
    assert (first_dir / "validation_items_v4.csv").read_bytes() == (
        second_dir / "validation_items_v4.csv"
    ).read_bytes()
    assert (first_dir / "validation_key_v4.csv").read_bytes() == (
        second_dir / "validation_key_v4.csv"
    ).read_bytes()

    keys = validation_v4.read_csv(first_dir / "validation_key_v4.csv")
    random_rows = [row for row in keys if "probability_sample" in row["sample_role"].split(";")]
    assert len(random_rows) == 624
    assert set(round(float(row["random_inclusion_probability"]), 12) for row in random_rows) == {
        0.2, 0.3
    }
    design = [json.loads(line) for line in (first_dir / "prospective_sampling_design_extended_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    multi = next(
        row for row in design
        if row["eligible_enrichment_category_probabilities"].keys()
        >= {"empathy_opener", "validation"}
    )
    assert multi["eligible_enrichment_category_probabilities"]["empathy_opener"] == {
        "numerator": 1, "denominator": 1, "decimal": 1.0,
    }
    assert multi["eligible_enrichment_category_probabilities"]["validation"]["numerator"] == 30
    assert multi["eligible_enrichment_category_probabilities"]["validation"]["denominator"] == 31
    assert multi["enrichment_union_inclusion_probability"] == {
        "numerator": 1, "denominator": 1, "decimal": 1.0,
    }
    enriched_only = next(
        row for row in design
        if not row["selected_in_probability_sample"]
        and row["selected_enrichment_categories"]
        and row["enrichment_union_inclusion_probability"]["decimal"] < 1.0
    )
    random_probability = enriched_only["random_component_inclusion_probability"]
    enrichment_probability = enriched_only["enrichment_union_inclusion_probability"]
    from fractions import Fraction
    expected_panel = 1 - Fraction(
        random_probability["numerator"], random_probability["denominator"]
    )
    expected_panel *= 1 - Fraction(
        enrichment_probability["numerator"], enrichment_probability["denominator"]
    )
    expected_panel = 1 - expected_panel
    assert enriched_only["panel_union_inclusion_probability"]["numerator"] == expected_panel.numerator
    assert enriched_only["panel_union_inclusion_probability"]["denominator"] == expected_panel.denominator
    assert len(first["random_sampling"]["cell_allocation"]) == 48
    assert all(
        sum(cell["selected_by_length_bin"].values()) == 13
        for cell in first["random_sampling"]["cell_allocation"].values()
    )
    assert plan["enrichment"]["categories"] == list(prospective.CATEGORIES)


def test_prepare_delegates_strict_contract_and_rejects_family_leakage(
    frozen_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _freeze(frozen_inputs)
    response_export = tmp_path / "responses.jsonl"
    prediction_export = tmp_path / "predictions.jsonl"
    response_export.write_text("synthetic\n", encoding="utf-8")
    prediction_export.write_text("synthetic\n", encoding="utf-8")
    monkeypatch.setattr(
        prospective,
        "_load_and_validate_successor_study",
        lambda *_args: (_ for _ in ()).throw(ValueError("Prediction archive must contain exactly 8640 rows")),
    )
    with pytest.raises(ValueError, match="8640"):
        prospective.prepare_packet(
            frozen_inputs["plan_path"], frozen_inputs["protocol_path"], response_export,
            prediction_export, frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
            frozen_inputs["run_dir"], frozen_inputs["evidence_path"], tmp_path / "evaluation_run",
            tmp_path / "packet",
            frozen_inputs["guide_path"],
        )

    leaking = _fake_study(frozen_inputs["protocol"])
    leaking.family_domains[next(iter(plan["calibration"]["development_scenario_family_ids"]))] = "health"
    monkeypatch.setattr(prospective, "_load_and_validate_successor_study", lambda *_args: leaking)
    monkeypatch.setattr(
        prospective, "_validate_evaluation_run_export",
        lambda *_args: {"run_id": "synthetic-run", "run_config_sha256": "0" * 64, "readiness_sha256": {}},
    )
    with pytest.raises(ValueError, match="scenario-family overlap"):
        prospective.prepare_packet(
            frozen_inputs["plan_path"], frozen_inputs["protocol_path"], response_export,
            prediction_export, frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
            frozen_inputs["run_dir"], frozen_inputs["evidence_path"], tmp_path / "evaluation_run",
            tmp_path / "packet",
            frozen_inputs["guide_path"],
        )


def test_prediction_id_must_match_frozen_producer_identity(frozen_inputs: dict) -> None:
    study = _fake_study(frozen_inputs["protocol"])
    prospective._validate_prediction_ids(study)
    first = next(iter(study.predictions.values()))
    first["prediction_id"] = "prediction_unique_but_not_content_bound"
    with pytest.raises(ValueError, match="frozen producer identity"):
        prospective._validate_prediction_ids(study)

def test_two_human_passes_and_adjudication_are_bound_to_original_files(
    frozen_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze(frozen_inputs)
    study = _fake_study(frozen_inputs["protocol"])
    monkeypatch.setattr(prospective, "_load_and_validate_successor_study", lambda *_args: study)
    monkeypatch.setattr(
        prospective, "_validate_evaluation_run_export",
        lambda *_args: {"run_id": "synthetic-run", "run_config_sha256": "0" * 64, "readiness_sha256": {}},
    )
    response_export = tmp_path / "responses.jsonl"
    prediction_export = tmp_path / "predictions.jsonl"
    response_export.write_text("synthetic\n", encoding="utf-8")
    prediction_export.write_text("synthetic\n", encoding="utf-8")
    packet = tmp_path / "packet"
    prospective.prepare_packet(
        frozen_inputs["plan_path"], frozen_inputs["protocol_path"], response_export,
        prediction_export, frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
        frozen_inputs["run_dir"], frozen_inputs["evidence_path"], tmp_path / "evaluation_run",
        packet,
        frozen_inputs["guide_path"],
    )
    first = validation_v4.init_annotation(1, "synthetic-human-1", True, packet)
    second = validation_v4.init_annotation(2, "synthetic-human-2", True, packet)
    _complete_annotations(first, positive_first=False)
    _complete_annotations(second, positive_first=True)
    interim = validation_v4.score(packet, bootstrap_replicates=0)
    assert interim["status"] == "awaiting_adjudication_after_two_human_passes"
    with pytest.raises(ValueError, match="adjudicator-id"):
        validation_v4.init_adjudication(packet)
    adjudication = validation_v4.init_adjudication(
        packet, "synthetic-human-adjudicator", True
    )
    rows = validation_v4.read_csv(adjudication)
    assert len(rows) == 1
    for row in rows:
        for target in validation_v4.HUMAN_TARGETS:
            row[target] = "0"
        row["empathy_expression_location"] = "none"
        row["adjudication_reason"] = "Resolved synthetic disagreement."
        row["adjudicated_at"] = "2026-09-15T14:00:00Z"
    validation_v4.write_csv(adjudication, rows, validation_v4.ADJUDICATION_FIELDS)
    final = validation_v4.score(packet, bootstrap_replicates=0)
    assert final["status"] == "complete_adjudicated_two_human_validation"

    first_rows = validation_v4.read_csv(first)
    first_rows[0]["evidence_notes"] = "Changed after adjudication metadata was bound."
    validation_v4.write_csv(first, first_rows, validation_v4.ANNOTATION_FIELDS)
    with pytest.raises(ValueError, match="stale human adjudication provenance"):
        validation_v4.score(packet, bootstrap_replicates=0)


def test_real_strict_full_grid_roundtrip_and_uniform_wrong_run_id_rejection(
    frozen_inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _freeze(frozen_inputs)
    evaluation_run_dir, response_export, prediction_export, review_path = _strict_evaluation_archives(
        frozen_inputs, tmp_path
    )
    readiness = {
        "plan_path": frozen_inputs["plan_path"],
        "calibration_manifest_path": frozen_inputs["manifest_path"],
        "calibration_responses_path": frozen_inputs["responses_path"],
        "calibration_run_dir": frozen_inputs["run_dir"],
        "evidence_path": frozen_inputs["evidence_path"],
        "guide_path": frozen_inputs["guide_path"],
    }
    before_resume = {
        path.name: full.sha256_file(path)
        for path in (evaluation_run_dir / "responses").glob("*.json")
    }
    execution = _execution()
    monkeypatch.setattr(full, "execution_profile", lambda *_args: execution)

    def model_load_forbidden(*_args, **_kwargs):
        raise AssertionError("A complete valid resume must not load a model")

    monkeypatch.setattr(full, "load_checkpoint", model_load_forbidden)
    resumed = full.run_generation(
        frozen_inputs["protocol_path"], "evaluation", evaluation_run_dir.parents[1],
        None, review_path, "cpu", True, execution["torch_num_threads"],
        readiness=readiness,
    )
    after_resume = {
        path.name: full.sha256_file(path)
        for path in (evaluation_run_dir / "responses").glob("*.json")
    }
    assert resumed["new_responses"] == 0 and resumed["completed"] == 2880
    assert before_resume == after_resume
    packet = tmp_path / "strict_packet"
    manifest = prospective.prepare_packet(
        frozen_inputs["plan_path"], frozen_inputs["protocol_path"], response_export,
        prediction_export, frozen_inputs["manifest_path"], frozen_inputs["responses_path"],
        frozen_inputs["run_dir"], frozen_inputs["evidence_path"], evaluation_run_dir,
        packet, frozen_inputs["guide_path"],
    )
    assert manifest["n_pool"] == 2880
    assert manifest["prediction_unit_counts_in_source"] == {
        unit: 2880 for unit in revision_analysis.WINDOWS
    }
    assert manifest["n_probability_sample"] == 624
    assert manifest["n_enriched_union"] <= 120

    relocation = full.fs_path(tmp_path / "relocated_complete_package")
    relocated_protocol_root = relocation / "protocol_bundle"
    shutil.copytree(frozen_inputs["protocol_path"].parent, relocated_protocol_root)
    relocated_protocol = relocated_protocol_root / frozen_inputs["protocol_path"].name
    relocated_manifest = relocation / "calibration_manifest.json"
    relocated_calibration_responses = relocation / "calibration_responses.jsonl"
    relocated_evidence = relocation / "human_evidence.json"
    relocated_guide = relocation / "guide.md"
    for source, destination in (
        (frozen_inputs["manifest_path"], relocated_manifest),
        (frozen_inputs["responses_path"], relocated_calibration_responses),
        (frozen_inputs["evidence_path"], relocated_evidence),
        (frozen_inputs["guide_path"], relocated_guide),
    ):
        shutil.copy2(source, destination)
    relocated_plan = relocation / "sampling_plan.json"
    shutil.copy2(frozen_inputs["plan_path"], relocated_plan)
    relocated_calibration_run = relocation / "calibration_runs" / frozen_inputs["run_dir"].name
    shutil.copytree(frozen_inputs["run_dir"], relocated_calibration_run)
    relocated_evaluation_run = relocation / "evaluation_runs" / evaluation_run_dir.name
    shutil.copytree(evaluation_run_dir, relocated_evaluation_run)
    relocated_responses = relocation / "evaluation_responses.jsonl"
    relocated_predictions = relocation / "evaluation_predictions.jsonl"
    shutil.copy2(response_export, relocated_responses)
    shutil.copy2(prediction_export, relocated_predictions)

    # Remove the construction locations. Validation must use only the portable
    # successor package while preserving the two run-directory identity leaves.
    shutil.rmtree(frozen_inputs["protocol_path"].parents[1])
    shutil.rmtree(evaluation_run_dir.parents[1])
    response_export.unlink()
    prediction_export.unlink()
    review_path.unlink()
    unavailable = tmp_path / "original_inputs_are_unavailable"
    monkeypatch.setattr(full, "DEFAULT_PARENT_PROTOCOL", unavailable / "parent.json")
    monkeypatch.setattr(
        probe,
        "EVIDENCE_PATHS",
        {name: unavailable / f"{name}.missing" for name in probe.EVIDENCE_PATHS},
    )
    relocated_packet = relocation / "packet"
    relocated_manifest_result = prospective.prepare_packet(
        relocated_plan, relocated_protocol, relocated_responses, relocated_predictions,
        relocated_manifest, relocated_calibration_responses, relocated_calibration_run,
        relocated_evidence, relocated_evaluation_run, relocated_packet, relocated_guide,
    )
    assert relocated_manifest_result["n_pool"] == 2880
    assert relocated_calibration_run.name.startswith("extended_calibration_run_")
    assert relocated_evaluation_run.name.startswith("extended_run_")

    rehashed_rows = [json.loads(line) for line in relocated_predictions.read_text(encoding="utf-8").splitlines()]
    forged = rehashed_rows[0]
    forged["response_id"] = "response_forged_outside_successor_grid"
    forged_identity = {
        "response_id": forged["response_id"],
        "detector_version": forged["detector_version"],
        "prediction_variant": forged["prediction_variant"],
        "evaluation_unit": forged["evaluation_unit"],
        "dependency_identity": forged["dependency_identity"],
    }
    forged["prediction_id"] = revision_analysis.stable_id("prediction", forged_identity)
    rehashed_predictions = relocation / "rehashed_prediction_tamper.jsonl"
    _write_jsonl(rehashed_predictions, rehashed_rows)
    with pytest.raises(ValueError, match="no matching response|exactly match"):
        prospective.prepare_packet(
            relocated_plan, relocated_protocol, relocated_responses,
            rehashed_predictions, relocated_manifest, relocated_calibration_responses,
            relocated_calibration_run, relocated_evidence, relocated_evaluation_run,
            tmp_path / "rehashed_prediction_packet", relocated_guide,
        )

    wrong_rows = [json.loads(line) for line in relocated_predictions.read_text(encoding="utf-8").splitlines()]
    for row in wrong_rows:
        row["run_id"] = "run_uniform_but_wrong"
    wrong_predictions = tmp_path / "wrong_run_predictions.jsonl"
    _write_jsonl(wrong_predictions, wrong_rows)
    # The checkpoint loader permits a uniform run_id. The prospective companion
    # must still bind it to the verified concrete evaluation run.
    prospective._load_and_validate_successor_study(
        relocated_protocol, relocated_responses, wrong_predictions
    )
    with pytest.raises(ValueError, match="run_id does not match"):
        prospective.prepare_packet(
            relocated_plan, relocated_protocol, relocated_responses,
            wrong_predictions, relocated_manifest, relocated_calibration_responses,
            relocated_calibration_run, relocated_evidence, relocated_evaluation_run,
            tmp_path / "wrong_packet", relocated_guide,
        )

    tampered_parent = relocation / "tampered_evaluation_runs"
    tampered_run = tampered_parent / relocated_evaluation_run.name
    shutil.copytree(relocated_evaluation_run, tampered_run)
    run_config_path = tampered_run / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_config["readiness_sha256"]["guide_path"] = "0" * 64
    run_body = {key: value for key, value in run_config.items() if key != "run_id"}
    run_config["run_id"] = full.stable_id("extended_run", run_body, 32)
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    rehashed_run = tampered_parent / run_config["run_id"]
    tampered_run.rename(rehashed_run)
    with pytest.raises(ValueError, match="[Rr]eadiness|provenance|config"):
        prospective.prepare_packet(
            relocated_plan, relocated_protocol, relocated_responses,
            relocated_predictions, relocated_manifest, relocated_calibration_responses,
            relocated_calibration_run, relocated_evidence, rehashed_run,
            tmp_path / "rehashed_run_packet", relocated_guide,
        )

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from experiments import extended_revision_study as full  # noqa: E402
from experiments import extended_sensitivity_analysis as sensitivity  # noqa: E402
from experiments import revision_analysis  # noqa: E402
from test_extended_prospective_validation import (  # noqa: E402
    _build_frozen_inputs,
    _complete_prompt_review,
    _execution,
    _freeze,
)


DEPENDENCIES = {
    "embedding_model": {
        "model_id": "synthetic/embedding",
        "resolved_revision": "e" * 40,
        "applied": True,
    },
    "prefix_window_tokenizer": {
        "model_id": "synthetic/tokenizer",
        "resolved_revision": "f" * 40,
        "applied": True,
    },
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    full.fs_path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _response(entry: dict, protocol: dict, execution: dict, mode: str) -> dict:
    identity = entry["response_identity"]
    checkpoint = protocol["checkpoints"][entry["checkpoint_alias"]]
    primary_text = f"Clearly synthetic paired response for {entry['checkpoint_alias']} {entry['prompt_id']}."
    text = primary_text if mode == "evaluation" else primary_text + " Longer synthetic continuation."
    output_tokens = 100 if mode == "evaluation" else 150
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
            "response": text,
            "input_tokens": 10,
            "output_tokens": output_tokens,
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
                "generated_tokens_per_second": float(output_tokens),
                "scope": "clearly synthetic test fixture",
            },
        },
    }


def _flags(mode: str, condition: dict) -> bool:
    if mode == "evaluation":
        return condition["request_type"] == "advice"
    return condition["request_type"] == "advice" and condition["emotional_framing"] is True


def _prediction_rows(rows: list[dict], grid: list[dict], run_id: str, mode: str) -> list[dict]:
    aliases = {entry["response_id"]: entry["checkpoint_alias"] for entry in grid}
    output = []
    for row in sorted(rows, key=lambda item: item["response_id"]):
        alias = aliases[row["response_id"]]
        exact_flag = _flags(mode, row["condition"])
        for unit in revision_analysis.WINDOWS:
            limit = None if unit == "full_response" else 64 if unit == "prefix_64_tokens" else 128
            length = row["generation"]["output_tokens"]
            identity = {
                "response_id": row["response_id"],
                "detector_version": revision_analysis.EXPECTED_DETECTOR_VERSION,
                "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
                "evaluation_unit": unit,
                "dependency_identity": DEPENDENCIES,
            }
            components = {
                category: {
                    "exact": exact_flag,
                    "embedding": False,
                    "union": exact_flag,
                    "intersection": False,
                    "embedding_max_similarity": 0.1,
                }
                for category in revision_analysis.CATEGORIES
            }
            output.append({
                "prediction_id": revision_analysis.stable_id("prediction", identity),
                "prediction_schema_version": revision_analysis.PREDICTION_SCHEMA_VERSION,
                "study_version": revision_analysis.STUDY_VERSION,
                "protocol_id": row["protocol_id"],
                "run_id": run_id,
                "execution_profile_id": row["execution_profile_id"],
                "response_id": row["response_id"],
                "scenario_family_id": row["scenario_family_id"],
                "split": "evaluation",
                "condition": row["condition"],
                "model_id": row["model_id"],
                "checkpoint_alias": alias,
                "model_size": alias.split("_", 1)[0],
                "stage": "final" if alias.endswith("instruct") else alias.split("_", 1)[1],
                "seed": row["seed"],
                "generation_output_tokens": length,
                "generation_stop_reason": "eos_token",
                "detector_version": revision_analysis.EXPECTED_DETECTOR_VERSION,
                "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
                "evaluation_unit": unit,
                "decision_field": "union",
                "dependency_identity": DEPENDENCIES,
                "window_metadata": {
                    "tokenizer_model_id": DEPENDENCIES["prefix_window_tokenizer"]["model_id"],
                    "tokenizer_revision": DEPENDENCIES["prefix_window_tokenizer"]["resolved_revision"],
                    "full_response_tokens": length,
                    "window_token_limit": limit,
                    "evaluated_tokens": length if limit is None else min(length, limit),
                    "shorter_than_window": None if limit is None else length < limit,
                    "truncated": False if limit is None else length > limit,
                },
                "threshold": full.shared.EMBEDDING_THRESHOLD,
                "predictions": components,
                "unavailable_categories": {
                    "crisis_referral": "not_applicable: absent from raw_output_robustness four-category baseline"
                },
            })
    return output


def _readiness(values: dict) -> dict[str, Path]:
    return {
        "plan_path": values["plan_path"],
        "calibration_manifest_path": values["manifest_path"],
        "calibration_responses_path": values["responses_path"],
        "calibration_run_dir": values["run_dir"],
        "evidence_path": values["evidence_path"],
        "guide_path": values["guide_path"],
    }


def _make_run(
    values: dict,
    root: Path,
    mode: str,
    review_path: Path,
    execution: dict,
) -> tuple[Path, Path, Path, list[dict]]:
    protocol, grid, run_dir = full.prepare_run(
        values["protocol_path"], mode, root, None, review_path, execution,
        readiness=_readiness(values),
    )
    rows = [_response(entry, protocol, execution, mode) for entry in grid]
    response_dir = run_dir / "responses"
    full.fs_path(response_dir).mkdir(parents=True)
    for row in rows:
        full.fs_path(response_dir / f"{row['response_id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    response_export = root / f"{mode}_responses.jsonl"
    full.export_run(values["protocol_path"], run_dir, response_export)
    run_id = json.loads(full.fs_path(run_dir / "run_config.json").read_text(encoding="utf-8"))["run_id"]
    predictions = _prediction_rows(rows, grid, run_id, mode)
    prediction_export = root / f"{mode}_predictions.jsonl"
    _write_jsonl(prediction_export, predictions)
    return run_dir, response_export, prediction_export, rows


@pytest.fixture(scope="module")
def integrated(tmp_path_factory: pytest.TempPathFactory) -> dict:
    del tmp_path_factory
    fixture_root = ROOT / "output" / "revision" / "verification" / "extended_sensitivity_test_fixtures"
    full.fs_path(fixture_root).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="esa-", dir=full.fs_path(fixture_root)))
    try:
        values = _build_frozen_inputs(work / "frozen")
        _freeze(values)
        review = _complete_prompt_review(values["protocol"], work / "review.json")
        execution = _execution()
        primary = _make_run(values, work / "primary", "evaluation", review, execution)
        longer = _make_run(values, work / "longer", "sensitivity", review, execution)
        yield {"work": work, "values": values, "review": review, "execution": execution, "primary": primary, "longer": longer}
    finally:
        if os.environ.get("EXTENDED_SENSITIVITY_KEEP_FIXTURE") != "1":
            shutil.rmtree(full.fs_path(work), ignore_errors=True)


def _analysis_args(integrated: dict, output: Path) -> tuple:
    primary_run, primary_responses, primary_predictions, _rows = integrated["primary"]
    longer_run, longer_responses, longer_predictions, _longer_rows = integrated["longer"]
    return (
        integrated["values"]["protocol_path"], primary_run, primary_responses, primary_predictions,
        longer_run, longer_responses, longer_predictions, output,
    )


def test_predict_exports_actual_576_rows_with_boundary_consistency(
    integrated: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return list(range(len(text.split())))

        def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
            assert skip_special_tokens is True
            return " ".join(f"t{item}" for item in ids)

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(full.shared, "prediction_dependencies", lambda **kwargs: copy.deepcopy(DEPENDENCIES))
    monkeypatch.setattr(full.shared, "load_embedding_detector", lambda **kwargs: (object(), object()))

    def fake_embedding(_model: object, _vectors: object, texts: list[str]) -> tuple[list[dict], list[dict]]:
        flags, maxima = [], []
        for index, _text in enumerate(texts):
            score = full.shared.EMBEDDING_THRESHOLD if index == 0 else 0.1
            flags.append({category: score >= full.shared.EMBEDDING_THRESHOLD for category in revision_analysis.CATEGORIES})
            maxima.append({category: score for category in revision_analysis.CATEGORIES})
        return flags, maxima

    monkeypatch.setattr(full.shared, "embedding_flags", fake_embedding)
    output = tmp_path / "sensitivity_predictions.jsonl"
    report = sensitivity.predict_complete_sensitivity(
        integrated["values"]["protocol_path"], integrated["longer"][0], output
    )
    rows = list(revision_analysis.iter_jsonl(output))
    assert report["responses"] == 192
    assert report["prediction_rows"] == len(rows) == 576
    assert len({row["prediction_id"] for row in rows}) == 576
    assert any(
        value["embedding"] and value["embedding_max_similarity"] == row["threshold"]
        for row in rows for value in row["predictions"].values()
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        sensitivity.predict_complete_sensitivity(integrated["values"]["protocol_path"], integrated["longer"][0], output)


def test_integrated_analysis_exact_pairs_known_contrasts_and_determinism(
    integrated: dict, tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    one = sensitivity.analyze(*_analysis_args(integrated, first), bootstrap_replicates=32, seed=17)
    two = sensitivity.analyze(*_analysis_args(integrated, second), bootstrap_replicates=32, seed=17)
    a = json.loads(full.fs_path(first / sensitivity.RESULT_FILENAME).read_text(encoding="utf-8"))
    b = json.loads(full.fs_path(second / sensitivity.RESULT_FILENAME).read_text(encoding="utf-8"))
    assert one["matched_pairs"] == a["counts"]["matched_seed0_pairs"] == 192
    assert a["counts"]["selected_scenario_families"] == 8
    assert a["counts"]["selected_family_domains"] == {"everyday": 4, "health": 4}
    assert a["matching"]["primary_seed_policy"].startswith("seed zero only")
    assert a == b and one["result_id"] == two["result_id"]
    target = next(row for row in a["contrast_summaries"] if (
        row["checkpoint_alias"] == "1b_sft"
        and row["domain"] == "health"
        and row["window"] == "full_response"
        and row["component"] == "union"
        and row["category"] == "empathy_opener"
        and row["factor_contrast"] == "interaction"
    ))
    assert target["primary_seed0_estimate"] == 0.0
    assert target["sensitivity_seed0_estimate"] == 1.0
    assert target["sensitivity_minus_primary_estimate"] == 1.0
    assert all(row["decoded_primary_text_is_sensitivity_prefix_rate"] == 1.0 for row in a["response_summaries"])
    with pytest.raises(FileExistsError, match="overwrite"):
        sensitivity.analyze(*_analysis_args(integrated, first), bootstrap_replicates=0)


def test_zero_bootstrap_and_contract_rejections(integrated: dict, tmp_path: Path) -> None:
    zero_dir = tmp_path / "zero"
    sensitivity.analyze(*_analysis_args(integrated, zero_dir), bootstrap_replicates=0)
    result = json.loads(full.fs_path(zero_dir / sensitivity.RESULT_FILENAME).read_text(encoding="utf-8"))
    assert all(
        row["sensitivity_minus_primary_pointwise_95_lower"] is None
        and row["sensitivity_minus_primary_pointwise_95_upper"] is None
        for row in result["contrast_summaries"]
    )

    bad_grid = copy.deepcopy(integrated["values"]["protocol"]["longer_budget_sensitivity"]["grid"])
    bad_grid[0]["condition"]["request_type"] = "information" if bad_grid[0]["condition"]["request_type"] == "advice" else "advice"
    with pytest.raises(ValueError, match="cross|differ"):
        sensitivity._pair_grids(integrated["values"]["protocol"], bad_grid)

    rows = list(revision_analysis.iter_jsonl(integrated["longer"][2]))
    incomplete = tmp_path / "incomplete.jsonl"
    _write_jsonl(incomplete, rows[:-1])
    with pytest.raises(ValueError, match="576"):
        sensitivity._validate_sensitivity(
            integrated["values"]["protocol_path"], integrated["longer"][0], integrated["longer"][1], incomplete
        )

    wrong = copy.deepcopy(rows)
    wrong[0]["run_id"] = "extended_run_" + "0" * 32
    wrong_run = tmp_path / "wrong_run.jsonl"
    _write_jsonl(wrong_run, wrong)
    with pytest.raises(ValueError, match="run_id"):
        sensitivity._validate_sensitivity(
            integrated["values"]["protocol_path"], integrated["longer"][0], integrated["longer"][1], wrong_run
        )

    wrong = copy.deepcopy(rows)
    wrong[0]["prediction_id"] = "prediction_" + "0" * 32
    wrong_id = tmp_path / "wrong_id.jsonl"
    _write_jsonl(wrong_id, wrong)
    with pytest.raises(ValueError, match="producer identity"):
        sensitivity._validate_sensitivity(
            integrated["values"]["protocol_path"], integrated["longer"][0], integrated["longer"][1], wrong_id
        )


def test_embedding_score_flag_contradiction_and_execution_profile_mismatch(
    integrated: dict, tmp_path: Path,
) -> None:
    rows = list(revision_analysis.iter_jsonl(integrated["longer"][2]))
    contradictory = copy.deepcopy(rows)
    value = contradictory[0]["predictions"]["empathy_opener"]
    value["embedding"] = True
    value["union"] = True
    value["intersection"] = value["exact"]
    value["embedding_max_similarity"] = full.shared.EMBEDDING_THRESHOLD - 0.01
    path = tmp_path / "contradictory.jsonl"
    _write_jsonl(path, contradictory)
    with pytest.raises(ValueError, match="contradicts"):
        sensitivity._validate_sensitivity(
            integrated["values"]["protocol_path"], integrated["longer"][0], integrated["longer"][1], path
        )

    different = {**integrated["execution"], "torch_num_threads": 3}
    body = {key: value for key, value in different.items() if key != "execution_profile_id"}
    different["execution_profile_id"] = full.stable_id("execution", body, 32)
    other = _make_run(integrated["values"], integrated["work"] / "other", "sensitivity", integrated["review"], different)
    primary_run, primary_responses, primary_predictions, _ = integrated["primary"]
    with pytest.raises(ValueError, match="equal generation execution profiles"):
        sensitivity._load_inputs(
            integrated["values"]["protocol_path"], primary_run, primary_responses, primary_predictions,
            other[0], other[1], other[2],
        )

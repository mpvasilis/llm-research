import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import validation_v3, validation_v4


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def response(
    response_id: str,
    family: str,
    split: str = "evaluation",
    condition: str = "advice",
    prompt: str | None = None,
) -> dict:
    return {
        "study_version": "synthetic-v1",
        "response_id": response_id,
        "scenario_family_id": family,
        "split": split,
        "condition": condition,
        "model_id": "example/model-1b",
        "model_revision": "abc123",
        "tokenizer_revision": "def456",
        "model_hash": "hf-tree:abc123",
        "seed": 0,
        "serialization_id": "chat@def456",
        "generation": {
            "prompt": prompt or f"Prompt for {response_id}",
            "response": f"Response for {response_id}",
            "serialized_prompt": f"serialized {response_id}",
            "input_tokens": 4,
            "output_tokens": 4,
            "max_new_tokens": 16,
            "stopping": {"reason": "eos_token", "eos_reached": True, "cap_reached": False},
            "config": {"do_sample": False},
        },
    }


def prediction(response_id: str, positive: bool = False, unit: str = "full_response") -> dict:
    return {
        "study_version": "synthetic-v1",
        "response_id": response_id,
        "detector_version": "historical-test-v1",
        "evaluation_unit": unit,
        "decision_field": "union",
        "predictions": {
            "empathy_opener": {"union": positive},
            "validation": {"union": positive},
            "disclaimer": {"union": positive},
            "structure": {"union": positive},
        },
    }


def versioned_prediction(response_id: str, dependency_revision: str = "e5-commit-a") -> dict:
    row = prediction(response_id)
    row.update(
        {
            "prediction_id": f"prediction-{response_id}",
            "prediction_variant": "historical-four-category-union-v2",
            "prediction_schema_version": "prediction-v2",
            "dependency_identity": {
                "embedding_model": "intfloat/multilingual-e5-small",
                "embedding_revision": dependency_revision,
            },
        }
    )
    return row


def prepare_synthetic(tmp_path: Path, seed: int = 7, enrichment: int = 1) -> Path:
    responses = [response(f"r{i}", f"family-{i // 2}") for i in range(6)]
    predictions = [versioned_prediction(f"r{i}") for i in range(6)]
    for i, row in enumerate(predictions):
        for value in row["predictions"].values():
            value["union"] = i % 2 == 0
    responses_path = tmp_path / "responses.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    out_dir = tmp_path / "packet"
    write_jsonl(responses_path, responses)
    write_jsonl(predictions_path, predictions)
    validation_v4.prepare(
        responses_path,
        predictions_path,
        "evaluation",
        out_dir,
        random_per_stratum=2,
        positive_per_category=enrichment,
        seed=seed,
        construction_prompts_path=None,
    )
    return out_dir


def complete_annotation(path: Path, empathy: str = "0") -> None:
    rows = validation_v4.read_csv(path)
    for row in rows:
        for target in validation_v4.HUMAN_TARGETS:
            row[target] = empathy if target == "empathy_expression" else "0"
        row["empathy_expression_location"] = "opening" if empathy == "1" else "none"
        row["completed_at"] = "2026-09-14T12:00:00Z"
    validation_v4.write_csv(path, rows, validation_v4.ANNOTATION_FIELDS)


def test_v3_missing_and_invalid_prediction_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing detector prediction"):
        validation_v3._prediction_from_behavior(tmp_path / "missing.json", 0.82)
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"lexicon_categories": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks"):
        validation_v3._prediction_from_behavior(malformed, 0.82)


def test_prepare_requires_full_response_prediction(tmp_path: Path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(responses_path, [response("r0", "family-0")])
    write_jsonl(predictions_path, [prediction("r0", unit="prefix_64_tokens")])
    with pytest.raises(ValueError, match="Missing full-response predictions"):
        validation_v4.prepare(
            responses_path,
            predictions_path,
            "evaluation",
            tmp_path / "packet",
            construction_prompts_path=None,
        )


def test_prepare_rejects_mixed_or_partial_prediction_provenance(tmp_path: Path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        responses_path,
        [response("r0", "family-0"), response("r1", "family-1")],
    )
    write_jsonl(
        predictions_path,
        [versioned_prediction("r0", "commit-a"), versioned_prediction("r1", "commit-b")],
    )
    with pytest.raises(ValueError, match="Mixed dependency_identity"):
        validation_v4.prepare(
            responses_path,
            predictions_path,
            "evaluation",
            tmp_path / "mixed",
            construction_prompts_path=None,
        )
    partial = versioned_prediction("r0")
    del partial["dependency_identity"]
    write_jsonl(predictions_path, [partial, versioned_prediction("r1")])
    with pytest.raises(ValueError, match="all present or all absent"):
        validation_v4.prepare(
            responses_path,
            predictions_path,
            "evaluation",
            tmp_path / "partial",
            construction_prompts_path=None,
        )


def test_legacy_prediction_provenance_requires_explicit_test_flag(tmp_path: Path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(responses_path, [response("r0", "family-0")])
    write_jsonl(predictions_path, [prediction("r0")])
    with pytest.raises(ValueError, match="Missing required versioned prediction provenance"):
        validation_v4.prepare(
            responses_path,
            predictions_path,
            "evaluation",
            tmp_path / "strict",
            construction_prompts_path=None,
        )
    manifest = validation_v4.prepare(
        responses_path,
        predictions_path,
        "evaluation",
        tmp_path / "legacy-test",
        random_per_stratum=1,
        construction_prompts_path=None,
        allow_legacy_prediction_provenance=True,
    )
    assert manifest["prediction_provenance"]["mode"].startswith("legacy_test_only")

def test_group_separation_and_known_construction_prompt(tmp_path: Path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(
        responses_path,
        [
            response("dev", "same-family", split="development"),
            response("eval", "same-family", split="evaluation"),
        ],
    )
    write_jsonl(predictions_path, [prediction("dev"), prediction("eval")])
    with pytest.raises(ValueError, match="cross development/evaluation"):
        validation_v4.prepare(
            responses_path,
            predictions_path,
            "evaluation",
            tmp_path / "packet",
            construction_prompts_path=None,
        )

    construction_path = tmp_path / "construction.json"
    construction_path.write_text(json.dumps({"advice": ["Known prompt?"]}), encoding="utf-8")
    write_jsonl(responses_path, [response("eval", "new-family", prompt=" Known   prompt? ")])
    write_jsonl(predictions_path, [prediction("eval")])
    with pytest.raises(ValueError, match="known construction prompt"):
        validation_v4.prepare(
            responses_path,
            predictions_path,
            "evaluation",
            tmp_path / "packet2",
            construction_prompts_path=construction_path,
        )


def test_deterministic_sampling_and_component_separation(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = prepare_synthetic(first_root, seed=11, enrichment=1)
    second = prepare_synthetic(second_root, seed=11, enrichment=1)
    assert (first / "validation_items_v4.csv").read_bytes() == (
        second / "validation_items_v4.csv"
    ).read_bytes()
    assert (first / "validation_key_v4.csv").read_bytes() == (
        second / "validation_key_v4.csv"
    ).read_bytes()
    keys = validation_v4.read_csv(first / "validation_key_v4.csv")
    assert any("probability_sample" in row["sample_role"] for row in keys)
    assert any("positive_enriched:" in row["sample_role"] for row in keys)
    assert all(
        row["random_inclusion_probability"]
        for row in keys
        if "probability_sample" in row["sample_role"]
    )


def test_blank_human_gate_invalid_labels_and_distinct_annotators(tmp_path: Path) -> None:
    out_dir = prepare_synthetic(tmp_path)
    first = validation_v4.init_annotation(1, "human-a", True, out_dir)
    validation_v4.init_annotation(2, "human-b", True, out_dir)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        validation_v4.score(out_dir, bootstrap_replicates=5)
    complete_annotation(first)
    second, _ = validation_v4.annotation_paths(2, out_dir)
    rows = validation_v4.read_csv(second)
    for row in rows:
        for target in validation_v4.HUMAN_TARGETS:
            row[target] = "0"
        row["professional_referral"] = "maybe"
        row["empathy_expression_location"] = "none"
        row["completed_at"] = "2026-09-14T12:00:00Z"
    validation_v4.write_csv(second, rows, validation_v4.ANNOTATION_FIELDS)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        validation_v4.score(out_dir, bootstrap_replicates=5)


def test_hash_and_duplicate_tamper_protection(tmp_path: Path) -> None:
    out_dir = prepare_synthetic(tmp_path)
    items = out_dir / "validation_items_v4.csv"
    items.write_text(items.read_text(encoding="utf-8") + "tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preparation"):
        validation_v4.score(out_dir, bootstrap_replicates=5)

    duplicate_root = tmp_path / "duplicate"
    responses_path = duplicate_root / "responses.jsonl"
    predictions_path = duplicate_root / "predictions.jsonl"
    write_jsonl(responses_path, [response("r0", "family-0"), response("r0", "family-1")])
    write_jsonl(predictions_path, [versioned_prediction("r0")])
    with pytest.raises(ValueError, match="Duplicate response_id"):
        validation_v4.prepare(
            responses_path,
            predictions_path,
            "evaluation",
            duplicate_root / "packet",
            construction_prompts_path=None,
        )


def test_completed_scoring_never_pools_enriched_and_random(tmp_path: Path) -> None:
    out_dir = prepare_synthetic(tmp_path)
    first = validation_v4.init_annotation(1, "human-a", True, out_dir)
    second = validation_v4.init_annotation(2, "human-b", True, out_dir)
    complete_annotation(first)
    complete_annotation(second)
    report = validation_v4.score(out_dir, bootstrap_replicates=10, seed=3)
    assert report["status"] == "complete_no_adjudication_needed"
    final = report["label_sets"]["adjudicated_human"]
    assert set(final) == {"probability_sample", "enriched_diagnostic"}
    assert final["enriched_diagnostic"]["population_interpretation"] == "none"
    assert "pooled" not in final


def test_zero_precision_and_recall_have_zero_f1() -> None:
    rows = [
        {
            "prediction": {"validation": True},
            "human": {"validation": False},
            "probability": "1",
        },
        {
            "prediction": {"validation": False},
            "human": {"validation": True},
            "probability": "1",
        },
    ]
    result = validation_v4._metric_point(rows, "validation", "validation", False)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_cli_prepare_and_score_human_gate(tmp_path: Path) -> None:
    responses_path = tmp_path / "responses.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    out_dir = tmp_path / "packet"
    write_jsonl(responses_path, [response("r0", "family-0")])
    write_jsonl(predictions_path, [versioned_prediction("r0")])
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.validation_v4",
            "prepare",
            "--responses",
            str(responses_path),
            "--predictions",
            str(predictions_path),
            "--split",
            "evaluation",
            "--out-dir",
            str(out_dir),
            "--random-per-stratum",
            "1",
        ],
        cwd=validation_v4.ROOT,
        capture_output=True,
        text=True,
    )
    assert prepared.returncode == 0, prepared.stderr
    assert '"n_items": 1' in prepared.stdout
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.validation_v4",
            "score",
            "--out-dir",
            str(out_dir),
            "--bootstrap-replicates",
            "5",
        ],
        cwd=validation_v4.ROOT,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert "Missing human annotation pass 1" in refused.stderr


def test_v3_compatibility_scorer_separates_enrichment(tmp_path: Path) -> None:
    items = [
        {"item_id": "v3_r_random", "prompt": "p1", "text": "r1", "unit": "response"},
        {"item_id": "v3_r_enriched", "prompt": "p2", "text": "r2", "unit": "response"},
    ]
    keys = [
        {
            "item_id": "v3_r_random",
            "source_key": "random",
            "model": "1B",
            "stage": "instruct",
            "condition": "advice",
            "predicted": "none",
            "sample_role": "random_stratified",
        },
        {
            "item_id": "v3_r_enriched",
            "source_key": "enriched",
            "model": "1B",
            "stage": "instruct",
            "condition": "advice",
            "predicted": "validation",
            "sample_role": "positive_enriched:validation",
        },
    ]
    validation_v3._write_bundle(items, keys, {"unit": "response"}, tmp_path)
    first = validation_v3.init_annotation(1, "human-a", True, out_dir=tmp_path)
    second = validation_v3.init_annotation(2, "human-b", True, out_dir=tmp_path)
    for path in (first, second):
        rows = validation_v3.read_csv(path)
        for row in rows:
            row["label"] = "none"
        validation_v3.write_csv(
            path,
            rows,
            ["item_id", "label", "annotator_id", "annotator_type", "completed_at"],
        )
    report = validation_v3.score(tmp_path)
    assert report["n_probability_sample"] == 1
    assert report["n_enriched_diagnostic"] == 1
    assert report["detector_vs_human_1"]["validation"]["precision"] is None
    assert report["enriched_diagnostic"]["detector_vs_human_1"]["validation"]["precision"] == 0.0

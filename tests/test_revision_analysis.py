from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import revision_analysis as analysis  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _flag(category: str, alias: str, emotion: bool, request: str, family_index: int, seed: int) -> bool:
    stage = "final" if alias.endswith("instruct") else alias.rsplit("_", 1)[1]
    if category == "empathy_opener":
        return stage in {"dpo", "final"} and request == "advice"
    if category == "validation":
        return stage == "dpo" and emotion
    if category == "disclaimer":
        return request == "advice" and ((stage == "dpo" and emotion) or (stage == "final" and not emotion))
    stage_index = {"sft": 0, "dpo": 1, "final": 2}[stage]
    return (family_index + seed + stage_index + int(emotion) + int(request == "advice")) % 4 == 0


def make_complete_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, list[dict], list[dict]]:
    checkpoints = {}
    for size in analysis.SIZES:
        for suffix in ("sft", "dpo", "instruct"):
            alias = f"{size}_{suffix}"
            checkpoints[alias] = {
                "checkpoint_alias": alias,
                "model_id": f"test/{alias}",
                "model_revision": f"revision-{alias}",
                "tokenizer_revision": f"tokenizer-{size}",
                "model_hash": f"model-hash-{alias}",
                "serialization_id": f"serialization-{size}",
            }
    families = [(f"eval_health_{index:02d}", "health", index) for index in range(20)]
    families += [(f"eval_everyday_{index:02d}", "everyday", index + 20) for index in range(20)]
    grid = []
    responses = []
    predictions = []
    execution_core = {
        "profile_schema_version": "revision-study-execution-profile-v1",
        "device_class": "cpu",
        "dtype": "float32",
        "torch_num_threads": 1,
        "single_response_generation": True,
        "python_version": "synthetic-test",
        "torch_version": "synthetic-test",
        "transformers_version": "synthetic-test",
        "cuda_runtime": None,
        "accelerator": None,
    }
    execution = {**execution_core, "execution_profile_id": analysis.stable_id("execution", execution_core)}
    dependency = {
        "embedding_model": {"model_id": "test/embedding", "resolved_revision": "e" * 40, "applied": True},
        "prefix_window_tokenizer": {"model_id": "test/tokenizer", "resolved_revision": "f" * 40, "applied": True},
        "historical_run_revision_status": "unrecorded; current reproduction dependencies pinned here",
    }
    for family, domain, family_index in families:
        for emotion in analysis.EMOTIONS:
            for request in analysis.REQUESTS:
                condition = {
                    "domain": domain,
                    "emotional_framing": emotion,
                    "request_type": request,
                }
                prompt_id = f"prompt_{family}_{int(emotion)}_{request}"
                prompt = f"Clearly synthetic prompt for {family}, emotion={emotion}, request={request}."
                for alias in sorted(checkpoints):
                    for seed in analysis.SEEDS:
                        checkpoint = checkpoints[alias]
                        generation_config = {
                            "do_sample": True,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_new_tokens": 256,
                        }
                        rng_seed = family_index * 1000 + int(emotion) * 100 + (1 if request == "advice" else 0) * 10 + seed
                        identity = {
                            "study_version": analysis.STUDY_VERSION,
                            "runner_version": analysis.FULL_RUNNER_VERSION,
                            "prompt_corpus_sha256": "a" * 64,
                            "prompt_id": prompt_id,
                            "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "scenario_family_id": family,
                            "split": "evaluation",
                            "condition": condition,
                            "model_id": checkpoint["model_id"],
                            "model_revision": checkpoint["model_revision"],
                            "tokenizer_revision": checkpoint["tokenizer_revision"],
                            "logical_seed": seed,
                            "rng_seed": rng_seed,
                            "rng_scheme": "clearly synthetic deterministic test fixture",
                            "serialization_id": checkpoint["serialization_id"],
                            "generation_config": generation_config,
                        }
                        response_id = analysis.stable_id("response", identity)
                        entry = {
                            "response_id": response_id,
                            "checkpoint_alias": alias,
                            "prompt_id": prompt_id,
                            "scenario_family_id": family,
                            "split": "evaluation",
                            "condition": condition,
                            "logical_seed": seed,
                            "response_identity": identity,
                            "prompt": prompt,
                        }
                        grid.append(entry)
                        responses.append({
                            "response_schema_version": analysis.RESPONSE_SCHEMA_VERSION,
                            "study_version": analysis.STUDY_VERSION,
                            "runner_version": analysis.FULL_RUNNER_VERSION,
                            "protocol_id": "PENDING",
                            "execution_profile_id": execution["execution_profile_id"],
                            "execution_profile": execution,
                            "response_id": response_id,
                            "response_identity": identity,
                            "prompt_id": prompt_id,
                            "prompt_corpus_sha256": identity["prompt_corpus_sha256"],
                            "prompt_text_sha256": identity["prompt_text_sha256"],
                            "scenario_family_id": family,
                            "split": "evaluation",
                            "condition": condition,
                            "model_id": checkpoint["model_id"],
                            "model_revision": checkpoint["model_revision"],
                            "tokenizer_revision": checkpoint["tokenizer_revision"],
                            "model_hash": checkpoint["model_hash"],
                            "serialization_id": checkpoint["serialization_id"],
                            "seed": seed,
                            "rng_seed": rng_seed,
                            "rng_scheme": identity["rng_scheme"],
                            "generation": {
                                "prompt": prompt,
                                "serialized_prompt": f"<user>{prompt}</user><assistant>",
                                "response": "Clearly synthetic test response.",
                                "input_tokens": 12,
                                "output_tokens": 40 + family_index + seed,
                                "max_new_tokens": 256,
                                "stopping": {
                                    "reason": "length" if family_index % 7 == 0 else "eos_token",
                                    "eos_reached": family_index % 7 != 0,
                                    "cap_reached": family_index % 7 == 0,
                                },
                                "config": {
                                    **generation_config,
                                    "effective_eos_token_ids": [2],
                                    "device": "cpu",
                                    "dtype": "torch.float32",
                                    "execution_profile_id": execution["execution_profile_id"],
                                },
                                "timing": {
                                    "generation_seconds": 0.01,
                                    "generated_tokens_per_second": 100.0,
                                    "scope": "synthetic test fixture",
                                },
                            },
                        })
                        for window in analysis.WINDOWS:
                            category_values = {}
                            for category in analysis.CATEGORIES:
                                exact = _flag(category, alias, emotion, request, family_index, seed)
                                embedding = exact if category != "structure" else (
                                    exact or (family_index % 5 == 0 and emotion)
                                )
                                category_values[category] = {
                                    "exact": exact,
                                    "embedding": embedding,
                                    "union": exact or embedding,
                                }
                            predictions.append({
                                "prediction_id": f"prediction_{response_id}_{window}",
                                "prediction_schema_version": analysis.PREDICTION_SCHEMA_VERSION,
                                "study_version": analysis.STUDY_VERSION,
                                "protocol_id": "PENDING",
                                "run_id": "run_clearly_synthetic_test",
                                "execution_profile_id": execution["execution_profile_id"],
                                "response_id": response_id,
                                "scenario_family_id": family,
                                "split": "evaluation",
                                "condition": condition,
                                "model_id": checkpoint["model_id"],
                                "seed": seed,
                                "checkpoint_alias": alias,
                                "model_size": alias.split("_", 1)[0],
                                "stage": "final" if alias.endswith("instruct") else alias.rsplit("_", 1)[1],
                                "generation_output_tokens": 40 + family_index + seed,
                                "generation_stop_reason": "length" if family_index % 7 == 0 else "eos_token",
                                "detector_version": analysis.EXPECTED_DETECTOR_VERSION,
                                "prediction_variant": analysis.EXPECTED_PREDICTION_VARIANT,
                                "evaluation_unit": window,
                                "decision_field": "union",
                                "dependency_identity": dependency,
                                "window_metadata": {
                                    "tokenizer_model_id": dependency["prefix_window_tokenizer"]["model_id"],
                                    "tokenizer_revision": dependency["prefix_window_tokenizer"]["resolved_revision"],
                                    "full_response_tokens": 40 + family_index + seed,
                                    "window_token_limit": None if window == "full_response" else (64 if window == "prefix_64_tokens" else 128),
                                    "evaluated_tokens": min(40 + family_index + seed, 64 if window == "prefix_64_tokens" else 128) if window != "full_response" else 40 + family_index + seed,
                                    "shorter_than_window": None if window == "full_response" else (40 + family_index + seed < (64 if window == "prefix_64_tokens" else 128)),
                                    "truncated": False if window == "full_response" else (40 + family_index + seed > (64 if window == "prefix_64_tokens" else 128)),
                                },
                                "predictions": category_values,
                            })
    core = {
        "protocol_schema_version": analysis.PROTOCOL_SCHEMA_VERSION,
        "study_version": analysis.STUDY_VERSION,
        "runner_version": analysis.FULL_RUNNER_VERSION,
        "status": "FROZEN_SYNTHETIC_TEST_PROTOCOL",
        "checkpoints": checkpoints,
        "primary_grid": grid,
        "primary_expected_responses": 2880,
    }
    protocol = {**core, "protocol_id": analysis.stable_id("protocol", core)}
    for row in responses:
        row["protocol_id"] = protocol["protocol_id"]
    for row in predictions:
        row["protocol_id"] = protocol["protocol_id"]
    protocol_path = tmp_path / "synthetic_protocol.json"
    response_path = tmp_path / "synthetic_responses.jsonl"
    prediction_path = tmp_path / "synthetic_predictions.jsonl"
    _write_json(protocol_path, protocol)
    _write_jsonl(response_path, responses)
    _write_jsonl(prediction_path, predictions)
    return protocol_path, response_path, prediction_path, protocol, responses, predictions


@pytest.fixture(scope="module")
def complete_fixture(tmp_path_factory: pytest.TempPathFactory):
    return make_complete_fixture(tmp_path_factory.mktemp("revision-analysis-synthetic"))


def test_known_factor_and_stage_effects_and_independent_n(complete_fixture) -> None:
    protocol_path, response_path, prediction_path, *_rest = complete_fixture
    study = analysis.load_and_validate(protocol_path, response_path, prediction_path)
    result = analysis.analyze(study, replicates=200, seed=17)
    by_key = {
        (row["model_size"], row["category"], row["factor_contrast"], row["stage_change"]): row
        for row in result["primary_metrics"]
    }
    for size in analysis.SIZES:
        assert by_key[(size, "empathy_opener", "advice_minus_information", "dpo_minus_sft")]["estimate_percentage_points"] == pytest.approx(100)
        assert by_key[(size, "empathy_opener", "advice_minus_information", "final_minus_dpo")]["estimate_percentage_points"] == pytest.approx(0)
        assert by_key[(size, "validation", "emotion_minus_neutral", "dpo_minus_sft")]["estimate_percentage_points"] == pytest.approx(100)
        assert by_key[(size, "validation", "emotion_minus_neutral", "final_minus_dpo")]["estimate_percentage_points"] == pytest.approx(-100)
        assert by_key[(size, "disclaimer", "interaction", "dpo_minus_sft")]["estimate_percentage_points"] == pytest.approx(100)
    assert result["input_summary"]["responses"] == 2880
    assert result["input_summary"]["independent_n"] == 40
    assert {row["independent_scenario_families"] for row in result["primary_metrics"]} == {40}
    assert {row["seed_replicates_per_cell"] for row in result["primary_metrics"]} == {3}


def test_protocol_missing_factorial_crossing_is_rejected(complete_fixture) -> None:
    *_paths, protocol, _responses, _predictions = complete_fixture
    broken = copy.deepcopy(protocol)
    broken["primary_grid"][-1]["condition"]["emotional_framing"] = False
    core = {key: value for key, value in broken.items() if key != "protocol_id"}
    broken["protocol_id"] = analysis.stable_id("protocol", core)
    with pytest.raises(ValueError, match="missing or duplicate factorial crossing"):
        analysis._validate_protocol_grid(broken)


def test_wrong_protocol_and_prediction_id_mismatch_are_rejected(tmp_path: Path, complete_fixture) -> None:
    protocol_path, _response_path, _prediction_path, _protocol, responses, predictions = complete_fixture
    wrong_responses = copy.deepcopy(responses)
    wrong_responses[0]["protocol_id"] = "protocol_wrong"
    wrong_response_path = tmp_path / "wrong_responses.jsonl"
    _write_jsonl(wrong_response_path, wrong_responses)
    with pytest.raises(ValueError, match="protocol_id mismatch"):
        analysis.load_and_validate(protocol_path, wrong_response_path, _prediction_path)

    wrong_predictions = copy.deepcopy(predictions)
    wrong_predictions[0]["response_id"] = "response_not_in_grid"
    wrong_prediction_path = tmp_path / "wrong_predictions.jsonl"
    _write_jsonl(wrong_prediction_path, wrong_predictions)
    with pytest.raises(ValueError, match="no matching response"):
        analysis.load_and_validate(protocol_path, _response_path, wrong_prediction_path)


def test_development_row_and_unpinned_dependency_are_rejected(tmp_path: Path, complete_fixture) -> None:
    protocol_path, response_path, _prediction_path, _protocol, _responses, predictions = complete_fixture
    development = copy.deepcopy(predictions)
    development[0]["split"] = "development"
    development_path = tmp_path / "development_predictions.jsonl"
    _write_jsonl(development_path, development)
    with pytest.raises(ValueError, match="non-evaluation"):
        analysis.load_and_validate(protocol_path, response_path, development_path)

    unpinned = copy.deepcopy(predictions)
    for row in unpinned:
        row["dependency_identity"]["embedding_model"]["resolved_revision"] = None
    unpinned_path = tmp_path / "unpinned_predictions.jsonl"
    _write_jsonl(unpinned_path, unpinned)
    with pytest.raises(ValueError, match="not consistently applied and pinned"):
        analysis.load_and_validate(protocol_path, response_path, unpinned_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda row: row["generation"]["config"].__setitem__("temperature", 0.2), "response_identity"),
        (lambda row: row["generation"]["config"].__setitem__("dtype", "torch.float16"), "dtype"),
    ),
)
def test_actual_decoding_and_dtype_must_match_declared_provenance(
    tmp_path: Path, complete_fixture, mutation, message: str,
) -> None:
    protocol_path, _response_path, prediction_path, _protocol, responses, _predictions = complete_fixture
    altered = copy.deepcopy(responses)
    mutation(altered[0])
    altered_path = tmp_path / f"altered_{message}.jsonl"
    _write_jsonl(altered_path, altered)
    with pytest.raises(ValueError, match=message):
        analysis.load_and_validate(protocol_path, altered_path, prediction_path)


def test_bootstrap_is_deterministic_and_zero_effects_are_explicit() -> None:
    domains = ["health"] * 20 + ["everyday"] * 20
    zero = np.zeros((40, 48))
    first = analysis.bootstrap_primary(zero, domains, replicates=250, seed=11)
    second = analysis.bootstrap_primary(zero, domains, replicates=250, seed=11)
    assert np.array_equal(first["indices"], second["indices"])
    assert np.array_equal(first["replicate_estimates"], second["replicate_estimates"])
    assert first["estimable_count"] == 0
    assert first["simultaneous_critical_value"] is None
    assert np.all(first["pointwise_lower"] == 0)
    assert np.all(first["pointwise_upper"] == 0)


def test_constant_noninteger_values_within_domains_are_degenerate() -> None:
    domains = ["health"] * 20 + ["everyday"] * 20
    values = np.zeros((40, 48))
    values[:20, 0] = 1 / 3
    values[20:, 0] = 2 / 3
    result = analysis.bootstrap_primary(values, domains, replicates=300, seed=29)
    assert result["bootstrap_standard_error"][0] >= 0
    assert not result["within_stratum_variation"][0]
    assert not result["estimable"][0]
    assert np.isnan(result["simultaneous_lower"][0])


def test_shared_cluster_resampling_preserves_metric_relationships() -> None:
    domains = ["health"] * 20 + ["everyday"] * 20
    values = np.zeros((40, 48))
    values[:, 0] = np.arange(40)
    values[:, 1] = 2 * values[:, 0]
    result = analysis.bootstrap_primary(values, domains, replicates=300, seed=23)
    assert result["indices"].shape == (300, 40)
    assert np.all(result["indices"][:, :20] < 20)
    assert np.all(result["indices"][:, 20:] >= 20)
    assert np.allclose(result["replicate_estimates"][:, 1], 2 * result["replicate_estimates"][:, 0])


def test_output_writer_refuses_synthetic_or_unconfirmed_results(tmp_path: Path) -> None:
    output = tmp_path / "forbidden-output"
    with pytest.raises(ValueError, match="real complete-archive confirmation"):
        analysis.write_outputs({"protocol_id": "synthetic"}, output, confirmed_real_complete_archive=False)
    assert not output.exists()

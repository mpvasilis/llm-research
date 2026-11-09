"""Strict matched-cap sensitivity prediction and analysis for the successor study.

This module never generates model responses. Prediction requires a complete,
strictly verified 192-response sensitivity run. Analysis pairs those seed-zero
responses only with the corresponding seed-zero rows from the complete primary
run; it never compares a longer response with a three-seed primary average.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from . import extended_revision_study as core
    from . import revision_analysis
except ImportError:  # pragma: no cover
    import extended_revision_study as core
    import revision_analysis


ANALYSIS_VERSION = "extended-budget-sensitivity-analysis-v1"
RESULT_SCHEMA_VERSION = "extended-budget-sensitivity-results-v1"
PREDICTION_VARIANT = "raw_output_robustness_four_category_union_provenance_v2"
WINDOWS = revision_analysis.WINDOWS
COMPONENTS = revision_analysis.COMPONENTS
CATEGORIES = revision_analysis.CATEGORIES
DOMAINS = revision_analysis.DOMAINS
REQUESTS = revision_analysis.REQUESTS
EMOTIONS = revision_analysis.EMOTIONS
CONTRASTS = revision_analysis.CONTRASTS
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_SEED = 20260915
EXPECTED_PRIMARY_RESPONSES = 2880
EXPECTED_SENSITIVITY_RESPONSES = 192
EXPECTED_SENSITIVITY_PREDICTIONS = 576
RESULT_FILENAME = "extended_sensitivity_results.json"
REPORT_FILENAME = "EXTENDED_SENSITIVITY_REPORT.md"
PRODUCER_PREDICTION_FIELDS = {
    "prediction_id", "prediction_schema_version", "study_version", "protocol_id", "run_id",
    "execution_profile_id", "response_id", "scenario_family_id", "split", "condition", "model_id",
    "checkpoint_alias", "model_size", "stage", "seed", "generation_output_tokens",
    "generation_stop_reason", "detector_version", "prediction_variant", "evaluation_unit",
    "decision_field", "dependency_identity", "window_metadata", "threshold", "predictions",
    "unavailable_categories",
}


def _read_json(path: Path) -> dict[str, Any]:
    path = core.fs_path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    yield from revision_analysis.iter_jsonl(core.fs_path(path))


def _canonical_rows(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(core.canonical_json(row) for row in rows)


def _run_rows(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((core.fs_path(run_dir) / "responses").glob("*.json"))
    return [_read_json(path) for path in paths]


def _validate_exact_export(
    run_dir: Path,
    export_path: Path,
    expected_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    run_rows = _run_rows(run_dir)
    export_rows = list(_iter_jsonl(export_path))
    run_ids = [row.get("response_id") for row in run_rows]
    export_ids = [row.get("response_id") for row in export_rows]
    if len(run_ids) != len(set(run_ids)) or set(run_ids) != expected_ids:
        raise ValueError("Run response files do not exactly equal the frozen response-ID grid")
    if len(export_ids) != len(set(export_ids)) or set(export_ids) != expected_ids:
        raise ValueError("Response export does not exactly equal the frozen response-ID grid")
    if _canonical_rows(run_rows) != _canonical_rows(export_rows):
        raise ValueError("Response export rows are not exactly equal to the actual run rows")
    return export_rows, {row["response_id"]: row for row in export_rows}


def _prediction_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "response_id": row["response_id"],
        "detector_version": row["detector_version"],
        "prediction_variant": row["prediction_variant"],
        "evaluation_unit": row["evaluation_unit"],
        "dependency_identity": row["dependency_identity"],
    }


def _validate_producer_predictions(
    rows: list[dict[str, Any]],
    run_id: str,
    expected_response_ids: set[str],
) -> None:
    expected_keys = {(response_id, window) for response_id in expected_response_ids for window in WINDOWS}
    observed_keys: set[tuple[str, str]] = set()
    for row in rows:
        if set(row) != PRODUCER_PREDICTION_FIELDS:
            raise ValueError("Prediction fields do not exactly reconstruct the producer schema")
        key = (row.get("response_id"), row.get("evaluation_unit"))
        if key in observed_keys:
            raise ValueError(f"Duplicate prediction response/window key: {key}")
        observed_keys.add(key)
        if row.get("run_id") != run_id:
            raise ValueError("Prediction run_id does not match the actual strict run")
        expected_prediction_id = core.shared.stable_id("prediction", _prediction_identity(row), 32)
        if row.get("prediction_id") != expected_prediction_id:
            raise ValueError("Prediction ID does not match the producer identity formula")
        if row.get("threshold") != core.shared.EMBEDDING_THRESHOLD:
            raise ValueError("Prediction threshold differs from the actual detector")
        if row.get("unavailable_categories") != {
            "crisis_referral": "not_applicable: absent from raw_output_robustness four-category baseline"
        }:
            raise ValueError("Prediction unavailable-category convention differs from the producer")
        for category in CATEGORIES:
            values = row["predictions"][category]
            if values.get("intersection") != (values.get("exact") and values.get("embedding")):
                raise ValueError("Prediction intersection flag is inconsistent")
            maximum = values.get("embedding_max_similarity")
            if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not np.isfinite(maximum):
                raise ValueError("Prediction embedding similarity must be a finite number")
            if values.get("embedding") != (maximum >= row["threshold"]):
                raise ValueError("Prediction embedding flag contradicts its recorded similarity and threshold")
    if observed_keys != expected_keys:
        raise ValueError("Prediction response/window keys do not exactly match the required grid")


def _strict_run(
    protocol_path: Path,
    run_dir: Path,
    responses_path: Path,
    expected_mode: str,
    expected_count: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol_path = core.fs_path(protocol_path)
    run_dir = core.fs_path(run_dir)
    responses_path = core.fs_path(responses_path)
    protocol = core.load_protocol(protocol_path)
    report = core.verify_run(protocol_path, run_dir, require_complete=True)
    strict_protocol, config, grid = core._validate_run_config(protocol_path, run_dir)
    if core.canonical_json(protocol) != core.canonical_json(strict_protocol):
        raise ValueError("Strict run protocol differs from the loaded successor protocol")
    if report.get("status") != "complete" or report.get("mode") != expected_mode:
        raise ValueError(f"Expected a complete {expected_mode} run")
    if len(grid) != expected_count or report.get("verified_responses") != expected_count:
        raise ValueError(f"Expected exactly {expected_count} verified {expected_mode} responses")
    expected_ids = {entry["response_id"] for entry in grid}
    if len(expected_ids) != expected_count or config.get("response_ids") != [entry["response_id"] for entry in grid]:
        raise ValueError("Run config does not contain the exact ordered frozen grid")
    _rows, by_id = _validate_exact_export(run_dir, responses_path, expected_ids)
    return protocol, config, grid, by_id


def _validate_primary(
    protocol_path: Path,
    run_dir: Path,
    responses_path: Path,
    predictions_path: Path,
) -> tuple[revision_analysis.ValidatedStudy, dict[str, Any], list[dict[str, Any]]]:
    protocol_path = core.fs_path(protocol_path)
    run_dir = core.fs_path(run_dir)
    responses_path = core.fs_path(responses_path)
    predictions_path = core.fs_path(predictions_path)
    protocol, config, grid, _responses = _strict_run(
        protocol_path, run_dir, responses_path, "evaluation", EXPECTED_PRIMARY_RESPONSES
    )
    study = revision_analysis.load_and_validate(protocol_path, responses_path, predictions_path)
    if core.canonical_json(study.protocol) != core.canonical_json(protocol):
        raise ValueError("Primary analyzer loaded a different protocol")
    if set(study.responses) != {entry["response_id"] for entry in grid}:
        raise ValueError("Primary analyzer did not validate the exact successor primary grid")
    prediction_rows = list(_iter_jsonl(predictions_path))
    if len(prediction_rows) != EXPECTED_PRIMARY_RESPONSES * len(WINDOWS):
        raise ValueError("Primary prediction export must contain exactly 8,640 rows")
    _validate_producer_predictions(prediction_rows, config["run_id"], set(study.responses))
    if {core.canonical_json(row["dependency_identity"]) for row in prediction_rows} != {
        core.canonical_json(study.dependency_identity)
    }:
        raise ValueError("Primary producer dependency identity is inconsistent")
    return study, config, prediction_rows


def _validate_sensitivity(
    protocol_path: Path,
    run_dir: Path,
    responses_path: Path,
    predictions_path: Path,
) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]],
    dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]],
    dict[str, Any], list[dict[str, Any]],
]:
    protocol, config, grid, responses = _strict_run(
        protocol_path, run_dir, responses_path, "sensitivity", EXPECTED_SENSITIVITY_RESPONSES
    )
    frozen_grid = protocol["longer_budget_sensitivity"]["grid"]
    if core.canonical_json(grid) != core.canonical_json(frozen_grid):
        raise ValueError("Sensitivity run is not bound to the frozen 192-response grid")
    aliases = {entry["response_id"]: entry["checkpoint_alias"] for entry in grid}
    prediction_rows = list(_iter_jsonl(predictions_path))
    predictions, dependency = revision_analysis._validate_predictions(
        prediction_rows, responses, aliases
    )
    if len(prediction_rows) != EXPECTED_SENSITIVITY_PREDICTIONS or len(predictions) != EXPECTED_SENSITIVITY_PREDICTIONS:
        raise ValueError("Sensitivity prediction export must contain exactly 576 rows")
    _validate_producer_predictions(prediction_rows, config["run_id"], set(responses))
    return protocol, config, grid, responses, predictions, dependency, prediction_rows


def _validate_sensitivity_grid(protocol: dict[str, Any], grid: list[dict[str, Any]]) -> dict[str, str]:
    if len(grid) != EXPECTED_SENSITIVITY_RESPONSES:
        raise ValueError("Sensitivity grid must contain exactly 192 rows")
    selected = protocol["longer_budget_sensitivity"].get("scenario_family_ids")
    families = sorted({entry["scenario_family_id"] for entry in grid})
    if not isinstance(selected, list) or sorted(selected) != families or len(families) != 8:
        raise ValueError("Sensitivity grid must use exactly the eight frozen selected families")
    family_domains: dict[str, str] = {}
    crossings: Counter[tuple[Any, ...]] = Counter()
    aliases = set(protocol["checkpoints"])
    for entry in grid:
        condition = entry["condition"]
        family = entry["scenario_family_id"]
        domain = condition["domain"]
        if family in family_domains and family_domains[family] != domain:
            raise ValueError("Sensitivity family crosses domains")
        family_domains[family] = domain
        if entry["logical_seed"] != 0:
            raise ValueError("Sensitivity grid must contain only logical seed zero")
        crossings[(family, condition["emotional_framing"], condition["request_type"], entry["checkpoint_alias"])] += 1
    if Counter(family_domains.values()) != Counter({"health": 4, "everyday": 4}):
        raise ValueError("Sensitivity family selection must contain four families per domain")
    expected = {
        (family, emotion, request, alias)
        for family in families for emotion in EMOTIONS for request in REQUESTS for alias in aliases
    }
    if set(crossings) != expected or any(count != 1 for count in crossings.values()):
        raise ValueError("Sensitivity grid must cross four variants with all six checkpoints exactly once")
    return family_domains


def _pair_grids(
    protocol: dict[str, Any], sensitivity_grid: list[dict[str, Any]]
) -> dict[tuple[str, str, int], tuple[dict[str, Any], dict[str, Any]]]:
    family_domains = _validate_sensitivity_grid(protocol, sensitivity_grid)
    selected_families = set(family_domains)
    primary_rows = [
        entry for entry in protocol["primary_grid"]
        if entry["logical_seed"] == 0 and entry["scenario_family_id"] in selected_families
    ]
    primary = {(entry["checkpoint_alias"], entry["prompt_id"], entry["logical_seed"]): entry for entry in primary_rows}
    sensitivity = {(entry["checkpoint_alias"], entry["prompt_id"], entry["logical_seed"]): entry for entry in sensitivity_grid}
    if len(primary) != EXPECTED_SENSITIVITY_RESPONSES or set(primary) != set(sensitivity):
        raise ValueError("Primary seed-zero rows do not pair exactly with all sensitivity rows")
    pairs = {}
    compare_fields = (
        "checkpoint_alias", "prompt_id", "scenario_family_id", "split", "condition", "prompt", "logical_seed"
    )
    for key in sorted(primary):
        short, long = primary[key], sensitivity[key]
        if any(short[field] != long[field] for field in compare_fields):
            raise ValueError("Matched cap grids differ outside generation budget")
        short_identity = dict(short["response_identity"])
        long_identity = dict(long["response_identity"])
        short_generation = dict(short_identity.pop("generation_config"))
        long_generation = dict(long_identity.pop("generation_config"))
        if core.canonical_json(short_identity) != core.canonical_json(long_identity):
            raise ValueError("Matched cap identity metadata or RNG differs")
        short_cap = short_generation.pop("max_new_tokens")
        long_cap = long_generation.pop("max_new_tokens")
        if (
            core.canonical_json(short_generation) != core.canonical_json(long_generation)
            or short_cap != protocol["primary_cap"]
            or long_cap != protocol["sensitivity_cap"]
        ):
            raise ValueError("Matched generation configs must differ only by their frozen cap")
        pairs[key] = (short, long)
    return pairs


def predict_complete_sensitivity(protocol_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    """Score all windows of a complete frozen 192-response sensitivity run."""
    protocol_path = core.fs_path(protocol_path)
    run_dir = core.fs_path(run_dir)
    output = core.fs_path(output)
    protocol = core.load_protocol(protocol_path)
    report = core.verify_run(protocol_path, run_dir, require_complete=True)
    strict_protocol, config, grid = core._validate_run_config(protocol_path, run_dir)
    if core.canonical_json(protocol) != core.canonical_json(strict_protocol):
        raise ValueError("Strict sensitivity run protocol mismatch")
    if report.get("mode") != "sensitivity" or report.get("verified_responses") != EXPECTED_SENSITIVITY_RESPONSES:
        raise ValueError("Prediction export requires the complete 192-response sensitivity run")
    _validate_sensitivity_grid(protocol, grid)
    expected_ids = {entry["response_id"] for entry in protocol["longer_budget_sensitivity"]["grid"]}
    rows = sorted(_run_rows(run_dir), key=lambda row: row["response_id"])
    if {row["response_id"] for row in rows} != expected_ids or len(rows) != EXPECTED_SENSITIVITY_RESPONSES:
        raise ValueError("Actual sensitivity responses do not exactly match the frozen grid")
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite sensitivity prediction export: {core.display_path(output)}"
        )

    dependencies = core.shared.prediction_dependencies(include_embedding=True, include_prefix=True)
    tokenizer_dependency = dependencies["prefix_window_tokenizer"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dependency["model_id"],
        revision=tokenizer_dependency["resolved_revision"],
        local_files_only=True,
    )
    prepared = [
        (row, unit, text, metadata)
        for row in rows
        for unit, text, metadata in core.evaluation_windows(
            row["generation"]["response"], tokenizer, tokenizer_dependency
        )
    ]
    model, exemplar_vectors = core.shared.load_embedding_detector(
        local_files_only=True,
        revision=dependencies["embedding_model"]["resolved_revision"],
    )
    embedding, maxima = core.shared.embedding_flags(
        model, exemplar_vectors, [text for _row, _unit, text, _metadata in prepared]
    )
    by_id = {entry["response_id"]: entry for entry in grid}
    predictions: list[dict[str, Any]] = []
    for index, (row, unit, text, metadata) in enumerate(prepared):
        exact = core.shared.exact_flags(text)
        components = {
            category: {
                "exact": exact[category],
                "embedding": embedding[index][category],
                "union": exact[category] or embedding[index][category],
                "intersection": exact[category] and embedding[index][category],
                "embedding_max_similarity": maxima[index][category],
            }
            for category in core.shared.CATEGORIES
        }
        entry = by_id[row["response_id"]]
        alias = entry["checkpoint_alias"]
        size, raw_stage = alias.split("_", 1)
        identity = {
            "response_id": row["response_id"],
            "detector_version": core.shared.DETECTOR_VERSION,
            "prediction_variant": PREDICTION_VARIANT,
            "evaluation_unit": unit,
            "dependency_identity": dependencies,
        }
        prediction = {
            "prediction_id": core.shared.stable_id("prediction", identity, 32),
            "prediction_schema_version": core.shared.PREDICTION_SCHEMA_VERSION,
            "study_version": row["study_version"],
            "protocol_id": protocol["protocol_id"],
            "run_id": config["run_id"],
            "execution_profile_id": config["execution_profile"]["execution_profile_id"],
            "response_id": row["response_id"],
            "scenario_family_id": row["scenario_family_id"],
            "split": row["split"],
            "condition": row["condition"],
            "model_id": row["model_id"],
            "checkpoint_alias": alias,
            "model_size": size,
            "stage": "final" if raw_stage == "instruct" else raw_stage,
            "seed": row["seed"],
            "generation_output_tokens": row["generation"]["output_tokens"],
            "generation_stop_reason": row["generation"]["stopping"]["reason"],
            "detector_version": core.shared.DETECTOR_VERSION,
            "prediction_variant": PREDICTION_VARIANT,
            "evaluation_unit": unit,
            "decision_field": "union",
            "dependency_identity": dependencies,
            "window_metadata": metadata,
            "threshold": core.shared.EMBEDDING_THRESHOLD,
            "predictions": components,
            "unavailable_categories": {
                "crisis_referral": "not_applicable: absent from raw_output_robustness four-category baseline"
            },
        }
        core.shared.validate_prediction_record(prediction)
        predictions.append(prediction)
    if len(predictions) != EXPECTED_SENSITIVITY_PREDICTIONS:
        raise ValueError("Prediction export must contain exactly 576 rows")
    aliases = {entry["response_id"]: entry["checkpoint_alias"] for entry in grid}
    response_map = {row["response_id"]: row for row in rows}
    revision_analysis._validate_predictions(predictions, response_map, aliases)
    _validate_producer_predictions(predictions, config["run_id"], expected_ids)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sorted(predictions, key=lambda item: item["prediction_id"]))
    _atomic_new_file(output, text)
    return {
        "status": "complete",
        "responses": EXPECTED_SENSITIVITY_RESPONSES,
        "prediction_rows": EXPECTED_SENSITIVITY_PREDICTIONS,
        "evaluation_units": list(WINDOWS),
        "dependency_identity": dependencies,
        "run_id": config["run_id"],
        "output": core.display_path(output),
        "output_sha256": core.sha256_file(output),
    }


def _factor_contrasts(cell: Mapping[tuple[bool, str], float]) -> dict[str, float]:
    return revision_analysis._factor_contrasts(dict(cell))


def _condition_rates(
    pair_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[tuple[float, float]]] = defaultdict(list)
    families: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in pair_rows:
        condition = row["condition"]
        for window in WINDOWS:
            for component in COMPONENTS:
                for category in CATEGORIES:
                    key = (
                        row["checkpoint_alias"], condition["domain"], condition["emotional_framing"],
                        condition["request_type"], window, component, category,
                    )
                    buckets[key].append((
                        float(row["primary_predictions"][window][category][component]),
                        float(row["sensitivity_predictions"][window][category][component]),
                    ))
                    families[key].add(row["scenario_family_id"])
    results = []
    for key in sorted(buckets):
        values = np.asarray(buckets[key], dtype=float)
        alias, domain, emotion, request, window, component, category = key
        results.append({
            "checkpoint_alias": alias,
            "domain": domain,
            "emotional_framing": emotion,
            "request_type": request,
            "window": window,
            "component": component,
            "category": category,
            "matched_pairs": len(values),
            "scenario_families": len(families[key]),
            "primary_seed0_rate": float(values[:, 0].mean()),
            "sensitivity_seed0_rate": float(values[:, 1].mean()),
            "sensitivity_minus_primary_rate": float((values[:, 1] - values[:, 0]).mean()),
        })
    return results


def _response_summaries(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        condition = row["condition"]
        key = (
            row["checkpoint_alias"], condition["domain"],
            condition["emotional_framing"], condition["request_type"],
        )
        buckets[key].append(row)
    results = []
    for key in sorted(buckets):
        rows = buckets[key]
        primary_lengths = np.asarray([row["primary_response"]["generation"]["output_tokens"] for row in rows])
        sensitivity_lengths = np.asarray([row["sensitivity_response"]["generation"]["output_tokens"] for row in rows])
        alias, domain, emotion, request = key
        results.append({
            "checkpoint_alias": alias,
            "domain": domain,
            "emotional_framing": emotion,
            "request_type": request,
            "matched_pairs": len(rows),
            "scenario_families": len({row["scenario_family_id"] for row in rows}),
            "primary_mean_output_tokens": float(primary_lengths.mean()),
            "sensitivity_mean_output_tokens": float(sensitivity_lengths.mean()),
            "mean_paired_output_token_change": float((sensitivity_lengths - primary_lengths).mean()),
            "primary_median_output_tokens": float(np.median(primary_lengths)),
            "sensitivity_median_output_tokens": float(np.median(sensitivity_lengths)),
            "primary_cap_rate": float(np.mean([row["primary_response"]["generation"]["stopping"]["cap_reached"] for row in rows])),
            "sensitivity_cap_rate": float(np.mean([row["sensitivity_response"]["generation"]["stopping"]["cap_reached"] for row in rows])),
            "primary_eos_rate": float(np.mean([row["primary_response"]["generation"]["stopping"]["eos_reached"] for row in rows])),
            "sensitivity_eos_rate": float(np.mean([row["sensitivity_response"]["generation"]["stopping"]["eos_reached"] for row in rows])),
            "decoded_primary_text_is_sensitivity_prefix_rate": float(np.mean([row["decoded_prefix_agreement"] for row in rows])),
            "decoded_prefix_comparison": "exact decoded Python string prefix; not token-ID identity",
        })
    return results


def _family_contrasts(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[Any, ...], dict[tuple[bool, str], tuple[float, float]]] = defaultdict(dict)
    domains: dict[str, str] = {}
    for row in pair_rows:
        family = row["scenario_family_id"]
        condition = row["condition"]
        domains[family] = condition["domain"]
        factor = (condition["emotional_framing"], condition["request_type"])
        for window in WINDOWS:
            for component in COMPONENTS:
                for category in CATEGORIES:
                    key = (family, row["checkpoint_alias"], window, component, category)
                    if factor in cells[key]:
                        raise ValueError("Duplicate family-condition pair while forming contrasts")
                    cells[key][factor] = (
                        float(row["primary_predictions"][window][category][component]),
                        float(row["sensitivity_predictions"][window][category][component]),
                    )
    records = []
    for key in sorted(cells):
        family, alias, window, component, category = key
        values = cells[key]
        primary = _factor_contrasts({factor: pair[0] for factor, pair in values.items()})
        sensitivity = _factor_contrasts({factor: pair[1] for factor, pair in values.items()})
        for contrast in CONTRASTS:
            records.append({
                "scenario_family_id": family,
                "domain": domains[family],
                "checkpoint_alias": alias,
                "window": window,
                "component": component,
                "category": category,
                "factor_contrast": contrast,
                "primary_seed0": primary[contrast],
                "sensitivity_seed0": sensitivity[contrast],
                "sensitivity_minus_primary": sensitivity[contrast] - primary[contrast],
            })
    return records


def _interval(values: np.ndarray, replicates: int) -> tuple[float | None, float | None]:
    if replicates == 0:
        return None, None
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _contrast_summaries(
    family_records: list[dict[str, Any]], replicates: int, seed: int
) -> list[dict[str, Any]]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 0:
        raise ValueError("bootstrap_replicates must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    by_spec: dict[tuple[Any, ...], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in family_records:
        spec = (
            row["checkpoint_alias"], row["domain"], row["window"], row["component"],
            row["category"], row["factor_contrast"],
        )
        by_spec[spec][row["scenario_family_id"]] = {
            "primary_seed0": row["primary_seed0"],
            "sensitivity_seed0": row["sensitivity_seed0"],
            "sensitivity_minus_primary": row["sensitivity_minus_primary"],
        }
    rng = np.random.default_rng(seed)
    results = []
    for domain in DOMAINS:
        specs = sorted(spec for spec in by_spec if spec[1] == domain)
        domain_families = sorted({family for spec in specs for family in by_spec[spec]})
        if len(domain_families) != 4 or any(set(by_spec[spec]) != set(domain_families) for spec in specs):
            raise ValueError("Each domain-level contrast must contain the same four sensitivity families")
        indices = rng.integers(0, 4, size=(replicates, 4)) if replicates else None
        for spec in specs:
            alias, _domain, window, component, category, contrast = spec
            matrix = np.asarray([
                [by_spec[spec][family][name] for name in ("primary_seed0", "sensitivity_seed0", "sensitivity_minus_primary")]
                for family in domain_families
            ], dtype=float)
            boot = matrix[indices].mean(axis=1) if replicates else None
            result = {
                "checkpoint_alias": alias,
                "domain": domain,
                "window": window,
                "component": component,
                "category": category,
                "factor_contrast": contrast,
                "scenario_families": 4,
                "primary_seed0_estimate": float(matrix[:, 0].mean()),
                "sensitivity_seed0_estimate": float(matrix[:, 1].mean()),
                "sensitivity_minus_primary_estimate": float(matrix[:, 2].mean()),
            }
            for index, name in enumerate(("primary_seed0", "sensitivity_seed0", "sensitivity_minus_primary")):
                lower, upper = _interval(boot[:, index], replicates) if boot is not None else (None, None)
                result[f"{name}_pointwise_95_lower"] = lower
                result[f"{name}_pointwise_95_upper"] = upper
            results.append(result)
    return results


def _load_inputs(
    protocol_path: Path,
    primary_run_dir: Path,
    primary_responses_path: Path,
    primary_predictions_path: Path,
    sensitivity_run_dir: Path,
    sensitivity_responses_path: Path,
    sensitivity_predictions_path: Path,
) -> dict[str, Any]:
    primary, primary_config, primary_prediction_rows = _validate_primary(
        protocol_path, primary_run_dir, primary_responses_path, primary_predictions_path
    )
    (
        protocol, sensitivity_config, sensitivity_grid, sensitivity_responses,
        sensitivity_predictions, sensitivity_dependency, sensitivity_prediction_rows,
    ) = _validate_sensitivity(
        protocol_path, sensitivity_run_dir, sensitivity_responses_path, sensitivity_predictions_path
    )
    if core.canonical_json(primary.protocol) != core.canonical_json(protocol):
        raise ValueError("Primary and sensitivity inputs do not use the same successor protocol")
    if core.canonical_json(primary_config["execution_profile"]) != core.canonical_json(sensitivity_config["execution_profile"]):
        raise ValueError("Primary and sensitivity runs must use equal generation execution profiles")
    if core.canonical_json(primary.dependency_identity) != core.canonical_json(sensitivity_dependency):
        raise ValueError("Primary and sensitivity predictions must use the same detector dependency identity")
    grid_pairs = _pair_grids(protocol, sensitivity_grid)
    pair_rows = []
    for key, (primary_entry, sensitivity_entry) in grid_pairs.items():
        primary_response = primary.responses[primary_entry["response_id"]]
        sensitivity_response = sensitivity_responses[sensitivity_entry["response_id"]]
        if primary_response["execution_profile_id"] != sensitivity_response["execution_profile_id"]:
            raise ValueError("Matched responses have different execution profiles")
        primary_by_window = {window: primary.predictions[(primary_entry["response_id"], window)]["predictions"] for window in WINDOWS}
        sensitivity_by_window = {window: sensitivity_predictions[(sensitivity_entry["response_id"], window)]["predictions"] for window in WINDOWS}
        pair_rows.append({
            "pair_key": {"checkpoint_alias": key[0], "prompt_id": key[1], "logical_seed": key[2]},
            "checkpoint_alias": key[0],
            "prompt_id": key[1],
            "scenario_family_id": primary_entry["scenario_family_id"],
            "condition": primary_entry["condition"],
            "primary_response": primary_response,
            "sensitivity_response": sensitivity_response,
            "primary_predictions": primary_by_window,
            "sensitivity_predictions": sensitivity_by_window,
            "decoded_prefix_agreement": sensitivity_response["generation"]["response"].startswith(primary_response["generation"]["response"]),
        })
    if len(pair_rows) != EXPECTED_SENSITIVITY_RESPONSES:
        raise ValueError("Matched analysis must contain exactly 192 seed-zero pairs")
    return {
        "protocol": protocol,
        "primary": primary,
        "primary_config": primary_config,
        "primary_prediction_rows": primary_prediction_rows,
        "sensitivity_config": sensitivity_config,
        "sensitivity_dependency": sensitivity_dependency,
        "sensitivity_prediction_rows": sensitivity_prediction_rows,
        "pair_rows": pair_rows,
    }


def _analysis_result(
    loaded: dict[str, Any],
    paths: Mapping[str, Path],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    protocol = loaded["protocol"]
    pair_rows = loaded["pair_rows"]
    family_records = _family_contrasts(pair_rows)
    summaries = _contrast_summaries(family_records, bootstrap_replicates, seed)
    source_hashes = dict(protocol["source_sha256"])
    source_hashes["experiments/extended_sensitivity_analysis.py"] = core.sha256_file(core.fs_path(__file__))
    result_core = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete_limited_eight_family_seed0_sensitivity",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": core.sha256_file(paths["protocol"]),
        "caps": {"primary": protocol["primary_cap"], "sensitivity": protocol["sensitivity_cap"]},
        "runs": {
            "primary": {
                "run_id": loaded["primary_config"]["run_id"],
                "mode": loaded["primary_config"]["mode"],
                "execution_profile": loaded["primary_config"]["execution_profile"],
                "run_config_sha256": core.sha256_file(paths["primary_run"] / "run_config.json"),
            },
            "sensitivity": {
                "run_id": loaded["sensitivity_config"]["run_id"],
                "mode": loaded["sensitivity_config"]["mode"],
                "execution_profile": loaded["sensitivity_config"]["execution_profile"],
                "run_config_sha256": core.sha256_file(paths["sensitivity_run"] / "run_config.json"),
            },
        },
        "input_sha256": {name: core.sha256_file(path) for name, path in paths.items() if name not in {"primary_run", "sensitivity_run"}},
        "source_sha256": dict(sorted(source_hashes.items())),
        "detector_dependency_identity": loaded["primary"].dependency_identity,
        "counts": {
            "primary_validated_responses": len(loaded["primary"].responses),
            "primary_validated_predictions": len(loaded["primary"].predictions),
            "sensitivity_validated_responses": EXPECTED_SENSITIVITY_RESPONSES,
            "sensitivity_validated_predictions": len(loaded["sensitivity_prediction_rows"]),
            "matched_seed0_pairs": len(pair_rows),
            "selected_scenario_families": len({row["scenario_family_id"] for row in pair_rows}),
            "selected_family_domains": dict(sorted(Counter({
                row["scenario_family_id"]: row["condition"]["domain"] for row in pair_rows
            }.values()).items())),
            "family_contrast_rows": len(family_records),
            "contrast_summary_rows": len(summaries),
        },
        "matching": {
            "key": ["checkpoint_alias", "prompt_id", "logical_seed=0"],
            "primary_seed_policy": "seed zero only; primary seeds one and two are excluded",
            "required_equal_metadata": "family, condition, prompt, checkpoint, model/tokenizer/serialization pins, RNG identity, and generation config except max_new_tokens",
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "method": "pointwise 2.5th/97.5th percentile domain-stratified scenario-family bootstrap",
            "independent_units": "four selected scenario families per domain",
            "shared_resampling": "one family-index draw per domain and replicate, shared across caps, checkpoints, detector components, categories, windows, and contrasts",
            "zero_replicates": "interval fields are null",
        },
        "condition_rates": _condition_rates(pair_rows),
        "response_summaries": _response_summaries(pair_rows),
        "family_contrasts": family_records,
        "contrast_summaries": summaries,
        "interpretation_limits": [
            "This is a limited pre-specified eight-family, seed-zero sensitivity analysis.",
            "Longer-minus-primary values are paired point-estimate changes; they do not establish sign reversals.",
            "Decoded-prefix agreement is exact string-prefix evidence, not token-ID identity.",
            "The detector is not semantic or human validation of completeness, accuracy, or advice quality.",
            "The analysis is descriptive and supports no causal claim or generation-budget adequacy claim.",
        ],
    }
    return {**result_core, "result_id": core.stable_id("extended_sensitivity_result", result_core, 32)}


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Extended longer-budget sensitivity",
        "",
        f"Protocol `{result['protocol_id']}` pairs exactly {result['counts']['matched_seed0_pairs']} sensitivity responses with primary seed-zero responses from eight pre-specified evaluation families (four per domain) across six checkpoints and four prompt variants.",
        "",
        f"Primary and sensitivity caps are {result['caps']['primary']} and {result['caps']['sensitivity']} tokens. The analysis validated {result['counts']['primary_validated_responses']:,}/{result['counts']['primary_validated_predictions']:,} primary response/prediction rows and {result['counts']['sensitivity_validated_responses']}/{result['counts']['sensitivity_validated_predictions']} sensitivity rows. Primary seeds 1 and 2 are excluded from matched comparisons.",
        "",
        f"Pointwise intervals use {result['bootstrap']['replicates']:,} domain-stratified family bootstrap replicates with shared resampling across both caps and every reported metric. Each domain contributes only four scenario families.",
        "",
        "## Response length and stopping",
        "",
        "| Checkpoint | Primary mean tokens | Longer mean tokens | Paired change | Primary cap rate | Longer cap rate | Decoded prefix rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result["response_summaries"]:
        grouped[row["checkpoint_alias"]].append(row)
    for alias in sorted(grouped):
        rows = grouped[alias]
        def mean(field: str) -> float:
            return float(np.mean([row[field] for row in rows]))

        lines.append(
            f"| {alias} | {mean('primary_mean_output_tokens'):.1f} | {mean('sensitivity_mean_output_tokens'):.1f} | "
            f"{mean('mean_paired_output_token_change'):+.1f} | {mean('primary_cap_rate'):.3f} | "
            f"{mean('sensitivity_cap_rate'):.3f} | {mean('decoded_primary_text_is_sensitivity_prefix_rate'):.3f} |"
        )
    lines.extend([
        "",
        "The results JSON contains all exact/embedding/union rates for full, prefix-64, and prefix-128 windows, plus every family-level factorial contrast and its paired longer-minus-primary change.",
        "",
        "These are descriptive point estimates and pointwise intervals for a limited eight-family, seed-zero sensitivity. They do not establish sign reversals, causal effects, semantic completeness, human validity, or budget adequacy. Decoded prefix agreement is exact string-prefix evidence and does not establish token-ID identity.",
        "",
    ])
    return "\n".join(lines)


def _atomic_new_file(path: Path, text: str) -> None:
    path = core.fs_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {core.display_path(path)}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_new_output_dir(output_dir: Path, files: Mapping[str, str]) -> None:
    output_dir = core.fs_path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing analysis output directory: {core.display_path(output_dir)}"
        )
    temporary = core.fs_path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for name, text in files.items():
            path = temporary / name
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        os.rename(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def analyze(
    protocol_path: Path,
    primary_run_dir: Path,
    primary_responses_path: Path,
    primary_predictions_path: Path,
    sensitivity_run_dir: Path,
    sensitivity_responses_path: Path,
    sensitivity_predictions_path: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Strictly validate both runs and atomically persist matched sensitivity results."""
    paths = {
        "protocol": core.fs_path(protocol_path),
        "primary_run": core.fs_path(primary_run_dir),
        "primary_responses": core.fs_path(primary_responses_path),
        "primary_predictions": core.fs_path(primary_predictions_path),
        "sensitivity_run": core.fs_path(sensitivity_run_dir),
        "sensitivity_responses": core.fs_path(sensitivity_responses_path),
        "sensitivity_predictions": core.fs_path(sensitivity_predictions_path),
    }
    output_dir = core.fs_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing analysis output directory: {core.display_path(output_dir)}"
        )
    loaded = _load_inputs(
        paths["protocol"], paths["primary_run"], paths["primary_responses"], paths["primary_predictions"],
        paths["sensitivity_run"], paths["sensitivity_responses"], paths["sensitivity_predictions"],
    )
    result = _analysis_result(loaded, paths, bootstrap_replicates, seed)
    result_text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report_text = _markdown_report(result)
    _atomic_new_output_dir(output_dir, {RESULT_FILENAME: result_text, REPORT_FILENAME: report_text})
    return {
        "status": result["status"],
        "result_id": result["result_id"],
        "matched_pairs": result["counts"]["matched_seed0_pairs"],
        "result_path": core.display_path(output_dir / RESULT_FILENAME),
        "result_sha256": core.sha256_file(output_dir / RESULT_FILENAME),
        "report_path": core.display_path(output_dir / REPORT_FILENAME),
        "report_sha256": core.sha256_file(output_dir / REPORT_FILENAME),
    }


def _preview_prediction(protocol_path: Path, run_dir: Path, output: Path) -> dict[str, Any]:
    protocol_path = core.fs_path(protocol_path)
    run_dir = core.fs_path(run_dir)
    output = core.fs_path(output)
    protocol = core.load_protocol(protocol_path)
    report = core.verify_run(protocol_path, run_dir, require_complete=True)
    _protocol, config, grid = core._validate_run_config(protocol_path, run_dir)
    if report.get("mode") != "sensitivity" or len(grid) != EXPECTED_SENSITIVITY_RESPONSES:
        raise ValueError("Prediction preview requires a complete sensitivity run")
    _validate_sensitivity_grid(protocol, grid)
    return {
        "status": "validated_dry_run",
        "execute_required": True,
        "run_id": config["run_id"],
        "responses": EXPECTED_SENSITIVITY_RESPONSES,
        "expected_prediction_rows": EXPECTED_SENSITIVITY_PREDICTIONS,
        "output": core.display_path(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict")
    predict.add_argument("--protocol", type=Path, required=True)
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--execute", action="store_true")
    analyze_parser = commands.add_parser("analyze")
    for name in (
        "protocol", "primary-run-dir", "primary-responses", "primary-predictions",
        "sensitivity-run-dir", "sensitivity-responses", "sensitivity-predictions", "output-dir",
    ):
        analyze_parser.add_argument(f"--{name}", type=Path, required=True)
    analyze_parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    analyze_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    analyze_parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "predict":
        result = (
            predict_complete_sensitivity(args.protocol, args.run_dir, args.output)
            if args.execute else _preview_prediction(args.protocol, args.run_dir, args.output)
        )
    elif args.execute:
        result = analyze(
            args.protocol, args.primary_run_dir, args.primary_responses, args.primary_predictions,
            args.sensitivity_run_dir, args.sensitivity_responses, args.sensitivity_predictions,
            args.output_dir, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed,
        )
    else:
        loaded = _load_inputs(
            args.protocol, args.primary_run_dir, args.primary_responses, args.primary_predictions,
            args.sensitivity_run_dir, args.sensitivity_responses, args.sensitivity_predictions,
        )
        result = {
            "status": "validated_dry_run",
            "execute_required": True,
            "primary_responses": len(loaded["primary"].responses),
            "sensitivity_pairs": len(loaded["pair_rows"]),
            "output_dir": core.display_path(core.fs_path(args.output_dir)),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

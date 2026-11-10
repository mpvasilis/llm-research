"""Strict prospective checkpoint analysis for the revision study.

The CLI writes results only for a complete real evaluation archive.  The
library functions accept complete synthetic archives so the frozen analysis
can be tested before responses exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_VERSION = "revision-checkpoint-analysis-v1"
PROTOCOL_SCHEMA_VERSION = "revision-study-protocol-v1"
RESPONSE_SCHEMA_VERSION = "revision-study-response-v2"
PREDICTION_SCHEMA_VERSION = "historical-detector-prediction-v2"
STUDY_VERSION = "revision-study-v1"
FULL_RUNNER_VERSION = "revision-study-full-runner-v2"
EXPECTED_PREDICTION_VARIANT = "raw_output_robustness_four_category_union_provenance_v2"
EXPECTED_DETECTOR_VERSION = "historical-raw-output-robustness-v1"
WINDOWS = ("full_response", "prefix_64_tokens", "prefix_128_tokens")
COMPONENTS = ("exact", "embedding", "union")
CATEGORIES = ("empathy_opener", "validation", "disclaimer", "structure")
SIZES = ("1b", "7b")
STAGES = ("sft", "dpo", "final")
SEEDS = (0, 1, 2)
DOMAINS = ("health", "everyday")
REQUESTS = ("information", "advice")
EMOTIONS = (False, True)
CONTRASTS = ("advice_minus_information", "emotion_minus_neutral", "interaction")
STAGE_CHANGES = ("dpo_minus_sft", "final_minus_dpo")
PRIMARY_METRIC_COUNT = 48
DEFAULT_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260915
DEFAULT_OUTPUT = ROOT / "output" / "revision" / "checkpoint_analysis"
ROUND_OFF_MULTIPLIER = 64.0


@dataclass(frozen=True)
class ValidatedStudy:
    protocol: dict[str, Any]
    responses: dict[str, dict[str, Any]]
    predictions: dict[tuple[str, str], dict[str, Any]]
    grid_by_response_id: dict[str, dict[str, Any]]
    alias_by_response_id: dict[str, str]
    family_domains: dict[str, str]
    dependency_identity: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any, length: int = 32) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _require_keys(row: dict[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in row or row[key] in (None, "")]
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")


def _require_present_keys(row: dict[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(missing)}")


def _is_pinned_revision(value: Any) -> bool:
    """Accept immutable hexadecimal content revisions, never moving refs."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", value) is not None


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    if protocol.get("protocol_schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Wrong protocol schema version")
    if protocol.get("study_version") != STUDY_VERSION:
        raise ValueError("Wrong study version")
    if protocol.get("runner_version") != FULL_RUNNER_VERSION:
        raise ValueError("Protocol was not frozen by the supported full runner")
    claimed = protocol.get("protocol_id")
    core = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if claimed != stable_id("protocol", core):
        raise ValueError("Protocol content does not match protocol_id")
    return protocol


def _validate_protocol_grid(protocol: dict[str, Any]) -> tuple[
    dict[str, dict[str, Any]], dict[str, str], dict[str, str]
]:
    grid = protocol.get("primary_grid")
    if protocol.get("primary_expected_responses") != 2880 or not isinstance(grid, list) or len(grid) != 2880:
        raise ValueError("Protocol does not contain the complete 2,880-response primary grid")
    checkpoints = protocol.get("checkpoints")
    expected_aliases = {f"{size}_{suffix}" for size in SIZES for suffix in ("sft", "dpo", "instruct")}
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected_aliases:
        raise ValueError("Protocol checkpoint aliases differ from the frozen six-checkpoint design")

    grid_by_id: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    family_domains: dict[str, str] = {}
    crossings: Counter[tuple[Any, ...]] = Counter()
    for entry in grid:
        _require_keys(entry, (
            "response_id", "checkpoint_alias", "prompt_id", "scenario_family_id", "split",
            "condition", "logical_seed", "response_identity", "prompt",
        ), "protocol grid entry")
        response_id = entry["response_id"]
        if response_id in grid_by_id:
            raise ValueError(f"Duplicate response_id in protocol grid: {response_id}")
        if entry["split"] != "evaluation":
            raise ValueError("Protocol primary grid contains a non-evaluation row")
        condition = entry["condition"]
        if not isinstance(condition, dict) or set(condition) != {"domain", "emotional_framing", "request_type"}:
            raise ValueError(f"Invalid condition for {response_id}")
        domain = condition["domain"]
        emotion = condition["emotional_framing"]
        request = condition["request_type"]
        if domain not in DOMAINS or type(emotion) is not bool or request not in REQUESTS:
            raise ValueError(f"Invalid factor value for {response_id}")
        alias = entry["checkpoint_alias"]
        seed = entry["logical_seed"]
        if alias not in expected_aliases or seed not in SEEDS:
            raise ValueError(f"Invalid checkpoint or seed for {response_id}")
        family = entry["scenario_family_id"]
        if family in family_domains and family_domains[family] != domain:
            raise ValueError(f"Scenario family crosses domains: {family}")
        family_domains[family] = domain
        crossing = (family, emotion, request, alias, seed)
        crossings[crossing] += 1
        grid_by_id[response_id] = entry
        aliases[response_id] = alias

    domain_counts = Counter(family_domains.values())
    if domain_counts != Counter({"health": 20, "everyday": 20}):
        raise ValueError(f"Protocol family balance is not 20 health/20 everyday: {dict(domain_counts)}")
    expected_crossings = {
        (family, emotion, request, alias, seed)
        for family in family_domains
        for emotion in EMOTIONS
        for request in REQUESTS
        for alias in expected_aliases
        for seed in SEEDS
    }
    if set(crossings) != expected_crossings or any(count != 1 for count in crossings.values()):
        missing = len(expected_crossings - set(crossings))
        extra = len(set(crossings) - expected_crossings)
        raise ValueError(f"Protocol grid has a missing or duplicate factorial crossing (missing={missing}, extra={extra})")
    return grid_by_id, aliases, family_domains


def _validate_responses(
    rows: list[dict[str, Any]],
    protocol: dict[str, Any],
    grid_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        from . import full_revision_study as full_runner
    except ImportError:  # direct script execution
        import full_revision_study as full_runner

    if len(rows) != 2880:
        raise ValueError(f"Response archive must contain exactly 2,880 rows, found {len(rows)}")
    responses: dict[str, dict[str, Any]] = {}
    execution_ids: set[str] = set()
    execution_bodies: set[str] = set()
    for row in rows:
        _require_keys(row, (
            "response_schema_version", "study_version", "runner_version", "protocol_id",
            "execution_profile_id", "execution_profile", "response_id", "response_identity",
            "prompt_id", "scenario_family_id", "split", "condition", "model_id",
            "model_revision", "tokenizer_revision", "model_hash", "serialization_id", "seed",
            "generation",
        ), "response")
        response_id = row["response_id"]
        if response_id in responses:
            raise ValueError(f"Duplicate response_id in response archive: {response_id}")
        expected = grid_by_id.get(response_id)
        if expected is None:
            raise ValueError(f"Response ID is outside the frozen evaluation grid: {response_id}")
        if row["response_schema_version"] != RESPONSE_SCHEMA_VERSION:
            raise ValueError("Wrong response schema version")
        if row["study_version"] != STUDY_VERSION or row["runner_version"] != FULL_RUNNER_VERSION:
            raise ValueError("Response is not from the prospective full runner")
        if row["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("Response protocol_id mismatch")
        if row["split"] != "evaluation" or expected["split"] != "evaluation":
            raise ValueError("Development or historical rows cannot enter checkpoint analysis")
        for field in ("response_identity", "prompt_id", "scenario_family_id", "condition"):
            expected_field = "logical_seed" if field == "seed" else field
            if row[field] != expected[expected_field]:
                raise ValueError(f"Response {field} differs from frozen grid for {response_id}")
        if row["seed"] != expected["logical_seed"]:
            raise ValueError(f"Response seed differs from frozen grid for {response_id}")
        checkpoint = protocol["checkpoints"][expected["checkpoint_alias"]]
        for field in ("model_id", "model_revision", "tokenizer_revision", "model_hash", "serialization_id"):
            if row[field] != checkpoint[field]:
                raise ValueError(f"Response {field} differs from frozen checkpoint for {response_id}")
        # Reuse the runner's own validation so the declared identity cannot
        # conceal a changed prompt, decoding configuration, serialization, or
        # execution dtype/device.
        full_runner.validate_response(row, expected, protocol["protocol_id"], row["execution_profile"])
        generation = row["generation"]
        _require_keys(generation, ("response", "output_tokens", "stopping"), f"generation {response_id}")
        if not isinstance(generation["response"], str):
            raise ValueError(f"Response text is not a string for {response_id}")
        if not isinstance(generation["output_tokens"], int) or generation["output_tokens"] < 0:
            raise ValueError(f"Invalid output token count for {response_id}")
        stopping = generation["stopping"]
        _require_keys(stopping, ("reason", "eos_reached", "cap_reached"), f"stopping {response_id}")
        if type(stopping["eos_reached"]) is not bool or type(stopping["cap_reached"]) is not bool:
            raise ValueError(f"Stopping flags must be Boolean for {response_id}")
        execution_ids.add(row["execution_profile_id"])
        execution_bodies.add(canonical_json(row["execution_profile"]))
        responses[response_id] = row

    expected_ids = set(grid_by_id)
    if set(responses) != expected_ids:
        raise ValueError(f"Response IDs do not exactly match protocol (missing={len(expected_ids-set(responses))})")
    if len(execution_ids) != 1 or len(execution_bodies) != 1:
        raise ValueError("Response archive mixes execution profiles")
    body = json.loads(next(iter(execution_bodies)))
    execution_core = {key: value for key, value in body.items() if key != "execution_profile_id"}
    recomputed_execution_id = stable_id("execution", execution_core)
    if (
        body.get("execution_profile_id") != next(iter(execution_ids))
        or body.get("execution_profile_id") != recomputed_execution_id
    ):
        raise ValueError("Execution profile body does not match execution_profile_id")
    return responses


def _validate_predictions(
    rows: list[dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    alias_by_response_id: dict[str, str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    expected_count = len(responses) * len(WINDOWS)
    if len(rows) != expected_count:
        raise ValueError(f"Prediction archive must contain exactly {expected_count} rows, found {len(rows)}")
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    prediction_ids: set[str] = set()
    dependencies: set[str] = set()
    run_ids: set[str] = set()
    variants: set[str] = set()
    detectors: set[str] = set()
    schemas: set[str] = set()
    for row in rows:
        _require_keys(row, (
            "prediction_id", "prediction_schema_version", "study_version", "response_id",
            "detector_version", "prediction_variant", "evaluation_unit", "decision_field",
            "dependency_identity", "predictions", "scenario_family_id", "split", "condition",
            "model_id", "seed", "checkpoint_alias", "model_size", "stage", "window_metadata",
            "protocol_id", "run_id", "execution_profile_id", "generation_output_tokens",
            "generation_stop_reason",
        ), "prediction")
        response_id = row["response_id"]
        unit = row["evaluation_unit"]
        key = (response_id, unit)
        if response_id not in responses:
            raise ValueError(f"Prediction response_id has no matching response: {response_id}")
        if unit not in WINDOWS:
            raise ValueError(f"Unexpected evaluation window: {unit}")
        if key in predictions:
            raise ValueError(f"Duplicate response/window prediction: {key}")
        if row["prediction_id"] in prediction_ids:
            raise ValueError(f"Duplicate prediction_id: {row['prediction_id']}")
        prediction_ids.add(row["prediction_id"])
        if row["study_version"] != STUDY_VERSION:
            raise ValueError("Wrong prediction study version")
        if row["protocol_id"] != responses[response_id]["protocol_id"]:
            raise ValueError(f"Prediction protocol_id disagrees with response {response_id}")
        if row["execution_profile_id"] != responses[response_id]["execution_profile_id"]:
            raise ValueError(f"Prediction execution_profile_id disagrees with response {response_id}")
        if row["generation_output_tokens"] != responses[response_id]["generation"]["output_tokens"]:
            raise ValueError(f"Prediction generation_output_tokens disagrees with response {response_id}")
        if row["generation_stop_reason"] != responses[response_id]["generation"]["stopping"]["reason"]:
            raise ValueError(f"Prediction generation_stop_reason disagrees with response {response_id}")
        if row["decision_field"] != "union":
            raise ValueError("Prospective archive must include the union detector")
        if "split" in row and row["split"] != "evaluation":
            raise ValueError("Prediction archive contains a non-evaluation row")
        for field in ("scenario_family_id", "condition", "model_id", "seed"):
            if row[field] != responses[response_id].get(field):
                raise ValueError(f"Prediction {field} disagrees with response {response_id}")
        expected_alias = alias_by_response_id[response_id]
        expected_size, expected_stage = _stage_from_alias(expected_alias)
        if row["checkpoint_alias"] != expected_alias or row["model_size"] != expected_size or row["stage"] != expected_stage:
            raise ValueError(f"Prediction checkpoint metadata is inconsistent for {response_id}")
        window_metadata = row["window_metadata"]
        if not isinstance(window_metadata, dict):
            raise ValueError(f"Prediction window_metadata must be an object for {response_id}/{unit}")
        _require_present_keys(window_metadata, (
            "tokenizer_model_id", "tokenizer_revision", "full_response_tokens", "window_token_limit",
            "evaluated_tokens", "shorter_than_window", "truncated",
        ), f"window_metadata {response_id}/{unit}")
        full_tokens = window_metadata["full_response_tokens"]
        if not isinstance(full_tokens, int) or full_tokens < 0:
            raise ValueError(f"Invalid full_response_tokens for {response_id}/{unit}")
        limit = {"full_response": None, "prefix_64_tokens": 64, "prefix_128_tokens": 128}[unit]
        expected_evaluated = full_tokens if limit is None else min(full_tokens, limit)
        expected_shorter = None if limit is None else full_tokens < limit
        expected_truncated = False if limit is None else full_tokens > limit
        if (
            window_metadata["window_token_limit"] != limit
            or window_metadata["evaluated_tokens"] != expected_evaluated
            or window_metadata["shorter_than_window"] != expected_shorter
            or window_metadata["truncated"] != expected_truncated
        ):
            raise ValueError(f"Inconsistent window_metadata for {response_id}/{unit}")
        category_values = row["predictions"]
        if not isinstance(category_values, dict) or set(category_values) != set(CATEGORIES):
            raise ValueError(f"Prediction categories differ from frozen four-category detector for {response_id}")
        for category in CATEGORIES:
            values = category_values[category]
            if not isinstance(values, dict):
                raise ValueError(f"Prediction component object missing for {response_id}/{category}")
            for component in COMPONENTS:
                if type(values.get(component)) is not bool:
                    raise ValueError(f"Prediction flag must be Boolean: {response_id}/{unit}/{category}/{component}")
            if values["union"] != (values["exact"] or values["embedding"]):
                raise ValueError(f"Union flag is inconsistent for {response_id}/{unit}/{category}")
        dependencies.add(canonical_json(row["dependency_identity"]))
        run_ids.add(row["run_id"])
        variants.add(row["prediction_variant"])
        detectors.add(row["detector_version"])
        schemas.add(row["prediction_schema_version"])
        predictions[key] = row

    for unit in WINDOWS:
        unit_ids = {response_id for response_id, candidate in predictions if candidate == unit}
        if unit_ids != set(responses):
            raise ValueError(f"Response/prediction IDs do not exactly match for {unit}")
    if schemas != {PREDICTION_SCHEMA_VERSION}:
        raise ValueError(f"Inconsistent or unsupported prediction schemas: {sorted(schemas)}")
    if variants != {EXPECTED_PREDICTION_VARIANT}:
        raise ValueError(f"Inconsistent or unsupported prediction variants: {sorted(variants)}")
    if detectors != {EXPECTED_DETECTOR_VERSION}:
        raise ValueError(f"Inconsistent or unsupported detector versions: {sorted(detectors)}")
    if len(dependencies) != 1:
        raise ValueError("Prediction archive mixes dependency identities")
    if len(run_ids) != 1:
        raise ValueError("Prediction archive mixes full-run run_ids")
    dependency = json.loads(next(iter(dependencies)))
    for key in ("embedding_model", "prefix_window_tokenizer"):
        item = dependency.get(key)
        if not isinstance(item, dict) or item.get("applied") is not True or not _is_pinned_revision(item.get("resolved_revision")):
            raise ValueError(f"Prediction dependency is not consistently applied and pinned: {key}")
    tokenizer_dependency = dependency["prefix_window_tokenizer"]
    full_tokens_by_response: dict[str, int] = {}
    for (response_id, _unit), row in predictions.items():
        metadata = row["window_metadata"]
        if (
            metadata["tokenizer_model_id"] != tokenizer_dependency.get("model_id")
            or metadata["tokenizer_revision"] != tokenizer_dependency.get("resolved_revision")
        ):
            raise ValueError(f"Window tokenizer metadata disagrees with pinned dependency for {response_id}")
        if response_id in full_tokens_by_response and full_tokens_by_response[response_id] != metadata["full_response_tokens"]:
            raise ValueError(f"Window token counts disagree across evaluation units for {response_id}")
        full_tokens_by_response[response_id] = metadata["full_response_tokens"]
    return predictions, dependency


def load_and_validate(protocol_path: Path, responses_path: Path, predictions_path: Path) -> ValidatedStudy:
    protocol = _load_protocol(protocol_path)
    grid, aliases, family_domains = _validate_protocol_grid(protocol)
    responses = _validate_responses(list(iter_jsonl(responses_path)), protocol, grid)
    predictions, dependency = _validate_predictions(list(iter_jsonl(predictions_path)), responses, aliases)
    return ValidatedStudy(protocol, responses, predictions, grid, aliases, family_domains, dependency)


def _stage_from_alias(alias: str) -> tuple[str, str]:
    size, suffix = alias.split("_", 1)
    return size, "final" if suffix == "instruct" else suffix


def _factor_contrasts(cell: dict[tuple[bool, str], float]) -> dict[str, float]:
    expected = {(emotion, request) for emotion in EMOTIONS for request in REQUESTS}
    if set(cell) != expected:
        raise ValueError("Missing four-way prompt crossing while computing a family contrast")
    advice = 0.5 * sum(cell[(emotion, "advice")] - cell[(emotion, "information")] for emotion in EMOTIONS)
    emotion = 0.5 * sum(cell[(True, request)] - cell[(False, request)] for request in REQUESTS)
    interaction = (
        cell[(True, "advice")] - cell[(True, "information")]
        - cell[(False, "advice")] + cell[(False, "information")]
    )
    return {
        "advice_minus_information": advice,
        "emotion_minus_neutral": emotion,
        "interaction": interaction,
    }


def family_stage_change_records(study: ValidatedStudy) -> list[dict[str, Any]]:
    seed_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for response_id, response in study.responses.items():
        family = response["scenario_family_id"]
        condition = response["condition"]
        alias = study.alias_by_response_id[response_id]
        for window in WINDOWS:
            prediction = study.predictions[(response_id, window)]["predictions"]
            for component in COMPONENTS:
                for category in CATEGORIES:
                    key = (
                        family, alias, condition["emotional_framing"], condition["request_type"],
                        window, component, category,
                    )
                    seed_values[key].append(float(prediction[category][component]))
    if any(len(values) != 3 for values in seed_values.values()):
        raise ValueError("Seed averaging requires exactly three values per family cell")

    checkpoint_contrasts: dict[tuple[Any, ...], float] = {}
    for family in sorted(study.family_domains):
        for alias in sorted(study.protocol["checkpoints"]):
            for window in WINDOWS:
                for component in COMPONENTS:
                    for category in CATEGORIES:
                        cell = {
                            (emotion, request): float(np.mean(seed_values[(
                                family, alias, emotion, request, window, component, category,
                            )]))
                            for emotion in EMOTIONS for request in REQUESTS
                        }
                        for contrast, value in _factor_contrasts(cell).items():
                            checkpoint_contrasts[(family, alias, window, component, category, contrast)] = value

    records = []
    for family in sorted(study.family_domains):
        for size in SIZES:
            aliases = {stage: f"{size}_{'instruct' if stage == 'final' else stage}" for stage in STAGES}
            pairs = {
                "dpo_minus_sft": (aliases["dpo"], aliases["sft"]),
                "final_minus_dpo": (aliases["final"], aliases["dpo"]),
            }
            for window in WINDOWS:
                for component in COMPONENTS:
                    for category in CATEGORIES:
                        for contrast in CONTRASTS:
                            for stage_change, (after, before) in pairs.items():
                                value = (
                                    checkpoint_contrasts[(family, after, window, component, category, contrast)]
                                    - checkpoint_contrasts[(family, before, window, component, category, contrast)]
                                )
                                records.append({
                                    "scenario_family_id": family,
                                    "domain": study.family_domains[family],
                                    "window": window,
                                    "component": component,
                                    "category": category,
                                    "model_size": size,
                                    "factor_contrast": contrast,
                                    "stage_change": stage_change,
                                    "value": value,
                                })
    return records


def primary_family_matrix(
    family_records: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, str]], np.ndarray]:
    selected = [row for row in family_records if row["window"] == "full_response" and row["component"] == "union"]
    families = sorted({row["scenario_family_id"] for row in selected})
    domains_by_family = {row["scenario_family_id"]: row["domain"] for row in selected}
    specs = [
        {
            "model_size": size,
            "category": category,
            "factor_contrast": contrast,
            "stage_change": stage_change,
        }
        for size in SIZES
        for category in CATEGORIES
        for contrast in CONTRASTS
        for stage_change in STAGE_CHANGES
    ]
    if len(specs) != PRIMARY_METRIC_COUNT:
        raise AssertionError("Primary metric definition must contain exactly 48 entries")
    lookup = {
        (row["scenario_family_id"], row["model_size"], row["category"], row["factor_contrast"], row["stage_change"]): row["value"]
        for row in selected
    }
    matrix = np.array([
        [lookup[(family, spec["model_size"], spec["category"], spec["factor_contrast"], spec["stage_change"])] for spec in specs]
        for family in families
    ], dtype=float)
    return families, [domains_by_family[family] for family in families], specs, matrix


def stratified_bootstrap_indices(domains: Sequence[str], replicates: int, seed: int) -> np.ndarray:
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    domains_array = np.asarray(domains)
    strata = [np.flatnonzero(domains_array == domain) for domain in DOMAINS]
    if [len(item) for item in strata] != [20, 20]:
        raise ValueError("Bootstrap requires exactly 20 health and 20 everyday families")
    rng = np.random.default_rng(seed)
    sampled = [rng.choice(indices, size=(replicates, len(indices)), replace=True) for indices in strata]
    return np.concatenate(sampled, axis=1)


def bootstrap_primary(
    values: np.ndarray,
    domains: Sequence[str],
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if values.shape != (40, PRIMARY_METRIC_COUNT):
        raise ValueError(f"Primary family matrix must have shape (40, 48), found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Primary family matrix contains non-finite values")
    indices = stratified_bootstrap_indices(domains, replicates, seed)
    estimates = values[indices].mean(axis=1)
    point = values.mean(axis=0)
    lower, upper = np.quantile(estimates, (0.025, 0.975), axis=0)
    standard_error = estimates.std(axis=0, ddof=1)
    # A stratified bootstrap has no scenario uncertainty when a metric is
    # constant within both domains. Floating summation can nevertheless create
    # a tiny positive SD, so classify variation from the original family
    # values with a scale-aware roundoff threshold (64 machine epsilons).
    scale = np.maximum(1.0, np.max(np.abs(values), axis=0))
    tolerance = ROUND_OFF_MULTIPLIER * np.finfo(float).eps * scale
    domains_array = np.asarray(domains)
    within_stratum_variation = np.zeros(PRIMARY_METRIC_COUNT, dtype=bool)
    for domain in DOMAINS:
        stratum = values[domains_array == domain]
        within_stratum_variation |= np.ptp(stratum, axis=0) > tolerance
    estimable = (
        np.isfinite(standard_error)
        & (standard_error > tolerance)
        & within_stratum_variation
    )
    critical: float | None = None
    simultaneous_lower = np.full(PRIMARY_METRIC_COUNT, np.nan)
    simultaneous_upper = np.full(PRIMARY_METRIC_COUNT, np.nan)
    if estimable.any():
        standardized = np.abs((estimates[:, estimable] - point[estimable]) / standard_error[estimable])
        maxima = standardized.max(axis=1)
        if np.isfinite(maxima).all():
            critical = float(np.quantile(maxima, 0.95))
            simultaneous_lower[estimable] = point[estimable] - critical * standard_error[estimable]
            simultaneous_upper[estimable] = point[estimable] + critical * standard_error[estimable]
        else:
            estimable[:] = False
    return {
        "point": point,
        "pointwise_lower": lower,
        "pointwise_upper": upper,
        "bootstrap_standard_error": standard_error,
        "roundoff_tolerance": tolerance,
        "within_stratum_variation": within_stratum_variation,
        "estimable": estimable,
        "simultaneous_lower": simultaneous_lower,
        "simultaneous_upper": simultaneous_upper,
        "simultaneous_critical_value": critical,
        "estimable_count": int(estimable.sum()),
        "replicates": replicates,
        "seed": seed,
        "indices": indices,
        "replicate_estimates": estimates,
    }


def _metric_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return row["model_size"], row["category"], row["factor_contrast"], row["stage_change"]


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def aggregate_sensitivities(family_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in family_records:
        key = (row["window"], row["component"], *_metric_key(row))
        grouped[key].append(row["value"])
    primary = {
        key[2:]: float(np.mean(values))
        for key, values in grouped.items()
        if key[0] == "full_response" and key[1] == "union"
    }
    records = []
    for key in sorted(grouped):
        window, component, size, category, contrast, stage_change = key
        estimate = float(np.mean(grouped[key]))
        reference = primary[(size, category, contrast, stage_change)]
        if window == "full_response" and component == "union":
            continue
        records.append({
            "window": window,
            "component": component,
            "model_size": size,
            "category": category,
            "factor_contrast": contrast,
            "stage_change": stage_change,
            "estimate_percentage_points": estimate * 100.0,
            "primary_reference_percentage_points": reference * 100.0,
            "direction": _sign(estimate),
            "primary_direction": _sign(reference),
            "direction_change": _sign(estimate) != _sign(reference),
            "interpretation": "descriptive sensitivity; no p-value-based selection",
            "independent_scenario_families": 40,
        })
    return records


def condition_rate_records(study: ValidatedStudy) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    families: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for response_id, response in study.responses.items():
        size, stage = _stage_from_alias(study.alias_by_response_id[response_id])
        condition = response["condition"]
        base = (size, stage, condition["domain"], condition["emotional_framing"], condition["request_type"])
        for window in WINDOWS:
            prediction = study.predictions[(response_id, window)]["predictions"]
            for component in COMPONENTS:
                for category in CATEGORIES:
                    key = (*base, window, component, category)
                    grouped[key].append(float(prediction[category][component]))
                    families[key].add(response["scenario_family_id"])
    records = []
    for key in sorted(grouped, key=str):
        size, stage, domain, emotion, request, window, component, category = key
        records.append({
            "model_size": size,
            "stage": stage,
            "domain": domain,
            "emotional_framing": emotion,
            "request_type": request,
            "window": window,
            "component": component,
            "category": category,
            "rate": float(np.mean(grouped[key])),
            "response_rows": len(grouped[key]),
            "independent_scenario_families": len(families[key]),
        })
    return records


def response_summary_records(study: ValidatedStudy) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for response_id, response in study.responses.items():
        size, stage = _stage_from_alias(study.alias_by_response_id[response_id])
        condition = response["condition"]
        key = (size, stage, condition["domain"], condition["emotional_framing"], condition["request_type"])
        grouped[key].append(response)
    records = []
    for key in sorted(grouped, key=str):
        rows = grouped[key]
        tokens = np.array([row["generation"]["output_tokens"] for row in rows], dtype=float)
        window_tokens = np.array([
            study.predictions[(row["response_id"], "full_response")]["window_metadata"]["full_response_tokens"]
            for row in rows
        ], dtype=float)
        shorter_64 = [
            study.predictions[(row["response_id"], "prefix_64_tokens")]["window_metadata"]["shorter_than_window"]
            for row in rows
        ]
        shorter_128 = [
            study.predictions[(row["response_id"], "prefix_128_tokens")]["window_metadata"]["shorter_than_window"]
            for row in rows
        ]
        reasons = Counter(row["generation"]["stopping"]["reason"] for row in rows)
        size, stage, domain, emotion, request = key
        records.append({
            "model_size": size,
            "stage": stage,
            "domain": domain,
            "emotional_framing": emotion,
            "request_type": request,
            "response_rows": len(rows),
            "independent_scenario_families": len({row["scenario_family_id"] for row in rows}),
            "mean_output_tokens": float(tokens.mean()),
            "median_output_tokens": float(np.median(tokens)),
            "mean_detector_window_tokens": float(window_tokens.mean()),
            "median_detector_window_tokens": float(np.median(window_tokens)),
            "cap_rate": float(np.mean([row["generation"]["stopping"]["cap_reached"] for row in rows])),
            "eos_rate": float(np.mean([row["generation"]["stopping"]["eos_reached"] for row in rows])),
            "shorter_than_64_rate": float(np.mean(shorter_64)),
            "shorter_than_128_rate": float(np.mean(shorter_128)),
            "stopping_reason_counts": dict(sorted(reasons.items())),
        })
    return records


def analyze(study: ValidatedStudy, replicates: int = DEFAULT_REPLICATES, seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
    family_records = family_stage_change_records(study)
    families, domains, specs, values = primary_family_matrix(family_records)
    boot = bootstrap_primary(values, domains, replicates, seed)
    primary = []
    for index, spec in enumerate(specs):
        included = bool(boot["estimable"][index])
        primary.append({
            **spec,
            "estimate_percentage_points": float(boot["point"][index] * 100),
            "pointwise_95_lower_percentage_points": float(boot["pointwise_lower"][index] * 100),
            "pointwise_95_upper_percentage_points": float(boot["pointwise_upper"][index] * 100),
            "bootstrap_standard_error_percentage_points": float(boot["bootstrap_standard_error"][index] * 100),
            "simultaneous_95_lower_percentage_points": float(boot["simultaneous_lower"][index] * 100) if included else None,
            "simultaneous_95_upper_percentage_points": float(boot["simultaneous_upper"][index] * 100) if included else None,
            "simultaneous_included": included,
            "uncertainty_status": "estimable" if included else "degenerate_uninformative",
            "independent_scenario_families": 40,
            "seed_replicates_per_cell": 3,
        })
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete_prospective_analysis",
        "protocol_id": study.protocol["protocol_id"],
        "input_summary": {
            "responses": len(study.responses),
            "predictions": len(study.predictions),
            "scenario_families": len(families),
            "domain_family_counts": dict(Counter(domains)),
            "independent_n": 40,
            "seed_replicates_per_cell": 3,
            "dependency_identity": study.dependency_identity,
        },
        "bootstrap": {
            "method": "stratified scenario-family bootstrap with shared indices across all primary metrics",
            "replicates": replicates,
            "seed": seed,
            "pointwise_interval": "unclipped 2.5th and 97.5th percentiles",
            "simultaneous_method": "95th percentile of replicate maximum absolute standardized deviation",
            "simultaneous_critical_value": boot["simultaneous_critical_value"],
            "simultaneous_scope": {
                "estimable_metrics": boot["estimable_count"],
                "predefined_metrics": PRIMARY_METRIC_COUNT,
                "label": f"simultaneous max-t-style band over estimable subset {boot['estimable_count']}/{PRIMARY_METRIC_COUNT}",
            },
        },
        "primary_metrics": primary,
        "sensitivity_metrics": aggregate_sensitivities(family_records),
        "condition_rates": condition_rate_records(study),
        "response_summaries": response_summary_records(study),
        "interpretation_limits": [
            "Intervals approximate scenario-family variation within the balanced chosen health/everyday frame.",
            "They do not guarantee coverage for new topics or identify randomized causal training-stage effects.",
            "Prefix and detector-component comparisons are descriptive sensitivities.",
            "Detector flags are not calibrated semantic judgments in the absence of real human validation labels.",
        ],
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _flatten_csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for section, entries in (
        ("primary_metric", result["primary_metrics"]),
        ("sensitivity_metric", result["sensitivity_metrics"]),
        ("condition_rate", result["condition_rates"]),
        ("response_summary", result["response_summaries"]),
    ):
        for entry in entries:
            flat = {"record_type": section}
            for key, value in entry.items():
                flat[key] = canonical_json(value) if isinstance(value, (dict, list)) else value
            rows.append(flat)
    return rows


def _markdown_report(result: dict[str, Any]) -> str:
    scope = result["bootstrap"]["simultaneous_scope"]
    lines = [
        "# Prospective checkpoint analysis",
        "",
        f"Protocol: `{result['protocol_id']}`. The complete archive contains 2,880 responses from 40 independent scenario families (20 health and 20 everyday); three seeds per cell were averaged before contrasts.",
        "",
        f"Estimates are percentage-point changes in factorial contrasts across adjacent checkpoints. Intervals use a {result['bootstrap']['replicates']:,}-replicate stratified scenario-family bootstrap. They approximate variation within the balanced chosen scenario frame and do not establish randomized causal stage effects or guaranteed coverage for new topics.",
        "",
        f"The simultaneous interval is labeled **{scope['label']}**. Zero-variance metrics are explicitly `degenerate_uninformative` and have no simultaneous limits.",
        "",
        "## Primary full-response union estimates",
        "",
        "| Size | Marker | Factor contrast | Stage change | Estimate | Pointwise 95% interval | Simultaneous 95% interval | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in result["primary_metrics"]:
        pointwise = f"[{row['pointwise_95_lower_percentage_points']:.2f}, {row['pointwise_95_upper_percentage_points']:.2f}]"
        simultaneous = (
            f"[{row['simultaneous_95_lower_percentage_points']:.2f}, {row['simultaneous_95_upper_percentage_points']:.2f}]"
            if row["simultaneous_included"] else "unavailable"
        )
        lines.append(
            f"| {row['model_size']} | {row['category']} | {row['factor_contrast']} | {row['stage_change']} | "
            f"{row['estimate_percentage_points']:.2f} | {pointwise} | {simultaneous} | {row['uncertainty_status']} |"
        )
    lines += [
        "",
        "## Sensitivities and descriptive summaries",
        "",
        "The CSV and JSON contain exact, embedding, and union contrasts for full, 64-token, and 128-token windows, with direction-change flags relative to the matching primary metric. They also contain condition rates, output-token summaries, stop and cap rates, and shorter-than-prefix opportunities. These sensitivities are descriptive and are not used to select a winning detector or window by p-value.",
        "",
        "Human-validation results are outside this detector checkpoint report unless a separate real annotation input is supplied and analyzed under its own protocol.",
        "",
    ]
    return "\n".join(lines)


def _write_plots(result: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = result["primary_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharex=True)
    for axis, size in zip(axes, SIZES):
        rows = [row for row in primary if row["model_size"] == size]
        y = np.arange(len(rows))
        x = np.array([row["estimate_percentage_points"] for row in rows])
        lo = np.array([row["pointwise_95_lower_percentage_points"] for row in rows])
        hi = np.array([row["pointwise_95_upper_percentage_points"] for row in rows])
        axis.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", markersize=3, capsize=2)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(f"{size.upper()} full-response union")
        axis.set_yticks(y)
        axis.set_yticklabels([f"{r['category']} | {r['factor_contrast']} | {r['stage_change']}" for r in rows], fontsize=6)
        axis.set_xlabel("Percentage-point stage change (pointwise 95% interval)")
    fig.tight_layout()
    contrast_path = output_dir / "primary_checkpoint_contrasts.png"
    fig.savefig(contrast_path, dpi=180)
    plt.close(fig)

    summaries = result["response_summaries"]
    labels = [f"{size.upper()} {stage}" for size in SIZES for stage in STAGES]
    means = []
    for size in SIZES:
        for stage in STAGES:
            values = [row["mean_output_tokens"] for row in summaries if row["model_size"] == size and row["stage"] == stage]
            means.append(float(np.mean(values)))
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, means)
    axis.set_ylabel("Mean generated tokens (balanced condition cells)")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    length_path = output_dir / "response_lengths.png"
    fig.savefig(length_path, dpi=180)
    plt.close(fig)
    return [str(contrast_path), str(length_path)]


def write_outputs(
    result: dict[str, Any], output_dir: Path, *, confirmed_real_complete_archive: bool
) -> dict[str, Any]:
    if not confirmed_real_complete_archive:
        raise ValueError("Refusing to write empirical outputs without explicit real complete-archive confirmation")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "checkpoint_analysis.json"
    csv_path = output_dir / "checkpoint_analysis.csv"
    markdown_path = output_dir / "checkpoint_analysis.md"
    _atomic_write_text(json_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    flat_rows = _flatten_csv_rows(result)
    fieldnames = sorted({key for row in flat_rows for key in row})
    temp_csv = csv_path.with_suffix(".csv.tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)
    os.replace(temp_csv, csv_path)
    _atomic_write_text(markdown_path, _markdown_report(result))
    plots = _write_plots(result, output_dir)
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "protocol_id": result["protocol_id"],
        "confirmed_real_complete_archive": True,
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (json_path, csv_path, markdown_path, *(Path(item) for item in plots))
        ],
    }
    manifest_path = output_dir / "checkpoint_analysis_manifest.json"
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"output_dir": str(output_dir), "manifest": str(manifest_path), "artifacts": manifest["artifacts"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--confirm-real-complete-archive", action="store_true",
        help="Required assertion that these are real outputs from the completed frozen evaluation run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.confirm_real_complete_archive:
        raise SystemExit("--confirm-real-complete-archive is required; no empirical artifacts were written")
    if args.bootstrap_replicates != DEFAULT_REPLICATES:
        raise SystemExit(f"The frozen real-data analysis requires exactly {DEFAULT_REPLICATES} bootstrap replicates")
    study = load_and_validate(args.protocol, args.responses, args.predictions)
    result = analyze(study, args.bootstrap_replicates, args.bootstrap_seed)
    result["input_provenance"] = {
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "responses_path": str(args.responses),
        "responses_sha256": sha256_file(args.responses),
        "predictions_path": str(args.predictions),
        "predictions_sha256": sha256_file(args.predictions),
    }
    written = write_outputs(result, args.output_dir, confirmed_real_complete_archive=True)
    print(json.dumps(written, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

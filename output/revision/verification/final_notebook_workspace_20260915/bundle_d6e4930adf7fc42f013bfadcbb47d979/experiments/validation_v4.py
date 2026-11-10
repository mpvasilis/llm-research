"""Versioned full-response human-validation workflow for the revision study.

This module keeps human targets distinct from every historical detector's category
names.  It creates blinded items, a separate provenance/prediction key, and two
independent blank annotation templates.  Population estimates are produced only
from the probability sample.  Detector-positive enrichment is reported as a
separate diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from config import ROOT


VERSION = 4
DEFAULT_OUT = ROOT / "output" / "revision" / "annotation_v4"
HUMAN_TARGETS = (
    "empathy_expression",
    "validation",
    "professional_referral",
    "professional_disclaimer",
    "professional_mention",
    "structuring",
)
ORIGINAL_CATEGORIES = (
    "empathy_opener",
    "validation",
    "disclaimer",
    "crisis_referral",
    "structure",
)
EVALUATION_UNITS = {"full_response", "prefix_64_tokens", "prefix_128_tokens"}
SPLITS = {"development", "evaluation", "historical_diagnostic"}
LOCATION_VALUES = {"none", "opening", "later", "both"}

ITEM_FIELDS = ["item_id", "prompt", "response"]
KEY_FIELDS = [
    "item_id",
    "response_id",
    "study_version",
    "scenario_family_id",
    "split",
    "condition_json",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "model_hash",
    "seed",
    "serialization_id",
    "detector_version",
    "evaluation_unit",
    "prediction_json",
    "sampling_stratum",
    "sample_role",
    "random_inclusion_probability",
    "enrichment_inclusion_probabilities_json",
    "provenance_json",
]
ANNOTATION_FIELDS = [
    "item_id",
    *HUMAN_TARGETS,
    "empathy_expression_location",
    "evidence_notes",
    "completed_at",
]
ADJUDICATION_FIELDS = [
    "item_id",
    *HUMAN_TARGETS,
    "empathy_expression_location",
    "adjudication_reason",
    "adjudicated_at",
]

COMPARISON_MAP = (
    {
        "original_category": "empathy_opener",
        "human_target": "empathy_expression",
        "alignment": "partial: historical prediction is response-wide, so its name does not support an opener claim",
    },
    {
        "original_category": "validation",
        "human_target": "validation",
        "alignment": "aligned",
    },
    {
        "original_category": "disclaimer",
        "human_target": "professional_any",
        "alignment": "combined comparison declared explicitly; component human targets remain separate",
    },
    {
        "original_category": "crisis_referral",
        "human_target": None,
        "alignment": "not scored: v4 has no crisis-referral human target",
    },
    {
        "original_category": "structure",
        "human_target": "structuring",
        "alignment": "aligned",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _required_text(row: dict[str, object], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires nonblank {key}")
    return value.strip()


def _normalized_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def _construction_prompts(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Construction-prompt file must contain an object: {path}")
    prompts = [prompt for values in data.values() for prompt in values]
    if not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"Construction prompts must all be strings: {path}")
    return {_normalized_prompt(prompt) for prompt in prompts}


def _validate_responses(
    rows: list[dict[str, object]], construction_prompts_path: Path | None
) -> dict[str, dict[str, object]]:
    by_id: dict[str, dict[str, object]] = {}
    family_splits: dict[str, set[str]] = defaultdict(set)
    known_construction = _construction_prompts(construction_prompts_path)
    for index, row in enumerate(rows, 1):
        context = f"response record {index}"
        response_id = _required_text(row, "response_id", context)
        if response_id in by_id:
            raise ValueError(f"Duplicate response_id: {response_id}")
        _required_text(row, "study_version", context)
        family = _required_text(row, "scenario_family_id", context)
        split = _required_text(row, "split", context)
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r} for {response_id}")
        condition = row.get("condition")
        if not isinstance(condition, (str, dict)) or not condition:
            raise ValueError(f"{context} requires a nonempty condition string/object")
        for field in (
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "serialization_id",
        ):
            _required_text(row, field, context)
        if not isinstance(row.get("seed"), int):
            raise ValueError(f"{context} requires integer seed")
        generation = row.get("generation")
        if not isinstance(generation, dict):
            raise ValueError(f"{context} requires generation object")
        prompt = _required_text(generation, "prompt", context + " generation")
        _required_text(generation, "response", context + " generation")
        if split == "evaluation" and _normalized_prompt(prompt) in known_construction:
            raise ValueError(
                f"Prospective evaluation includes a known construction prompt: {response_id}"
            )
        family_splits[family].add(split)
        by_id[response_id] = row
    leaking = sorted(
        family
        for family, splits in family_splits.items()
        if "development" in splits and "evaluation" in splits
    )
    if leaking:
        raise ValueError(
            "Scenario families cross development/evaluation: " + ", ".join(leaking)
        )
    return by_id


def _prediction_decisions(row: dict[str, object]) -> dict[str, bool | None]:
    predictions = row.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError("Prediction record requires predictions object")
    unknown = set(predictions) - set(ORIGINAL_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown original detector categories: {sorted(unknown)}")
    decision_field = row.get("decision_field", "union")
    if not isinstance(decision_field, str) or not decision_field:
        raise ValueError("decision_field must be a nonblank string")
    decisions: dict[str, bool | None] = {}
    for category in ORIGINAL_CATEGORIES:
        if category not in predictions:
            decisions[category] = None
            continue
        value = predictions[category]
        if value is None:
            decisions[category] = None
        elif isinstance(value, bool):
            decisions[category] = value
        elif isinstance(value, dict):
            if value.get("available") is False or value.get("status") in {
                "unavailable",
                "not_applicable",
            }:
                decisions[category] = None
            elif isinstance(value.get(decision_field), bool):
                decisions[category] = bool(value[decision_field])
            elif isinstance(value.get("predicted"), bool):
                decisions[category] = bool(value["predicted"])
            else:
                raise ValueError(
                    f"Prediction for {category} lacks boolean {decision_field!r}/predicted"
                )
        else:
            raise ValueError(f"Invalid prediction value for {category}: {value!r}")
    return decisions


def _validate_predictions(
    rows: list[dict[str, object]],
    response_ids: set[str],
    required_response_ids: set[str] | None = None,
    allow_legacy_prediction_provenance: bool = False,
) -> tuple[dict[str, dict[str, object]], dict[str, int], dict[str, object]]:
    full: dict[str, dict[str, object]] = {}
    unit_counts: dict[str, int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows, 1):
        context = f"prediction record {index}"
        response_id = _required_text(row, "response_id", context)
        detector_version = _required_text(row, "detector_version", context)
        unit = _required_text(row, "evaluation_unit", context)
        if unit not in EVALUATION_UNITS:
            raise ValueError(f"Invalid evaluation_unit {unit!r} for {response_id}")
        identity = (response_id, detector_version, unit)
        if identity in seen:
            raise ValueError(f"Duplicate prediction identity: {identity}")
        seen.add(identity)
        if response_id not in response_ids:
            raise ValueError(f"Prediction has no response: {response_id}")
        _prediction_decisions(row)
        unit_counts[unit] += 1
        if unit == "full_response":
            if response_id in full:
                raise ValueError(
                    f"Multiple full-response detector versions for {response_id}; prepare one at a time"
                )
            full[response_id] = row
    required = response_ids if required_response_ids is None else required_response_ids
    missing = sorted(required - set(full))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing full-response predictions for {len(missing)} responses: {preview}")
    chosen = [full[response_id] for response_id in sorted(required)]
    detector_versions = {_required_text(row, "detector_version", "prediction") for row in chosen}
    if len(detector_versions) != 1:
        raise ValueError(f"Mixed detector_version values in chosen predictions: {sorted(detector_versions)}")

    provenance_fields = {
        "prediction_id",
        "prediction_variant",
        "dependency_identity",
        "prediction_schema_version",
    }
    present_counts = {
        field: sum(field in row and row[field] not in (None, "") for row in chosen)
        for field in provenance_fields
    }
    if any(present_counts.values()):
        incomplete = {
            field: count for field, count in present_counts.items() if count != len(chosen)
        }
        if incomplete:
            raise ValueError(
                "Prediction provenance fields must be all present or all absent; "
                f"incomplete counts={incomplete}"
            )
        prediction_ids = [str(row["prediction_id"]) for row in chosen]
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("Duplicate prediction_id in chosen predictions")
        variants = {_required_text(row, "prediction_variant", "prediction") for row in chosen}
        schema_versions = {
            _required_text(row, "prediction_schema_version", "prediction") for row in chosen
        }
        dependency_identities = {
            canonical_json(row["dependency_identity"]) for row in chosen
        }
        if len(variants) != 1:
            raise ValueError(f"Mixed prediction_variant values: {sorted(variants)}")
        if len(schema_versions) != 1:
            raise ValueError(
                f"Mixed prediction_schema_version values: {sorted(schema_versions)}"
            )
        if len(dependency_identities) != 1:
            raise ValueError("Mixed dependency_identity values in chosen predictions")
        prediction_provenance: dict[str, object] = {
            "mode": "strict_versioned",
            "prediction_variant": next(iter(variants)),
            "prediction_schema_version": next(iter(schema_versions)),
            "dependency_identity": json.loads(next(iter(dependency_identities))),
            "prediction_ids_unique": True,
        }
    else:
        if not allow_legacy_prediction_provenance:
            raise ValueError(
                "Missing required versioned prediction provenance: prediction_id, "
                "prediction_variant, prediction_schema_version, dependency_identity"
            )
        prediction_provenance = {
            "mode": "legacy_test_only_missing_provenance_fields",
            "warning": (
                "Accepted only for legacy/synthetic compatibility; assignable packets require "
                "prediction_id, prediction_variant, prediction_schema_version, and dependency_identity."
            ),
        }
    return full, dict(sorted(unit_counts.items())), prediction_provenance


def _item_id(response_id: str, prompt: str, response: str) -> str:
    digest = hashlib.sha256(
        f"v4\n{response_id}\n{prompt}\n{response}".encode("utf-8")
    ).hexdigest()[:20]
    return f"v4_r_{digest}"


def _stratum(row: dict[str, object]) -> str:
    return canonical_json(
        {
            "condition": row["condition"],
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
        }
    )


def _bundle_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    return (
        out_dir / "validation_items_v4.csv",
        out_dir / "validation_key_v4.csv",
        out_dir / "validation_manifest_v4.json",
    )


def prepare(
    responses_path: Path,
    predictions_path: Path,
    split: str,
    out_dir: Path = DEFAULT_OUT,
    random_per_stratum: int = 8,
    positive_per_category: int = 0,
    seed: int = 0,
    construction_prompts_path: Path | None = ROOT / "experiments" / "prompts.json",
    overwrite: bool = False,
    allow_legacy_prediction_provenance: bool = False,
) -> dict[str, object]:
    """Prepare one split with probability and optional enriched components."""

    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    if random_per_stratum <= 0 or positive_per_category < 0:
        raise ValueError("random_per_stratum must be positive; enrichment cannot be negative")
    responses = _validate_responses(read_jsonl(responses_path), construction_prompts_path)
    pool = [row for row in responses.values() if row["split"] == split]
    if not pool:
        raise ValueError(f"No response records for split={split}")
    pool_ids = {str(row["response_id"]) for row in pool}
    predictions, unit_counts, prediction_provenance = _validate_predictions(
        read_jsonl(predictions_path),
        set(responses),
        pool_ids,
        allow_legacy_prediction_provenance,
    )

    strata: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pool:
        strata[_stratum(row)].append(row)
    rng = random.Random(seed)
    selected_ids: set[str] = set()
    random_probability: dict[str, float] = {}
    roles: dict[str, set[str]] = defaultdict(set)
    enrichment_probabilities: dict[str, dict[str, float]] = defaultdict(dict)
    stratum_population: dict[str, int] = {}
    stratum_selected: dict[str, int] = {}
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum], key=lambda row: str(row["response_id"]))
        count = min(random_per_stratum, len(candidates))
        chosen = rng.sample(candidates, count)
        probability = count / len(candidates)
        stratum_population[stratum] = len(candidates)
        stratum_selected[stratum] = count
        for row in chosen:
            response_id = str(row["response_id"])
            selected_ids.add(response_id)
            random_probability[response_id] = probability
            roles[response_id].add("probability_sample")

    if positive_per_category:
        for category in ORIGINAL_CATEGORIES:
            candidates = [
                row
                for row in pool
                if _prediction_decisions(predictions[str(row["response_id"])]).get(category)
                is True
            ]
            candidates.sort(key=lambda row: str(row["response_id"]))
            count = min(positive_per_category, len(candidates))
            if not count:
                continue
            probability = count / len(candidates)
            for row in rng.sample(candidates, count):
                response_id = str(row["response_id"])
                selected_ids.add(response_id)
                roles[response_id].add(f"positive_enriched:{category}")
                enrichment_probabilities[response_id][category] = probability

    ordered = sorted(
        (responses[response_id] for response_id in selected_ids),
        key=lambda row: str(row["response_id"]),
    )
    rng.shuffle(ordered)
    items: list[dict[str, object]] = []
    keys: list[dict[str, object]] = []
    for row in ordered:
        response_id = str(row["response_id"])
        generation = row["generation"]
        assert isinstance(generation, dict)
        prompt = str(generation["prompt"])
        response = str(generation["response"])
        item_id = _item_id(response_id, prompt, response)
        prediction = predictions[response_id]
        decisions = _prediction_decisions(prediction)
        items.append({"item_id": item_id, "prompt": prompt, "response": response})
        keys.append(
            {
                "item_id": item_id,
                "response_id": response_id,
                "study_version": row["study_version"],
                "scenario_family_id": row["scenario_family_id"],
                "split": row["split"],
                "condition_json": canonical_json(row["condition"]),
                "model_id": row["model_id"],
                "model_revision": row["model_revision"],
                "tokenizer_revision": row["tokenizer_revision"],
                "model_hash": row.get("model_hash", ""),
                "seed": row["seed"],
                "serialization_id": row["serialization_id"],
                "detector_version": prediction["detector_version"],
                "evaluation_unit": "full_response",
                "prediction_json": canonical_json(decisions),
                "sampling_stratum": _stratum(row),
                "sample_role": ";".join(sorted(roles[response_id])),
                "random_inclusion_probability": (
                    f"{random_probability[response_id]:.17g}"
                    if response_id in random_probability
                    else ""
                ),
                "enrichment_inclusion_probabilities_json": canonical_json(
                    enrichment_probabilities.get(response_id, {})
                ),
                "provenance_json": canonical_json(row),
            }
        )

    item_ids = [row["item_id"] for row in items]
    key_ids = [row["item_id"] for row in keys]
    if len(set(item_ids)) != len(item_ids) or set(item_ids) != set(key_ids):
        raise ValueError("Item/key IDs must be unique and identical")
    items_path, key_path, manifest_path = _bundle_paths(out_dir)
    existing_annotations = list(out_dir.glob("human_annotations_*_v4.csv"))
    if existing_annotations:
        raise ValueError("Refusing to replace a bundle with existing human annotations")
    if not overwrite and any(path.exists() for path in (items_path, key_path, manifest_path)):
        raise ValueError("v4 bundle already exists; use --overwrite only before annotation")
    write_csv(items_path, items, ITEM_FIELDS)
    write_csv(key_path, keys, KEY_FIELDS)
    blank_rows = []
    for item in items:
        blank = {field: "" for field in ANNOTATION_FIELDS}
        blank["item_id"] = item["item_id"]
        blank_rows.append(blank)
    blank_template_hashes: dict[str, str] = {}
    for annotator in (1, 2):
        blank_path = out_dir / f"blank_human_annotations_{annotator}_v4.csv"
        write_csv(blank_path, blank_rows, ANNOTATION_FIELDS)
        blank_template_hashes[blank_path.name] = file_sha256(blank_path)
    detector_versions = sorted({str(row["detector_version"]) for row in predictions.values()})
    manifest: dict[str, object] = {
        "version": VERSION,
        "status": "awaiting_two_independent_human_annotations",
        "created_at": utc_now(),
        "split": split,
        "scientific_scope": (
            "historical diagnostic only; not a fresh test set or stagewise validation"
            if split == "historical_diagnostic"
            else "prospective split declared in source records"
        ),
        "human_targets": list(HUMAN_TARGETS),
        "empathy_location_definition": (
            "opening means the first sentence, ending at the first . ! or ? boundary; "
            "the historical response-wide empathy_opener detector does not establish an opener claim"
        ),
        "comparison_map": list(COMPARISON_MAP),
        "responses_source": portable_path(responses_path),
        "responses_sha256": file_sha256(responses_path),
        "predictions_source": portable_path(predictions_path),
        "predictions_sha256": file_sha256(predictions_path),
        "detector_versions": detector_versions,
        "prediction_provenance": prediction_provenance,
        "prediction_unit_used": "full_response",
        "prediction_unit_counts_in_source": unit_counts,
        "prefix_rule": "full-response human labels never validate token-prefix predictions",
        "n_pool": len(pool),
        "n_items": len(items),
        "n_probability_sample": sum(
            "probability_sample" in roles[response_id] for response_id in selected_ids
        ),
        "n_enriched_union": sum(
            any(role.startswith("positive_enriched:") for role in roles[response_id])
            for response_id in selected_ids
        ),
        "n_known_construction_prompt_items": sum(
            bool(row.get("known_construction_prompt")) for row in ordered
        ),
        "random_sampling": {
            "seed": seed,
            "without_replacement": True,
            "stratum_fields": ["condition", "model_id", "model_revision"],
            "requested_per_stratum": random_per_stratum,
            "population_by_stratum": stratum_population,
            "selected_by_stratum": stratum_selected,
            "inclusion_probabilities_recorded_in_key": True,
        },
        "enrichment": {
            "requested_per_original_category": positive_per_category,
            "role": "separate detector-positive diagnostic",
            "population_estimation": "forbidden unless a future analysis supplies justified design weights",
        },
        "uncertainty_unit": "scenario_family_id cluster",
        "items_sha256": file_sha256(items_path),
        "key_sha256": file_sha256(key_path),
        "blank_annotation_template_sha256": blank_template_hashes,
        "human_provenance": "not yet collected; AI labels are prohibited",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def prepare_historical_frame(
    source: Path = ROOT / "out" / "raw_cache" / "final_7b_answers.jsonl",
    out_dir: Path = DEFAULT_OUT,
    construction_prompts_path: Path = ROOT / "experiments" / "prompts.json",
) -> dict[str, object]:
    """Normalize the recovered final-7B archive without claiming fresh evaluation."""

    raw = read_jsonl(source)
    construction = _construction_prompts(construction_prompts_path)
    records: list[dict[str, object]] = []
    for index, row in enumerate(raw, 1):
        for key in ("key", "tag", "question", "answer", "model_id"):
            _required_text(row, key, f"historical row {index}")
        prompt = str(row["question"])
        source_key = str(row["key"])
        response_id = "hist4_" + hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:20]
        scenario_index = row.get("i")
        if not isinstance(scenario_index, int):
            raise ValueError(f"historical row {index} requires integer i")
        records.append(
            {
                "study_version": "historical-final7b-recovered-v1",
                "response_id": response_id,
                "scenario_family_id": f"historical::{row['tag']}::{scenario_index:03d}",
                "split": "historical_diagnostic",
                "condition": str(row["tag"]),
                "model_id": str(row["model_id"]),
                "model_revision": "unrecorded_historical",
                "tokenizer_revision": "unrecorded_historical",
                "model_hash": "unrecorded_historical",
                "seed": int(row.get("seed", 0)),
                "serialization_id": (
                    "historical_chat_serialization_unrecorded"
                    if row.get("is_chat")
                    else "historical_plain_serialization_unrecorded"
                ),
                "generation": {
                    "prompt": prompt,
                    "response": str(row["answer"]),
                    "serialized_prompt": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "max_new_tokens": None,
                    "stopping": {
                        "reason": "unrecorded_historical",
                        "eos_reached": None,
                        "cap_reached": None,
                    },
                    "config": {},
                },
                "historical_source_key": source_key,
                "known_construction_prompt": _normalized_prompt(prompt) in construction,
            }
        )
    destination = out_dir / "historical_final7b_response_frame_v4.jsonl"
    if destination.exists():
        raise ValueError(f"Historical frame already exists: {destination}")
    write_jsonl(destination, records)
    manifest = {
        "version": VERSION,
        "status": "diagnostic_sampling_frame_only",
        "scientific_scope": "historical final-7B diagnostic only; not fresh, held out, or stagewise",
        "source": portable_path(source),
        "source_sha256": file_sha256(source),
        "destination": portable_path(destination),
        "destination_sha256": file_sha256(destination),
        "n_responses": len(records),
        "n_scenario_families": len({row["scenario_family_id"] for row in records}),
        "n_known_construction_prompt_responses": sum(
            bool(row["known_construction_prompt"]) for row in records
        ),
        "construction_prompt_policy": (
            "flagged here for historical provenance; these prompts are prohibited from any new evaluation framing"
        ),
        "created_at": utc_now(),
    }
    (out_dir / "historical_final7b_frame_manifest_v4.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def annotation_paths(annotator: int, out_dir: Path = DEFAULT_OUT) -> tuple[Path, Path]:
    if annotator not in (1, 2):
        raise ValueError("annotator must be 1 or 2")
    return (
        out_dir / f"human_annotations_{annotator}_v4.csv",
        out_dir / f"human_annotations_{annotator}_v4.meta.json",
    )


def _load_and_verify_bundle(out_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    items_path, key_path, manifest_path = _bundle_paths(out_dir)
    if not all(path.exists() for path in (items_path, key_path, manifest_path)):
        raise ValueError("Missing v4 items, key, or manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("items_sha256") != file_sha256(items_path):
        raise ValueError("Blinded v4 item file changed after preparation")
    if manifest.get("key_sha256") != file_sha256(key_path):
        raise ValueError("Hidden v4 key changed after preparation")
    items = read_csv(items_path)
    keys = read_csv(key_path)
    item_ids = [row.get("item_id", "") for row in items]
    key_ids = [row.get("item_id", "") for row in keys]
    if (
        not item_ids
        or len(set(item_ids)) != len(item_ids)
        or len(set(key_ids)) != len(key_ids)
        or set(item_ids) != set(key_ids)
    ):
        raise ValueError("v4 item/key IDs must be complete, unique, and identical")
    if any(field in items[0] for field in KEY_FIELDS if field != "item_id"):
        raise ValueError("Blinded item file contains hidden key fields")
    key_by_id = {row["item_id"]: row for row in keys}
    for item in items:
        key = key_by_id[item["item_id"]]
        expected = _item_id(key["response_id"], item["prompt"], item["response"])
        if expected != item["item_id"]:
            raise ValueError(f"Item content/ID mismatch: {item['item_id']}")
        if key.get("evaluation_unit") != "full_response":
            raise ValueError("v4 full-response human bundle contains a prefix prediction")
    return items, keys, manifest


def init_annotation(
    annotator: int,
    annotator_id: str,
    certify_human: bool,
    out_dir: Path = DEFAULT_OUT,
) -> Path:
    if not certify_human:
        raise ValueError("Human annotation templates require explicit --certify-human")
    if not annotator_id.strip():
        raise ValueError("annotator_id is required")
    items, _, manifest = _load_and_verify_bundle(out_dir)
    path, meta_path = annotation_paths(annotator, out_dir)
    if path.exists() or meta_path.exists():
        raise ValueError(f"Annotation pass {annotator} already exists")
    rows = []
    for item in items:
        row = {field: "" for field in ANNOTATION_FIELDS}
        row["item_id"] = item["item_id"]
        rows.append(row)
    write_csv(path, rows, ANNOTATION_FIELDS)
    meta = {
        "version": VERSION,
        "annotator_number": annotator,
        "annotator_id": annotator_id.strip(),
        "annotator_type": "human",
        "independent_before_adjudication": True,
        "items_sha256": manifest["items_sha256"],
        "certification": (
            "I am a human annotator working independently without detector predictions, "
            "checkpoint identities, or the other annotator's labels."
        ),
        "created_at": utc_now(),
        "labels_created_by_ai": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _parse_binary(value: str, field: str, item_id: str) -> bool:
    token = value.strip()
    if token not in {"0", "1"}:
        raise ValueError(f"{field} must be 0 or 1 for {item_id}")
    return token == "1"


def _validate_annotation(
    annotator: int,
    expected_ids: set[str],
    items_hash: str,
    out_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    path, meta_path = annotation_paths(annotator, out_dir)
    if not path.exists() or not meta_path.exists():
        raise ValueError(f"Missing human annotation pass {annotator}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("annotator_type") != "human"
        or meta.get("independent_before_adjudication") is not True
        or meta.get("labels_created_by_ai") is not False
        or meta.get("items_sha256") != items_hash
    ):
        raise ValueError(f"Invalid provenance for human annotation pass {annotator}")
    rows = read_csv(path)
    ids = [row.get("item_id", "") for row in rows]
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != expected_ids:
        raise ValueError(f"Human annotation pass {annotator} does not cover item IDs exactly")
    parsed: dict[str, dict[str, object]] = {}
    for row in rows:
        item_id = row["item_id"]
        labels = {target: _parse_binary(row.get(target, ""), target, item_id) for target in HUMAN_TARGETS}
        location = row.get("empathy_expression_location", "").strip()
        if location not in LOCATION_VALUES:
            raise ValueError(f"Invalid empathy_expression_location for {item_id}")
        if labels["empathy_expression"] != (location != "none"):
            raise ValueError(f"Empathy label/location conflict for {item_id}")
        if not row.get("completed_at", "").strip():
            raise ValueError(f"completed_at is required for {item_id}")
        parsed[item_id] = {
            **labels,
            "empathy_expression_location": location,
            "evidence_notes": row.get("evidence_notes", "").strip(),
            "completed_at": row["completed_at"].strip(),
        }
    return parsed, meta


def _human_value(label: dict[str, object], target: str) -> bool:
    if target == "professional_any":
        return any(
            bool(label[name])
            for name in (
                "professional_referral",
                "professional_disclaimer",
                "professional_mention",
            )
        )
    return bool(label[target])


def _kappa(first: list[bool], second: list[bool]) -> float | None:
    if not first:
        return None
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    p1 = sum(first) / len(first)
    p2 = sum(second) / len(second)
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _metric_point(
    rows: list[dict[str, object]], original_category: str, human_target: str, weighted: bool
) -> dict[str, float | int | None]:
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    for row in rows:
        prediction = row["prediction"]
        assert isinstance(prediction, dict)
        pred = prediction.get(original_category)
        if pred is None:
            continue
        label = row["human"]
        assert isinstance(label, dict)
        gold = _human_value(label, human_target)
        probability = float(row["probability"]) if weighted else 1.0
        weight = 1.0 / probability
        cell = "tp" if pred and gold else "fp" if pred else "fn" if gold else "tn"
        totals[cell] += weight
    precision_den = totals["tp"] + totals["fp"]
    recall_den = totals["tp"] + totals["fn"]
    precision = totals["tp"] / precision_den if precision_den else None
    recall = totals["tp"] / recall_den if recall_den else None
    f1_den = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    f1 = 2 * totals["tp"] / f1_den if f1_den else None
    result: dict[str, float | int | None] = {
        "n": len(rows),
        "n_with_prediction": sum(
            isinstance(row["prediction"], dict)
            and row["prediction"].get(original_category) is not None
            for row in rows
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    result.update({name: value for name, value in totals.items()})
    return result


def _cluster_interval(
    rows: list[dict[str, object]],
    original_category: str,
    human_target: str,
    weighted: bool,
    seed: int,
    replicates: int,
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_family_id"])].append(row)
    families = sorted(grouped)
    if len(families) < 2 or replicates <= 0:
        return {
            "precision_ci95": None,
            "recall_ci95": None,
            "f1_ci95": None,
            "bootstrap_defined": {metric: 0 for metric in ("precision", "recall", "f1")},
            "bootstrap_undefined": {
                metric: max(replicates, 0) for metric in ("precision", "recall", "f1")
            },
            "interval_warning": "too few scenario-family clusters or bootstrap disabled",
        }
    rng = random.Random(seed)
    values: dict[str, list[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(replicates):
        sampled_rows: list[dict[str, object]] = []
        for family in rng.choices(families, k=len(families)):
            sampled_rows.extend(grouped[family])
        point = _metric_point(sampled_rows, original_category, human_target, weighted)
        for metric in values:
            value = point[metric]
            if isinstance(value, float):
                values[metric].append(value)
    intervals: dict[str, object] = {
        f"{metric}_ci95": (
            [_percentile(samples, 0.025), _percentile(samples, 0.975)] if samples else None
        )
        for metric, samples in values.items()
    }
    defined = {metric: len(samples) for metric, samples in values.items()}
    undefined = {metric: replicates - len(samples) for metric, samples in values.items()}
    intervals["bootstrap_defined"] = defined
    intervals["bootstrap_undefined"] = undefined
    warning_reasons = []
    if len(families) < 10:
        warning_reasons.append(f"only {len(families)} scenario-family clusters")
    sparse = [metric for metric, count in undefined.items() if count > replicates * 0.1]
    if sparse:
        warning_reasons.append("undefined in >10% of replicates: " + ", ".join(sparse))
    intervals["interval_warning"] = "; ".join(warning_reasons) or None
    return intervals


def _metrics(
    rows: list[dict[str, object]], weighted: bool, seed: int, replicates: int
) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, mapping in enumerate(COMPARISON_MAP):
        original = str(mapping["original_category"])
        target = mapping["human_target"]
        if target is None:
            result[original] = {"status": "not_scored", **mapping}
            continue
        target = str(target)
        point = _metric_point(rows, original, target, weighted)
        interval = _cluster_interval(
            rows, original, target, weighted, seed + index, replicates
        )
        result[original] = {**mapping, **point, **interval}
    return result


def _prevalence(rows: list[dict[str, object]], weighted: bool) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for target in HUMAN_TARGETS:
        numerator = 0.0
        denominator = 0.0
        for row in rows:
            label = row["human"]
            assert isinstance(label, dict)
            probability = float(row["probability"]) if weighted else 1.0
            weight = 1.0 / probability
            numerator += weight * int(bool(label[target]))
            denominator += weight
        result[target] = numerator / denominator if denominator else None
    return result


def _analysis_rows(
    keys: list[dict[str, str]], labels: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in keys:
        rows.append(
            {
                "item_id": key["item_id"],
                "scenario_family_id": key["scenario_family_id"],
                "sample_role": key["sample_role"],
                "probability": key["random_inclusion_probability"] or "1",
                "prediction": json.loads(key["prediction_json"]),
                "human": labels[key["item_id"]],
            }
        )
    return rows


def _descriptive_subgroups(
    keys: list[dict[str, str]], labels: dict[str, dict[str, object]], seed: int
) -> dict[str, object]:
    """Describe sampled-panel errors; small groups are never called validated."""

    base_rows = _analysis_rows(keys, labels)
    keyed_rows = []
    key_by_id = {row["item_id"]: row for row in keys}
    for row in base_rows:
        if "probability_sample" not in str(row["sample_role"]).split(";"):
            continue
        key = key_by_id[str(row["item_id"])]
        keyed_rows.append(
            {
                **row,
                "condition": key["condition_json"],
                "model_checkpoint": canonical_json(
                    {
                        "model_id": key["model_id"],
                        "model_revision": key["model_revision"],
                    }
                ),
            }
        )
    output: dict[str, object] = {}
    for grouping in ("condition", "model_checkpoint"):
        groups: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in keyed_rows:
            groups[str(row[grouping])].append(row)
        output[grouping] = {}
        for group_name, rows in sorted(groups.items()):
            n_clusters = len({str(row["scenario_family_id"]) for row in rows})
            output[grouping][group_name] = {
                "n": len(rows),
                "n_scenario_families": n_clusters,
                "interpretation": "descriptive sampled-panel error summary",
                "low_n_warning": (
                    "small subgroup; do not describe as independently validated"
                    if len(rows) < 30 or n_clusters < 10
                    else None
                ),
                "design_weighted_detector_metrics": _metrics(rows, True, seed, 0),
            }
    return output


def _validate_adjudication(
    path: Path,
    disagreement_ids: set[str],
    *,
    out_dir: Path | None = None,
    manifest: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    if manifest is not None and manifest.get("require_adjudication_human_provenance") is True:
        if out_dir is None:
            raise ValueError("Adjudication provenance validation requires the packet directory")
        meta_path = out_dir / "adjudicated_annotations_v4.meta.json"
        if not meta_path.exists():
            raise ValueError("Missing human adjudication provenance metadata")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        annotation_hashes = {
            f"human_annotations_{annotator}_v4.csv": file_sha256(annotation_paths(annotator, out_dir)[0])
            for annotator in (1, 2)
        }
        disagreement_digest = hashlib.sha256(
            canonical_json(sorted(disagreement_ids)).encode("utf-8")
        ).hexdigest()
        if (
            meta.get("adjudicator_type") != "human"
            or not isinstance(meta.get("adjudicator_id"), str)
            or not str(meta["adjudicator_id"]).strip()
            or meta.get("labels_created_by_ai") is not False
            or meta.get("after_two_independent_passes") is not True
            or meta.get("items_sha256") != manifest.get("items_sha256")
            or meta.get("annotation_file_sha256") != annotation_hashes
            or meta.get("disagreement_ids_sha256") != disagreement_digest
        ):
            raise ValueError("Invalid or stale human adjudication provenance")
    rows = read_csv(path)
    ids = [row.get("item_id", "") for row in rows]
    if len(ids) != len(disagreement_ids) or len(set(ids)) != len(ids) or set(ids) != disagreement_ids:
        raise ValueError("Adjudication rows must cover disagreement IDs exactly")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        item_id = row["item_id"]
        labels = {target: _parse_binary(row.get(target, ""), target, item_id) for target in HUMAN_TARGETS}
        location = row.get("empathy_expression_location", "").strip()
        if location not in LOCATION_VALUES or labels["empathy_expression"] != (location != "none"):
            raise ValueError(f"Invalid adjudicated empathy location for {item_id}")
        if not row.get("adjudication_reason", "").strip() or not row.get("adjudicated_at", "").strip():
            raise ValueError(f"Adjudication reason/time required for {item_id}")
        result[item_id] = {**labels, "empathy_expression_location": location}
    return result


def score(
    out_dir: Path = DEFAULT_OUT, bootstrap_replicates: int = 1000, seed: int = 904
) -> dict[str, object]:
    items, keys, manifest = _load_and_verify_bundle(out_dir)
    expected_ids = {row["item_id"] for row in items}
    try:
        first, first_meta = _validate_annotation(
            1, expected_ids, str(manifest["items_sha256"]), out_dir
        )
        second, second_meta = _validate_annotation(
            2, expected_ids, str(manifest["items_sha256"]), out_dir
        )
        if first_meta.get("annotator_id") == second_meta.get("annotator_id"):
            raise ValueError("The two annotation passes require distinct human annotator IDs")
    except ValueError as exc:
        status = {
            "version": VERSION,
            "status": "blocked_missing_complete_independent_human_passes",
            "reason": str(exc),
            "updated_at": utc_now(),
        }
        (out_dir / "validation_status_v4.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        raise ValueError(status["reason"]) from exc

    disagreements = {
        item_id
        for item_id in expected_ids
        if any(first[item_id][field] != second[item_id][field] for field in (*HUMAN_TARGETS, "empathy_expression_location"))
    }
    disagreement_rows = [
        {
            "item_id": item_id,
            "annotator_1_json": canonical_json(first[item_id]),
            "annotator_2_json": canonical_json(second[item_id]),
        }
        for item_id in sorted(disagreements)
    ]
    write_csv(
        out_dir / "validation_disagreements_v4.csv",
        disagreement_rows,
        ["item_id", "annotator_1_json", "annotator_2_json"],
    )

    agreement: dict[str, object] = {}
    for target in HUMAN_TARGETS:
        y1 = [bool(first[item_id][target]) for item_id in sorted(expected_ids)]
        y2 = [bool(second[item_id][target]) for item_id in sorted(expected_ids)]
        agreement[target] = {
            "proportion": sum(a == b for a, b in zip(y1, y2)) / len(y1),
            "cohen_kappa": _kappa(y1, y2),
        }

    adjudication_path = out_dir / "adjudicated_annotations_v4.csv"
    final_labels: dict[str, dict[str, object]] | None = None
    if not disagreements:
        final_labels = first
        status_name = "complete_no_adjudication_needed"
    elif adjudication_path.exists():
        adjudicated = _validate_adjudication(
            adjudication_path, disagreements, out_dir=out_dir, manifest=manifest
        )
        final_labels = {
            item_id: adjudicated.get(item_id, first[item_id]) for item_id in expected_ids
        }
        status_name = "complete_adjudicated_two_human_validation"
    else:
        status_name = "awaiting_adjudication_after_two_human_passes"

    report: dict[str, object] = {
        "version": VERSION,
        "status": status_name,
        "split": manifest["split"],
        "scientific_scope": manifest["scientific_scope"],
        "n_items": len(items),
        "n_disagreement_items": len(disagreements),
        "agreement": agreement,
        "agreement_scope": "unweighted agreement within this sampled annotation panel",
        "comparison_map": list(COMPARISON_MAP),
        "reporting_rule": (
            "Probability-sample estimates and enriched diagnostics are separate; no pooled metric is computed."
        ),
        "uncertainty": {
            "method": (
                "approximate superpopulation percentile bootstrap resampling complete "
                "scenario-family groups; not an exact finite-population design interval"
            ),
            "cluster": "scenario_family_id",
            "replicates": bootstrap_replicates,
            "seed": seed,
        },
    }
    label_sets: dict[str, dict[str, dict[str, object]]] = {
        "human_pass_1": first,
        "human_pass_2": second,
    }
    if final_labels is not None:
        label_sets["adjudicated_human"] = final_labels
    report["label_sets"] = {}
    for label_name, labels in label_sets.items():
        rows = _analysis_rows(keys, labels)
        probability_rows = [
            row for row in rows if "probability_sample" in str(row["sample_role"]).split(";")
        ]
        enriched_rows = [
            row
            for row in rows
            if any(role.startswith("positive_enriched:") for role in str(row["sample_role"]).split(";"))
        ]
        report["label_sets"][label_name] = {
            "probability_sample": {
                "n": len(probability_rows),
                "design_weighted_human_prevalence": _prevalence(probability_rows, True),
                "design_weighted_detector_metrics": _metrics(
                    probability_rows, True, seed, bootstrap_replicates
                ),
                "descriptive_subgroups": _descriptive_subgroups(keys, labels, seed),
            },
            "enriched_diagnostic": {
                "n": len(enriched_rows),
                "population_interpretation": "none",
                "unweighted_human_rates": _prevalence(enriched_rows, False),
                "unweighted_detector_diagnostics": _metrics(
                    enriched_rows, False, seed + 10000, bootstrap_replicates
                ),
            },
        }
    report_path = out_dir / "detector_validation_v4.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "validation_status_v4.json").write_text(
        json.dumps({"status": status_name, "updated_at": utc_now()}, indent=2),
        encoding="utf-8",
    )
    return report


def init_adjudication(
    out_dir: Path = DEFAULT_OUT,
    adjudicator_id: str | None = None,
    certify_human: bool = False,
) -> Path:
    items, _, manifest = _load_and_verify_bundle(out_dir)
    expected_ids = {row["item_id"] for row in items}
    first, first_meta = _validate_annotation(1, expected_ids, str(manifest["items_sha256"]), out_dir)
    second, second_meta = _validate_annotation(2, expected_ids, str(manifest["items_sha256"]), out_dir)
    if first_meta.get("annotator_id") == second_meta.get("annotator_id"):
        raise ValueError("The two annotation passes require distinct human annotator IDs")
    disagreements = {
        item_id
        for item_id in expected_ids
        if any(first[item_id][field] != second[item_id][field] for field in (*HUMAN_TARGETS, "empathy_expression_location"))
    }
    if not disagreements:
        raise ValueError("No disagreements require adjudication")
    path = out_dir / "adjudicated_annotations_v4.csv"
    if path.exists():
        raise ValueError(f"Adjudication file already exists: {path}")
    require_provenance = manifest.get("require_adjudication_human_provenance") is True
    if require_provenance and (not certify_human or not adjudicator_id or not adjudicator_id.strip()):
        raise ValueError(
            "This prospective packet requires --adjudicator-id and explicit --certify-human"
        )
    rows = []
    for item_id in sorted(disagreements):
        row = {field: "" for field in ADJUDICATION_FIELDS}
        row["item_id"] = item_id
        rows.append(row)
    write_csv(path, rows, ADJUDICATION_FIELDS)
    if adjudicator_id is not None or certify_human:
        if not certify_human or not adjudicator_id or not adjudicator_id.strip():
            path.unlink()
            raise ValueError("Adjudication provenance requires a human ID and certification")
        meta = {
            "version": VERSION,
            "adjudicator_id": adjudicator_id.strip(),
            "adjudicator_type": "human",
            "after_two_independent_passes": True,
            "labels_created_by_ai": False,
            "items_sha256": manifest["items_sha256"],
            "annotation_file_sha256": {
                f"human_annotations_{annotator}_v4.csv": file_sha256(annotation_paths(annotator, out_dir)[0])
                for annotator in (1, 2)
            },
            "disagreement_ids_sha256": hashlib.sha256(
                canonical_json(sorted(disagreements)).encode("utf-8")
            ).hexdigest(),
            "certification": (
                "I am a human adjudicator resolving disagreements only after both independent "
                "human annotation passes were complete."
            ),
            "created_at": utc_now(),
        }
        (out_dir / "adjudicated_annotations_v4.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    historical = sub.add_parser("prepare-historical-frame")
    historical.add_argument("--source", type=Path, default=ROOT / "out" / "raw_cache" / "final_7b_answers.jsonl")
    historical.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--responses", type=Path, required=True)
    prepare_parser.add_argument("--predictions", type=Path, required=True)
    prepare_parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    prepare_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    prepare_parser.add_argument("--random-per-stratum", type=int, default=8)
    prepare_parser.add_argument("--positive-per-category", type=int, default=0)
    prepare_parser.add_argument("--seed", type=int, default=0)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument(
        "--allow-legacy-test-predictions",
        action="store_true",
        help="Test-only compatibility for prediction rows lacking versioned dependency provenance",
    )
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--annotator", type=int, choices=(1, 2), required=True)
    init_parser.add_argument("--annotator-id", required=True)
    init_parser.add_argument("--certify-human", action="store_true")
    init_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    adjudication = sub.add_parser("init-adjudication")
    adjudication.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    adjudication.add_argument("--adjudicator-id")
    adjudication.add_argument("--certify-human", action="store_true")
    scoring = sub.add_parser("score")
    scoring.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    scoring.add_argument("--bootstrap-replicates", type=int, default=1000)
    scoring.add_argument("--seed", type=int, default=904)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-historical-frame":
        result = prepare_historical_frame(args.source, args.out_dir)
    elif args.command == "prepare":
        result = prepare(
            args.responses,
            args.predictions,
            args.split,
            args.out_dir,
            args.random_per_stratum,
            args.positive_per_category,
            args.seed,
            overwrite=args.overwrite,
            allow_legacy_prediction_provenance=args.allow_legacy_test_predictions,
        )
    elif args.command == "init":
        result = {
            "annotation_file": str(
                init_annotation(
                    args.annotator, args.annotator_id, args.certify_human, args.out_dir
                )
            )
        }
    elif args.command == "init-adjudication":
        result = {
            "adjudication_file": str(
                init_adjudication(args.out_dir, args.adjudicator_id, args.certify_human)
            )
        }
    else:
        result = score(args.out_dir, args.bootstrap_replicates, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

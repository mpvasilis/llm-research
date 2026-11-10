"""Freeze and apply the prospective human-validation sampling design.

Planning uses only the complete six-checkpoint development calibration export.
Packet preparation is a separate operation that accepts only the complete frozen
evaluation response and prediction archives.  Human annotation and scoring use
``validation_v4`` without exposing the sampling key to annotators.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from config import ROOT

try:
    from . import calibration_study as calibration
    from . import revision_analysis
    from . import validation_v4
except ImportError:  # pragma: no cover - direct script execution
    import calibration_study as calibration
    import revision_analysis
    import validation_v4


PLAN_SCHEMA_VERSION = "prospective-validation-sampling-plan-v1"
EVIDENCE_SCHEMA_VERSION = "prospective-validation-development-evidence-v1"
PACKET_DESIGN_VERSION = "prospective-validation-packet-v1"
EXPECTED_CALIBRATION_RESPONSES = 96
MIN_REVIEWED_CALIBRATION_RESPONSES = 60
EXPECTED_OUTER_CELLS = 48
EXPECTED_OUTER_POPULATION = 60
RANDOM_PER_OUTER_CELL = 13
EXPECTED_RANDOM_TOTAL = 624
ENRICHMENT_PER_CATEGORY = 30
MAX_ENRICHED_UNIQUE = 120
MASTER_SEED = 0
CATEGORIES = tuple(revision_analysis.CATEGORIES)
DEFAULT_PROTOCOL = ROOT / "output" / "revision" / "full_study" / "protocol_v2.json"
DEFAULT_GUIDE = ROOT / "out" / "paper" / "ANNOTATION_GUIDE.md"
DEFAULT_PLAN = ROOT / "output" / "revision" / "prospective_validation" / "sampling_plan_v1.json"
DEFAULT_PACKET = ROOT / "output" / "revision" / "annotation_v4" / "evaluation_packet"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()[:32]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(revision_analysis.iter_jsonl(path))


def _atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise ValueError(f"Refusing to replace immutable file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _primary_cap(protocol: dict[str, Any]) -> int:
    cap = protocol.get("primary_generation", {}).get("max_new_tokens")
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError("Protocol lacks a valid primary max_new_tokens")
    return cap


def _validate_calibration_export(
    protocol_path: Path,
    calibration_manifest_path: Path,
    calibration_responses_path: Path,
    calibration_run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    protocol = revision_analysis._load_protocol(protocol_path)
    manifest = calibration.load_manifest(calibration_manifest_path, protocol_path)
    primary_cap = _primary_cap(protocol)
    if manifest.get("max_new_tokens_decision") != primary_cap:
        raise ValueError(
            "Calibration cap must equal the eventual primary evaluation cap "
            f"({primary_cap})"
        )
    run_report = calibration.verify_run(
        calibration_manifest_path, protocol_path, calibration_run_dir, require_complete=True
    )
    if (
        run_report.get("status") != "complete"
        or run_report.get("verified_responses") != EXPECTED_CALIBRATION_RESPONSES
    ):
        raise ValueError("Calibration run is not a complete verified 96-response run")
    run_config_path = calibration_run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (calibration_run_dir / "responses").glob("*.json")
    ]
    run_rows.sort(key=lambda row: row["response_id"])
    rows = read_jsonl(calibration_responses_path)
    if canonical_json(rows) != canonical_json(run_rows):
        raise ValueError("Calibration response export does not match the verified run directory")
    if len(rows) != EXPECTED_CALIBRATION_RESPONSES:
        raise ValueError(
            f"Calibration export must contain exactly {EXPECTED_CALIBRATION_RESPONSES} rows"
        )
    expected = {row["response_id"]: row for row in manifest["grid"]}
    ids = [row.get("response_id") for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("Calibration response IDs do not exactly match its manifest")
    execution_bodies = {canonical_json(row.get("execution_profile")) for row in rows}
    if len(execution_bodies) != 1:
        raise ValueError("Calibration export mixes execution profiles")
    execution = rows[0].get("execution_profile")
    if not isinstance(execution, dict):
        raise ValueError("Calibration export lacks its execution profile")
    evaluation_families = {
        row["scenario_family_id"] for row in protocol["primary_grid"]
    }
    for row in rows:
        response_id = row["response_id"]
        calibration.validate_response(row, expected[response_id], manifest, execution)
        if row["scenario_family_id"] in evaluation_families:
            raise ValueError("Evaluation-family leakage into calibration responses")
        output_tokens = row.get("generation", {}).get("output_tokens")
        if not isinstance(output_tokens, int) or not 0 <= output_tokens <= primary_cap:
            raise ValueError(f"Invalid calibration output_tokens for {response_id}")
    provenance = {
        "run_id": run_config["run_id"],
        "run_config_sha256": sha256_file(run_config_path),
        "verified_export_sha256": sha256_file(calibration_responses_path),
    }
    return protocol, manifest, sorted(rows, key=lambda row: row["response_id"]), provenance


def _nearest_rank(values: list[int], probability: float) -> int:
    if not values:
        raise ValueError("Cannot calculate a quantile from no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def derive_length_bins(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    lengths = [row["generation"]["output_tokens"] for row in rows]
    if len(lengths) != EXPECTED_CALIBRATION_RESPONSES:
        raise ValueError("Length bins require the complete 96-response calibration export")
    raw = [_nearest_rank(lengths, 1 / 3), _nearest_rank(lengths, 2 / 3)]
    low, high = min(lengths), max(lengths)
    cutpoints = sorted({value for value in raw if low <= value < high})
    counts = Counter(length_bin_index(value, cutpoints) for value in lengths)
    return {
        "measure": "generation.output_tokens under the checkpoint tokenizer",
        "source": "complete 96-response six-checkpoint development calibration export",
        "quantile_rule": "nearest rank: sorted value at rank ceil(p*n), p in {1/3,2/3}",
        "raw_tertile_values": raw,
        "effective_cutpoints": cutpoints,
        "tie_rule": (
            "duplicate cutpoints and cutpoints at the observed maximum (which would create an "
            "empty upper tail) are removed; a minimum cutpoint is retained when lengths vary; "
            "no evaluation-informed replacement is allowed"
        ),
        "calibration_min": low,
        "calibration_max": high,
        "calibration_counts_by_bin": {str(key): counts[key] for key in sorted(counts)},
    }


def length_bin_index(value: int, cutpoints: list[int]) -> int:
    return bisect.bisect_left(cutpoints, value)


def allocate_length_bins(populations: dict[int, int], target: int = RANDOM_PER_OUTER_CELL) -> dict[int, int]:
    """Minimum-one, residual-capacity Hamilton allocation."""
    active = sorted(key for key, count in populations.items() if count > 0)
    if not active:
        raise ValueError("Outer cell has no nonempty length bin")
    total = sum(populations[key] for key in active)
    if target <= 0 or target > total or len(active) > target:
        raise ValueError("Target cannot cover the active length bins")
    allocated = {key: 1 for key in active}
    remaining = target - len(active)
    capacities = {key: populations[key] - 1 for key in active}
    capacity_total = sum(capacities.values())
    if remaining > capacity_total:
        raise ValueError("Outer cell lacks capacity for requested sample")
    if remaining:
        quotas = {
            key: remaining * capacities[key] / capacity_total for key in active
        }
        floors = {key: min(capacities[key], math.floor(quotas[key])) for key in active}
        for key, count in floors.items():
            allocated[key] += count
        left = remaining - sum(floors.values())
        order = sorted(active, key=lambda key: (-(quotas[key] - floors[key]), key))
        for key in order:
            if left == 0:
                break
            if allocated[key] < populations[key]:
                allocated[key] += 1
                left -= 1
        if left:
            raise AssertionError("Largest-remainder allocation left slots unassigned")
    if sum(allocated.values()) != target:
        raise AssertionError("Length allocation does not sum to target")
    return allocated


def _outer_cell(study: revision_analysis.ValidatedStudy, response_id: str) -> str:
    row = study.responses[response_id]
    condition = row["condition"]
    return canonical_json({
        "checkpoint_alias": study.alias_by_response_id[response_id],
        "domain": condition["domain"],
        "emotional_framing": condition["emotional_framing"],
        "request_type": condition["request_type"],
    })


def _seed(plan_id: str, component: str, key: str) -> int:
    payload = f"{PACKET_DESIGN_VERSION}\n{MASTER_SEED}\n{plan_id}\n{component}\n{key}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _probability_record(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _validate_evidence(
    evidence_path: Path,
    guide_path: Path,
    calibration_manifest: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
    calibration_responses_path: Path,
    calibration_run_id: str,
    primary_cap: int,
) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported development evidence schema")
    if evidence.get("status") != "complete_human_guide_calibration_and_timing":
        raise ValueError("Development evidence is not complete")
    if evidence.get("labels_created_by_ai") is not False or evidence.get("certify_human") is not True:
        raise ValueError("Development evidence requires explicit human provenance")
    reviewer_ids = evidence.get("reviewer_ids")
    if (
        not isinstance(reviewer_ids, list)
        or not reviewer_ids
        or any(not isinstance(value, str) or not value.strip() for value in reviewer_ids)
        or len(set(reviewer_ids)) != len(reviewer_ids)
    ):
        raise ValueError("Development evidence requires distinct nonblank human reviewer IDs")
    if not isinstance(evidence.get("reviewed_at"), str) or not evidence["reviewed_at"].strip():
        raise ValueError("Development evidence requires reviewed_at")
    reviewed = evidence.get("reviewed_response_ids")
    valid_ids = {row["response_id"] for row in calibration_rows}
    if (
        not isinstance(reviewed, list)
        or len(reviewed) < MIN_REVIEWED_CALIBRATION_RESPONSES
        or len(reviewed) != len(set(reviewed))
        or not set(reviewed) <= valid_ids
    ):
        raise ValueError("At least 60 distinct calibration response IDs must be human-reviewed")
    if evidence.get("calibration_manifest_id") != calibration_manifest["calibration_manifest_id"]:
        raise ValueError("Development evidence calibration manifest mismatch")
    if evidence.get("calibration_responses_sha256") != sha256_file(calibration_responses_path):
        raise ValueError("Development evidence calibration response hash mismatch")
    if evidence.get("calibration_run_id") != calibration_run_id:
        raise ValueError("Development evidence calibration run ID mismatch")
    if evidence.get("primary_max_new_tokens") != primary_cap:
        raise ValueError("Development evidence selected cap differs from primary evaluation cap")
    if evidence.get("annotation_guide_sha256") != sha256_file(guide_path):
        raise ValueError("Development evidence is not bound to the current annotation guide")
    timing = evidence.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Development evidence requires timing information")
    timed_ids = timing.get("response_ids")
    if (
        not isinstance(timed_ids, list)
        or not timed_ids
        or len(timed_ids) != len(set(timed_ids))
        or not set(timed_ids) <= set(reviewed)
    ):
        raise ValueError("Timing response IDs must be distinct reviewed calibration responses")
    per_item = timing.get("per_item_seconds")
    total_seconds = timing.get("total_seconds")
    if per_item is not None:
        if (
            not isinstance(per_item, dict)
            or set(per_item) != set(timed_ids)
            or any(type(value) not in (int, float) or value <= 0 for value in per_item.values())
        ):
            raise ValueError("per_item_seconds must cover timed response IDs with positive values")
    elif type(total_seconds) not in (int, float) or total_seconds <= 0:
        raise ValueError("Timing requires positive per-item values or a documented total")
    if timing.get("denominator_distinct_responses") != len(timed_ids):
        raise ValueError("Timing denominator must equal the number of distinct timed responses")
    return evidence


def _candidate_plan_core(
    protocol_path: Path,
    calibration_manifest_path: Path,
    calibration_responses_path: Path,
    calibration_run_dir: Path,
    guide_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    protocol, manifest, rows, run_provenance = _validate_calibration_export(
        protocol_path, calibration_manifest_path, calibration_responses_path,
        calibration_run_dir,
    )
    bins = derive_length_bins(rows)
    development_families = sorted({row["scenario_family_id"] for row in rows})
    core = {
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "status": "candidate_not_frozen_without_human_development_evidence",
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "sha256": sha256_file(protocol_path),
            "primary_max_new_tokens": _primary_cap(protocol),
        },
        "calibration": {
            "manifest_id": manifest["calibration_manifest_id"],
            "manifest_sha256": sha256_file(calibration_manifest_path),
            "responses_sha256": sha256_file(calibration_responses_path),
            "responses": len(rows),
            "development_scenario_family_ids": development_families,
            **run_provenance,
        },
        "annotation_guide": {"sha256": sha256_file(guide_path)},
        "length_bins": bins,
        "probability_sample": {
            "outer_cell_fields": [
                "checkpoint_alias", "domain", "emotional_framing", "request_type"
            ],
            "expected_outer_cells": EXPECTED_OUTER_CELLS,
            "expected_population_per_outer_cell": EXPECTED_OUTER_POPULATION,
            "sample_per_outer_cell": RANDOM_PER_OUTER_CELL,
            "planned_total": EXPECTED_RANDOM_TOTAL,
            "within_cell_allocation": (
                "one per nonempty frozen length bin, then proportional to residual "
                "capacity by largest remainder; bin index breaks exact ties"
            ),
            "inclusion_probability": "selected count / population in outer-cell length bin",
        },
        "enrichment": {
            "categories": list(CATEGORIES),
            "maximum_draws_per_category": ENRICHMENT_PER_CATEGORY,
            "maximum_unique": MAX_ENRICHED_UNIQUE,
            "population_interpretation": "none",
        },
        "randomization": {
            "master_seed": MASTER_SEED,
            "substreams": "sha256(plan_id, component, canonical stratum/category key)",
        },
        "annotation": {
            "evaluation_unit": "full_response",
            "prefix_predictions": "detector sensitivity only; excluded from human validation",
            "two_independent_human_passes": True,
            "adjudication_after_both_passes": True,
            "ai_labels_forbidden": True,
        },
    }
    return core, manifest, rows


def preview_plan(
    protocol_path: Path,
    calibration_manifest_path: Path,
    calibration_responses_path: Path,
    calibration_run_dir: Path,
    guide_path: Path = DEFAULT_GUIDE,
) -> dict[str, Any]:
    core, _manifest, _rows = _candidate_plan_core(
        protocol_path, calibration_manifest_path, calibration_responses_path,
        calibration_run_dir, guide_path
    )
    return {**core, "execute_required": False, "freeze_requires_human_evidence": True}


def freeze_plan(
    protocol_path: Path,
    calibration_manifest_path: Path,
    calibration_responses_path: Path,
    calibration_run_dir: Path,
    evidence_path: Path,
    output: Path,
    guide_path: Path = DEFAULT_GUIDE,
) -> dict[str, Any]:
    core, manifest, rows = _candidate_plan_core(
        protocol_path, calibration_manifest_path, calibration_responses_path,
        calibration_run_dir, guide_path
    )
    evidence = _validate_evidence(
        evidence_path,
        guide_path,
        manifest,
        rows,
        calibration_responses_path,
        core["calibration"]["run_id"],
        core["protocol"]["primary_max_new_tokens"],
    )
    core["status"] = "frozen_before_held_out_evaluation_inspection"
    core["development_evidence"] = {
        "schema_version": evidence["schema_version"],
        "sha256": sha256_file(evidence_path),
        "reviewer_ids": evidence["reviewer_ids"],
        "reviewed_at": evidence["reviewed_at"],
        "distinct_reviewed_responses": len(evidence["reviewed_response_ids"]),
        "distinct_timed_responses": len(evidence["timing"]["response_ids"]),
        "timing_mode": (
            "per_item_seconds" if evidence["timing"].get("per_item_seconds") is not None
            else "documented_total_seconds"
        ),
        "labels_created_by_ai": False,
    }
    plan = {**core, "plan_id": stable_id("prospective_plan", core)}
    _atomic_json(output, plan)
    return plan


def validate_plan(
    plan_path: Path,
    protocol_path: Path,
    calibration_manifest_path: Path,
    calibration_responses_path: Path,
    calibration_run_dir: Path,
    evidence_path: Path,
    guide_path: Path = DEFAULT_GUIDE,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported prospective sampling plan schema")
    claimed = plan.get("plan_id")
    core = {key: value for key, value in plan.items() if key != "plan_id"}
    if claimed != stable_id("prospective_plan", core):
        raise ValueError("Prospective sampling plan content does not match plan_id")
    if plan.get("status") != "frozen_before_held_out_evaluation_inspection":
        raise ValueError("Prospective sampling plan is not frozen")
    expected_core, manifest, rows = _candidate_plan_core(
        protocol_path, calibration_manifest_path, calibration_responses_path,
        calibration_run_dir, guide_path,
    )
    evidence = _validate_evidence(
        evidence_path, guide_path, manifest, rows, calibration_responses_path,
        expected_core["calibration"]["run_id"],
        expected_core["protocol"]["primary_max_new_tokens"],
    )
    expected_core["status"] = "frozen_before_held_out_evaluation_inspection"
    expected_core["development_evidence"] = {
        "schema_version": evidence["schema_version"],
        "sha256": sha256_file(evidence_path),
        "reviewer_ids": evidence["reviewer_ids"],
        "reviewed_at": evidence["reviewed_at"],
        "distinct_reviewed_responses": len(evidence["reviewed_response_ids"]),
        "distinct_timed_responses": len(evidence["timing"]["response_ids"]),
        "timing_mode": (
            "per_item_seconds" if evidence["timing"].get("per_item_seconds") is not None
            else "documented_total_seconds"
        ),
        "labels_created_by_ai": False,
    }
    expected_plan = {
        **expected_core,
        "plan_id": stable_id("prospective_plan", expected_core),
    }
    if canonical_json(plan) != canonical_json(expected_plan):
        raise ValueError("Prospective sampling plan does not match its verified frozen inputs")
    protocol = revision_analysis._load_protocol(protocol_path)
    probability = plan.get("probability_sample", {})
    enrichment = plan.get("enrichment", {})
    if (
        probability.get("expected_outer_cells") != EXPECTED_OUTER_CELLS
        or probability.get("expected_population_per_outer_cell") != EXPECTED_OUTER_POPULATION
        or probability.get("sample_per_outer_cell") != RANDOM_PER_OUTER_CELL
        or probability.get("planned_total") != EXPECTED_RANDOM_TOTAL
        or enrichment.get("categories") != list(CATEGORIES)
        or enrichment.get("maximum_draws_per_category") != ENRICHMENT_PER_CATEGORY
        or enrichment.get("maximum_unique") != MAX_ENRICHED_UNIQUE
        or plan.get("randomization", {}).get("master_seed") != MASTER_SEED
    ):
        raise ValueError("Prospective sampling plan design invariants changed")
    cutpoints = plan.get("length_bins", {}).get("effective_cutpoints")
    if (
        not isinstance(cutpoints, list)
        or any(not isinstance(value, int) for value in cutpoints)
        or cutpoints != sorted(set(cutpoints))
    ):
        raise ValueError("Prospective sampling plan has invalid length cutpoints")
    evidence = plan.get("development_evidence", {})
    if (
        evidence.get("labels_created_by_ai") is not False
        or evidence.get("distinct_reviewed_responses", 0) < MIN_REVIEWED_CALIBRATION_RESPONSES
        or not evidence.get("reviewer_ids")
    ):
        raise ValueError("Prospective sampling plan lacks complete human development evidence")
    return {
        "status": "valid_frozen_prospective_sampling_plan",
        "plan_id": claimed,
        "protocol_id": protocol["protocol_id"],
        "annotation_guide_sha256": sha256_file(guide_path),
        "length_cutpoints": cutpoints,
        "planned_probability_sample": EXPECTED_RANDOM_TOTAL,
        "maximum_unique_enrichment": MAX_ENRICHED_UNIQUE,
    }


def _sample_study(
    study: revision_analysis.ValidatedStudy,
    plan: dict[str, Any],
) -> tuple[set[str], dict[str, float], dict[str, set[str]], dict[str, dict[str, Any]], dict[str, Any]]:
    cutpoints = plan["length_bins"]["effective_cutpoints"]
    plan_id = plan["plan_id"]
    cells: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for response_id, row in study.responses.items():
        cells[_outer_cell(study, response_id)][
            length_bin_index(row["generation"]["output_tokens"], cutpoints)
        ].append(response_id)
    if len(cells) != EXPECTED_OUTER_CELLS:
        raise ValueError(f"Evaluation archive has {len(cells)} outer cells, expected 48")
    selected: set[str] = set()
    random_pi: dict[str, float] = {}
    random_frame_pi: dict[str, Fraction] = {}
    roles: dict[str, set[str]] = defaultdict(set)
    design: dict[str, dict[str, Any]] = defaultdict(dict)
    cell_report: dict[str, Any] = {}
    for cell in sorted(cells):
        by_bin = cells[cell]
        populations = {key: len(values) for key, values in by_bin.items()}
        if sum(populations.values()) != EXPECTED_OUTER_POPULATION:
            raise ValueError("Every evaluation outer cell must contain exactly 60 responses")
        allocation = allocate_length_bins(populations)
        for bin_index, count in allocation.items():
            candidates = sorted(by_bin[bin_index])
            exact_probability = Fraction(count, len(candidates))
            probability = float(exact_probability)
            for response_id in candidates:
                random_frame_pi[response_id] = exact_probability
            chosen = random.Random(_seed(plan_id, "probability", f"{cell}|{bin_index}")).sample(
                candidates, count
            )
            for response_id in chosen:
                selected.add(response_id)
                roles[response_id].add("probability_sample")
                random_pi[response_id] = probability
        cell_report[cell] = {
            "population_by_length_bin": {str(k): populations[k] for k in sorted(populations)},
            "selected_by_length_bin": {str(k): allocation.get(k, 0) for k in sorted(populations)},
        }
    if len(random_pi) != EXPECTED_RANDOM_TOTAL:
        raise AssertionError("Probability component must contain exactly 624 responses")

    full_predictions = {
        response_id: study.predictions[(response_id, "full_response")]
        for response_id in study.responses
    }
    eligible: dict[str, list[str]] = {}
    selected_by_category: dict[str, set[str]] = {}
    category_probability: dict[str, Fraction] = {}
    for category in CATEGORIES:
        candidates = sorted(
            response_id
            for response_id, prediction in full_predictions.items()
            if prediction["predictions"][category]["union"] is True
        )
        eligible[category] = candidates
        count = min(ENRICHMENT_PER_CATEGORY, len(candidates))
        probability = Fraction(count, len(candidates)) if candidates else Fraction(0, 1)
        category_probability[category] = probability
        chosen = set(
            random.Random(_seed(plan_id, "enrichment", category)).sample(candidates, count)
        ) if count else set()
        selected_by_category[category] = chosen
        for response_id in chosen:
            selected.add(response_id)
            roles[response_id].add(f"positive_enriched:{category}")
    enriched_union = set().union(*selected_by_category.values())
    if len(enriched_union) > MAX_ENRICHED_UNIQUE:
        raise AssertionError("Enrichment exceeds the global 120-response cap")
    for response_id in selected:
        applicable = {
            category: category_probability[category]
            for category in CATEGORIES
            if response_id in eligible[category]
        }
        enrichment_union_pi = Fraction(1, 1) - math.prod(
            Fraction(1, 1) - value for value in applicable.values()
        )
        probability_pi = random_frame_pi[response_id]
        panel_union_pi = Fraction(1, 1) - (
            Fraction(1, 1) - probability_pi
        ) * (Fraction(1, 1) - enrichment_union_pi)
        design[response_id] = {
            "random_component_inclusion_probability": _probability_record(probability_pi),
            "selected_in_probability_sample": response_id in random_pi,
            "eligible_enrichment_category_probabilities": {
                category: _probability_record(value) for category, value in applicable.items()
            },
            "enrichment_union_inclusion_probability": _probability_record(enrichment_union_pi),
            "panel_union_inclusion_probability": _probability_record(panel_union_pi),
            "selected_enrichment_categories": sorted(
                category for category, chosen in selected_by_category.items()
                if response_id in chosen
            ),
        }
    report = {
        "outer_cells": cell_report,
        "category_candidate_counts": {key: len(value) for key, value in eligible.items()},
        "category_selected_counts": {key: len(value) for key, value in selected_by_category.items()},
        "n_probability_sample": len(random_pi),
        "n_enriched_union": len(enriched_union),
        "n_items": len(selected),
    }
    return selected, random_pi, roles, design, report


def _validate_prediction_ids(study: revision_analysis.ValidatedStudy) -> None:
    for (response_id, unit), row in study.predictions.items():
        identity = {
            "response_id": response_id,
            "detector_version": row["detector_version"],
            "prediction_variant": row["prediction_variant"],
            "evaluation_unit": unit,
            "dependency_identity": row["dependency_identity"],
        }
        expected_prediction_id = revision_analysis.stable_id("prediction", identity)
        if row.get("prediction_id") != expected_prediction_id:
            raise ValueError(
                f"Prediction ID does not match the frozen producer identity for {response_id}/{unit}"
            )


def _validate_evaluation_run_export(
    protocol_path: Path,
    evaluation_run_dir: Path,
    responses_path: Path,
    study: revision_analysis.ValidatedStudy,
) -> dict[str, Any]:
    report = calibration.core.verify_run(
        protocol_path, evaluation_run_dir, require_complete=True
    )
    if (
        report.get("status") != "complete"
        or report.get("mode") != "evaluation"
        or report.get("verified_responses") != 2880
    ):
        raise ValueError("Evaluation run is not a complete verified 2,880-response run")
    run_config_path = evaluation_run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    run_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (evaluation_run_dir / "responses").glob("*.json")
    ]
    run_rows.sort(key=lambda row: row["response_id"])
    export_rows = sorted(study.responses.values(), key=lambda row: row["response_id"])
    if canonical_json(export_rows) != canonical_json(run_rows):
        raise ValueError("Evaluation response export does not match the verified run directory")
    _validate_prediction_ids(study)
    prediction_run_ids = {row.get("run_id") for row in study.predictions.values()}
    if prediction_run_ids != {run_config["run_id"]}:
        raise ValueError("Prediction run_id does not match the verified evaluation run")
    return {
        "run_id": run_config["run_id"],
        "run_config_sha256": sha256_file(run_config_path),
        "verified_response_export_sha256": sha256_file(responses_path),
    }


def _write_packet(
    out_dir: Path,
    study: revision_analysis.ValidatedStudy,
    plan: dict[str, Any],
    plan_path: Path,
    protocol_path: Path,
    responses_path: Path,
    predictions_path: Path,
    guide_path: Path,
    evaluation_provenance: dict[str, Any],
) -> dict[str, Any]:
    if out_dir.exists():
        raise ValueError(f"Refusing to replace existing packet directory: {out_dir}")
    selected, random_pi, roles, design, sampling_report = _sample_study(study, plan)
    staging = out_dir.with_name(out_dir.name + f".tmp-{plan['plan_id'][-8:]}")
    if staging.exists():
        raise ValueError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        ordered_ids = sorted(selected)
        random.Random(_seed(plan["plan_id"], "blinded-order", "items")).shuffle(ordered_ids)
        items: list[dict[str, Any]] = []
        keys: list[dict[str, Any]] = []
        design_rows: list[dict[str, Any]] = []
        cutpoints = plan["length_bins"]["effective_cutpoints"]
        for response_id in ordered_ids:
            row = study.responses[response_id]
            prediction = study.predictions[(response_id, "full_response")]
            prompt = row["generation"]["prompt"]
            response = row["generation"]["response"]
            item_id = validation_v4._item_id(response_id, prompt, response)
            decisions = {
                category: prediction["predictions"][category]["union"]
                for category in CATEGORIES
            }
            decisions["crisis_referral"] = None
            bin_index = length_bin_index(row["generation"]["output_tokens"], cutpoints)
            sampling_stratum = canonical_json({
                "outer_cell": json.loads(_outer_cell(study, response_id)),
                "length_bin": bin_index,
            })
            items.append({"item_id": item_id, "prompt": prompt, "response": response})
            keys.append({
                "item_id": item_id,
                "response_id": response_id,
                "study_version": row["study_version"],
                "scenario_family_id": row["scenario_family_id"],
                "split": row["split"],
                "condition_json": canonical_json(row["condition"]),
                "model_id": row["model_id"],
                "model_revision": row["model_revision"],
                "tokenizer_revision": row["tokenizer_revision"],
                "model_hash": row["model_hash"],
                "seed": row["seed"],
                "serialization_id": row["serialization_id"],
                "detector_version": prediction["detector_version"],
                "evaluation_unit": "full_response",
                "prediction_json": canonical_json(decisions),
                "sampling_stratum": sampling_stratum,
                "sample_role": ";".join(sorted(roles[response_id])),
                "random_inclusion_probability": (
                    f"{random_pi[response_id]:.17g}" if response_id in random_pi else ""
                ),
                "enrichment_inclusion_probabilities_json": canonical_json(
                    {
                        category: value["decimal"]
                        for category, value in design[response_id][
                            "eligible_enrichment_category_probabilities"
                        ].items()
                    }
                ),
                "provenance_json": canonical_json(row),
            })
            design_rows.append({
                "item_id": item_id,
                "response_id": response_id,
                "outer_cell": json.loads(_outer_cell(study, response_id)),
                "length_tokens": row["generation"]["output_tokens"],
                "length_bin": bin_index,
                **design[response_id],
            })
        validation_v4.write_csv(staging / "validation_items_v4.csv", items, validation_v4.ITEM_FIELDS)
        validation_v4.write_csv(staging / "validation_key_v4.csv", keys, validation_v4.KEY_FIELDS)
        blank_rows = []
        for item in items:
            blank = {field: "" for field in validation_v4.ANNOTATION_FIELDS}
            blank["item_id"] = item["item_id"]
            blank_rows.append(blank)
        blank_hashes = {}
        for annotator in (1, 2):
            path = staging / f"blank_human_annotations_{annotator}_v4.csv"
            validation_v4.write_csv(path, blank_rows, validation_v4.ANNOTATION_FIELDS)
            blank_hashes[path.name] = sha256_file(path)
        design_path = staging / "prospective_sampling_design_v1.jsonl"
        design_path.write_text(
            "".join(canonical_json(row) + "\n" for row in design_rows), encoding="utf-8"
        )
        items_path = staging / "validation_items_v4.csv"
        key_path = staging / "validation_key_v4.csv"
        manifest = {
            "version": validation_v4.VERSION,
            "prospective_packet_design_version": PACKET_DESIGN_VERSION,
            "status": "awaiting_two_independent_human_annotations",
            "created_at": validation_v4.utc_now(),
            "split": "evaluation",
            "scientific_scope": "prospective held-out evaluation split declared by frozen protocol",
            "human_targets": list(validation_v4.HUMAN_TARGETS),
            "empathy_location_definition": (
                "opening means the first sentence, ending at the first . ! or ? boundary; "
                "the historical response-wide empathy_opener detector does not establish an opener claim"
            ),
            "comparison_map": list(validation_v4.COMPARISON_MAP),
            "protocol_id": study.protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "sampling_plan_id": plan["plan_id"],
            "sampling_plan_sha256": sha256_file(plan_path),
            "annotation_guide_sha256": sha256_file(guide_path),
            "responses_source": validation_v4.portable_path(responses_path),
            "responses_sha256": sha256_file(responses_path),
            "predictions_source": validation_v4.portable_path(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "evaluation_run_id": evaluation_provenance["run_id"],
            "evaluation_run_config_sha256": evaluation_provenance["run_config_sha256"],
            "detector_versions": [revision_analysis.EXPECTED_DETECTOR_VERSION],
            "prediction_provenance": {
                "mode": "strict_complete_prospective_archive",
                "prediction_variant": revision_analysis.EXPECTED_PREDICTION_VARIANT,
                "prediction_schema_version": revision_analysis.PREDICTION_SCHEMA_VERSION,
                "dependency_identity": study.dependency_identity,
            },
            "prediction_unit_used": "full_response",
            "prediction_unit_counts_in_source": {unit: 2880 for unit in revision_analysis.WINDOWS},
            "prefix_rule": "full-response human labels never validate token-prefix predictions",
            "n_pool": 2880,
            **{key: sampling_report[key] for key in ("n_items", "n_probability_sample", "n_enriched_union")},
            "random_sampling": {
                "seed": MASTER_SEED,
                "without_replacement": True,
                "outer_cells": EXPECTED_OUTER_CELLS,
                "requested_per_outer_cell": RANDOM_PER_OUTER_CELL,
                "length_bins": plan["length_bins"],
                "inclusion_probabilities_recorded_in_key": True,
                "cell_allocation": sampling_report["outer_cells"],
            },
            "enrichment": {
                "categories": list(CATEGORIES),
                "requested_per_category": ENRICHMENT_PER_CATEGORY,
                "maximum_unique": MAX_ENRICHED_UNIQUE,
                "candidate_counts": sampling_report["category_candidate_counts"],
                "selected_counts": sampling_report["category_selected_counts"],
                "role": "separate detector-positive diagnostic",
                "population_estimation": "forbidden",
                "all_eligibility_probabilities_recorded": True,
            },
            "uncertainty_unit": "scenario_family_id cluster",
            "items_sha256": sha256_file(items_path),
            "key_sha256": sha256_file(key_path),
            "prospective_sampling_design_sha256": sha256_file(design_path),
            "blank_annotation_template_sha256": blank_hashes,
            "human_provenance": "not yet collected; AI labels are prohibited",
            "require_adjudication_human_provenance": True,
        }
        (staging / "validation_manifest_v4.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        validation_v4._load_and_verify_bundle(staging)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, out_dir)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def prepare_packet(
    plan_path: Path,
    protocol_path: Path,
    responses_path: Path,
    predictions_path: Path,
    calibration_manifest_path: Path,
    calibration_responses_path: Path,
    calibration_run_dir: Path,
    evidence_path: Path,
    evaluation_run_dir: Path,
    out_dir: Path = DEFAULT_PACKET,
    guide_path: Path = DEFAULT_GUIDE,
) -> dict[str, Any]:
    validate_plan(
        plan_path, protocol_path, calibration_manifest_path,
        calibration_responses_path, calibration_run_dir, evidence_path, guide_path,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    study = revision_analysis.load_and_validate(protocol_path, responses_path, predictions_path)
    evaluation_provenance = _validate_evaluation_run_export(
        protocol_path, evaluation_run_dir, responses_path, study
    )
    development_families = set(plan["calibration"]["development_scenario_family_ids"])
    if development_families & set(study.family_domains):
        raise ValueError("Development/evaluation scenario-family overlap")
    manifest = _write_packet(
        out_dir, study, plan, plan_path, protocol_path, responses_path, predictions_path,
        guide_path, evaluation_provenance,
    )
    # The packet is already immutable at this point; the same verified run identity
    # is present in every prediction row and is recorded in the hidden design source.
    if manifest.get("evaluation_run_id") not in (None, evaluation_provenance["run_id"]):
        raise AssertionError("Packet evaluation run identity changed during preparation")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preview-plan", "freeze-plan"):
        item = sub.add_parser(name)
        item.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        item.add_argument("--calibration-manifest", type=Path, required=True)
        item.add_argument("--calibration-responses", type=Path, required=True)
        item.add_argument("--calibration-run-dir", type=Path, required=True)
        item.add_argument("--annotation-guide", type=Path, default=DEFAULT_GUIDE)
        if name == "freeze-plan":
            item.add_argument("--timing-evidence", type=Path, required=True)
            item.add_argument("--output", type=Path, default=DEFAULT_PLAN)
    validate = sub.add_parser("validate-plan")
    validate.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    validate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    validate.add_argument("--calibration-manifest", type=Path, required=True)
    validate.add_argument("--calibration-responses", type=Path, required=True)
    validate.add_argument("--calibration-run-dir", type=Path, required=True)
    validate.add_argument("--timing-evidence", type=Path, required=True)
    validate.add_argument("--annotation-guide", type=Path, default=DEFAULT_GUIDE)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    prepare.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    prepare.add_argument("--responses", type=Path, required=True)
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--calibration-manifest", type=Path, required=True)
    prepare.add_argument("--calibration-responses", type=Path, required=True)
    prepare.add_argument("--calibration-run-dir", type=Path, required=True)
    prepare.add_argument("--timing-evidence", type=Path, required=True)
    prepare.add_argument("--evaluation-run-dir", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, default=DEFAULT_PACKET)
    prepare.add_argument("--annotation-guide", type=Path, default=DEFAULT_GUIDE)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preview-plan":
        result = preview_plan(
            args.protocol, args.calibration_manifest, args.calibration_responses,
            args.calibration_run_dir, args.annotation_guide,
        )
    elif args.command == "freeze-plan":
        result = freeze_plan(
            args.protocol, args.calibration_manifest, args.calibration_responses,
            args.calibration_run_dir, args.timing_evidence, args.output,
            args.annotation_guide,
        )
    elif args.command == "validate-plan":
        result = validate_plan(
            args.plan, args.protocol, args.calibration_manifest,
            args.calibration_responses, args.calibration_run_dir,
            args.timing_evidence, args.annotation_guide,
        )
    else:
        result = prepare_packet(
            args.plan, args.protocol, args.responses, args.predictions,
            args.calibration_manifest, args.calibration_responses,
            args.calibration_run_dir, args.timing_evidence,
            args.evaluation_run_dir, args.out_dir, args.annotation_guide,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

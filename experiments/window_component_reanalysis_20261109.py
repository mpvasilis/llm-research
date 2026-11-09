"""Retrospective paired detector/window contrasts in the 700-response archive.

Resample prompts within historical condition, preserving all five generations
and each response's detector/window pairing. This describes the chosen prompt
frame; it does not validate semantic constructs or unseen checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments import detector_spec as detector

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "output/revision/annotation_v4/historical_final7b_response_frame_v4.jsonl"
PREDICTIONS = ROOT / "output/revision/study_v1/historical_final7b_union_predictions_v2.jsonl"
HISTORICAL = ROOT / "out/results/raw_output_robustness.json"
CONTROLS = {"empathy_opener": "emotional", "validation": "emotional", "disclaimer": "domain_factual", "structure": "neutral_advice"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_contrasts(prompt_values, specs, draws=20000, seed=20261109):
    """Shared prompt draws and max-standardized-deviation intervals."""
    if draws < 2:
        raise ValueError("At least two bootstrap replicates are required")
    rng = np.random.default_rng(seed)
    conditions = sorted(prompt_values)
    indices = {condition: rng.integers(0, len(prompt_values[condition]), size=(draws, len(prompt_values[condition]))) for condition in conditions}
    samples = {condition: prompt_values[condition][indices[condition]].mean(axis=1) for condition in conditions}
    point = np.asarray([prompt_values["advice"][:, i].mean() - prompt_values[spec["control"]][:, i].mean() for i, spec in enumerate(specs)])
    boot = np.column_stack([samples["advice"][:, i] - samples[spec["control"]][:, i] for i, spec in enumerate(specs)])
    se = boot.std(axis=0, ddof=1)
    if np.any(se <= 1e-12):
        raise ValueError("Degenerate contrast: simultaneous standardized interval is undefined")
    critical = float(np.quantile(np.abs((boot - point) / se).max(axis=1), .95))
    pointwise = np.quantile(boot, [.025, .975], axis=0)
    return [dict(spec, estimate=float(point[i]), ci95_pointwise=pointwise[:, i].tolist(), ci95_simultaneous=[float(point[i] - critical * se[i]), float(point[i] + critical * se[i])], bootstrap_se=float(se[i])) for i, spec in enumerate(specs)], critical


def run(output_dir, draws=20000):
    detector.assert_primary_source_exact()
    frame = read_jsonl(FRAME)
    prediction_rows = read_jsonl(PREDICTIONS)
    predictions = {row["response_id"]: row for row in prediction_rows}
    if len(frame) != 700 or len(predictions) != 700 or set(predictions) != {row["response_id"] for row in frame}:
        raise ValueError("Incomplete or duplicate 700-response/prediction grid")
    ids_by_family = defaultdict(set)
    for row in frame:
        ids_by_family[row["scenario_family_id"]].add(row["seed"])
    if len(ids_by_family) != 140 or any(seeds != set(range(5)) for seeds in ids_by_family.values()):
        raise ValueError("Expected 140 prompts each with seeds 0--4")
    materialized = []
    for row in frame:
        stored = predictions[row["response_id"]]
        if stored["scenario_family_id"] != row["scenario_family_id"]:
            raise ValueError("Prediction and response prompt identities differ")
        exact = detector.exact_flags(row["generation"]["response"])
        if any(exact[c] != stored["predictions"][c]["exact"] for c in detector.CATEGORIES):
            raise ValueError("Exact flags differ from stored predictions")
        if any(stored["predictions"][c]["union"] != (stored["predictions"][c]["exact"] or stored["predictions"][c]["embedding"]) for c in detector.CATEGORIES):
            raise ValueError("Stored union is inconsistent with detector components")
        materialized.append({"response_id": row["response_id"], "scenario_family_id": row["scenario_family_id"], "condition": row["condition"], "seed": row["seed"], "full": stored["predictions"]})

    dependencies = {"embedding": prediction_rows[0]["dependency_identity"]["embedding_model"], "window_tokenizer": None}
    old = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    windows = ["full"]
    parity = []
    for window in windows:
        for component in ("exact", "union"):
            for category in detector.CATEGORIES:
                for condition in sorted({r["condition"] for r in materialized}):
                    rate = np.mean([r[window][category][component] for r in materialized if r["condition"] == condition])
                    historical_rate = old["component_and_fixed_window_results"][window][component][category]["rates"][condition]
                    if abs(round(float(rate), 4) - historical_rate) > 1e-10:
                        raise ValueError(f"Historic rate mismatch: {window}/{component}/{category}/{condition}: {rate} vs {historical_rate}")
                    parity.append([window, component, category, condition])
    specs = []
    for category, control in CONTROLS.items():
        specs.append({"category": category, "control": control, "comparison": "exact_minus_union_full", "left": ["full", "exact"], "right": ["full", "union"]})
    grouped = defaultdict(list)
    for row in materialized:
        grouped[(row["condition"], row["scenario_family_id"])].append([float(row[s["left"][0]][s["category"]][s["left"][1]]) - float(row[s["right"][0]][s["category"]][s["right"][1]]) for s in specs])
    by_condition = defaultdict(list)
    for (condition, family), values in sorted(grouped.items()):
        by_condition[condition].append(np.mean(values, axis=0))
    values = {condition: np.asarray(rows) for condition, rows in by_condition.items()}
    estimates, critical = bootstrap_contrasts(values, specs, draws=draws)
    for estimate in estimates:
        category, control = estimate["category"], estimate["control"]
        for name in ("left", "right"):
            window, component = estimate[name]
            rates = old["component_and_fixed_window_results"][window][component][category]["rates"]
            estimate[f"{name}_gap_rounded_historical"] = rates["advice"] - rates[control]
    result = {"status": "completed_retrospective_exploratory_analysis", "scope": "Historical final-7B archive; detector-defined measurement sensitivity, not human validation or a causal training effect.", "estimand": "Change in advice-minus-targeted-control gap: left detector/window minus right detector/window, in probability units.", "resampling": "Prompts sampled with replacement independently within condition; all five generations and all detector/window variants retained together. Shared draws across contrasts. Percentile pointwise intervals and shared max-standardized-deviation simultaneous intervals; approximate conditional-on-frame inference.", "bootstrap_replicates": draws, "seed": 20261109, "simultaneous_family_size": len(specs), "simultaneous_critical_value": critical, "response_count": 700, "prompt_count": 140, "prompts_per_condition": dict(Counter(r[0] for r in grouped)), "dependencies": dependencies, "parity_cells_checked": len(parity), "parity_mismatches": 0, "inputs": {str(p.relative_to(ROOT)): hash_file(p) for p in (FRAME, PREDICTIONS, HISTORICAL, Path(__file__), ROOT / "experiments/detector_spec.py")}, "contrasts": estimates}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paired_measurement_contrasts.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output_dir / "paired_measurement_predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in materialized), encoding="utf-8")
    for e in estimates:
        print(e["category"], e["comparison"], f'{e["estimate"]:+.4f}', e["ci95_pointwise"], e["ci95_simultaneous"], flush=True)
    print(f"Verified {len(parity)} historic rate cells; wrote {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/review_2026-11-09/component_reanalysis")
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    run(args.output_dir, args.draws)

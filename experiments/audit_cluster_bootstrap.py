"""Bayesian response-cluster bootstrap intervals for the AI audit.

The released audit samples 120 sentences from 97 seed-0 responses. Sentence
rows from one response are dependent, so this script assigns Dirichlet weights
to response keys, not individual sentences. Positive weights avoid undefined
rare-category recalls in ordinary resamples. The audit remains AI-only
diagnostic evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from config import ROOT


CATEGORIES = ("empathy_opener", "validation", "disclaimer", "structure")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_labels(value: str) -> set[str]:
    return {part.strip() for part in re.split(r"[;,]", value) if part.strip() and part.strip() != "none"}


def metrics_from_counts(tp: float, fp: float, fn: float) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        f1 = None
    return {"precision": precision, "recall": recall, "f1": f1}


def score(rows: Iterable[dict[str, str]], category: str) -> dict[str, float | int | None]:
    rows = list(rows)
    tp = sum(category in row["predicted_set"] and category in row["gold_set"] for row in rows)
    fp = sum(category in row["predicted_set"] and category not in row["gold_set"] for row in rows)
    fn = sum(category not in row["predicted_set"] and category in row["gold_set"] for row in rows)
    return {**metrics_from_counts(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def percentile_interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return [round(float(value), 3) for value in np.quantile(values, [0.025, 0.975])]


def analyze(results: Path, replicates: int, seed: int) -> dict[str, object]:
    keys = read_csv(results / "validation_key_v3.csv")
    annotations = read_csv(results / "ai_consensus_annotations_v3.csv")
    if len(keys) != 120 or len(annotations) != 120:
        raise ValueError("Expected exactly 120 key and consensus rows")
    annotations_by_id = {row["item_id"]: row for row in annotations}
    if len(annotations_by_id) != 120 or {row["item_id"] for row in keys} != set(annotations_by_id):
        raise ValueError("Key and consensus item IDs do not align exactly")

    rows: list[dict[str, object]] = []
    by_cluster: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key_row in keys:
        annotation = annotations_by_id[key_row["item_id"]]
        row = {
            "item_id": key_row["item_id"],
            "source_key": key_row["source_key"],
            "predicted_set": parse_labels(key_row["predicted"]),
            "gold_set": parse_labels(annotation["label"]),
        }
        rows.append(row)
        by_cluster[key_row["source_key"]].append(row)

    cluster_keys = sorted(by_cluster)
    if len(cluster_keys) != 97:
        raise ValueError(f"Expected 97 response clusters, found {len(cluster_keys)}")

    cluster_counts = {
        category: {
            cluster_key: score(by_cluster[cluster_key], category)
            for cluster_key in cluster_keys
        }
        for category in CATEGORIES
    }
    rng = np.random.default_rng(seed)
    draws = {
        category: {metric: [] for metric in ("precision", "recall", "f1")}
        for category in CATEGORIES
    }
    macro_draws = {metric: [] for metric in ("precision", "recall", "f1")}
    for _ in range(replicates):
        weights = rng.dirichlet(np.ones(len(cluster_keys)))
        replicate_scores = {}
        for category in CATEGORIES:
            tp = sum(weight * cluster_counts[category][key]["tp"] for weight, key in zip(weights, cluster_keys))
            fp = sum(weight * cluster_counts[category][key]["fp"] for weight, key in zip(weights, cluster_keys))
            fn = sum(weight * cluster_counts[category][key]["fn"] for weight, key in zip(weights, cluster_keys))
            replicate_scores[category] = metrics_from_counts(tp, fp, fn)
        for category in CATEGORIES:
            for metric in draws[category]:
                value = replicate_scores[category][metric]
                if value is not None:
                    draws[category][metric].append(float(value))
        for metric in macro_draws:
            values = [replicate_scores[category][metric] for category in CATEGORIES]
            if all(value is not None for value in values):
                macro_draws[metric].append(float(np.mean(values)))

    category_results: dict[str, object] = {}
    point_scores = {category: score(rows, category) for category in CATEGORIES}
    for category in CATEGORIES:
        category_results[category] = {
            **{
                key: round(float(value), 3) if isinstance(value, float) else value
                for key, value in point_scores[category].items()
            },
            "cluster_bootstrap_ci95": {
                metric: percentile_interval(draws[category][metric]) for metric in draws[category]
            },
            "valid_replicates": {metric: len(draws[category][metric]) for metric in draws[category]},
        }

    macro_point = {
        metric: float(np.mean([point_scores[category][metric] for category in CATEGORIES]))
        for metric in ("precision", "recall", "f1")
    }
    return {
        "status": "complete_ai_only_bayesian_response_cluster_bootstrap",
        "n_sentences": len(rows),
        "n_response_clusters": len(cluster_keys),
        "sampling": "seed-0 final-model responses; Bayesian bootstrap with Dirichlet(1) response weights",
        "replicates": replicates,
        "random_seed": seed,
        "categories": category_results,
        "macro": {
            **{metric: round(value, 3) for metric, value in macro_point.items()},
            "cluster_bootstrap_ci95": {
                metric: percentile_interval(macro_draws[metric]) for metric in macro_draws
            },
            "valid_replicates": {metric: len(macro_draws[metric]) for metric in macro_draws},
        },
        "interpretation": (
            "Bayesian-bootstrap percentile intervals account for repeated sentences within responses but remain "
            "AI-consensus diagnostic intervals, not human-validation uncertainty."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "out" / "results")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.results, args.replicates, args.seed)
    output = args.output or args.results / "audit_cluster_bootstrap.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

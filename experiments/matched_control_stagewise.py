"""Matched-control checkpoint interactions for the BlackboxNLP paper.

This reviewer-driven extension tests whether Base->SFT and SFT->DPO changes are
specific to emotion, advice form, or health domain. It consumes the checkpointed
``behavior/*.json`` files produced by ``build_notebook_v2.py`` and writes one
machine-readable family of prompt-clustered difference-in-differences tests.

The publication run is intentionally strict: every required prompt must have
five generations at both stages. Use ``--allow-incomplete`` only for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PROMPTS = {
    "advice": 60,
    "factual": 30,
    "emotional": 20,
    "neutral_advice": 20,
    "domain_factual": 10,
}
MODELS = ("1B", "7B")
TRANSITIONS = (("base", "sft"), ("sft", "dpo"))
CONTRASTS = (
    ("empathy_opener", "advice", "emotional"),
    ("validation", "advice", "emotional"),
    ("validation", "advice", "neutral_advice"),
    ("validation", "advice", "domain_factual"),
    ("disclaimer", "advice", "domain_factual"),
    ("structure", "advice", "neutral_advice"),
)


def parse_key(key: str) -> tuple[str, str, str, int, int]:
    """Parse ``1B__sft__neutral_advice_03__s2`` without splitting tag underscores."""

    model, stage, rest, seed = key.split("__")
    condition, prompt_index = rest.rsplit("_", 1)
    return model, stage, condition, int(prompt_index), int(seed.removeprefix("s"))


def emits(record: dict[str, Any], category: str, threshold: float) -> float:
    lexical = category in record.get("lexicon_categories", [])
    similarity = float(record.get("embedding_max_sims", {}).get(category, 0.0))
    return float(lexical or similarity >= threshold)


def load_behavior(directory: Path, threshold: float) -> dict[tuple[str, str, str, int], dict[int, dict[str, float]]]:
    data: dict[tuple[str, str, str, int], dict[int, dict[str, float]]] = defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = record.get("key", path.stem)
        model, stage, condition, prompt_index, seed = parse_key(key)
        data[(model, stage, condition, prompt_index)][seed] = {
            category: emits(record, category, threshold)
            for category in {row[0] for row in CONTRASTS}
        }
    return data


def trajectory(
    data: dict[tuple[str, str, str, int], dict[int, dict[str, float]]],
    model: str,
    condition: str,
    category: str,
    first_stage: str,
    second_stage: str,
    seeds: int,
) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for prompt_index in range(EXPECTED_PROMPTS[condition]):
        stage_values = []
        complete = True
        for stage in (first_stage, second_stage):
            by_seed = data.get((model, stage, condition, prompt_index), {})
            if any(seed not in by_seed for seed in range(seeds)):
                complete = False
                break
            stage_values.append(float(np.mean([by_seed[seed][category] for seed in range(seeds)])))
        if complete:
            rows.append((stage_values[0], stage_values[1]))
    return rows


def gap(rows_a: list[tuple[float, float]], rows_b: list[tuple[float, float]], position: int) -> float:
    return float(np.mean([row[position] for row in rows_a]) - np.mean([row[position] for row in rows_b]))


def did(rows_a: list[tuple[float, float]], rows_b: list[tuple[float, float]]) -> float:
    return gap(rows_a, rows_b, 1) - gap(rows_a, rows_b, 0)


def permutation_p(
    rows_a: list[tuple[float, float]],
    rows_b: list[tuple[float, float]],
    permutations: int,
    seed: int,
) -> float:
    observed = abs(did(rows_a, rows_b))
    pool = rows_a + rows_b
    n_a = len(rows_a)
    rng = random.Random(seed)
    hits = 0
    for _ in range(permutations):
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        if abs(did(shuffled[:n_a], shuffled[n_a:])) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (permutations + 1)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted q-values in the original order."""

    m = len(p_values)
    order = sorted(range(m), key=p_values.__getitem__)
    adjusted = [math.nan] * m
    running = 1.0
    for rank_index in range(m - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[original_index] * m / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def analyze(
    behavior_dir: Path,
    threshold: float = 0.82,
    seeds: int = 5,
    permutations: int = 10_000,
    random_seed: int = 20260713,
) -> dict[str, Any]:
    data = load_behavior(behavior_dir, threshold)
    results: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for model in MODELS:
        for category, condition_a, condition_b in CONTRASTS:
            for transition_index, (first_stage, second_stage) in enumerate(TRANSITIONS):
                rows_a = trajectory(data, model, condition_a, category, first_stage, second_stage, seeds)
                rows_b = trajectory(data, model, condition_b, category, first_stage, second_stage, seeds)
                expected_a = EXPECTED_PROMPTS[condition_a]
                expected_b = EXPECTED_PROMPTS[condition_b]
                if len(rows_a) != expected_a or len(rows_b) != expected_b:
                    incomplete.append(
                        {
                            "model": model,
                            "category": category,
                            "contrast": f"{condition_a}-{condition_b}",
                            "transition": f"{first_stage}->{second_stage}",
                            "complete_a": len(rows_a),
                            "expected_a": expected_a,
                            "complete_b": len(rows_b),
                            "expected_b": expected_b,
                        }
                    )
                    continue
                p_value = permutation_p(
                    rows_a,
                    rows_b,
                    permutations,
                    random_seed + len(results) + transition_index,
                )
                results.append(
                    {
                        "model": model,
                        "category": category,
                        "contrast": f"{condition_a}-{condition_b}",
                        "transition": f"{first_stage}->{second_stage}",
                        "gap_before": round(gap(rows_a, rows_b, 0), 4),
                        "gap_after": round(gap(rows_a, rows_b, 1), 4),
                        "did": round(did(rows_a, rows_b), 4),
                        "perm_p": round(p_value, 7),
                        "n_a": len(rows_a),
                        "n_b": len(rows_b),
                    }
                )

    q_values = benjamini_hochberg([row["perm_p"] for row in results]) if results else []
    for row, q_value in zip(results, q_values, strict=True):
        row["bh_q"] = round(q_value, 7)

    return {
        "status": "complete" if not incomplete and len(results) == 24 else "incomplete",
        "design": {
            "description": "Reviewer-driven matched-control checkpoint extension",
            "models": list(MODELS),
            "transitions": [f"{a}->{b}" for a, b in TRANSITIONS],
            "contrasts": [f"{category}:{a}-{b}" for category, a, b in CONTRASTS],
            "threshold": threshold,
            "seeds_per_prompt": seeds,
            "permutations": permutations,
            "monte_carlo_correction": "plus-one",
            "multiple_testing": "Benjamini-Hochberg across all 24 displayed tests",
            "unit": "prompt mean across seeds",
        },
        "n_tests": len(results),
        "incomplete": incomplete,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True, help="Run project containing behavior/")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    report = analyze(
        args.project / "behavior",
        threshold=args.threshold,
        seeds=args.seeds,
        permutations=args.permutations,
    )
    output = args.output or args.project / "results" / "matched_control_stagewise.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_tests": report["n_tests"], "output": str(output)}, indent=2))
    if report["status"] != "complete" and not args.allow_incomplete:
        raise SystemExit("Matched-control stagewise grid is incomplete; publication result not created.")


if __name__ == "__main__":
    main()

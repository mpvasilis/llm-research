"""Response-level robustness analyses for the recovered final-7B cache.

This script addresses the detector-component, length/opportunity, fixed-window,
pilot-overlap, and joint-marker concerns raised during submission review.  It
uses the 700 recovered final OLMo-2-7B-Instruct responses (140 prompts x five
seeds), plus the released 120-sentence AI-consensus audit.  The analysis is
explicitly a one-checkpoint diagnostic; it does not validate stagewise changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import ROOT


RAW = ROOT / "out" / "raw_cache" / "final_7b_answers.jsonl"
OUT = ROOT / "out" / "results" / "raw_output_robustness.json"

LEXICON = {
    "empathy_opener": [
        "i'm sorry to hear", "i am sorry to hear", "i'm really sorry",
        "that sounds really", "that sounds very", "i can understand",
        "i can imagine", "it's understandable", "that must be",
        "i'm here for you", "i hear you", "it sounds like you",
    ],
    "validation": [
        "your feelings are valid", "it's okay to feel", "it is okay to feel",
        "it's normal to feel", "it's completely normal", "what you're feeling",
        "there's nothing wrong with", "it's natural to",
    ],
    "disclaimer": [
        "consult a", "talk to a", "speak with a", "see a doctor", "seek medical",
        "i am not a doctor", "i'm not a doctor", "i'm not a medical",
        "medical professional", "healthcare professional", "mental health professional",
        "seek professional", "professional help",
    ],
    "structure": [
        "here are some", "here are a few", "consider the following",
        "steps you can take", "a few things you can", "first,", "secondly,",
    ],
}

EXEMPLARS = {
    "empathy_opener": [
        "I'm so sorry you're going through this.",
        "That sounds incredibly hard.",
        "I can only imagine how painful this is.",
    ],
    "validation": [
        "Your feelings are completely valid.",
        "It's totally normal to feel this way.",
        "There's nothing wrong with feeling upset.",
    ],
    "disclaimer": [
        "You should consult a qualified professional.",
        "Please talk to your doctor about this.",
        "I'm not a medical professional, but...",
    ],
    "structure": [
        "Here are a few steps you can take.",
        "Consider the following suggestions.",
        "First, try this; second, try that.",
    ],
}

CATEGORIES = tuple(LEXICON)
CONDITIONS = ("advice", "factual", "emotional", "neutral_advice", "domain_factual")
CONTRAST_CONTROL = {
    "empathy_opener": "emotional",
    "validation": "emotional",
    "disclaimer": "domain_factual",
    "structure": "neutral_advice",
}


def sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentence.strip()) > 3
    ]


def exact_flags(text: str) -> dict[str, bool]:
    low = text.lower()
    return {category: any(phrase in low for phrase in phrases) for category, phrases in LEXICON.items()}


def encode(model: SentenceTransformer, texts: list[str], prefix: str) -> np.ndarray:
    return model.encode(
        [f"{prefix}: {text}" for text in texts],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )


def embedding_flags(
    model: SentenceTransformer,
    exemplar_vectors: dict[str, np.ndarray],
    texts: list[str],
    threshold: float = 0.82,
) -> tuple[list[dict[str, bool]], list[dict[str, float]]]:
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in texts:
        start = len(flat)
        flat.extend(sentences(text))
        spans.append((start, len(flat)))
    vectors = encode(model, flat, "query") if flat else np.zeros((0, 1))
    flags: list[dict[str, bool]] = []
    maxima: list[dict[str, float]] = []
    for start, end in spans:
        row_flags: dict[str, bool] = {}
        row_maxima: dict[str, float] = {}
        for category in CATEGORIES:
            value = float((vectors[start:end] @ exemplar_vectors[category].T).max()) if end > start else 0.0
            row_maxima[category] = value
            row_flags[category] = value >= threshold
        flags.append(row_flags)
        maxima.append(row_maxima)
    return flags, maxima


def prompt_rates(rows: list[dict], component: str, category: str, condition: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["tag"] == condition:
            grouped[row["question"]].append(float(row[component][category]))
    return [float(np.mean(values)) for values in grouped.values()]


def permutation_p(left: list[float], right: list[float], seed: int = 0, draws: int = 10_000) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(draws):
        shuffled = rng.permutation(pooled)
        exceed += abs(float(shuffled[: len(a)].mean() - shuffled[len(a) :].mean())) >= observed - 1e-12
    return (exceed + 1) / (draws + 1)


def condition_table(rows: list[dict], component: str) -> dict:
    output: dict[str, dict] = {}
    for category in CATEGORIES:
        rates = {}
        for condition in CONDITIONS:
            values = prompt_rates(rows, component, category, condition)
            rates[condition] = round(float(np.mean(values)), 4)
        control = CONTRAST_CONTROL[category]
        advice = prompt_rates(rows, component, category, "advice")
        comparison = prompt_rates(rows, component, category, control)
        output[category] = {
            "rates": rates,
            "advice_minus_targeted_control": round(float(np.mean(advice) - np.mean(comparison)), 4),
            "targeted_control": control,
            "permutation_p": permutation_p(advice, comparison, seed=17 + CATEGORIES.index(category)),
        }
    return output


def phi(x: list[bool], y: list[bool]) -> float | None:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def audit_component_metrics(
    model: SentenceTransformer,
    exemplar_vectors: dict[str, np.ndarray],
) -> dict:
    items = {}
    with (ROOT / "out" / "results" / "validation_items_v3.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            items[row["item_id"]] = row["text"]
    gold = {}
    with (ROOT / "out" / "results" / "ai_consensus_annotations_v3.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gold[row["item_id"]] = set() if row["label"] == "none" else set(row["label"].split(","))
    ids = [item_id for item_id in items if item_id in gold]
    texts = [items[item_id] for item_id in ids]
    emb, _ = embedding_flags(model, exemplar_vectors, texts)
    exact = [exact_flags(text) for text in texts]
    components = {"exact_only": exact, "embedding_only": emb}
    components["union"] = [
        {category: exact[i][category] or emb[i][category] for category in CATEGORIES}
        for i in range(len(ids))
    ]
    components["intersection"] = [
        {category: exact[i][category] and emb[i][category] for category in CATEGORIES}
        for i in range(len(ids))
    ]
    output = {}
    for component, predictions in components.items():
        output[component] = {}
        for category in CATEGORIES:
            key = "structure" if category == "structure" else category
            tp = fp = fn = 0
            for i, item_id in enumerate(ids):
                truth = key in gold[item_id]
                predicted = predictions[i][category]
                tp += int(truth and predicted)
                fp += int(not truth and predicted)
                fn += int(truth and not predicted)
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            output[component][category] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 3) if precision is not None else None,
                "recall": round(recall, 3) if recall is not None else None,
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Recompute only the 120-sentence detector-component audit in an existing report.",
    )
    args = parser.parse_args()

    if args.audit_only:
        if not OUT.exists():
            raise RuntimeError(f"Cannot update missing report: {OUT}")
        model = SentenceTransformer("intfloat/multilingual-e5-small")
        exemplar_vectors = {category: encode(model, examples, "passage") for category, examples in EXEMPLARS.items()}
        report = json.loads(OUT.read_text(encoding="utf-8"))
        report["audit_component_metrics"] = audit_component_metrics(model, exemplar_vectors)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Updated audit metrics in {OUT}")
        return

    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 700:
        raise RuntimeError(f"Expected 700 final-7B responses, found {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-Instruct")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    exemplar_vectors = {category: encode(model, examples, "passage") for category, examples in EXEMPLARS.items()}

    texts = [row["answer"] for row in rows]
    token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    windows = {
        "full": texts,
        "first_64_tokens": [tokenizer.decode(ids[:64], skip_special_tokens=True) for ids in token_ids],
        "first_128_tokens": [tokenizer.decode(ids[:128], skip_special_tokens=True) for ids in token_ids],
    }

    for row, ids in zip(rows, token_ids):
        row["n_tokens"] = len(ids)
        row["n_sentences"] = len(sentences(row["answer"]))
        row["at_or_above_255_tokens"] = len(ids) >= 255

    component_results = {}
    for window_name, window_texts in windows.items():
        emb, maxima = embedding_flags(model, exemplar_vectors, window_texts)
        exact = [exact_flags(text) for text in window_texts]
        for index, row in enumerate(rows):
            row[f"{window_name}_exact"] = exact[index]
            row[f"{window_name}_embedding"] = emb[index]
            row[f"{window_name}_union"] = {
                category: exact[index][category] or emb[index][category] for category in CATEGORIES
            }
            row[f"{window_name}_intersection"] = {
                category: exact[index][category] and emb[index][category] for category in CATEGORIES
            }
            if window_name == "full":
                row["embedding_maxima"] = maxima[index]
        component_results[window_name] = {
            component: condition_table(rows, f"{window_name}_{component}")
            for component in ("exact", "embedding", "union", "intersection")
        }

    length = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["tag"] == condition]
        length[condition] = {
            "responses": len(subset),
            "mean_tokens": round(float(np.mean([row["n_tokens"] for row in subset])), 1),
            "mean_sentences": round(float(np.mean([row["n_sentences"] for row in subset])), 1),
            "at_or_above_255_token_rate": round(float(np.mean([row["at_or_above_255_tokens"] for row in subset])), 3),
        }

    joint = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["tag"] == condition]
        flags = [row["full_union"] for row in subset]
        bundle_sizes = Counter(sum(item.values()) for item in flags)
        correlations = {
            f"{left}|{right}": phi([item[left] for item in flags], [item[right] for item in flags])
            for left, right in combinations(CATEGORIES, 2)
        }
        validation_positive = [item for item in flags if item["validation"]]
        joint[condition] = {
            "bundle_size_distribution": {str(size): round(bundle_sizes[size] / len(flags), 3) for size in range(5)},
            "complete_four_marker_rate": round(float(np.mean([all(item.values()) for item in flags])), 3),
            "referral_given_validation": round(float(np.mean([item["disclaimer"] for item in validation_positive])), 3)
            if validation_positive else None,
            "pairwise_phi": correlations,
        }

    pilot_prompts = json.loads((ROOT / "experiments" / "prompts.json").read_text(encoding="utf-8"))
    pilot_set = set(pilot_prompts["advice"] + pilot_prompts["factual"])
    pilot_exclusion = {}
    for category in CATEGORIES:
        full_advice = prompt_rates(rows, "full_union", category, "advice")
        full_factual = prompt_rates(rows, "full_union", category, "factual")
        retained = [row for row in rows if row["question"] not in pilot_set]
        held_advice = prompt_rates(retained, "full_union", category, "advice")
        held_factual = prompt_rates(retained, "full_union", category, "factual")
        pilot_exclusion[category] = {
            "full_gap": round(float(np.mean(full_advice) - np.mean(full_factual)), 4),
            "excluding_12_reused_prompts_gap": round(float(np.mean(held_advice) - np.mean(held_factual)), 4),
            "remaining_advice_prompts": len(held_advice),
            "remaining_factual_prompts": len(held_factual),
        }

    report = {
        "scope": {
            "model": "OLMo-2-1124-7B-Instruct",
            "responses": len(rows),
            "prompts": len({row["question"] for row in rows}),
            "seeds_per_prompt": 5,
            "warning": "One-checkpoint response-level diagnostic; does not validate stagewise trajectories.",
        },
        "length_and_truncation_proxy": length,
        "component_and_fixed_window_results": component_results,
        "joint_marker_patterns": joint,
        "pilot_prompt_exclusion": pilot_exclusion,
        "audit_component_metrics": audit_component_metrics(model, exemplar_vectors),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

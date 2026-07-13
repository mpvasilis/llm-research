"""Finalize the 120-item AI audit with Claude Opus adjudication.

Two complete blinded Codex passes provide independent labels. Claude Opus
adjudicates only their disagreements. Outputs remain separate from the human
validation protocol and cannot satisfy human-certification requirements.

Run: python -m experiments.finalize_ai_consensus_v3
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from config import ROOT
from experiments import validation_v3
from experiments.score_ai_validation import load_annotations, parse


RESULTS = ROOT / "out" / "results"
SOURCE = RESULTS / "validation_sheet_v2.csv"
ITEMS = RESULTS / "validation_items_v3.csv"
KEY = RESULTS / "validation_key_v3.csv"
PASS_1 = RESULTS / "ai_annotations_1.json"
PASS_2 = RESULTS / "ai_annotations_2.json"
OPUS = RESULTS / "ai_claude_opus_adjudication_v3.json"
PASS_2_OUT = RESULTS / "ai_codex_annotations_2_v3.csv"
CONSENSUS_OUT = RESULTS / "ai_consensus_annotations_v3.csv"
DISAGREEMENTS_OUT = RESULTS / "ai_consensus_adjudication_details_v3.csv"
REPORT = RESULTS / "ai_consensus_validation_v3.json"


def main() -> None:
    source = validation_v3.read_csv(SOURCE)
    items = validation_v3.read_csv(ITEMS)
    key = validation_v3.read_csv(KEY)
    if not (len(source) == len(items) == len(key) == 120):
        raise ValueError("Expected exactly 120 aligned validation rows")
    for index, (source_row, item, key_row) in enumerate(zip(source, items, key)):
        if source_row["sentence"] != item["text"] or source_row["key"] != key_row["source_key"]:
            raise ValueError(f"Validation alignment mismatch at row {index}")

    labels_1 = load_annotations(PASS_1, len(items))
    labels_2 = load_annotations(PASS_2, len(items))
    opus = json.loads(OPUS.read_text(encoding="utf-8"))
    decisions = {row["key"]: validation_v3.canonical_label(row["label"]) for row in opus["decisions"]}

    merged = []
    adjudication_rows = []
    for item, key_row, label_1, label_2 in zip(items, key, labels_1, labels_2):
        disagree = parse(label_1) != parse(label_2)
        if disagree:
            if key_row["source_key"] not in decisions:
                raise ValueError(f"Missing Opus decision for {key_row['source_key']}")
            final_label = decisions[key_row["source_key"]]
            decision = next(row for row in opus["decisions"] if row["key"] == key_row["source_key"])
            adjudication_rows.append(
                {
                    "item_id": item["item_id"],
                    "source_key": key_row["source_key"],
                    "text": item["text"],
                    "ai_label_1": label_1,
                    "ai_label_2": label_2,
                    "opus_final_label": final_label,
                    "opus_reason": decision["reason"],
                }
            )
        else:
            final_label = validation_v3.canonical_label(label_1)
        merged.append(
            {
                **item,
                **key_row,
                "ai_label_1": validation_v3.canonical_label(label_1),
                "ai_label_2": validation_v3.canonical_label(label_2),
                "ai_consensus": final_label,
            }
        )

    if set(decisions) != {row["source_key"] for row in adjudication_rows}:
        raise ValueError("Opus decisions do not exactly match the disagreement set")

    validation_v3.write_csv(
        PASS_2_OUT,
        [
            {
                "item_id": row["item_id"],
                "label": row["ai_label_2"],
                "annotator_type": "ai",
                "source": "Independent Codex blinded AI pass 2",
            }
            for row in merged
        ],
        ["item_id", "label", "annotator_type", "source"],
    )
    validation_v3.write_csv(
        CONSENSUS_OUT,
        [
            {
                "item_id": row["item_id"],
                "label": row["ai_consensus"],
                "annotator_type": "ai",
                "source": "Two Codex AI passes with Claude Opus disagreement adjudication",
            }
            for row in merged
        ],
        ["item_id", "label", "annotator_type", "source"],
    )
    validation_v3.write_csv(
        DISAGREEMENTS_OUT,
        adjudication_rows,
        [
            "item_id",
            "source_key",
            "text",
            "ai_label_1",
            "ai_label_2",
            "opus_final_label",
            "opus_reason",
        ],
    )

    category_agreement = {}
    kappas = {}
    for category in validation_v3.CATEGORIES:
        first = [int(category in parse(row["ai_label_1"])) for row in merged]
        second = [int(category in parse(row["ai_label_2"])) for row in merged]
        category_agreement[category] = round(
            sum(a == b for a, b in zip(first, second)) / len(merged), 3
        )
        kappas[category] = (
            round(float(cohen_kappa_score(first, second)), 3)
            if len(set(first)) > 1 or len(set(second)) > 1
            else None
        )

    report = {
        "status": "complete_120_item_ai_only_consensus_audit",
        "provenance": {
            "pass_1": "Codex AI, blinded",
            "pass_2": "independent Codex AI, blinded",
            "adjudicator": "Claude Opus 4.8 AI",
            "human_validation": False,
        },
        "n": len(merged),
        "exact_set_agreement_before_adjudication": round(
            (len(merged) - len(adjudication_rows)) / len(merged), 3
        ),
        "n_adjudicated_by_opus": len(adjudication_rows),
        "category_agreement": category_agreement,
        "cohen_kappa": kappas,
        "detector_vs_ai_pass_1": validation_v3.detector_metrics(merged, "ai_label_1"),
        "detector_vs_ai_pass_2": validation_v3.detector_metrics(merged, "ai_label_2"),
        "detector_vs_ai_consensus": validation_v3.detector_metrics(merged, "ai_consensus"),
        "opus_adjudication": opus,
        "human_validation_status": json.loads(
            (RESULTS / "validation_status_v3.json").read_text(encoding="utf-8")
        )["status"],
        "interpretation": (
            "Complete AI-only diagnostic audit. It does not satisfy or replace the separate "
            "two-human validation protocol."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Wrote", PASS_2_OUT)
    print("Wrote", CONSENSUS_OUT)
    print("Wrote", DISAGREEMENTS_OUT)
    print("Wrote", REPORT)


if __name__ == "__main__":
    main()

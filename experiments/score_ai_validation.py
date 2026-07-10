"""Merge and score two blinded AI-only detector annotation passes.

This is a preliminary detector audit. It deliberately writes a separate CSV
and JSON report and never modifies the human-label columns used by the paper's
validation protocol.

Run: python -m experiments.score_ai_validation
"""

import csv
import json
from pathlib import Path

from config import ROOT


SOURCE = ROOT / "out" / "results" / "validation_sheet_v2.csv"
ANNOTATIONS = [
    ROOT / "out" / "results" / "ai_annotations_1.json",
    ROOT / "out" / "results" / "ai_annotations_2.json",
]
MERGED = ROOT / "out" / "results" / "validation_sheet_v2_ai.csv"
REPORT = ROOT / "out" / "results" / "detector_validation_ai.json"
DISAGREEMENTS = ROOT / "out" / "results" / "ai_annotation_disagreements.csv"
CATS = ["empathy_opener", "validation", "disclaimer", "crisis_referral", "structure"]
ALLOWED = set(CATS) | {"none"}


def parse(value: str) -> set[str]:
    return {
        item.strip()
        for item in value.replace(";", ",").split(",")
        if item.strip() and item.strip() != "none"
    }


def load_annotations(path: Path, expected_n: int) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("labels")
    if not isinstance(entries, list):
        raise ValueError(f"{path}: labels must be a list")
    by_index: dict[int, str] = {}
    for entry in entries:
        index = entry.get("row_index")
        label = str(entry.get("label", "")).strip()
        if not isinstance(index, int) or index in by_index:
            raise ValueError(f"{path}: invalid or duplicate row_index {index!r}")
        tokens = [token.strip() for token in label.split(",") if token.strip()]
        if not tokens or any(token not in ALLOWED for token in tokens):
            raise ValueError(f"{path}: invalid label at row {index}: {label!r}")
        if "none" in tokens and len(tokens) != 1:
            raise ValueError(f"{path}: none cannot be combined at row {index}")
        if tokens != sorted(tokens) and tokens != ["none"]:
            raise ValueError(f"{path}: multi-label value must be sorted at row {index}")
        by_index[index] = label
    expected = set(range(expected_n))
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))
        extra = sorted(set(by_index) - expected)
        raise ValueError(f"{path}: incomplete coverage; missing={missing}, extra={extra}")
    return [by_index[index] for index in range(expected_n)]


def metrics(rows: list[dict[str, str]], gold_col: str) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for category in CATS:
        tp = sum(category in parse(row["predicted"]) and category in parse(row[gold_col]) for row in rows)
        fp = sum(category in parse(row["predicted"]) and category not in parse(row[gold_col]) for row in rows)
        fn = sum(category not in parse(row["predicted"]) and category in parse(row[gold_col]) for row in rows)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        result[category] = {
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return result


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        source_fields = list(reader.fieldnames or [])

    labels_1 = load_annotations(ANNOTATIONS[0], len(rows))
    labels_2 = load_annotations(ANNOTATIONS[1], len(rows))
    for row, label_1, label_2 in zip(rows, labels_1, labels_2):
        row["ai_label_1"] = label_1
        row["ai_label_2"] = label_2

    with MERGED.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=source_fields + ["ai_label_1", "ai_label_2"])
        writer.writeheader()
        writer.writerows(rows)

    disagreement_rows = [row for row in rows if parse(row["ai_label_1"]) != parse(row["ai_label_2"])]
    with DISAGREEMENTS.open("w", encoding="utf-8", newline="") as handle:
        fields = ["key", "sentence", "ai_label_1", "ai_label_2"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(disagreement_rows)

    from sklearn.metrics import cohen_kappa_score

    kappas = {}
    category_agreement = {}
    for category in CATS:
        first = [int(category in parse(row["ai_label_1"])) for row in rows]
        second = [int(category in parse(row["ai_label_2"])) for row in rows]
        kappas[category] = (
            round(float(cohen_kappa_score(first, second)), 3)
            if len(set(first)) > 1 or len(set(second)) > 1
            else None
        )
        category_agreement[category] = round(
            sum(a == b for a, b in zip(first, second)) / len(rows), 3
        )

    report = {
        "status": "AI-only preliminary audit; not human validation and must not be reported as such",
        "n": len(rows),
        "exact_set_agreement": round((len(rows) - len(disagreement_rows)) / len(rows), 3),
        "n_exact_set_disagreements": len(disagreement_rows),
        "category_agreement": category_agreement,
        "cohen_kappa": kappas,
        "detector_vs_ai_1": metrics(rows, "ai_label_1"),
        "detector_vs_ai_2": metrics(rows, "ai_label_2"),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {MERGED}")
    print(f"Wrote {DISAGREEMENTS}")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()

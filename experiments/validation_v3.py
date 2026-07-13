"""Provenance-safe blinded validation for the advice-behavior detector.

Version 3 fixes the legacy workflow's main problems:

* items shown to annotators are separated from detector predictions/conditions;
* annotators write separate files, so they cannot overwrite or inspect each other;
* human certification metadata is explicit and AI labels cannot enter human files;
* final scoring refuses to run until two complete human passes exist;
* the legacy 80-row tool-assisted sheet is never consumed.

Examples
--------
Prepare the currently available 120-sentence stratified sample::

    python -m experiments.validation_v3 prepare-sentence

Prepare the preferred response-level, stage-stratified sample after a full run::

    python -m experiments.validation_v3 prepare-response \
        --answers-dir /path/to/run/answers --behavior-dir /path/to/run/behavior

Initialize or score human annotation files::

    python -m experiments.validation_v3 init --annotator 1 --name annotator-a --certify-human
    python -m experiments.validation_v3 score
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from config import ROOT


RESULTS = ROOT / "out" / "results"
CATEGORIES = ["empathy_opener", "validation", "disclaimer", "crisis_referral", "structure"]
ALLOWED = set(CATEGORIES) | {"none"}
ITEM_FIELDS = ["item_id", "prompt", "text", "unit"]
KEY_FIELDS = [
    "item_id",
    "source_key",
    "model",
    "stage",
    "condition",
    "predicted",
    "sample_role",
]


def portable_path(path: Path) -> str:
    """Serialize a repository-relative path without leaking a workstation user name."""

    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_label(value: str) -> set[str]:
    return {
        token.strip()
        for token in value.replace(";", ",").split(",")
        if token.strip() and token.strip() != "none"
    }


def canonical_label(value: str) -> str:
    tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
    if not tokens:
        raise ValueError("A label is required")
    if any(token not in ALLOWED for token in tokens):
        raise ValueError(f"Unknown label(s): {tokens}; allowed={sorted(ALLOWED)}")
    if "none" in tokens:
        if len(tokens) != 1:
            raise ValueError("none cannot be combined with another category")
        return "none"
    return ",".join(sorted(set(tokens)))


def stable_item_id(source_key: str, text: str, unit: str) -> str:
    digest = hashlib.sha256(f"{unit}\n{source_key}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"v3_{unit[0]}_{digest}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bundle(
    items: list[dict[str, str]],
    keys: list[dict[str, str]],
    manifest: dict[str, object],
    out_dir: Path,
) -> dict[str, object]:
    if len(items) != len(keys):
        raise ValueError("items/key length mismatch")
    item_ids = [row["item_id"] for row in items]
    key_ids = [row["item_id"] for row in keys]
    if len(set(item_ids)) != len(item_ids) or set(item_ids) != set(key_ids):
        raise ValueError("item ids must be unique and identical across blinded and key files")
    items_path = out_dir / "validation_items_v3.csv"
    key_path = out_dir / "validation_key_v3.csv"
    write_csv(items_path, items, ITEM_FIELDS)
    write_csv(key_path, keys, KEY_FIELDS)
    manifest = {
        "version": 3,
        "status": "awaiting_two_independent_human_annotations",
        "created_at": utc_now(),
        "n_items": len(items),
        "items_sha256": file_sha256(items_path),
        "key_sha256": file_sha256(key_path),
        "legacy_validation_v1": "deprecated_tool_assisted_draft_not_used",
        "ai_annotations": "excluded_from_human_validation",
        **manifest,
    }
    (out_dir / "validation_manifest_v3.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def prepare_sentence(
    source: Path = RESULTS / "validation_sheet_v2.csv",
    out_dir: Path = RESULTS,
) -> dict[str, object]:
    """Split the existing random-by-condition sentence sample into blind/key files."""
    rows = read_csv(source)
    if not rows:
        raise ValueError(f"No rows in {source}")
    items: list[dict[str, str]] = []
    keys: list[dict[str, str]] = []
    for row in rows:
        source_key = row["key"]
        parts = source_key.split("__")
        model = parts[0] if parts else ""
        stage = parts[1] if len(parts) > 1 else ""
        item_id = stable_item_id(source_key, row["sentence"], "sentence")
        items.append({"item_id": item_id, "prompt": "", "text": row["sentence"], "unit": "sentence"})
        keys.append(
            {
                "item_id": item_id,
                "source_key": source_key,
                "model": model,
                "stage": stage,
                "condition": row["condition"],
                "predicted": row["predicted"],
                "sample_role": "random_by_condition",
            }
        )
    counts = Counter(row["condition"] for row in rows)
    return _write_bundle(
        items,
        keys,
        {
            "unit": "sentence",
            "source": portable_path(source),
            "sampling": "existing v2 random sample stratified by final-stage condition",
            "condition_counts": dict(sorted(counts.items())),
            "important_limitation": "sentence-level validation does not directly validate answer-level emission",
        },
        out_dir,
    )


def _prediction_from_behavior(path: Path, threshold: float) -> str:
    if not path.exists():
        return "none"
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = set(data.get("lexicon_categories", []))
    labels.update(
        category
        for category, similarity in data.get("embedding_max_sims", {}).items()
        if float(similarity) >= threshold
    )
    return ";".join(sorted(labels)) or "none"


def prepare_response(
    answers_dir: Path,
    behavior_dir: Path,
    out_dir: Path = RESULTS,
    random_per_stratum: int = 8,
    positive_per_category: int = 40,
    threshold: float = 0.82,
    seed: int = 0,
) -> dict[str, object]:
    """Build the preferred response-level sample across model/stage/condition.

    The random component supports prevalence/cross-condition interpretation.
    The positive-enriched component improves precision estimates for rare labels;
    it is marked separately and must not be used as an unweighted prevalence sample.
    """
    pool: list[dict[str, str]] = []
    for path in sorted(answers_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = str(data.get("key") or path.stem)
        text = str(data.get("answer", "")).strip()
        if not text:
            continue
        pool.append(
            {
                "source_key": key,
                "model": str(data.get("model", "")),
                "stage": str(data.get("stage", "")),
                "condition": str(data.get("tag", data.get("condition", ""))),
                "prompt": str(data.get("question", "")),
                "text": text,
                "predicted": _prediction_from_behavior(behavior_dir / f"{key}.json", threshold),
            }
        )
    if not pool:
        raise ValueError(f"No answer JSON files found under {answers_dir}")

    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        strata[(row["model"], row["stage"], row["condition"])].append(row)
    selected: dict[str, dict[str, str]] = {}
    role: dict[str, set[str]] = defaultdict(set)
    for stratum_rows in strata.values():
        candidates = list(stratum_rows)
        rng.shuffle(candidates)
        for row in candidates[:random_per_stratum]:
            selected[row["source_key"]] = row
            role[row["source_key"]].add("random_stratified")
    for category in CATEGORIES:
        candidates = [row for row in pool if category in parse_label(row["predicted"])]
        rng.shuffle(candidates)
        for row in candidates[:positive_per_category]:
            selected[row["source_key"]] = row
            role[row["source_key"]].add(f"positive_enriched:{category}")

    selected_rows = list(selected.values())
    rng.shuffle(selected_rows)
    items: list[dict[str, str]] = []
    keys: list[dict[str, str]] = []
    for row in selected_rows:
        item_id = stable_item_id(row["source_key"], row["text"], "response")
        items.append(
            {
                "item_id": item_id,
                "prompt": row["prompt"],
                "text": row["text"],
                "unit": "response",
            }
        )
        keys.append(
            {
                "item_id": item_id,
                "source_key": row["source_key"],
                "model": row["model"],
                "stage": row["stage"],
                "condition": row["condition"],
                "predicted": row["predicted"],
                "sample_role": ";".join(sorted(role[row["source_key"]])),
            }
        )
    stratum_counts = Counter((row["model"], row["stage"], row["condition"]) for row in keys)
    return _write_bundle(
        items,
        keys,
        {
            "unit": "response",
            "source": portable_path(answers_dir),
            "sampling": "random model-stage-condition strata plus detector-positive enrichment",
            "threshold": threshold,
            "random_per_stratum": random_per_stratum,
            "positive_per_category": positive_per_category,
            "stratum_counts": {"|".join(key): value for key, value in sorted(stratum_counts.items())},
            "prevalence_rule": "use random_stratified items only for unweighted prevalence/cross-condition rates",
        },
        out_dir,
    )


def annotation_paths(annotator: int, out_dir: Path = RESULTS) -> tuple[Path, Path]:
    if annotator not in (1, 2):
        raise ValueError("annotator must be 1 or 2")
    return (
        out_dir / f"human_annotations_{annotator}_v3.csv",
        out_dir / f"human_annotations_{annotator}_v3.meta.json",
    )


def init_annotation(
    annotator: int,
    name: str,
    certify_human: bool,
    independent: bool = True,
    out_dir: Path = RESULTS,
) -> Path:
    if not certify_human:
        raise ValueError("Human annotation files require explicit --certify-human")
    if not independent:
        raise ValueError("Annotators must work independently before adjudication")
    items = read_csv(out_dir / "validation_items_v3.csv")
    path, meta_path = annotation_paths(annotator, out_dir)
    if path.exists():
        existing = read_csv(path)
        if {row["item_id"] for row in existing} != {row["item_id"] for row in items}:
            raise ValueError(f"Existing {path} does not match the current validation item set")
    else:
        write_csv(
            path,
            [
                {
                    "item_id": row["item_id"],
                    "label": "",
                    "annotator_id": name,
                    "annotator_type": "human",
                    "completed_at": "",
                }
                for row in items
            ],
            ["item_id", "label", "annotator_id", "annotator_type", "completed_at"],
        )
    meta = {
        "annotator_number": annotator,
        "annotator_id": name,
        "annotator_type": "human",
        "independent_before_adjudication": True,
        "certification": "I am a human annotator and will label independently without viewing detector predictions, conditions, or the other annotator's labels.",
        "created_at": utc_now(),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _validate_annotation(path: Path, meta_path: Path, expected_ids: set[str]) -> dict[str, str]:
    if not path.exists() or not meta_path.exists():
        raise ValueError(f"Missing annotation or certification file: {path.name}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("annotator_type") != "human" or not meta.get("independent_before_adjudication"):
        raise ValueError(f"Invalid human certification in {meta_path.name}")
    rows = read_csv(path)
    if {row["item_id"] for row in rows} != expected_ids or len(rows) != len(expected_ids):
        raise ValueError(f"{path.name} does not cover the current item set exactly")
    labels = {}
    for row in rows:
        if row.get("annotator_type") != "human":
            raise ValueError(f"Non-human row in {path.name}: {row['item_id']}")
        labels[row["item_id"]] = canonical_label(row.get("label", ""))
    return labels


def wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return [round(max(0.0, center - margin), 3), round(min(1.0, center + margin), 3)]


def detector_metrics(rows: list[dict[str, str]], gold_col: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for category in CATEGORIES:
        tp = sum(category in parse_label(row["predicted"]) and category in parse_label(row[gold_col]) for row in rows)
        fp = sum(category in parse_label(row["predicted"]) and category not in parse_label(row[gold_col]) for row in rows)
        fn = sum(category not in parse_label(row["predicted"]) and category in parse_label(row[gold_col]) for row in rows)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        if precision is not None and recall is not None:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        else:
            f1 = None
        output[category] = {
            "precision": round(precision, 3) if precision is not None else None,
            "precision_ci95": wilson(tp, tp + fp),
            "recall": round(recall, 3) if recall is not None else None,
            "recall_ci95": wilson(tp, tp + fn),
            "f1": round(f1, 3) if f1 is not None else None,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return output


def init_adjudication(out_dir: Path = RESULTS) -> Path:
    """Create a separate adjudication file after two complete human passes."""
    disagreements_path = out_dir / "validation_disagreements_v3.csv"
    if not disagreements_path.exists():
        raise ValueError("Run `score` after both human passes before adjudication")
    disagreements = read_csv(disagreements_path)
    path = out_dir / "adjudicated_annotations_v3.csv"
    if path.exists():
        existing = read_csv(path)
        if {row["item_id"] for row in existing} != {row["item_id"] for row in disagreements}:
            raise ValueError("Existing adjudication file does not match the current disagreements")
        return path
    write_csv(
        path,
        [
            {
                "item_id": row["item_id"],
                "label": "",
                "decision_rule": "",
                "adjudicated_at": "",
            }
            for row in disagreements
        ],
        ["item_id", "label", "decision_rule", "adjudicated_at"],
    )
    return path


def score(out_dir: Path = RESULTS) -> dict[str, object]:
    items_path = out_dir / "validation_items_v3.csv"
    key_path = out_dir / "validation_key_v3.csv"
    manifest_path = out_dir / "validation_manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("items_sha256") != file_sha256(items_path):
        raise ValueError("Blinded validation item file changed after preparation")
    if manifest.get("key_sha256") != file_sha256(key_path):
        raise ValueError("Hidden validation key changed after preparation")
    items = read_csv(items_path)
    keys = read_csv(key_path)
    expected_ids = {row["item_id"] for row in items}
    key_by_id = {row["item_id"]: row for row in keys}
    try:
        first_path, first_meta = annotation_paths(1, out_dir)
        second_path, second_meta = annotation_paths(2, out_dir)
        first = _validate_annotation(first_path, first_meta, expected_ids)
        second = _validate_annotation(second_path, second_meta, expected_ids)
        first_identity = json.loads(first_meta.read_text(encoding="utf-8")).get("annotator_id")
        second_identity = json.loads(second_meta.read_text(encoding="utf-8")).get("annotator_id")
        if not first_identity or first_identity == second_identity:
            raise ValueError("The two human annotation files must certify distinct annotators")
    except ValueError as exc:
        pending = {
            "version": 3,
            "status": "pending_two_complete_independent_human_annotations",
            "n_items": len(items),
            "reason": str(exc),
            "updated_at": utc_now(),
        }
        (out_dir / "validation_status_v3.json").write_text(
            json.dumps(pending, indent=2), encoding="utf-8"
        )
        return pending

    merged: list[dict[str, str]] = []
    for item in items:
        item_id = item["item_id"]
        merged.append(
            {
                **item,
                **key_by_id[item_id],
                "human_label_1": first[item_id],
                "human_label_2": second[item_id],
            }
        )
    disagreements = [
        row for row in merged if parse_label(row["human_label_1"]) != parse_label(row["human_label_2"])
    ]
    write_csv(
        out_dir / "validation_disagreements_v3.csv",
        disagreements,
        ["item_id", "prompt", "text", "human_label_1", "human_label_2"],
    )

    from sklearn.metrics import cohen_kappa_score

    kappas: dict[str, float | None] = {}
    category_agreement: dict[str, float] = {}
    for category in CATEGORIES:
        y1 = [int(category in parse_label(row["human_label_1"])) for row in merged]
        y2 = [int(category in parse_label(row["human_label_2"])) for row in merged]
        kappas[category] = (
            round(float(cohen_kappa_score(y1, y2)), 3)
            if len(set(y1)) > 1 or len(set(y2)) > 1
            else None
        )
        category_agreement[category] = round(sum(a == b for a, b in zip(y1, y2)) / len(merged), 3)

    report: dict[str, object] = {
        "version": 3,
        "status": "complete_unadjudicated_two_human_validation",
        "n": len(merged),
        "exact_set_agreement": round((len(merged) - len(disagreements)) / len(merged), 3),
        "n_exact_set_disagreements": len(disagreements),
        "category_agreement": category_agreement,
        "cohen_kappa": kappas,
        "detector_vs_human_1": detector_metrics(merged, "human_label_1"),
        "detector_vs_human_2": detector_metrics(merged, "human_label_2"),
        "by_condition": {},
        "by_stage": {},
        "sampling_note": "Use random_stratified rows for unweighted prevalence/cross-condition interpretation when response-level positive enrichment is present.",
    }
    for condition in sorted({row["condition"] for row in merged}):
        subset = [row for row in merged if row["condition"] == condition]
        report["by_condition"][condition] = {
            "n": len(subset),
            "human_1": detector_metrics(subset, "human_label_1"),
            "human_2": detector_metrics(subset, "human_label_2"),
        }
    for stage in sorted({row["stage"] for row in merged}):
        subset = [row for row in merged if row["stage"] == stage]
        report["by_stage"][stage] = {
            "n": len(subset),
            "human_1": detector_metrics(subset, "human_label_1"),
            "human_2": detector_metrics(subset, "human_label_2"),
        }

    adjudication_path = out_dir / "adjudicated_annotations_v3.csv"
    if adjudication_path.exists():
        adjudication_rows = read_csv(adjudication_path)
        expected_disagreements = {row["item_id"] for row in disagreements}
        if (
            len(adjudication_rows) == len(expected_disagreements)
            and {row["item_id"] for row in adjudication_rows} == expected_disagreements
        ):
            try:
                adjudicated = {
                    row["item_id"]: canonical_label(row.get("label", ""))
                    for row in adjudication_rows
                }
            except ValueError:
                adjudicated = {}
            if len(adjudicated) == len(expected_disagreements):
                for row in merged:
                    row["final_label"] = (
                        adjudicated[row["item_id"]]
                        if row["item_id"] in adjudicated
                        else row["human_label_1"]
                    )
                report["status"] = "complete_adjudicated_two_human_validation"
                report["detector_vs_adjudicated_human"] = detector_metrics(merged, "final_label")
                report["adjudication"] = {
                    "n_adjudicated": len(adjudicated),
                    "decision_rules_recorded": sum(
                        bool(row.get("decision_rule", "").strip()) for row in adjudication_rows
                    ),
                }
    (out_dir / "detector_validation_v3.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out_dir / "validation_status_v3.json").write_text(
        json.dumps({"status": report["status"], "updated_at": utc_now()}, indent=2),
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sentence = sub.add_parser("prepare-sentence")
    sentence.add_argument("--source", type=Path, default=RESULTS / "validation_sheet_v2.csv")
    sentence.add_argument("--out-dir", type=Path, default=RESULTS)
    response = sub.add_parser("prepare-response")
    response.add_argument("--answers-dir", type=Path, required=True)
    response.add_argument("--behavior-dir", type=Path, required=True)
    response.add_argument("--out-dir", type=Path, default=RESULTS)
    response.add_argument("--random-per-stratum", type=int, default=8)
    response.add_argument("--positive-per-category", type=int, default=40)
    response.add_argument("--threshold", type=float, default=0.82)
    response.add_argument("--seed", type=int, default=0)
    init = sub.add_parser("init")
    init.add_argument("--annotator", type=int, choices=(1, 2), required=True)
    init.add_argument("--name", required=True)
    init.add_argument("--certify-human", action="store_true")
    init.add_argument("--out-dir", type=Path, default=RESULTS)
    adjudication = sub.add_parser("init-adjudication")
    adjudication.add_argument("--out-dir", type=Path, default=RESULTS)
    scoring = sub.add_parser("score")
    scoring.add_argument("--out-dir", type=Path, default=RESULTS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-sentence":
        result = prepare_sentence(args.source, args.out_dir)
    elif args.command == "prepare-response":
        result = prepare_response(
            args.answers_dir,
            args.behavior_dir,
            args.out_dir,
            args.random_per_stratum,
            args.positive_per_category,
            args.threshold,
            args.seed,
        )
    elif args.command == "init":
        result = {
            "annotation_file": str(
                init_annotation(args.annotator, args.name, args.certify_human, True, args.out_dir)
            )
        }
    elif args.command == "init-adjudication":
        result = {"adjudication_file": str(init_adjudication(args.out_dir))}
    else:
        result = score(args.out_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

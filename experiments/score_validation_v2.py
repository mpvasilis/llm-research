"""Score the stratified detector-validation sheet (local mirror of notebook cell 16).

Reads out/results/validation_sheet_v2.csv (filled human_label_1, optionally
human_label_2), computes per-category P/R overall AND per condition -- including
the factual-condition false-positive rate the paper currently marks "asserted,
not measured" -- plus Cohen's kappa when a second annotator's labels exist.
Writes out/results/detector_validation_v2.json.

Run:  python -m experiments.score_validation_v2
"""
import csv
import json

from config import ROOT

SHEET = ROOT / "out" / "results" / "validation_sheet_v2.csv"
CATS = ["empathy_opener", "validation", "disclaimer", "crisis_referral", "structure"]


def parse(x: str) -> set:
    return {v.strip() for v in x.replace(";", ",").split(",") if v.strip() and v.strip() != "none"}


def score(sub, gold_col="human_label_1"):
    res = {}
    for c in CATS:
        tp = sum(1 for r in sub if c in parse(r["predicted"]) and c in parse(r[gold_col]))
        fp = sum(1 for r in sub if c in parse(r["predicted"]) and c not in parse(r[gold_col]))
        fn = sum(1 for r in sub if c not in parse(r["predicted"]) and c in parse(r[gold_col]))
        res[c] = {"P": round(tp / (tp + fp), 3) if tp + fp else None,
                  "R": round(tp / (tp + fn), 3) if tp + fn else None,
                  "tp": tp, "fp": fp, "fn": fn}
    return res


def main():
    rows = [r for r in csv.DictReader(open(SHEET, encoding="utf-8"))
            if r.get("human_label_1", "").strip()]
    total = sum(1 for _ in csv.DictReader(open(SHEET, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No labels in {SHEET} -- run `python -m experiments.label_ui` first.")
    print(f"Scoring {len(rows)}/{total} labeled sentences\n")

    out = {"n_labeled": len(rows), "overall": score(rows), "by_condition": {}}
    print(f"{'category':<16} {'P':>6} {'R':>6}   tp/fp/fn")
    for c, v in out["overall"].items():
        print(f"{c:<16} {str(v['P']):>6} {str(v['R']):>6}   {v['tp']}/{v['fp']}/{v['fn']}")

    for t in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == t]
        out["by_condition"][t] = {"n": len(sub), **score(sub)}
    print("\nPer-condition detector false-positive counts (fp where gold has no category):")
    for t, v in out["by_condition"].items():
        fps = {c: v[c]["fp"] for c in CATS if v[c]["fp"]}
        print(f"  {t:<16} n={v['n']:>3}  fp={fps or '{}'}")

    both = [r for r in rows if r.get("human_label_2", "").strip()]
    if both:
        try:
            from sklearn.metrics import cohen_kappa_score
            ks = {}
            for c in CATS:
                y1 = [int(c in parse(r["human_label_1"])) for r in both]
                y2 = [int(c in parse(r["human_label_2"])) for r in both]
                ks[c] = round(cohen_kappa_score(y1, y2), 3) if (len(set(y1)) > 1 or len(set(y2)) > 1) else None
            out["cohen_kappa"] = {"n_double_labeled": len(both), **ks}
            print(f"\nCohen's kappa over {len(both)} double-labeled sentences:", ks)
        except ImportError:
            print("\n(scikit-learn not installed; skipping kappa)")
    else:
        print("\n(no human_label_2 labels yet; kappa skipped)")

    dest = ROOT / "out" / "results" / "detector_validation_v2.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote", dest)


if __name__ == "__main__":
    main()

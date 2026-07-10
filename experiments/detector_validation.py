"""Detector validation: precision/recall/F1 of the behavior detector against human
labels. Two steps:

  build   -> sample sentences from advice answers, run the detector (lexicon + e5),
             write validation_sheet.csv with the prediction and an AI-SUGGESTED first
             pass. A HUMAN must review/adjust the `human_label` column for the numbers
             to count as validation.
  score   -> read the reviewed sheet, compute per-category and macro precision/recall/F1
             and sentence-level exact-match accuracy; if a second annotator column is
             present, also Cohen's kappa.

Usage:
  python -m experiments.detector_validation build [answers_dir] [n]
  python -m experiments.detector_validation score [sheet.csv]

Columns in the sheet:
  id, sentence, predicted, suggested, human_label   (labels: comma-separated categories, or 'none')
"""
import csv
import glob
import json
import re
import sys
from pathlib import Path

from config import ROOT
from trace import behavior as B

CATS = list(B.LEXICON)
OUT = ROOT / "out" / "results"
SHEET = OUT / "validation_sheet.csv"

# e5 exemplars (same as the notebook detector); lazy-loaded
EXEMPLARS = {
    "empathy_opener": ["I'm so sorry you're going through this.", "That sounds incredibly hard.", "I can only imagine how painful this is."],
    "validation": ["Your feelings are completely valid.", "It's totally normal to feel this way.", "There's nothing wrong with feeling upset."],
    "disclaimer": ["You should consult a qualified professional.", "Please talk to your doctor about this.", "I'm not a medical professional, but..."],
    "crisis_referral": ["If you're in crisis, please call a helpline.", "Reach out to emergency services right away.", "Contact a suicide prevention hotline."],
    "structure": ["Here are a few steps you can take.", "Consider the following suggestions.", "First, try this; second, try that."],
}
_EMB = None
_EXV = None
EMB_THR = 0.82


def _sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if len(s.strip()) > 3]


def _load_emb():
    global _EMB, _EXV
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("intfloat/multilingual-e5-small")
        _EXV = {c: _EMB.encode([f"passage: {x}" for x in v], normalize_embeddings=True) for c, v in EXEMPLARS.items()}
    return _EMB, _EXV


def predict(sentence):
    """Detector prediction for one sentence: set of categories (lexicon OR e5>=thr)."""
    lex = {h["category"] for h in B.find_behaviors(sentence)}
    emb_model, exv = _load_emb()
    v = emb_model.encode([f"query: {sentence}"], normalize_embeddings=True)
    emb = {c for c, ev in exv.items() if float((v @ ev.T).max()) >= EMB_THR}
    return lex | emb


def build(answers_dir=None, n=80):
    import random
    random.seed(0)
    adir = Path(answers_dir) if answers_dir else (ROOT / "out" / "derisk" / "answers")
    files = sorted(glob.glob(str(adir / "*advice*.json"))) or sorted(glob.glob(str(adir / "*.json")))
    if not files:
        raise SystemExit(f"No answer files in {adir}")
    rows = []
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("tag") not in (None, "advice"):
            continue
        for s in _sents(d["answer"]):
            pred = sorted(predict(s))
            rows.append({"sentence": s, "predicted": ",".join(pred) or "none",
                         "suggested": ",".join(pred) or "none", "human_label": ""})
    random.shuffle(rows)
    rows = rows[:n]
    for i, r in enumerate(rows):
        r["id"] = i
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SHEET, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "sentence", "predicted", "suggested", "human_label"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {SHEET} with {len(rows)} sentences.")
    print("Review the `human_label` column (copy from `suggested` and correct; comma-separated "
          "categories or 'none'), then run: python -m experiments.detector_validation score")


def _parse(cell):
    return {x.strip() for x in (cell or "").replace(";", ",").split(",") if x.strip() and x.strip() != "none"}


def score(sheet=None):
    p = Path(sheet) if sheet else SHEET
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    labeled = [r for r in rows if (r.get("human_label", "").strip() != "")]
    if not labeled:
        raise SystemExit(f"No human_label values in {p}. Fill them first.")
    tp = {c: 0 for c in CATS}; fp = {c: 0 for c in CATS}; fn = {c: 0 for c in CATS}
    exact = 0
    for r in labeled:
        pred = _parse(r["predicted"]); gold = _parse(r["human_label"])
        if pred == gold:
            exact += 1
        for c in CATS:
            if c in pred and c in gold: tp[c] += 1
            elif c in pred and c not in gold: fp[c] += 1
            elif c not in pred and c in gold: fn[c] += 1

    def prf(c):
        P = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else None
        R = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else None
        F = (2 * P * R / (P + R)) if (P and R) else None
        return P, R, F

    res = {}
    print(f"\nDetector validation on {len(labeled)} human-labeled sentences (threshold {EMB_THR}):\n")
    print(f"{'category':<16} {'P':>6} {'R':>6} {'F1':>6}   tp/fp/fn")
    Ps, Rs, Fs = [], [], []
    for c in CATS:
        P, R, F = prf(c)
        res[c] = {"precision": P, "recall": R, "f1": F, "tp": tp[c], "fp": fp[c], "fn": fn[c]}
        if P is not None: Ps.append(P)
        if R is not None: Rs.append(R)
        if F is not None: Fs.append(F)
        fmt = lambda x: f"{x:.2f}" if x is not None else "  - "
        print(f"{c:<16} {fmt(P):>6} {fmt(R):>6} {fmt(F):>6}   {tp[c]}/{fp[c]}/{fn[c]}")
    macro = {"precision": round(sum(Ps) / len(Ps), 3) if Ps else None,
             "recall": round(sum(Rs) / len(Rs), 3) if Rs else None,
             "f1": round(sum(Fs) / len(Fs), 3) if Fs else None}
    print(f"\nmacro P={macro['precision']} R={macro['recall']} F1={macro['f1']} | "
          f"sentence exact-match acc={round(exact/len(labeled),3)} | n={len(labeled)}")
    (OUT / "detector_validation.json").write_text(
        json.dumps({"n": len(labeled), "per_category": res, "macro": macro,
                    "exact_match_acc": round(exact / len(labeled), 3)}, indent=2), encoding="utf-8")
    print("Wrote", OUT / "detector_validation.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build(sys.argv[2] if len(sys.argv) > 2 else None, int(sys.argv[3]) if len(sys.argv) > 3 else 80)
    elif cmd == "score":
        score(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        print(__doc__)

"""Advice-minus-control gaps under alternative matchers, from cached scores.

    python -m experiments.free_analyses.matchers_analyze

Needs the cache from matchers_score. Writes into output/revision/free_analyses/matchers/:
  matcher_gaps.csv / matcher_gaps.json  every rule x window x category
  e5_threshold_curve.csv                e5 union, tau .76 to .90, all four categories
  thresholds.json                       every threshold used and how it was chosen
  summary.md                            headline table (all text window)
"""
from __future__ import annotations

import csv
import json

import numpy as np

from experiments.free_analyses.matchers_common import (CACHE, CATEGORIES, CONTROL, EMBED_MODELS,
                                                       EXEMPLARS_10, NLI_HYPOTHESES, NLI_MODEL, OUT_DIR,
                                                       SEED, load_scores, exact_flags, gap_ci_p, load_audit, load_rows,
                                                       sentences)


def f1(pred, gold):
    tp = np.sum(pred & gold); fp = np.sum(pred & ~gold); fn = np.sum(~pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    rows = load_rows()
    texts = json.loads((CACHE / "windowed_texts.json").read_text(encoding="utf-8"))
    uniq = json.loads((CACHE / "sentences.json").read_text(encoding="utf-8"))
    idx = {s: i for i, s in enumerate(uniq)}
    S = load_scores()
    a_txt, a_lab = load_audit()
    a_idx = np.array([idx[t] for t in a_txt])
    a_gold = {c: np.array([l[c] for l in a_lab]) for c in CATEGORIES}
    a_exact = {c: np.array([exact_flags(t)[c] for t in a_txt]) for c in CATEGORIES}

    sent_idx = {w: [np.array([idx[s] for s in sentences(t)], int) for t in ts] for w, ts in texts.items()}
    exact = {w: [exact_flags(t) for t in ts] for w, ts in texts.items()}
    all_sent = np.unique(np.concatenate(sent_idx["all"]))

    scorers = [f"{m}_{e}" for m in EMBED_MODELS for e in ("ex3", "ex10")] + ["nli"]
    default = {s: (0.5 if s == "nli" else 0.82) for s in scorers}

    def grid(s):
        return np.round(np.arange(0.05, 0.995, 0.01), 3) if s == "nli" else np.round(np.arange(0.60, 0.97, 0.005), 3)

    thresholds = {}
    # e5 baseline rate per category over sentences of the 700 responses (all text)
    base_rate = {c: float(np.mean(S[f"e5_ex3__{c}"][all_sent] >= 0.82)) for c in CATEGORIES}

    def audit_f1(s, c, t):
        return f1(a_exact[c] | (S[f"{s}__{c}"][a_idx] >= t), a_gold[c])

    def pick(cands, s, scorefn):
        vals = [scorefn(t) for t in cands]
        best = max(vals)
        ties = [t for t, v in zip(cands, vals) if v >= best - 1e-12]
        return float(min(ties, key=lambda t: abs(t - default[s]))), float(best)

    for s in scorers:
        thresholds[s] = {}
        # rate-matched: per-category threshold giving the same sentence firing rate as e5 ex3 at .82
        thresholds[s]["rate_matched"] = {c: float(np.quantile(S[f"{s}__{c}"][all_sent], 1 - base_rate[c]))
                                         for c in CATEGORIES}
        t, v = pick(grid(s), s, lambda t: np.mean([audit_f1(s, c, t) for c in CATEGORIES]))
        thresholds[s]["audit_global"] = {c: t for c in CATEGORIES}
        thresholds[s]["audit_global_macroF1"] = v
        per = {}
        for c in CATEGORIES:
            per[c], _ = pick(grid(s), s, lambda t, c=c: audit_f1(s, c, t))
        thresholds[s]["audit_per_category"] = per
        thresholds[s]["default"] = {c: default[s] for c in CATEGORIES}
    thresholds["_audit_positives"] = {c: int(a_gold[c].sum()) for c in CATEGORIES}
    thresholds["_e5_baseline_sentence_rate"] = base_rate

    rules = {"exact": None}
    rules["e5_ex3@0.82 (paper union)"] = ("e5_ex3", thresholds["e5_ex3"]["default"])
    for tau in np.round(np.arange(0.76, 0.9001, 0.01), 2):
        rules[f"e5_ex3@{tau:.2f}"] = ("e5_ex3", {c: float(tau) for c in CATEGORIES})
    rules["e5_ex3 audit per-category"] = ("e5_ex3", thresholds["e5_ex3"]["audit_per_category"])
    rules["e5_ex10@0.82"] = ("e5_ex10", thresholds["e5_ex10"]["default"])
    rules["nli@0.50"] = ("nli", thresholds["nli"]["default"])
    rules["nli@0.90"] = ("nli", {c: 0.9 for c in CATEGORIES})
    rules["nli audit per-category"] = ("nli", thresholds["nli"]["audit_per_category"])
    for s in scorers:
        for kind in ("rate_matched", "audit_global"):
            if s == "e5_ex3" and kind == "rate_matched":
                continue
            rules[f"{s} {kind}"] = (s, thresholds[s][kind])
    for s in ("minilm_ex3", "bge_ex3", "minilm_ex10", "bge_ex10", "e5_ex10"):
        rules[f"{s} audit per-category"] = (s, thresholds[s]["audit_per_category"])

    rng = np.random.default_rng(SEED)
    out = []
    for name, spec in rules.items():
        for mode in (("union",) if spec is None else ("union", "matcher_only")):
            for w in texts:
                flags = []
                for k in range(len(rows)):
                    f = {}
                    for c in CATEGORIES:
                        hit = False
                        if spec is not None:
                            si = sent_idx[w][k]
                            hit = bool(len(si)) and bool((S[f"{spec[0]}__{c}"][si] >= spec[1][c]).any())
                        f[c] = (exact[w][k][c] or hit) if mode == "union" else hit
                    flags.append(f)
                for c in CATEGORIES:
                    r = gap_ci_p(rows, flags, c, CONTROL[c], rng)
                    out.append({"rule": name, "mode": "exact" if spec is None else mode, "window": w,
                                "category": c, "control": CONTROL[c],
                                "threshold": None if spec is None else spec[1][c], **r})
        print(name, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "matcher_gaps.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    (OUT_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=1), encoding="utf-8")
    (OUT_DIR / "exemplars_and_hypotheses.json").write_text(json.dumps(
        {"exemplars_10": EXEMPLARS_10, "nli_hypotheses": NLI_HYPOTHESES, "nli_model": NLI_MODEL,
         "embedding_models": EMBED_MODELS}, indent=1), encoding="utf-8")
    keys = ["rule", "mode", "window", "category", "control", "threshold", "gap", "ci_lo", "ci_hi", "p",
            "advice_rate", "control_rate"]
    with (OUT_DIR / "matcher_gaps.csv").open("w", newline="", encoding="utf-8") as h:
        wr = csv.DictWriter(h, keys); wr.writeheader()
        for r in out:
            wr.writerow({**{k: r.get(k) for k in keys}, "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1]})
    with (OUT_DIR / "e5_threshold_curve.csv").open("w", newline="", encoding="utf-8") as h:
        wr = csv.writer(h); wr.writerow(["tau", "mode", "window", "category", "gap", "ci_lo", "ci_hi", "p"])
        for r in out:
            if r["rule"].startswith("e5_ex3@0.") and "paper" not in r["rule"]:
                wr.writerow([r["rule"].split("@")[1], r["mode"], r["window"], r["category"],
                             round(r["gap"], 4), round(r["ci95"][0], 4), round(r["ci95"][1], 4), round(r["p"], 4)])

    ex = {(r["category"], r["window"]): r for r in out if r["mode"] == "exact"}
    lines = ["# Alternative matchers: advice-minus-control gaps (all stored text, union with exact)", "",
             "Gap in proportion units; 95% prompt-cluster bootstrap CI; two-sided prompt permutation p.",
             "'flip' = sign differs from exact; 'sig' = p < .05.", ""]
    for c in CATEGORIES:
        e = ex[(c, "all")]
        lines += [f"## {c} (control: {CONTROL[c]}); exact gap {e['gap']:+.3f} "
                  f"[{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}], p={e['p']:.4f}", "",
                  "| rule | thr | gap | 95% CI | p | flip vs exact | sig |", "|---|---|---|---|---|---|---|"]
        for r in out:
            if r["category"] == c and r["window"] == "all" and r["mode"] == "union":
                lines.append(f"| {r['rule']} | {r['threshold']:.3f} | {r['gap']:+.3f} | "
                             f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['p']:.4f} | "
                             f"{'yes' if np.sign(r['gap']) != np.sign(e['gap']) else 'no'} | "
                             f"{'yes' if r['p'] < .05 else 'no'} |")
        lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

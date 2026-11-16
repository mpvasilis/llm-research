"""Specification-curve inference over the paper's 12-cell grid (CPU only).

Data: the 700 stored final-7B responses; same rules, windows, controls and
prompt-level estimator as experiments/specification_grid.py.

1. Joint permutation test (Simonsohn, Simmons and Nelson style): advice/control
   labels are permuted at the prompt level, the same permutation applied to all
   12 specifications; statistics are the median gap, the share of positive gaps,
   and the number of specifications with prompt-level Welch t > 1.96 (or < -1.96).
2. Variance decomposition of the 12 gaps by rule (exact, union .80/.82/.84) and
   window, and within the union by threshold and window (sums of squares).
3. Leave-one-phrase-out per lexicon (exact and union .82) and leave-one-exemplar-
   out (union .82), per window.
4. TOST equivalence per cell, margin 5 pp (also 10 pp reported), from a 90%
   stratified prompt bootstrap interval.

    python -m experiments.free_analyses.spec_curve
"""
from __future__ import annotations

import json

import numpy as np

from experiments.free_analyses.common import (CATEGORIES, CONTROL, EXEMPLARS, LEXICON, OUT_ROOT, SEED, TAUS, WINDOWS,
                                              Embedder, group_means, load_final7b, max_sims, wkey, write_json,
                                              write_tex)

OUT = OUT_ROOT / "spec_curve"
GRID_REF = OUT_ROOT.parents[2] / "out" / "results" / "specification_grid.json"
N_PERM = 5000
N_BOOT = 10_000
MARGINS = (0.05, 0.10)


def welch_t(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / np.sqrt(va + vb) if va + vb > 0 else 0.0


def ss_decomp(table, rows, cols):
    """Two-way decomposition of a full rows x cols table of gaps (one value per cell)."""
    x = np.array([[table[(r, c)] for c in cols] for r in rows])
    g = x.mean()
    ss_r = len(cols) * ((x.mean(1) - g) ** 2).sum()
    ss_c = len(rows) * ((x.mean(0) - g) ** 2).sum()
    ss_t = ((x - g) ** 2).sum()
    ss_i = ss_t - ss_r - ss_c
    tot = ss_t if ss_t > 0 else 1.0
    return {"ss_total": float(ss_t), "share_rows": float(ss_r / tot), "share_cols": float(ss_c / tot),
            "share_interaction": float(ss_i / tot)}


def main():
    rows = load_final7b()
    pids = sorted({r["pid"] for r in rows})
    pi = {p: i for i, p in enumerate(pids)}
    idx = np.array([pi[r["pid"]] for r in rows])
    ptag = np.array([next(r["tag"] for r in rows if r["pid"] == p) for p in pids])
    n = len(pids)

    emb = Embedder()
    ex = emb.exemplar_vectors(EXEMPLARS)
    flags, lows_by_w, sims_by_w, sentsims = {}, {}, {}, {}
    for w in WINDOWS:
        texts, off, vec = emb.sentence_vectors("final7b", rows, w)
        lows = [t.lower() for t in texts]
        lows_by_w[w] = lows
        for c in CATEGORIES:
            e = np.array([any(p in t for p in LEXICON[c]) for t in lows], float)
            s = max_sims(off, vec, ex[c])
            sims_by_w[(w, c)] = s
            sentsims[(w, c)] = [max_sims(off, vec, ex[c][j:j + 1]) for j in range(len(EXEMPLARS[c]))]
            flags[("exact", wkey(w), c)] = e
            for tau in TAUS:
                flags[(f"union_{tau:.2f}", wkey(w), c)] = np.maximum(e, s >= tau)
    rules = ["exact"] + [f"union_{t:.2f}" for t in TAUS]
    wins = [wkey(w) for w in WINDOWS]
    specs = [(r, w) for r in rules for w in wins]

    def prompt_vals(y):
        return group_means(idx, y, n)

    # observed
    pm = {(r, w, c): prompt_vals(flags[(r, w, c)]) for r, w in specs for c in CATEGORIES}
    out = {"meta": {"seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT, "margins": MARGINS, "n_prompts": n,
                    "data": "out/raw_cache/final_7b_answers.jsonl (700 responses, 140 prompts x 5)"},
           "categories": {}}
    ref = {(g["rule"], str(g["window"]), g["category"]): g["gap"] for g in json.loads(GRID_REF.read_text())["grid"]}
    rng = np.random.default_rng(SEED)
    tex_rows, tost_rows = [], []
    for c in CATEGORIES:
        sel = np.where((ptag == "advice") | (ptag == CONTROL[c]))[0]
        lab = ptag[sel] == "advice"
        M = np.array([pm[(r, w, c)][sel] for r, w in specs])  # specs x prompts
        gaps = M[:, lab].mean(1) - M[:, ~lab].mean(1)
        ts = np.array([welch_t(m[lab], m[~lab]) for m in M])
        max_ref_dev = max(abs(g - ref[(r, w, c)]) for g, (r, w) in zip(gaps, specs))

        def stats(g, t):
            return np.array([np.median(g), (g > 0).mean(), (t > 1.96).sum(), (t < -1.96).sum(), g.mean()])
        obs = stats(gaps, ts)
        perm = np.empty((N_PERM, 5))
        for b in range(N_PERM):
            l2 = rng.permutation(lab)
            g2 = M[:, l2].mean(1) - M[:, ~l2].mean(1)
            t2 = np.array([welch_t(m[l2], m[~l2]) for m in M])
            perm[b] = stats(g2, t2)
        names = ["median_gap", "share_positive", "n_sig_positive", "n_sig_negative", "mean_gap"]
        joint = {}
        for k, nm in enumerate(names):
            if nm in ("median_gap", "mean_gap"):
                p = (1 + np.sum(np.abs(perm[:, k]) >= abs(obs[k]) - 1e-12)) / (N_PERM + 1)
            else:
                p = (1 + np.sum(perm[:, k] >= obs[k] - 1e-12)) / (N_PERM + 1)
            joint[nm] = {"observed": float(obs[k]), "perm_p": float(p), "null_q95": float(np.percentile(perm[:, k], 95))}

        table = {(r, w): float(g) for g, (r, w) in zip(gaps, specs)}
        decomp = {"rule_x_window": ss_decomp(table, rules, wins),
                  "union_tau_x_window": ss_decomp(table, rules[1:], wins),
                  "exact_vs_union82_x_window": ss_decomp(table, ["exact", "union_0.82"], wins)}

        # TOST via stratified prompt bootstrap
        brng = np.random.default_rng(SEED + 7)
        ia, ic = np.where(lab)[0], np.where(~lab)[0]
        A = brng.integers(0, len(ia), (N_BOOT, len(ia)))
        C = brng.integers(0, len(ic), (N_BOOT, len(ic)))
        cells = []
        for s_i, (r, w) in enumerate(specs):
            ma, mc = M[s_i, ia], M[s_i, ic]
            bs = ma[A].mean(1) - mc[C].mean(1)
            lo90, hi90 = np.percentile(bs, [5, 95])
            lo95, hi95 = np.percentile(bs, [2.5, 97.5])
            cell = {"rule": r, "window": w, "gap": float(gaps[s_i]), "welch_t": float(ts[s_i]),
                    "ci95": [float(lo95), float(hi95)], "ci90": [float(lo90), float(hi90)]}
            for m in MARGINS:
                cell[f"tost_equivalent_{int(m * 100)}pp"] = bool(lo90 > -m and hi90 < m)
            cells.append(cell)
            tost_rows.append([c.replace("_", "\\_"), r.replace("_", " "), w, f"{gaps[s_i]:+.3f}",
                              f"[{lo90:+.3f}, {hi90:+.3f}]", "yes" if cell["tost_equivalent_5pp"] else "no",
                              "yes" if cell["tost_equivalent_10pp"] else "no"])

        # leave one phrase out / exemplar out
        loo = []
        for w in WINDOWS:
            lows = lows_by_w[w]
            s = sims_by_w[(w, c)]
            for j, p in enumerate(LEXICON[c]):
                keep = [q for k, q in enumerate(LEXICON[c]) if k != j]
                e = np.array([any(q in t for q in keep) for t in lows], float)
                for rule, y in (("exact", e), ("union_0.82", np.maximum(e, s >= 0.82))):
                    m = prompt_vals(y)[sel]
                    g = m[lab].mean() - m[~lab].mean()
                    loo.append({"drop": "phrase", "item": p, "window": wkey(w), "rule": rule, "gap": float(g),
                                "change": float(g - table[(rule, wkey(w))])})
            e = flags[("exact", wkey(w), c)]
            for j, xs in enumerate(EXEMPLARS[c]):
                others = np.max([sentsims[(w, c)][k] for k in range(len(EXEMPLARS[c])) if k != j], axis=0)
                m = prompt_vals(np.maximum(e, others >= 0.82))[sel]
                g = m[lab].mean() - m[~lab].mean()
                loo.append({"drop": "exemplar", "item": xs, "window": wkey(w), "rule": "union_0.82", "gap": float(g),
                            "change": float(g - table[("union_0.82", wkey(w))])})

        def loo_range(rule, w, drop="phrase"):
            v = [x["gap"] for x in loo if x["rule"] == rule and x["window"] == w and x["drop"] == drop]
            return [float(min(v)), float(max(v))]
        loo_summary = {f"{d}|{r}|{w}": loo_range(r, w, d) for w in wins for r, d in
                       (("exact", "phrase"), ("union_0.82", "phrase"), ("union_0.82", "exemplar"))}
        biggest = max((x for x in loo if x["drop"] == "phrase" and x["rule"] == "exact" and x["window"] == "all"),
                      key=lambda x: abs(x["change"]))
        out["categories"][c] = {"control": CONTROL[c], "max_abs_dev_from_specification_grid_json": float(max_ref_dev),
                                "joint_permutation": joint, "variance_decomposition": decomp, "cells": cells,
                                "leave_one_out": loo, "leave_one_out_range": loo_summary,
                                "largest_exact_all_phrase_influence": biggest}
        tex_rows.append([c.replace("_", "\\_"), f"{joint['median_gap']['observed']:+.3f}",
                         f"{joint['median_gap']['perm_p']:.4f}",
                         f"{int(joint['n_sig_positive']['observed'])}/12", f"{joint['n_sig_positive']['perm_p']:.4f}",
                         f"{decomp['rule_x_window']['share_rows']:.2f}", f"{decomp['rule_x_window']['share_cols']:.2f}",
                         f"{decomp['rule_x_window']['share_interaction']:.2f}",
                         "[{:+.3f}, {:+.3f}]".format(*loo_summary["phrase|exact|all"])])
        print(c, json.dumps(joint), flush=True)

    write_json(OUT / "spec_curve.json", out)
    write_tex(OUT / "spec_curve_joint.tex", ["Marker", "Median gap", "$p$", "Sig.\\ $+$", "$p$", "Rule SS",
                                             "Window SS", "Inter.\\ SS", "LOO exact, all"], tex_rows,
              "Joint permutation test, variance shares and leave-one-phrase-out range")
    write_tex(OUT / "tost_cells.tex", ["Marker", "Rule", "Window", "Gap", "90\\% CI", "Equiv.\\ 5pp", "Equiv.\\ 10pp"],
              tost_rows, "TOST equivalence per grid cell")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

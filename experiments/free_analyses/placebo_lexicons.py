"""Placebo lexicons and placebo seed sentences for the RQ1 gaps (CPU only).

Real data: the 700 stored final-7B responses (out/raw_cache/final_7b_answers.jsonl),
gap = personal-advice prompt mean minus the category's control prompt mean
(the estimator and controls of experiments/specification_grid.py).

Exact placebo: for each category, each real phrase is replaced by a random word
n-gram with the same number of whitespace words and a matched document
frequency (same log2 bin) in an independent reference corpus, the 6,240 pivot
generations (output/pivot/results). Phrase count per lexicon is preserved.
Candidates that contain, or are contained in, any real lexicon phrase are
excluded.

Embedding placebo: three random sentences (20 to 160 characters) drawn from
the pivot generations replace the category's three seed sentences; the union is
real exact OR placebo embedding at tau .82.

RQ3 placebo is not computed: the local DPO pair cache
(output/revision/dpo_conditional/pair_features_*.parquet) stores only the
precomputed flags of the real phrase groups, not the response text, so random
lexicons cannot be scored without rescanning the remote preference mixes.

    python -m experiments.free_analyses.placebo_lexicons
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from experiments.free_analyses.common import (CATEGORIES, CONTROL, EXEMPLARS, LEXICON, OUT_ROOT, SEED, WINDOWS,
                                              Embedder, group_means, load_final7b, load_pivot, max_sims, sentences,
                                              wkey, write_json, write_tex)

OUT = OUT_ROOT / "placebo_lexicons"
N_DRAW = 200
TAU = 0.82


def ngrams(text, nmax=5):
    toks = text.lower().split()
    out = set()
    for n in range(1, nmax + 1):
        for i in range(len(toks) - n + 1):
            out.add(" ".join(toks[i:i + n]))
    return out


def doc_freq(texts, phrases):
    low = [t.lower() for t in texts]
    return {p: sum(p in t for t in low) for p in phrases}


def fbin(f):
    return int(math.floor(math.log2(f + 1)))


class GapEstimator:
    def __init__(self, rows):
        pids = sorted({r["pid"] for r in rows})
        self.n = len(pids)
        pi = {p: i for i, p in enumerate(pids)}
        self.idx = np.array([pi[r["pid"]] for r in rows])
        ptag = {r["pid"]: r["tag"] for r in rows}
        self.ptag = np.array([ptag[p] for p in pids])

    def gap(self, y, cat):
        m = group_means(self.idx, y, self.n)
        return float(m[self.ptag == "advice"].mean() - m[self.ptag == CONTROL[cat]].mean())


def emp_p(null, real):
    null = np.asarray(null)
    return {"p_two_sided_abs": float((1 + np.sum(np.abs(null) >= abs(real) - 1e-12)) / (len(null) + 1)),
            "p_upper": float((1 + np.sum(null >= real - 1e-12)) / (len(null) + 1)),
            "p_lower": float((1 + np.sum(null <= real + 1e-12)) / (len(null) + 1)),
            "percentile": float(np.mean(null < real) * 100)}


def dist(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "q025": float(np.percentile(x, 2.5)),
            "q50": float(np.median(x)), "q975": float(np.percentile(x, 97.5))}


def main():
    rng = np.random.default_rng(SEED)
    rows = load_final7b()
    est = GapEstimator(rows)
    pivot = load_pivot()
    ref_texts = [r["text"] for rs in pivot.values() for r in rs]

    # candidate pool from the reference corpus
    df = Counter()
    for t in ref_texts:
        df.update(ngrams(t))
    real_all = [p for c in CATEGORIES for p in LEXICON[c]]
    real_df = doc_freq(ref_texts, real_all)
    pool = defaultdict(list)
    for g, f in df.items():
        if f < 1 or any(g in p or p in g for p in real_all):
            continue
        pool[(len(g.split()), fbin(f))].append(g)
    for k in pool:
        pool[k].sort()

    def match(p):
        n, b = len(p.split()), fbin(real_df[p])
        for d in (0, 1, -1, 2, -2):
            cand = pool.get((n, max(b + d, 0)))
            if cand:
                return cand[rng.integers(len(cand))], d
        raise RuntimeError(p)

    # real values
    emb = Embedder()
    ex_real = emb.exemplar_vectors(EXEMPLARS)
    wdata = {}
    for w in WINDOWS:
        texts, off, vec = emb.sentence_vectors("final7b", rows, w)
        wdata[w] = (texts, [t.lower() for t in texts], off, vec)

    def exact_y(lows, phrases):
        return np.array([any(p in t for p in phrases) for t in lows], float)

    real = {}
    for w in WINDOWS:
        _, lows, off, vec = wdata[w]
        for c in CATEGORIES:
            ex_y = exact_y(lows, LEXICON[c])
            em_y = (max_sims(off, vec, ex_real[c]) >= TAU).astype(float)
            real[(wkey(w), c)] = {"exact": est.gap(ex_y, c), "union_0.82": est.gap(np.maximum(ex_y, em_y), c),
                                  "embedding_only_0.82": est.gap(em_y, c)}

    # exact placebo
    null_exact = defaultdict(list)
    lexicons, bin_shift = [], Counter()
    for d in range(N_DRAW):
        lex = {}
        for c in CATEGORIES:
            chosen = []
            for p in LEXICON[c]:
                g, s = match(p)
                chosen.append(g)
                bin_shift[s] += 1
            lex[c] = chosen
        lexicons.append(lex)
        for w in WINDOWS:
            lows = wdata[w][1]
            for c in CATEGORIES:
                null_exact[(wkey(w), c)].append(est.gap(exact_y(lows, lex[c]), c))

    # embedding placebo
    sent_pool = sorted({s for t in ref_texts for s in sentences(t) if 20 <= len(s) <= 160})
    null_union, null_emb, null_delta = defaultdict(list), defaultdict(list), defaultdict(list)
    seed_sets = []
    for d in range(N_DRAW):
        picks = {c: [sent_pool[i] for i in rng.choice(len(sent_pool), 3, replace=False)] for c in CATEGORIES}
        seed_sets.append(picks)
        exv = emb.exemplar_vectors(picks)
        for w in WINDOWS:
            _, lows, off, vec = wdata[w]
            for c in CATEGORIES:
                ex_y = exact_y(lows, LEXICON[c])
                em_y = (max_sims(off, vec, exv[c]) >= TAU).astype(float)
                u = est.gap(np.maximum(ex_y, em_y), c)
                null_union[(wkey(w), c)].append(u)
                null_emb[(wkey(w), c)].append(est.gap(em_y, c))
                null_delta[(wkey(w), c)].append(real[(wkey(w), c)]["exact"] - u)
        if d % 20 == 0:
            print("embedding draw", d, flush=True)

    results = []
    for key in real:
        w, c = key
        rv = real[key]
        results.append({
            "window": w, "category": c, "control": CONTROL[c], "real": rv,
            "exact_placebo": {**dist(null_exact[key]), **emp_p(null_exact[key], rv["exact"])},
            "union_placebo_seeds": {**dist(null_union[key]), **emp_p(null_union[key], rv["union_0.82"])},
            "embedding_only_placebo_seeds": {**dist(null_emb[key]), **emp_p(null_emb[key], rv["embedding_only_0.82"])},
            "exact_minus_union_placebo_seeds": {**dist(null_delta[key]),
                                                **emp_p(null_delta[key], rv["exact"] - rv["union_0.82"])},
        })
    meta = {"n_draws": N_DRAW, "seed": SEED, "tau": TAU, "reference_corpus": "pivot generations, 13 tags, 6240 rows",
            "frequency_matching": "same whitespace word count and same floor(log2(df+1)) bin in reference corpus; "
                                  "fallback to adjacent bins counted in bin_shift",
            "bin_shift_counts": dict(bin_shift), "real_phrase_reference_df": real_df,
            "rq3": "not computed: local DPO cache has flags for real phrase groups only, no text"}
    write_json(OUT / "placebo_lexicons.json", {"meta": meta, "results": results,
                                               "example_lexicons": lexicons[:5], "example_seed_sets": seed_sets[:3]})
    rows_tex = []
    for r in results:
        if r["window"] != "all":
            continue
        e, u = r["exact_placebo"], r["union_placebo_seeds"]
        rows_tex.append([r["category"].replace("_", "\\_"), f"{r['real']['exact']:+.3f}",
                         f"[{e['q025']:+.3f}, {e['q975']:+.3f}]", f"{e['p_two_sided_abs']:.3f}",
                         f"{r['real']['union_0.82']:+.3f}", f"[{u['q025']:+.3f}, {u['q975']:+.3f}]",
                         f"{u['p_two_sided_abs']:.3f}"])
    write_tex(OUT / "placebo_all_text.tex", ["Marker", "Exact gap", "Placebo 95\\%", "$p$", "Union gap",
                                             "Placebo-seed 95\\%", "$p$"], rows_tex,
              "Real RQ1 gaps (all text) against 200 frequency-matched placebo lexicons and placebo seed sets")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

"""Specification grid over the 700 stored final-7B responses (CPU only).

For each marker, the advice-minus-control gap is computed under every
combination of matching rule (exact; exact-or-embedding union at tau .80, .82,
.84) and text window (first 64 tokens, first 128 tokens, all stored text):
twelve specifications. A second analysis asks whether the embedding rule's
firing rate grows with the number of sentences searched, by holding the number
of sentences fixed (first k sentences) and by regressing firing on sentence
count within condition.

    python -m experiments.specification_grid
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.detector_spec import (CATEGORIES, EXEMPLARS, EMBEDDING_MODEL_ID, WINDOW_TOKENIZER_ID,
                                       encode, exact_flags, sentences)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "out" / "raw_cache" / "final_7b_answers.jsonl"
OUT = ROOT / "out" / "results" / "specification_grid.json"
CONTROL = {"empathy_opener": "emotional", "validation": "emotional",
           "disclaimer": "domain_factual", "structure": "neutral_advice"}
TAUS = (0.80, 0.82, 0.84)
WINDOWS = (64, 128, None)
N_PERM = 10_000
FIRST_K = (3, 6)


def gap_and_p(rows, flags, cat, control, rng):
    """Advice minus control on prompt-level means; two-sided prompt permutation p."""
    per_prompt = defaultdict(list)
    for r, f in zip(rows, flags):
        if r["tag"] in ("advice", control):
            per_prompt[(r["tag"], r["i"])].append(float(f[cat]))
    keys = sorted(per_prompt)
    vals = np.array([np.mean(per_prompt[k]) for k in keys])
    is_adv = np.array([k[0] == "advice" for k in keys])
    obs = vals[is_adv].mean() - vals[~is_adv].mean()
    perm = np.empty(N_PERM)
    for b in range(N_PERM):
        lab = rng.permutation(is_adv)
        perm[b] = vals[lab].mean() - vals[~lab].mean()
    p = (1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (N_PERM + 1)
    return float(obs), float(p)


def main():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    rows = [json.loads(x) for x in RAW.read_text(encoding="utf-8").splitlines() if x.strip()]
    tok = AutoTokenizer.from_pretrained(WINDOW_TOKENIZER_ID)
    model = SentenceTransformer(EMBEDDING_MODEL_ID)
    ex = {c: encode(model, EXEMPLARS[c], "passage") for c in CATEGORIES}

    def window(text, w):
        if w is None:
            return text
        return tok.decode(tok.encode(text, add_special_tokens=False)[:w], skip_special_tokens=True)

    rng = np.random.default_rng(0)
    grid, sims_by_window = [], {}
    for w in WINDOWS:
        texts = [window(r["answer"], w) for r in rows]
        sents = [sentences(t) for t in texts]
        flat = [s for ss in sents for s in ss]
        vec = encode(model, flat, "query")
        sims, pos = [], 0
        for ss in sents:
            v = vec[pos:pos + len(ss)]
            pos += len(ss)
            sims.append({c: (v @ ex[c].T).max(1) if len(ss) else np.zeros(0) for c in CATEGORIES})
        sims_by_window[w] = (sents, sims)
        exact = [exact_flags(t) for t in texts]
        rules = {"exact": exact}
        for tau in TAUS:
            rules[f"union_{tau:.2f}"] = [{c: e[c] or bool((s[c] >= tau).any()) for c in CATEGORIES}
                                         for e, s in zip(exact, sims)]
        for rule, flags in rules.items():
            for c in CATEGORIES:
                gap, p = gap_and_p(rows, flags, c, CONTROL[c], rng)
                grid.append({"rule": rule, "window": w or "all", "category": c, "control": CONTROL[c],
                             "gap": gap, "p": p})

    summary = {}
    for c in CATEGORIES:
        g = [x["gap"] for x in grid if x["category"] == c]
        summary[c] = {"n_specs": len(g), "share_positive": float(np.mean(np.array(g) > 0)),
                      "min": float(min(g)), "max": float(max(g)),
                      "n_positive_p05": sum(x["gap"] > 0 and x["p"] < .05 for x in grid if x["category"] == c),
                      "n_negative_p05": sum(x["gap"] < 0 and x["p"] < .05 for x in grid if x["category"] == c)}

    # Opportunity: embedding-only firing (tau .82, all text) by sentences searched
    sents, sims = sims_by_window[None]
    opp = {}
    for c in CATEGORIES:
        per_cond = {}
        for cond in ("advice", CONTROL[c], "factual"):
            idx = [i for i, r in enumerate(rows) if r["tag"] == cond]
            n_sent = np.array([len(sents[i]) for i in idx])
            fire_all = np.array([(sims[i][c] >= .82).any() for i in idx], float)
            first = {k: float(np.mean([(sims[i][c][:k] >= .82).any() for i in idx])) for k in FIRST_K}
            per_cond[cond] = {"mean_sentences": float(n_sent.mean()), "rate_all": float(fire_all.mean()),
                              **{f"rate_first_{k}": v for k, v in first.items()}}
        # pooled within-condition slope of firing on sentence count (per 10 sentences)
        xs, ys = [], []
        for cond in ("advice", CONTROL[c], "factual"):
            idx = [i for i, r in enumerate(rows) if r["tag"] == cond]
            n = np.array([len(sents[i]) for i in idx], float)
            y = np.array([(sims[i][c] >= .82).any() for i in idx], float)
            xs.append(n - n.mean())
            ys.append(y - y.mean())
        x, y = np.concatenate(xs), np.concatenate(ys)
        slope = float((x @ y) / (x @ x)) * 10 if x @ x > 0 else 0.0
        bs = []
        brng = np.random.default_rng(1)
        for _ in range(2000):
            j = brng.integers(0, len(x), len(x))
            bs.append((x[j] @ y[j]) / (x[j] @ x[j]) * 10)
        opp[c] = {"conditions": per_cond, "slope_per_10_sentences": slope,
                  "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"grid": grid, "summary": summary, "opportunity": opp}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(json.dumps({c: {"slope": v["slope_per_10_sentences"], "ci": v["slope_ci95"]} for c, v in opp.items()}, indent=1))


if __name__ == "__main__":
    main()

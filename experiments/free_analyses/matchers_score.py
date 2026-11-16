"""Score every unique sentence (all windows of the 700 responses + 120 audit
sentences) with each alternative matcher and cache the scores.

    python -m experiments.free_analyses.matchers_score embed
    python -m experiments.free_analyses.matchers_score nli

Writes output/revision/free_analyses/matchers/cache/{sentences.json, scores_embed.npz, scores_nli.npz}.
"""
from __future__ import annotations

import json
import time

import numpy as np

from experiments.free_analyses.matchers_common import (CACHE, CATEGORIES, EMBED_MODELS, EXEMPLARS,
                                                       EXEMPLARS_10, NLI_HYPOTHESES, NLI_MODEL, SEED,
                                                       load_audit, load_rows, sentences, windowed_texts)


def main():
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else "embed"
    import torch
    torch.manual_seed(SEED)
    torch.set_num_threads(int(__import__("os").environ.get("MATCHERS_THREADS", "8")))
    from sentence_transformers import CrossEncoder, SentenceTransformer

    rows = load_rows()
    texts = windowed_texts(rows)
    audit_texts, _ = load_audit()
    uniq = sorted({s for ts in texts.values() for t in ts for s in sentences(t)} | set(audit_texts))
    print("unique sentences", len(uniq), flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "sentences.json").write_text(json.dumps(uniq), encoding="utf-8")
    (CACHE / "windowed_texts.json").write_text(json.dumps(texts), encoding="utf-8")

    scores = {}
    if part == "nli":
        return run_nli(uniq)
    for name, (mid, qpre, ppre) in EMBED_MODELS.items():
        t0 = time.time()
        m = SentenceTransformer(mid, device="cpu")
        enc = lambda xs, pre: m.encode([pre + x for x in xs], normalize_embeddings=True,
                                       batch_size=64, show_progress_bar=False)
        sv = enc(uniq, qpre)
        for exname, exs in (("ex3", EXEMPLARS), ("ex10", EXEMPLARS_10)):
            for c in CATEGORIES:
                scores[f"{name}_{exname}__{c}"] = (sv @ enc(exs[c], ppre).T).max(1).astype(np.float32)
        print(name, "done", round(time.time() - t0), "s", flush=True)
        np.savez_compressed(CACHE / "scores_embed.npz", **scores)


def run_nli(uniq):
    from sentence_transformers import CrossEncoder
    scores = {}

    t0 = time.time()
    ce = CrossEncoder(NLI_MODEL, device="cpu", max_length=128)
    labels = [k.lower() for k in ce.model.config.id2label.values()]
    ent = labels.index("entailment")
    print("nli labels", labels, flush=True)
    for c in CATEGORIES:
        best = np.zeros(len(uniq), np.float32)
        for h in NLI_HYPOTHESES[c]:
            logits = ce.predict([(s, h) for s in uniq], batch_size=64, show_progress_bar=False)
            pr = np.exp(logits - logits.max(1, keepdims=True))
            pr = pr / pr.sum(1, keepdims=True)
            best = np.maximum(best, pr[:, ent].astype(np.float32))
        scores[f"nli__{c}"] = best
        print("nli", c, round(time.time() - t0), "s", flush=True)
    np.savez_compressed(CACHE / "scores_nli.npz", **scores)
    print("saved", flush=True)


if __name__ == "__main__":
    main()

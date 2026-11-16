"""Shared CPU helpers for the free analyses.

Everything detector related is imported from experiments.detector_spec (lexicon,
exemplars, sentence splitter, exact matcher, e5 encoder call) and from
experiments.pivot.analyze_pivot (base-format cleaning, token windows). Nothing
here redefines a detector; it only caches sentence embeddings so that many
placebo exemplar sets can be scored without re-encoding.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from experiments.detector_spec import (CATEGORIES, EMBEDDING_MODEL_ID, EXEMPLARS, LEXICON,  # noqa: E402
                                       WINDOW_TOKENIZER_ID, encode, exact_flags, sentences)
from experiments.pivot.analyze_pivot import clean, windowed  # noqa: E402
from experiments.specification_grid import CONTROL, TAUS, WINDOWS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output" / "revision" / "free_analyses"
CACHE = OUT_ROOT / "_cache"
FINAL7B = ROOT / "out" / "raw_cache" / "final_7b_answers.jsonl"
PIVOT = ROOT / "output" / "pivot" / "results"
SEED = 20261114

__all__ = ["CATEGORIES", "EXEMPLARS", "LEXICON", "CONTROL", "TAUS", "WINDOWS", "exact_flags", "sentences",
           "clean", "windowed", "load_final7b", "load_pivot", "Embedder", "OUT_ROOT", "SEED", "wkey",
           "write_json", "write_tex", "group_means", "boot_ci", "max_sims"]


def wkey(w):
    return "all" if w is None else str(w)


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_final7b():
    rows = load_jsonl(FINAL7B)
    for r in rows:
        r["text"] = r["answer"]
        r["pid"] = f"{r['tag']}::{r['i']}"
    return rows


def load_pivot():
    out = {}
    for p in sorted(PIVOT.glob("gen_*.jsonl")):
        tag = p.stem[4:]
        rows = load_jsonl(p)
        for r in rows:
            r["text"] = clean(r)
        out[tag] = rows
    return out


class Embedder:
    """Lazy e5 model + OLMo tokenizer, with a disk cache of sentence vectors per (dataset, window)."""

    def __init__(self):
        self._model = self._tok = None
        CACHE.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL_ID, local_files_only=True, device="cpu")
        return self._model

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(WINDOW_TOKENIZER_ID, local_files_only=True)
        return self._tok

    def window_texts(self, rows, w):
        return [windowed(r["text"], w, self.tok) for r in rows]

    def sentence_vectors(self, name, rows, w):
        """Return (texts, offsets, vectors float32) for the rows under window w, cached."""
        path = CACHE / f"{name}_{wkey(w)}.npz"
        texts = self.window_texts(rows, w)
        if path.exists():
            z = np.load(path)
            return texts, z["offsets"], z["vec"].astype(np.float32)
        if w is not None and not (CACHE / f"{name}_all.npz").exists():
            self.sentence_vectors(name, rows, None)
        sents = [sentences(t) for t in texts]
        offsets = np.cumsum([0] + [len(s) for s in sents])
        flat = [s for ss in sents for s in ss]
        # sentence-level memo: prefix windows share most sentences with the full text
        memo_path = CACHE / f"{name}_sentmemo.npz"
        memo = {}
        if memo_path.exists():
            z = np.load(memo_path, allow_pickle=False)
            memo = dict(zip(z["keys"].tolist(), z["vec"].astype(np.float32)))
        todo = sorted({s for s in flat if s not in memo})
        if todo:
            for s, v in zip(todo, encode(self.model, todo, "query")):
                memo[s] = np.asarray(v, np.float32)
            keys = sorted(memo)
            np.savez_compressed(memo_path, keys=np.array(keys), vec=np.array([memo[k] for k in keys], np.float16))
        vec = np.array([memo[s] for s in flat], np.float32) if flat else np.zeros((0, 384), np.float32)
        np.savez_compressed(path, offsets=offsets, vec=vec.astype(np.float16))
        return texts, offsets, np.asarray(vec, np.float32)

    def exemplar_vectors(self, exemplars):
        return {c: np.asarray(encode(self.model, list(v), "passage"), np.float32) for c, v in exemplars.items()}


def max_sims(offsets, vec, ex):
    """Per response maximum cosine to the exemplar matrix ex (k x d)."""
    s = vec @ ex.T
    smax = s.max(1) if len(s) else np.zeros(0)
    out = np.zeros(len(offsets) - 1)
    for i in range(len(out)):
        a, b = offsets[i], offsets[i + 1]
        out[i] = smax[a:b].max() if b > a else 0.0
    return out


def group_means(idx, y, n):
    """Mean of y within integer groups idx (0..n-1)."""
    return np.bincount(idx, weights=np.asarray(y, float), minlength=n) / np.maximum(np.bincount(idx, minlength=n), 1)


def boot_ci(per_cluster, n_boot=10_000, seed=SEED, level=0.95):
    """Percentile bootstrap of the mean over clusters (rows of per_cluster; extra dims kept)."""
    x = np.asarray(per_cluster, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bs = x[idx].mean(1)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return np.percentile(bs, lo, axis=0), np.percentile(bs, hi, axis=0), bs


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def write_tex(path, header, rows, caption=None):
    cols = len(header)
    lines = ["% generated by experiments/free_analyses; fragment for \\input", "\\begin{tabular}{l" + "r" * (cols - 1) + "}",
             "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if caption:
        lines.insert(1, f"% {caption}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

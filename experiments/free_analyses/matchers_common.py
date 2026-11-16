"""Shared helpers for the alternative-matcher analyses (CPU only).

Reuses the frozen detector specification in experiments/detector_spec.py
(lexicon, e5 exemplars, sentence splitter, e5 prefixes) and the 700 stored
final-7B responses in out/raw_cache/final_7b_answers.jsonl.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.detector_spec import (CATEGORIES, EXEMPLARS, LEXICON, WINDOW_TOKENIZER_ID,
                                       exact_flags, sentences)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "out" / "raw_cache" / "final_7b_answers.jsonl"
OUT_DIR = ROOT / "output" / "revision" / "free_analyses" / "matchers"
CACHE = OUT_DIR / "cache"
AUDIT_ITEMS = ROOT / "out" / "results" / "validation_items_v3.csv"
AUDIT_LABELS = ROOT / "out" / "results" / "ai_consensus_annotations_v3.csv"
CONTROL = {"empathy_opener": "emotional", "validation": "emotional",
           "disclaimer": "domain_factual", "structure": "neutral_advice"}
WINDOWS = (64, 128, None)
SEED = 20261114

# (c) Ten exemplars per category, written from the definitions in
# out/paper/ANNOTATION_GUIDE.md (empathy_expression, validation,
# professional_referral + professional_disclaimer, structuring).
EXEMPLARS_10 = {
    "empathy_opener": [
        "I'm so sorry you're going through this.",
        "That sounds incredibly hard.",
        "I can only imagine how painful this is.",
        "It must be exhausting to deal with this every day.",
        "I'm really sorry that you are feeling so alone right now.",
        "That sounds like a very stressful situation to be in.",
        "I can hear how much this has been weighing on you.",
        "Losing someone close to you is heartbreaking, and I'm sorry for your loss.",
        "It sounds like you've been carrying a lot lately.",
        "I'm sorry things have been so difficult for you.",
    ],
    "validation": [
        "Your feelings are completely valid.",
        "It's totally normal to feel this way.",
        "There's nothing wrong with feeling upset.",
        "It makes sense that you would feel anxious about this.",
        "Anyone in your position would feel frustrated.",
        "Feeling overwhelmed in a situation like this is a natural reaction.",
        "It's okay to grieve and to take the time you need.",
        "Your reaction is understandable given what happened.",
        "Many people feel exactly the same way, and that is perfectly okay.",
        "You have every right to feel hurt by what they said.",
    ],
    "disclaimer": [
        "You should consult a qualified professional.",
        "Please talk to your doctor about this.",
        "I'm not a medical professional, but...",
        "Consider reaching out to a licensed therapist or counselor.",
        "A mental health professional can help you work through this.",
        "Please see a healthcare provider if your symptoms continue.",
        "I can't diagnose this, so it's best to get a professional opinion.",
        "This is not a substitute for professional medical advice.",
        "It may help to speak with a financial advisor before deciding.",
        "If you are in crisis, please contact a helpline or emergency services.",
    ],
    "structure": [
        "Here are a few steps you can take.",
        "Consider the following suggestions.",
        "First, try this; second, try that.",
        "Here are some strategies that might help:",
        "1. Set a regular schedule.",
        "Below are several options to consider.",
        "Step 1: Make a list of your priorities.",
        "There are three main things to keep in mind.",
        "**Communication:** Talk openly with your partner.",
        "Let's break this down into a few parts.",
    ],
}

# (a) NLI hypotheses per category (premise = sentence). Max entailment over the list.
NLI_HYPOTHESES = {
    "empathy_opener": ["The speaker expresses sympathy for the listener's feelings.",
                       "The speaker acknowledges that the listener is going through something hard."],
    "validation": ["The speaker says the listener's feelings are normal or valid.",
                   "The speaker reassures the listener that their emotional reaction is okay."],
    "disclaimer": ["The speaker recommends consulting a doctor, therapist or other professional.",
                   "The speaker says they are not a professional and cannot give professional advice."],
    "structure": ["The text introduces a list of steps or tips.",
                  "The text is an item in a numbered or bulleted list."],
}

EMBED_MODELS = {
    "e5": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge": ("BAAI/bge-small-en-v1.5", "", ""),
}
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def load_rows():
    return [json.loads(x) for x in RAW.read_text(encoding="utf-8").splitlines() if x.strip()]


def windowed_texts(rows):
    """{window: [text per row]} using the historical OLMo-2 tokenizer prefix rule."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(WINDOW_TOKENIZER_ID)
    out = {}
    for w in WINDOWS:
        if w is None:
            out["all"] = [r["answer"] for r in rows]
        else:
            out[str(w)] = [tok.decode(tok.encode(r["answer"], add_special_tokens=False)[:w],
                                      skip_special_tokens=True) for r in rows]
    return out


def load_audit():
    items = {r["item_id"]: r["text"] for r in csv.DictReader(AUDIT_ITEMS.open(encoding="utf-8"))}
    labels = {}
    for r in csv.DictReader(AUDIT_LABELS.open(encoding="utf-8")):
        labs = set(r["label"].split(",")) - {"none", ""}
        if "crisis_referral" in labs:
            labs = (labs - {"crisis_referral"}) | {"disclaimer"}
        labels[r["item_id"]] = labs
    ids = sorted(items)
    return [items[i] for i in ids], [{c: c in labels[i] for c in CATEGORIES} for i in ids]


def sentence_exact(s):
    return exact_flags(s)


def prompt_means(rows, flags, cat, control):
    per = defaultdict(list)
    for r, f in zip(rows, flags):
        if r["tag"] in ("advice", control):
            per[(r["tag"], r["i"])].append(float(f[cat]))
    keys = sorted(per)
    vals = np.array([np.mean(per[k]) for k in keys])
    is_adv = np.array([k[0] == "advice" for k in keys])
    return vals, is_adv


def gap_ci_p(rows, flags, cat, control, rng, n_boot=2000, n_perm=5000):
    vals, is_adv = prompt_means(rows, flags, cat, control)
    a, b = vals[is_adv], vals[~is_adv]
    obs = a.mean() - b.mean()
    bs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])
    perm = np.empty(n_perm)
    for k in range(n_perm):
        lab = rng.permutation(is_adv)
        perm[k] = vals[lab].mean() - vals[~lab].mean()
    p = (1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (n_perm + 1)
    return {"gap": float(obs), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p": float(p), "advice_rate": float(a.mean()), "control_rate": float(b.mean())}


__all__ = ["CATEGORIES", "EXEMPLARS", "LEXICON", "sentences", "exact_flags"]


def load_scores():
    out = {}
    for f in ("scores_embed.npz", "scores_nli.npz"):
        out.update(dict(np.load(CACHE / f)))
    return out

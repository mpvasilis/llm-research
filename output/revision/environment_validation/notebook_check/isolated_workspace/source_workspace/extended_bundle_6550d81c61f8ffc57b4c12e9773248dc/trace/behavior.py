"""Behavioral-move lexicon + pretraining source-domain bucketing.

The advice behaviors we trace are *moves*, not topics: empathy openers, emotional
validation, safety disclaimers, crisis referral, and structuring. The lexicon is
deliberately small and auditable (this is a feature: the whole pitch is a
human-inspectable selector). `bucket_source` maps an infini-gram Dolma document
path to a coarse web-source domain (forum vs curated) for the "where does the
advice come from" analysis.
"""
import re

# Behavioral-move lexicon. Lowercased surface forms; matched case-insensitively.
LEXICON = {
    "empathy_opener": [
        "i'm sorry to hear", "i am sorry to hear", "i'm really sorry",
        "that sounds really", "that sounds very", "i can understand",
        "i can imagine", "it's understandable", "that must be", "i'm here for you",
        "i hear you", "it sounds like you",
    ],
    "validation": [
        "your feelings are valid", "it's okay to feel", "it is okay to feel",
        "it's normal to feel", "it's completely normal", "what you're feeling",
        "there's nothing wrong with", "it's natural to",
    ],
    "disclaimer": [
        "consult a", "talk to a", "speak with a", "see a doctor", "seek medical",
        "i am not a doctor", "i'm not a doctor", "i'm not a medical",
        "medical professional", "healthcare professional", "mental health professional",
        "seek professional", "professional help",
    ],
    "crisis_referral": [
        "crisis", "hotline", "helpline", "988", "call 911", "emergency services",
        "suicide", "crisis line",
    ],
    "structure": [
        "here are some", "here are a few", "consider the following",
        "steps you can take", "a few things you can", "first,", "secondly,",
    ],
}
ALL_PHRASES = [(cat, p) for cat, ps in LEXICON.items() for p in ps]

_WORD = re.compile(r"[A-Za-z0-9']+")


def find_behaviors(text: str) -> list[dict]:
    """All lexicon occurrences in `text`: [{category, phrase, start, end}] (char spans)."""
    low = text.lower()
    hits = []
    for cat, p in ALL_PHRASES:
        i = low.find(p)
        while i != -1:
            hits.append({"category": cat, "phrase": p, "start": i, "end": i + len(p)})
            i = low.find(p, i + 1)
    return hits


def categories_present(text: str) -> set[str]:
    return {h["category"] for h in find_behaviors(text)}


def behavioral_coverage(text: str) -> float:
    """Fraction of word tokens that fall inside any behavioral-phrase match."""
    words = list(_WORD.finditer(text))
    if not words:
        return 0.0
    spans = [(h["start"], h["end"]) for h in find_behaviors(text)]
    covered = sum(1 for w in words if any(s <= w.start() < e for s, e in spans))
    return covered / len(words)


# --- pretraining source-domain bucketing ---------------------------------
_FORUM = {"reddit", "falcon", "cc_tail"}
_CURATED = {"cc_head", "c4", "wiki", "books", "academic"}


def bucket_source(src) -> str:
    """Map an infini-gram Dolma doc metadata (dict or str) to a coarse source bucket."""
    if isinstance(src, dict):
        p = (src.get("path") or src.get("url") or
             (src.get("metadata", {}) or {}).get("url") or "")
    else:
        p = str(src or "")
    p = p.lower()
    if "reddit" in p: return "reddit"
    if "cc_en_head" in p: return "cc_head"
    if "cc_en_middle" in p: return "cc_middle"
    if "cc_en_tail" in p: return "cc_tail"
    if "falcon" in p: return "falcon"
    if "c4" in p: return "c4"
    if "wiki" in p: return "wiki"
    if "gutenberg" in p or "books" in p: return "books"
    if "pes2o" in p or "s2orc" in p or "peS2o" in p or "paper" in p: return "academic"
    if "stack" in p or "starcoder" in p or "code" in p: return "code"
    if "math" in p: return "math"
    return "other"


def source_split(buckets: list[str]) -> dict:
    """Forum vs curated share over a list of buckets."""
    n = len(buckets) or 1
    forum = sum(1 for b in buckets if b in _FORUM)
    curated = sum(1 for b in buckets if b in _CURATED)
    return {
        "n": len(buckets),
        "forum_share": round(forum / n, 3),
        "curated_share": round(curated / n, 3),
        "by_bucket": {b: buckets.count(b) for b in sorted(set(buckets))},
    }

"""Extract salient phrases from an answer and attribute them to a corpus.

Method (same idea as infini-gram's attribution): walk the answer left to right;
at each position greedily extend an n-gram word by word as long as it still
occurs in the corpus (count > 0). The maximal still-matching span is a phrase
the model could have lifted verbatim from pretraining. Longer spans and rarer
counts = stronger, more distinctive provenance.
"""
import re
from .infinigram import count, passages_for

_WORD = re.compile(r"[A-Za-z0-9']+(?:[-/][A-Za-z0-9']+)*")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def longest_matches(index: str, text: str, min_words: int = 4,
                    max_words: int = 16, max_calls: int = 80) -> list[dict]:
    """Find maximal verbatim-matching spans of `text` in the corpus `index`."""
    words = tokenize(text)
    i, calls = 0, 0
    matches: list[dict] = []
    while i < len(words) and calls < max_calls:
        best = None
        j = i + min_words
        while j <= len(words) and (j - i) <= max_words and calls < max_calls:
            ngram = " ".join(words[i:j])
            calls += 1
            c = count(index, ngram).get("count", 0)
            if c and c > 0:
                best = {"phrase": ngram, "words": j - i, "count": c, "start": i}
                j += 1
            else:
                break
        if best:
            matches.append(best)
            i = best["start"] + best["words"]  # skip past matched span
        else:
            i += 1
    # most distinctive first: longer spans, then rarer
    matches.sort(key=lambda m: (m["words"], -m["count"]), reverse=True)
    return matches


def attribute(index: str, answer: str, top_passages: int = 5,
              max_calls: int = 80) -> dict:
    """Full pretraining attribution: matched spans + corpus passages for the top ones."""
    matches = longest_matches(index, answer, max_calls=max_calls)
    for m in matches[:top_passages]:
        m["passages"] = passages_for(index, m["phrase"], maxnum=2)
    return {
        "index": index,
        "matched_spans": matches,
        "n_matches": len(matches),
        "longest_span_words": matches[0]["words"] if matches else 0,
    }

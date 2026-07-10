"""Pretraining-corpus search via the infini-gram API.

infini-gram is an n-gram index over large pretraining corpora (Dolma, The Pile,
RedPajama, ...). We use it to ask: does this exact phrase from the answer occur
in the model's pretraining data, and how often? Long phrases with low but
nonzero counts are strong evidence the model saw that wording in pretraining.
"""
import json
from config import INFINIGRAM_API
from ._http import post_json


def _parse_source(raw):
    """infini-gram returns doc metadata as a JSON string; parse to dict."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            return {"raw": raw}
    return raw or {}


def count(index: str, query: str) -> dict:
    """Occurrence count of an exact phrase in the corpus."""
    return post_json(INFINIGRAM_API, {
        "index": index, "query_type": "count", "query": query,
    })


def search_docs(index: str, query: str, maxnum: int = 3, max_disp_len: int = 250) -> dict:
    """Return up to `maxnum` corpus documents containing the exact phrase."""
    return post_json(INFINIGRAM_API, {
        "index": index, "query_type": "search_docs", "query": query,
        "maxnum": maxnum, "max_disp_len": max_disp_len,
    })


def passages_for(index: str, query: str, maxnum: int = 2) -> list[dict]:
    """Normalize search_docs output into [{text, doc_meta}] passages."""
    res = search_docs(index, query, maxnum=maxnum)
    out = []
    docs = res.get("documents") or res.get("docs") or []
    for d in docs:
        spans = d.get("spans")
        if spans:  # spans = list of [text, relevance] segments
            text = "".join(s[0] if isinstance(s, (list, tuple)) else str(s) for s in spans)
        else:
            text = d.get("text", "")
        src = _parse_source(d.get("metadata") or d.get("doc_meta") or d.get("doc") or {})
        out.append({
            "text": text.strip()[:400],
            "source": src,
        })
    return out

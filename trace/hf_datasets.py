"""Instruction-tuning provenance via HuggingFace datasets-server full-text search.

Pretraining shapes *what words* the model knows; instruction tuning shapes *how
it responds* (gives advice, is empathetic, lists steps). We search the model's
SFT / preference datasets for examples on the same topic to show where the
response behavior was learned.
"""
import json
import re
from urllib.parse import quote
from config import DATASETS_SERVER, INSTRUCT_DATASETS
from ._http import get_json

_STOP = set("""a an the and or but if then of to in on at for with about into over
after is are was were be been being do does did have has had i you he she it we they
how what why when where who which this that these those my your our their can could
would should will my me your stop while during without my mine""".split())


def keywords(question: str, k: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z']+", question.lower())
    seen, out = set(), []
    for w in words:
        if len(w) > 3 and w not in _STOP and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return out


def _snippet(row: dict, limit: int = 300) -> str:
    """Flatten an arbitrary dataset row to a readable snippet."""
    val = row.get("messages") or row.get("conversations") or row.get("text") or row
    s = json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val
    s = re.sub(r"\s+", " ", s)
    return s[:limit]


def search_instruction(model: str, question: str, token: str = "", query: str = "") -> dict:
    # Reliability: we search the dataset's local parquet (via DuckDB), not HF's
    # flaky search index. A single distinctive term gives precise hits; `query`
    # is normally the rarest non-junk topical term picked by the pipeline.
    from .parquet_search import search as pq_search
    terms = keywords(question)
    if not query:
        query = max(terms, key=len) if terms else question[:40]
    results = [pq_search(ds, query) for ds, _c, _s, _h in INSTRUCT_DATASETS.get(model, [])]
    return {"query_terms": terms, "query_used": query, "datasets": results}

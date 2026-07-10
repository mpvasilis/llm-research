"""Reliable instruction-dataset search via local parquet + DuckDB.

HF's datasets-server search index is flaky (intermittent 500s). Instead we read
the dataset's stable auto-generated parquet (the refs/convert/parquet branch),
cache the shards locally once, and run substring search with DuckDB. After the
first download it's fully local: fast and never 500s.

Big datasets are capped to the first N shards (a labelled sample) so we don't
pull gigabytes.
"""
import json
import urllib.request
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
from config import ROOT, load_token

CACHE = ROOT / "out" / "cache"

# How to extract searchable text per dataset (column or SQL expression). The
# OLMo-specific Tulu-3 mixtures share the generic mixture's schema (id, messages,
# source[, dataset]) with messages a native STRUCT(content, role)[].
_TEXT_EXPR = {
    "OpenAssistant/oasst1": "text",
    "allenai/tulu-3-sft-mixture": "to_json(messages)",              # generic (legacy)
    "allenai/tulu-3-sft-olmo-2-mixture": "to_json(messages)",       # OLMo-2 7B line
    "allenai/tulu-3-sft-olmo-2-mixture-0225": "to_json(messages)",  # OLMo-2 0425-1B line
}
# Max parquet shards to scan (None = all). Caps download for big datasets.
_SHARD_CAP = {
    "allenai/tulu-3-sft-mixture": 1,   # 1 of 6 shards (~361MB) — sampled
}

# Role-scoped predicates: count a phrase only in ASSISTANT turns (the model's
# target behaviour) vs the whole-conversation "blob" (which inflates counts
# because user turns also match). Tulu `messages` is a native STRUCT(content,
# role)[]; oasst1 has a native `role` column (prompter/assistant).
_ASSISTANT_PRED = {
    "allenai/tulu-3-sft-mixture":
        "len(list_filter(messages, m -> m.role = 'assistant' AND m.content ILIKE $q)) > 0",
    "allenai/tulu-3-sft-olmo-2-mixture":
        "len(list_filter(messages, m -> m.role = 'assistant' AND m.content ILIKE $q)) > 0",
    "allenai/tulu-3-sft-olmo-2-mixture-0225":
        "len(list_filter(messages, m -> m.role = 'assistant' AND m.content ILIKE $q)) > 0",
    "OpenAssistant/oasst1": "role = 'assistant' AND text ILIKE $q",
}


@lru_cache(maxsize=1)
def _con():
    import duckdb
    c = duckdb.connect()
    c.execute("SET enable_progress_bar=false;")
    return c


def _shard_count(dataset: str, token: str) -> int:
    u = (f"https://huggingface.co/api/datasets/{dataset}/tree/"
         f"refs%2Fconvert%2Fparquet/default/train")
    req = urllib.request.Request(u, headers={"Authorization": f"Bearer {token}"})
    import json
    tree = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return sum(1 for f in tree if f["path"].endswith(".parquet"))


def _ensure_cached(dataset: str, token: str) -> tuple[list[Path], int, int]:
    """Download capped parquet shards locally; return (paths, n_used, n_total)."""
    total = _shard_count(dataset, token)
    cap = _SHARD_CAP.get(dataset)
    n = total if cap is None else min(cap, total)
    d = CACHE / dataset.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        fn = f"{i:04d}.parquet"
        p = d / fn
        if not p.exists():
            url = (f"https://huggingface.co/datasets/{dataset}/resolve/"
                   f"refs%2Fconvert%2Fparquet/default/train/{fn}?download=true")
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=600) as r, open(p, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
        paths.append(p)
    return paths, n, total


def search(dataset: str, query: str, limit: int = 3) -> dict:
    """Substring-search a dataset's parquet for `query`; reliable, local."""
    token = load_token()
    expr = _TEXT_EXPR.get(dataset, "text")
    try:
        paths, n_used, n_total = _ensure_cached(dataset, token)
        files = [str(p) for p in paths]
        con = _con()
        like = f"%{query}%"
        total = con.execute(
            f"SELECT count(*) FROM read_parquet($f) WHERE {expr} ILIKE $q",
            {"f": files, "q": like}).fetchone()[0]
        rows = con.execute(
            f"SELECT {expr} AS t FROM read_parquet($f) WHERE {expr} ILIKE $q LIMIT $n",
            {"f": files, "q": like, "n": limit}).fetchall()
        sampled = n_used < n_total
        return {
            "dataset": dataset,
            "query": query,
            "total_matches": total,
            "sampled": sampled,
            "scan_note": (f"sampled {n_used}/{n_total} shards" if sampled else "full dataset"),
            "viewer_url": viewer_url(dataset, query),
            "hits": [_to_hit(r[0]) for r in rows],
        }
    except Exception as e:  # noqa: BLE001
        return {"dataset": dataset, "query": query, "error": str(e),
                "viewer_url": viewer_url(dataset, query), "hits": []}


def count_roles(dataset: str, query: str) -> dict:
    """Count exact-substring matches for `query` in the whole conversation (blob)
    vs ASSISTANT turns only. The gap is the role-scoping inflation artifact:
    phrases counted in the blob include user turns that didn't come from the model.
    """
    token = load_token()
    blob_expr = _TEXT_EXPR.get(dataset, "text")
    asst_pred = _ASSISTANT_PRED.get(dataset)
    try:
        paths, n_used, n_total = _ensure_cached(dataset, token)
        files = [str(p) for p in paths]
        con = _con()
        like = f"%{query}%"
        blob = con.execute(
            f"SELECT count(*) FROM read_parquet($f) WHERE {blob_expr} ILIKE $q",
            {"f": files, "q": like}).fetchone()[0]
        if asst_pred:
            assistant = con.execute(
                f"SELECT count(*) FROM read_parquet($f) WHERE {asst_pred}",
                {"f": files, "q": like}).fetchone()[0]
        else:
            assistant = None
        return {
            "dataset": dataset, "query": query,
            "blob_matches": blob, "assistant_matches": assistant,
            "inflation": (round(blob / assistant, 2) if assistant else None),
            "scan_note": (f"sampled {n_used}/{n_total} shards" if n_used < n_total else "full dataset"),
        }
    except Exception as e:  # noqa: BLE001
        return {"dataset": dataset, "query": query, "error": str(e)}


def viewer_url(dataset: str, query: str) -> str:
    """Link straight to the HF dataset viewer, full-text-searched for the query."""
    return (f"https://huggingface.co/datasets/{dataset}/viewer/default/train"
            f"?q={quote(query)}")


def _to_hit(raw: str) -> dict:
    """Turn a matched row into {snippet, turns|full}. tulu rows are message
    lists (multi-turn prompt+response); oasst1 rows are a single text."""
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:  # noqa: BLE001
        pass
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        turns = [{"role": m.get("role", ""), "content": str(m.get("content", ""))}
                 for m in parsed]
        snippet = _clean(" / ".join(t["content"] for t in turns))
        return {"snippet": snippet, "turns": turns}
    return {"snippet": _clean(raw), "full": _clean(raw, 6000)}


def _clean(s: str, limit: int = 300) -> str:
    import re
    return re.sub(r"\s+", " ", str(s)).strip()[:limit]

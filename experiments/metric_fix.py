"""Length-matched SFT recoverability (fixes the confound in the phase-1 chart).

The original metric compared behavioral *phrases* (multi-word) to topical *words*
(single-word). Longer strings match fewer docs by construction, so the gap was a
length artifact, not provenance. Fix: for each behavioral phrase of length L
words, compare it to topical (non-behavioral) n-grams of the SAME length L drawn
from the answer body. Report per-million assistant prevalence within each length
bucket — an apples-to-apples behavioral-vs-topical comparison.

Run:  python -m experiments.metric_fix
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import duckdb
from config import ROOT
from trace.parquet_search import count_roles
from trace import behavior as B

# NOTE (2026-07-04): the released OLMo-2 models trained on model-SPECIFIC mixtures
# (config.SFT_BY_MODEL): tulu-3-sft-olmo-2-mixture (7B) / -0225 (1B). oasst1 is
# already an internal subset of each mixture, so it must NOT be added separately
# (double-count). This local helper still points at whatever is cached in
# out/cache/; refresh the cache to the correct per-model mixture before trusting
# its numbers. The paper's counts come from the Colab notebook (build_notebook.py),
# which now uses the correct per-model mixtures.
SFT = ["allenai/tulu-3-sft-mixture", "OpenAssistant/oasst1"]  # legacy proxy cache
_con = duckdb.connect(); _con.execute("SET enable_progress_bar=false")
_NDOCS = {}


def ndocs(ds: str) -> int:
    if ds not in _NDOCS:
        files = glob.glob(str(ROOT / "out" / "cache" / ds.replace("/", "__") / "*.parquet"))
        _NDOCS[ds] = _con.execute("SELECT count(*) FROM read_parquet($f)", {"f": files}).fetchone()[0]
    return _NDOCS[ds]
ANS = ROOT / "out" / "derisk" / "answers"
_WORD = re.compile(r"[A-Za-z0-9']+")
_STOP = set("the a an and or but if then of to in on at for with about into over after is are "
            "was were be been being do does did have has had you your my me it this that these "
            "those can could would should will what how why when".split())
_cache = {}


def permil(phrase: str) -> float:
    if phrase in _cache:
        return _cache[phrase]
    tot = 0.0
    for ds in SFT:
        r = count_roles(ds, phrase)
        a = r.get("assistant_matches") or 0
        tot += 1e6 * a / max(1, ndocs(ds))
    _cache[phrase] = round(tot, 2)
    return _cache[phrase]


def body_ngrams(answer: str, length: int, beh_spans, k: int = 3) -> list[str]:
    """Non-behavioral content n-grams of exactly `length` words from the body."""
    toks = list(_WORD.finditer(answer))
    out, seen = [], set()
    for i in range(len(toks) - length + 1):
        gram_toks = toks[i:i + length]
        s, e = gram_toks[0].start(), gram_toks[-1].end()
        if any(bs <= s < be or bs < e <= be for bs, be in beh_spans):
            continue  # overlaps a behavioral phrase
        words = [t.group().lower() for t in gram_toks]
        if all(w in _STOP for w in words):
            continue
        g = " ".join(words)
        if g in seen:
            continue
        seen.add(g)
        out.append(answer[s:e])
        if len(out) >= k:
            break
    return out


def main():
    files = sorted(ANS.glob("advice_*.json"))
    if not files:
        raise SystemExit("No cached advice answers in out/derisk/answers — run experiments.derisk first.")
    by_len = defaultdict(lambda: {"behavioral": [], "topical": []})
    for f in files:
        ans = json.loads(f.read_text(encoding="utf-8"))["answer"]
        beh = B.find_behaviors(ans)
        beh_spans = [(h["start"], h["end"]) for h in beh]
        beh_phrases = {h["phrase"] for h in beh}
        for p in beh_phrases:
            L = len(p.split())
            by_len[L]["behavioral"].append(permil(p))
        # length-matched topical controls for each behavioral length present
        for L in {len(p.split()) for p in beh_phrases}:
            for g in body_ngrams(ans, L, beh_spans):
                by_len[L]["topical"].append(permil(g))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 1) if xs else None

    print("Length-matched SFT-assistant recoverability (per-million docs):\n")
    print(f"{'len':>4} | {'behavioral':>12} | {'topical(ctrl)':>14} | {'n_beh':>5} {'n_top':>5}")
    print("-" * 52)
    rows = []
    for L in sorted(by_len):
        b = mean(by_len[L]["behavioral"]); t = mean(by_len[L]["topical"])
        nb, nt = len(by_len[L]["behavioral"]), len(by_len[L]["topical"])
        print(f"{L:>4} | {str(b):>12} | {str(t):>14} | {nb:>5} {nt:>5}")
        rows.append({"length": L, "behavioral_permil": b, "topical_permil": t, "n_beh": nb, "n_top": nt})
    out = ROOT / "out" / "derisk" / "recoverability_length_matched.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # verdict
    pooled_b = mean([v for L in by_len for v in by_len[L]["behavioral"]])
    pooled_t = mean([v for L in by_len for v in by_len[L]["topical"]])
    print(f"\npooled (NOT length-matched, for reference): behavioral {pooled_b} vs topical {pooled_t}")
    print("Per-length rows are the honest comparison. Wrote", out)


if __name__ == "__main__":
    main()

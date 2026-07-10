"""EXACT SFT recount against the real OLMo-2 model-specific mixtures.

The paper's SFT counts were computed against the generic tulu-3-sft-mixture +
oasst1 (a proxy, with oasst1 double-counted). This recomputes the role-scoped
assistant-only counts against the corpora the released models were ACTUALLY
trained on -- tulu-3-sft-olmo-2-mixture (7B line) and -0225 (1B line), with no
separate oasst1 -- so the role-scoping table and the distinct-phrase 3-word
leave-one-out become exact and per-model.

Downloads ~2.7 GB of parquet once (cached in out/cache/), then all search is
local DuckDB. Corpus-search only: needs no model outputs.

Run:  python -m experiments.recount_sft_olmo2
"""
import glob
import json
from pathlib import Path

import duckdb
from config import ROOT, load_token
from trace.parquet_search import count_roles, _ensure_cached

MIX = {
    "1B": "allenai/tulu-3-sft-olmo-2-mixture-0225",
    "7B": "allenai/tulu-3-sft-olmo-2-mixture",
}
_con = duckdb.connect(); _con.execute("SET enable_progress_bar=false;")


def phrases_from_summary():
    s = json.loads((ROOT / "out" / "results" / "summary.json").read_text())
    out = {}
    for m in ("1B", "7B"):
        out[m] = [r["phrase"] for r in s["per_model"][m]["role_scoping"]]
    return out


def ndocs(dataset: str) -> int:
    files = glob.glob(str(ROOT / "out" / "cache" / dataset.replace("/", "__") / "*.parquet"))
    return _con.execute("SELECT count(*) FROM read_parquet($f)", {"f": files}).fetchone()[0]


def main():
    token = load_token()
    surfaced = phrases_from_summary()
    result = {"mixtures": MIX, "per_model": {}}

    for model, ds in MIX.items():
        print(f"\n=== {model}: {ds} ===")
        print("  ensuring parquet cached (downloads once) ...")
        _ensure_cached(ds, token)
        nd = ndocs(ds)
        print(f"  NDOCS (rows/conversations) = {nd:,}")
        rows = []
        for p in surfaced[model]:
            r = count_roles(ds, p)
            a = r.get("assistant_matches")
            b = r.get("blob_matches")
            permil = round(1e6 * a / nd, 1) if a is not None else None
            infl = round(b / a, 2) if a else None
            rows.append({"phrase": p, "len": len(p.split()), "assistant": a,
                         "blob": b, "permil": permil, "inflation": infl})
            print(f"    {p:<28} asst={a} blob={b} /M={permil} infl={infl}")

        # distinct-phrase 3-word leave-one-out (drop the single dominant marker)
        three = [r for r in rows if r["len"] == 3 and r["assistant"] is not None]
        three_sorted = sorted(three, key=lambda r: -r["assistant"])
        mean_all = round(sum(r["permil"] for r in three) / len(three), 1) if three else None
        drop = three_sorted[0]["phrase"] if three_sorted else None
        rest = [r for r in three if r["phrase"] != drop]
        mean_drop = round(sum(r["permil"] for r in rest) / len(rest), 1) if rest else None
        loo = {"n_distinct_3word": len(three), "dropped": drop,
               "mean_permil_all": mean_all, "mean_permil_drop_top": mean_drop}
        print(f"  3-word distinct leave-one-out: {len(three)} types, "
              f"mean {mean_all} -> {mean_drop} (drop '{drop}')")
        result["per_model"][model] = {"dataset": ds, "ndocs": nd,
                                      "phrases": rows, "leave_one_out_3word": loo}

    dest = ROOT / "out" / "results" / "sft_recount_olmo2.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nWrote", dest)


if __name__ == "__main__":
    main()

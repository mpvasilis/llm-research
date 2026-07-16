"""EXACT SFT recount against the real OLMo-2 model-specific mixtures.

The paper's SFT counts were computed against the generic tulu-3-sft-mixture +
oasst1 (a proxy, with oasst1 double-counted). This recomputes the role-scoped
assistant-only counts against the corpora the released models were ACTUALLY
trained on -- tulu-3-sft-olmo-2-mixture (7B line) and -0225 (1B line), with no
separate oasst1 -- so the role-scoping table and the distinct-phrase 3-word
leave-one-out become exact and per-model.

By default this materializes the exact counts already produced by the canonical
Colab DuckDB scan in ``summary_v3.json`` so the phrase inventory cannot drift.
Pass ``--verify-corpus`` for an independent local parquet recount. That check
uses the ~2.7 GB cache and can be slow on nested message columns.

Run:  python -m experiments.recount_sft_olmo2
"""
import argparse
import json

import duckdb
from config import ROOT, load_token
from trace.parquet_search import _ensure_cached

MIX = {
    "1B": "allenai/tulu-3-sft-olmo-2-mixture-0225",
    "7B": "allenai/tulu-3-sft-olmo-2-mixture",
}
N_DOCS = {"1B": 866_138, "7B": 939_344}
_con = duckdb.connect(); _con.execute("SET enable_progress_bar=false;")


def load_summary():
    # summary.json is a frozen legacy artifact. The surfaced phrase inventory
    # must follow the current canonical generation run.
    return json.loads((ROOT / "out" / "results" / "summary_v3.json").read_text())


def count_roles_batch(paths, phrases):
    """Count every phrase in one parquet scan instead of two scans per phrase."""
    columns = ["count(*) AS ndocs"]
    params = {"f": [str(path) for path in paths]}
    for index, phrase in enumerate(phrases):
        key = f"q{index}"
        params[key] = f"%{phrase}%"
        columns.append(
            f"sum(CASE WHEN to_json(messages) ILIKE ${key} THEN 1 ELSE 0 END)"
        )
        columns.append(
            "sum(CASE WHEN len(list_filter(messages, m -> "
            f"m.role = 'assistant' AND m.content ILIKE ${key})) > 0 "
            "THEN 1 ELSE 0 END)"
        )
    row = _con.execute(
        f"SELECT {', '.join(columns)} FROM read_parquet($f)", params
    ).fetchone()
    counts = {
        phrase: {"blob": int(row[1 + 2 * index]), "assistant": int(row[2 + 2 * index])}
        for index, phrase in enumerate(phrases)
    }
    return int(row[0]), counts


def main(verify_corpus=False):
    summary = load_summary()
    surfaced_rows = {
        model: summary["per_model"][model]["role_scoping"]
        for model in ("1B", "7B")
    }
    result = {
        "source": "summary_v3.json canonical Colab DuckDB counts",
        "independent_corpus_verification": bool(verify_corpus),
        "mixtures": MIX,
        "per_model": {},
    }
    token = load_token() if verify_corpus else None

    for model, ds in MIX.items():
        print(f"\n=== {model}: {ds} ===")
        phrases = [row["phrase"] for row in surfaced_rows[model]]
        if verify_corpus:
            print("  ensuring parquet cached (downloads once) ...")
            paths, _, _ = _ensure_cached(ds, token)
            nd, counts = count_roles_batch(paths, phrases)
        else:
            nd = N_DOCS[model]
            counts = {
                row["phrase"]: {"assistant": row["assistant"], "blob": row["blob"]}
                for row in surfaced_rows[model]
            }
        print(f"  NDOCS (rows/conversations) = {nd:,}")
        rows = []
        for p in phrases:
            a = counts[p]["assistant"]
            b = counts[p]["blob"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-corpus", action="store_true")
    args = parser.parse_args()
    main(verify_corpus=args.verify_corpus)

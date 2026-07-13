"""Deterministic derived tables for the paper, from the canonical summary artifact:
  (1) Wilson 95% CIs for every emission cell (counts = rate * N).
  (2) Threshold-sweep table (advice) at 0.80/0.82/0.84.
  (3) v2-only stagewise and 5-condition emission tables when available.
  (4) v2-only prompt-clustered permutation tests when available.
  (5) Per-phrase appendix: role_scoping assistant counts + per-million over each
      model's exact SFT assistant-document count.
  (6) Distinct-phrase leave-one-out at 3 words (drop the top generic phrase).
Writes out/results/polish_tables.md and .json. No network, no parquet download.
Prefers out/results/summary_v2.json when present, falling back to summary.json.

Run:  python -m experiments.polish_tables
"""
import json
import math

from config import ROOT
from experiments.summary_adapter import (
    CATS,
    condition_n,
    conditions,
    emission_by_threshold,
    emission_rates,
    load_sft_ndocs,
    load_summary,
    stagewise_emission,
)

S, SUMMARY_PATH = load_summary()
NDOCS = load_sft_ndocs()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (round(p, 3), round(max(0, c - h), 3), round(min(1, c + h), 3))


def fmt_cell(x):
    if x is None:
        return "NA"
    if isinstance(x, float) and not math.isfinite(x):
        return "NA"
    return str(x)


def emission_table():
    L = ["### Emission rates: counts and Wilson 95% CIs (primary threshold 0.82)\n",
         "| model | behavior | advice k/n (rate, CI) | factual k/n (rate, CI) |",
         "|---|---|---|---|"]
    js = {}
    for m, blk in S["per_model"].items():
        na, nf = condition_n(S, m, "advice"), condition_n(S, m, "factual"); js[m] = {}
        er = emission_rates(S, m)
        for c in CATS:
            v = er[c]
            ka, kf = round(v["advice"] * na), round(v["factual"] * nf)
            pa, pf = wilson(ka, na), wilson(kf, nf)
            js[m][c] = {"advice": {"k": ka, "n": na, "ci": [pa[1], pa[2]]},
                        "factual": {"k": kf, "n": nf, "ci": [pf[1], pf[2]]}}
            L.append(f"| {m} | {c} | {ka}/{na} ({pa[0]}, [{pa[1]},{pa[2]}]) | "
                     f"{kf}/{nf} ({pf[0]}, [{pf[1]},{pf[2]}]) |")
    return "\n".join(L), js


def threshold_table():
    L = ["\n### Threshold sensitivity: advice emission at e5 thresholds 0.80 / 0.82 / 0.84\n",
         "| model | behavior | 0.80 | 0.82 | 0.84 |", "|---|---|---|---|---|"]
    for m, blk in S["per_model"].items():
        sw = emission_by_threshold(S, m)
        for c in CATS:
            L.append(f"| {m} | {c} | {sw['0.8'][c]['advice']} | {sw['0.82'][c]['advice']} | {sw['0.84'][c]['advice']} |")
    return "\n".join(L)


def stagewise_table():
    if not any(stagewise_emission(S, m) for m in S["per_model"]):
        return "", {}
    L = ["\n### Stagewise emergence: advice emission at primary threshold 0.82\n",
         "| model | stage | empathy | validation | disclaimer | crisis_referral | structure |",
         "|---|---|---|---|---|---|---|"]
    js = {}
    for m in S["per_model"]:
        js[m] = {}
        for stage, block in stagewise_emission(S, m).items():
            js[m][stage] = {c: block[c]["advice"] for c in CATS}
            L.append(f"| {m} | {stage} | " + " | ".join(fmt_cell(block[c]["advice"]) for c in CATS) + " |")
    return "\n".join(L), js


def condition_table():
    conds = conditions(S)
    if conds == ["advice", "factual"]:
        return "", {}
    L = ["\n### Final model emission by condition at primary threshold 0.82\n",
         "| model | behavior | " + " | ".join(conds) + " |",
         "|---|---|" + "|".join("---" for _ in conds) + "|"]
    js = {}
    for m in S["per_model"]:
        er = emission_rates(S, m); js[m] = {}
        for c in CATS:
            js[m][c] = {cond: er[c].get(cond) for cond in conds}
            L.append(f"| {m} | {c} | " + " | ".join(fmt_cell(er[c].get(cond)) for cond in conds) + " |")
    return "\n".join(L), js


def tests_table():
    if not any(S["per_model"][m].get("tests") for m in S["per_model"]):
        return "", {}
    L = ["\n### Prompt-clustered permutation tests: advice vs control conditions\n",
         "| model | behavior | control | diff | perm_p |",
         "|---|---|---|---|---|"]
    js = {}
    for m, blk in S["per_model"].items():
        js[m] = {}
        for c, tests in blk.get("tests", {}).items():
            js[m][c] = tests
            for control, values in tests.items():
                diff = values.get("diff")
                perm_p = values.get("perm_p") if fmt_cell(diff) != "NA" else None
                L.append(f"| {m} | {c} | {control} | {fmt_cell(diff)} | {fmt_cell(perm_p)} |")
    return "\n".join(L), js


def perphrase_and_loo():
    L = ["\n### Per-phrase SFT-assistant recoverability (role-scoped)\n",
         "Per-million uses each model's exact SFT assistant-document count when available. "
         "`infl` = whole-conversation (blob) count / assistant-only count.\n",
         "| model | words | phrase | assistant | per-million | infl |",
         "|---|---|---|---|---|---|"]
    loo = {}
    for m, blk in S["per_model"].items():
        ndocs = NDOCS.get(m)
        rows = []
        for r in blk["role_scoping"]:
            w = len(r["phrase"].split())
            pm = round(1e6 * r["assistant"] / ndocs, 1) if ndocs else r.get("per_million")
            rows.append({"words": w, "phrase": r["phrase"], "assistant": r["assistant"],
                         "permil": pm, "infl": r["inflation"]})
        rows.sort(key=lambda x: (x["words"], -x["assistant"]))
        for x in rows:
            L.append(f"| {m} | {x['words']} | `{x['phrase']}` | {x['assistant']} | {x['permil']} | {x['infl']} |")
        # distinct-phrase leave-one-out at 3 words
        three = [x for x in rows if x["words"] == 3]
        if three:
            dom = max(three, key=lambda x: x["assistant"])
            full = sum(x["permil"] for x in three) / len(three)
            rest = [x for x in three if x["phrase"] != dom["phrase"]]
            wo = sum(x["permil"] for x in rest) / len(rest) if rest else 0.0
            loo[m] = {"n_distinct_3word": len(three), "dominant": dom["phrase"],
                      "mean_permil_with": round(full, 1), "mean_permil_without": round(wo, 1)}
    L.append("\n**Distinct-phrase leave-one-out (3-word behavioral phrases, per-model-denominator per-million):**")
    for m, d in loo.items():
        L.append(f"- {m}: over {d['n_distinct_3word']} distinct 3-word phrases, dropping the top generic "
                 f"phrase `{d['dominant']}` lowers the mean from {d['mean_permil_with']} to "
                 f"{d['mean_permil_without']} per-million.")
    return "\n".join(L), loo


def main():
    et, ej = emission_table()
    tt = threshold_table()
    st, sj = stagewise_table()
    ct, cj = condition_table()
    tst, tsj = tests_table()
    pt, loo = perphrase_and_loo()
    portable_source = SUMMARY_PATH.resolve().relative_to(ROOT.resolve()).as_posix()
    parts = [f"Source: `{portable_source}`\n", et, tt, st, ct, tst, pt]
    md = "\n".join(p for p in parts if p) + "\n"
    out = ROOT / "out" / "results"
    (out / "polish_tables.md").write_text(md, encoding="utf-8")
    (out / "polish_tables.json").write_text(json.dumps(
        {"summary_source": SUMMARY_PATH.resolve().relative_to(ROOT.resolve()).as_posix(), "ndocs": NDOCS, "emission_ci": ej,
         "stagewise_advice": sj, "condition_emission": cj, "prompt_clustered_tests": tsj,
         "leave_one_out_3word": loo},
        indent=2), encoding="utf-8")
    print(md)
    print("\nWrote", out / "polish_tables.md")


if __name__ == "__main__":
    main()

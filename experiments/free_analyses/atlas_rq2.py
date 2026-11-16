"""Atlas RQ2: stagewise marker contrasts on the pivot generations (CPU only).

Data: output/pivot/results/gen_*.jsonl (13 checkpoint/format tags, 40 topic
families x {neutral, emotional} x {information, advice} x 3 seeds = 480 rows).
Base-format (qa) text is cut at the first generated "\\nQuestion:" exactly as
experiments/pivot/analyze_pivot.clean does. Rules and windows are the paper's
12-cell grid (exact; union at tau .80/.82/.84; 64 tokens, 128 tokens, all).

Contrasts per family (cluster = topic family, 40 clusters):
  advice_vs_information   mean(advice cells) - mean(information cells)   (analyze_pivot.gap, closest to the
                          paper's personal-advice minus general-facts gap)
  emotional_advice_vs_emotional_information  advice vs the emotion-mention, no-request control
  neutral_advice_vs_neutral_information
Stage changes are paired over families; 95% percentile bootstrap (10k), plus
TOST at +-5 pp (90% bootstrap interval inside the margin).

    python -m experiments.free_analyses.atlas_rq2
"""
from __future__ import annotations

import numpy as np

from experiments.free_analyses.common import (CATEGORIES, EXEMPLARS, OUT_ROOT, SEED, TAUS, WINDOWS, Embedder,
                                              boot_ci, exact_flags, load_pivot, max_sims, wkey, write_json,
                                              write_tex)

OUT = OUT_ROOT / "atlas_rq2"
CHAINS = {"1B": ["base", "sft", "dpo", "rlvr", "instruct"], "7B": ["base", "sft", "dpo", "instruct"]}
CONTRASTS = ("advice_vs_information", "emotional_advice_vs_emotional_information",
             "neutral_advice_vs_neutral_information")
MARGIN = 0.05
N_BOOT = 10_000


def family_contrasts(rows, y):
    fams = sorted({r["family"] for r in rows})
    fi = {f: i for i, f in enumerate(fams)}
    cell = np.zeros((len(fams), 2, 2))
    cnt = np.zeros((len(fams), 2, 2))
    for r, v in zip(rows, y):
        k = (fi[r["family"]], int(r["emotional"]), int(r["request"] == "advice"))
        cell[k] += v
        cnt[k] += 1
    m = cell / np.maximum(cnt, 1)
    return fams, {
        "advice_vs_information": m[:, :, 1].mean(1) - m[:, :, 0].mean(1),
        "emotional_advice_vs_emotional_information": m[:, 1, 1] - m[:, 1, 0],
        "neutral_advice_vs_neutral_information": m[:, 0, 1] - m[:, 0, 0],
    }


def summarise(x, seed):
    lo, hi, bs = boot_ci(x, N_BOOT, seed)
    lo90, hi90, _ = boot_ci(x, N_BOOT, seed, level=0.90)
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / N_BOOT)
    return {"estimate": float(np.mean(x)), "ci95": [float(lo), float(hi)], "ci90": [float(lo90), float(hi90)],
            "boot_p": float(p), "tost_equivalent_5pp": bool(lo90 > -MARGIN and hi90 < MARGIN)}


def main():
    data = load_pivot()
    emb = Embedder()
    ex = emb.exemplar_vectors(EXEMPLARS)
    # per tag -> spec -> category -> contrast -> per-family vector
    fam = {}
    rates = {}
    fams_ref = None
    for tag, rows in data.items():
        fam[tag] = {}
        for w in WINDOWS:
            texts, off, vec = emb.sentence_vectors(f"pivot_{tag}", rows, w)
            exact = [exact_flags(t) for t in texts]
            sims = {c: max_sims(off, vec, ex[c]) for c in CATEGORIES}
            rules = {"exact": {c: np.array([e[c] for e in exact], float) for c in CATEGORIES}}
            for tau in TAUS:
                rules[f"union_{tau:.2f}"] = {c: np.maximum(rules["exact"][c], sims[c] >= tau) for c in CATEGORIES}
            for rule, flags in rules.items():
                spec = f"{rule}|{wkey(w)}"
                fam[tag][spec] = {}
                for c in CATEGORIES:
                    fams, con = family_contrasts(rows, flags[c])
                    fams_ref = fams_ref or fams
                    assert fams == fams_ref
                    fam[tag][spec][c] = con
                    adv = np.array([r["request"] == "advice" for r in rows])
                    rates.setdefault(tag, {}).setdefault(spec, {})[c] = {
                        "advice": float(flags[c][adv].mean()), "information": float(flags[c][~adv].mean())}
        print("scored", tag, flush=True)

    specs = list(next(iter(fam.values())))
    levels = []
    for tag in fam:
        for spec in specs:
            for c in CATEGORIES:
                for k in CONTRASTS:
                    levels.append({"tag": tag, "spec": spec, "category": c, "contrast": k,
                                   **summarise(fam[tag][spec][c][k], SEED)})
    changes = []
    for size, chain in CHAINS.items():
        for a, b in zip(chain, chain[1:]):
            ta, tb = f"{size}_{a}_chat", f"{size}_{b}_chat"
            for spec in specs:
                for c in CATEGORIES:
                    for k in CONTRASTS:
                        d = fam[tb][spec][c][k] - fam[ta][spec][c][k]
                        changes.append({"size": size, "from": a, "to": b, "spec": spec, "category": c,
                                        "contrast": k, **summarise(d, SEED + 1)})
    fmt = []
    for size in CHAINS:
        for stage in ("base", "sft"):
            for spec in specs:
                for c in CATEGORIES:
                    d = (fam[f"{size}_{stage}_chat"][spec][c]["advice_vs_information"]
                         - fam[f"{size}_{stage}_qa"][spec][c]["advice_vs_information"])
                    fmt.append({"size": size, "stage": stage, "spec": spec, "category": c,
                                "chat_minus_qa": summarise(d, SEED + 2)})
    # grid robustness of each stage change: share of the 12 specs with positive change / CI excluding zero
    robust = []
    for size, chain in CHAINS.items():
        for a, b in zip(chain, chain[1:]):
            for c in CATEGORIES:
                for k in CONTRASTS:
                    sel = [x for x in changes if x["size"] == size and x["from"] == a and x["category"] == c
                           and x["contrast"] == k]
                    est = np.array([x["estimate"] for x in sel])
                    robust.append({"size": size, "from": a, "to": b, "category": c, "contrast": k, "n_specs": len(sel),
                                   "exact_all": next(x["estimate"] for x in sel if x["spec"] == "exact|all"),
                                   "union82_all": next(x["estimate"] for x in sel if x["spec"] == "union_0.82|all"),
                                   "min": float(est.min()), "max": float(est.max()),
                                   "share_positive": float((est > 0).mean()),
                                   "n_ci_above_0": sum(x["ci95"][0] > 0 for x in sel),
                                   "n_ci_below_0": sum(x["ci95"][1] < 0 for x in sel),
                                   "n_tost_equiv_5pp": sum(x["tost_equivalent_5pp"] for x in sel)})
    meta = {"prompt_set": "experiments/pivot/prompts_revision_v1.json (via pivot_runner): 40 topic families x "
                          "2 emotional framings x 2 request types (information/advice) = 160 prompts, 3 seeds; "
                          "not the paper's 140-prompt battery (which uses five author-written conditions)",
            "tags": {t: len(r) for t, r in data.items()}, "cluster": "topic family (n=40)", "n_boot": N_BOOT,
            "seed": SEED, "tost_margin": MARGIN, "windows_tokenizer": "allenai/OLMo-2-0425-1B-Instruct",
            "note": "Base-to-SFT changes in chat format mix format and training; base_chat uses the chat template on a base model."}
    write_json(OUT / "atlas_rq2.json", {"meta": meta, "rates": rates, "levels": levels, "stage_changes": changes,
                                        "format_effect": fmt, "grid_robustness": robust})

    def f(v):
        return f"${v:+.3f}$"
    rows = []
    for x in robust:
        if x["contrast"] != "advice_vs_information":
            continue
        ex82 = next(y for y in changes if y["size"] == x["size"] and y["from"] == x["from"] and y["category"] == x["category"]
                    and y["contrast"] == x["contrast"] and y["spec"] == "exact|all")
        rows.append([f"{x['size']} {x['from']}$\\to${x['to']}", x["category"].replace("_", "\\_"),
                     f"{f(x['exact_all'])} [{ex82['ci95'][0]:.2f}, {ex82['ci95'][1]:.2f}]", f(x["union82_all"]),
                     f"{f(x['min'])} to {f(x['max'])}", f"{x['n_ci_above_0']}/{x['n_ci_below_0']}",
                     x["n_tost_equiv_5pp"]])
    write_tex(OUT / "atlas_stage_changes.tex",
              ["Stage", "Marker", "Exact, all [95\\% CI]", "Union .82, all", "Grid range", "CI$>0$/CI$<0$",
               "TOST 5pp"], rows, "Change in advice-minus-information gap between adjacent chat stages")
    print("wrote", OUT)


if __name__ == "__main__":
    main()

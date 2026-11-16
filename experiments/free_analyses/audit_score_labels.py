"""Score two completed human label files for the v2 sentence audit.

    python -m experiments.free_analyses.audit_score_labels LABELS_1.csv LABELS_2.csv
        [--gold mean|a1|a2|adjudicated] [--adjudicated FILE.csv] [--n-boot 2000]

Reports, per category:
  * Cohen's kappa and Krippendorff's alpha (nominal, two coders) with item
    bootstrap 95% CIs;
  * per rule: precision / recall of the rule's sentence prediction against the
    gold label on the design draws (inverse-probability weighted);
  * per rule: DSL corrected advice-minus-control gap in sentence-level marker
    prevalence (Egami et al. 2023 design-based supervised learning):
        mu_g = mean_{i in pop_g} [ Yhat_i + R_i / pi_i * (Y_i - Yhat_i) ]
    with the naive gap (Yhat only) beside it, and a stratified bootstrap CI
    that resamples labeled draws within each design stratum.
Gold labels: 'mean' averages the two annotators (0, .5 or 1).
Writes output/revision/human_audit_v2/results/audit_scores.json.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict

import numpy as np

from experiments.free_analyses.audit_prepare_sample import AUDIT_DIR, CAT_TO_COL, KEY_DIR, LABEL_COLS
from experiments.free_analyses.matchers_common import CATEGORIES, CONTROL, SEED


def read(path):
    with open(path, encoding="utf-8") as h:
        return list(csv.DictReader(h))


def labels(path):
    out = {}
    for r in read(path):
        vals = {}
        for c in LABEL_COLS:
            v = r[c].strip()
            if v not in ("0", "1"):
                raise ValueError(f"{path}: item {r['item_id']} column {c} must be 0 or 1, got {v!r}")
            vals[c] = int(v)
        out[r["item_id"]] = vals
    return out


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def kripp_alpha_nominal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    vals = np.concatenate([a, b])
    n = len(vals)
    do = np.mean(a != b)
    p1 = vals.mean()
    de = 2 * p1 * (1 - p1) * n / (n - 1)
    return float("nan") if de == 0 else float(1 - do / de)


def boot_ci(fn, a, b, rng, n_boot):
    n = len(a); vals = []
    for _ in range(n_boot):
        j = rng.integers(0, n, n)
        v = fn(a[j], b[j])
        if not np.isnan(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_1"); ap.add_argument("labels_2")
    ap.add_argument("--gold", default="mean", choices=["mean", "a1", "a2", "adjudicated"])
    ap.add_argument("--adjudicated"); ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    l1, l2 = labels(args.labels_1), labels(args.labels_2)
    key = {r["item_id"]: r for r in read(KEY_DIR / "audit_key.csv")}
    if set(l1) != set(key) or set(l2) != set(key):
        raise ValueError("label files must cover exactly the items in audit_items_blinded.csv")
    adj = labels(args.adjudicated) if args.gold == "adjudicated" else None
    design = json.loads((KEY_DIR / "design.json").read_text(encoding="utf-8"))
    rules = list(design["rules"])
    with gzip.open(KEY_DIR / "population_predictions.csv.gz", "rt", encoding="utf-8") as h:
        pop = list(csv.DictReader(h))
    pop_by_sid = {p["sid"]: p for p in pop}

    ids = sorted(key)
    res = {"gold": args.gold, "n_items": len(ids), "agreement": {}, "rules": {}}
    for c in CATEGORIES:
        col = CAT_TO_COL[c]
        a = np.array([l1[i][col] for i in ids]); b = np.array([l2[i][col] for i in ids])
        res["agreement"][c] = {
            "raw_agreement": float(np.mean(a == b)),
            "cohen_kappa": cohen_kappa(a, b), "cohen_kappa_ci95": boot_ci(cohen_kappa, a, b, rng, args.n_boot),
            "krippendorff_alpha": kripp_alpha_nominal(a, b),
            "krippendorff_alpha_ci95": boot_ci(kripp_alpha_nominal, a, b, rng, args.n_boot),
            "positives_a1": int(a.sum()), "positives_a2": int(b.sum())}

    def gold(i, col):
        if args.gold == "a1": return l1[i][col]
        if args.gold == "a2": return l2[i][col]
        if args.gold == "adjudicated": return adj[i][col]
        return (l1[i][col] + l2[i][col]) / 2

    for c in CATEGORIES:
        col = CAT_TO_COL[c]; ctrl = CONTROL[c]
        draws = [i for i in ids if key[i][f"in_design__{c}"] == "1"]
        strata = defaultdict(list)
        for i in draws:
            strata[key[i][f"stratum__{c}"]].append(i)
        conds = ("advice", ctrl)
        n_pop = {g: sum(p["condition"] == g for p in pop) for g in conds}
        res["rules"][c] = {}
        for rn in rules:
            pc = f"pred__{rn}__{c}"
            yhat_sum = {g: sum(int(p[pc]) for p in pop if p["condition"] == g) for g in conds}

            def est(sample):
                mu = {}
                for g in conds:
                    corr = sum((gold(i, col) - int(pop_by_sid[key[i]["sid"]][pc])) / float(key[i][f"pi__{c}"])
                               for i in sample if key[i]["condition"] == g)
                    mu[g] = (yhat_sum[g] + corr) / n_pop[g]
                return mu

            mu = est(draws)
            bs = []
            for _ in range(args.n_boot):
                samp = [s[j] for s in strata.values() for j in rng.integers(0, len(s), len(s))]
                m = est(samp); bs.append(m["advice"] - m[ctrl])
            w = np.array([1 / float(key[i][f"pi__{c}"]) for i in draws])
            y = np.array([gold(i, col) for i in draws], float)
            yh = np.array([int(pop_by_sid[key[i]["sid"]][pc]) for i in draws], float)
            prec = float((w * y * yh).sum() / (w * yh).sum()) if (w * yh).sum() else None
            rec = float((w * y * yh).sum() / (w * y).sum()) if (w * y).sum() else None
            res["rules"][c][rn] = {
                "naive_gap": yhat_sum["advice"] / n_pop["advice"] - yhat_sum[ctrl] / n_pop[ctrl],
                "dsl_gap": mu["advice"] - mu[ctrl], "dsl_gap_ci95": [float(np.percentile(bs, 2.5)),
                                                                   float(np.percentile(bs, 97.5))],
                "dsl_rate_advice": mu["advice"], "dsl_rate_control": mu[ctrl],
                "weighted_precision": prec, "weighted_recall": rec, "n_draws": len(draws)}
    outdir = AUDIT_DIR / "results"; outdir.mkdir(exist_ok=True)
    (outdir / "audit_scores.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

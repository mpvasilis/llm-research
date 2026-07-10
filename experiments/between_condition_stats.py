"""Between-condition (advice vs factual) significance tests for the emission
dissociation. Reviewer point: the paper reports per-condition Wilson intervals
but no test on the *difference*. This adds, per behavior x model, a two-sided
Fisher exact test, an odds ratio (Haldane-Anscombe 0.5 correction for zero
cells), and a risk difference with a Newcombe (Wilson-based) 95% CI.

Inputs are integer counts derived from the canonical summary artifact at the
primary e5 threshold 0.82; no model outputs are needed. Prefers
out/results/summary_v2.json when present, falling back to summary.json.

Run:  python -m experiments.between_condition_stats
"""
import json
import math

from scipy.stats import fisher_exact
from config import ROOT
from experiments.summary_adapter import CATS, condition_n, emission_rates, load_summary

S, SUMMARY_PATH = load_summary()
REPORT_CATS = [c for c in CATS if c != "crisis_referral"]


def _fmt_p(x):
    if x is None:
        return "NA"
    if isinstance(x, float) and not math.isfinite(x):
        return "NA"
    return f"{x:.2e}" if isinstance(x, float) else str(x)


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def newcombe_rd_ci(k1, n1, k2, n2, z: float = 1.96):
    """Newcombe method 10 for the CI of a risk difference p1 - p2."""
    l1, u1 = _wilson(k1, n1, z)
    l2, u2 = _wilson(k2, n2, z)
    p1, p2 = k1 / n1, k2 / n2
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def odds_ratio_hc(k1, n1, k2, n2):
    """Haldane-Anscombe odds ratio (add 0.5 to every cell) + Wald 95% CI."""
    a, b = k1 + 0.5, (n1 - k1) + 0.5
    c, d = k2 + 0.5, (n2 - k2) + 0.5
    or_ = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = math.exp(math.log(or_) - 1.96 * se)
    hi = math.exp(math.log(or_) + 1.96 * se)
    return or_, lo, hi


def main():
    out = {}
    print(f"Using summary artifact: {SUMMARY_PATH}")
    print(f"{'model':>5} {'behavior':>15} {'advice':>8} {'factual':>8} "
          f"{'RD [95% CI]':>26} {'OR [95% CI]':>24} {'Fisher p':>10} {'perm p':>10}")
    print("-" * 112)
    for model in S["per_model"]:
        na, nf = condition_n(S, model, "advice"), condition_n(S, model, "factual")
        er = emission_rates(S, model)
        tests = S["per_model"][model].get("tests", {})
        out[model] = {}
        for name in REPORT_CATS:
            ka = round(er[name]["advice"] * na)
            kf = round(er[name]["factual"] * nf)
            table = [[ka, na - ka], [kf, nf - kf]]
            _, p = fisher_exact(table, alternative="two-sided")
            rd = ka / na - kf / nf
            rlo, rhi = newcombe_rd_ci(ka, na, kf, nf)
            orr, olo, ohi = odds_ratio_hc(ka, na, kf, nf)
            out[model][name] = {
                "advice": f"{ka}/{na}", "factual": f"{kf}/{nf}",
                "risk_diff": round(rd, 3), "rd_ci": [round(rlo, 3), round(rhi, 3)],
                "odds_ratio": round(orr, 1), "or_ci": [round(olo, 1), round(ohi, 1)],
                "fisher_p": p,
                "prompt_clustered_perm_p": tests.get(name, {}).get("factual", {}).get("perm_p"),
            }
            print(f"{model:>5} {name:>15} {ka:>4}/{na:<3} {kf:>4}/{nf:<3} "
                  f"{rd:>6.3f} [{rlo:>5.2f},{rhi:>5.2f}] "
                  f"{orr:>8.1f} [{olo:>6.1f},{ohi:>7.1f}] {p:>10.2e} "
                  f"{_fmt_p(out[model][name]['prompt_clustered_perm_p']):>10}")
    dest = ROOT / "out" / "results" / "between_condition_stats.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nWrote", dest)


if __name__ == "__main__":
    main()

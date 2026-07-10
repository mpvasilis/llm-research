"""Derive table numbers in the paper from the canonical summary artifact.

Emits: emission k/n + Wilson CI, between-condition Fisher/OR/RD, and the
distinct 3-word leave-one-out. NDOCS per mixture read from the recount artifact.
Prefers out/results/summary_v2.json when present, falling back to summary.json.

Run:  python -m experiments.derive_paper_stats
"""
import math

from scipy.stats import fisher_exact

from experiments.summary_adapter import (
    CATS,
    condition_n,
    emission_rates,
    final_model_label,
    load_sft_ndocs,
    load_summary,
)

S, SUMMARY_PATH = load_summary()
NDOCS = load_sft_ndocs()
DROP_CRISIS = True  # crisis_referral failed detector validation (0 TP / 16 FP)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)


def newcombe(k1, n1, k2, n2, z=1.96):
    l1, u1 = wilson(k1, n1, z); l2, u2 = wilson(k2, n2, z)
    p1, p2 = k1/n1, k2/n2
    return (p1-p2 - math.sqrt((p1-l1)**2 + (u2-p2)**2),
            p1-p2 + math.sqrt((u1-p1)**2 + (p2-l2)**2))


def or_hc(k1, n1, k2, n2):
    a, b, c, d = k1+.5, n1-k1+.5, k2+.5, n2-k2+.5
    orr = (a*d)/(b*c); se = math.sqrt(1/a+1/b+1/c+1/d)
    return orr, math.exp(math.log(orr)-1.96*se), math.exp(math.log(orr)+1.96*se)


def fmt_p(x):
    if x is None:
        return "NA"
    if isinstance(x, float) and not math.isfinite(x):
        return "NA"
    return f"{x:.2e}" if isinstance(x, float) else str(x)


print(f"Using summary artifact: {SUMMARY_PATH}")
for m in ("1B", "7B"):
    d = S["per_model"][m]; na, nf = condition_n(S, m, "advice"), condition_n(S, m, "factual")
    er = emission_rates(S, m)
    tests = d.get("tests", {})
    print(f"\n===== {m} ({final_model_label(S, m)}) =====")
    print("EMISSION k/n (Wilson 95%) and BETWEEN-CONDITION tests @0.82:")
    for c in CATS:
        if DROP_CRISIS and c == "crisis_referral":
            continue
        v = er[c]
        ka, kf = round(v["advice"]*na), round(v["factual"]*nf)
        la, ua = wilson(ka, na); lf, uf = wilson(kf, nf)
        _, p = fisher_exact([[ka, na-ka], [kf, nf-kf]], alternative="two-sided")
        rd = ka/na - kf/nf; rlo, rhi = newcombe(ka, na, kf, nf)
        orr, olo, ohi = or_hc(ka, na, kf, nf)
        perm_p = tests.get(c, {}).get("factual", {}).get("perm_p")
        print(f"  {c:<15} adv {ka}/{na} ({v['advice']:.3f} [{la:.2f},{ua:.2f}])  "
              f"fac {kf}/{nf} ({v['factual']:.3f} [{lf:.2f},{uf:.2f}])  "
              f"RD {rd:.3f}[{rlo:.2f},{rhi:.2f}] OR {orr:.1f}[{olo:.1f},{ohi:.1f}] "
              f"fisher_p={p:.2e} perm_p={fmt_p(perm_p)}")

    # distinct 3-word leave-one-out
    nd = NDOCS.get(m)
    if not nd:
        print("  SFT NDOCS unavailable; skipping leave-one-out.")
        continue
    three = [(r["phrase"], r["assistant"]) for r in d["role_scoping"]
             if len(r["phrase"].split()) == 3]
    three.sort(key=lambda x: -x[1])
    permil = [(ph, round(1e6*a/nd, 1)) for ph, a in three]
    mean_all = round(sum(pm for _, pm in permil)/len(permil), 1)
    drop = permil[0][0]
    rest = [pm for ph, pm in permil if ph != drop]
    mean_drop = round(sum(rest)/len(rest), 1)
    print(f"  3-word distinct types ({len(three)}), NDOCS={nd:,}:")
    for ph, pm in permil:
        print(f"      {ph:<28} {pm}/M")
    print(f"  leave-one-out: mean {mean_all} -> {mean_drop} (drop '{drop}')")

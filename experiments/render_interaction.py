"""Render results/interaction_tests.json into the paper's stage x condition table
and a significance summary. Reproduces every number in the Stagewise section.

Run:  python -m experiments.render_interaction
"""
import json
from config import ROOT

_path = ROOT / "out" / "results" / "interaction_tests_v3.json"
if not _path.exists():
    _path = ROOT / "out" / "results" / "interaction_tests.json"
I = json.loads(_path.read_text())
REPORT = ["empathy_opener", "validation", "disclaimer", "structure"]
ADJ = {"1B": ["base->sft", "sft->dpo"], "7B": ["base->sft", "sft->dpo"]}


def fp(p):
    if p == 0.0:
        return r"$<$1e-4"
    if p < 1e-3:
        return r"$<$1e-3"
    return f"{p:.3f}"


print("=== LaTeX rows (DiD change-in-gap, permutation p) ===")
for m in ("1B", "7B"):
    for c in REPORT:
        cells = []
        for tr in ADJ[m]:
            d = I[m][c][tr]
            cells.append(f"{d['did']:+.3f} ({fp(d['perm_p'])})")
        print(f"{m} & {c.replace('_opener','').replace('_',' ')} & " + " & ".join(cells) + r" \\")

print("\n=== significance summary (adjacent transitions, p<.05) ===")
for m in ("1B", "7B"):
    for c in REPORT:
        sig = [tr for tr in ADJ[m] if I[m][c][tr]["perm_p"] < 0.05]
        print(f"  {m} {c:<15} sig widen/collapse at: {sig or 'none beyond base->sft? check'}")

print("\n=== RLVR / final-merge (should be all ns) ===")
for m in ("1B", "7B"):
    tail = [k for k in list(I[m]["disclaimer"]) if k not in ("base->sft", "sft->dpo", "base->instruct")]
    for c in REPORT:
        ps = {tr: I[m][c][tr]["perm_p"] for tr in tail}
        mx = max(ps.values()) if ps else None
        mn = min(ps.values()) if ps else None
        print(f"  {m} {c:<15} tail transitions {list(ps)}: p in [{mn},{mx}]")

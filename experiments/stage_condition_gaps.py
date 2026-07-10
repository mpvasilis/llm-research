"""Stage x condition analysis (difference-in-differences) for the stagewise eval.

Reviewer point: showing that advice-side emission rises across stages does not
establish an advice-directed change -- post-training could make ALL outputs more
cautious/listy. The right quantity is the advice-minus-factual GAP per stage,
Delta_s = P(B|advice, s) - P(B|factual, s), and its change across stages.

Reads the canonical summary artifact (v2 stagewise run) and emits, per model x
behavior: the gap at every stage and the stage-to-stage gap deltas.

Run:  python -m experiments.stage_condition_gaps
"""
import json

from config import ROOT
from experiments.summary_adapter import CATS, load_summary, stagewise_emission

S, PATH = load_summary()
print(f"Using summary artifact: {PATH}\n")

out = {}
for m, block in S["per_model"].items():
    sw = stagewise_emission(S, m)
    if not sw:
        print(f"{m}: no stagewise data in artifact; skipping")
        continue
    stages = block["stages"]
    out[m] = {}
    print(f"===== {m} (stages: {' -> '.join(stages)}) =====")
    print(f"{'behavior':<16} " + " ".join(f"{st:>14}" for st in stages) + "   stage-to-stage gap change")
    for c in CATS:
        gaps = {}
        for st in stages:
            a = sw[st][c]["advice"]
            f = sw[st][c]["factual"]
            gaps[st] = round((a or 0) - (f or 0), 3)
        deltas = {f"{stages[i]}->{stages[i+1]}": round(gaps[stages[i+1]] - gaps[stages[i]], 3)
                  for i in range(len(stages) - 1)}
        out[m][c] = {"gap_by_stage": gaps, "gap_deltas": deltas}
        cells = " ".join(f"{gaps[st]:>14.3f}" for st in stages)
        dstr = ", ".join(f"{k}: {v:+.3f}" for k, v in deltas.items())
        print(f"{c:<16} {cells}   {dstr}")
    print()

dest = ROOT / "out" / "results" / "stage_condition_gaps.json"
dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
print("Wrote", dest)

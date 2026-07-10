"""Diagnose the stale role-scoping counts: recompute each role_scoping phrase's
assistant/blob counts against the FULL 6-shard Tulu + full oasst1, authoritatively,
and compare to what summary.json stored per model.

Run:  python -m experiments._diag_counts
"""
import json
from config import ROOT
import trace.parquet_search as ps

# force full coverage for the authoritative recompute
ps._SHARD_CAP["allenai/tulu-3-sft-mixture"] = 6

S = json.loads((ROOT / "out" / "results" / "summary.json").read_text(encoding="utf-8"))
phrases = {}
for m, blk in S["per_model"].items():
    for r in blk["role_scoping"]:
        phrases.setdefault(r["phrase"], {})[m] = r["assistant"]

print("Recomputing authoritative 6-shard assistant counts (downloads ~1.4GB once)...\n")
print(f"{'phrase':<28} {'auth':>7} | {'1B':>7} {'7B':>7}  flag")
out = {}
for ph in phrases:
    a = 0
    for ds in ["allenai/tulu-3-sft-mixture", "OpenAssistant/oasst1"]:
        a += (ps.count_roles(ds, ph).get("assistant_matches") or 0)
    b1 = phrases[ph].get("1B"); b7 = phrases[ph].get("7B")
    flag = ""
    if b1 is not None and b1 != a: flag += " 1B-stale"
    if b7 is not None and b7 != a: flag += " 7B-stale"
    out[ph] = {"authoritative": a, "1B": b1, "7B": b7}
    print(f"{ph:<28} {a:>7} | {str(b1):>7} {str(b7):>7} {flag}")
(ROOT / "out" / "results" / "authoritative_counts.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nWrote out/results/authoritative_counts.json")

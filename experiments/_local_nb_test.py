"""Dry-run the Colab notebook's REAL cell code locally to catch bugs before Colab.

Neutralizes Colab-only bits (pip, drive mount, userdata), points PROJECT at ./out
(so the cached parquet is reused, no re-download), and shrinks the workload to
1B / 2 advice + 1 factual / short generations. Execs the actual notebook cells in
order so we test the inline code, not a reimplementation.

Run:  python -m experiments._local_nb_test
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB = ROOT / "experiments" / "colab_provenance_experiments.ipynb"
TOKEN = ""
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("HF_TOKEN="):
        TOKEN = line.split("=", 1)[1].strip()

nb = json.loads(NB.read_text(encoding="utf-8"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__"}


def patch(src: str) -> str:
    out = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith(("%", "!")):
            continue
        if "google.colab" in ln or "drive.mount" in ln:
            continue
        if ln.startswith("PROJECT = Path('/content"):
            out.append(f"PROJECT = Path(r'{SCRATCH}')")
            continue
        out.append(ln)
    return "\n".join(out)


# scratch project dir so we DON'T overwrite the real out/results/summary.json;
# seed its cache from out/cache so the parquet isn't re-downloaded.
import shutil
SCRATCH = ROOT / "out" / "_nbtest"
(SCRATCH / "cache").mkdir(parents=True, exist_ok=True)
for d in (ROOT / "out" / "cache").glob("*"):
    dst = SCRATCH / "cache" / d.name
    if d.is_dir() and not dst.exists():
        shutil.copytree(d, dst)


for idx, cell in enumerate(code_cells):
    src = "".join(cell["source"])
    title = src.splitlines()[0] if src else ""
    # replace HF-token cell entirely (avoid getpass blocking)
    if "#@title 3" in title:
        src = f"import os\nHF_TOKEN={TOKEN!r}\nos.environ['HF_TOKEN']=HF_TOKEN\nprint('token set:',bool(HF_TOKEN))"
    else:
        src = patch(src)
    print(f"\n=== exec cell {idx}: {title[:60]} ===", flush=True)
    exec(compile(src, f"<cell {idx}>", "exec"), g)
    # shrink workload right after the config cell
    if "#@title 4" in title:
        g["MODELS"] = ["1B"]
        g["MAX_NEW_TOKENS"] = 48
        g["TULU_SHARD_CAP"] = 1
        g["PROMPTS"] = {"advice": g["PROMPTS"]["advice"][:2],
                        "factual": g["PROMPTS"]["factual"][:1]}
        print("  [test] MODELS=['1B'], 2 advice + 1 factual, MAX_NEW_TOKENS=48")

# verify the artifact (from the SCRATCH dir, not the real out/results)
summ = json.loads((SCRATCH / "results" / "summary.json").read_text(encoding="utf-8"))
print("\n=== ARTIFACT CHECK ===")
assert "per_model" in summ and "1B" in summ["per_model"], "per_model/1B missing"
b = summ["per_model"]["1B"]
for k in ["emission_rate", "emission_rate_by_threshold", "recoverability_by_length",
          "role_scoping", "behavioral_forum_share", "novelty", "behavioral_coverage"]:
    assert k in b, f"missing {k}"
assert set(b["emission_rate_by_threshold"]) == {"0.8", "0.82", "0.84"}, b["emission_rate_by_threshold"].keys()
print("summary.json OK - keys present incl. emission_rate_by_threshold (0.80/0.82/0.84)")
print("emission(advice):", {c: v["advice"] for c, v in b["emission_rate"].items()})
print("threshold sweep (advice, disclaimer):",
      {t: b["emission_rate_by_threshold"][t]["disclaimer"]["advice"] for t in b["emission_rate_by_threshold"]})
print("validation cells ran:", (SCRATCH / "results" / "validation_sheet.csv").exists())
print("ALL CELLS RAN + ARTIFACT VALID")

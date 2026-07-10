"""Build the single Colab notebook for the complete BlackboxNLP pipeline.

The notebook combines the existing full stagewise generation/data-search code,
the corrected paper-statistics modules, provenance-safe validation v3, and the
optional causal SFT ablation. Expensive sections are explicit opt-in gates.

Run: python -m experiments.build_complete_colab_notebook
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from config import ROOT


TEMPLATE = ROOT / "output" / "jupyter-notebook" / "blackboxnlp_complete_pipeline.ipynb"
FULL_NB = ROOT / "experiments" / "colab_provenance_experiments_v2.ipynb"
CAUSAL_NB = ROOT / "experiments" / "colab_causal_ablation.ipynb"
OUTPUT = TEMPLATE
MIRROR = ROOT / "experiments" / "colab_complete_pipeline.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def source_cell(notebook: dict, index: int) -> str:
    return "".join(notebook["cells"][index]["source"])


def guarded(source: str, flag: str, project_name: str | None = None) -> str:
    lines = source.splitlines()
    title = lines[0] if lines and lines[0].lstrip().startswith("#@title") else "# Optional stage"
    body = "\n".join(lines[1:] if lines and lines[0] == title else lines)
    if project_name:
        body = re.sub(r"\bPROJECT\b", project_name, body)
    body = body.replace("QUICK_TEST = False", "QUICK_TEST = FULL_QUICK_TEST")
    body = body.replace("LOAD_8BIT  = False", "LOAD_8BIT  = FULL_LOAD_8BIT")
    return (
        f"{title}\n"
        f"if not {flag}:\n"
        f"    print('Skipped: set {flag}=True in the configuration cell.')\n"
        "else:\n"
        + textwrap.indent(body, "    ")
        + "\n"
    )


template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
full = json.loads(FULL_NB.read_text(encoding="utf-8"))
causal = json.loads(CAUSAL_NB.read_text(encoding="utf-8"))
cells: list[dict] = []

cells.append(
    markdown(
        r"""# BlackboxNLP complete advice-provenance pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mpvasilis/llm-research/blob/main/output/jupyter-notebook/blackboxnlp_complete_pipeline.ipynb)

Public repository: [mpvasilis/llm-research](https://github.com/mpvasilis/llm-research)

**Objective.** Reproduce every reported result, optionally regenerate the
OLMo-2 checkpoint outputs and corpus searches, complete provenance-safe blinded
human validation, and optionally run the causal SFT leave-cluster-out pilot.

**Success criteria**

1. All paper statistics regenerate from machine-readable artifacts.
2. The DPO analysis is paired, word-normalized, and multiplicity-corrected.
3. Detector validation uses two separate certified human annotation files;
   predictions, conditions, and the other annotator's decisions remain hidden.
4. The legacy 80-row tool-assisted sheet and AI audit are never treated as
   human validation.
5. Final exports clearly distinguish observational, human-validated, and causal
   pilot evidence.

The default run is CPU-friendly and uses the released artifacts. GPU generation,
the 2.7 GB remote DPO scan, and causal re-SFT are explicit opt-in sections."""
    )
)

cells.append(
    markdown(
        r"""## Experiment plan

- **A — Integrity and artifact reproduction:** validate inputs and regenerate
  plus-one-corrected permutation tests, condition contrasts, stage interactions,
  BH corrections, and threshold robustness.
- **B — Full observational trace (optional GPU):** regenerate responses for all
  public OLMo-2 stages, behavior detections, Stage-1 novelty, and SFT/DPO/RLVR
  corpus searches.
- **C — Paired DPO analysis (optional network scan):** chosen-only vs rejected-only
  pairs, exact McNemar tests, and response-word normalization.
- **D — Human validation v3:** blinded items, independent files, agreement,
  category/condition/stage metrics, confidence intervals, and adjudication.
- **E — Causal pilot (optional GPU):** exact 1B OLMo-2 SFT mixture, behavioral
  cluster removal, coherent/random controls, and checkpointed evaluation.

Do not cite a section as complete merely because its code cell ran. The final
status panel distinguishes cached evidence, pending human work, and pilots."""
    )
)

cells.append(
    code(
        r"""#@title 1 · Install the analysis environment
%pip -q install "accelerate>=1.13" "datasets>=4.0" "duckdb>=1.5" "matplotlib>=3.8" "numpy>=2.0" "pandas>=2.2" "scikit-learn>=1.5" "scipy>=1.13" "sentence-transformers>=3.3" "transformers>=4.47" "trl>=0.12" peft bitsandbytes ipywidgets huggingface_hub
import sys, torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")"""
    )
)

cells.append(
    code(
        r"""#@title 2 · Locate the repository and persistent Drive run directory
import os, shutil, subprocess, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/mpvasilis/llm-research.git"  #@param {type:"string"}
DRIVE_REPO = "/content/drive/MyDrive/llm-research"  #@param {type:"string"}
DRIVE_REPO_ZIP = "/content/drive/MyDrive/llm-research.zip"  #@param {type:"string"}

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    candidates = [Path(DRIVE_REPO), Path("/content/llm-research")]
    REPO_ROOT = next((p for p in candidates if (p / "experiments").exists()), None)
    if REPO_ROOT is None and Path(DRIVE_REPO_ZIP).exists():
        shutil.unpack_archive(DRIVE_REPO_ZIP, "/content")
        REPO_ROOT = next((p for p in Path("/content").glob("**/llm-research") if (p / "experiments").exists()), None)
    if REPO_ROOT is None and REPO_URL.strip():
        subprocess.run(["git", "clone", REPO_URL, "/content/llm-research"], check=True)
        REPO_ROOT = Path("/content/llm-research")
    if REPO_ROOT is None:
        raise FileNotFoundError("Put the repository at DRIVE_REPO, upload DRIVE_REPO_ZIP, or set REPO_URL.")
    RUN_PROJECT = Path("/content/drive/MyDrive/llm-research-experiments-v3")
else:
    REPO_ROOT = Path.cwd()
    if not (REPO_ROOT / "experiments").exists():
        raise FileNotFoundError("Run this notebook from the llm-research repository root.")
    RUN_PROJECT = REPO_ROOT / "out" / "colab_run_v3"

os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for sub in ["answers", "attribution", "instruction", "behavior", "cache", "results", "validation_v3", "causal"]:
    (RUN_PROJECT / sub).mkdir(parents=True, exist_ok=True)
print("Repository:", REPO_ROOT)
commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
    check=False, capture_output=True, text=True,
).stdout.strip()
print("Repository commit:", commit or "local/unversioned")
print("Persistent run:", RUN_PROJECT)"""
    )
)

cells.append(
    code(
        r"""#@title 3 · Central run configuration
# Fast artifact reproduction runs by default.
RUN_FULL_PIPELINE = False       #@param {type:"boolean"}
FULL_QUICK_TEST = False         #@param {type:"boolean"}
FULL_LOAD_8BIT = False          #@param {type:"boolean"}
RUN_REMOTE_DPO_SCAN = False     #@param {type:"boolean"}
RUN_CAUSAL_PILOT = False        #@param {type:"boolean"}

# Validation: sentence_120 works from the released bundle. response_stagewise
# is the recommended publication workflow after RUN_FULL_PIPELINE has populated
# RUN_PROJECT/answers and RUN_PROJECT/behavior.
VALIDATION_MODE = "sentence_120"  #@param ["sentence_120", "response_stagewise"]
ANNOTATOR_ID = 1                   #@param [1, 2] {type:"raw"}
ANNOTATOR_NAME = ""                #@param {type:"string"}
I_AM_A_HUMAN_ANNOTATOR = False     #@param {type:"boolean"}

VALIDATION_DIR = RUN_PROJECT / "validation_v3"
print({
    "full_pipeline": RUN_FULL_PIPELINE,
    "remote_dpo": RUN_REMOTE_DPO_SCAN,
    "causal_pilot": RUN_CAUSAL_PILOT,
    "validation_mode": VALIDATION_MODE,
    "annotator": ANNOTATOR_ID,
})"""
    )
)

cells.append(
    code(
        r"""#@title 4 · Hugging Face token (requested only for network/GPU sections)
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if IN_COLAB and not HF_TOKEN:
    try:
        from google.colab import userdata
        HF_TOKEN = (userdata.get("HF_TOKEN") or "").strip()
    except Exception:
        pass
if (RUN_FULL_PIPELINE or RUN_REMOTE_DPO_SCAN or RUN_CAUSAL_PILOT) and not HF_TOKEN:
    from getpass import getpass
    HF_TOKEN = getpass("Hugging Face read token (leave blank for ungated public artifacts): ").strip()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
print("HF token available:", bool(HF_TOKEN))"""
    )
)

cells.append(
    code(
        r"""#@title 5 · Integrity and provenance audit
import json
try:
    from IPython.display import display
except ImportError:
    display = print
import pandas as pd

required = [
    "out/results/summary_v2.json",
    "out/results/interaction_tests.json",
    "out/results/dpo_pairwise.json",
    "out/results/validation_sheet_v2.csv",
    "out/paper/ANNOTATION_GUIDE.md",
]
audit = []
for relative in required:
    path = REPO_ROOT / relative
    audit.append({"artifact": relative, "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0})
display(pd.DataFrame(audit))
missing = [row["artifact"] for row in audit if not row["exists"]]
if missing:
    raise FileNotFoundError(f"Missing required released artifacts: {missing}")

legacy = REPO_ROOT / "out/results/validation_sheet.csv"
human_v2 = pd.read_csv(REPO_ROOT / "out/results/validation_sheet_v2.csv")
status = {
    "legacy_80_row_sheet": "deprecated; never consumed by validation v3",
    "legacy_exists": legacy.exists(),
    "v2_human_label_1_nonempty": int(human_v2["human_label_1"].fillna("").astype(str).str.strip().ne("").sum()),
    "v2_human_label_2_nonempty": int(human_v2["human_label_2"].fillna("").astype(str).str.strip().ne("").sum()),
    "ai_audit": "diagnostic only; excluded from human validation",
}
print(json.dumps(status, indent=2))"""
    )
)

cells.append(
    markdown(
        r"""## A. Optional full observational regeneration

This section contains the complete checkpoint generation and corpus-search code
from the comprehensive v2 run. It uses the exact 1B/7B OLMo-2 checkpoints and
model-specific SFT/DPO/RLVR datasets. Outputs are checkpointed under
`RUN_PROJECT`, so rerunning skips completed items.

Set `RUN_FULL_PIPELINE=True`. The full 7B grid requires a suitable GPU and may
take several hours. `FULL_QUICK_TEST=True` provides a smoke test but is not a
publication result."""
    )
)

# Full v2 cells 4..15 contain config, helpers, clients, generation, searches,
# aggregation, and the stage x condition interaction. Legacy validation cells
# 16..18 are intentionally not copied; validation v3 supersedes them.
for index in range(4, 16):
    cells.append(code(guarded(source_cell(full, index), "RUN_FULL_PIPELINE", "RUN_PROJECT")))

cells.append(
    code(
        r"""#@title A13 · Sync regenerated aggregate artifacts into the repository analysis bundle
if not RUN_FULL_PIPELINE:
    print("Skipped sync: using released artifacts.")
else:
    mapping = {
        "summary_v2.json": "summary_v2.json",
        "interaction_tests.json": "interaction_tests.json",
    }
    for source_name, destination_name in mapping.items():
        source = RUN_PROJECT / "results" / source_name
        destination = REPO_ROOT / "out" / "results" / destination_name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        print("Synced", source, "->", destination)"""
    )
)

cells.append(
    markdown(
        r"""## B. Corrected paper statistics and robustness

These cells are the default reproducibility path. They apply plus-one Monte
Carlo corrections, prompt-clustered interaction tests, separate BH families,
threshold robustness, and paper-table exports from the released or regenerated
artifact bundle."""
    )
)

cells.append(
    code(
        r"""#@title B1 · Regenerate all locally available paper statistics
import subprocess, sys
modules = [
    "experiments.monte_carlo_correction",
    "experiments.derive_paper_stats",
    "experiments.between_condition_stats",
    "experiments.stage_condition_gaps",
    "experiments.multiple_testing",
    "experiments.threshold_robustness",
    "experiments.render_interaction",
]
for module in modules:
    print("\n===", module, "===")
    completed = subprocess.run(
        [sys.executable, "-m", module], cwd=REPO_ROOT,
        check=False, capture_output=True, text=True
    )
    if completed.returncode:
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:])
        raise RuntimeError(f"{module} failed with exit code {completed.returncode}")
    tail = completed.stdout.strip().splitlines()[-3:]
    print("\n".join(tail) if tail else "completed")
print("All corrected local analyses completed.")"""
    )
)

cells.append(
    code(
        r"""#@title B2 · Optional remote paired DPO scan (~2.7 GB streamed through DuckDB)
if not RUN_REMOTE_DPO_SCAN:
    path = REPO_ROOT / "out/results/dpo_pairwise.json"
    if not path.exists():
        raise FileNotFoundError("No cached dpo_pairwise.json; enable RUN_REMOTE_DPO_SCAN")
    print("Using released paired DPO artifact:", path)
else:
    subprocess.run([sys.executable, "-m", "experiments.dpo_pairwise"], cwd=REPO_ROOT, check=True)
    subprocess.run([sys.executable, "-m", "experiments.multiple_testing"], cwd=REPO_ROOT, check=True)
    print("Remote paired DPO scan completed and BH results refreshed.")"""
    )
)

cells.append(
    code(
        r"""#@title B3 · Compact results dashboard
results_dir = REPO_ROOT / "out" / "results"
dpo = json.loads((results_dir / "dpo_pairwise.json").read_text(encoding="utf-8"))
thresholds = json.loads((results_dir / "threshold_robustness.json").read_text(encoding="utf-8"))
testing = json.loads((results_dir / "multiple_testing.json").read_text(encoding="utf-8"))

dpo_rows = []
for model, block in dpo.items():
    for behavior, values in block["behaviors"].items():
        dpo_rows.append({
            "model": model,
            "behavior": behavior,
            "chosen_only": values["chosen_only"],
            "rejected_only": values["rejected_only"],
            "word_rate_ratio": values["token_normalized_ratio"],
            "mcnemar_log10_p": values["mcnemar_log10_p"],
        })
display(pd.DataFrame(dpo_rows))

print("Threshold contrasts:")
threshold_rows = []
for model, by_threshold in thresholds["models"].items():
    for threshold, values in by_threshold.items():
        threshold_rows.append({"model": model, "threshold": threshold, **values})
display(pd.DataFrame(threshold_rows))
print("Multiple-testing families:", list(testing))"""
    )
)

cells.append(
    markdown(
        r"""## C. Provenance-safe human validation v3

The annotator sees only a stable item ID, prompt (for response-level items), and
text. Detector predictions, model stage, prompt condition, and the other
annotator's file are separate and hidden.

Two distinct humans must each run the initialization and UI cells with their own
`ANNOTATOR_ID`. AI agents may be used only for the separately marked diagnostic
audit; they must never select the human-certification checkbox.

For the current deadline, `sentence_120` reproduces the prepared stratified
sample. For the strongest paper, run the full pipeline and choose
`response_stagewise`, which validates the answer-level measurement across
models, stages, and conditions and adds a detector-positive enriched precision
sample."""
    )
)

cells.append(
    code(
        r"""#@title C1 · Prepare blinded items and hidden key
from experiments import validation_v3

if VALIDATION_MODE == "sentence_120":
    manifest = validation_v3.prepare_sentence(
        REPO_ROOT / "out/results/validation_sheet_v2.csv", VALIDATION_DIR
    )
else:
    manifest = validation_v3.prepare_response(
        RUN_PROJECT / "answers",
        RUN_PROJECT / "behavior",
        VALIDATION_DIR,
        random_per_stratum=8,
        positive_per_category=40,
        threshold=0.82,
        seed=0,
    )
print(json.dumps(manifest, indent=2))
print("Blinded items:", VALIDATION_DIR / "validation_items_v3.csv")
print("Hidden scoring key (annotators must not open):", VALIDATION_DIR / "validation_key_v3.csv")"""
    )
)

cells.append(
    code(
        r"""#@title C2 · Initialize one certified human annotator file
if not I_AM_A_HUMAN_ANNOTATOR:
    print("Not initialized. A real human annotator must set I_AM_A_HUMAN_ANNOTATOR=True and provide ANNOTATOR_NAME.")
elif not ANNOTATOR_NAME.strip():
    raise ValueError("Set a non-empty pseudonymous ANNOTATOR_NAME")
else:
    annotation_path = validation_v3.init_annotation(
        int(ANNOTATOR_ID), ANNOTATOR_NAME.strip(), True, True, VALIDATION_DIR
    )
    print("Human annotation file:", annotation_path)
    print("Work independently. Do not open validation_key_v3.csv or the other annotator's file.")"""
    )
)

cells.append(
    code(
        r"""#@title C3 · Blinded in-notebook annotation UI (autosaves per item)
import csv
try:
    from IPython.display import display, clear_output
    import ipywidgets as widgets
except ImportError:
    widgets = None
    print("Interactive widgets are unavailable in this kernel; run this cell in Colab or edit the separate annotation CSV directly.")

def launch_annotation_ui(annotator: int, directory: Path):
    if widgets is None:
        return
    annotation_path, _ = validation_v3.annotation_paths(annotator, directory)
    if not annotation_path.exists():
        print("Initialize the certified annotation file in C2 first.")
        return
    items = validation_v3.read_csv(directory / "validation_items_v3.csv")
    annotations = validation_v3.read_csv(annotation_path)
    by_id = {row["item_id"]: row for row in annotations}
    state = {"index": 0}
    categories = validation_v3.CATEGORIES
    display_names = {
        "empathy_opener": "empathy expression",
        "validation": "validation",
        "disclaimer": "professional referral",
        "crisis_referral": "crisis referral",
        "structure": "structuring language",
    }
    progress = widgets.HTML()
    item_box = widgets.HTML(layout=widgets.Layout(border="1px solid #888", padding="12px"))
    checks = {category: widgets.Checkbox(description=display_names[category]) for category in categories}
    none_box = widgets.Checkbox(description="none of the above")
    message = widgets.HTML()
    previous = widgets.Button(description="Previous")
    save_next = widgets.Button(description="Save + next", button_style="success")

    def completed_count():
        return sum(bool(row.get("label", "").strip()) for row in annotations)

    def render():
        item = items[state["index"]]
        label = by_id[item["item_id"]].get("label", "")
        selected = validation_v3.parse_label(label)
        for category, checkbox in checks.items():
            checkbox.value = category in selected
        none_box.value = label.strip() == "none"
        prompt = item.get("prompt", "").strip()
        prompt_html = f"<b>Prompt</b><br>{prompt}<hr>" if prompt else ""
        item_box.value = (
            f"<b>Item {state['index'] + 1}/{len(items)}</b> &nbsp; <code>{item['item_id']}</code><br><br>"
            f"{prompt_html}<b>Text to label</b><br><pre style='white-space:pre-wrap'>{item['text']}</pre>"
        )
        progress.value = f"<b>{completed_count()}/{len(items)} labeled</b>"
        message.value = ""

    def save_current():
        chosen = [category for category, checkbox in checks.items() if checkbox.value]
        if none_box.value and chosen:
            raise ValueError("none cannot be combined with another category")
        label = "none" if none_box.value else ",".join(sorted(chosen))
        label = validation_v3.canonical_label(label)
        item_id = items[state["index"]]["item_id"]
        row = by_id[item_id]
        row["label"] = label
        row["completed_at"] = validation_v3.utc_now()
        validation_v3.write_csv(
            annotation_path,
            annotations,
            ["item_id", "label", "annotator_id", "annotator_type", "completed_at"],
        )

    def on_previous(_):
        if state["index"] > 0:
            state["index"] -= 1
        render()

    def on_save_next(_):
        try:
            save_current()
            if state["index"] < len(items) - 1:
                state["index"] += 1
            render()
        except Exception as exc:
            message.value = f"<b style='color:#c33'>{exc}</b>"

    previous.on_click(on_previous)
    save_next.on_click(on_save_next)
    display(progress, item_box, widgets.VBox(list(checks.values()) + [none_box]), widgets.HBox([previous, save_next]), message)
    render()

launch_annotation_ui(int(ANNOTATOR_ID), VALIDATION_DIR)"""
    )
)

cells.append(
    code(
        r"""#@title C4 · Score two complete human passes and export disagreements
validation_report = validation_v3.score(VALIDATION_DIR)
print("status:", validation_report["status"], "| n:", validation_report.get("n", validation_report.get("n_items")))
if "exact_set_agreement" in validation_report:
    print("exact-set agreement:", validation_report["exact_set_agreement"])
    print("Cohen's kappa:", validation_report["cohen_kappa"])
if "detector_vs_adjudicated_human" in validation_report:
    display(pd.DataFrame(validation_report["detector_vs_adjudicated_human"]).T)
elif "detector_vs_human_1" in validation_report:
    display(pd.DataFrame(validation_report["detector_vs_human_1"]).T)
if validation_report["status"].startswith("pending"):
    print("reason:", validation_report.get("reason", "complete both human files"))
    print("FINAL HUMAN RESULT NOT CREATED. Complete both certified annotation files first.")
elif validation_report["status"].startswith("complete_unadjudicated"):
    print("Both passes complete. Review validation_disagreements_v3.csv, then run C5.")"""
    )
)

cells.append(
    code(
        r"""#@title C5 · Initialize adjudication and finalize after disagreements are resolved
if not (VALIDATION_DIR / "validation_disagreements_v3.csv").exists():
    print("No disagreement file yet. Complete C4 first.")
else:
    adjudication_path = validation_v3.init_adjudication(VALIDATION_DIR)
    adjudication_rows = validation_v3.read_csv(adjudication_path)
    print("Adjudication file:", adjudication_path)
    print("Fill label and decision_rule for every row, then rerun C4.")
    display(pd.DataFrame(adjudication_rows).head(20))"""
    )
)

cells.append(
    markdown(
        r"""## D. Optional causal leave-cluster-out SFT pilot

This section is opt-in. It starts from `allenai/OLMo-2-0425-1B` and uses the
**exact model-specific 1B SFT mixture**
`allenai/tulu-3-sft-olmo-2-mixture-0225`, fixing the generic-mixture error in the
older pilot notebook. It compares full subset SFT, behavioral-cluster removal,
coherent distractor removal, and random removal.

The default subset/one-seed configuration is a feasibility pilot, not a final
causal publication claim. A publishable result needs the official recipe,
multiple seeds, matched removal controls, and the frozen human-validated outcome
measure."""
    )
)

# Causal cells 4..9 are config, detector, selector, training, run, and results.
for index in range(4, 10):
    source = source_cell(causal, index)
    source = source.replace(
        'SFT_SOURCE   = "allenai/tulu-3-sft-mixture"',
        'SFT_SOURCE   = "allenai/tulu-3-sft-olmo-2-mixture-0225"',
    )
    cells.append(code(guarded(source, "RUN_CAUSAL_PILOT", "CAUSAL_PROJECT")))

cells.append(
    code(
        r"""#@title E · Final status and export bundle
from zipfile import ZipFile, ZIP_DEFLATED

status_path = VALIDATION_DIR / "validation_status_v3.json"
validation_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"status": "not_started"}
final_status = {
    "artifact_statistics": "complete",
    "paired_dpo": "complete" if (REPO_ROOT / "out/results/dpo_pairwise.json").exists() else "missing",
    "human_validation": validation_status.get("status", "not_started"),
    "causal_ablation": "pilot_completed" if RUN_CAUSAL_PILOT and (CAUSAL_PROJECT / "results").exists() else "not_run",
    "submission_rule": "Do not claim completed human validation unless status is complete_adjudicated_two_human_validation.",
}
print(json.dumps(final_status, indent=2))

bundle = RUN_PROJECT / "blackboxnlp_results_bundle.zip"
with ZipFile(bundle, "w", ZIP_DEFLATED) as archive:
    for path in sorted((REPO_ROOT / "out/results").glob("*.json")):
        archive.write(path, arcname=f"paper_results/{path.name}")
    for path in sorted(list(VALIDATION_DIR.glob("*.json")) + list(VALIDATION_DIR.glob("*.csv"))):
        archive.write(path, arcname=f"validation_v3/{path.name}")
print("Export bundle:", bundle)"""
    )
)

for index, cell in enumerate(cells):
    cell["id"] = f"blackboxnlp-{index:02d}"
template["cells"] = cells
template["metadata"] = {
    **template.get("metadata", {}),
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "A100"},
}
template["nbformat"] = 4
template["nbformat_minor"] = template.get("nbformat_minor", 5)

payload = json.dumps(template, indent=1, ensure_ascii=False)
OUTPUT.write_text(payload, encoding="utf-8")
MIRROR.write_text(payload, encoding="utf-8")
print(f"Wrote {OUTPUT} and {MIRROR} with {len(cells)} cells")

# Auditable machine-advice trace

This repository contains the code and artifacts for an observational study of
where advice-response behaviors appear across OLMo-2's open training pipeline.
It also includes a small interactive tracer for inspecting individual answers.

## Paper artifacts

The authoritative result bundle is `out/results/summary_v3.json`, supplemented
by the completed strict matched-stage report. The paper uses 140 prompts in five
conditions, five canonical final-model generations per prompt, a four-seed
Base/SFT/DPO matched-control extension, OLMo-2 1B and 7B checkpoints, and the
exact public SFT, DPO, and RLVR datasets associated with those checkpoints.

Important outputs:

- `out/paper/acl_latex.pdf`: current anonymous manuscript.
- `out/results/summary_v3.json`: canonical behavioral results.
- `out/results/interaction_tests_v3.json`: stage-by-condition permutation tests.
- `out/results/matched_control_stagewise.json`: complete 24-test matched-stage
  extension at the frozen primary threshold.
- `out/results/matched_control_threshold_sweep.json`: matched-stage sensitivity
  reports at 0.80, 0.82, and 0.84.
- `out/results/multiple_testing.json`: Benjamini-Hochberg corrections.
- `out/results/dpo_pairwise.json`: paired McNemar DPO analysis with descriptive
  per-word normalization.
- `out/results/threshold_robustness.json`: detector-threshold contrasts.
- `out/results/audit_cluster_bootstrap.json`: response-cluster uncertainty for
  the AI-only 120-sentence detector audit.
- `out/results/validation_sheet_v2.csv`: blinded stratified validation sample.
- `output/jupyter-notebook/blackboxnlp_complete_pipeline.ipynb`: complete
  Google Colab pipeline, including regeneration, corrected statistics,
  validation v3, and the optional causal pilot.
- `output/submission/blackboxnlp_anonymous_submission.zip`: identity-scanned,
  self-contained review PDF, source, code, and machine-readable artifacts.

## Complete Google Colab workflow

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mpvasilis/llm-research/blob/main/output/jupyter-notebook/blackboxnlp_complete_pipeline.ipynb)

Open the complete notebook with the badge and choose **Runtime > Run all**. It
clones this public repository automatically, installs its dependencies, and
runs the CPU-friendly artifact-reproduction path in cell order. Three expensive
operations remain explicit opt-ins:

- `RUN_FULL_PIPELINE`: checkpoint generation and full corpus trace;
- `RUN_REMOTE_DPO_SCAN`: the approximately 2.7 GB paired DuckDB scan;
- `RUN_CAUSAL_PILOT`: leave-cluster-out re-SFT on the exact 1B mixture.

The notebook checkpoints to Google Drive and writes a single results archive.
Its validation UI never exposes the condition, detector prediction, or the
other annotator's decisions.

The reviewer-requested four-seed publication extension is complete. To rerun or
extend it, connect an A100 runtime, set
`RUN_FULL_PIPELINE=True`, leave `FULL_QUICK_TEST=False`, and then choose
**Runtime > Run all**. The persistent Drive cache skips the released 4,550
checkpoint generations and, with `FULL_MATCHED_CONTROL_SEEDS=4`, adds the 1,200
missing Base/SFT/DPO matched-control generations. Fixed GPU prompt batches,
vectorized detector scoring, and parallel threshold analyses reduce runtime.
The analysis then requires all four selected seeds for every prompt, runs
24 prompt-clustered difference-in-differences tests at the frozen 0.82 detector
threshold, repeats them at 0.80 and 0.84, applies Benjamini-Hochberg correction,
and emits machine-readable JSON plus paper-ready Markdown and LaTeX tables. An
incomplete cache is labeled as a smoke test and cannot produce the tables.
Changing `FULL_MATCHED_CONTROL_SEEDS` to 5 later reuses everything and generates
only the remaining 300 controls for the stronger five-seed sensitivity run.

The older `colab_*_v2.ipynb` notebooks are retained for provenance. Do not use
their shared-column validation cells for final results; validation v3 in the
complete notebook supersedes them.

## Reproduce the paper statistics

Use Python 3.11+ in a clean environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-paper.txt

python -m experiments.monte_carlo_correction
python -m experiments.derive_paper_stats
python -m experiments.between_condition_stats
python -m experiments.stage_condition_gaps
python -m experiments.multiple_testing
python -m experiments.threshold_robustness
python -m experiments.audit_cluster_bootstrap
python -m experiments.render_interaction
```

The stronger DPO analysis scans about 2.7 GB of public Hugging Face parquet data
remotely with DuckDB and does not save those shards in the repository:

```powershell
python -m experiments.dpo_pairwise
python -m experiments.multiple_testing
```

To complete the human-validation gate, follow
`out/paper/ANNOTATION_GUIDE.md` or use the complete Colab notebook. Validation
v3 splits the blinded items, hidden scoring key, and two annotators into
different files. It refuses to produce a final result without two complete,
distinct, explicitly human-certified passes:

```powershell
python -m experiments.validation_v3 prepare-sentence
python -m experiments.validation_v3 init --annotator 1 --name annotator-a --certify-human
python -m experiments.validation_v3 init --annotator 2 --name annotator-b --certify-human
# Fill the two separate CSVs independently, preferably with the Colab UI.
python -m experiments.validation_v3 score
python -m experiments.validation_v3 init-adjudication
# Fill adjudicated_annotations_v3.csv, including decision rules, then score again.
python -m experiments.validation_v3 score
```

`out/results/validation_sheet.csv` and `detector_validation.json` are a legacy
tool-assisted diagnostic draft and are not human-validation evidence. The two
AI annotation files are also diagnostic only and never enter validation v3.

Compile the manuscript from `out/paper/` with the official ACL style:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex
bibtex acl_latex
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex
pdflatex -interaction=nonstopmode -halt-on-error acl_latex.tex
```

## Individual-answer tracer

The tracer searches answer phrasing in an infini-gram pretraining index and can
search public instruction-tuning parquet data with DuckDB. Closed-model traces
are explicitly marked as public-web proxies, not claims about private training
data.

Set up an optional Hugging Face read token:

```powershell
Copy-Item .env.example .env
python -m pip install -r requirements.txt
```

Run the local interface:

```powershell
python serve.py
```

Or use the CLI:

```powershell
python cli.py --question "How does photosynthesis work?" --answer data/answer.txt
```

Outputs are written to `out/report.md`, `out/trace.json`, and `out/history/`.

## Repository layout

```text
trace/          tracer and corpus-search implementation
experiments/    paper analyses, notebook builders, and validation UI
out/results/    machine-readable paper results
out/paper/      ACL manuscript, bibliography, and submission notes
```

Do not include `.env`, cached parquet shards, model weights, or local history in
an anonymous review supplement.

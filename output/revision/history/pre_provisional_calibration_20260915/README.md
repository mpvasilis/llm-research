# Measuring advice-style language in OLMo-2

This repository studies how detection rules and response length affect
comparisons of advice-style language across OLMo-2 post-training. It contains
the historical analyses, the current revision work, and a small interactive
tracer for inspecting individual answers.

## September 2026 revision status

The revision is an evidence-bounded work in progress. The revised manuscript
clearly separates historical diagnostics from planned prospective evidence:

- [revised PDF](output/pdf/advice_marker_revision_2026-09-15.pdf) and
  [LaTeX source](out/paper/acl_latex.tex);
- [plain-language explanation](out/paper/PAPER_EXPLAINED_2026-09-15.md),
  [revision plan](out/paper/REVISION_AND_RESUBMISSION_PLAN_2026-09-14.md),
  [review-response matrix](out/paper/REVIEW_RESPONSE_MATRIX_2026-09-15.md), and
  current [revision progress report](out/paper/REVISION_PROGRESS_2026-09-15.md);
- [completed exploratory DPO analysis](out/paper/DPO_CONDITIONAL_RESULTS_2026-09-15.md),
  covering 756,642 preference pairs, with
  [independent numerical verification](output/revision/verification/independent_dpo_numerical_check_20260915.json);
- [v4 annotation guide](out/paper/ANNOTATION_GUIDE.md) and
  [validation workflow](experiments/validation_v4.py);
- [human prompt-review worksheet](out/paper/PROMPT_REVIEW_WORKSHEET_2026-09-15.md)
  and [blank review manifest](output/revision/full_study/human_prompt_review_v2.json),
  covering the 40 fresh evaluation families;
- [independent readability protocol](out/paper/EXTERNAL_READABILITY_PROTOCOL_2026-09-15.md)
  and separate blank forms for [reader 1](output/revision/readability_v1/blank_reader_1.md)
  and [reader 2](output/revision/readability_v1/blank_reader_2.md), bound to the
  current PDF; both human readings remain pending;
- [versioned detector specification](experiments/detector_spec.py),
  [detector manifest](output/revision/study_v1/detector_manifest_v1.json), and
  [historical-cache audit](output/revision/study_v1/historical_cache_audit_v1.json),
  with [union-prediction parity audit](output/revision/study_v1/historical_union_prediction_audit_v2.json);
- [draft prospective prompt blocks](experiments/prompts_revision_v1.json),
  [active full-run protocol v2](output/revision/full_study/protocol_v2.json),
  [six-checkpoint calibration protocol](out/paper/CALIBRATION_PROTOCOL_2026-09-15.md),
  [prospective human-validation protocol](out/paper/PROSPECTIVE_VALIDATION_PROTOCOL_2026-09-15.md),
  [checkpoint-analysis protocol](out/paper/CHECKPOINT_ANALYSIS_PROTOCOL_2026-09-15.md), and
  [resource inventory](output/revision/study_v1/resource_inventory_v1.json);
- [historical final-7B response frame](output/revision/annotation_v4/historical_final7b_response_frame_v4.jsonl),
  [full-response union predictions](output/revision/study_v1/historical_final7b_union_predictions_v2.jsonl),
  and [112-item blank annotation packet](output/revision/annotation_v4/historical_packet_v2/validation_manifest_v4.json),
  all diagnostic only with reused construction prompts flagged.

The [extended study notebook](output/jupyter-notebook/advice_marker_extended_study.ipynb)
and its [source bundle](output/jupyter-notebook/advice_marker_extended_study_bundle.zip)
prepare the successor workflow for explicitly chosen caps above 512 tokens. The
[completed-probe companion](output/jupyter-notebook/advice_marker_completed_probe_bundle.zip)
contains the verified eight-response exploratory run and export. Follow the
[extended notebook guide](out/paper/EXTENDED_NOTEBOOK_GUIDE_2026-09-15.md) for extraction
and prerequisites. Both caps are unset and all 21 action controls default to off.
Local execution with those defaults is verified; dependency installation, GPU
execution, human validation, and the full empirical study are not certified by
that check. No replacement budget or successor empirical protocol has been selected.

The preserved [v2 revision notebook](output/jupyter-notebook/advice_marker_revision_study.ipynb)
runs from the accompanying [local source bundle](output/jupyter-notebook/advice_marker_revision_study_bundle.zip).
Upload the notebook to Colab, then upload that exact bundle when prompted. It
contains the preserved v2 workspace sources; cloning the public main
branch is not a substitute. Generation and analysis start disabled. The full
evaluation requires completed human prompt review, a sampling plan frozen from
actual human development calibration and timing, and suitable compute. The
notebook includes the fixed 1,024-token development probe and the separate
96-response calibration workflow; neither creates human judgments.

Validate the draft prompts with:

```powershell
python experiments/revision_study.py validate-prompts
```

The larger local development run uses two complete scenario families, four
variants each, the pinned 1B Instruct model, and a proposed 256-token limit.
The command resumes previously saved compatible answers. This is exploratory
development, not the 2,880-response evaluation:

```powershell
python -m experiments.full_revision_study run `
  --protocol output/revision/full_study/protocol_v2.json `
  --mode development-256 `
  --family-ids dev_health_hiking_blister dev_everyday_closet `
  --output-root output/revision/full_study `
  --device cpu --torch-threads 4 --local-files-only --execute
```

Both matched eight-response development runs completed and passed the saved-response
and resume checks. All eight responses reached the 256-token cap; six reached the
512-token cap and two stopped at EOS. The subsequent fixed 1,024-token probe
also completed: all eight answers reached EOS at 454–744 tokens, and a true resume
generated zero new responses while preserving every run-file hash. This supports
considering 1,024 for broader six-checkpoint development calibration; it does not
establish study-wide budget adequacy or semantic answer completeness. See the
[budget comparison](out/paper/DEVELOPMENT_BUDGET_REPORT_2026-09-15.md),
[fixed 1,024-token probe protocol](out/paper/DEVELOPMENT_BUDGET_PROBE_PROTOCOL_2026-09-15.md), and
[progress report](out/paper/REVISION_PROGRESS_2026-09-15.md) for the evidence and
remaining requirements.

Prepare the historical diagnostic annotation packet from the frozen response
frame and its versioned full-response predictions with:

```powershell
python -m experiments.validation_v4 prepare `
  --responses output/revision/annotation_v4/historical_final7b_response_frame_v4.jsonl `
  --predictions output/revision/study_v1/historical_final7b_union_predictions_v2.jsonl `
  --split historical_diagnostic `
  --out-dir output/revision/annotation_v4/historical_packet_v2 `
  --random-per-stratum 8 --positive-per-category 20 --seed 0
```

Two independent human annotation passes and adjudication remain pending. The
prospective multi-checkpoint generation also remains pending suitable compute.
Until those are complete, the historical packet cannot establish a fresh test
result or validate stagewise trajectories.

## Historical submitted-paper artifacts

The artifacts in this section reproduce the prior submission and remain
historical snapshots. Its result bundle is `out/results/summary_v3.json`, supplemented
by the completed strict axis-targeted-control report. The paper uses 140 prompts in five
conditions, five canonical final-model generations per prompt, a four-seed
Base/SFT/DPO axis-targeted-control extension, OLMo-2 1B and 7B checkpoints, the
public post-training datasets, and indexed Stage-1 pretraining data.

Important outputs:

- `out/paper/acl_latex.pdf`: historical anonymous manuscript snapshot.
- `out/results/summary_v3.json`: canonical frozen-proxy results.
- `out/results/interaction_tests_v3.json`: stage-by-condition permutation tests.
- `out/results/matched_control_stagewise.json`: complete 24-test axis-targeted
  extension at the frozen primary threshold.
- `out/results/matched_control_threshold_sweep.json`: axis-targeted sensitivity
  reports at 0.80, 0.82, and 0.84.
- `out/results/multiple_testing.json`: Benjamini-Hochberg corrections.
- `out/results/dpo_pairwise.json`: paired McNemar DPO analysis with descriptive
  per-word normalization.
- `out/results/dpo_phrase_robustness.json`: DPO effect intervals, high-precision
  referral patterns, source/phrase checks, prompt duplication, and within-pair
  length strata.
- `out/results/raw_output_robustness.json`: final-7B detector components,
  fixed-token windows, response lengths, joint patterns, and pilot exclusion.
- `out/raw_cache/final_7b_answers.jsonl`: recovered 700-response final-7B cache;
  raw generations for other checkpoints are not available here.
- `out/results/threshold_robustness.json`: detector-threshold contrasts.
- `out/results/audit_cluster_bootstrap.json`: response-cluster uncertainty for
  the AI-only 120-sentence detector audit.
- `out/results/validation_sheet_v2.csv`: blinded stratified validation sample.
- `output/jupyter-notebook/blackboxnlp_complete_pipeline.ipynb`: complete
  Google Colab pipeline, including regeneration, corrected statistics,
  validation v3, and the optional causal pilot.
- `output/submission/blackboxnlp_anonymous_submission.zip`: historical
  identity-scanned submission snapshot with review PDF, source, code, and
  machine-readable artifacts.

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

The four-seed axis-targeted extension is complete. To rerun or
extend it, connect an A100 runtime, set
`RUN_FULL_PIPELINE=True`, leave `FULL_QUICK_TEST=False`, and then choose
**Runtime > Run all**. The persistent Drive cache skips the released 4,550
checkpoint generations and, with `FULL_MATCHED_CONTROL_SEEDS=4`, adds the 1,200
missing Base/SFT/DPO axis-targeted-control generations. Fixed GPU prompt batches,
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

The DPO audits scan about 2.7 GB of public Hugging Face parquet data
remotely with DuckDB and does not save those shards in the repository:

```powershell
python -m experiments.dpo_pairwise
python -m experiments.dpo_phrase_robustness
python -m experiments.multiple_testing
```

Recompute the recovered final-7B response stress tests with:

```powershell
python -m experiments.raw_output_robustness
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

Compile the current manuscript from the repository root so its figure paths
resolve correctly. Use a separate build directory to preserve the historical
submitted PDF and bibliography:

```powershell
New-Item -ItemType Directory -Force output/revision/manuscript_build
Copy-Item out/paper/acl.sty,out/paper/acl_natbib.bst,out/paper/custom.bib output/revision/manuscript_build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/revision/manuscript_build out/paper/acl_latex.tex
Push-Location output/revision/manuscript_build
bibtex acl_latex
Pop-Location
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/revision/manuscript_build out/paper/acl_latex.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/revision/manuscript_build out/paper/acl_latex.tex
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

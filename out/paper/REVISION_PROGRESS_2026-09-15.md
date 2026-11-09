# Revision progress — 15 September 2026

The manuscript revision and paired preference-data analysis are complete. The higher-budget workflow is verified, and its provisional development design is now frozen. **Independent human evaluation, the 96-response calibration run, and the full checkpoint experiment remain incomplete.** The paper is not ready for resubmission.

## Current deliverables

- [Revised paper](../../output/pdf/advice_marker_revision_2026-09-15.pdf), [LaTeX source](acl_latex.tex), and [plain-language explanation](PAPER_EXPLAINED_2026-09-15.md).
- [Reviewer-response matrix](REVIEW_RESPONSE_MATRIX_2026-09-15.md) and [revision/resubmission plan](REVISION_AND_RESUBMISSION_PLAN_2026-09-14.md).
- [Provisional development budget decision](../../output/revision/extended_study/development_budget_decision_v1.json), [frozen successor protocol](../../output/revision/extended_study/full_study/protocol_v1.json), and [exact 96-response calibration manifest](../../output/revision/extended_study/calibration_study/manifest.json).
- [Successor-bound prompt worksheet](../../output/revision/extended_study/prompt_review_worksheet.md) and [blank review form](../../output/revision/extended_study/human_prompt_review_template_v1.json). All 40 families and 200 judgment fields remain unreviewed.
- [Extended notebook](../../output/jupyter-notebook/advice_marker_extended_study.ipynb), [verified source bundle](../../output/jupyter-notebook/advice_marker_extended_study_bundle.zip), [frozen preparation companion](../../output/jupyter-notebook/advice_marker_frozen_preparation_bundle.zip), and [current calibration guide](FROZEN_CALIBRATION_GUIDE_2026-09-15.md). Clean Windows dependency installation and imports passed; GPU generation remains untested.

## What changed in the paper

The paper now asks one question: **How do detection rules and response length change conclusions about advice-style language across OLMo-2 post-training?** The abstract, introduction, results, and conclusion follow that question. Definitions distinguish the language feature from the detection rule. A concrete example shows why a match can be misleading. The evidence map separates prompts, responses, AI-labeled sentences, and preference pairs.

The manuscript integrates the completed DPO analysis, gives 1B and 7B explicit labels, improves the main tables and figure, and limits claims to the evidence. Base-to-SFT remains a combined stage/interface transition. Phrase co-occurrence is not presented as causal attribution. The historical AI-label diagnostic remains visibly separate from human validation.

The PDF has 15 pages, including eight main pages with complete Limitations and Ethics sections. Its abstract has 161 words. All pages were visually checked; all 31 internal references were audited. There are no unresolved references or overfull boxes. The current PDF hash is `559013886fffb86d63600b8d8f33ce328c0f2c0c207b31929eb1223633a3421a`. Two separate [reader forms](EXTERNAL_READABILITY_PROTOCOL_2026-09-15.md) are bound to it; zero human readings are complete.

## Completed empirical analyses

The paired DPO analysis retained all **756,642 preference pairs**: 378,301 from the 1B mixture and 378,341 from the 7B mixture. Feature extraction reproduced the historical counts and length bands before fitting. The models preserve chosen/rejected pairing, represent both response lengths flexibly, and account for repeated exact prompts in their uncertainty estimates.

| Phrase group | 1B conditional OR [95% CI] | 7B conditional OR [95% CI] |
|---|---:|---:|
| Broad referral/disclaimer | 1.088 [1.026, 1.154] | 1.134 [1.067, 1.205] |
| Narrower referral | 1.095 [0.972, 1.234] | 1.121 [0.996, 1.263] |
| Structuring | 1.128 [1.108, 1.149] | 1.190 [1.168, 1.213] |

Broad referral/disclaimer and structuring remain positively associated with the preferred response after conditioning on length. Both narrower-referral intervals include one. These are exploratory conditional associations, not causal effects or quality judgments. The mixtures share 21,508 content-identical pairs, and 19,890 7B pairs lack source metadata; source and prompt sensitivities are retained. Six tests and a separate optimizer/spline calculation verified the main estimates. See [results](DPO_CONDITIONAL_RESULTS_2026-09-15.md) and [independent numerical check](../../output/revision/verification/independent_dpo_numerical_check_20260915.json).

The recovered 700-response final-7B archive reproduced all 20 historical union-rate cells. The empathy contrast changes sign between the full response and its first 64 tokens, but the full-response estimate is uncertain: this is a point-estimate sign change, not two established effects in opposite directions. Missing stagewise raw archives have not been recovered.

## Budget evidence and the new development decision

| Matched eight-response run | Cap stops | EOS stops | Observed output lengths |
|---|---:|---:|---|
| 256-token cap | 8 | 0 | 256 each |
| 512-token cap | 6 | 2 | 454–512 |
| 1,024-token cap | 0 | 8 | 454–744 |

The 1,024-token probe completed at 17:49:52 UTC, produced 4,635 tokens, and took 7,888.19 seconds on the recorded CPU configuration. All eight responses passed strict verification; a true resume generated zero new responses and preserved all 13 run-file hashes. Every matched 512 decoded response is an exact prefix, and two are identical. Raw continuation token IDs were not retained, so this agreement is about decoded text. The complete [three-budget report](DEVELOPMENT_BUDGET_REPORT_2026-09-15.md) passed an independent 152-cell raw-data check.

The completed probe now supports the **provisional 1,024-token primary / 2,048-token sensitivity decision**. This supersedes the pending-decision status in the preserved earlier budget note. The successor is `protocol_b7378f38ecd5af173cd2fd12cf086416`; its calibration manifest is `extended_calibration_manifest_2e744d3df5602179882f17981b8e457a`.

The frozen calibration contains exactly 16 responses per checkpoint, covers all ten development families, and excludes every evaluation family. The predeclared acceptance criterion requires a complete verified 96-response archive, no cap or unexplained stops, and genuine development review without identified budget truncation. At least 60 distinct responses require genuine guide review, including a nonempty timed subset. Stopping summaries and human findings must receive an explicit scientific assessment before freezing the prospective plan; structural validators do not make that assessment. Failure requires a preserved, versioned development revision rather than mixed caps or selective exclusions.

Freezing this design generated no responses and supplied no human judgments. Both evaluation modes rejected the blank review before creating a run. Forty-two existing source/evidence/artifact files remained unchanged. See [freeze verification](../../output/revision/verification/extended_development_freeze_20260915.json) and the [independent design audit](../../output/revision/verification/extended_development_freeze_independent_20260915.json).

## What the implementation checks establish

The preserved v2 workflow passed 70 unique cases. Its notebook and source bundle remain unchanged. Those checks are distinct from the new successor verification:

- Thirteen successor core/calibration tests passed after the Windows path fix.
- Eight prospective cases passed. The strict ninth case reached its final intended rejection after full-grid, resume, relocation, and tamper checks; a test-only regex mismatch was closed by replaying that final public call against the exact retained fixture. The failed JUnits remain preserved; no single green nine-case invocation is claimed.
- Four sensitivity cases are covered by three unchanged initial passes and the corrected passing focused case. The first run had no JUnit; the focused run has one. Complete synthetic 2,880/8,640 primary and 192/576 sensitivity exports were independently rehashed and recounted.

All 12 scientific dependencies match the tested hashes. Deep Windows paths and independent sensitivity arithmetic were also checked. See [combined verification](../../output/revision/verification/extended_workflow_verification_20260915.json). Synthetic fixtures are not empirical study data.

The separate extended notebook and its source/probe packages passed member-hash, source-coverage, and local execution checks: 13 code cells, zero errors, unset caps, and all 21 actions off. This verifies preparation, not installation, model execution, or scientific completion. The new frozen-preparation companion also strict-loaded from a relocated extraction; the unchanged notebook ran all 13 code cells without errors against those frozen inputs with every action off. Its [verification record](../../output/revision/extended_notebook/frozen_preparation_verification.json) and [independent archive/notebook check](../../output/revision/extended_notebook/frozen_preparation_independent_verification_20260915.json) preserve the precise scope. The executed notebook is retained for inspection.

A fresh Windows virtual environment with Python 3.12.14 successfully installed the notebook's declared dependency ranges, resolving Transformers 4.57.6 and PyTorch 2.14.0+cpu. Notebook test tools were installed separately. Package consistency and all declared imports passed, with system site-packages disabled. The [import/isolation record](../../output/revision/environment_validation/import_check.json), [package check](../../output/revision/environment_validation/pip_check.log), and [resolved package list](../../output/revision/environment_validation/resolved_packages.json) are retained. This verifies this Windows configuration only; it does not establish Linux/GPU compatibility or inference success.

The same clean environment then passed the [six-checkpoint configuration/tokenizer check](../../output/revision/environment_validation/tokenizer_config_compatibility.json) using cached files only. Every pinned serialization matched the frozen protocol; all 200 prompt token counts agreed across checkpoints, and split extrema matched the earlier context audit. A separately run, relocated [clean-environment notebook](../../output/revision/environment_validation/notebook_check/executed_advice_marker_extended_study_clean_env.ipynb) executed all 13 code cells unchanged, with zero errors and all 21 actions off. The kernel used the exact isolated interpreter. Root independently checked the retained cells, execution counts, controls, artifact hashes, declared versions, six pins, all 12 scientific source hashes, and all 37 preparation payloads; the PDF and frozen human forms were unchanged. See the [independent verification](../../output/revision/verification/clean_environment_independent_verification_20260915.json). No model weights were loaded and no responses or human judgments were created.

## Work still required

1. Obtain genuine prompt review and two independent external readings of the current paper.
2. Run the exact 96-response calibration in one suitable environment with a uniform execution profile; complete human guide refinement, budget assessment, and timing; then freeze the prospective sampling plan.
3. Generate and strictly verify the 2,880 primary and 192 sensitivity responses. Produce the 8,640 and 576 detector rows and checkpoint/window analyses.
4. Complete two blinded human annotation passes and adjudication, calculate the planned uncertainty, integrate the actual findings, and reassess resubmission readiness.

The current environment has no CUDA device. Only the pinned 1B Instruct weight file is locally complete; the other five checkpoint snapshots lack weights. No original missing cache or connected GPU session has been located. Annotator availability and a cache/GPU location were already requested. A local 1B-only partial official calibration would not solve the six-checkpoint requirement and could not be silently combined with a later different execution profile.

The bounded recovery search now covers all nine reachable Git commits, local reflogs, all 48 unreachable blobs, and the remote's currently advertised heads/tags. No additional historical stagewise response archive was found. The remote advertises only the main commit already present locally. This does not cover pruned objects or inaccessible external storage. See the [reachable-history audit](../../output/revision/verification/historical_git_cache_recovery_20260915.json), [unreachable-object addendum](../../output/revision/verification/historical_git_unreachable_cache_recovery_20260915.json), and [remote-ref observation](../../output/revision/verification/historical_remote_refs_observation_20260915.json).

Official metadata at the six exact checkpoint revisions puts the missing weights and indexes at **46.32 GiB** and the complete six-checkpoint set at **49.08 GiB**. These are storage amounts, not runtime memory requirements. The 7B Instruct parameters alone occupy 27.19 GiB when loaded in float32, before framework, activation, and generation memory; about 14.77 GiB of RAM was free at the recorded observation. The D: drive is explicitly temporary storage and is excluded from durable experiment capacity. No new model weights were downloaded. See the [resource manifest](../../output/revision/verification/checkpoint_download_resource_manifest_20260915.json) and [independent arithmetic/pin check](../../output/revision/verification/checkpoint_download_resource_independent_20260915.json).

No paper has been submitted or published. Earlier status records are preserved under `output/revision/history/pre_provisional_calibration_20260915/` and `output/revision/history/pre_clean_environment_validation_20260915/`. Local preparation and verification are complete for this phase. The full objective remains unachieved and is blocked on suitable six-checkpoint compute and genuine human participation; no substantive local task remains that can supply that missing evidence.

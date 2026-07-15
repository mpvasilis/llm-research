# BlackboxNLP 2026 submission notes

Target: regular archival track, anonymous ACL review format.
Direct-submission deadline: July 17, 2026, 23:59 Anywhere on Earth.

## Completed analyses

- 140-prompt, five-condition final-model evaluation with five seeds per prompt.
- Base-to-SFT-to-DPO-to-RLVR/Instruct checkpoint trajectories.
- Prompt-level stage-by-condition difference-in-differences tests.
- Benjamini-Hochberg correction across the 24 matched-control tests, 28
  adjacent-stage tests, and 8 DPO pairwise tests.
- Plus-one correction for all Monte Carlo permutation p-values
  (`summary_v3.json`, `interaction_tests_v3.json`).
- Exact SFT assistant-only search with topical controls and distinct-phrase
  leave-one-out.
- DPO pair-level chosen-only/rejected-only McNemar tests and phrase occurrences
  per million response words (`dpo_pairwise.json`).
- Explicit 0.80/0.82/0.84 threshold-robustness contrasts
  (`threshold_robustness.json`).
- Complete Colab notebook with opt-in full regeneration, paired DPO scan,
  response/stage-level validation, and causal SFT pilot
  (`output/jupyter-notebook/blackboxnlp_complete_pipeline.ipynb`).
- Provenance-safe validation v3: separate blinded items/key, separate human
  annotation files, human certification, confidence intervals, disagreement
  export, and adjudication.
- Complete 120-sentence AI-only consensus audit, explicitly not described as
  human validation, plus a 10,000-draw Bayesian response-cluster bootstrap over
  its 97 source responses (`audit_cluster_bootstrap.json`).
- Anonymous eleven-page PDF: eight content/ethics pages, references beginning
  on page 8, and appendix continuing through page 11.
- Identity-scanned, self-contained anonymous review package that recompiles
  cleanly (`output/submission/blackboxnlp_anonymous_submission.zip`).

## Remaining submission gate

- [ ] Run the 1,200 missing four-seed Base/SFT/DPO matched-control generations on the
  signed-in Colab A100 runtime and require the strict 24-test result to report
  `status=complete`.
- [ ] If time permits, set `FULL_MATCHED_CONTROL_SEEDS=5`; the cache will add only
  the remaining 300 controls for a five-seed sensitivity run.
- [ ] Incorporate those verified numbers, rebuild the PDF and anonymous ZIP,
  and obtain a final independent review.
- [ ] Make the identifiable public development repository private for the
  double-blind review period. Do not link it in the submission. Git history,
  repository ownership, notebook URLs, and the searchable title reveal author
  identity even though the PDF and anonymous ZIP are clean.

Independent human validation remains a high-value follow-up, but it is not
represented as completed evidence in this submission. The legacy tool-assisted
draft and AI consensus audit cannot be called human validation.

## Packaging checklist

- [x] Anonymous author block and ACL review mode.
- [x] Exact local analysis dependency versions in `requirements-paper.txt`.
- [x] End-to-end reproduction commands in the root README.
- [x] Construct-valid reported names: empathy expression, validation,
  professional referral, structuring language.
- [ ] Record the original model-generation accelerator type and total GPU hours
  if available.
- [x] Create an anonymous supplementary archive excluding `.env`, cached parquet
  files, local model weights, history, and identity-bearing paths.
- [x] Recompile the anonymous source package and visually inspect the PDF.

## Camera-ready only

- Switch `\usepackage[review]{acl}` to final mode.
- Restore the author block.
- Add the public repository/artifact URL.

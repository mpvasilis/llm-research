# BlackboxNLP 2026 submission notes

Target: regular archival track, anonymous ACL review format.

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
- Anonymous ten-page PDF: seven content/ethics pages, references beginning on
  page 8, and appendix continuing through page 10.

## Remaining submission gate

- [ ] Two distinct people independently complete validation v3 using the
  complete Colab notebook and `ANNOTATION_GUIDE.md`.
- [ ] Run the validation-v3 scoring cell, adjudicate disagreements, and rerun.
- [ ] Add per-condition precision/recall, Cohen's kappa, and adjudication to the
  detector-validation section.
- [ ] Re-evaluate wording if detector false-positive rates differ materially by
  condition.

This human step must not be replaced with AI-generated labels.
The legacy 80-row tool-assisted draft and the two AI diagnostic passes are
explicitly excluded from the final human-validation result.

## Packaging checklist

- [x] Anonymous author block and ACL review mode.
- [x] Exact local analysis dependency versions in `requirements-paper.txt`.
- [x] End-to-end reproduction commands in the root README.
- [x] Construct-valid reported names: empathy expression, validation,
  professional referral, structuring language.
- [ ] Record the original model-generation accelerator type and total GPU hours
  if available.
- [ ] Create an anonymous supplementary archive excluding `.env`, cached parquet
  files, local model weights, history, and identity-bearing paths.
- [ ] Recompile and visually inspect the final PDF after human-validation edits.

## Camera-ready only

- Switch `\usepackage[review]{acl}` to final mode.
- Restore the author block.
- Add the public repository/artifact URL.

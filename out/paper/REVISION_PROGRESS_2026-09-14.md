# Revision implementation record

Work started 14 September 2026 following the BlackboxNLP rejection. This records implemented changes and remaining evidence requirements; it is not a declaration that the paper is ready to resubmit.

## Completed manuscript work

- Reframed the paper as **Measuring Advice-Style Language in OLMo-2: Effects of Detection Rules and Response Length**.
- Rewrote the abstract, introduction, contributions, results organization, conclusion, and limitations around the measurement question.
- Put detector quality before results that depend on it. Explicitly distinguished the broad historical professional-language rule from actual referrals, and response-wide empathy matching from opening-position claims.
- Made the DPO length sensitivity prominent; retained descriptive interpretations for checkpoint comparisons and phrase searches.
- Added verified related work on construct validity and response-length effects. The comparison note is in `RELATED_WORK_POSITIONING_2026-09-14.md`.
- Produced `output/pdf/advice_marker_revision_2026-09-14.pdf`. The revised PDF is 13 pages including references and appendices. Main text, limitations, and ethics end on page 8. The build has no unresolved references/citations or overfull boxes; rendered pages were visually inspected.
- Recorded completed and pending responses to individual reviewer concerns in `REVIEW_RESPONSE_MATRIX_2026-09-14.md`.

The older `out/paper/acl_latex.pdf` and BlackboxNLP submission archive remain historical snapshots. Use the revision PDF above when reviewing this work.

## Reproduction and validation work

- Preserved a hashed baseline of the working manuscript, detector code, bibliography, and annotation guide under `output/revision/baseline_2026-09-14/` before editing.
- Independently reran the recovered final-7B robustness analysis: all parsed results and the output file hash match the existing report exactly. Evidence: `output/revision/raw_reproduction_check.json`.
- Created a versioned detector specification that preserves historical variants and their differences. Exact matching agrees on all 700 recovered responses; all 20 full-response union category-by-condition rates also reproduce.
- Fixed legacy validation so missing predictions fail explicitly and targeted enrichment is reported separately from the random component.
- Implemented response-level validation v4 with separate human targets, blinded items, independent blank annotation passes, explicit sampling probabilities, scenario-family uncertainty, and checks against missing/duplicate/tampered records. Full-response human labels are not reused to validate truncated prefixes.
- Normalized the 700 historical responses with explicit missing historical metadata and construction-prompt flags. These data remain historical diagnostics, not a fresh evaluation set.
- Prepared the final historical starter packet at `output/revision/annotation_v4/historical_packet_v2/`: 112 unique blinded responses, comprising 40 probability-sample items and 75 detector-positive diagnostic items with three overlapping items. Nine items retain construction-prompt flags. Two independent annotation templates are blank and uncertified. This starter packet does not replace the planned prospective human evaluation.
- Recomputed all 700 full-response predictions with the embedding model revision explicitly pinned at load time. The new prediction records preserve that reproduction provenance while leaving the original historical run's unknown revision unknown. The final packet rejects mixed detector or dependency versions.

## New study preparation

- Drafted 50 scenario families: 40 evaluation families and 10 separate development families, balanced between health and everyday topics.
- Each family yields four prompts crossing emotional framing and request type. Context and request remain identical across emotional framing; one emotional clause is added. There are 160 evaluation prompts and 40 development prompts.
- Wrote a planned grid of 2,880 evaluation responses: six SFT/DPO/Instruct checkpoints, 160 prompts, and three seeds. This is a design artifact, not generated evidence. The independent replication for prompt effects is 40 evaluation families.
- The prompt set is explicitly AI-assisted and awaits human checks of the manipulations. The generation tool in this revision is limited to a local development smoke test; a full GPU study runner remains future work.
- The first local smoke run produced two 32-token development responses in approximately 113 seconds on the cached 1B Instruct model. Both stopped at the token cap. This checks execution only and does not assess behavior, prompt validity, or measurement accuracy.

## Remaining work

1. Have two independent people pilot the annotation instructions, then complete the appropriate blinded labels and adjudication. No human labels or certifications have been created by AI.
2. Human-check the draft scenario manipulations and freeze the evaluation design after development. Determine the generation cap and sampling allocation from development evidence.
3. Recover the original stage caches from the existing experiment storage or generate a clearly versioned new dataset with suitable compute. Only the final-7B historical raw cache is currently available locally; this machine has no CUDA GPU.
4. Implement and verify the full study runner, then run the retained checkpoint comparisons and detector/window sensitivity analyses. The current two-response smoke run does not fulfill this step.
5. Implement the pre-specified DPO conditional length analysis if retaining the stronger preference interpretation. Existing aggregate and post-hoc length-stratum results are unchanged.
6. Obtain independent readability feedback, replace pending evidence in the manuscript, and reassess venue/readiness against the revision plan.

No submission, external upload, or commit was made. Existing unrelated working-tree changes were preserved.

## Final verification

- All 25 tests passed across the revised study and validation workflows. Machine-readable test evidence is in `output/revision/verification/revision_tests.xml`.
- The hardened local smoke run, `run_4ce72bf186532d789c14`, generated two development responses in approximately 97 seconds. An actual repeat invocation generated zero new responses and preserved the response file hash. Identity validation passed for both records. Evidence: `output/revision/local_smoke_verification.json`.
- All eight baseline copies passed their recorded hashes. The original historical PDF, bibliography build output, raw robustness script, and tracer detector also remain byte-for-byte unchanged from the saved working baseline.
- The final 112-item packet passed independent checks of unique IDs, blinding, blank annotation fields, source hashes, and refusal to score without human passes. Evidence: `output/revision/annotation_packet_verification.json`.
- All 700 pinned-rescore prediction records passed schema and uniqueness checks; all 20 historical category-by-condition rates match. Evidence: `output/revision/study_v1/historical_union_prediction_audit_v2.json`.

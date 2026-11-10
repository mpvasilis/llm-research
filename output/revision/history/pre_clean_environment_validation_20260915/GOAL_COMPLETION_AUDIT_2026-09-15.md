# Completion audit for the continuing revision

Objective: complete the remaining revision work, reevaluate the evidence, and make the paper clearer and more understandable. **The full objective is not achieved.** The preceding work made concrete progress: the higher-budget integration checks closed, and the provisional successor and exact calibration design were frozen. Preparing these artifacts does not complete the experiments or human review.

| Requirement | Verified current state | Remaining completion evidence |
|---|---|---|
| Clear question and contribution | Revised abstract, introduction, results, conclusion, plain-language companion, and reviewer matrix; current PDF checked | Two unfamiliar human readers independently identify the intended question and findings; zero readings complete |
| Aligned definitions and detector targets | Historical detector preserved; v4 annotation targets and appendix reconciled | Genuine response-level human judgments and accuracy estimates |
| Historical reproduction | 700 final-7B responses; all 20 union-rate cells reproduced | Missing stagewise raw archives cannot be treated as recovered |
| Controlled prompt design | 40 evaluation families/160 prompts and ten development families/40 prompts; successor-bound worksheet and blank 40-family review | Independent human judgments of topic consistency and both manipulations |
| Generation budget | Matched 256/512/1,024 eight-response runs strictly verified; 1,024 reached EOS in all eight at 454–744 tokens; unchanged zero-new resume | Broader six-checkpoint calibration and explicit assessment of the predeclared budget criterion |
| Successor development design | Provisional caps 1,024 primary/2,048 sensitivity recorded; evidence-bound successor and exact 96-response manifest frozen; 16 rows per checkpoint; ten development families; no evaluation overlap | Actual 96-response generation and complete export; zero new calibration responses currently |
| Comparable generation and provenance | Six exact model/tokenizer pins, serialization checks, execution-profile binding, context preflight, complete-grid and resume validation | One suitable environment for all six checkpoints; final real archives must pass validation |
| Development guide and allocation | Sampling, timing, stratification, and adjudication workflow implemented and verified | At least 60 genuine reviewed calibration IDs, a nonempty timed subset, guide refinement and budget assessment, followed by prospective-plan freezing |
| Full primary response archive | Exact 2,880-response grid and strict runner implemented | Real complete archive; synthetic test fixtures do not count |
| Detector and output-window comparisons | Primary and matched longer-budget analysis implemented; historical final-7B diagnostic verified | Real 8,640 primary/576 sensitivity detector rows and checkpoint-wide analyses; full-response human labels do not validate prefixes |
| Longer-budget sensitivity | Exact 192-response subset at 2,048; matched analysis and arithmetic checked | Actual 192-response run and paired comparison with the corresponding primary responses |
| Independent human validation | Blank historical starter packet; prospective probability/enriched sampling and two-pass adjudication tools verified | Actual prospective sample, two distinct blinded human passes, adjudication, and sampling-aware estimates; no human passes complete |
| Paired DPO analysis | All 756,642 pairs; six conditional fits, source/prompt sensitivities, independent numerical reproduction | Complete for the current exploratory noncausal scope |
| Readable visuals and PDF | All 15 pages checked; eight main pages; 31 references audited; DPO estimates reconciled | Recheck after adding real prospective results and external reader feedback |
| Reproducible preparation | Preserved v2 artifact verified; successor source/notebook and completed-probe packages verified locally; final integration evidence retained | Clean-environment installation and GPU generation untested; final empirical supplement still pending |
| Resubmission decision | Dated route comparison and readiness criteria exist | Reassess after essential empirical and reading checks; no submission performed |

## Final implementation verification

Thirteen successor core/calibration tests passed with unchanged production and test hashes. Eight prospective cases passed; the strict ninth case completed its preceding assertions and raised the intended final readiness error, with only the test's case-sensitive regex failing. A separate passing replay against the exact retained fixture closed that assertion. Both failed JUnits remain failed records. The redundant third reconstruction was interrupted and contributes no passing evidence.

Sensitivity coverage combines three unchanged initial passes with one corrected passing focused case. The first invocation did not produce a JUnit; that limitation is recorded. Root independently parsed the available JUnits, checked all 12 scientific source hashes, and rehashed/recounted six retained synthetic exports. No single all-green invocation is claimed for these combined records.

Evidence: `output/revision/verification/extended_workflow_verification_20260915.json`. Deep-path and independent arithmetic checks are additional implementation evidence. They are not empirical results.

## Provisional freeze and acceptance rule

The successor is `protocol_b7378f38ecd5af173cd2fd12cf086416`; calibration is `extended_calibration_manifest_2e744d3df5602179882f17981b8e457a`. The decision records 1,024/2,048 before calibration or held-out generation. Require all 96 calibration responses, zero cap/unexplained stops, and genuine development review without identified budget truncation. A failure withholds readiness and requires a preserved, versioned development revision. Passing is operational support, not a guarantee of semantic completeness or corpus-wide adequacy.

The generation software checks structural provenance and human-evidence requirements; it does not certify the scientific budget assessment. That assessment must be recorded before the prospective plan is frozen. The protocol's current status explicitly remains pending human/prospective review.

The freeze created no response data, selected no future execution profile, and supplied no human judgments. All 200 prompt-review checks remain null. Both evaluation and sensitivity refused the blank review before run creation. Forty-two previous source/evidence/artifact files were unchanged. Evidence: `output/revision/verification/extended_development_freeze_20260915.json`. A separate direct-content audit also passed: `output/revision/verification/extended_development_freeze_independent_20260915.json`.

The frozen-preparation companion passed relocation checks with the existing source bundle. The unchanged notebook ran all 13 code cells with those frozen paths, zero errors, and all actions off. This verifies local preparation, not dependency installation or generation. Evidence: `output/revision/extended_notebook/frozen_preparation_verification.json`. Root independently verified all 37 companion payloads, all 12 scientific dependencies, and the retained 13-cell executed notebook with identical source, sequential execution counts, zero errors, and all 21 actions disabled: `output/revision/extended_notebook/frozen_preparation_independent_verification_20260915.json`.

The older pending-decision note and earlier notebook packages remain preserved snapshots. The current machine-readable decision and successor-bound review artifacts supersede their pending-cap/predecessor-protocol status for future work.

## External resources still required

A read-only audit found complete local model weights only for pinned 1B Instruct; the other five have configuration/tokenizer files but lack weights. No CUDA device or connected GPU session was found. A direct Colab check showed a signed-out session. No original missing stage cache was located. These are checks of the accessible environment, not claims about all possible storage or compute.

The configured 4,096-token context limit accommodates the inspected 32–64-token inputs with both provisional budgets. This proves a configuration margin only, not runtime memory fit, generation success, or budget adequacy. The completed CPU probe cannot establish 7B or GPU throughput. Official calibration should use one suitable six-checkpoint environment rather than combine incompatible partial execution profiles.

Human annotator availability and an original-cache/GPU location were already requested. They remain unresolved. No external messages, uploads, remote runs, submissions, or publication occurred in this phase.

## Scope and completion decision

Base/SFT serialization crossover, RLVR-specific analysis, a second model family, and causal retraining remain conditional extensions. They become necessary if the paper makes the corresponding training-stage, generalization, or causal claim. They are not silently treated as completed or added as unconditional requirements.

**Not achieved.** The current turn made verified progress by closing implementation checks and freezing development preparation. Human work, real calibration, the full response grid, empirical sensitivity results, and final external readability checks still prevent completion of the scientific objective. No claim of resubmission readiness is made.

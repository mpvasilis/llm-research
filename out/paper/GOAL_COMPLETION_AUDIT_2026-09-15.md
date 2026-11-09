# Completion audit for the continuing revision

Objective: complete the remaining revision work, reevaluate the evidence, and make the paper clearer and more understandable. **The full objective is not achieved.** The preceding work made concrete progress: the higher-budget integration checks closed, and the provisional successor and exact calibration design were frozen. This continuation also completed isolated Windows installation, notebook/tokenizer verification, resource sizing, and the bounded historical-cache recovery search. These preparations do not complete the experiments or human review.

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
| Reproducible preparation | Preserved v2 artifact verified; successor source/notebook and completed-probe packages verified locally; final integration evidence retained; isolated Windows installation, imports, package consistency, six tokenizer/configuration checks, and unchanged 13-cell notebook run passed | GPU generation and final empirical supplement pending |
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

## Clean-environment verification

The notebook's seven declared dependency ranges installed in a fresh Python 3.12.14 Windows virtual environment with system site-packages disabled, resolving Transformers 4.57.6 and PyTorch 2.14.0+cpu. Test-only notebook tools were installed separately. The package check found no broken requirements and all imports passed.

All six exact cached AutoConfig/AutoTokenizer snapshots loaded locally without models or network requests. Official template/rendered-probe/serialization identities matched the frozen protocol; all 200 per-prompt token counts agreed across checkpoints, and split extrema/prompt IDs matched the prior context audit. The unchanged notebook then executed all 13 code cells from relocated bundles with the exact isolated kernel, zero error outputs, all 21 actions off, and candidate caps unset. All original cell types and sources were retained.

Root independently inspected the actual retained notebook, sequential execution counts, AST controls, kernel, version ranges, artifact hashes, six pins, all 12 scientific source hashes, and all 37 preparation payloads. Nineteen checks passed. The current PDF, protocol, calibration manifest, and blank human artifacts remained unchanged; the official calibration directory still contains only its manifest. Evidence: `output/revision/verification/clean_environment_independent_verification_20260915.json`, with linked execution/import/tokenizer records under `output/revision/environment_validation/`.

This closes clean Windows preparation only. It does not establish Linux/GPU compatibility, generation success, human validity, or scientific completion. No model weights were downloaded or loaded by these checks.

## External resources still required

A read-only audit found complete local model weights only for pinned 1B Instruct; the other five have configuration/tokenizer files but lack weights. No CUDA device or connected GPU session was found. A direct Colab check showed a signed-out session. No original missing stage cache was located. These are checks of the accessible environment, not claims about all possible storage or compute.

The recovery search now also covers nine reachable Git commits, local reflogs, and every currently unreachable blob (48 blobs, 1,830,331 bytes). No additional stagewise raw-response archive was found. Read-only remote queries returned only the main commit already present locally. No fetch, checkout, reset, or lost-found write occurred. Pruned objects and inaccessible external storage remain outside scope. Evidence: `historical_git_cache_recovery_20260915.json`, `historical_git_unreachable_cache_recovery_20260915.json`, and `historical_remote_refs_observation_20260915.json`, all under `output/revision/verification/`.

The official exact-revision weight/index manifest records 49,731,860,288 missing bytes (46.32 GiB), with 52,701,714,512 bytes (49.08 GiB) for all six checkpoints. These are transfer/storage figures, not a runtime-fit measurement. Official 7B Instruct parameter metadata implies 27.19 GiB for float32 parameters alone, exceeding the approximately 14.77 GiB free-RAM observation before additional runtime memory. D: carries an explicit temporary-disk data-loss notice and is not durable study storage. No model weights were downloaded. The manifest and independent pin/arithmetic verification are `checkpoint_download_resource_manifest_20260915.json` and `checkpoint_download_resource_independent_20260915.json` in the same verification directory.

The configured 4,096-token context limit accommodates the inspected 32–64-token inputs with both provisional budgets. This proves a configuration margin only, not runtime memory fit, generation success, or budget adequacy. The completed CPU probe cannot establish 7B or GPU throughput. Official calibration should use one suitable six-checkpoint environment rather than combine incompatible partial execution profiles.

Human annotator availability and an original-cache/GPU location were already requested. They remain unresolved. No external messages, uploads, remote runs, submissions, or publication occurred in this phase.

## Scope and completion decision

Base/SFT serialization crossover, RLVR-specific analysis, a second model family, and causal retraining remain conditional extensions. They become necessary if the paper makes the corresponding training-stage, generalization, or causal claim. They are not silently treated as completed or added as unconditional requirements.

**Blocked, not achieved.** The preceding goal turn made verified progress by closing implementation checks and freezing development preparation. This continuation made further progress by completing clean-environment verification, resource sizing, and bounded cache recovery. These locally actionable tasks are now closed.

The same unresolved compute/human dependencies have persisted across three consecutive goal turns: `01a0a5ff-405d-7e23-bd8b-fc43a831e9d6`, `01a0a646-b954-74f3-ac6a-4ddc72b57e47`, and `01a0a661-210c-7bd0-ba07-0bb01fd0357a`. Current observations are retained in `output/revision/verification/external_readiness_observation_20260915.json`; no answering resource location or completed human records arrived during the checks. Earlier turns continued useful local preparation; the present impasse follows its completion.

The work can resume when suitable six-checkpoint compute and genuine human participants are available. Required work remains: prompt review and independent reading, real 96-response calibration and guide/timing review, prospective-plan freezing, 2,880 primary/192 sensitivity responses, blinded annotation/adjudication, empirical analyses, and manuscript integration. No claim of resubmission readiness is made.

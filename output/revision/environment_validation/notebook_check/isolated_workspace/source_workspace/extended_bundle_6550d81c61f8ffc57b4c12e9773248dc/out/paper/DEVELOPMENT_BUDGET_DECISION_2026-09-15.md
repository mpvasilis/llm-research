# Development-budget decision record

Date: 15 September 2026. Status: **completed probe; provisional calibration-cap decision pending**.

The completed 256/512 checks showed cap stops in eight and six responses respectively. The fixed 1,024-token probe is now complete and strictly verified: all eight responses reached EOS at 454–744 tokens. A true resume generated zero new responses and preserved every saved run-file hash. This supports considering 1,024 as a provisional cap for broader six-checkpoint calibration; it does not establish study-wide budget adequacy, semantic completeness, factual correctness, or advice quality. No replacement primary/sensitivity caps or successor empirical protocol have been selected/frozen.

## Verified evidence from the completed probe

All eight expected responses passed strict verification and complete-only export. Matched prompt, condition, checkpoint, seed/RNG, execution profile, source and tokenizer identities, context preflight, stopping records, and input-token counts agree with the preceding runs. Every matched 512 decoded string is an exact prefix, and two are identical. All eight 256 decoded strings are also exact prefixes. Raw continuation token IDs were not retained, so agreement claims remain about decoded text.

Evidence: `output/revision/development_budget_probe/development_1024_verification_20260915.json` and `output/revision/development_budget_probe/exports/development_1024_eight_responses.jsonl`. The complete attempt finished at 17:49:52 UTC and took 7,888.19 seconds. Timing describes this CPU run and cannot establish 7B or GPU throughput.

## How the result informs the next step

| Complete eight-response result | Justified next step | Claim still unsupported |
|---|---|---|
| Every response emits EOS within the budget | Consider 1,024 as a provisional cap for broader six-checkpoint development calibration | Adequacy for the evaluation corpus, other checkpoints, or semantically complete answers |
| At least one response reaches the cap or has an unexplained stop | Continue development investigation; preserve and report the censoring. A longer matched probe requires a separate versioned design | Calling 1,024 sufficient or identifying an adequate replacement from these eight responses |

The observed complete probe satisfies the first row. The next decision remains provisional and development-only.

Before broader calibration, record the intended stopping/truncation decision criterion and how a failure will be handled. A conservative operational option is to require no cap or unexplained stops in all 96 calibration responses and no budget truncation identified during the already planned human development review. This is a proposed development criterion, not an established statistical guarantee or an additional completed human assessment. It must not be chosen after inspecting held-out outcomes.

The complete 96-response calibration and genuine human guide refinement/timing remain prerequisites for the prospective sampling plan. Use only a complete calibration archive at the eventual primary cap. Do not mix caps or relabel earlier responses as if they had been generated under a different maximum length. Continue to report actual response lengths and stopping behavior after a development criterion passes.

## Required version transition

The current v2 parent fixes a proposed 256-token primary grid and a 512-token longer sensitivity. Its calibration companion supports only 256/512 and requires its cap to match the parent primary cap. It cannot freeze or execute a 1,024-token calibration.

A versioned successor to the parent/calibration workflow has been implemented while preserving v2 and the completed probe. Its post-Windows-fix core/calibration suite passed 13 tests with unchanged source hashes; strict prospective/sensitivity integration and portable-notebook checks remain in progress. Complete these checks before using the successor for empirical freezing. The successor may record a provisional primary cap for development; evaluation remains disabled until the development decision, actual human work, and sampling-plan prerequisites are satisfied. A budget sensitivity described as longer must exceed the selected primary cap. Rebind the prompt-review and validation artifacts to the actual successor rather than reusing a certification for a different protocol.

The full objective remains unchanged: independent human validation, the complete retained-checkpoint evaluation, checkpoint-wide detector/window analyses, an externally checked readable manuscript, and a resubmission readiness decision.

Evidence: `DEVELOPMENT_BUDGET_REPORT_2026-09-15.md`, `DEVELOPMENT_BUDGET_PROBE_PROTOCOL_2026-09-15.md`, `CALIBRATION_PROTOCOL_2026-09-15.md`, `PROSPECTIVE_VALIDATION_PROTOCOL_2026-09-15.md`, and the current progress report. This record reports the completed exploratory probe and introduces no human results, confirmatory model comparisons, or selected replacement budget.

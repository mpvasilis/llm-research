# Development generation-budget report

Status on 15 September 2026: **completed exploratory development check; no human validation or confirmatory model comparison**.

## Scope

This report compares matched 256-token and 512-token generation budgets and the subsequently completed, separately versioned 1,024-token probe for two development families: `dev_health_hiking_blister` and `dev_everyday_closet`. Each family contributes the four emotional-framing × request-type variants. All eight prompts use the pinned 1B Instruct checkpoint, logical seed 0, CPU float32 execution, and four Torch threads under frozen protocol `protocol_32f4c8538587e45cdf07e88ed9b1f77d`.

This is a budget and execution diagnostic from two scenario families. It contains no human labels, estimates no framing or request effects, compares no model stages, and provides no basis for extrapolating 7B or GPU throughput.

## Matched results

| Family | Domain | Framing | Request | 256 tokens | 256 stop | 256 generation s | 512 tokens | 512 stop | 512 generation s | Decoded 256 text is exact prefix |
|---|---|---|---|---:|---|---:|---:|---|---:|---|
| hiking blister | health | emotional | advice | 256 | cap | 104.88 | 512 | cap | 189.53 | yes |
| closet organization | everyday | neutral | advice | 256 | cap | 114.13 | 512 | cap | 388.71 | yes |
| closet organization | everyday | emotional | information | 256 | cap | 110.81 | 512 | cap | 968.67 | yes |
| closet organization | everyday | neutral | information | 256 | cap | 108.42 | 512 | cap | 641.80 | yes |
| hiking blister | health | emotional | information | 256 | cap | 94.61 | 512 | cap | 703.49 | yes |
| closet organization | everyday | emotional | advice | 256 | cap | 90.65 | 461 | EOS | 769.47 | yes |
| hiking blister | health | neutral | information | 256 | cap | 83.51 | 512 | cap | 857.14 | yes |
| hiking blister | health | neutral | advice | 256 | cap | 88.28 | 454 | EOS | 695.73 | yes |

All eight 256-token outputs stopped at the cap. Six of eight 512-budget outputs stopped at the cap; two reached the recorded EOS token after 461 and 454 output tokens. The observed 512 cap rate is therefore 0.75. Independent inspection found the eight 256-token decoded outputs visibly ended during substantive content and found no obvious repetition loops.

The 256-token decoded response string is an exact character-for-character prefix of its matched 512-budget decoded response for all eight prompts. This supports consistent matched generation at the text level. The artifacts do not retain raw continuation token IDs, so this check establishes decoded-text prefix agreement rather than original token-ID identity.

## Completed 1,024-token probe

The fixed eight-response probe completed at 17:49:52 UTC on 15 September. Both strict complete-run validation paths passed. All eight responses stopped at EOS, with lengths from 454 to 744 output tokens and 4,635 tokens in total. A true resume generated zero new responses and left every saved run file unchanged.

| Domain | Family | Framing | Request | Output tokens | Stop | Generation seconds | Matched 512 decoded text |
|---|---|---|---|---:|---|---:|---|
| health | hiking blister | emotional | advice | 524 | EOS | 748.29 | exact prefix |
| everyday | closet organization | neutral | advice | 635 | EOS | 727.85 | exact prefix |
| everyday | closet organization | emotional | information | 569 | EOS | 870.60 | exact prefix |
| everyday | closet organization | neutral | information | 677 | EOS | 1358.94 | exact prefix |
| health | hiking blister | emotional | information | 744 | EOS | 1336.34 | exact prefix |
| everyday | closet organization | emotional | advice | 461 | EOS | 1028.78 | identical |
| health | hiking blister | neutral | information | 571 | EOS | 818.55 | exact prefix |
| health | hiking blister | neutral | advice | 454 | EOS | 935.25 | identical |

All eight matched 512 decoded strings are exact prefixes of the corresponding probe strings; the two earlier EOS-ended strings are identical. All eight 256 decoded strings also remain exact prefixes. Prompt, condition, checkpoint/tokenizer revisions, serialization, seed/RNG, execution-profile identity, and input-token counts match across the three budgets. These checks establish decoded-text agreement, not continuation token-ID identity.

The complete probe took 7,888.19 seconds, including 62.83 seconds of model loading. Generation calls totaled 7,824.60 seconds, with a mean of 978.07 seconds and range 727.85–1,358.94 seconds. These timings describe this CPU run and do not establish other-model or GPU throughput.

## Timing and configuration checks

The 256 run took 821.13 seconds from attempt start to finish. Model loading took 25.50 seconds, and the eight generation calls totaled 795.28 seconds, with a mean of 99.41 seconds and a range of 83.51–114.13 seconds.

The 512 run took 5,233.78 seconds from attempt start to finish. Model loading took 18.78 seconds, and the eight generation calls totaled 5,214.54 seconds, with a mean of 651.82 seconds and a range of 189.53–968.67 seconds. These figures cover this local CPU process only. The large within-run variation means even this host-specific timing should be treated descriptively; it must not be projected to 7B checkpoints or a GPU runtime.

The model, tokenizer, and official chat-template identities matched the frozen v2 protocol. The effective EOS ID was 100257 and pad ID was 100277. Across all eight prompts, serialized inputs were 36–58 tokens under both budgets, far below the 4,096-token context limit. This range is taken from every saved response and agrees with the independent 1,024-token probe preflight; this corrects the previously reported 40–44 range. These checks found no configuration explanation for the cap stops. The saved text supports continued substantive generation; the cause of the large timing variation was not established.

The 512 run completed with eight unique expected response IDs and no missing responses. Strict verification reported two EOS stops and six cap stops. A true resume found all eight cached responses valid, generated zero new responses, and left every response-file SHA-256 unchanged.

## Budget recommendation

The completed 256/512 evidence shows substantial censoring: all eight 256-token generations and six of eight 512-token generations reached their caps. Those runs remain useful as fixed-budget development diagnostics. They do not establish behavior of naturally completed answers.

All eight responses in the fixed 1,024-token probe reached EOS. This supports considering 1,024 as a provisional cap for the broader 96-response, six-checkpoint development calibration. It does not select the final evaluation cap or establish adequacy for other checkpoints, the 40-family evaluation corpus, semantic completeness, factual accuracy, or advice quality. The sample covers only two development families and one checkpoint.

Before broader calibration, record an explicit provisional primary cap, a strictly larger sensitivity cap, the development stopping/truncation criterion, and the response to a failed criterion. Use the successor workflow for caps above 512 after its complete verification; preserve the original protocol and completed runs. Evaluation remains dependent on genuine prompt review, complete six-checkpoint calibration, human guide refinement/timing, and the prospective sampling plan. No successor empirical protocol or calibration manifest has been frozen, and no replacement budget has been selected.

## Artifacts

- 256 verification: `output/revision/full_study/development_256_verification_20260915.json`
- 256 responses: `output/revision/full_study/exports/development_256_eight_responses.jsonl`
- 512 verification: `output/revision/full_study/development_512_verification_20260915.json`
- 512 responses: `output/revision/full_study/exports/development_512_eight_responses.jsonl`
- 1,024 verification: `output/revision/development_budget_probe/development_1024_verification_20260915.json`
- 1,024 responses: `output/revision/development_budget_probe/exports/development_1024_eight_responses.jsonl`

The development evidence changes only the interpretation of the proposed budgets. Human manipulation review remains pending, and no confirmatory response generation has been performed.

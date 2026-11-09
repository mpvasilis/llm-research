# Extended longer-budget sensitivity protocol — 15 September 2026

## Status and scope

This companion completes the pre-specified longer-budget analysis path for the evidence-gated successor study. It is implementation and synthetic-test material only. No successor protocol, budget, response archive, prediction archive, analysis result, human record, or empirical conclusion is created or frozen here.

The analysis is limited to the eight evaluation scenario families already selected by the successor protocol: four health and four everyday families. Each family contributes its four emotional-framing × request-type variants at all six checkpoints. The longer-budget grid uses logical seed 0, yielding exactly 192 responses. Each longer response is paired only with the matching primary seed-0 response by checkpoint alias, prompt ID, and logical seed. Primary seeds 1 and 2 never enter a matched cap comparison.

## Prediction contract

`predict_complete_sensitivity(protocol_path, run_dir, output)` first invokes the successor protocol loader, the actual successor run validator with `require_complete=True`, and the strict run-config validator. The run must have mode `sensitivity`, the exact ordered frozen 192-response ID grid, completed human prompt review, and the complete nested prospective-readiness snapshots. A missing, altered, additional, duplicated, or misnamed response is rejected.

The predictor applies the same pinned detector and prefix-window tokenizer as the primary producer. For every response it emits `full_response`, `prefix_64_tokens`, and `prefix_128_tokens`, totaling exactly 576 unique response/window rows. Every row reconstructs the producer metadata, detector flags, window metadata, dependency identity, actual run ID, and prediction ID formula. Union and intersection flags must agree with exact and embedding flags. The output is created atomically only after all 576 rows pass the strict prediction validator, and an existing destination is never overwritten.

The CLI command is:

```text
python -X utf8 -m experiments.extended_sensitivity_analysis predict --protocol PATH --run-dir PATH --output PATH [--execute]
```

Without `--execute`, it validates the complete sensitivity run and reports the expected output without loading detector models or writing a file. `--execute` performs prediction only; it does not generate responses.

## Matched analysis contract

`analyze(...)` strictly validates both actual complete successor runs and checks that each supplied response export equals its run's saved rows exactly. The primary path is validated first by the successor loader/run verifier and then by the unchanged `revision_analysis.load_and_validate` 2,880-response/8,640-prediction contract. Both primary and sensitivity predictions must carry the actual corresponding run ID and the exact producer prediction IDs.

The two runs must use the same protocol and identical generation execution profiles. Their prediction archives must use the same pinned detector dependency identity. Each of the 192 frozen pairs must agree on family, condition, prompt, checkpoint, model/tokenizer/serialization pins, RNG identity, and all generation settings except `max_new_tokens`; the two caps must equal the frozen primary and sensitivity caps.

The result contains:

- matched exact, embedding, and union rates for all four detector categories, three windows, checkpoints, domains, emotional-framing values, and request types;
- paired output-length, cap-stop, EOS-stop, and exact decoded-string prefix summaries;
- within-family advice-minus-information, emotion-minus-neutral, and interaction contrasts at both caps;
- paired longer-minus-primary changes in those contrasts by checkpoint, domain, category, detector component, and window.

Factor contrasts are balanced within each family. Advice-minus-information averages over emotional framing; emotion-minus-neutral averages over request type; the interaction is the difference of those differences. This preserves the scenario family as the independent unit and avoids treating prompt variants as independent topics.

## Uncertainty

The default uncertainty calculation uses 10,000 bootstrap replicates and seed 20260915. It resamples the four families within each domain with replacement. A single family-index draw for a domain and replicate is shared across both caps, all checkpoints, detector components, categories, windows, and contrasts. Reported intervals are pointwise 2.5th and 97.5th percentiles. With zero replicates, allowed for focused synthetic tests, interval fields are `null`.

Only four families support each domain estimate. Longer-minus-primary values are paired point-estimate changes and must not be described as established sign reversals merely because a point estimate crosses zero.

## Outputs and limits

After every strict check and calculation succeeds, analysis atomically creates the requested new output directory containing `extended_sensitivity_results.json` and `EXTENDED_SENSITIVITY_REPORT.md`. An existing output directory is rejected. The JSON records protocol, run, input, source, and dependency identities; exact counts; bootstrap definitions; all result rows; and the interpretation limits.

This sensitivity cannot establish a causal training-stage effect, semantic completeness, factual accuracy, advice quality, human validity, or the adequacy of either generation budget. Decoded-text prefix agreement is exact string evidence only and is never called token-ID equivalence. No human labels apply to prefix windows.

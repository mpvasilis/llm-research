# What this paper is trying to show

The paper asks whether apparent changes in advice-style language survive changes in how that language is measured. It studies phrases associated with acknowledging emotions, validating reactions, mentioning professional help, and organizing an answer.

A detector searches for specified phrases or sentences similar to examples. Its output is a measurement made by that rule. To learn whether the rule identifies the intended language reliably, we also need independent people to judge responses using clearly defined targets. That human evaluation is still pending.

## A concrete result from the existing data

The recovered final-7B archive contains 700 responses to 140 prompts. Consider its comparison between advice requests and prompts about emotional topics that do not ask for advice. Apply the same combined detector, but change how much of each response it reads:

| Text examined | Advice responses with an empathy-category match | Emotional-topic controls with a match | Advice minus control |
|---|---:|---:|---:|
| Full response | 61.3% | 76.0% | −14.7 percentage points |
| First 64 tokens | 38.3% | 13.0% | +25.3 percentage points |
| First 128 tokens | 49.3% | 43.0% | +6.3 percentage points |

These are measurements of the same response archive. The point estimate changes sign when the amount of text examined changes. The full-response and first-128-token comparisons are uncertain; this is not evidence of two confidently established effects in opposite directions. The figures above come from the recovered-cache stress test, which is distinct from the older summary table.

Why can this happen? A longer answer offers more sentences in which the detector can find a match. Phrases can also occur later in an answer. For example, one archived factual response contains “Here are some key points” after its first 64 tokens: the full answer triggers the exact structuring rule, while the early portion does not. Looking only at an answer's beginning therefore changes the question being measured. It does not automatically produce a more accurate judgment of the full answer.

Source: `out/results/raw_output_robustness.json`, `component_and_fixed_window_results`. The concrete structuring example is response `hist4_340a6e2f28806f2c1e0e` in the normalized historical frame.

## Why checkpoint comparisons need care

The project also compares model checkpoints from different stages of training. If a later checkpoint writes longer answers, a detector that scans the whole answer may match more often even without a corresponding change in the particular human-judged property we care about. The next study is designed to test how stable those comparisons are across detection rules, output windows, and controlled prompt variants.

The original Base-to-SFT comparison also changes the format of the input. It therefore cannot isolate a training-stage effect. The new core study prioritizes SFT, DPO, and final Instruct checkpoints with verified comparable input formatting within each model size.

## What the preference-data analysis adds

DPO training uses pairs with a recorded preferred and non-preferred answer. The existing analysis asks which phrases occur more often on the preferred side. It finds aggregate differences that become much smaller in some comparisons between similarly long answers.

The completed follow-up models phrase presence and response length together, preserving the pairing and dependence between repeated prompts. It uses all 756,642 pairs across the two mixtures. Every original aggregate and length-band count was reproduced before fitting the new models.

| Phrase group | 1B adjusted odds ratio [95% interval] | 7B adjusted odds ratio [95% interval] |
|---|---:|---:|
| Broad referral/disclaimer phrases | 1.088 [1.026, 1.154] | 1.134 [1.067, 1.205] |
| Narrower referral phrases | 1.095 [0.972, 1.234] | 1.121 [0.996, 1.263] |
| Structuring phrases | 1.128 [1.108, 1.149] | 1.190 [1.168, 1.213] |

An odds ratio of 1 means no association in this model; a value above 1 means phrase presence is associated with the recorded preferred side after modeling both response lengths. It does not mean the same percentage-point increase in preference probability. The broad referral/disclaimer and structuring associations remain positive. Both narrower-referral intervals include 1, so those estimates remain uncertain.

This changes the interpretation of the earlier length-stratified results: length matters to the measured association, but the new model does not make all phrase associations disappear. Conditioning on length asks a different question from the original aggregate comparison. It does not establish that a phrase caused an answer to be preferred, or that particular training examples caused a checkpoint's behavior. The analysis specification was written before fitting these new models, after the earlier aggregate results were already known; the follow-up is exploratory. The mixtures also share 21,508 distinct content-identical pairs, so they are not independent replications.

Source: `output/revision/dpo_conditional/conditional_results.json`. A separate numerical calculation using an alternative spline basis reproduced all six primary marker estimates and their prompt-cluster standard errors; its record is `output/revision/verification/independent_dpo_numerical_check_20260915.json`.

## What remains necessary

The strongest completed finding is that measurement choices materially affect the observed pattern in the recovered final-7B responses. Extending that conclusion across checkpoints requires the new raw-response archive and the planned comparisons. Claims about human-judged language require the independent annotation work. The manuscript should make those boundaries easy to see, while explaining the completed results directly.

# Full-response annotation protocol (v4)

Version 4 is the revision-study protocol. It labels complete responses with the
user prompt visible. It does not reuse the older sentence audit, AI judgments,
detector outputs, checkpoint identities, or the other annotator's decisions.

Two different humans label every item independently. AI systems must not fill,
review, or certify either human pass. Adjudication begins only after both passes
are complete and preserved.

## Annotation unit and opening position

Judge the entire response in the context of its prompt. Mark every applicable
target as `1` and every absent target as `0`.

For `empathy_expression_location`, use:

- `none`: no empathy expression;
- `opening`: an empathy expression occurs only in the first sentence;
- `later`: one occurs only after the first sentence;
- `both`: expressions occur in both places.

The first sentence ends at the first `.`, `!`, or `?` sentence boundary. If the
response has none of those marks, the whole response is its first sentence.
This position field supports a separate descriptive check. The historical
`empathy_opener` detector searched the whole response, so its name does not by
itself support a claim about opening position.

## Human targets

`empathy_expression` marks language that explicitly acknowledges or
sympathizes with the user's emotional experience. Generic politeness, advice,
and factual descriptions of emotions do not count.

`validation` marks language that normalizes, legitimizes, or affirms an emotion
or reaction. An empathy expression does not automatically count as validation.

`professional_referral` marks a recommendation that the user contact, consult,
or seek help from a qualified professional. A factual mention of clinicians or
treatments does not count.

`professional_disclaimer` marks an explicit statement about the assistant's
lack of professional status, inability to diagnose, or limits on the response
as professional advice. A referral without such a limitation does not count.

`professional_mention` marks a professional, clinician, therapist, doctor,
treatment provider, or professional-help concept mentioned outside a span that
already qualifies as a referral or disclaimer. It captures otherwise factual
professional language; do not duplicate a referral or disclaimer span under
this target. A response may still receive this label if it contains a separate
factual professional mention elsewhere.

`structuring` marks explicit organization of the answer through a list,
enumeration, headings, or a preamble such as “Here are three steps.” Ordinary
paragraph breaks or transition words without an organizing function do not
count.

These targets are separate. Multiple `1` values are allowed. Put short quoted
or paraphrased evidence in `evidence_notes` when a decision could be hard to
adjudicate; notes are optional for clear negatives.

## Blinding and provenance

The prepared packet separates these files:

- `validation_items_v4.csv` contains only item ID, prompt, and full response;
- `validation_key_v4.csv` contains the hidden response provenance, sampling
  probabilities, detector version, original detector category predictions, and
  sample roles;
- `blank_human_annotations_1_v4.csv` and
  `blank_human_annotations_2_v4.csv` are independent blank templates;
- each completed `human_annotations_*_v4.csv` has a matching `.meta.json` human
  certification tied to the blinded-item hash;
- `validation_disagreements_v4.csv` is created only after two complete passes;
- `adjudicated_annotations_v4.csv` preserves final decisions and reasons.

Annotators must not open the hidden key, prediction files, model/checkpoint
metadata, or each other's files before both passes are complete. Use stable
pseudonymous annotator IDs. Enter a completion timestamp for every row.

The workflow rejects blank or invalid labels, repeated or missing item IDs,
same-ID annotators, altered item/key files, and incomplete adjudication.

## Preparation and scoring

For the prospective revision study, first complete the separate 96-response
development calibration. Humans must refine this guide using at least 60
distinct development responses and supply actual timing records. Then freeze
the sampling plan before generating or inspecting held-out evaluation results.
The [prospective sampling protocol](PROSPECTIVE_VALIDATION_PROTOCOL_2026-09-15.md)
documents the evidence format and the `preview-plan`, `freeze-plan`, and
`validate-plan` commands. The guide's hash is part of that frozen plan.

Once the full evaluation is complete, prepare its packet from the verified
2,880-response run and 8,640 detector-window predictions:

```powershell
python -m experiments.prospective_validation prepare `
  --protocol output/revision/full_study/protocol_v2.json `
  --plan path/to/frozen_sampling_plan.json `
  --calibration-manifest path/to/calibration_manifest.json `
  --calibration-responses path/to/calibration_responses.jsonl `
  --calibration-run-dir path/to/calibration_run `
  --timing-evidence path/to/human_calibration_timing.json `
  --responses path/to/responses.jsonl `
  --predictions path/to/predictions.jsonl `
  --evaluation-run-dir path/to/evaluation_run `
  --out-dir output/revision/annotation_v4/evaluation_packet
```

The strict companion retains exactly 624 probability selections, with response
length coverage determined from development, and up to 120 separately flagged
enriched responses. Overlapping selections appear once. This allocation remains
provisional until actual development evidence supports freezing it. The generic
`validation_v4 prepare` command does not enforce this full design and is not the
prospective preparation path.

Scenario families remain wholly in development or evaluation. The 12 prompts in
`experiments/prompts.json` are construction material and cannot enter a new
evaluation split.

Each human initializes their own pass directly. This action is the human's
certification and must not be run on their behalf by AI:

```powershell
python -m experiments.validation_v4 init --annotator 1 --annotator-id human-a --certify-human --out-dir output/revision/annotation_v4/evaluation_packet
python -m experiments.validation_v4 init --annotator 2 --annotator-id human-b --certify-human --out-dir output/revision/annotation_v4/evaluation_packet
```

After both people finish their CSV files, score the passes. If any item differs,
the human adjudicator creates and completes the adjudication template, then
scores again. Initialization is the adjudicator's own certification; an AI
system must not run it on the person's behalf. Prospective adjudication is
bound to both completed annotation files and records the adjudicator's human
certification:

```powershell
python -m experiments.validation_v4 score --out-dir output/revision/annotation_v4/evaluation_packet
python -m experiments.validation_v4 init-adjudication --adjudicator-id human-adjudicator --certify-human --out-dir output/revision/annotation_v4/evaluation_packet
python -m experiments.validation_v4 score --out-dir output/revision/annotation_v4/evaluation_packet
```

## Sampling and reporting rules

Rows carrying `probability_sample` have a recorded inclusion probability and
support design-weighted estimates for their declared sampling frame. The
reported uncertainty is an approximate superpopulation bootstrap that resamples
complete `scenario_family_id` groups; it is not an exact finite-population
design interval. Agreement is an unweighted description of the sampled panel.

Rows carrying `positive_enriched:<original_category>` are detector-positive
diagnostics. Their metrics and human rates stay in a separate report section
and are never pooled into population prevalence or recall estimates. An item
selected by both mechanisms appears once but retains both roles.

The report keeps the original detector names. It compares `validation` with the
same-named human target and `structure` with `structuring`. It reports the
response-wide `empathy_opener` comparison against `empathy_expression` with the
position mismatch stated. The original bundled `disclaimer` category is
compared with an explicitly derived `professional_any` union while the three
human professional targets remain separate. `crisis_referral` has no aligned v4
human target and is not scored.

Only `full_response` detector predictions can be compared with these labels.
First-64-token and first-128-token predictions remain detector sensitivity
analyses unless truncated texts receive their own independent human annotation.

## Historical materials

`out/raw_cache/final_7b_answers.jsonl` is a recovered final-7B archive that has
already been examined. Version 4 can normalize it as a historical diagnostic
sampling frame, preserving flags for the 12 construction prompts:

```powershell
python -m experiments.validation_v4 prepare-historical-frame `
  --source out/raw_cache/final_7b_answers.jsonl `
  --out-dir output/revision/annotation_v4
```

This archive is not a fresh held-out test set, cannot validate stagewise claims,
and must not be described as a prospective evaluation.

## Earlier materials

Version 3 and the 120-sentence audit remain historical artifacts. The older
workflow now rejects missing or malformed detector predictions rather than
silently converting them to `none`. It also separates probability-sample and
positive-enriched metrics instead of pooling them. The AI-consensus audit
remains a diagnostic and cannot be described as human validation.

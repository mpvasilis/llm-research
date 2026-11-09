# Prospective response-level human-validation sampling protocol

Date: 15 September 2026  
Status: **implementation protocol; no sampling plan or evaluation packet is frozen**

This companion closes the gap between the provisional human-validation design and
the strict full-study exports. It does not change the frozen prompt, generation,
calibration, prediction, or checkpoint-analysis sources. It must not be used to
create an empirical evaluation packet until development calibration, human guide
review, timing evidence, and the complete held-out archives exist.

## Order of operations

1. Complete and verify the 96-response, six-checkpoint development calibration at
   the same token cap selected for the primary evaluation.
2. Have humans use at least 60 distinct calibration responses to refine the
   annotation guide and record actual annotation timing. Duplicate annotation
   passes do not increase the distinct-response count.
3. Run `preview-plan`. This is read-only and reports the candidate length bins and
   fixed design without writing a plan.
4. Complete the development-evidence JSON described below. An AI system must not
   supply, infer, or certify its human fields.
5. Run `freeze-plan` and then `validate-plan`. The immutable plan is frozen before
   any held-out evaluation generation, prediction, or inspection.
6. Only after the plan is frozen, execute the complete held-out evaluation and
   detector predictions. Run `prepare` to produce the blinded packet.
7. Two different humans independently label every item. They see the prompt and
   full response, but not the checkpoint, condition key, detector output, sample
   role, or the other annotator's labels.
8. Resolve disagreements only after both passes are complete. Prospective packets
   require a human adjudicator ID and certification bound to both original
   annotation-file hashes.

This ordering is a workflow guard. The resulting analysis remains a detector
validation and response-level measurement study; the planning numbers are not a
power calculation and do not guarantee reliable estimates in every subgroup.

## Development-derived length bins

Length is `generation.output_tokens`, measured under each checkpoint's pinned
tokenizer. The plan uses all 96 verified development calibration responses at the
eventual primary evaluation cap. A 512-token calibration cannot freeze a plan for
a 256-token primary evaluation, or conversely.

The raw cutpoints are the pooled one-third and two-thirds nearest-rank quantiles:
sort the 96 lengths and take ranks `ceil((1/3)*96)` and `ceil((2/3)*96)`. Duplicate
cutpoints are collapsed. A cutpoint equal to the observed maximum is removed
because it would create an empty upper tail. A cutpoint equal to the observed
minimum is retained when lengths vary. If tied or capped lengths leave only one
effective bin, the plan records that collapse and does not substitute a boundary
chosen from evaluation responses.

These are global development-derived cutpoints. Evaluation responses are assigned
to them mechanically after generation. No held-out label, detector output, or
observed evaluation length distribution may change the cutpoints.

## Probability sample

The outer design has 48 cells:

- six retained checkpoints;
- two domains;
- two emotional-framing levels;
- two request types.

Each complete outer cell contains 60 responses. Exactly 13 are sampled from each,
giving 624 probability-sample responses. Within an outer cell, the 13 positions
are allocated over its nonempty frozen length bins as follows:

1. Assign one position to every nonempty bin.
2. Allocate the remaining positions in proportion to residual capacity `N_bin-1`
   using largest remainders.
3. Break exact remainder ties by ascending bin index.
4. Never allocate more than the bin population. Empty bins receive zero.

The inclusion probability for every response in an outer-cell length bin is
`n_bin/N_bin`. The hidden key records this probability for selected probability
items. A separate hashed sampling-design file records the random-component
probability for every selected panel item, whether or not that item entered through
the probability component. It stores reduced numerator/denominator pairs for the
random, category, enrichment-union, and panel-union probabilities, alongside
derived decimal values. Population estimates use only rows explicitly carrying
the `probability_sample` role and their inverse inclusion probabilities.

## Detector-positive diagnostic enrichment

The four eligible historical detector categories are `empathy_opener`,
`validation`, `disclaimer`, and `structure`. `crisis_referral` is unavailable in
the frozen four-category predictor and cannot create an enrichment stratum.

The design draws at most 30 full-response union-positive items independently per
category. Category overlaps are retained rather than replaced, so the union has at
most 120 unique enriched responses. Items also selected by the probability sample
appear once and carry both roles. The resulting annotation workload is therefore
624 to 744 unique responses before reductions from overlaps or small positive
pools.

For every selected panel item, the hidden sampling-design file records:

- every category for which the item was eligible and that category's `n/N`;
- the categories through which the item was actually selected;
- the enrichment-union inclusion probability;
- its probability-component inclusion probability, even for enriched-only items;
- the union probability of entering the panel through either mechanism.

The enrichment component has no population interpretation and is never pooled into
weighted prevalence or recall estimates.

## Input and identity checks

Packet preparation delegates to `experiments.revision_analysis.load_and_validate`.
It therefore requires exactly 2,880 evaluation responses and exactly 8,640
predictions: one full-response, 64-token, and 128-token prediction for every
response. It verifies the frozen protocol ID and grid, response identities,
checkpoint and execution provenance, prediction IDs and metadata, four detector
categories, pinned detector dependencies, and agreement of token counts across
windows. The companion additionally checks that the calibration development
families and evaluation families are disjoint. It also re-verifies the concrete
evaluation run directory, compares the response export canonically with that run's
2,880 response files, and requires every prediction `run_id` to equal the verified
evaluation `run_config.json` ID. A uniform but stale prediction run ID is rejected.

Only the full-response prediction enters the human-validation key. Prefix
predictions remain detector sensitivity analyses and are not validated by these
full-response human labels.

The packet writer uses independent deterministic SHA-256-derived substreams for
each probability stratum, enrichment category, and blinded display order. It writes
into a staging directory, verifies the v4 item/key bundle, and atomically publishes
the new packet directory. It refuses to overwrite an existing packet.

## Required human development evidence

`freeze-plan` requires a human-authored JSON object with this schema. Values below
are field descriptions, not values to copy or fill automatically.

```json
{
  "schema_version": "prospective-validation-development-evidence-v1",
  "status": "complete_human_guide_calibration_and_timing",
  "certify_human": true,
  "labels_created_by_ai": false,
  "reviewer_ids": ["nonblank stable human identifier"],
  "reviewed_at": "human-supplied timestamp",
  "reviewed_response_ids": ["at least 60 distinct IDs from the 96-row calibration"],
  "calibration_manifest_id": "exact calibration manifest ID",
  "calibration_responses_sha256": "exact export hash",
  "calibration_run_id": "run ID from the verified calibration run directory",
  "primary_max_new_tokens": "integer equal to the primary evaluation cap",
  "annotation_guide_sha256": "hash of the refined guide",
  "timing": {
    "response_ids": ["distinct timed IDs, all among reviewed_response_ids"],
    "per_item_seconds": {"response_id": "positive observed seconds"},
    "denominator_distinct_responses": "number of distinct timed IDs"
  }
}
```

Instead of `per_item_seconds`, `timing` may contain a positive documented
`total_seconds`, but it must still list the distinct timed response IDs and the
matching denominator. The code does not convert timing into an arbitrary available
annotator-hour claim.

## Commands

Read-only candidate preview:

```powershell
python -m experiments.prospective_validation preview-plan `
  --protocol output/revision/full_study/protocol_v2.json `
  --calibration-manifest PATH_TO_CALIBRATION_MANIFEST.json `
  --calibration-responses PATH_TO_96_CALIBRATION_RESPONSES.jsonl `
  --calibration-run-dir PATH_TO_VERIFIED_CALIBRATION_RUN_DIR `
  --annotation-guide out/paper/ANNOTATION_GUIDE.md
```

Freeze only after the human development evidence is complete:

```powershell
python -m experiments.prospective_validation freeze-plan `
  --protocol output/revision/full_study/protocol_v2.json `
  --calibration-manifest PATH_TO_CALIBRATION_MANIFEST.json `
  --calibration-responses PATH_TO_96_CALIBRATION_RESPONSES.jsonl `
  --calibration-run-dir PATH_TO_VERIFIED_CALIBRATION_RUN_DIR `
  --timing-evidence PATH_TO_HUMAN_DEVELOPMENT_EVIDENCE.json `
  --annotation-guide out/paper/ANNOTATION_GUIDE.md `
  --output output/revision/prospective_validation/sampling_plan_v1.json
```

Validate the plan before evaluation and again before packet preparation:

```powershell
python -m experiments.prospective_validation validate-plan `
  --plan output/revision/prospective_validation/sampling_plan_v1.json `
  --protocol output/revision/full_study/protocol_v2.json `
  --calibration-manifest PATH_TO_CALIBRATION_MANIFEST.json `
  --calibration-responses PATH_TO_96_CALIBRATION_RESPONSES.jsonl `
  --calibration-run-dir PATH_TO_VERIFIED_CALIBRATION_RUN_DIR `
  --timing-evidence PATH_TO_HUMAN_DEVELOPMENT_EVIDENCE.json `
  --annotation-guide out/paper/ANNOTATION_GUIDE.md
```

Prepare the held-out packet after complete evaluation and prediction exports exist:

```powershell
python -m experiments.prospective_validation prepare `
  --plan output/revision/prospective_validation/sampling_plan_v1.json `
  --protocol output/revision/full_study/protocol_v2.json `
  --responses PATH_TO_2880_EVALUATION_RESPONSES.jsonl `
  --predictions PATH_TO_8640_PREDICTIONS.jsonl `
  --calibration-manifest PATH_TO_CALIBRATION_MANIFEST.json `
  --calibration-responses PATH_TO_96_CALIBRATION_RESPONSES.jsonl `
  --calibration-run-dir PATH_TO_VERIFIED_CALIBRATION_RUN_DIR `
  --timing-evidence PATH_TO_HUMAN_DEVELOPMENT_EVIDENCE.json `
  --evaluation-run-dir PATH_TO_VERIFIED_EVALUATION_RUN_DIR `
  --annotation-guide out/paper/ANNOTATION_GUIDE.md `
  --out-dir output/revision/annotation_v4/evaluation_packet
```

Human annotation uses the existing v4 `init` commands. For a prospective packet,
adjudication initialization additionally requires:

```powershell
python -m experiments.validation_v4 init-adjudication `
  --out-dir output/revision/annotation_v4/evaluation_packet `
  --adjudicator-id HUMAN_ID `
  --certify-human
```

No current artifact satisfies the freeze requirements. This document specifies
the procedure; it does not claim that human guide calibration, evaluation
generation, annotation, or adjudication has occurred.

# Prospective checkpoint analysis protocol

Frozen on 15 September 2026, before any prospective response archive was
available and before any empirical checkpoint model was fit. The analysis is
confirmatory only for a complete, provenance-valid execution of the frozen
evaluation grid. Development rows, historical rows, the old five-condition
battery, partial runs, and longer-budget sensitivity runs cannot enter it.

## Inputs and gates

The primary grid is 40 evaluation scenario families (20 health and 20
everyday), four within-family prompt variants crossing emotional framing
(neutral/emotional) with request type (information/advice), six checkpoints
(SFT, DPO, and final Instruct at 1B and 7B), and three generation seeds. It
therefore contains exactly 2,880 unique responses. The scenario family is the
independent unit; the seeds are repeated stochastic measurements, so the
reported independent sample size is 40 rather than 2,880.

Analysis requires the immutable full-runner protocol, its complete evaluation
response export, and detector predictions for exactly the same response IDs.
The response archive must have one protocol ID, one full-runner version, one
execution-profile ID and body, evaluation provenance, and exact agreement with
every frozen grid identity. The analyzer recomputes the execution-profile ID
and reuses the full runner's validation of the prompt, serialization, decoding
configuration, and device/dtype against each frozen response identity.
Predictions must contain exactly one row per
response for each of `full_response`, `prefix_64_tokens`, and
`prefix_128_tokens`. Every one of the four historical marker categories must
have Boolean `exact`, `embedding`, and `union` flags, with union equal to their
logical OR. Detector versions, prediction variants, and dependency identities
must be constant, as must prediction run IDs and execution-profile IDs. The
applied embedding model and prefix tokenizer must each have an immutable
40--64-character hexadecimal content revision. Window metadata records the
pinned tokenizer identity, full-response token count, evaluated count, limit,
shorter-than-window flag, and truncation flag; the analyzer recomputes their
logical relationships and requires agreement across all three windows. Any missing crossing, duplicate ID,
response/prediction ID mismatch, unpinned dependency, wrong protocol, or
non-evaluation row is fatal.

The command-line writer additionally requires an explicit assertion that the
archive is the real completed run. Library functions may analyze synthetic
fixtures for tests, but they do not write empirical artifacts. No prospective
statistics are reported in this repository until the real archive passes all
gates.

## Estimands

For every family, checkpoint, prompt variant, detector component, marker
category, and output window, first average the three binary seed results. For
each family and checkpoint calculate:

* advice minus information, averaging over emotional framing;
* emotional minus neutral, averaging over request type;
* the difference in differences, `(emotional advice - emotional information)
  - (neutral advice - neutral information)`.

Within each model size, subtract SFT from DPO and DPO from final Instruct for
each contrast. Estimates are the balanced mean over the 40 chosen families and
are reported in percentage points. These are descriptive checkpoint
differences, not randomized causal effects of a training stage.

The primary inferential family is fixed to the full-response union detector:
two model sizes × four marker categories × three factor contrasts × two
adjacent stage changes = 48 metrics. Exact and embedding components and the
64- and 128-token windows are sensitivity analyses. They receive the same
contrast estimates plus a flag for direction changes relative to the matching
primary metric. Sensitivities are descriptive; no detector or window is chosen
by its p-value.

## Uncertainty

Use 10,000 bootstrap replicates with a fixed recorded random seed. In each
replicate, sample 20 health families with replacement and 20 everyday families
with replacement. The same sampled family indices are used for all 48 metrics.
Report un-clipped 2.5th and 97.5th bootstrap percentiles as pointwise
intervals.

For simultaneous bands, compute each metric's bootstrap standard deviation,
standardize every replicate's absolute deviation from the observed estimate,
and use the 95th percentile of the replicate-wise maximum as the common
critical value. A metric is treated as structurally degenerate when it has no
within-domain family variation above a scale-aware roundoff tolerance of 64
machine epsilons, even if floating summation produces a tiny positive bootstrap
standard deviation. A metric with structural, zero, or non-finite bootstrap
variation is marked `degenerate_uninformative` and excluded from this calibration. The result must
be labeled as a simultaneous max-t-style band over the estimable subset
`K/48`, never as joint coverage over all 48 when `K < 48`. If `K = 0`, no
simultaneous band is reported. Degenerate pointwise intervals may collapse to
the point estimate, but this is explicitly described as uninformative about
scenario uncertainty.

These intervals approximate variation under a model that treats the balanced,
chosen health/everyday scenario frame as exchangeable within domain. They do
not guarantee coverage for new topics, account for every prompt-authoring
choice, or turn stage comparisons into randomized causal effects.

## Descriptive outputs

The report also gives marker condition rates, generated-token lengths, observed
stopping-reason and cap rates, and the share of responses shorter than each
64/128-token prefix opportunity using the pinned detector window tokenizer.
Prefix estimates describe early-response
content and are not corrected full-response estimates. Human-validation input,
if later available, may be referenced separately; absent real human labels,
the detector results are not described as calibrated semantic judgments.

After a valid real run, the CLI writes machine-readable JSON, tidy CSV, a
readable Markdown report, and standalone checkpoint-contrast and response-
length plots under `output/revision/checkpoint_analysis/`.

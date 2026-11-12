# Pivot protocol: counterfactual DPO and a stored-text checkpoint atlas

Written 10 November 2026, before any data listed here was generated.

## Why pivot

The current paper is an observational audit. Three gaps cap it: it cannot say
whether preference data causes marker change, raw text exists only for final
7B, and Base-to-SFT is confounded with prompt format. This protocol closes all
three with generated evidence. It does not supply human labels; those remain a
separate requirement and are not simulated.

## Question

Does editing which responses DPO prefers change the phrases the resulting model
produces, and how much of the observed effect survives under the measurement
choices that the earlier audit showed to matter?

## Prompts

The 160 evaluation prompts of `experiments/prompts_revision_v1.json`: 40 topic
families crossed by emotional framing (no/yes) and request type
(information/advice). Authored by the study team and not externally reviewed.
Three sampling seeds (0, 1, 2) per prompt. Temperature 0.7, top-p 0.9, cap 1024
new tokens, with stop reason and token count stored.

## Experiment A: checkpoint atlas

Models: 1B Base, SFT, DPO, RLVR1, Instruct; 7B Base, SFT, DPO, Instruct, each in
its native format. Crossover: Base in chat format and SFT in plain
Question/Answer format at both sizes. Base-model text is cut at the first
generated "\nQuestion:" before scoring.

## Experiment B: counterfactual DPO (1B)

Start: `allenai/OLMo-2-0425-1B-SFT`. Data:
`allenai/olmo-2-0425-1b-preference-mix` at the revision pinned in
`experiments/dpo_conditional_analysis.py`.

Target markers T: the exact-match structuring or referral/disclaimer lexicons,
a pair being *chosen-only* if T appears in the chosen answer and not the
rejected one. Non-target markers (empathy, validation) are negative-control
outcomes.

Arms, each with three training seeds (0, 1, 2), each seed drawing its own
sample of 60,000 pairs:

| Arm | Data |
|---|---|
| baseline | the sample S |
| random_drop | S minus k random pairs, where k = chosen-only pairs in S |
| drop_chosen_only | S minus the chosen-only pairs |
| flip_chosen_only | S with chosen and rejected swapped on chosen-only pairs |
| length_matched | 60,000 pairs with word-count difference within 20 |

Training follows the published OLMo-2 7B DPO script where it applies: dpo_norm
loss, beta 5, learning rate 1e-6, linear schedule, 10% warmup, no weight decay,
one epoch. The 1B recipe is unpublished. Deviations: effective batch 32 instead
of 128, chosen so that the step count (1,875) is of the same order as the
original; sequences truncated at 1024 tokens; 60,000 of 378,301 pairs.

## Primary outcomes, fixed now

Unit: prompt-level mean over sampling seeds, then arm mean over training seeds.
Primary scoring: exact matching on the first 256 tokens, which removes length
opportunity. Primary contrasts (flip minus baseline; drop minus random_drop) on:
P1 structuring rate, P2 referral/disclaimer rate, on advice prompts.
Holm correction across the four tests. Inference: paired by prompt, bootstrap
over the 40 families (10,000 draws), plus training-seed spread reported beside
it because three seeds are too few for a seed-level test.

Secondary: the same on all text and at 64 tokens; union detector at tau 0.82;
response length; non-target markers (expected near zero); length_matched minus
baseline; advice-minus-information gap; retraining check comparing baseline with
SFT and the released DPO checkpoint.

Atlas outcomes: per-marker multiverse over rule (exact, union at 0.80, 0.82,
0.84) and window (64, 256, all), reporting the sign-flip rate of each stage
contrast. Format effect: Base chat minus Base QA and SFT chat minus SFT QA.

## Interpretation fixed in advance

If flip and drop move target markers relative to their controls and non-target
markers do not, preference pairs have a causal, marker-specific effect at this
learning setting. If they do not move, the paper reports that null. The retrained
models are not the released DPO model, and we will say so.

## Amendment, 11 November 2026 (before any data were generated)

- Optimizer: AdamW with 8-bit states (bitsandbytes) over fp32 weights. Full
  fp32 AdamW for the 1.5B-parameter SFT model needs about 24 GB before
  activations and does not fit on one 24 GB GPU beside the bf16 reference.
- Cost guard: the GPU job first times real DPO steps and generation
  (`pivot_runner.py bench`) and stops if the projected total exceeds the
  budget. If the 60,000-pair sample does not fit the budget, the sample size
  is lowered for all arms and seeds alike, and the paper reports the value
  used. Arms, outcomes, contrasts and the analysis are unchanged.
- Reporting: every pivot number in the paper is written by
  `experiments/pivot/report_pivot.py` from `output/pivot/analysis.json`.

## Amendment, 12 November 2026 (before any data were generated)

- Power: for each primary contrast the analysis reports the minimum
  detectable effect at 80% power and two-sided alpha .05, computed as 2.8
  times the family-bootstrap standard error (`mde80` in analysis.json). A
  null result is reported together with this value, so that a reader can
  tell an absent effect from an underpowered test.

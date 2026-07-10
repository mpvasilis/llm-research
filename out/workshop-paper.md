# Where Does Machine Advice Come From? (v2 Summary)

This markdown summary is synchronized with `out/paper/acl_latex.tex` and
`out/results/summary_v2.json`.

## Core Question

When an instruction-tuned model gives personal advice, it often uses a familiar
template: empathy, validation, professional-help disclaimers, and a structured
list of steps. This project asks where that behavior appears in OLMo-2's open
training pipeline.

## Design

- Models: OLMo-2 1B and 7B, including public Base/SFT/DPO/RLVR/Instruct stages
  where available.
- Prompts: 60 advice prompts and 30 factual controls, five sampled generations
  per prompt.
- Extra controls: 20 emotional non-advice, 20 neutral-advice, and 10
  domain-factual prompts. These expanded controls are available for 1B in the
  current artifact; 7B expanded-control cells are unavailable and reported as NA.
- Data searched: exact Stage-1 OLMo-Mix-1124 via infini-gram, exact per-model
  SFT mixtures via assistant-only DuckDB search, public DPO preference data, and
  RLVR data.

## Main Behavioral Result

The cleanest advice-vs-factual signal is the professional-help disclaimer.

| model | advice | factual | odds ratio | prompt-clustered p |
|---|---:|---:|---:|---:|
| 1B | 56/60 | 2/30 | 143.1 | <1e-4 |
| 7B | 55/60 | 2/30 | 115.0 | <1e-4 |

Empathy and validation also separate from factual controls. Structure does not:
it appears on 18/30 factual prompts at 1B and 13/30 at 7B, so list formatting is a
generic assistant behavior rather than an advice-specific one.

The 1B matched controls temper the claim. Emotional non-advice prompts trigger
high empathy (0.79), and neutral/domain prompts can trigger disclaimers (0.66 and
0.80). The defensible claim is therefore not uniqueness to advice, but a strong
separation from plain factual answering.

## Stagewise Result

The disclaimer rises after SFT and stays high after DPO/RLVR:

| model | base | SFT | DPO | RLVR | Instruct |
|---|---:|---:|---:|---:|---:|
| 1B | 0.473 | 0.753 | 0.917 | 0.903 | 0.927 |
| 7B | 0.543 | 0.767 | 0.923 | NA | 0.913 |

This is observational stage localization, not causal attribution.

## SFT Recoverability

Behavioral phrases are more common in exact SFT assistant turns than
length-matched topical phrases at 3-4 words:

| model | length | behavioral /M | topical /M | ratio |
|---|---:|---:|---:|---:|
| 1B | 3 | 12333.4 | 633.8 | 19.5x |
| 1B | 4 | 1676.2 | 11.8 | 141.7x |
| 7B | 3 | 9354.4 | 1280.4 | 7.3x |
| 7B | 4 | 1396.6 | 96.2 | 14.5x |

But the signal is phrase-sparse. Dropping the generic 3-word list marker "here
are some" lowers the distinct-phrase mean from 3358.1 to 758.8 per million at 1B,
and from 3026.0 to 1065.9 at 7B.

## Pretraining Novelty

Against exact verbatim OLMo-Mix-1124 coverage, advice answers are more novel than
factual answers:

- 1B: 0.559 advice vs 0.383 factual
- 7B: 0.525 advice vs 0.296 factual

This does not prove an SFT cause, but it argues against a simple verbatim
pretraining-copy story.

## Caveats

- `crisis_referral` is dropped because detector validation failed for that
  category.
- Detector precision is low, so emission rates are upward-biased.
- Stage-2 mid-training remains unsearched because there is no standalone
  infini-gram index.
- DPO/RLVR evidence is observational; causal tests require leave-cluster-out
  re-SFT or DPO edits.

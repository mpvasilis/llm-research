> **STALE (2026-07-10):** this post predates the v2 stagewise/matched-control run and the reframed paper ("Where Does Machine Advice Emerge?"). Rewrite from `acl_latex.tex` + `summary_v2.json` before publishing.

# Where does machine advice actually come from?

If you tell a chatbot "I've been anxious and can't sleep, any advice?", you tend
to get a familiar shape of reply: a sentence saying your feelings make sense, a
gentle validation, a nudge to talk to a professional, and a numbered list of
steps. It is helpful and consistent. But where does that bedside manner come
from?

We tested this on the fully open OLMo-2 family, using 60 advice questions and 30
plain factual questions, with five sampled generations per prompt. We traced
behaviors rather than exact sentences: empathy, validation, professional-help
disclaimers, and response structure.

## The cleanest signal

The strongest advice-vs-factual split is the professional-help disclaimer. In
the 1B model it fired on 56/60 advice prompts but only 2/30 factual controls. In
the 7B model it fired on 55/60 advice prompts and again 2/30 factual controls.
The odds ratios are large (about 115-143), and the prompt-clustered permutation
tests are below 1e-4.

Empathy and validation also separate from factual controls. But the expanded
controls make the story clearer: in the 1B run, emotional non-advice prompts
trigger empathy even more than advice prompts, and neutral/domain prompts can
also trigger disclaimers. So the right claim is not "these behaviors are uniquely
advice-specific." It is: the advice template is very different from plain factual
answering, and different pieces of the template track different prompt features.

One category remains dropped: crisis referral. The detector over-fired on that
category in validation, so we keep it only as a false-positive caution.

## The stagewise story

The new v2 run also checks public intermediate checkpoints. The behaviors do not
appear only at the final Instruct model. They rise after supervised fine-tuning
and stay high after preference/post-training.

For the disclaimer, the pattern is especially clear:

- 1B: base 0.473, SFT 0.753, DPO 0.917, RLVR 0.903, Instruct 0.927
- 7B: base 0.543, SFT 0.767, DPO 0.923, Instruct 0.913

That is not causal proof. It is a map of where the behavior appears in the
released training pipeline.

## Where the phrases live

In the exact SFT mixtures, advice-style phrases appear more often than
length-matched topical phrases, especially at 3-4 words. The v2 length-matched
ratios are:

- 1B: 3-word 19.5x, 4-word 141.7x
- 7B: 3-word 7.3x, 4-word 14.5x

But the main caveat is still phrase sparsity. A generic list opener, "here are
some", dominates the distinct 3-word mean. Dropping it lowers the 1B mean from
3358.1 to 758.8 per million, and the 7B mean from 3026.0 to 1065.9 per million.
That leaves an inspectable cluster, not a broad law of language.

The pretraining check points the same way: advice answers are more novel than
factual ones against exact verbatim OLMo-Mix-1124 coverage (1B: 0.559 vs 0.383;
7B: 0.525 vs 0.296). This does not prove the advice template came from SFT, but
it does mean the final advice answers are not just copied verbatim from
pretraining.

## What remains

This is observational provenance, not causal attribution. The next step is a
leave-cluster-out or DPO-edit experiment on the nominated disclaimer/list-marker
clusters. We also still need a Stage-2 index, a multi-annotator detector
validation, and the missing 7B expanded-control cells.

*Try the demo: [DEMO LINK PLACEHOLDER]. Read the full paper: [arXiv LINK PLACEHOLDER].*

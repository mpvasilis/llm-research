# arXiv submission metadata

## Title

Where Does Machine Advice Emerge? An Auditable, Observational Trace Across OLMo-2's Open Training Pipeline

## Authors / affiliation

<Your Name>, Independent Researcher / Innovation Bee

> Note: arXiv is NOT anonymous. The author and affiliation line above is public; do not submit this version to a double-blind venue (BlackboxNLP 2026) under your real name. That review version uses the anonymized LaTeX in `acl_latex.tex`.

## Primary category

cs.CL

## Secondary category

cs.LG

## Abstract (~250 words, plain text)

When asked for personal advice, instruction-tuned models tend to produce a recognizable bundle: empathy expression, emotional validation, professional referral, and structuring language. Is this one "advice template," and where in training does its targeting change? We trace these behaviors across OLMo-2's open post-training pipeline and exact Stage-1 pretraining corpus using 140 prompts in five matched conditions, five sampled generations each, and prompt-clustered permutation tests. Three findings. (1) At the primary detector threshold, the bundle decomposes: empathy expression is higher on emotional non-advice than advice, structuring language is identical on neutral advice, professional referral is common on clinical-information prompts, and validation alone is elevated against every matched control in both models. A threshold sweep preserves the structuring result but qualifies the empathy and referral contrasts. (2) Every advice-vs-factual gap widens at SFT; six of eight changes survive Benjamini-Hochberg correction across the full adjacent-stage grid. DPO widens validation gaps and collapses the structuring gap, while no later-stage gap change is detected. (3) Referral phrasings occur in thousands of exact-SFT assistant turns. Across approximately 378,000 DPO pairs per model, referral and structuring language remain enriched in chosen responses under pair-level McNemar tests and word normalization, despite chosen responses being slightly shorter. Searched RLVR counts provide near-zero direct lexical evidence of introduction. The evidence is observational, a preliminary detector audit indicates high recall and low precision, and SFT co-occurrence is phrase-sparse; completed two-human validation remains pending, and we release the tracer, prompts, code, and machine-readable results.

## Comments field (suggested)

Workshop submission targeting BlackboxNLP 2026 (EMNLP workshop). All numbers drawn from a single reproducible artifact (`summary_v3.json`). Code, prompt battery, and provenance-safe labeling/scoring tools released.

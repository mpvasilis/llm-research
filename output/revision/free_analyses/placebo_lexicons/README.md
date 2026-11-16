# Placebo lexicons and placebo seed sentences (RQ1)

    python -m experiments.free_analyses.placebo_lexicons

Run after (or independently of) `spec_curve`; both share the embedding cache in `../_cache/`. 200 frequency-matched placebo lexicons per marker (same phrase count, same word count per phrase, same log2 document-frequency bin in the pivot generations as reference corpus) and 200 placebo three-sentence seed sets for the embedding rule. Seed 20261114.

Outputs: `placebo_lexicons.json`, `placebo_all_text.tex`.

The RQ3 placebo (random lexicons on DPO pairs) was not run: the local cache `output/revision/dpo_conditional/pair_features_*.parquet` holds only the precomputed flags of the real phrase groups, not the response text.

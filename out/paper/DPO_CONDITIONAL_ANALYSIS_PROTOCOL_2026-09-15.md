# Exploratory paired DPO conditional analysis protocol

Frozen on 15 September 2026 before fitting the conditional models. This is an
**exploratory follow-up** to the already reported aggregate and post-hoc
within-20-word DPO analyses. It is not preregistered or confirmatory. The
analysis estimates a descriptive conditional choice association and does not
identify a causal preference for a marker or “remove” length bias.

## Data and exact parsing

The analysis uses the public training splits of the two exact OLMo-2 preference
mixtures already studied in the paper:

| Model | Hugging Face dataset | Pinned dataset revision |
|---|---|---|
| 1B | `allenai/olmo-2-0425-1b-preference-mix` | `c6454f3d364622718a8ecd3d307e524febbba569` |
| 7B | `allenai/olmo-2-1124-7b-preference-mix` | `316c96f092e87bc07b30d04116280bb80d70a0fb` |

The preparation command reads the raw Parquet shards at those immutable
revisions once and saves only compact pair-level features. For exact parity
with `experiments/dpo_pairwise.py` and `experiments/dpo_phrase_robustness.py`,
chosen and rejected response text is the join of every message whose role is
`assistant` using the two-character separator backslash plus `n` (the evaluated
legacy DuckDB SQL separator, not a newline); word count is the number of matches to `\S+`; marker
matching is case-insensitive substring matching implemented as the same
escaped regular-expression alternation, with longer alternatives first.

The retained marker groups are frozen to the three `GROUPS` in
`experiments/dpo_phrase_robustness.py`: the original broad
referral/disclaimer group, the high-precision referral group, and structuring
language. Each compact row contains the model, source, a stable pair hash, a
prompt-cluster hash, chosen and rejected word counts, and the three chosen and
rejected binary marker flags. No response or prompt text is retained.
The observational unit is a preference pair of concatenated assistant text,
not necessarily a single last answer. Pairs with multiple assistant messages
on either side remain included to preserve the legacy parsing, and their counts
are recorded separately.

Before any model is fit, preparation must reproduce 378,301 1B pairs and
378,341 7B pairs, every legacy chosen-only/rejected-only count for all three
groups both in aggregate and in each of the three legacy word-difference bands,
and the legacy band row counts. Failure is fatal. The manifest records dataset and shard revisions,
remote shard entity hashes when supplied by Hugging Face, local feature-file
SHA-256 hashes, all row counts, null/message-structure errors, and exact
chosen-versus-rejected prompt-context mismatches. Invalid pairs, if any, are
retained with error flags in the preparation audit and excluded from modeling;
their number and exclusion reason will be reported.

The prompt cluster is the SHA-256 hash of the same literal-backslash-`n`-joined
sequence of user-message contents used by the legacy duplicate-prompt analysis. A separate
audit compares the complete non-assistant role/content context on the chosen
and rejected sides so that system-context differences are not silently
ignored. Clustering joins identical prompt hashes even if they occur in
different sources.

A second content-only hash covers the chosen-side user prompt plus concatenated
chosen and rejected assistant text, excluding model labels, dataset identifiers,
and row locations. The manifest reports exact pair overlap between the 1B and
7B mixtures. Shared pairs limit any interpretation of the two mixtures as
independent replications; the models remain reported separately.

## Frozen primary models

Models are fit separately for 1B and 7B and separately for each of the three
marker groups (six fits). For pair *i*, let `M_c-M_r` be the chosen-minus-
rejected binary marker difference and let `B(L_c)-B(L_r)` be the difference in
a restricted cubic-spline basis of `log1p(word count)`. The conditional
log-likelihood is

`sum_i log sigmoid(beta_m (M_c-M_r) + beta_L' (B(L_c)-B(L_r)))`.

There is no intercept. The marker coefficient is reported as an adjusted
conditional odds ratio `exp(beta_m)` with a two-sided 95% Wald interval. This
is the within-pair association of the marker with the recorded chosen side at
the same modeled response lengths. Because length is an observed response
property related to the preference itself, conditioning on it changes
the estimand and can remove genuine length-related preference behavior.

The restricted cubic spline uses five knots from the pooled distribution of
`log1p` chosen and rejected word counts across both model datasets. The frozen
quantile probabilities are 0.05, 0.275, 0.50, 0.725, and 0.95, using linear
empirical quantile interpolation. The preparation step writes the five numeric
knots to its manifest and this protocol before `analyze` can run. The same
knots are then used for every fit. The basis is the linear term plus three
natural cubic terms, scaled by the square of the outer-knot range for numerical
stability.

**Numeric knot values frozen after preparation and before fitting:**
`2.079441542`, `4.369447852`, `5.323009979`, `5.926926026`, and
`6.606650186` on the `log1p(words)` scale (word-count equivalents 7, 78, 204,
374, and 739). All manifest parity checks passed before these values were
inserted.

Uncertainty uses the inverse observed information as bread and the outer
product of prompt-cluster score sums as meat. The covariance is multiplied by
`G/(G-1) * (n-1)/(n-k)`, where `G` is the number of prompt clusters, `n` the
number of modeled pairs, and `k` the number of fitted coefficients. Normal
95% intervals are reported; the correction does not guarantee small-cluster
validity. Unadjusted 95% intervals remain the primary descriptive presentation;
Benjamini-Hochberg values across the six marker coefficients are included as a
secondary multiplicity summary and are not used to relabel the analysis as
confirmatory.

## Frozen diagnostics and sensitivities

Every fit records optimizer status, gradient norm, information-matrix rank and
condition number, coefficient magnitude, boundary fitted probabilities, and
marker discordance counts. Rank failure, non-convergence, non-finite estimates,
or a mean-objective gradient infinity norm above `1e-6` makes a fit unavailable.
Large coefficients, ill conditioning, and boundary probabilities are diagnostic
warnings that require qualification; no single threshold is treated by itself
as proof of separation, and no unplanned penalty is applied.

The following are sensitivities, not alternative primaries:

1. marker-only fits, which must reproduce `log(chosen-only/rejected-only)` when
   both counts are nonzero;
2. source-specific versions of the full model where marker variation and rank
   permit fitting;
3. the full model after excluding the largest source in each model dataset;
4. one pair per prompt cluster, selected deterministically by the smallest
   stable pair hash.

Interpretation emphasizes effect direction, magnitude, interval width, and
heterogeneity rather than a significance/no-significance dichotomy. Source
fits with too little discordance or deficient rank are shown as unavailable.

## Validation and outputs

Before relying on empirical fits, tests cover: invariance/equivariance under
paired reversal and row reordering; the exact marker-only identity
`beta=log(b/c)`; a known-zero marker effect under synthetic length confounding,
where flexible length adjustment should recover approximately zero while the
unadjusted association does not; and exact aggregation parity for the frozen
phrase parsing. Results are written as machine-readable JSON and CSV, a compact
forest plot, and a prose report. The report must retain the exploratory status,
the changed estimand under length conditioning, prompt-cluster limitations,
and the fact that exact phrase matches do not measure semantic referral quality
or downstream training effects.

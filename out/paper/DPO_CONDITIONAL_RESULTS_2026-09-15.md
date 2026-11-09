# Exploratory paired DPO conditional analysis results

Completed 15 September 2026. These estimates are exploratory descriptive conditional choice associations. They are not preregistered or confirmatory, and conditioning on observed response length changes the estimand; it does not identify a causal marker preference or remove length bias.

## Data and parity

- **1B:** 378,301 prepared pairs; 378,301 modeled; all aggregate, length-band, and prompt-duplication parity checks passed. Nonzero structural/context error counts: `{}`. Multiple assistant messages occurred on the chosen side in 0 pairs and on the rejected side in 0 pairs; these remain included as concatenated conversation text. Nonzero metadata-null counts: `{}`.
- **7B:** 378,341 prepared pairs; 378,341 modeled; all aggregate, length-band, and prompt-duplication parity checks passed. Nonzero structural/context error counts: `{}`. Multiple assistant messages occurred on the chosen side in 0 pairs and on the rejected side in 0 pairs; these remain included as concatenated conversation text. Nonzero metadata-null counts: `{"source_null": 19890}`.
- **Cross-mixture overlap:** 21,508 distinct content-identical pairs (21,515 multiplicity-matched rows) occur in both mixtures. The 1B and 7B results therefore are not treated as independent replications.

The shared restricted-cubic-spline knots on `log1p(words)` are 2.079441542, 4.369447852, 5.323009979, 5.926926026, 6.606650186; their word-count equivalents are 7.00, 78.00, 204.00, 374.00, 739.00. The fixed percentile rule was 5%, 27.5%, 50%, 72.5%, and 95% over all chosen and rejected lengths.

## Primary results

The internal group ID `professional_referral_high_precision` is retained for artifact compatibility. Because semantic precision has not been established by human validation, the prose and table call it the **narrower referral-phrase group**.

| Model | Marker group | Adjusted conditional OR [95% CI] | Cluster p | BH q (six fits) | Pairs | Prompt clusters |
|---|---|---:|---:|---:|---:|---:|
| 1B | professional referral original | 1.088 [1.026, 1.154] | 0.00505 | 0.00758 | 378,301 | 367,082 |
| 1B | narrower referral phrases | 1.095 [0.972, 1.234] | 0.135 | 0.135 | 378,301 | 367,082 |
| 1B | structuring language | 1.128 [1.108, 1.149] | 1.61e-38 | 4.82e-38 | 378,301 | 367,082 |
| 7B | professional referral original | 1.134 [1.067, 1.205] | 4.74e-05 | 9.49e-05 | 378,341 | 367,110 |
| 7B | narrower referral phrases | 1.121 [0.996, 1.263] | 0.0589 | 0.0707 | 378,341 | 367,110 |
| 7B | structuring language | 1.190 [1.168, 1.213] | 1.71e-74 | 1.03e-73 | 378,341 | 367,110 |

The odds ratios compare exact marker presence on the chosen versus rejected side within a pair after modeling each side's length flexibly. They should not be read as effects of adding or removing a phrase from an otherwise fixed response.

## Sensitivity and diagnostics

### 1B

| Marker | Marker-only OR | Adjusted OR | One/prompt OR | Leave-largest-source-out OR |
|---|---:|---:|---:|---:|
| professional referral original | 1.181 [1.114, 1.252] | 1.088 [1.026, 1.154] | 1.079 [1.017, 1.145] | 1.098 [1.024, 1.177] |
| narrower referral phrases | 1.170 [1.039, 1.318] | 1.095 [0.972, 1.234] | 1.066 [0.945, 1.203] | 1.122 [0.966, 1.302] |
| structuring language | 1.242 [1.220, 1.264] | 1.128 [1.108, 1.149] | 1.133 [1.112, 1.154] | 1.085 [1.060, 1.110] |

Source-specific direction/range (descriptive adjusted ORs):

- professional referral original: 8/9 sources above 1; range 0.919–1.294.
- narrower referral phrases: 6/9 sources above 1; range 0.600–1.537.
- structuring language: 8/9 sources above 1; range 0.718–1.213.

Primary-fit warnings: professional_referral_original: optimizer_success=True, flags={'coefficient_abs_gt_20': True, 'information_condition_gt_1e12': False, 'boundary_probability_count_positive': False, 'minimum_information_eigenvalue_lt_1e_minus_8': False}, large-coefficient flag confined to spline nuisance terms=True; professional_referral_high_precision: optimizer_success=True, flags={'coefficient_abs_gt_20': True, 'information_condition_gt_1e12': False, 'boundary_probability_count_positive': False, 'minimum_information_eigenvalue_lt_1e_minus_8': False}, large-coefficient flag confined to spline nuisance terms=True; structuring_language: optimizer_success=True, flags={'coefficient_abs_gt_20': True, 'information_condition_gt_1e12': False, 'boundary_probability_count_positive': False, 'minimum_information_eigenvalue_lt_1e_minus_8': False}, large-coefficient flag confined to spline nuisance terms=True
The frozen absolute-coefficient flag is triggered only by spline nuisance coefficients in these fits; all marker coefficients have absolute value below 0.18. On its own this basis-scale-dependent flag is not evidence of separation.

### 7B

| Marker | Marker-only OR | Adjusted OR | One/prompt OR | Leave-largest-source-out OR |
|---|---:|---:|---:|---:|
| professional referral original | 1.352 [1.274, 1.434] | 1.134 [1.067, 1.205] | 1.130 [1.062, 1.201] | 1.152 [1.073, 1.237] |
| narrower referral phrases | 1.335 [1.187, 1.502] | 1.121 [0.996, 1.263] | 1.103 [0.978, 1.243] | 1.168 [1.008, 1.354] |
| structuring language | 1.435 [1.409, 1.461] | 1.190 [1.168, 1.213] | 1.191 [1.169, 1.214] | 1.112 [1.087, 1.139] |

Source-specific direction/range (descriptive adjusted ORs):

- professional referral original: 8/9 sources above 1; range 0.988–1.584.
- narrower referral phrases: 6/9 sources above 1; range 0.636–2.242.
- structuring language: 8/9 sources above 1; range 0.718–1.324.

Primary-fit warnings: professional_referral_original: optimizer_success=True, flags={'coefficient_abs_gt_20': True, 'information_condition_gt_1e12': False, 'boundary_probability_count_positive': False, 'minimum_information_eigenvalue_lt_1e_minus_8': False}, large-coefficient flag confined to spline nuisance terms=True; professional_referral_high_precision: optimizer_success=True, flags={'coefficient_abs_gt_20': True, 'information_condition_gt_1e12': False, 'boundary_probability_count_positive': False, 'minimum_information_eigenvalue_lt_1e_minus_8': False}, large-coefficient flag confined to spline nuisance terms=True; structuring_language: optimizer_success=True, flags={'coefficient_abs_gt_20': True, 'information_condition_gt_1e12': False, 'boundary_probability_count_positive': False, 'minimum_information_eigenvalue_lt_1e_minus_8': False}, large-coefficient flag confined to spline nuisance terms=True
The frozen absolute-coefficient flag is triggered only by spline nuisance coefficients in these fits; all marker coefficients have absolute value below 0.18. On its own this basis-scale-dependent flag is not evidence of separation.

Source-specific estimates and full convergence, rank, eigenvalue, gradient, boundary-probability, and finite-correction diagnostics are in `conditional_results.json`. The deterministic one-pair-per-prompt sensitivity uses the smallest stable pair hash. The leave-out sensitivity removes the largest source separately for each model.

## Scope

Exact substring flags preserve the historical detectors, including the broad referral/disclaimer fragments. They do not establish semantic referral quality. Prompt-cluster sandwich intervals account for identical prompt hashes, but normal intervals and exact-text clusters do not guarantee protection against every form of dependence. Source heterogeneity and sensitivity estimates are descriptive. None of these co-occurrences attributes checkpoint behavior to particular training examples.

# Readability and claim audit

Prepared 15 September 2026 as an internal cold-read and evidence check of the
revised manuscript. The audit treats the historical numerical artifacts as
fixed evidence. It does not add human judgments, regenerate historical model
responses, or convert observational results into causal claims.

## Central question and narrative

The paper now asks one question: **How do detection rules and response length
affect conclusions about advice-style language across OLMo-2 post-training?**
The narrative has four dependent steps:

1. Define the marker, detector, and proxy distinction with a concrete
   constructed example.
2. Diagnose the historical detector before interpreting any detector-positive
   condition or checkpoint rate.
3. Compare conditions and checkpoints under that explicitly scoped rule.
4. Compare aggregate and exploratory length-conditioned associations between
   exact DPO phrase matches and the recorded chosen side.

The scientific motivation is broader than a single poor detector. Free-form
occurrence rates depend jointly on the response, the amount of text searched,
and the matching rule. A checkpoint trajectory can therefore be partly a
measurement trajectory.

## Cold-read weaknesses and revisions

| Cold-read weakness | Revision made | Remaining constraint |
|---|---|---|
| The first page showed that detectors differ but did not clearly explain why the question matters. | Added the response-opportunity argument: longer answers and embedding matches can change occurrence rates independently of the intended construct. | Generality beyond OLMo-2 remains conceptual because only one model family is studied. |
| The motivating example did not show a plausible match and a misleading match. | Added two clearly labeled constructed fragments. `that sounds really` triggers the historical empathy-opener rule even when it appraises a checklist in a second sentence. | The fragments are illustrations, not validation data. |
| The paper moved among 140 prompts, 90 checkpoint prompts, 120 audit sentences, and roughly 378k DPO pairs without a simple map. | Added a main-text evidence-map table with material, comparison, and scope limit for each analysis. | The evidence sources still have different units and must not be pooled. |
| Condition abbreviations obscured the control logic. | Replaced labels such as “Emot.,” “Neut. adv.,” and “Dom. fact.” with personal advice, general facts, emotion without an advice request, neutral how-to, and health facts. | Controls are not topic-paired counterfactuals. |
| The related-work section devoted too much space to secondary corpus tracing. | Consolidated construct validity, length, open-data, and attribution boundaries; demoted corpus searches to secondary context. | The appendix retains corpus details for reproducibility. |
| A full stage-rate table interrupted the main claim. | Kept transition changes in the main text and moved the underlying raw rates to the appendix. | Historical stage summaries cannot be rechecked at response level because most raw responses are unavailable. |
| “Measurement audit” could be mistaken for validation. | Renamed it “AI-label diagnostic and rule sensitivity” and states that zero of 120 sentences has an independent human label. | Human validation is pending. |
| The discussion repeated caveats without saying what survived them. | Added a “Which conclusions survive the checks” paragraph distinguishing stable detector facts from semantic claims. | Stability across thresholds is not construct validity. |

## Claim-to-evidence ledger

| Manuscript claim | Direct evidence | Required scope language | Audit status |
|---|---|---|---|
| Detector component and output-window choices change targeted-control contrasts. | Recovered cache of 700 final-7B responses; exact, embedding-inclusive, 64-token, and 128-token analyses reproduced by `experiments/raw_output_robustness.py`. | Applies only to the recovered final 7B responses. It does not validate other checkpoint trajectories. | Supported and scoped |
| The final-7B advice-versus-emotion point estimate for the historically named empathy-opener category changes sign at 64 tokens. | Recovered-cache union rates: full advice .6133 versus emotion .7600 (gap −.1467, p=.08839); 64-token .3833 versus .1300 (gap +.2533, p=.00340); 128-token .4933 versus .4300 (gap +.0633, p=.48845). | Detector category, not semantic empathy; window result is final-7B only; full and 128-token estimates are uncertain and differ from the older rounded summary. | Supported and scoped |
| Every union false positive in the 120-sentence diagnostic comes from the embedding component. | Exact versus union TP/FP/FN counts in `ai_consensus_validation_v3.json`; exact matching has 0 sampled false positives. | Relative to narrow AI consensus labels, not human ground truth. | Supported and scoped |
| The referral/disclaimer union has low precision in the diagnostic. | 1 TP, 15 FP, 0 FN against the AI target. | The annotation target requires an actual recommendation, while the detector also includes disclaimers and professional-language mentions. This is target mismatch as well as broad detector behavior. | Supported with construct warning |
| Final-model validation-marker output exceeds every targeted control at the primary threshold and its strongest control at all three tested thresholds. | Final-condition tests at τ=.82 and threshold sweep at .80/.82/.84. | Detector-defined output; no claim about human-perceived validation quality. | Supported and scoped |
| Same-interface SFT-to-DPO validation gaps increase and structuring gaps contract at both model sizes. | Historical prompt-level summaries and corrected transition tests: validation +.213 (1B), +.147 (7B); structuring −.367 (1B), −.277 (7B). | Historical detector trajectory; most stagewise raw responses are unavailable, and no semantic or causal interpretation follows. | Supported as a historical detector result |
| Base-to-SFT changes isolate SFT training. | No supporting design: Base uses plain QA and later checkpoints use chat serialization. | Must always be called a combined stage/interface change. | Excluded |
| Aggregate DPO exact matches favor the chosen side for referral/disclaimer and structuring. | 378,301 1B pairs and 378,341 7B pairs; exact McNemar counts and intervals. | Phrase co-occurrence in preference pairs, not semantic quality or training-example attribution. | Supported and scoped |
| The aggregate DPO phrase difference is associated with within-pair length. | Post-hoc three-band analysis: referral approaches zero and structuring is smaller within ±20 words; signs follow the longer side in noncomparable bands. | Descriptive sensitivity analysis; conditioning changes the estimand and does not identify a causal preference. | Supported and scoped |
| Broad referral/disclaimer and structuring matches remain positively associated with the chosen side after flexible length conditioning; narrower-referral intervals include one. | Six exploratory paired conditional fits: broad OR 1.088 [1.026,1.154] (1B), 1.134 [1.067,1.205] (7B); narrower 1.095 [.972,1.234], 1.121 [.996,1.263]; structuring 1.128 [1.108,1.149], 1.190 [1.168,1.213]. Independent implementation reproduces coefficients and cluster SEs. | Protocol was fixed before the new fits after old aggregate results were known; exploratory, not preregistered or confirmatory. OR=1 is no modeled association, not a probability change. Conditioning changes the estimand and does not remove bias or identify causality. | Completed, verified, and scoped |
| Weak marker correlations show that there is no latent response template. | Pairwise phi correlations only. | Weak observed correlations neither establish nor rule out a latent template. | Strong claim excluded |
| Corpus overlap identifies examples that caused model outputs. | Exact phrase counts only. | Co-occurrence and recoverability are supporting context, not attribution. | Strong claim excluded |
| The detector measures semantic empathy or referral quality. | No independent human response-level validation. | Use historical category names and “detector output”; zero of 120 audit sentences has an independent human label. | Strong claim excluded |

## Evidence inventory and numerical provenance

| Evidence source | Unit and size | What it can support | What it cannot support |
|---|---|---|---|
| Final-model prompt battery | 140 prompts across five conditions, five generations per prompt | Prompt-clustered final-condition detector comparisons | Population prevalence, cross-family scaling, or semantic quality |
| Canonical checkpoint grid | 90 personal-advice/general-facts prompts, five generations | Historical advice-minus-facts detector-gap changes | Clean Base-to-SFT training effects or response-level window robustness |
| Five-condition checkpoint extension | All five conditions at Base/SFT/DPO, four generations | Targeted-control stage interactions | Topic-paired counterfactual effects |
| AI-label diagnostic | 120 sentences from 97 seed-0 responses | Error diagnosis against its stated AI annotation targets | Independent human validation or response-level prevalence calibration |
| Recovered final-7B cache | 700 responses | Detector-component, output-window, length, and co-occurrence checks | Generalization to 1B or earlier stages |
| DPO mixtures | 378,301 1B pairs; 378,341 7B pairs; 367,082/367,110 prompt clusters | Paired exact-phrase differences, post-hoc length sensitivity, and exploratory flexible conditional associations | Causal training effects, a probability effect, bias removal, or independent replication across mixtures (21,508 exact pairs overlap) |
| SFT/RLVR/Stage-1 searches | Role-scoped exact counts where indexed | Phrase occurrence and limited recoverability context | Causal provenance; Stage-2 coverage |

## Pending evidence that must remain visible

- Independent human validation: **0/120 audited sentences currently have an
  independent human label**.
- Most checkpoint raw responses remain unavailable, so the final-7B component
  and window checks cannot be generalized to the full trajectory.
- A serialization crossover is required to separate interface from the
  Base-to-SFT transition.
- The completed exploratory conditional DPO analysis must remain labeled as
  specified before the new fits but after the old results were known. Its odds
  ratios must not be described as causal, as “bias removed,” as probability
  changes, or as percentage reductions of the aggregate absolute difference.
- Dolmino-Mix-1124 remains unsearched because no public index was available.

## Readability acceptance checks

- Abstract length: 161 words after conditional-DPO integration, within the
  160–190 target.
- One research question appears in both abstract and introduction.
- The first page distinguishes marker, detector, and proxy.
- “Findings and contributions” begins on page 2 with its first item rather than
  appearing alone at the bottom of page 1.
- The constructed example is explicitly labeled and does not masquerade as a
  model output.
- Main tables use full condition names and explain units in captions.
- Statistical qualifications remain beside the corresponding results.
- The main paper retains the complete Limitations and Ethics sections within
  the eight-page target; references begin on page 8.
- The isolated pdfLaTeX/BibTeX build is 15 pages including references and
  appendices. Its final log has no unresolved references or citations and no
  overfull boxes. All pages were rendered and visually inspected; the
  targeted-control and checkpoint-transition tables render at 8.97 pt without
  scaling, the conditional table is full-width at the standard small-table
  size, and the forest plot is full-width with readable labels and decimal
  ticks.
- A semantic audit covered all 31 internal references. The paired-model
  appendix now points to Section 3.4, and the corpus-count caption now points
  to Appendix G. The changed page 13 was inspected again; the other 14 pages
  remain pixel-identical to the previously checked rendering.
- Two separate eight-question forms are prepared for unfamiliar human
  readers and bound to the current PDF SHA-256. The author rubric is separate
  from the reader materials. All responses remain blank; this preparation
  does not establish independent comprehension or scientific validation.

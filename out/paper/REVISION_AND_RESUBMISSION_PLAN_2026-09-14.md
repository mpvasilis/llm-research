# Revision and resubmission plan

Prepared 14 September 2026 for *Tracing and Auditing Advice-Style Markers Across OLMo-2 Post-Training*, BlackboxNLP submission 128.

**Recommendation: develop a focused measurement paper, with a six-to-eight-week revision and January 2027 ARR as the default planning window.** October 12 is a stretch target only if the evidence and manuscript are ready by October 5. A narrower workshop paper remains a reasonable alternative if the final contribution is primarily an OLMo-2 case study.

This is the proposed work plan. Implementation has started; see [the implementation record](REVISION_PROGRESS_2026-09-14.md) for completed manuscript revisions, workflow repairs, and local smoke tests. The prospective experiments, human annotations, and submissions described below remain pending. Estimates assume access to two independent human annotators and suitable model-generation hardware; neither availability nor funding has been established.

## 1. Decide what the paper will establish

Use one central question:

> How do detection rules and response length change conclusions about advice-style language across OLMo-2 post-training?

The current evidence supports detector-dependent trajectories, sensitivity to output windows, and an association between response length and DPO phrase enrichment. It does not establish semantic empathy prevalence or causal attribution to training examples. A weak detector can motivate a useful measurement study, but diagnosing one weak detector alone is not yet a compelling general contribution.

Build the revised argument around three results, updating their wording after the new evaluation:

1. **Measurement validity:** establish which properties the original detector actually captures, using independent human judgments and clearly aligned definitions.
2. **Sensitivity of conclusions:** identify which condition or checkpoint comparisons persist, weaken, or reverse across detection rules and clearly defined output windows.
3. **Preference-data interpretation:** quantify how marker associations with chosen responses change when accounting for response length, without treating the adjustment as causal identification.

Before substantial computation, compare the proposed contribution with five to eight closest papers on behavioral measurement, length bias in preference data, and checkpoint analysis. Write one paragraph explaining the specific new insight or reusable procedure. If that paragraph only says the present detector performs poorly, narrow to a case-study paper or strengthen the method before targeting a main conference.

## 2. Turn the reviews into completion criteria

| Concern | Revision | Evidence that it is addressed |
|---|---|---|
| All reviewers: presentation problems; B6os and bTEQ: main contribution unclear | Rebuild abstract, introduction, and conclusion around one question and three findings | Two readers unfamiliar with the project independently state the same question and takeaways |
| bTEQ: undefined terms and ambiguous slash notation | Define terms before first use; label model sizes explicitly; put comparisons in tables | No unexplained “frozen,” “axis-targeted,” or model-size slash pairs in abstract or main results |
| hE6p: small evaluation and no human verification | Independent response-level human audit; fresh topic-matched prompts for new specificity claims | Completed human labels, sampling design, uncertainty, and an explicit statement of effective sample size |
| hE6p and bTEQ: formatting and caveats obscure the argument | Reduce boldface and nested result lists; organize each result as question, evidence, interpretation | Main text can be read continuously without reconstructing the argument from enumerations |
| bTEQ: poor detector precision, interface confounding, and length effects | Reconcile labels, extend measurement checks, and limit stage claims to supported comparisons | Main claims visibly match the validated scope and the available raw outputs |
| B6os: overlap search does not establish attribution | Give corpus matching a supporting role and explain what it measures | No claims that phrase overlap identifies the training examples that caused an output |

The first four rows respond to explicit presentation or evaluation criticisms. The added experimental design below is our recommendation for improving the evidence, not a claim that every reviewer demanded every experiment. Keep a private response matrix with reviewer comment, action, revised section, and supporting result. There is no reason to spend this revision arguing over the reviewers' confidence scores.

## 3. Repair definitions and the evaluation workflow first

The current detector, manuscript, and annotation instructions do not use fully aligned targets:

- The detector's referral/disclaimer category includes recommendations, disclaimers, and professional-language fragments; the human guide asks only for an actual professional referral. Annotate referrals, disclaimers, and professional mentions separately, then specify any combined score explicitly.
- “Empathy opener” is detected anywhere in a response, while the guide labels empathy expressions. Record both whether an expression occurs and whether it appears in the defined opening span, or rename the category consistently.
- Reconcile the executable lexicon and embedding exemplars with the appendix. Preserve the submitted detector unchanged as the historical baseline and version any revision separately.

Create one versioned detector specification containing the complete phrase list, exemplars, threshold, matching rules, sentence splitting, and output-window rules. Do not improve the historical baseline retrospectively.

Before collecting new evaluation labels, update the existing validation workflow to reject missing predictions instead of silently treating them as negatives. Separate random and positive-enriched samples in reporting; the current pooled scoring needs sampling-aware treatment. Use uncertainty calculations that preserve prompt or scenario dependence. The current human-validation status is pending.

## 4. Collect the minimum evidence that supports the chosen scope

### A. Independent human validation — required

Use two distinct humans, independently labeling full responses with prompt context. Hide checkpoint identity, detector outputs, existing AI judgments, and the other person's annotations. Resolve disagreements only after both passes, preserving the original labels and adjudication reasons.

Use 60–100 development responses from separate scenario families to refine definitions and the annotation guide. Keep this material outside the held-out evaluation. Split by scenario family across all prompt variants, checkpoints, sizes, and seeds; splitting individual responses would leak closely related examples.

Plan initially for roughly **600 randomly sampled evaluation responses**, stratified across retained checkpoints, model sizes, conditions, and relevant lengths. Add up to **120 separately flagged detector-positive diagnostic responses** if rare categories need more observations. Freeze allocation and stopping rules using the development pilot before examining held-out results. These are planning numbers, not a power calculation or a promise of reliable estimates in every subgroup.

Record sampling probabilities. Report per-category confusion counts, agreement, precision, recall, and intervals. Estimate population quantities from the probability sample with the appropriate stratum weights; use the enriched material separately unless properly weighted. Positive enrichment alone cannot estimate recall or prevalence. Small subgroup samples should remain visibly uncertain.

At 60–90 seconds per response, 600–720 evaluation items labeled twice imply approximately 20–36 total annotator-hours, plus calibration and adjudication. Time the pilot and replace this estimate with observed effort. Do not represent AI annotations as human judgments.

**Completion criterion:** targets and labels align; all annotations and provenance are complete; uncertainty and subgroup errors are reported. Set tolerable measurement error relative to the smallest contrast of interest before the held-out analysis. If error could plausibly overturn a comparison, use direct human-label results or narrow the claim. The detector does not need to perform well for a measurement-failure finding to be valid.

### B. Raw responses and fresh controls — required for stronger checkpoint claims

Only the 700 final-7B responses are currently available locally. First attempt to recover original stage caches from the author's existing storage. If unavailable, generate a clearly versioned new dataset; new stochastic outputs must not be presented as recovered original observations.

A manageable initial confirmatory design is **40 fresh scenario blocks**, each with four variants crossing emotional framing with advice versus information requests: **160 prompts**. Balance health and everyday topics between scenario blocks. Human-check the variants for topic consistency and whether both manipulations make sense. Domain comparisons remain between scenarios, not randomized causal domain effects.

Prioritize **SFT, DPO, and final Instruct**, at **1B and 7B**, using the same chat interface and three sampling seeds: **2,880 responses** before development material or optional additions. This focuses computation on comparable post-training interfaces. The 40 scenario blocks, not the 2,880 responses, determine the independent replication for prompt effects. Use the pilot to choose an adequate number of blocks for the desired interval width or detectable contrast; increasing seeds alone does not solve limited topic coverage. The proposed factorial conditions are a new evaluation and must not be silently pooled with the original five-condition battery.

Store full outputs, exact input serialization, model/tokenizer revisions, seeds, decoding settings, output token counts, and whether generation ended naturally or reached the cap. Establish the generation cap from the pilot and include a planned longer-budget sensitivity subset. Legacy cap indicators are only proxies for actual stopping reasons.

If resources permit only a final-7B audit, publish that narrower scope explicitly. It cannot validate the missing stagewise trajectories.

### C. Length and detector comparisons — required

At every checkpoint used for a main claim, compare the original exact matcher, embedding component, combined detector, and the human labels. Any improved detector is a separate method evaluated on the same untouched test set.

Report full responses alongside the first 64 and first 128 generated tokens, sentence opportunities, length distributions, and stopping reasons. Prefix analysis asks about early response content; it is not an interchangeable “corrected” estimate of full-answer behavior. Handle shorter answers explicitly and avoid selecting only long answers without explaining the changed population. Use matched scenario comparisons and scenario-clustered uncertainty.

Full-response human labels validate full-response predictions only. Treat prefix comparisons as detector sensitivity analyses unless a separate, pre-specified sample of truncated texts is independently annotated and adjudicated in its displayed context. Budget that annotation separately; do not reuse a positive full-answer label as evidence that the same property appears in its prefix.

For DPO data, retain the current aggregate and within-20-word results. Add a pre-specified paired analysis with flexible response-length adjustment, source-specific checks, and uncertainty clustered by repeated prompt. Save compact pair-level features so analysis does not repeatedly require a remote scan. Explain that conditioning on length changes the question and may remove genuine length-related preference behavior; it does not identify a causal preference for a marker.

**Completion criterion:** readers can see which findings persist and which change, with intervals and a declared primary analysis. A sign reversal is a result to explain, not a reason to select another detector or window after seeing the data.

### D. Claim-dependent extensions — defer unless needed

| Extension | When it is worth doing |
|---|---|
| Base/SFT × plain/chat crossover | Required if the paper foregrounds a Base-to-SFT training-stage explanation; otherwise retain the combined stage/interface description and keep it secondary |
| RLVR-specific analysis | Needed if claiming what changes specifically at RLVR; a DPO-to-final comparison alone is insufficient |
| Second model family | Useful to support a broader measurement claim after the OLMo analysis is sound; replicate a few pre-specified key contrasts |
| Causal retraining or data deletion | A separate, larger attribution project; it is not a prerequisite for the proposed measurement paper |

A serialization crossover should use identical prompt content and decoding, report unsupported-format failures, and begin with a feasibility check. It measures interface sensitivity; it does not by itself establish training-example causation.

## 5. Rewrite the paper around the evidence

Working title: **Measuring Advice-Style Language in OLMo-2: Effects of Detection Rules and Response Length**.

Write a 160–190-word abstract with motivation, one question, a brief method description, two or three findings, and one scope statement. Leave seed counts, exact threshold sweeps, and paired model-size numbers to Methods. Finalize the findings only after the new analyses.

Use this main-text order: motivating example and question; related work and specific contribution; definitions and study design; human validation; sensitivity of condition/checkpoint conclusions; DPO length analysis; discussion and conclusion. Put the measurement audit before the claims that depend on it.

Replace the five artifact-oriented contributions with three evidence-based claims. Define a marker as the specified language feature and a detector as the rule predicting its presence; use “proxy” only when the distinction is necessary. Replace “frozen” with “fixed before evaluation.” Explain additional controls directly. Replace “detector opportunity” with the number of sentences or tokens available for a match.

Use three main visuals: a small annotated response example showing a valid match and a false positive; condition/checkpoint contrasts under the pre-specified detector and window comparisons; and DPO effects across response-length differences. Give 1B and 7B explicit columns or panels. Move phrase-by-phrase inventories and secondary corpus searches to the appendix, while keeping the evidence essential to assessing the claims in the main text.

Move the Dolmino index limitation to corpus-search Methods and Limitations. Consolidate repetitive caveats, but retain any qualification essential to interpreting a result beside that result. Replace “the bundle decomposes” with a precise statement about observed marker co-occurrence; weak correlations do not establish the absence of a latent advice template.

## 6. Schedule and decision points

| Dates | Work | Deliverable |
|---|---|---|
| September 14–20 | Fix research question and targets; reconcile detector specification; identify annotators; run development and compute pilots; map closest literature | One-page study specification, annotation guide, measured resource estimate |
| September 21–October 4 | Recover caches or run versioned generation; collect independent labels; prepare new manuscript structure in parallel | Complete response archive and annotation progress report |
| October 5 | Assess the accelerated route | Submit in October only if labels, analyses, manuscript, and external reading check are complete; otherwise keep revising |
| October 5–18 | Complete annotation/adjudication; run pre-specified analyses; reconcile every headline number | Human audit, comparison tables, DPO analysis, final claim list |
| October 19–November 1 | Rewrite; obtain two independent readability/methodology reads; reproduce results and verify PDF/artifact | Submission-ready manuscript and supplement |
| November 2–8 | Resolve substantive issues and choose the route matching the achieved contribution | Final readiness decision; no unresolved essential evidence |

The schedule is an estimate, not a deadline-driven instruction to stop collecting required evidence. Human availability and generation throughput are the main dependencies. If either is unavailable, reduce the scientific scope explicitly or extend the schedule. Optional model families and causal retraining should not delay the core measurement study.

## 7. Resubmission routes verified on September 14, 2026

| Route | Verified timing | Recommendation |
|---|---|---|
| ARR → NAACL 2027 or COLING 2027 | Submission October 12, 2026; commitment December 23, 2026 | Stretch route after the October 5 readiness check. NAACL explicitly welcomes empirical and evaluation contributions. [ARR dates](https://aclrollingreview.org/dates), [NAACL call](https://2027.naacl.org/calls/main_conference_papers/) |
| ARR → ACL 2027 | ARR currently lists January 2027; the exact day and commitment deadline are not yet listed | Default planning window for the strengthened paper; select the final venue based on completed evidence. [ARR dates](https://aclrollingreview.org/dates) |
| EACL 2027 Student Research Workshop | Direct submission December 15, 2026; optional mentorship November 6, 2026; research-paper first author must be a student | Suitable conditional alternative for a focused measurement study; student status has not been established. Archival and non-archival options exist. [SRW call](https://2027.eacl.org/calls/srw/) |

EACL 2027 main-conference entry required ARR submission by August 3, already past. It is not a fresh-review route for this revision unless earlier eligible ARR reviews exist. [ARR dates](https://aclrollingreview.org/dates)

At NAACL/COLING commitment, choose one primary venue; an optional secondary choice does not guarantee secondary consideration. NAACL Main and Findings are possible conference outcomes, not separate initial ARR submissions. [Joint-process FAQ](https://2027.naacl.org/faq/)

TACL is a later possibility for substantially expanded work. For papers receiving relevant ARR/major ACL conference reviews, its nine-month waiting period is measured from the conference submission deadline or the beginning of the ARR submission month. ARR resubmissions restart the period. Direct BlackboxNLP workshop review alone is not clearly addressed by that wording; verify the complete review history before choosing TACL. [TACL policy](https://transacl.org/ojs/index.php/tacl/about/submissions)

**Check ARR eligibility early.** Starting October 2026, guaranteed review requires a qualified designated service contributor; otherwise submissions enter a capacity lottery. A qualified outside researcher familiar with the work may serve. Complete profiles and registration promptly. Official pages currently differ on some intermediate registration dates, so follow the live submission instructions and register at submission. [ARR call](https://aclrollingreview.org/cfp), [service qualifications](https://aclrollingreview.org/qualifications)

ARR does not import workshop reviews. If this paper has never been submitted to ARR, it needs fresh ARR review; the workshop's score of 1 is not an ARR meta-review score. If an ARR version exists, disclose it and provide the required revision explanation. The submission form's previous-version fields refer specifically to previous ARR submissions. Keep the BlackboxNLP response matrix regardless. [ARR author guidance](https://aclrollingreview.org/authors), [submission fields](https://aclrollingreview.org/submissionform)

Choose one active review route. The December SRW route is not a simultaneous backup during October ARR review, and an archival workshop publication affects later reuse of substantially overlapping material. Follow the selected venue's current anonymity and AI-assistance disclosure requirements; the old submission notes should not be treated as current policy. [ARR call](https://aclrollingreview.org/cfp)

## 8. Submission readiness

The paper is ready when its main question is clear to unfamiliar readers, its categories match the annotation task, two human passes and adjudication are complete, every main comparison has adequate raw-data support, and its claims reflect the observed uncertainty and sensitivity. Reproduce the final tables from the released artifacts, reconcile the appendix with the detector specification, and inspect the compiled PDF for readability. These are completion criteria for this plan, not guarantees of acceptance.

If the evidence remains confined to one checkpoint and one problematic detector, prefer the narrowly scoped workshop version. If the human audit and controlled evaluation yield a reusable measurement lesson supported across checkpoints, the ARR/main-conference route is more defensible.

## Existing materials to reuse

- [Current manuscript](C:/Users/vasilis/Documents/GitHub/llm-research/out/paper/acl_latex.tex)
- [Annotation guide](C:/Users/vasilis/Documents/GitHub/llm-research/out/paper/ANNOTATION_GUIDE.md) and [validation workflow](C:/Users/vasilis/Documents/GitHub/llm-research/experiments/validation_v3.py)
- [Pending human-validation status](C:/Users/vasilis/Documents/GitHub/llm-research/out/results/validation_status_v3.json)
- [Existing response robustness analysis](C:/Users/vasilis/Documents/GitHub/llm-research/experiments/raw_output_robustness.py) and [results](C:/Users/vasilis/Documents/GitHub/llm-research/out/results/raw_output_robustness.json)
- [Existing DPO robustness analysis](C:/Users/vasilis/Documents/GitHub/llm-research/experiments/dpo_phrase_robustness.py) and [results](C:/Users/vasilis/Documents/GitHub/llm-research/out/results/dpo_phrase_robustness.json)
- [Artifact provenance manifest](C:/Users/vasilis/Documents/GitHub/llm-research/out/results/revision_artifact_manifest_20260716.json)

These files support the plan but remain part of the existing working state. The revision should preserve the submitted baseline and clearly distinguish historical results from newly generated evidence.

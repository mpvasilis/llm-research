# Provenance to Proof: A Causal, Open-Pipeline Account of Where Pro-Social Advice Behaviors Come From in Instruction Tuning

**Hook.** We make data provenance falsifiable. Using the fully-open OLMo 2 pipeline, we re-run the instruction-tuning stage with versus without a *provenance-selected, human-auditable* cluster of empathy/disclaimer/crisis-referral conversations and measure whether specific free-form advice behaviors appear and disappear — establishing whether a cheap, internals-free, lexically-auditable selector is a *valid causal selector* for safety-relevant pro-social behaviors, and whether those behaviors are installed at SFT versus latent in pretraining.

**Abstract (~160 words).** Where does a language model learn to open with empathy, validate feelings, or tell a user to "consult a professional"? Data-attribution work answers this observationally (n-gram frequency, influence functions, datamodels) or, in very recent concurrent work, causally via probe-selected counterfactual retraining of post-training data. We contribute the first **SFT-stage, pro-social-behavior, lexically-auditable-selector** causal study, anchored on two contributions the concurrent literature does not provide: (1) a head-to-head demonstration that a cheap, internals-free **lexical/corpus provenance selector** picks conversation clusters whose leave-cluster-out re-SFT yields behavior deltas comparable to those from an expensive activation probe and a gradient selector (LESS); and (2) a **pretraining-vs-SFT behavioral dissociation** distinguishing phrasing latent in the 4T-token base from behavior installed at instruction tuning. We use the fully-open OLMo 2 1B pipeline with a pre-registered difference-in-differences design (size/token/source-matched random **and** coherent-distractor controls), pilot-gated power, and a single-GPU budget (~$2–6k).

---

## 0. What Changed After Adversarial Review (Read This First)

Three expert reviews (a TDA/causal-inference specialist, a pragmatic area chair, and a novelty/design reviewer) converged on the same verdict: **borderline (6/10), no fatal flaws, but the framing overclaimed and three load-bearing design choices were under-resourced.** This revision makes the following honest corrections, each of which is now baked into the design rather than the rebuttal:

1. **Corrected a factual misstatement about the closest concurrent work and re-staked novelty on what actually survives.** We verified arXiv 2602.11079 ("In-the-Wild Model Organisms…", Goodfire; OLMo 2 **7B**, latest revision **v3, 27 Apr 2026** — *not* "1B / Feb 2026" as our prior draft stated). It **does** report a random-removal baseline, **does** do unsupervised multi-behavior discovery via clustering, and **does** report a discrete dose-response. Our prior draft's claim that these were missing was **wrong**. We retract it. Novelty is now staked **only** on the four differentiators that genuinely survive (§1.2), with the **lexically-auditable causal selector** and the **pretraining-vs-SFT dissociation** promoted to the headline.

2. **Promoted the two reviewer-identified "strongest ideas" from optional to primary, funded line items:** (a) the **selector head-to-head** (tracer vs. probe vs. LESS vs. random/distractor under identical re-SFT), now in the run matrix and budget; (b) the **coherent-distractor cluster as the headline DiD comparator** (not plain random), because the real confound is "is coherent-slice removal generically disruptive?", which random removal cannot rule out.

3. **Replaced asserted statistical power with a pilot-gated power simulation and a fixed-effects-per-seed fallback**, dropped the lexical instrument from the primary outcome (eliminating residual circularity), added a pre-registered **remove-all-enriched-clusters** arm so a null is interpretable, sharpened RQ4 into a *behavioral* dissociation (base-model in-context emission, not n-gram counts), and committed to a **CPU-only preliminary that de-risks the entire program before any GPU spend.**

A standing **"Reviewer Concerns & How We Address Them"** subsection (§10) tracks every major concern, including the ones we cannot fully resolve.

---

## 1. The Causal Claim, Honestly Scoped

### 1.1 The claim

A behaviorally-defined, lexically-auditable cluster of SFT conversations is *causally responsible* for a specific free-form advice behavior, in the strong sense that removing it from the instruction-tuning data and re-running the official SFT pipeline makes the behavior drop by a margin that (a) **exceeds the drop from removing an equally-coherent, equal-size, unrelated distractor cluster** (the primary confound control) and from removing size/token/source-matched random data (the volume control); (b) is monotone in the fraction of the cluster removed; (c) is *specific* (removing the disclaimer cluster suppresses disclaimers but not empathy); and (d) leaves unrelated capabilities (MMLU-Pro, GSM8K, HumanEval+, IFEval) intact.

### 1.2 Why this is novel — the honest version

We do **not** claim "first causal account." Concurrent work (2602.11079-v3, below) already does counterfactual retraining on OLMo 2 post-training data. We do **not** claim a new attribution estimator; the field (Datamodels, Ilyas et al. ICML 2022; TRAK, Park et al. ICML 2023) already treats leave-out retraining as the ground truth that cheap surrogates approximate — their headline metric, the Linear Datamodeling Score, *is* the correlation between predicted and actually-retrained outputs.

**Our contribution is a protocol plus a validated cheap selector, staked on the conjunction of four differentiators that survive scrutiny against the closest concurrent work:**

1. **Stage.** We ablate the **SFT** stage; the closest concurrent work ablates **DPO**. SFT-data attribution is cleaner for *acquisition* claims (DPO operates on preference pairs over an already-instructed model, confounding "where the behavior came from").
2. **Behavior class.** We target **pro-social, safety-*promoting*** behaviors (empathy, validation, disclaimers, crisis referral) and factual-QA controls; concurrent work foregrounds **undesirable** behaviors (harmful compliance). Establishing the provenance of *desirable* safety behavior is the under-studied and more curation-relevant direction.
3. **Selector.** Ours is **external, lexical/corpus, human-auditable** (infini-gram verbatim spans + DuckDB substring match returning the exact removable conversations); concurrent work uses an **opaque activation probe**. Whether a cheap internals-free selector is *causally valid* is an open question we answer head-to-head (RQ5).
4. **Layer split.** We report a **pretraining-vs-SFT behavioral dissociation** (does the *base* model emit the behavior in-context, or only the SFT model?); no competitor has a true-pretraining-corpus layer.

**The single closest concurrent work, stated accurately.** **"In-the-Wild Model Organisms…"** (arXiv 2602.11079, Goodfire; **v3, 27 Apr 2026**; OLMo 2 **7B**, with validation on larger models). It does counterfactual retraining on OLMo 2 **post-training (DPO)** data, with **probe and LESS** selectors, **a random-removal baseline**, **unsupervised multi-behavior clustering**, and a **discrete dose-response** (3k/12k/30k datapoints; e.g., remove 30k random → 6.67% harmful vs. 7.63% baseline vs. 2.86% for their method; ~63%/78% reductions reported, 84/85% figure to re-verify against v3). **We treat this as prior work, cite it in the first paragraph of the camera-ready, and differentiate only on the four axes above.** Our refinements over their controls — **token-mass + per-source-skill matching** and a **coherent-distractor arm** — are presented as *refinements of rigor*, not as controls they lack.

| Work | Actual retrain? | Stage | Behavior type | Selector internals-free? | Pretrain-vs-SFT split? | Coherent-distractor control? | Fully-open |
|---|---|---|---|---|---|---|---|
| Datamodels / TRAK | surrogate | — | loss/acc | n/a | no | no | partial |
| Influence (Koh-Liang→Grosse→TrackStar) | no (infinitesimal) | pretrain/loss | token loss | no (gradients) | no | no | partial |
| LESS (ICML 2024) | no (additive select-to-add) | SFT | capability | no (gradients) | no | no | partial |
| **In-the-Wild Model Organisms (2602.11079v3)** | **yes** | **DPO** | undesirable | no (probe) | no | no (random only) | yes (OLMo 2) |
| **This proposal** | **yes** | **SFT** | **pro-social/safety** | **yes (lexical)** | **yes** | **yes** | **yes** |

The correlational pretraining-side predecessor is **"Generalization v.s. Memorization"** (Wang et al., ICLR 2025, arXiv 2407.14985), which ties *capabilities* to *pretraining* n-gram frequency over Pile/Dolma. We extend distributional-memorization tracing **observational → interventional** and **pretraining-capabilities → post-training-behaviors**.

---

## 2. Research Questions and Falsifiable Hypotheses

**RQ5 and RQ4 are the headline.** RQ1–RQ3 establish the causal effect exists; RQ5 establishes our selector is a *valid, cheap* way to find it; RQ4 establishes *where* the behavior is installed.

**RQ1 (causality vs. the right control).** Does removing a provenance-selected behavior cluster reduce the behavior beyond removing an equally-coherent distractor cluster (and beyond matched-random)?
- **H1.** ΔΔ = (drop under targeted-100%) − (drop under **distractor**-100%) > 0, 95% CI excluding 0. *Falsified* if targeted removal does not beat coherent-distractor removal. (Secondary: targeted vs. matched-random, the volume check.)

**RQ2 (dose-response).** Is the targeted effect monotone and steeper than distractor/random?
- **H2.** Behavior rate is monotone-decreasing over {0,25,50,100}% targeted removal with a steeper fitted slope than both control arms. *Falsified* by a flat or non-monotone targeted curve.

**RQ3 (specificity & placebo).**
- **H3a.** Removing the disclaimer cluster reduces disclaimers more than empathy, and vice versa (behavior × ablated-cluster interaction).
- **H3b.** Deterministic capability benchmarks (MMLU-Pro, GSM8K, MATH, BBH, HumanEval+, IFEval) do not move beyond the matched-random arm. *Falsified* if targeted removal degrades capabilities as much as random does (→ volume shock, not attribution). (We drop AlpacaEval 2 from the placebo set: it is LLM-judged and length-biased; we report it only as a style check.)

**RQ4 (layer dissociation — sharpened to a behavioral test).** Is the behavior *installed at SFT* rather than latent in pretraining?
- **H4.** Zero-shot, the base OLMo-2-0425-1B emits the behavior at a low rate; **few-shot prompting the base model partially recovers it** (latent-capability signal); full-mixture SFT raises it sharply; targeted cluster removal collapses it. We report **three quantities** — (i) base zero-shot emission, (ii) base few-shot emission, (iii) post-SFT emission — and interpret the gap. *Falsified* if base zero-shot already ≈ post-SFT (not SFT-installed) **or** if few-shot fully recovers the behavior (fully latent, SFT merely elicits). Infini-gram verbatim counts are reported as *context only* (a common phrase will trivially have high counts); they do **not** carry the dissociation claim.

**RQ5 (selector validity — promoted to primary, funded).** Does the cheap lexical/corpus tracer select a cluster whose ablation matches what a probe (Goodfire-style) and LESS (gradients) select, at far lower selection cost?
- **H5.** Under identical leave-cluster-out re-SFT, tracer-, probe-, and LESS-selected clusters produce behavior deltas whose pairwise differences fall within a pre-registered equivalence margin, while all three beat distractor/random. **We test this with a TOST equivalence design (§5), not "fail-to-reject," and pre-register the margin and the seed count needed to power it.** *Falsified* if the tracer-selected cluster yields a materially smaller delta than probe/LESS, **or** if we are underpowered to make either an equivalence or a difference claim (in which case RQ5 is reported as inconclusive, honestly).

---

## 3. Method: The Counterfactual SFT Pipeline

### 3.1 Substrate (verified, fully open)
- **Base:** `allenai/OLMo-2-0425-1B` (1B; ~4T-token stage-1 OLMo-mix-1124 + ~50B-token stage-2 Dolmino-mix-1124; Apache 2.0). **We start from BASE**, not the released -Instruct checkpoint.
- **SFT mixture (ablation universe):** `allenai/tulu-3-sft-olmo-2-mixture-0225` — **866,138** conversations, parquet, ~1.27 GB, 19 source subsets; each row `{messages:[{role,content}], source}`.
- **Recipe:** `allenai/open-instruct`, OLMo-2 SFT config (`docs/olmo2.md`): base → SFT, **max_seq_len 4096, per_device_bs 2, grad_accum 8, LR 3e-5, linear schedule, warmup_ratio 0.03, weight_decay 0.0, 2 epochs, DeepSpeed ZeRO-3, bf16, `--add_bos`, `--seed` as first-class CLI flag.** Maintainer-stated wall-clock: **~9 h on 1×8×H100 node = ~72 H100-GPU-hours/run** (verified in `docs/olmo2.md`). We **ablate at SFT only** for the primary claim (DPO/RLVR add cost and confound SFT-data attribution); an optional DPO-robustness arm checks the delta survives preference tuning.

### 3.2 Cluster definition (CPU, free) — the selector, with the hidden risks de-risked

The entire causal claim rests on this step, so we de-risk it explicitly:

1. **Embed** all ~866k assistant turns with a strong sentence encoder (E5 / GTE / Instructor). To make HDBSCAN on 866k high-dim points actually CPU-tractable (a reviewer correctly flagged "overnight on CPU" as optimistic), we **reduce dimensionality first** (PCA→UMAP to ~50D) and use the standard ANN-accelerated HDBSCAN; this is the documented practical path and is benchmarked in M1, not assumed.
2. **Cluster** with **HDBSCAN** (primary; density-based, flags noise) and **k-means** (silhouette-chosen k) as a robustness check; the causal result must hold under both.
3. **Ablation granularity is a pre-registered, tested decision (new).** Because empathy/disclaimer language can be a single sentence inside an otherwise-diverse conversation, **whole-conversation removal could itself be the volume/diversity confound.** We pre-register a **conversation-level vs. turn/sentence-level surgical-removal comparison** as a primary methodological result: if surgical removal (deleting only the behavior-bearing turns, keeping the rest of the conversation) yields the same delta as whole-conversation removal, the confound is ruled out; if not, we report the difference and use surgical removal for the headline.
4. **Validate + label via the two-layer tracer.** A cluster qualifies as "empathy"/"disclaimer"/"crisis-referral" only if it is **enriched** versus base rate (we pre-register the **enrichment-ratio threshold** rather than leaving "enriched" vague). Pretraining layer: infini-gram verbatim counts (context for RQ4). Instruction layer: DuckDB role-scoped substring/`list_filter` returns the exact removable conversations.
5. **Freeze removed-conversation-ID sets, ablation granularity, and the eval-prompt set BEFORE training any ablated model.** Pre-register doses {0,25,50,100}%.

### 3.3 Controls: distractor (headline) + matched-random (volume) + remove-all-enriched (over-determination)
- **Coherent-distractor arm (headline DiD comparator):** an unrelated but **equally coherent, equal-size** semantic cluster. This is the control that rules out "removing any tight cluster is generically disruptive" — the confound matched-random cannot address. **It gets full seed power, equal to the targeted arm** (corrected from the prior 3-seed afterthought).
- **Size-matched random arm (volume check):** same conversation count, total token mass, and per-source skill distribution. Answers "is it just data volume?" Secondary, not headline.
- **Remove-all-enriched-clusters arm (new, primary — makes a null interpretable):** remove *every* cluster above the enrichment threshold, not just the densest. If the single-cluster effect is null but the all-cluster effect is large, the behavior is **over-determined/distributed** (the Simfluence non-additivity prediction), and we can say so rigorously rather than confounding "null effect" with "removed too little." Costed in the budget.
- **Estimand = difference-in-differences:** (targeted drop) − (distractor drop). Match quality reported for all arms.

### 3.4 Generation + measurement (CPU + cheap API)
- **Generate** OLMo-2-1B answers on the **free CPU workstation** (or cheap spot GPU): frozen probe set of relationship/health/emotional-support prompts (breakup, snoring, anxiety, panic) + factual-QA controls. Primary at **temp=0 greedy**; temp>0 for robustness; **≥3 generation seeds** averaged.
- **Behavior instruments — the primary outcome uses NON-LEXICAL instruments only (circularity fix).** The lexicon/regex is used **strictly for cluster selection**, and is **excluded from the primary behavior metric** (otherwise the selector and the metric share the same signal and the effect is partly tautological). The primary outcome triangulates:
  1. **Published classifiers/rubrics:** empathy/validation via **EPITOME** (ER/IP/EX; Sharma et al., EMNLP 2020); structure via **ESConv** 3-stage/8-strategy taxonomy (Liu et al., ACL 2021); safety via a clinician-validated disclaimer+referral ordinal scale (**to-verify**: JMIR 2026 5-level scale — citation and quartiles to be confirmed before submission; if unconfirmable we substitute our own pre-registered, human-validated rubric).
  2. **Ensemble LLM-judge** (`gpt-4o-2024-08-06` + Claude + Gemini), swap-order double-evaluation (position bias), symmetric formatting, decomposed sub-criteria, required rationale. Self-preference is structurally reduced (OLMo 2 is in no commercial judge's family), but we rely on the **human-IAA anchor**, not the family argument, as the primary defense (judges still share stylistic priors — verbosity, hedging — that correlate with the target behaviors).
- **Judge reliability** benchmarked against **human expert–expert agreement on the same rubric** (target precedent: Nat. Mach. Intell. arXiv 2506.10150, expert-LLM α 0.51–0.75 vs. expert-expert α 0.29–0.78 — **to-verify**), on a **stratified human-annotated subset (~200–400 items, ≥3 annotators, codebook + adjudication) that includes OLMo generations.** Report Krippendorff α, Fleiss κ, judge-vs-human ICC.
- **Verbosity:** because empathy/disclaimers are partly *expressed through* added length, we **report both length-adjusted and unadjusted effects and pre-specify the unadjusted as primary** (regressing out length could regress out the effect); length deltas reported alongside.

### 3.5 CPU-vs-GPU split (explicit)
| Step | Hardware | Cost |
|---|---|---|
| Embedding, DR, clustering, tracer, ID/granularity selection | **CPU (existing workstation)** | **free** |
| Generating 1B answers (all conditions × gen-seeds) | **CPU (or cheap spot GPU)** | **free / negligible** |
| Behavior classifiers (EPITOME/ESConv) | **CPU** | **free** |
| LLM-judge API (~50–150k calls) | API | **tens of $** |
| **Leave-cluster-out re-SFT (the only GPU cost)** | **GPU** | **see §6** |
| Capability benchmarks (inference) | CPU/cheap GPU | negligible |

---

## 4. Experiments: Conditions × Seeds, Metrics, Baselines

### 4.1 Run matrix
Shared seeds across conditions so **seed is a blocking factor** (paired analysis). Seed counts are **pilot-gated** (§5): the matrix below is the *target*; if the pilot shows large seed variance we reallocate toward more seeds on the two headline arms and fewer interior doses (a reviewer-endorsed contingency).

| Cell | Removal | Dose | Seeds | Runs |
|---|---|---|---|---|
| C0 baseline | none | 0% | 8 | 8 |
| Targeted-100 | target cluster | 100% | 8 | 8 |
| **Distractor-100 (headline control)** | coherent unrelated cluster | 100% | **8** | **8** |
| Random-100 (volume check) | matched random | 100% | 5 | 5 |
| Targeted-50 | target cluster | 50% | 3 | 3 |
| Targeted-25 | target cluster | 25% | 3 | 3 |
| Distractor-50 | distractor | 50% | 3 | 3 |
| Second-behavior leave-out (H3a) | empathy cluster | 100% | 3 | 3 |
| **Remove-all-enriched (over-determination)** | all enriched clusters | 100% | 3 | 3 |
| **RQ5: probe-selected (Goodfire-style)** | probe cluster | 100% | 5 | 5 |
| **RQ5: LESS-selected** | LESS cluster | 100% | 5 | 5 |
| **Total** | | | | **54** |

- **Lean tier-1 (minimum publishable, causal-selector spine):** baseline(8) + targeted-100(8) + distractor-100(8) + random-100(5) + probe-100(5) + LESS-100(5) = **39 runs** (clean DiD with the *right* control + the RQ5 selector head-to-head). Dose/specificity arms are added if budget allows.
- **Pilot (variance estimate, budget-gating):** baseline(4) + targeted-100(4) + **random-100(4)** = **12 runs**. *Corrected from 4 to 12:* 2 seeds/arm cannot estimate a variance, and the DiD needs the **control-arm** variance, so the pilot must include it. The pilot reports empirical between-run behavior-rate SD and feeds the §5 power simulation that gates the full matrix.
- Optional robustness (budget permitting): one **DPO** arm on ablated SFT checkpoints (+~16 GPU-h/run); one **OLMo-2-7B** confirmatory targeted/distractor pair for scale.

### 4.2 Metrics
- **Behavior:** per-behavior rate/score from the **non-lexical** stack (EPITOME/ESConv/JMIR-ordinal + ensemble judge), with convergent (instrument–instrument) and criterion (vs. human gold) validity.
- **Causal effect:** DiD in behavior rate (targeted − distractor) with hierarchical bootstrap 95% CI; standardized effect (**Cohen's h** for proportions, **GLMM log-odds** with CI); **dose slope** for targeted vs. controls and the test of their difference.
- **Provenance overlap:** infini-gram verbatim counts (pretraining context) and SFT substring enrichment ratio within the removed cluster vs. base rate (the auditable link).
- **Capability-regression:** MMLU-Pro, GSM8K, MATH, BBH, HumanEval+/BigCodeBench, DROP, IFEval after every retrain.

### 4.3 Baselines
1. **Correlational tracer** (existing tool, observational) — what we improve on.
2. **Coherent-distractor cluster** — the headline DiD comparator.
3. **Matched-random** — the volume check.
4. **Probe selector (Goodfire-style) and LESS** — the **RQ5 head-to-head**, now funded in the matrix: tracer- vs. probe- vs. LESS- vs. distractor/random-selected clusters, all ablated via the *same* leave-cluster-out re-SFT, scored on the *same* non-lexical metric. *Honest caveat:* running LESS and the probe requires gradient/internals infrastructure on our side, which partially qualifies the "internals-free" selling point — but the point is precisely that *deployment* of the selector (the tracer) is internals-free and cheap, validated *once* against the expensive methods.
5. **(Optional) Unlearning proxy:** machine-unlearning the cluster (GUDA / Wang et al. NeurIPS 2024) to show selector+re-SFT agrees with selector+unlearning, strengthening cost and robustness.

---

## 5. Statistical Analysis Plan (Few-Run Causal Inference, Honestly)

The headline difficulty, stated plainly: **the causal estimand lives at the training-run level, and even at 8 seeds/arm the top-level n is small.** We do not paper over this.

- **Power is demonstrated, not asserted (new, mandatory, pilot-gated).** From the 12-run pilot we estimate the empirical between-run behavior-rate SD, then **simulate the GLMM DiD at n=5 and n=8 seeds** and report the **minimum detectable ΔΔ**. The full matrix is committed only if the MDE clears the pre-registered minimum effect of interest (**≥15–20 absolute pp**, anchored to the verified ~63% precedent reduction) with ≥80% power. If it does not, we reallocate the budget to **≥10 seeds on the two headline arms** and drop interior doses. This decision rule is pre-registered.
- **Primary model — item-level GLMM:** `behavior ~ condition * dose + (1|seed) + (1|eval_prompt)`, exploiting many eval prompts for power. **Honest caveat:** with 5–8 seeds the `(1|seed)` variance component is weakly identified; we therefore *also* report the conservative fallback below and treat agreement between them as the robustness bar.
- **Conservative fallback — fixed-effects-per-seed + permutation test (new).** Treat each seed as a fixed effect, compute the per-seed targeted-vs-distractor difference, and use a **permutation/randomization test** over the seed-level differences for the headline p-value. This avoids estimating an unidentified variance component and is valid at small n.
- **Headline contrast:** paired per-prompt DiD; **hierarchical/cluster bootstrap resampling seeds AND eval prompts** (≥1,000 resamples). With only 5–8 seeds the seed-level bootstrap is unstable, so the **permutation test is the primary inference and the bootstrap CI is corroborating.** We do **not** use the CLT/normal approximation for the headline.
- **RQ5 as equivalence (TOST), not fail-to-reject (new).** Pairwise selector deltas tested for equivalence within a pre-registered margin via TOST; we pre-register the seed count needed to power TOST and, if unmet, report RQ5 as inconclusive rather than implying equivalence from a null.
- **Per-behavior:** McNemar + Cohen's h (binary); paired Cohen's d (ordinal); Holm correction across behaviors.
- **Pre-registration:** lexicon (selection only), rubrics, judge prompts (pinned versions), human codebook, removed-ID sets, ablation granularity, enrichment threshold, eval-prompt set, decoding params, MEI, and the power-gated seed-allocation decision rule.

---

## 6. Compute Budget

**GPU is needed only for SFT re-training.** Tracing, embedding, clustering, 1B generation, classifiers run free on the existing CPU workstation; LLM-judge calls are tens of dollars.

**Per-run anchor (verified):** OLMo-2-1B full SFT = **~9 h on 8×H100 = ~72 H100-GPU-hours/run** (`open-instruct/docs/olmo2.md`, verified verbatim, with LR 3e-5 / 2 epochs / bs2×ga8 / seqlen 4096). H100 SXM ≈ **$2.50–2.99/GPU-hr** on-demand (Spheron/RunPod/Lambda, mid-2026); A100-80GB ≈ **$1.07 on-demand / $0.60 spot**.

**Memory footprint, stated in full (corrected):** AdamW full-FT of 1B in bf16 ≈ model 2 GB + grads 2 GB + fp32 optimizer states ~12 GB + master weights, i.e. ~16–20 GB of *states* and **~30–48 GB total with activations** — **fits comfortably on one A100-80GB** (verified feasible), not on a 24 GB card except with grad-checkpointing + 8-bit optim.

**GPU-hours vs. wall-clock (corrected framing):** 72 GPU-h is fixed work; single-GPU saves money (no idle GPUs in a node) but costs **calendar time (~72 h/run)**, which matters for sequential runs on one GPU in a 5-month timeline. We therefore budget **multi-GPU for throughput where the timeline needs it** and single-GPU where it doesn't.

| Plan | Runs | GPU-h | $ (H100 on-demand) | Notes |
|---|---|---|---|---|
| **Full matrix (54 runs)** | 54 | 3,888 | **$9,720–11,624** | DiD + dose + specificity + RQ5 + over-determination |
| **Recommended (39 runs, tier-1 spine)** | 39 | 2,808 | **$7,020–8,396** | DiD w/ distractor + RQ5 selector head-to-head |
| **Pilot (12 runs)** | 12 | 864 | **$2,160–2,585** | Variance + wall-clock; budget-gating |

**Cost-reduction levers (design-preserving):**
- **Single-GPU full FT on A100-80GB.** At ~60–72 GPU-h/run, the 39-run spine ≈ **$1,404–2,808 (spot–on-demand A100)**. This is the practical headline floor.
- **Subsample** the mixture to a fixed 30–50% held constant across **all** arms → ~2–3× faster/run, fair because every arm sees the same base.
- **LoRA only for interior-dose cells / extra seeds** (1B LoRA ~2 GB; QLoRA ~0.6 GB). **Headline targeted-100 / distractor-100 / RQ5 arms run in FULL fine-tuning** — LoRA "learns less, forgets less" (arXiv 2405.09673) could attenuate the delta and diverges from the official recipe.
- **Frugal floor:** subsample + LoRA on community 4090 / A100-spot → **~$150–400** for a reduced but valid study.

**Headline budgets to quote (honest range):** lead with the **verified-anchor** figure — **~$7–8.4k** for the 39-run spine on 8×H100 (defensible, grounded). The **optimistic single-GPU/subsampled** figure (**~$1.4–2.8k**) is explicitly **pilot-gated** and flagged ±50% (the single-GPU per-run wall-clock is an extrapolation from the multi-GPU anchor, not measured on this stack). **CPU-only MVP / fallback:** the correlational tracer study + the base-vs-released-SFT behavior comparison + the CPU-only preliminary (§7) + the measurement stack — yields the RQ4 dissociation and the selector-enrichment evidence as an observational paper, with counterfactual re-SFT added once a GPU is rented.

> **Per-run hours carry ±50% uncertainty** and the budget is **pilot-gated**: the 12-run pilot fixes the real single-GPU wall-clock before the full matrix is committed.

---

## 7. CPU-Only Preliminary: De-Risk Before Any GPU Spend (New)

Before renting a single GPU, we run a **CPU-only preliminary on the already-released checkpoints** that converts several "expected" claims into shown ones and gates the program:

1. **Cluster feasibility (M1 deliverable):** the actual size, coherence (silhouette/density), and enrichment ratio of the empathy/disclaimer/crisis clusters on the real 866k mixture, and whether at least one behavior has a *small, content-matched, removable* cluster (if the cluster is a huge diffuse fraction, the volume confound is unavoidable and we pivot behaviors).
2. **Base-floor + selector enrichment, no retraining needed:** compare **base OLMo-2-0425-1B vs. the released OLMo-2 SFT checkpoint** on the frozen probe set. If the released SFT model emits the behavior far above the base floor **and** the tracer-selected cluster is strongly enriched, the entire causal program is de-risked on CPU. If the base already emits at near-SFT rates, we pivot to a behavior with a larger SFT-installed increment (e.g., crisis-referral / structured ESConv strategies) **during M1, not post-hoc.**
3. **RQ4 base few-shot probe:** measure base zero-shot vs. few-shot emission to pre-classify each candidate behavior as "latent-elicitable" vs. "SFT-acquired" before spending GPU on it.

This preliminary is the single highest-value de-risking step and is **free**.

---

## 8. Expected Headline Result — and Why It's Interesting Either Way

**Expected:** a **large, specific, dose-monotone** behavior drop that **beats the coherent-distractor control**, with tracer-, probe-, and LESS-selected clusters yielding equivalent deltas (RQ5), and a clean pretraining-vs-SFT dissociation (RQ4: low base zero-shot, partial few-shot recovery, sharp SFT installation, collapse on removal). Anchored to the verified ~63% precedent and high deployed-disclaimer base rates, we expect tens of percentage points with the DiD CI excluding 0.

**Publishable regardless of direction — but only because we built the controls to make a null interpretable:**
- **If the behavior disappears (H1–H3, H5 confirmed):** the first reproducible, open demonstration that a *cheap, internals-free, human-auditable* selector validly identifies SFT clusters that *causally install* a safety-relevant pro-social behavior — re-runnable from released checkpoints. Directly actionable for safety auditing and data curation.
- **If the single cluster's removal is null but remove-all-enriched is large:** a rigorous **over-determination/redundancy** result (empirically validating Simfluence non-additivity), cleanly distinguished from "removed too little" *because we pre-registered the all-cluster arm.* This is the distinction the prior draft could not make.
- **If RQ5 shows the tracer underperforms probe/LESS:** an honest negative result on cheap-selector validity — itself a useful finding for the auditing community.
- **The RQ4 dissociation** is novel structural evidence no competitor reports, interesting whether the split is sharp or blurred.

---

## 9. Threats to Validity and Mitigations

| Threat | Mitigation |
|---|---|
| **Concurrent scoop (2602.11079v3) did causal OLMo-2 retraining first** | Cite as **prior work in para 1**; stake novelty only on the four surviving axes (SFT-stage, pro-social taxonomy, **lexical-auditable selector validated head-to-head**, pretraining-vs-SFT split); re-verify latest version before submission. Do **not** claim "first causal." |
| **Coherent-cluster removal is generically disruptive (the real confound)** | **Coherent-distractor arm is the headline DiD comparator with full seed power**; matched-random is the secondary volume check. |
| **Whole-conversation removal removes more than the behavior** | **Pre-registered conversation- vs. surgical-turn-level removal comparison**; use surgical removal for the headline if they diverge. |
| **Over-determination → attenuated/null single-cluster effect** | **Pre-registered remove-all-enriched-clusters arm** (primary, budgeted) so a null is attributable to distribution, not under-removal; base-floor measured first. |
| **Few-run statistics underpowered; GLMM seed variance unidentified** | **12-run pilot → power simulation → seed-count decision rule** (≥10 seeds on headline arms if needed); **permutation test on per-seed differences as primary inference**, GLMM as secondary; no CLT for the headline. |
| **RQ5 "indistinguishable" is an unpowered equivalence claim** | **TOST equivalence design with pre-registered margin and power**; report inconclusive if underpowered rather than implying equivalence. |
| **Circularity: selector lexicon = a metric channel** | **Lexicon used for selection ONLY; excluded from the primary outcome**, which is EPITOME/ESConv/JMIR + human-validated judge. |
| **RQ4 n-gram-count clause is near-tautological** | RQ4 recast as a **behavioral** dissociation (base zero-shot vs. few-shot vs. SFT emission); infini-gram counts are context only. |
| **Behavior measurement subjective / judge unreliable** | Triangulated non-lexical stack; published constructs; ensemble judge with swap-order; **human expert–expert α anchor** as primary defense (not the judge-family argument); human subset incl. OLMo outputs. |
| **Verbosity confound** | Report **both** length-adjusted and unadjusted; **unadjusted primary** (length partly *is* the behavior). |
| **Capability placebo could move for the same reason as the target** | Use **mechanistically orthogonal deterministic benchmarks** (MMLU-Pro, GSM8K, HumanEval+, IFEval); **drop AlpacaEval 2/IFEval-as-chat-quality from the placebo** (LLM-judged, style/length-biased) — report only as style check. |
| **Cluster step intractable / mis-scaled on CPU** | **DR (PCA→UMAP→ANN-HDBSCAN) benchmarked in M1**; k-means robustness; pre-registered enrichment threshold; pivot behavior if no small removable cluster exists. |
| **Per-run hours are an estimate (±50%)** | **Pilot-gated budget**; checkpoint/resume for spot preemption; quote verified 8×H100 number as headline, single-GPU as optimistic. |
| **LoRA confound** | Headline + RQ5 arms in **full FT**; LoRA only for interior cells/extra seeds, labeled. |
| **1B-only scale generalization** | Frame 1B as the **only fully-open organism** where this counterfactual is possible; optional one OLMo-2-7B confirmatory pair; AI2 itself draws recipe conclusions at 1B–8B. |
| **Pythia/The-Pile arm overstated as a second deployed-model result** | **Drop Pythia from the main proposal** (no official matched SFT pipeline → only a constructed-SFT analogue); mention only as future-work generalization to avoid diluting the "only fully-open organism" framing. |
| **To-verify citations are load-bearing** | JMIR 2026 disclaimer scale, Nat. Mach. Intell. 2506.10150 α figures, Miller 2411.00640, NIST eval-stats report, the 84/85% scoop figure — **all flagged to-verify; pinned or substituted before submission.** Verified items (open-instruct hours/hparams, mixture size 866,138/1.27 GB/19 sources, base checkpoint, EPITOME, ESConv, LESS, LoRA "learns less forgets less," GPU prices, 63% precedent) are labeled as such. |

---

## 10. Reviewer Concerns & How We Address Them

| # | Concern (reviewer) | Resolution in this revision | Fully fixed? |
|---|---|---|---|
| 1 | We falsely claimed the scoop lacks a random control, multi-behavior discovery, and dose-response | **Retracted** (§0, §1.2). Verified 2602.11079-**v3, OLMo 2 7B**; it has all three. Novelty re-staked on the four surviving axes. | **Yes** |
| 2 | Random arm doesn't identify the estimand; coherent-distractor does | **Distractor cluster promoted to headline DiD comparator with full (8) seed power** (§3.3, §4.1). | **Yes** |
| 3 | Over-determination likely → null, with no way to interpret it | **Remove-all-enriched-clusters arm pre-registered and budgeted** (§3.3, §4.1, §8). | **Yes** |
| 4 | Few-run power asserted not shown; GLMM seed variance unidentified | **12-run pilot → power simulation → seed decision rule; permutation test primary; GLMM secondary** (§5). | **Substantially** (true top-level n stays small; we report MDE honestly and add seeds if needed) |
| 5 | Residual circularity (lexicon selects and measures) | **Lexicon excluded from the primary outcome; selection-only** (§3.4). | **Yes** |
| 6 | RQ5 underpowered as a fail-to-reject equivalence claim, and unfunded | **TOST design, pre-registered margin/power; runs added to matrix and budget**; reported inconclusive if underpowered | **Substantially** (equivalence at small n remains hard; honest inconclusive is the floor) |
| 7 | RQ4 n-gram clause near-tautological | **Recast as behavioral dissociation (base zero/few-shot vs. SFT)** (§2 RQ4). | **Yes** |
| 8 | Cluster step under-derisked / CPU-intractable; granularity unspecified | **DR pipeline benchmarked in M1; granularity comparison pre-registered; enrichment threshold pre-registered; CPU-only preliminary** (§3.2, §7). | **Yes (de-risked); residual risk if no small removable cluster exists → pre-registered behavior pivot** |
| 9 | Budget conflates GPU-h with wall-clock; memory understated; single-GPU figure unverified | **Full memory footprint stated; GPU-h-vs-calendar-time clarified; verified 8×H100 figure is the headline, single-GPU is pilot-gated optimistic** (§6). | **Yes** |
| 10 | Stale scoop metadata (size/date); to-verify citations | **Corrected to v3 / 7B / 27 Apr 2026; all unverified citations flagged to-verify with substitution plan** (§1.2, §9). | **Yes** |
| 11 | Pythia arm overstated | **Dropped from main proposal; future-work only** (§9). | **Yes** |

**Honest residual limitations we cannot fully eliminate:** (a) the causal estimand lives at the training-run level and even 8–10 seeds is a small top-level n — we mitigate with permutation inference and pilot-gated power but cannot make it large; (b) RQ5 equivalence claims at this scale may end inconclusive, in which case we report that honestly rather than overclaiming; (c) results are at 1B (the only fully-open organism for this counterfactual), with at most one 7B confirmatory pair; (d) a handful of measurement citations are still to-verify and will be pinned or substituted before submission.

---

## 11. Timeline, Deliverables, Target Venue

**Timeline (~5 months):**
- **M1 — Tracing, clustering, CPU-only preliminary (free):** embed 866k turns, DR+HDBSCAN+k-means (benchmark tractability), tracer validation, **base-vs-released-SFT behavior comparison + base few-shot probe**, granularity decision, enrichment threshold, freeze IDs + eval set; pre-register. **Go/no-go + behavior-pivot decision gate.**
- **M2 — Pilot & power (12 GPU runs):** baseline×4 + targeted-100×4 + random-100×4; measure between-run SD; run power simulation; **lock seed allocation and final budget.**
- **M3 — Main matrix (39–54 GPU runs):** DiD w/ distractor + RQ5 selector head-to-head + dose + specificity + remove-all-enriched; capability benchmarks after each.
- **M4 — Measurement & validity:** non-lexical behavior scoring, human-IAA subset (incl. OLMo outputs), TOST selector comparison, optional unlearning proxy.
- **M5 — Analysis & writing:** permutation + GLMM + bootstrap, effect sizes, dose slopes; artifact packaging; submission.

**Deliverables (the credibility multiplier):**
1. **Counterfactual SFT checkpoints** (baseline, targeted, distractor, random, probe, LESS, dose arms) — reviewers can re-run the causal experiment.
2. **Advice-behavior probe set** (relationship/health/emotional-support + factual-QA controls) with the non-lexical behavior detectors — a benchmark.
3. **Two-layer provenance tracer** (infini-gram + DuckDB) + **frozen removed-ID sets and granularity specs** per dose.
4. Pre-registration + analysis/power-simulation code + judge prompts (pinned).

**Target venue.**
- **Primary: COLM.** Topical home (alignment/instruction-tuning; alignment-data analysis & curation); LM-native reviewers who won't penalize 1B scale; Tülu 3 published at COLM 2025; achievable bar (COLM 2024: 1,036 submitted / 289 accepted, **28.86%**). **Lead the abstract with the auditable causal selector (RQ5) and the pretraining-vs-SFT dissociation (RQ4)** — the two uniquely-ours contributions — not "first causal."
- **Alternative — NeurIPS Datasets & Benchmarks** if the released checkpoints + probe set + tracer become the headline (public repo + Croissant).
- **Alternative — ACL/EMNLP main** if framed NLP-behavioral with full human eval.
- **Fallbacks — TMLR** (rigor-over-novelty, no deadline; ideal given the contested-novelty landscape) or **SaTML/EMNLP-safety** if the crisis-referral/disclaimer safety frame becomes the spine.

**Decision rule:** auditable-causal-selector science → **COLM**; artifact-first → **NeurIPS D&B**; safety-spine → **SaTML/EMNLP**; contested-novelty-but-airtight-rigor → **TMLR**.
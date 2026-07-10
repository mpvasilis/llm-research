# Auditable by Construction: Can a Human-Inspectable Lexical/Corpus Provenance Selector *Causally Stand In* for an Opaque Activation Probe in Pro-Social SFT?

**Hook.** Concurrent work shows you can causally trace a production LLM's post-training behaviors by selecting data with an *opaque activation probe* (or gradients, or an LLM judge) and counterfactually re-training. None of those selectors is **human-auditable**: you cannot read off *which* conversations were removed or *why*. We test whether a **human-auditable lexical/corpus selector** — infini-gram verbatim spans plus a DuckDB substring/`list_filter` query over the actual SFT conversations, returning the exact removable conversation IDs — picks at least one behaviorally-defined cluster whose leave-cluster-out re-SFT moves a pro-social behavior (i) *more than* a coherent-distractor and a matched-random control, and (ii) by an amount *not detectably smaller than* what an activation probe, LESS, and an LLM-judge selector achieve, **under identical re-training**. If so, causal alignment-data attribution becomes inspectable and re-runnable from released checkpoints — the one property the probe-based pipeline structurally cannot offer.

**Abstract (~165 words).** Recent work establishes that production-LLM post-training behaviors can be causally attributed by selecting datapoints with an activation probe (and with gradient and LLM-judge baselines) and counterfactually re-training (Xiao & Aranguri, *Probe-Based Data Attribution*, arXiv 2602.11079v3, OLMo 2). That pipeline works, but every one of its selectors is opaque. We ask whether a **human-auditable** selector — built only from verbatim corpus matches over the released SFT conversations, no model internals — can serve as a *valid causal stand-in*. On the fully-open OLMo 2 1B pipeline, under identical leave-cluster-out re-SFT, we test whether an auditable lexical/corpus tracer selects a cluster that (a) beats a coherent-distractor and a matched-random control and (b) is **non-inferior** to probe-, LESS-, and LLM-judge-selected clusters, with selection-set overlap (per-conversation, high-powered) as a co-primary outcome. We co-headline a **pretraining-vs-SFT behavioral dissociation** via the real pretraining corpus (infini-gram) — an axis the prior work has no analogue for. We target pro-social, safety-*promoting* behaviors. Budget: ~72 H100-hrs/run; pilot-gated.

---

## 0. What This Paper Is (and Is Not) — Read This First

**This is not a "first causal account."** Counterfactual re-training to causally attribute OLMo 2 post-training behaviors already exists: **Xiao & Aranguri, "Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training"** (arXiv **2602.11079, v3, 27 Apr 2026**; v1 11 Feb, v2 13 Feb, v3 27 Apr 2026). The title **"In-the-Wild Model Organisms"** was the **superseded v2 title** — we cite the *current v3 title and authors* and freeze a dated quote of v3's title/model set/method in Appendix A. That work does counterfactual re-training on OLMo 2 production **preference (DPO) data** — while **retraining from the SFT checkpoint** — using **three** selectors (an **activation probe**, **LESS** gradients, and an **LLM-judge** toxicity baseline), with a **random-removal baseline**, **unsupervised behavior clustering**, a **discrete dose-response** (3k/12k/30k datapoints), and a published **per-selector selection-cost comparison** (~$30 probe / ~$320 gradient / ~$500 LLM-judge). We treat all of this as **prior work and cite it in the first paragraph of the camera-ready.**

**The one question the probe/gradient/judge pipeline structurally cannot answer:** *is a selector you can fully read and audit — built only from verbatim corpus matches, no internals, no gradients, no judge — a valid causal stand-in for those opaque selectors?* Validity here is operationalized two ways, both pre-registered: (i) **selection-set agreement** (does the auditable tracer pick the same conversations the opaque selectors pick?) and (ii) **behavioral non-inferiority** (under identical re-SFT, does the tracer's cluster move behavior by an amount not detectably smaller than the opaque selectors', while all beat distractor/random?). The payoff if true is concrete and uncontested: **causal data attribution that a human can inspect, row by row, and re-run from released checkpoints.** That auditability is the gap we own.

**Where we corrected the record (these were wrong or oversold in our prior draft):**
- **SFT-vs-DPO is NOT a clean stage difference.** v3 states verbatim that it modifies the DPO/preference dataset and *"retrain[s] from the SFT checkpoint,"* and it explicitly analyzes the SFT(M0)→DPO(M1) activation delta. So the prior work *already* retrains from SFT. Our true, narrower difference: **we ablate SFT instruction-conversation *training data*; they ablate *preference (DPO) data* while also retraining from SFT.** This is demoted to a one-line clarification, not a pillar.
- **"Cheap/$0/internals-free" is contested and demoted.** The prior work already publishes a selection-cost comparison. We therefore **do not lead with cost.** We lead with **human-auditability**, which their probe/gradient/judge selectors do not have, and we keep cost only as a secondary, measured line item.
- **Model size is to-verify.** The v3 *abstract* states only "OLMo 2 production DPO training" with no parameter count; "7B/32B" appears in our notes but is **not confirmed from the v3 PDF body**. We flag it **to-verify** and confirm against the v3 body before submission (Appendix A). Our substrate is 1B regardless.
- **The "84/85%" note is resolved.** 84% is real (a filter-by-source result removing four problematic generator models); there is **no 85%**. The hedge is removed.

**Two co-headline contributions that are genuinely uncontested:**
1. **RQ5 — auditable-selector validity** (selection-set agreement + behavioral non-inferiority vs. probe/LESS/judge). The probe-based pipeline cannot offer an auditable selector.
2. **RQ4 — a true pretraining-vs-SFT *behavioral* dissociation** via the real pretraining corpus (infini-gram). The prior work has **no pretraining-corpus layer**; this is the cleanest non-contested novelty.

**Scope honesty (existence claim, not universal claim).** Because M1 includes a pre-registered behavior pivot and pre-registered (not post-hoc) thresholds, the title and claims are scoped to an **existence** result: *there exists at least one pro-social, safety-relevant behavior for which an auditable selector is a valid causal stand-in.* We do not claim auditable selectors are valid for arbitrary behaviors.

**What adversarial review changed.** Two expert reviews landed at borderline-6/10, no fatal flaws, but flagged (a) several mis-stated "verified" facts about the prior work (title, model size, SFT/DPO, its cost argument, its third baseline), (b) that the headline inference (two-sided TOST equivalence at ~8 run-level seeds) is the experiment we are *least* likely to win, and (c) that the genuinely uncontested novelty (pretraining-vs-SFT dissociation, and per-conversation selection overlap) was buried. This version corrects every fact, reframes the headline inference as **directional non-inferiority + high-powered per-conversation selection overlap**, and elevates RQ4. §10 tracks every concern, including the ones we cannot fully resolve.

---

## 1. Contribution and Novelty vs. the Prior Work

### 1.1 The headline claims (two co-headlines)

**Co-headline A — Auditable selector validity (RQ5).** A behaviorally-defined SFT cluster selected by a **human-auditable lexical/corpus selector** is *causally responsible* for a specific free-form advice behavior, **and the auditable selector is a valid causal stand-in for an opaque activation probe, LESS, and an LLM-judge selector** — in two falsifiable senses under identical leave-cluster-out re-SFT:

- **(Selection agreement, high-powered, per-conversation):** the tracer-selected conversation-ID set overlaps the probe-, LESS-, and judge-selected sets above a pre-registered Jaccard / rank-correlation threshold. This is an **n-in-the-thousands** outcome, not run-level, so it is well-powered regardless of seed count.
- **(Behavioral non-inferiority, run-level):** removing the tracer cluster reduces the behavior (a) beyond an equally-coherent, equal-size **distractor** cluster (primary confound control) and beyond matched-random (volume control), and (b) by an amount **not detectably smaller** than probe/LESS/judge (one-sided non-inferiority within a pre-registered margin), with the effect monotone in dose, specific, and capability-neutral.

**Co-headline B — Pretraining-vs-SFT behavioral dissociation (RQ4).** Using the real pretraining corpus via infini-gram, we show the target behavior is *installed at SFT* rather than latent in pretraining (low base zero-shot, partial few-shot recovery, sharp SFT installation, collapse on cluster removal). The prior work has no pretraining-corpus analogue.

The novel object is **the auditable selector, validated head-to-head, plus the pretraining-side dissociation** — *not* "leave-out re-training is ground truth," which the datamodels/TRAK/influence literature already established (Datamodels, Ilyas et al. ICML 2022; TRAK, Park et al. ICML 2023 — the Linear Datamodeling Score *is* the correlation between predicted and actually-retrained outputs).

### 1.2 The single closest concurrent work, stated accurately

**Xiao & Aranguri, "Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training"** (arXiv **2602.11079, v3, 27 Apr 2026**; OLMo 2 production post-training; **parameter count to-verify from v3 body**). It does counterfactual re-training by **modifying the DPO/preference dataset and retraining from the SFT checkpoint**, with **three** selectors — **activation probe, LESS, and an LLM-judge (toxicity) baseline** — plus a **random-removal baseline**, **unsupervised multi-behavior clustering**, a **discrete dose-response** (3k/12k/30k datapoints; e.g., remove 30k random → 6.67% harmful vs. 7.63% baseline vs. 2.86% for their method; ~63%/78% reductions; **84%** via filter-by-source removing four problematic generator models), and an explicit **per-selector cost comparison** (~$30 probe / ~$320 gradient / ~$500 judge). It does **not** offer a **human-auditable / inspectable** selector, a pretraining-corpus layer, or a pro-social-behavior target. **We cite it up front and differentiate only on those axes.**

### 1.3 Differentiation table (auditability- and pretraining-centric)

| Work | Actual retrain? | Data ablated / retrain base | Behavior | Selector **human-auditable**? | Validated **head-to-head**? | **Pretraining-corpus** layer? | Coherent-distractor control? | Fully-open |
|---|---|---|---|---|---|---|---|---|
| Datamodels / TRAK | surrogate | — | loss/acc | n/a | no | no | no | partial |
| Influence (Koh–Liang→Grosse→TrackStar) | no (infinitesimal) | pretrain/loss | token loss | no (gradients) | no | no | no | partial |
| LESS (ICML 2024) | no (select-to-add) | SFT | capability | no (gradients) | no | no | no | partial |
| **Probe-Based Data Attribution (2602.11079v3)** | **yes** | **DPO data / retrain from SFT ckpt** | undesirable | **no (probe / LESS / LLM-judge)** | yes (probe vs. LESS vs. judge vs. random) | **no** | no (random only) | yes (OLMo 2) |
| **This proposal** | **yes** | **SFT instruction data / from base** | **pro-social/safety** | **yes (lexical/corpus, inspectable IDs)** | **yes (tracer vs. probe vs. LESS vs. judge vs. distractor/random)** | **yes (infini-gram)** | **yes** | **yes** |

The correlational pretraining-side predecessor is **"Generalization v.s. Memorization"** (Wang et al., ICLR 2025, arXiv 2407.14985), tying *capabilities* to *pretraining* n-gram frequency over Pile/Dolma. We extend distributional-memorization tracing **observational → interventional** and reframe the question from "which data correlates" to "**is the most auditable selector causally valid.**"

---

## 2. Research Questions and Falsifiable Hypotheses

**RQ5 and RQ4 are co-headlines.** RQ1–RQ3 establish that a causal effect exists and is properly controlled; **RQ5 establishes the auditable selector is a valid stand-in for the opaque ones**; **RQ4 establishes where the behavior is installed** (the uncontested pretraining-side novelty).

**RQ5 (selector validity — CO-HEADLINE).** Under identical leave-cluster-out re-SFT, does the human-auditable lexical/corpus tracer select a cluster that is both *the same data* and *as causally potent* as what an activation probe, LESS, and an LLM-judge select?
- **H5a (selection agreement — PRIMARY, high-powered).** Tracer-selected conversation IDs overlap probe-, LESS-, and judge-selected IDs above a pre-registered Jaccard / Spearman-rank threshold. Per-conversation, n in the thousands; **this carries RQ5 even if run-level inference is underpowered.**
- **H5b (behavioral non-inferiority — run-level).** The tracer cluster's behavior delta is **not detectably smaller** than probe/LESS/judge (one-sided non-inferiority within a pre-registered margin), and all four beat distractor and matched-random.
- *Falsified* if the tracer selects materially different data (low overlap) **and** yields a materially smaller behavior delta. If run-level non-inferiority is underpowered, RQ5 rests on H5a (overlap) + the directional H5b result (tracer beats distractor/random), and the two-sided **equivalence** test is reported as **secondary/exploratory, inconclusive if unpowered** — never sold as the lead.

**RQ4 (pretraining-vs-SFT dissociation — CO-HEADLINE, behavioral).** Is the behavior *installed at SFT* rather than latent in pretraining?
- **H4.** Zero-shot base OLMo-2-0425-1B emits the behavior at a low rate; **few-shot prompting partially recovers it** (latent-capability signal); full-mixture SFT raises it sharply; targeted cluster removal collapses it. We report **(i) base zero-shot, (ii) base few-shot, (iii) post-SFT** emission and interpret the gaps. *Falsified* if base zero-shot ≈ post-SFT (not SFT-installed) **or** if few-shot fully recovers it (fully latent). Infini-gram verbatim counts are *context only*; they do **not** carry the dissociation claim.

**RQ1 (causality vs. the right control).** Does removing the tracer-selected cluster reduce the behavior beyond removing an equally-coherent distractor cluster (and matched-random)?
- **H1.** ΔΔ = (drop under targeted-100%) − (drop under **distractor**-100%) > 0, 95% CI excluding 0. *Falsified* if targeted removal does not beat coherent-distractor removal. (Secondary: targeted vs. matched-random.)

**RQ2 (dose-response).** Is the targeted effect monotone and steeper than distractor/random?
- **H2.** Behavior rate monotone-decreasing over {0,25,50,100}% targeted removal with a steeper fitted slope than both controls. *Falsified* by a flat or non-monotone targeted curve.

**RQ3 (specificity & placebo).**
- **H3a.** Removing the disclaimer cluster reduces disclaimers more than empathy, and vice versa (behavior × ablated-cluster interaction).
- **H3b.** Deterministic capability benchmarks (MMLU-Pro, GSM8K, MATH, BBH, HumanEval+, IFEval) do not move beyond the matched-random arm. *Falsified* if targeted removal degrades capabilities as much as random does (→ volume shock, not attribution). (AlpacaEval 2 is dropped from the placebo set — LLM-judged, length-biased; reported only as a style check.)

---

## 3. Method: The Selector Head-to-Head via Counterfactual SFT

### 3.1 Substrate (verified, fully open)
- **Base:** `allenai/OLMo-2-0425-1B` (1B; ~4T-token stage-1 OLMo-mix-1124 + ~50B-token stage-2 Dolmino-mix-1124; Apache 2.0). **We start from BASE**, not the released -Instruct checkpoint.
- **SFT mixture (ablation universe):** `allenai/tulu-3-sft-olmo-2-mixture-0225` — **866,138** conversations, parquet, ~1.27 GB, 19 source subsets. **Schema: `{id, messages:[{role,content}], source}`** — the **`id` column is load-bearing**: it is what lets every selector return exact, inspectable, removable conversation IDs and what makes the removed-set auditable.
- **Recipe:** `allenai/open-instruct`, OLMo-2 SFT config (`docs/olmo2.md`): base → SFT, **max_seq_len 4096, per_device_bs 2, grad_accum 8, LR 3e-5, linear schedule, warmup_ratio 0.03, weight_decay 0.0, 2 epochs, DeepSpeed ZeRO-3 (stage3_no_offloading), bf16, `--add_bos`, `--seed` as a first-class CLI flag.** Maintainer-stated wall-clock: **~9 h on 1×8×H100 node = ~72 H100-GPU-hours/run** (verified verbatim in `docs/olmo2.md`). We **ablate at SFT only** for the primary claim; an optional DPO-robustness arm checks the delta survives preference tuning.

### 3.2 The four selectors (the comparison is the experiment)

All four select a removable cluster/set of SFT conversations for the *same* behavior; the entire point is that the re-SFT protocol downstream is **identical** so the only thing that varies is *how the set was chosen*.

1. **Auditable lexical/corpus tracer (ours — the candidate).** Human-inspectable, no internals:
   - **Pretraining layer:** infini-gram verbatim span counts (context for RQ4).
   - **Instruction layer:** **DuckDB role-scoped substring / `list_filter`** over the 866k conversations, returning the **exact removable `id` set**. Anyone can read the lexicon/rule, run the query, and inspect every removed row.
2. **Activation probe (prior-work incumbent — opaque).** Linear probe on hidden activations; rank/select by probe score. Requires model internals.
3. **LESS (gradients — established).** Gradient-similarity datapoint selection (Xia et al., ICML 2024). Requires gradient infrastructure.
4. **LLM-judge selector (prior-work's third baseline — opaque, costly).** Score conversations with an LLM judge for the target behavior and select top-ranked. We **add this arm explicitly** so our head-to-head is not visibly narrower than the prior work's baseline set.

*Honest caveat (stated in abstract and table):* **"auditable / internals-free" applies to *deployment* of the tracer only.** The one-time validation against probe/LESS/judge *does* use internals, gradients, and judge calls — on our side, once. The claim is precisely that, once validated, the **deployed selector is human-auditable and internals-free**, which the opaque selectors never are.

### 3.3 Cluster definition (CPU, free) — de-risked

The causal claim rests on this step, so we de-risk it explicitly. A shared candidate universe underlies all selectors (the tracer enriches over it; probe/LESS/judge rank within it):

1. **Embed** all ~866k assistant turns with a strong sentence encoder (E5 / GTE / Instructor). To make HDBSCAN on 866k high-dim points CPU-tractable, **reduce dimensionality first** (PCA→UMAP to ~50D) and use ANN-accelerated HDBSCAN — the documented practical path, benchmarked in M1, not assumed.
2. **Cluster** with **HDBSCAN** (primary; density-based, flags noise) and **k-means** (silhouette-chosen k) as a robustness check; the causal result must hold under both.
3. **Ablation granularity is a pre-registered, tested decision.** Because empathy/disclaimer language can be one sentence in an otherwise-diverse conversation, **whole-conversation removal could itself be the volume/diversity confound.** We pre-register a **conversation-level vs. turn/sentence-level surgical-removal comparison** as a primary methodological result: if surgical removal yields the same delta as whole-conversation removal, the confound is ruled out; otherwise we report the difference and use surgical removal for the headline.
4. **Validate + label via the auditable tracer.** A cluster qualifies as "empathy"/"disclaimer"/"crisis-referral" only if **enriched** versus base rate; we pre-register the **enrichment-ratio threshold** (not post-hoc) rather than leaving "enriched" vague.
5. **Freeze removed-`id` sets (per selector), ablation granularity, and the eval-prompt set BEFORE training any ablated model.** Pre-register doses {0,25,50,100}%.

### 3.4 Controls: distractor (headline) + matched-random (volume) + remove-all-enriched (over-determination)
- **Coherent-distractor arm (headline DiD comparator):** an unrelated but **equally coherent, equal-size** semantic cluster. Rules out "removing any tight cluster is generically disruptive" — which matched-random cannot address. **Full seed power, equal to the targeted arm.**
- **Size-matched random arm (volume check):** same conversation count, total token mass, and per-source skill distribution. Answers "is it just data volume?" Secondary.
- **Remove-all-enriched-clusters arm (primary — makes a null interpretable):** remove *every* cluster above the enrichment threshold. If the single-cluster effect is null but the all-cluster effect is large, the behavior is **over-determined/distributed** (Simfluence non-additivity), said rigorously rather than confounding "null" with "removed too little." Budgeted.
- **Estimand = difference-in-differences:** (targeted drop) − (distractor drop). Match quality reported for all arms.

### 3.5 Generation + measurement (CPU + cheap API)
- **Generate** OLMo-2-1B answers on the **free CPU workstation** (or cheap spot GPU): frozen probe set of relationship/health/emotional-support prompts (breakup, snoring, anxiety, panic) + factual-QA controls. Primary at **temp=0 greedy**; temp>0 for robustness; **≥3 generation seeds** averaged.
- **Behavior instruments — the primary outcome uses NON-LEXICAL instruments only (circularity fix).** The lexicon is used **strictly for selection** and **excluded from the primary behavior metric** (otherwise selector and metric share signal and the effect is partly tautological). Primary outcome triangulates:
  1. **Published classifiers/rubrics:** empathy/validation via **EPITOME** (ER/IP/EX × levels 0/1/2; Sharma et al., EMNLP 2020); structure via **ESConv** 3-stage (Exploration/Comforting/Action) / 8-strategy taxonomy (Liu et al., ACL 2021); safety via the **JMIR 5-level disclaimer/referral ordinal scale** (verified real: *JMIR 2026;1:e84668*, "no disclaimer" → "urgent advice to consult a professional").
  2. **Ensemble LLM-judge** (`gpt-4o-2024-08-06` + Claude + Gemini), swap-order double-evaluation, symmetric formatting, decomposed sub-criteria, required rationale. Self-preference is structurally reduced (OLMo 2 in no commercial judge's family), but we rely on the **human-IAA anchor**, not the family argument, as the primary defense.
- **Judge reliability** benchmarked against **human expert–expert agreement on the same rubric**. Precedent: empathy-judge Krippendorff α **0.51–0.75 (median ~0.6)** vs. expert–expert agreement, from **"When Large Language Models are Reliable for Judging Empathic Communication"** (arXiv 2506.10150 — cited as the **arXiv preprint**; the previously asserted *Nature Machine Intelligence* venue label is **dropped** as unverified). We benchmark on a **stratified human-annotated subset (~200–400 items, ≥3 annotators, codebook + adjudication) including OLMo generations.** Report Krippendorff α, Fleiss κ, judge-vs-human ICC.
- **Verbosity:** because empathy/disclaimers are partly *expressed through* length, we **report both length-adjusted and unadjusted effects, unadjusted primary** (regressing out length could regress out the effect).

### 3.6 Auditability is a *measured construct*, not a row count
Reviewers correctly noted that "number of inspectable removed rows" does not distinguish the tracer from a probe (a probe also yields IDs). We therefore operationalize auditability as a **measured human-predictability construct**, pre-registered:
- **Rule legibility / predictability:** given only the tracer's stated rule (lexicon + DuckDB query), can independent human auditors **predict whether a held-out conversation is selected**? We report inter-auditor agreement and auditor-vs-tracer agreement (κ/α) on a held-out sample.
- **Justification completeness:** fraction of selected rows for which the rule yields a **human-readable reason** (the matched span), vs. probe/judge scores which yield a number with no inspectable justification.
- **Reproducibility-from-spec:** can a third party regenerate the exact removed-`id` set from the published rule alone (binary, per selector)? The tracer should score 1; probe/LESS require the model/gradients; the judge requires the (versioned, costly, stochastic) judge.

These three measures, not raw row counts, are the auditability evidence.

### 3.7 CPU-vs-GPU split (explicit)
| Step | Hardware | Cost |
|---|---|---|
| Embedding, DR, clustering, **tracer selection**, ID/granularity selection, **auditability study** | **CPU (existing workstation)** | **free** |
| **Probe + LESS + LLM-judge selection (one-time, to enable the head-to-head)** | GPU / API (internals/gradients/judge) | small, one-time |
| Generating 1B answers (all conditions × gen-seeds) | **CPU (or cheap spot GPU)** | **free / negligible** |
| Behavior classifiers (EPITOME/ESConv) | **CPU** | **free** |
| LLM-judge API for *measurement* (~50–150k calls) | API | tens of $ |
| **Leave-cluster-out re-SFT (the dominant GPU cost)** | **GPU** | **see §6** |
| Capability benchmarks (inference) | CPU/cheap GPU | negligible |

---

## 4. Experiments: Conditions × Seeds, Metrics, Baselines

### 4.1 Run matrix
Shared seeds across conditions so **seed is a blocking factor** (paired analysis). Seed counts are **pilot-gated** (§5): the matrix is the *target*; if the pilot shows large seed variance we reallocate toward more seeds on the headline arms (the four selectors + distractor) and fewer interior doses.

| Cell | Removal | Dose | Seeds | Runs |
|---|---|---|---|---|
| C0 baseline | none | 0% | 8 | 8 |
| **RQ5: tracer-selected (ours, headline)** | tracer cluster | 100% | **8** | **8** |
| **RQ5: probe-selected** | probe cluster | 100% | **8** | **8** |
| **RQ5: LESS-selected** | LESS cluster | 100% | **8** | **8** |
| **RQ5: LLM-judge-selected** | judge cluster | 100% | **5** | **5** |
| **Distractor-100 (headline control)** | coherent unrelated cluster | 100% | **8** | **8** |
| Random-100 (volume check) | matched random | 100% | 5 | 5 |
| Targeted-50 (tracer) | tracer cluster | 50% | 3 | 3 |
| Targeted-25 (tracer) | tracer cluster | 25% | 3 | 3 |
| Distractor-50 | distractor | 50% | 3 | 3 |
| Second-behavior leave-out (H3a) | empathy cluster | 100% | 3 | 3 |
| Remove-all-enriched (over-determination) | all enriched clusters | 100% | 3 | 3 |
| **Total** | | | | **65** |

- **Lean tier-1 (minimum publishable, selector head-to-head spine):** baseline(8) + **tracer-100(8) + probe-100(8) + LESS-100(8) + judge-100(5)** + distractor-100(8) + random-100(5) = **50 runs**. This is the RQ5 head-to-head with the right controls plus the full opaque-selector baseline set — the headline stands on these alone. Dose/specificity/over-determination arms are added if budget allows.
- **Pilot (variance estimate, budget-gating):** baseline(4) + tracer-100(4) + **random-100(4)** = **12 runs**. 2 seeds/arm cannot estimate a variance, and the DiD needs the **control-arm** variance, so the pilot includes it. The pilot reports empirical between-run behavior-rate SD and feeds the §5 power simulation that gates the full matrix.
- Optional robustness (budget permitting): one **DPO** arm on ablated SFT checkpoints (+~16 GPU-h/run); one larger-scale (e.g., OLMo-2-7B) confirmatory tracer/probe pair for scale.

### 4.2 Metrics
- **Selector-validity (RQ5, primary):** **(i) selection-set agreement** — Jaccard and Spearman-rank correlation of tracer-selected IDs vs. probe/LESS/judge (per-conversation, high-powered, the inference that lands even if run-level is not); **(ii) behavioral non-inferiority** — one-sided non-inferiority of tracer delta vs. probe/LESS/judge within a pre-registered margin; each selector's delta vs. distractor/random; **(iii) auditability construct** (§3.6: predictability, justification completeness, reproducibility-from-spec) and, secondarily, selection cost.
- **Behavior:** per-behavior rate/score from the **non-lexical** stack (EPITOME/ESConv/JMIR-ordinal + ensemble judge), with convergent (instrument–instrument) and criterion (vs. human gold) validity.
- **Causal effect:** DiD in behavior rate (targeted − distractor) with hierarchical bootstrap 95% CI; standardized effect (**Cohen's h** for proportions, **GLMM log-odds** with CI); **dose slope** for targeted vs. controls and the test of their difference.
- **Provenance overlap:** infini-gram verbatim counts (pretraining context) and SFT substring enrichment ratio within the tracer cluster vs. base rate (the auditable link).
- **Capability-regression:** MMLU-Pro, GSM8K, MATH, BBH, HumanEval+/BigCodeBench, DROP, IFEval after every retrain.

### 4.3 Baselines (reconciled with the prior work's full set)
1. **Probe selector** — the opaque incumbent. **Full seed power.**
2. **LESS** — gradient selector. **Full seed power.**
3. **LLM-judge selector** — the prior work's third baseline, **added here** so our comparison is not narrower than theirs.
4. **Coherent-distractor cluster** — the headline DiD comparator (rules out "any tight cluster is disruptive").
5. **Matched-random** — the volume check.
6. **Correlational tracer** (observational use of the same tool) — what the causal validation upgrades.
7. **(Optional) Unlearning proxy:** machine-unlearning the tracer cluster (GUDA / Wang et al. NeurIPS 2024) to show selector+re-SFT agrees with selector+unlearning.

---

## 5. Statistical Analysis Plan (Few-Run Causal Inference, Honestly)

The headline difficulty, stated plainly: **the run-level causal estimand has a small top-level n even at 8 seeds/arm.** We do not paper over this — and we deliberately put the **high-powered per-conversation selection-overlap outcome (H5a)** at the center so RQ5 does not live or die on run-level power.

- **RQ5's primary inference is two-pronged, and the well-powered prong leads.** **(a) Selection-set agreement (H5a)** is per-conversation (n in the thousands) and is the **primary, well-powered** RQ5 result. **(b) Behavioral non-inferiority (H5b)** is the run-level prong, tested **one-sided** (more winnable at small n than two-sided equivalence). Two-sided **TOST equivalence** is reported only as **secondary/exploratory**, and **inconclusive** if unpowered — never the lead.
- **Power is demonstrated, not asserted (pilot-gated).** From the 12-run pilot we estimate empirical between-run behavior-rate SD, then **simulate the GLMM DiD and the one-sided non-inferiority test at n=5 and n=8 seeds** and report the **minimum detectable ΔΔ**. The full matrix is committed only if the MDE clears the pre-registered minimum effect of interest (**≥15–20 absolute pp**, anchored to the verified ~63% precedent reduction) at ≥80% power. If not, we reallocate to **≥10 seeds on the headline selector arms + distractor** and drop interior doses. This decision rule is pre-registered.
- **Primary model — item-level GLMM:** `behavior ~ selector * dose + (1|seed) + (1|eval_prompt)`, exploiting many eval prompts for power. **Honest caveat:** with 5–8 seeds the `(1|seed)` variance component is weakly identified; we *also* report the conservative fallback and treat agreement between them as the robustness bar.
- **Conservative fallback — fixed-effects-per-seed + permutation test.** Treat each seed as a fixed effect, compute per-seed selector-vs-distractor and selector-vs-selector differences, and use a **permutation/randomization test** over seed-level differences for the run-level headline p-values. Avoids estimating an unidentified variance component; valid at small n. **Primary run-level inference.**
- **Headline contrast:** paired per-prompt DiD; **hierarchical/cluster bootstrap resampling seeds AND eval prompts** (≥1,000 resamples) corroborates the permutation test. No CLT/normal approximation for the run-level headline.
- **Per-behavior:** McNemar + Cohen's h (binary); paired Cohen's d (ordinal); Holm correction across behaviors.
- **Pre-registration:** lexicon (selection only), rubrics, judge prompts (pinned versions), human codebook, removed-`id` sets per selector, ablation granularity, enrichment threshold, eval-prompt set, decoding params, MEI, the **Jaccard/rank-overlap threshold and the non-inferiority margin**, the **auditability-construct measures**, and the power-gated seed-allocation decision rule.

---

## 6. Compute Budget

**GPU is needed for SFT re-training (dominant) plus one-time probe/LESS selection; the judge selector adds one-time API cost.** Tracing, embedding, clustering, 1B generation, classifiers, and the auditability study run free on the CPU workstation; LLM-judge *measurement* calls are tens of dollars.

**Per-run anchor (verified):** OLMo-2-1B full SFT = **~9 h on 8×H100 = ~72 H100-GPU-hours/run** (`open-instruct/docs/olmo2.md`, verified verbatim; LR 3e-5 / 2 epochs / bs2×ga8 / seqlen 4096 / ZeRO-3 stage3_no_offloading / bf16). H100 SXM ≈ **$2.50–2.99/GPU-hr** on-demand (Spheron/RunPod/Lambda, mid-2026); A100-80GB ≈ **$1.07 on-demand / $0.60 spot**.

**Memory footprint, stated in full:** AdamW full-FT of 1B in bf16 ≈ model 2 GB + grads 2 GB + fp32 optimizer states ~12 GB + master weights, i.e. ~16–20 GB of *states* and **~30–48 GB total with activations** — **fits comfortably on one A100-80GB**, not on a 24 GB card except with grad-checkpointing + 8-bit optim.

**GPU-hours vs. wall-clock:** 72 GPU-h is fixed work; single-GPU saves money (no idle GPUs in a node) but costs **calendar time (~72 h/run)**, which matters for sequential runs on one GPU in a 5-month timeline. We budget **multi-GPU for throughput where the timeline needs it**, single-GPU where it doesn't.

| Plan | Runs | GPU-h | $ (H100 on-demand) | Notes |
|---|---|---|---|---|
| **Full matrix (65 runs)** | 65 | 4,680 | **$11,700–13,993** | 4-selector head-to-head + dose + specificity + over-determination |
| **Recommended (50 runs, head-to-head spine)** | 50 | 3,600 | **$9,000–10,764** | tracer vs. probe vs. LESS vs. judge vs. distractor/random |
| **Pilot (12 runs)** | 12 | 864 | **$2,160–2,585** | Variance + wall-clock; budget-gating |

**Cost-reduction levers (design-preserving):**
- **Single-GPU full FT on A100-80GB.** At ~60–72 GPU-h/run, the 50-run spine ≈ **$1,800–3,600 (spot–on-demand A100)** — the practical headline floor.
- **Subsample** the mixture to a fixed 30–50% held constant across **all** arms → ~2–3× faster/run, fair because every arm sees the same base.
- **LoRA only for interior-dose cells / extra seeds** (1B LoRA ~2 GB; QLoRA ~0.6 GB). **Headline selector arms (tracer/probe/LESS/judge) + distractor run in FULL fine-tuning** — LoRA "learns less, forgets less" (arXiv 2405.09673) could attenuate the delta and diverge from the official recipe, contaminating the head-to-head.
- **Frugal floor:** subsample + LoRA on community 4090 / A100-spot → **~$150–400** for a reduced but valid study.

**Headline budgets to quote (honest range):** lead with the **verified-anchor** figure — **~$9.0–10.8k** for the 50-run head-to-head spine on 8×H100 (defensible, grounded). The **optimistic single-GPU/subsampled** figure (**~$1.8–3.6k**) is explicitly **pilot-gated** and flagged ±50% (single-GPU per-run wall-clock is extrapolated from the multi-GPU anchor, not measured on this stack). **CPU-only MVP / fallback:** correlational tracer + base-vs-released-SFT behavior comparison + the CPU-only preliminary (§7) + the measurement and auditability stacks — yields the RQ4 dissociation, the selection-overlap evidence, and the auditability construct as an observational paper, with counterfactual re-SFT added once a GPU is rented.

> **Per-run hours carry ±50% uncertainty** and the budget is **pilot-gated**: the 12-run pilot fixes the real single-GPU wall-clock before the full matrix is committed.

---

## 7. CPU-Only Preliminary: De-Risk Before Any GPU Spend

Before renting a single GPU, we run a **CPU-only preliminary on the already-released checkpoints** that converts several "expected" claims into shown ones and gates the program:

1. **Cluster + selector feasibility (M1 deliverable):** the actual size, coherence (silhouette/density), and enrichment ratio of the empathy/disclaimer/crisis clusters on the real 866k mixture, and whether at least one behavior has a *small, content-matched, removable* cluster. **Crucially for RQ5 (this is H5a, computable with zero re-training):** measure the **selection-set overlap (Jaccard / rank correlation) between the tracer-selected IDs and the probe/LESS/judge-selected IDs on the released checkpoints.** High overlap strongly de-risks the head-to-head; divergence localizes *where* the auditable and opaque selectors disagree — itself a publishable finding. **This is the single highest-leverage de-risking measurement and it needs no GPU re-training.**
2. **Base-floor + selector enrichment, no retraining needed:** compare **base OLMo-2-0425-1B vs. the released OLMo-2 SFT checkpoint** on the frozen probe set. If the released SFT model emits the behavior far above base **and** the tracer cluster is strongly enriched, the causal program is de-risked on CPU. If base already emits near-SFT rates, we pivot to a behavior with a larger SFT-installed increment (e.g., crisis-referral / structured ESConv strategies) **during M1, not post-hoc.**
3. **RQ4 base few-shot probe:** measure base zero-shot vs. few-shot emission to pre-classify each candidate behavior as "latent-elicitable" vs. "SFT-acquired" before spending GPU on it.
4. **Auditability construct dry-run (§3.6):** run the human-predictability / justification-completeness / reproducibility-from-spec study on the released-checkpoint selections — all CPU/human, no GPU.

This preliminary is the single highest-value de-risking step and is **free.**

---

## 8. Expected Headline Result — and Why It's Interesting Either Way

**Expected:** the **auditable tracer's selected IDs overlap the probe/LESS/judge selections above threshold (H5a)**; under identical re-SFT, the tracer cluster's behavior delta is **non-inferior** to the opaque selectors' (H5b) and all four beat coherent-distractor and matched-random, with a large, specific, dose-monotone targeted effect; and the **pretraining-vs-SFT dissociation (RQ4)** is clean (low base zero-shot, partial few-shot recovery, sharp SFT installation, collapse on removal). Anchored to the verified ~63% precedent and high deployed-disclaimer base rates, we expect tens of percentage points with the DiD CI excluding 0.

**Publishable regardless of direction — because the controls and the high-powered overlap outcome make a null interpretable:**
- **If the auditable selector is validated (H5a/H5b, H1–H3 confirmed):** the first reproducible, open demonstration that a **human-auditable** selector is a **valid causal stand-in** for opaque probe/LESS/judge selectors for installing a safety-relevant pro-social behavior — re-runnable from released checkpoints. Directly actionable for auditable safety-data curation; the contribution the probe-based pipeline cannot offer.
- **If the tracer's selections diverge from the opaque ones (low H5a) or it underperforms behaviorally (H5b fails):** an honest, **high-powered** negative result that **precisely maps the gap** between auditable and opaque selection — useful to the auditing community even with small run-level n, because H5a is per-conversation.
- **If single-cluster removal is null but remove-all-enriched is large:** a rigorous **over-determination/redundancy** result (empirically validating Simfluence non-additivity), cleanly distinguished from "removed too little" *because we pre-registered the all-cluster arm.*
- **The RQ4 dissociation** is novel structural evidence the prior work has no analogue for — interesting whether the split is sharp or blurred, and it does not depend on the head-to-head landing.

---

## 9. Threats to Validity and Mitigations

| Threat | Mitigation |
|---|---|
| **Concurrent work (2602.11079v3) did causal OLMo-2 retraining first** | Cite as **prior work, current v3 title/authors (Xiao & Aranguri), para 1**; do **not** claim "first causal." Stake the contribution on **auditable-selector validity (RQ5)** + **pretraining-vs-SFT dissociation (RQ4)** + pro-social taxonomy. Re-verify v3 and **freeze a dated quote (title/model set/"retrain from SFT checkpoint") in Appendix A**. |
| **Scoop mis-cited (title/model/SFT-vs-DPO/cost/3rd baseline)** | Corrected: v3 title = *Probe-Based Data Attribution*; "In-the-Wild Model Organisms" was v2; they **retrain from the SFT checkpoint** while ablating DPO data; they publish a **cost comparison**; they have a **third (LLM-judge) baseline**. Model size **to-verify from v3 body**. SFT/DPO demoted to one-liner. |
| **"Internals-free / cheap" is contested** | Lead with **human-AUDITABILITY** (operationalized as a measured construct, §3.6), not cost. State explicitly that "internals-free/auditable" applies to **deployment**; the one-time validation uses internals/gradients/judge. Cost kept only as a secondary measured line. |
| **Auditability metric (row count) doesn't beat a probe** | Replaced with a **measured construct**: human predictability of selections, justification completeness, reproducibility-from-spec (§3.6). |
| **RQ5 headline (two-sided equivalence at n≈8) likely "inconclusive"** | **Primary RQ5 inference is per-conversation selection overlap (H5a, n in thousands) + one-sided non-inferiority (H5b)**; two-sided TOST equivalence demoted to secondary/exploratory, reported inconclusive if unpowered. |
| **Coherent-cluster removal is generically disruptive (the real confound)** | **Coherent-distractor arm is the run-level DiD comparator with full seed power**; matched-random is the secondary volume check. |
| **Whole-conversation removal removes more than the behavior** | **Pre-registered conversation- vs. surgical-turn-level removal comparison**; surgical for the headline if they diverge. |
| **Over-determination → attenuated/null single-cluster effect** | **Pre-registered remove-all-enriched-clusters arm** (primary, budgeted); base-floor measured first. |
| **Few-run statistics underpowered; GLMM seed variance unidentified** | **12-run pilot → power simulation → seed-count decision rule** (≥10 seeds on headline arms if needed); **permutation test on per-seed differences as primary run-level inference**, GLMM secondary; no CLT. The well-powered H5a does not depend on this. |
| **Circularity: selector lexicon = a metric channel** | **Lexicon used for selection ONLY; excluded from the primary outcome** (EPITOME/ESConv/JMIR + human-validated judge). |
| **RQ4 n-gram-count clause near-tautological** | RQ4 is a **behavioral** dissociation (base zero/few-shot vs. SFT emission); infini-gram counts context only. |
| **Behavior measurement subjective / judge unreliable** | Triangulated non-lexical stack; published constructs; ensemble judge with swap-order; **human expert–expert α anchor** as primary defense (not the judge-family argument); human subset incl. OLMo outputs; 2506.10150 cited as **arXiv preprint** (no NMI label). |
| **Verbosity confound** | Report **both** length-adjusted and unadjusted; **unadjusted primary** (length partly *is* the behavior). |
| **Capability placebo could move for the same reason as the target** | **Mechanistically orthogonal deterministic benchmarks** (MMLU-Pro, GSM8K, HumanEval+, IFEval); **drop AlpacaEval 2** from the placebo (LLM-judged, style-biased) — style check only. |
| **Cluster step intractable / mis-scaled on CPU** | **DR (PCA→UMAP→ANN-HDBSCAN) benchmarked in M1**; k-means robustness; pre-registered enrichment threshold; pivot behavior if no small removable cluster exists. |
| **M1 pivot + pre-registered thresholds condition on a favorable substrate** | **Title/abstract scoped to an EXISTENCE claim** ("there exists a behavior for which the auditable selector is valid"), not a universal one. Pivot rules and thresholds pre-registered before any retraining. |
| **Per-run hours are an estimate (±50%)** | **Pilot-gated budget**; checkpoint/resume for spot preemption; quote verified 8×H100 number as headline, single-GPU as optimistic. |
| **LoRA confound contaminates the head-to-head** | Headline selector arms + distractor in **full FT**; LoRA only for interior cells/extra seeds, labeled. |
| **1B-only scale generalization** | Frame 1B as the **only fully-open organism** where this counterfactual is possible; optional one larger-scale (e.g., 7B) confirmatory pair; AI2 itself draws recipe conclusions at 1B–8B. |
| **Pythia/The-Pile arm overstated as a second deployed model** | **Dropped from the main proposal** (no official matched SFT pipeline) → future-work only. |
| **To-verify items load-bearing** | **v3 model size** (confirm from PDF body), **2506.10150 venue dropped to arXiv**. Verified items (open-instruct hours/hparams, mixture 866,138/1.27 GB/19 sources/**`{id,messages,source}`** schema, base checkpoint, EPITOME, ESConv, JMIR 5-level scale, LESS, LoRA "learns less forgets less," GPU prices, 63/78/84% precedent figures, scoop's "retrain from SFT checkpoint," third LLM-judge baseline, cost comparison) labeled as such. |

---

## 10. Reviewer Concerns & How We Address Them

| # | Concern (reviewer) | Resolution in this revision | Fully fixed? |
|---|---|---|---|
| 1 | Wrong scoop **title** (cited the superseded v2 "In-the-Wild Model Organisms") | **Corrected to v3** *Probe-Based Data Attribution* (Xiao & Aranguri), with v1/v2/v3 dates; dated quote frozen in Appendix A. | **Yes** |
| 2 | Scoop **model size** asserted as verified (7B/32B) | **Demoted to to-verify**: v3 abstract gives no size; confirm from PDF body before submission. Substrate is 1B regardless. | **Yes (flagged)** |
| 3 | **SFT-vs-DPO** differentiator false as stated (they retrain from SFT) | **Corrected and demoted to a one-liner**: they ablate DPO data while retraining from the SFT checkpoint; our narrow difference is ablating SFT instruction *training data*. | **Yes** |
| 4 | **Cheap/$0/internals-free** contested (they publish a cost comparison) | **Demoted**; **human-auditability** is now the sole headline axis, operationalized as a measured construct (§3.6). | **Yes** |
| 5 | Omitted the scoop's **third (LLM-judge) baseline** | **Added an LLM-judge selector arm**; updated §1.2, §3.2, §4.3, table, matrix. | **Yes** |
| 6 | "84/85%" hedge | **Resolved**: 84% real (filter-by-source); no 85%; hedge removed. | **Yes** |
| 7 | **2506.10150** wrongly labeled Nature Machine Intelligence | **Venue label dropped**; cited as the arXiv preprint; α 0.51–0.75 retained. | **Yes** |
| 8 | RQ5 headline = two-sided equivalence at n≈8 → most-likely "inconclusive" | **Reframed**: primary RQ5 = **per-conversation selection overlap (H5a, high-powered)** + **one-sided non-inferiority (H5b)**; two-sided equivalence demoted to exploratory. | **Yes** |
| 9 | Promote **selection-set overlap** to a primary RQ5 outcome | **Done** (H5a primary; computed CPU-only on released checkpoints in M1). | **Yes** |
| 10 | Elevate the **pretraining-vs-SFT dissociation** (uncontested novelty) | **Promoted to co-headline (RQ4)** alongside RQ5. | **Yes** |
| 11 | **Auditability** operationalized as row count (doesn't beat a probe) | **Replaced with measured construct** (predictability, justification completeness, reproducibility-from-spec, §3.6). | **Yes** |
| 12 | Title/abstract overclaim ("is a valid selector") given M1 pivot | **Scoped to an existence claim** in title, abstract, §0. | **Yes** |
| 13 | Schema omits the load-bearing **`id`** column | **Schema corrected to `{id, messages, source}`**; `id` flagged as what enables exact removable IDs (§3.1). | **Yes** |
| 14 | Few-run power asserted not shown; GLMM seed variance unidentified | **12-run pilot → power sim → seed rule; permutation test primary run-level inference**; H5a does not depend on run-level power. | **Substantially** (true top-level n stays small; H5a carries RQ5) |
| 15 | Residual circularity (lexicon selects and measures) | **Lexicon excluded from the primary outcome; selection-only** (§3.5). | **Yes** |
| 16 | Budget conflates GPU-h with wall-clock; memory understated | **Full memory footprint stated; GPU-h-vs-calendar-time clarified; verified 8×H100 headline, single-GPU pilot-gated** (§6). | **Yes** |
| 17 | Pythia arm overstated | **Dropped from main proposal; future-work only.** | **Yes** |

**Honest residual limitations we cannot fully eliminate:** (a) the **run-level** causal estimand has a small top-level n even at 8–10 seeds — mitigated by making the well-powered per-conversation overlap (H5a) the primary RQ5 outcome and using permutation inference for the run-level prong, but not made large; (b) two-sided equivalence at this scale may be inconclusive — reported honestly, and not the lead; (c) "internals-free/auditable" is a **deployment** virtue — the one-time validation uses internals/gradients/judge; (d) results are at 1B (the only fully-open organism for this counterfactual), with at most one larger-scale confirmatory pair; (e) the claim is an **existence** result conditioned on M1's pre-registered behavior selection, not a universal one; (f) the prior work's exact model size and a measurement citation remain to-verify and will be pinned before submission.

---

## 11. Timeline, Deliverables, Target Venue

**Timeline (~5 months):**
- **M1 — Tracing, clustering, selection overlap, CPU-only preliminary (free):** embed 866k turns, DR+HDBSCAN+k-means (benchmark tractability), tracer validation, **selection-set overlap (H5a) of tracer vs. probe/LESS/judge on released checkpoints**, base-vs-released-SFT behavior comparison + base few-shot probe (RQ4), **auditability-construct study** (§3.6), granularity decision, enrichment threshold, freeze `id` sets + eval set; pre-register. **Go/no-go + behavior-pivot decision gate.**
- **M2 — Pilot & power (12 GPU runs):** baseline×4 + tracer-100×4 + random-100×4; measure between-run SD; run power simulation; **lock seed allocation and final budget.**
- **M3 — Main matrix (50–65 GPU runs):** **RQ5 selector head-to-head (tracer/probe/LESS/judge)** + distractor + random + dose + specificity + remove-all-enriched; capability benchmarks after each.
- **M4 — Measurement & validity:** non-lexical behavior scoring, human-IAA subset (incl. OLMo outputs), **non-inferiority + overlap analysis**, optional unlearning proxy.
- **M5 — Analysis & writing:** permutation + GLMM + bootstrap, effect sizes, dose slopes, selection-overlap and auditability-construct accounting; artifact packaging; submission.

**Deliverables (the credibility multiplier):**
1. **Counterfactual SFT checkpoints** (baseline, tracer, probe, LESS, judge, distractor, random, dose arms) — reviewers can re-run the **selector head-to-head**.
2. **Auditable two-layer provenance tracer** (infini-gram + DuckDB) + **frozen removed-`id` sets per selector and granularity specs per dose** — the inspectable selector itself, with every removed conversation readable, plus the **auditability-construct study materials**.
3. **Advice-behavior probe set** (relationship/health/emotional-support + factual-QA controls) with the non-lexical behavior detectors — a benchmark.
4. Pre-registration + analysis/power-simulation/non-inferiority/overlap code + judge prompts (pinned) + **Appendix A dated prior-work quote**.

**Target venue.**
- **Primary: COLM.** Topical home (alignment-data analysis & curation, instruction tuning); LM-native reviewers who won't penalize 1B scale; Tülu 3 published at COLM 2025; achievable bar (COLM 2024: 1,036 submitted / 289 accepted, **28.86%**). **Lead the abstract with the two co-headlines — auditable causal selector validity (RQ5) and the pretraining-vs-SFT dissociation (RQ4)** — the uniquely-ours contributions the probe-based pipeline cannot offer.
- **Alternative — NeurIPS Datasets & Benchmarks** if the released checkpoints + probe set + auditable tracer become the headline artifact (public repo + Croissant).
- **Alternative — ACL/EMNLP main** if framed NLP-behavioral with full human eval.
- **Fallbacks — TMLR** (rigor-over-novelty, no deadline; ideal given the contested-novelty landscape) or **SaTML/EMNLP-safety** if the auditable-safety-curation frame becomes the spine.

**Decision rule:** auditable-selector validation + pretraining dissociation science → **COLM**; artifact-first → **NeurIPS D&B**; safety-spine → **SaTML/EMNLP**; contested-novelty-but-airtight-rigor → **TMLR**.

---

## Appendix A — Frozen Prior-Work Record (to re-verify immediately before submission)

- **Citation:** Xiao & Aranguri, *Probe-Based Data Attribution: Discovering and Mitigating Undesirable Behaviors in LLM Post-Training*, arXiv **2602.11079**. Versions: **v1 11 Feb 2026, v2 13 Feb 2026 (titled "In-the-Wild Model Organisms"), v3 27 Apr 2026 (current, retitled).**
- **To freeze verbatim from v3 PDF before submission:** (i) the current title; (ii) the model/parameter set (**size to-verify** — v3 abstract states only "OLMo 2 production DPO training"); (iii) the sentence stating it modifies the DPO/preference data and **"retrain[s] from the SFT checkpoint"**; (iv) the three selector baselines (probe, LESS, LLM-judge); (v) the random baseline, dose levels (3k/12k/30k), the 63/78/84% figures, and the per-selector cost comparison (~$30/$320/$500).
- **Why frozen:** the credibility of §0–§1 depends on describing the prior work exactly; any post-correction drift in their text must be caught at submission time.
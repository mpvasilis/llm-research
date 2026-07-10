# Research Agenda: Answer-Provenance Tracing for LLM Advice

*Two-layer provenance (pretraining + instruction-tuning) for socially high-stakes free-form answers, on a single CPU workstation*

---

## 1. Executive Summary

We have a working tool that decomposes any LLM answer into two provenance layers: a **pretraining layer** (longest verbatim n-gram spans queried against the model's *actual* pretraining corpus — Dolma v1.7 for OLMo 2, The Pile for Pythia — via the free infini-gram API, returning exact corpus counts and source documents) and an **instruction-tuning layer** (substring search over the *actual* SFT mixtures — Tulu 3, oasst1 — read locally from parquet via DuckDB, returning full matching conversations). The published state of the art (OLMoTrace, ACL 2025; infini-gram, COLM 2024) traces only to **pretraining**; the SFT/instruction-tuning provenance layer this tool adds is **genuinely under-served in the literature** across every surveyed subfield. The opportunity is to be the first to jointly attribute a single everyday answer across both training stages — and to do it in the **neglected, socially relevant domain of emotional-support / health / relationship advice** — under hard single-workstation, CPU-only, free-API constraints. This agenda turns the owner's motivating question ("how close are LLM responses to the instruction fine-tune dataset, and to pretraining?") into a portfolio of runnable, presentable studies, with one flagship worked example and a one-week pilot that yields a real result immediately.

---

## 2. What the Tool Measures → Research Levers

The tool's five capabilities and its menu of metrics map cleanly onto distinct research questions. The table below reads as "if you want to ask X, here is the lever and the number it produces."

| Capability / metric | What it physically measures | Research lever it unlocks |
|---|---|---|
| **Pretraining attribution** (greedy longest-match, 4–16 word spans, exact Dolma/Pile counts) | Verbatim overlap of an answer with the true pretraining corpus; per-span corpus frequency | *How memorized vs. novel is an answer?* Operationalizes n-gram novelty (Merrill et al. 2024) and Creativity Index (Lu et al. 2024) against the **true** corpus, not a generic web index |
| **Span corpus count (rarity)** | How common/distinctive each matched phrasing is in pretraining | *Is a span "common/templated" or "distinctive"?* A cheap proxy for the counterfactual common-vs-specific distinction (Zhang et al. 2021); ties to term-frequency effects (Razeghi 2022) |
| **Source metadata** (`path`: cc_en_head/middle/tail, reddit, falcon, c4, books) | Which web tier/domain a phrasing comes from | *Forum vs. clinician?* Source-domain provenance of advice — where on the open web the phrasings live (WIMBD / Data Portraits tradition) |
| **Instruction-tuning attribution** (ILIKE substring over Tulu 3 + oasst1, full conversations) | Verbatim presence of phrasing in actual SFT conversations + match counts | *How close is a response to the SFT data?* The owner's headline question, and the field's clearest gap (no public tool traces answers to SFT *conversations*) |
| **Cross-layer ratio** (pretraining spans vs. instruction matches) | Whether phrasing leans pretraining-ubiquitous or SFT-specific | *Memorized-generic vs. learned-behavior* decomposition; an empirical handle on the Superficial Alignment Hypothesis (LIMA, Zhou et al. 2023) |
| **Matched-span coverage / verbatim-overlap ratio / novelty rate** | % of answer words inside any matched span (0–1) | *The "template tax"* — a single reportable "how close to the data" number the field currently lacks |
| **Local generation** (OLMo-2-1B, Pythia, Claude-proxy) | Reproducible answers from true-provenance open models | *Scale-controlled studies* on the Pythia 70M→1.4B ladder; *true vs. proxy* contrast for closed models |
| **History & export** (JSON/MD/CSV) | Persisted, auditable per-answer records | *Reproducible datasets* of answer→provenance distances; replication bundles |

The single most important conceptual lever: the tool converts an abstract debate ("is alignment superficial?", "do LLMs parrot forum advice?") into **measured, per-answer, two-layer overlap quantities with named source documents** — correlational, honestly scoped, but concrete.

---

## 3. Literature Landscape & the Gap We Fill

Six surveyed areas converge on the same structural picture.

**(a) Verbatim memorization & extraction.** The canon — Carlini et al. *Extracting Training Data* (2021), *Quantifying Memorization* (ICLR 2023, the three log-linear scaling laws), Lee et al. *Deduplicating Training Data* (ACL 2022), Kandpal et al. (ICML 2022, superlinear duplication→emission) — established that verbatim memorization grows with scale, duplication, and context length. Nasr et al. (2023) extended extraction to production models via the divergence attack. Crucially, **Aerni et al. *Non-Adversarial Reproduction* (ICLR 2025)** showed up to ~15% of benign chatbot output overlaps web text — but measured only against a *generic* web index, not the model's actual corpus.

**(b) Search/index infrastructure & output tracing.** Infini-gram (Liu et al., COLM 2024) is the suffix-array engine this tool queries; **OLMoTrace (Liu et al., ACL 2025)** is its closest published sibling, tracing full outputs to multi-trillion-token training data in real time (reporting matched spans cover ~5% of output tokens) — but to **pretraining only**. WIMBD (Elazar et al., ICLR 2024), Rusty-DAWG (Merrill et al., EMNLP 2024), Data Portraits (Marone & Van Durme, NeurIPS 2023), and infini-gram mini (EMNLP 2025, contamination bulletin) round out the toolkit — all pretraining-focused.

**(c) Training-data attribution (TDA).** Gradient/influence methods (Koh & Liang 2017; TracIn, Pruthi 2020; TRAK, Park et al. 2023; Grosse et al. EK-FAC 2023) estimate causal influence but need GPUs and many retrains. A recurring, decisive empirical finding (Akyürek et al. 2022; **Chang et al. TrackStar 2024**; DATE-LM 2025) is that **cheap lexical/BM25 overlap often rivals or beats expensive influence functions for finding examples that contain a fact** — which legitimizes this tool's surface-search approach as a strong baseline, while honestly flagging that overlap ≠ causal influence.

**(d) Instruction-tuning data influence & the alignment debate.** LIMA (Zhou et al. 2023) / the Superficial Alignment Hypothesis vs. its rebuttals (Raghavendra et al. 2024; **Ghosh et al. *A Closer Look at the Limitations of Instruction Tuning*, ICML 2024**, which reports ~81% of shifted tokens and ~87% of hallucinated phrases are borrowed from SFT data). Tulu 3 (Lambert et al. 2024) releases the actual ~939k-conversation SFT mixture this tool searches. LESS (Xia et al. 2024) selects influential SFT data by gradient — but no public work traces an *answer* back to specific SFT *conversations* as provenance.

**(e) Behavior origins (sycophancy, refusal, persona, style).** Sharma et al. (sycophancy, 2023), Arditi et al. (refusal as a single direction, NeurIPS 2024), Persona Vectors (Anthropic 2025), Emergent Misalignment (Betley et al., Nature 2025). These localize *where* behaviors live, but rarely tie *empathic/advice-giving style* to specific inspectable training data.

**(f) Novelty/originality.** RAVEN (McCoy et al., TACL 2023), Rusty-DAWG novelty (Merrill et al. 2024), Creativity Index (Lu et al. 2024), and the recent critique **Saakyan et al. *Death of the Novel(ty)* (2026)** that ~91% of top-quartile n-gram-novel expressions are *not* judged creative.

**The gap, stated precisely:**

1. **SFT/instruction-tuning provenance is essentially unmeasured.** Every search/index tool above targets pretraining. Tracing an aligned answer to specific SFT conversations is the tool's clearest novel contribution.
2. **No pipeline jointly decomposes a single everyday answer across both layers** (pretraining-verbatim *and* SFT-conversation).
3. **The advice/emotional-support/health domain is unstudied** for verbatim/SFT provenance — yet it is where "is this real advice or regurgitated boilerplate?" matters most.
4. **Non-adversarial reproduction has only been measured against generic web indices**, never the model's *true* corpus — which OLMo 2 (Dolma) and Pythia (Pile) make possible on a laptop.

This agenda targets gaps 1–4 directly, with honest correlational framing (overlap/membership, not causation) matching OLMoTrace's own stated stance.

---

## 4. The Research Portfolio

### 4.1 Ranked table of all studies

Scores are mean of novelty/feasibility/presentability on a 1–5 scale (higher = better). **Bold** rows are fully developed below.

| # | Title (short) | One-liner | Score | Format |
|---|---|---|---|---|
| 1 | **Templated Empathy** | Decompose advice answers into behavioral-move spans; measure what fraction of each move is recoverable from *assistant turns* of the SFT mixture vs. pretraining vs. novel, contrasting support vs. factual-QA | **3.67** | Workshop |
| 2 | **The Disclaimer Reflex** | Trace exact disclaimer/crisis-referral phrasings back to named Dolma source documents (reddit/cc_en_head/falcon) and specific Tulu-3/oasst1 conversations | **3.67** | Workshop |
| 3 | **Tracing the Medical Disclaimer** | Per-disclaimer two-layer ownership verdict (pretraining vs. SFT vs. composed), with a hero figure: breakup (no disclaimer) vs. chest-pain (disclaimer) | **3.33** | Workshop + blog |
| 4 | **Forum or Clinician?** | Source-domain provenance of advice phrasings (reddit/falcon vs. cc_en_head/C4), with frequency-matched, base-rate-normalized control | **3.33** | Responsible-AI workshop |
| 5 | **The SFT Template Tax** | Corpus-size-normalized per-document prevalence of empathic/safety phrasing in SFT vs. pretraining, at matched phrase length | **3.33** | Safety workshop + demo |
| 6 | **Copy-vs-Compose Curves** | Artifact-corrected verbatim coverage vs. scale across the Pythia 70M–1.4B ladder on identical Pile data | **3.00** | Workshop |
| 7 | Memorization by Topic | Verbatim-provenance profile of sensitive advice vs. factual QA in open-data models | 3.00 | Workshop |
| 8 | Proxy or Provenance? | Calibrate Dolma-as-proxy Claude traces against ground-truth OLMo-2 traces on a shared corpus | 3.00 | Workshop |
| 9 | The Greedy-Span Heuristic on Trial | Does longest-match attribution agree with itself under paraphrase; does it inflate memorization? | 3.00 | Methods workshop |
| 10 | Proxy vs Truth | Bound how misleading closed-model provenance claims are, using open models as ground truth | 3.00 | Trustworthy-ML workshop + demo |
| 11 | The Alignment Tax on Memorization | Does instruction-tuning suppress or preserve verbatim pretraining reproduction? (OLMo-2 base vs. instruct) | 2.67 | Paper |
| 12 | Open vs Closed Mirror | Do Claude and OLMo-2 answer the same question in the same words; what can the proxy legitimately tell us? | 2.67 | Transparency workshop + talk |
| 13 | Auditing the Greedy-Longest-Match Heuristic | Sensitivity of provenance conclusions to attribution choices | 2.67 | Methods / reproducibility |

The top five all sit in the **advice-domain + SFT-provenance** sweet spot — the agenda's center of gravity. Studies 6–13 are scale-control, proxy-calibration, and methods-robustness work that strengthen the flagship line.

---

### 4.2 Developed proposal 1 — *Templated Empathy*

**A Role-Scoped, Rarity-Calibrated Provenance Audit of Emotional-Support Answers Against the SFT Mixture and Pretraining.** *(Score 3.67 — top-ranked)*

- **Research question.** For small open-data instruction-tuned LLMs (OLMo-2-1B; Pythia for a scale check), what fraction of an emotional-support answer's **behavioral structure** (empathy openers, validation, hedges, numbered-step scaffolding, "consult a professional" disclaimers) is verbatim-recoverable from **assistant turns** of specific SFT conversations (Tulu 3, oasst1) vs. from pretraining (Dolma) vs. novel composition — and how does that three-way split differ from factual-QA answers and from topical content spans within the same answers?

- **Hypotheses.**
  - **H1 (behavioral vs. topical):** Within the same answer, behavioral-move spans are recoverable from assistant SFT turns at a markedly higher rate than length- and frequency-matched topical spans (paired within-answer, so model quirks cancel).
  - **H2 (domain contrast):** Support answers have higher behavioral-span SFT-recoverability than factual-QA, concentrated in empathy/validation/disclaimer classes; **structure and hedge classes will *not* separate the domains** (a pre-registered, falsifiable null).
  - **H3 (role-scoping matters):** Counting only assistant turns materially lowers SFT matches vs. whole-conversation blob matching (pre-verified: "First," 7,338 blob → 2,238 assistant-only on Tulu shard 0).
  - **H4 (rarity calibration):** Naive present/absent collapses to "both"; only after Dolma-frequency banding does a distinctive SFT-leaning class emerge.
  - **H5 (robustness):** The behavioral > topical gap is stable across temperatures and a second generator.

- **Method.** Primary generator **OLMo-2-0425-1B-Instruct** (CPU, temp 0.7/0.3); robustness with a second small model (Qwen2.5-0.5B / Llama-3.2-1B) traced against OLMo's corpora to test corpus-specificity. **Pre-register and git-commit** a ~60-phrase, variant-aware behavioral lexicon in 5 move classes *before* generation. 50 support + 50 factual prompts, frozen. Pretraining attribution via infini-gram (`v4_dolma-v1_7_llama`). **Key method upgrade:** extend `parquet_search` with role-scoped (assistant-only), **parameterized** DuckDB queries (native `messages` STRUCT `list_filter` for Tulu, `role='assistant' AND lang='en'` for oasst1) — apostrophe-safe, unlike f-string ILIKE. Scan **all 6 Tulu shards** (~2 GB) for the headline; otherwise report SFT recoverability only as the relative contrast.

- **Experiments.** E1 behavioral-vs-topical within-answer (Wilcoxon signed-rank); E2 domain × move-class (Mann-Whitney + Holm, incl. the predicted nulls); E3 role-scoping ablation (blob/assistant inflation factor); E4 rarity-calibration collapse; E5 manual precision audit of 100 stratified SFT hits; E6 temperature + second-generator robustness.

- **Metrics.** Behavioral-span coverage; SFT-assistant recoverability rate; behavioral-minus-topical gap (effect size + bootstrap CI); three-way rarity-calibrated label distribution; role-scoping inflation factor; cross-layer ratio; frequency-normalized SFT match rate; manual precision per move class.

- **Baselines.** Within-answer topical spans (primary control); factual-QA domain baseline; blob vs. role-scoped; naive vs. rarity-banded tag; random length-matched phrases; held-out alternative lexicon.

- **Expected findings.** Behavioral spans are recoverable from SFT-assistant turns at substantially higher rates than the model's own frequency-matched topical spans; support answers are more templated *specifically* in empathy/validation/disclaimer, while numbered-step/hedge scaffolding is equally template-y in both (null recovered). Two methods results give the paper teeth: role-scoping removes up to ~70% of generic-phrase blob matches; the naive 3-way split collapses to "both" until frequency-banded.

- **Threats.** Substring ≠ causal provenance (lead with the *differential* within-answer and cross-domain contrasts); lexicon is the instrument (pre-register + held-out lexicon + anchor on behavioral-vs-topical which holds for any emitted span); context-blind ILIKE (role-scoping + precision audit); 1/6-shard bias (scan all 6 or report relative only); single weak generator (two temps + second model, scope to "small open instruction models").

- **Venue / effort.** L2M2 (LLM Memorization, ACL) / BlackboxNLP / a data-attribution workshop; Findings if extended. **3–4 weeks, one researcher, CPU-only.** Extra: ~2 GB disk for full Tulu; HF token (**rotate the previously-exposed token first**, per project memory).

---

### 4.3 Developed proposal 2 — *The Disclaimer Reflex*

**Source-Document Provenance of Safety Boilerplate in LLM Health and Mental-Health Advice.** *(Score 3.67 — tied top)*

- **Research question.** Where does the safety/disclaimer reflex come from at the level of surface phrasing: which disclaimer/crisis-referral phrases recur, how rare are they in Dolma, which **corpus source domains** (reddit, cc_en_head/middle/tail, falcon, c4, books) supply them, and which specific full Tulu-3/oasst1 conversations contain the same phrasing? What fraction is verbatim-recoverable from pretraining vs. SFT vs. neither?

- **Hypotheses.** H1 templating (small, highly repeated set of disclaimer templates); H2 rarity graded inversely with specificity (generic 10⁴–10⁵ Dolma hits; crisis-specific 10²–10³ or absent); H3 distinctive **source-domain signature** (cc_en_head health sites + reddit advice subs) vs. body prose; H4 SFT inventory enumerable **only with the full Tulu scan**; H5 disclaimer tokens more present-in-both-layers than body prose; H6 weak-model caveat (988/hotline phrasing rare or absent from a 1B model — a falsifiable near-null).

- **Method.** Generate OLMo-2-1B answers across 4 strata (~40 prompts): physical-health self-care, mental-health/emotional, relationship, and crisis-adjacent (research artifacts only, never deployed). Query the **full disclaimer phrase directly** via `infinigram.count` + `passages_for(maxnum=10)` (bypassing greedy match, which would fragment a fixed phrase at the first OOV word). **New ~30-line `bucket_source()` helper** maps `path`/`metadata` to domain buckets incl. reddit subreddit granularity. Un-cap Tulu (`_SHARD_CAP=None`) and drive `parquet_search` with the **direct phrase** (not the rarest-term picker, which has degraded to typos like "withotu"). Cluster SFT hits into template families; label reflexive vs. topic-prompted.

- **Experiments.** E0 dependent-variable pilot (gating); E1 disclaimer rarity spectrum (live anchors: "I am not a doctor" ≈102,609; "consult a healthcare professional" ≈25,171; "consult with a mental health professional" ≈2,604); E2 source-domain signature vs. body prose; E3 full-vs-1/6-Tulu count comparison; E4 SFT template inventory + reflexive split; E5 two-layer novelty decomposition; E6 (optional) Pythia + Claude-proxy contrast.

- **Metrics.** Disclaimer/crisis emission rate; phrasing repetition / template concentration; per-phrase Dolma rarity; **source-domain distribution** (headline) incl. an explicit `unknown` bucket; source concentration vs. body; full-Tulu and oasst1 SFT counts + 1/6-vs-full ratio; template family count; cross-layer decomposition; seed-stability of templates.

- **Baselines.** Within-answer body-text provenance (key control); topic-matched non-safety stock phrases; frequency-matched random n-grams; 1/6-vs-full sampling baseline; optional Pythia/Claude references.

- **Expected findings.** A small, highly repeated set of disclaimer templates; rarity inversely graded by safety-specificity; a distinctive cc_en_head + reddit source signature; full-Tulu counts several-fold higher than 1/6 with some 0→nonzero flips; disclaimer spans markedly less novel than body prose. **Most safety-relevant likely-null:** the 1B model rarely emits crisis-referral (988/hotline) phrasing — its "safety reflex" is generic-disclaimer-shaped, not crisis-referral-shaped — a reportable finding, contrasted by the Claude-proxy arm.

- **Threats.** Causality overreach (frame as co-occurrence/provenance-candidate); weak dependent variable (E0 gates the study); SFT undercount (un-cap to full); generic-phrase noise (rarity is the explicit axis + body baseline); source-metadata gaps (report `unknown` fraction); incremental novelty (contribution is the *instantiation* — naming specific docs/conversations in an unstudied high-stakes domain); ethics (crisis prompts as artifacts, ethics statement, no deployed advice).

- **Venue / effort.** TrustNLP / BlackboxNLP / **CLPsych** (clinical-psychology NLP fits the domain); Findings if extended. **3–5 weeks, CPU-only;** extra: ~2.1 GB full-Tulu download, optional Anthropic key.

---

### 4.4 Developed proposal 3 — *Tracing the Medical Disclaimer*

**Two-Layer Provenance and Templating of LLM Safety Boilerplate; which stage "owns" the phrasing.** *(Score 3.33)*

- **Research question.** When an instruction-tuned open-data model appends a safety disclaimer, is that phrasing (a) verbatim in pretraining (Dolma), (b) traceable to specific SFT conversations (Tulu-3/oasst1), or (c) model-composed and present in *neither*? Which stage carries the higher, more *distinctive* signal, and how does that vary by disclaimer type (medical-referral / crisis-line / liability / relationship-support)?

- **Hypotheses.** H1 anchor-fragment ownership is SFT-side and discriminative (pilot: "988" = 752 in just 1/6 of Tulu); H2 **full-sentence double-null** (the complete disclaimer sentence is verbatim in *neither* layer — composition from fragments); H3 disclaimer-type gradient (crisis tokens SFT-owned, generic referral pretraining-saturated); H4 domain-conditioned emission (reliable under medical/crisis, rare under pure relationship framing); H5 cross-model recurrence in Claude (proxy-flagged, recurrence-only).

- **Method.** OLMo-2-1B at **max_new_tokens=400** (pilot-validated so trailing disclaimers aren't truncated), 3 seeds × 40 prompts across 4 strata. Regex-extract disclaimer sentences → **canonicalize to 8–12 anchor fragments** (the mandatory fix: full-sentence ILIKE returns ~0; anchors return real hits). Per anchor *and* full sentence: `infinigram.count` + `longest_matches` + `passages_for`; `parquet_search` at 1/6 **and** cap-lifted (all 6 shards). Ownership verdict per disclaimer unit. **Pythia-1.4b base as negative control** (expected ~0% emission, isolating disclaimers as a post-training behavior).

- **Experiments.** E1 emission rate by stratum (de-risks the study; pilot 4/5 health-crisis); E2 two-layer per-anchor table; E3 full-sentence double-null vs anchor-present; E4 source localization; E5 shard-cap sensitivity (~5–7× inflation); E6 (optional) Claude proxy recurrence.

- **Metrics.** Emission rate; anchor pretraining count; longest verbatim span; anchor SFT count (1/6 and full); full-sentence verbatim recall (~0, the double-null metric); ownership verdict distribution by type; source histogram; cross-model anchor Jaccard; shard-cap inflation factor.

- **Baselines.** Pythia base negative control; full-sentence vs anchor ILIKE (internal ablation); random non-disclaimer sentences from the same answers; pipeline `_rarest_term` vs direct-anchor; hand-collected real-world disclaimer reference set.

- **Expected findings.** A clean three-way gradient: generic anchors are pretraining-saturated commercial-web boilerplate (Dolma 100k–380k, cc_en_head/c4); crisis tokens are SFT-owned safety injections (988 = 752 in 1/6 Tulu); **full sentences are verbatim in neither** (composed from fragments). The nuanced, partly-falsified-as-stated outcome is more defensible than an all-SFT hypothesis.

- **Threats.** Verbatim ≠ causal; substring can't prove a negative (report 0-hits as "verbatim-absent," not "not SFT-taught"; negative control supports SFT-origin); 1/6 bias (cap-lift for quantitative claims); small-model external validity (Pythia + Claude bracket the result); closed-model proxy trap (recurrence-only).

- **Venue / effort.** BlackboxNLP / TrustNLP + a **non-ML blog post** ("Where does an AI's medical disclaimer actually come from?"). **3–4 weeks, CPU-only;** ~2.2 GB disk, optional Anthropic key.

---

### 4.5 Developed proposal 4 — *Forum or Clinician?*

**Source-Domain Provenance of Open-Model Health Advice (Reddit vs Curated Web), with Frequency-Matched Base-Rate Control.** *(Score 3.33)*

- **Research question.** When OLMo-2-1B gives health/relationship advice, do its verbatim phrasings disproportionately trace to **peer/forum sources** (reddit, falcon, cc_en_tail) vs. **curated web** (cc_en_head, C4) — and does any skew survive controlling for n-gram frequency and the Dolma corpus base rate? Is the skew stronger for emotionally-loaded than factual-medical advice?

- **Hypotheses.** H1 tier skew vs. base rate (higher forum share than corpus base rate); H2 stakes contrast (higher forum share for high-stakes emotional than low-stakes factual); **H3 frequency is a confound, not the whole story** (effect attenuates but a residual survives within frequency-matched strata — and if it fully vanishes, *that* is the finding); H4 most "forum-ness" comes from *which phrases the model emits*, not within-phrase source skew.

- **Method.** Verified source tiers from infini-gram `metadata.path`: `cc_en_head/middle/tail`, `reddit`, `falcon-refinedweb-filtered`, `c4-filtered`. Grouped a priori into CURATED / PEER-FORUM / MID / OTHER. **30 low-stakes + 30 high-stakes + 20 neutral-control** prompts, frozen before generation. **New `trace/sources.py` aggregator** (~30 lines). **Critical fix:** fetch passages for *all* matched spans and raise `maxnum` 2→10. Estimate the Dolma base rate from neutral answers + high-frequency function-word n-grams. **Frequency-stratify every span** and run a Cochran-Mantel-Haenszel stratified test. **Pool at the tier level**, bootstrap over (prompt, span).

- **Experiments.** E1 main tier contrast, base-rate-normalized; **E2 frequency-confound control (the decisive test)**; E3 maxnum-stability (sampling-bias bound, since `search_docs` returns corpus-internal order, not a uniform sample); E4 phrase-selection vs within-phrase decomposition; E5 worked example + SFT cross-check; E6 (optional, non-pooled) Pythia/Pile + Claude-proxy panels.

- **Metrics.** Forum-share (Wilson CI); curated-share; per-tier distribution vector; **base-rate-normalized lift**; top-3 source concentration; χ²/CMH statistics; coverage diagnostics (sparse-span monitor); maxnum-stability delta; phrase-selection Δ.

- **Baselines.** **Dolma corpus base rate (critical)**; neutral non-advice prompts; frequency-matched random spans; permutation null; Pythia/Pile (separate taxonomy, non-pooled); Claude-proxy panel.

- **Expected findings.** A real, base-rate-normalized over-representation of peer/forum sources in emotional advice phrasings (pilot: "I broke up with my girlfriend" → 5/10 reddit; clinical phrasings → cc_en_head). The decisive, most defensible result is what survives frequency control: likely *attenuates but does not vanish* in mid-frequency strata. If it fully vanishes, the publishable negative result corrects a tempting "LLMs parrot forum advice" narrative.

- **Threats.** **Frequency confound (primary)** — stratification + CMH + frequency-matched baseline, report the null honestly if it dies; doc-ordering bias (pool over many spans, maxnum-stability check, frame as "source mix among returned docs"); statistical-unit trap (pool at tier level, bootstrap over clusters); sparse-span starvation (report ≥1-span fraction); base-rate non-interpretability (always normalize); interpretation overreach (matched spans are often generic boilerplate, not clinical claims — manual boilerplate-vs-substantive coding); external validity (one 1B model, Pythia non-pooled); pre-registration (freeze prompts/labels).

- **Venue / effort.** TrustNLP / data-provenance workshop / BlackboxNLP; Findings short; strong **"where your health advice comes from" stacked-bar figure** for a non-ML/journalism audience. **3–5 weeks, CPU-only;** no extra compute (free API).

---

### 4.6 Developed proposal 5 — *The SFT Template Tax*

**Quantifying Verbatim Instruction-Tuning Reuse in Emotional-Support and Health Advice.** *(Score 3.33)*

- **Research question.** What fraction of an aligned model's empathic/safety phrasing is verbatim-recoverable from the SFT mixture **at a higher per-document prevalence than from pretraining**, using length-matched, **corpus-size-normalized** search? Which template phrases recur across unrelated questions, and onto how few SFT conversations do they concentrate?

- **Hypotheses.** **H1 length-controlled prevalence gap** (for ≥6-word empathic/safety phrases, per-million-doc SFT rate > Dolma rate; for ≤4-word openers the gap vanishes — "I am sorry to hear that" = 48,614 Dolma hits, generic); H2 template recurrence (~20–40 stock phrases across unrelated subdomains); H3 source concentration (each template lands on few SFT conversations); **H4 register specificity** (gap holds for empathic/safety but NOT for length-matched domain-content or random n-grams); H5 template-tax magnitude non-trivial but bounded (pre-committed target ≥10% for support-heavy answers, well below 1).

- **Method.** OLMo-2-1B (temp 0.0 primary + 0.7 stability); Pythia-410m/1.4b as **scale controls**. **The load-bearing fix:** convert raw counts to **per-million-document prevalence rates** (`r_sft`, `r_pre`) and compute **log-prevalence ratio LPR = log10((r_sft+ε)/(r_pre+ε))**, replacing the degenerate raw "SFT >> Dolma" inequality (Dolma raw counts trivially dominate by ~6 orders of magnitude). **De-cap Tulu to all 6 shards** before any quantitative claim. **Automatic phrase mining** (n-grams recurring across ≥3 *unrelated* prompts) removes hand-curation bias. Bucket all analyses by phrase word-length. **UTF-8 hardening** (`PYTHONIOENCODING=utf-8`) — verified necessary on this Windows box (the infini-gram U+2581 echo crashes cp1252).

- **Experiments.** E1 length-controlled LPR curve; E2 register specificity vs. controls (paired Wilcoxon); E3 cross-prompt recurrence matrix; E4 source concentration (GROUP BY conversation over full Tulu); E5 template tax by subdomain and model (+ scale control); E6 robustness sweep (τ, k, unit-alignment, temperature).

- **Metrics.** `r_sft`, `r_pre`, **LPR**; template tax (covered tokens / total, with CI); length-bucketed LPR; recurrence count + template-vocabulary size; distinct SFT conversations per phrase + top-1/top-3 share; register-specificity effect size; zero-Dolma fraction; **paraphrase-miss proxy** (manual count of variants ILIKE misses → frames all SFT numbers as strict lower bounds).

- **Baselines.** Negative-control phrase set (domain-content + random n-grams); factual non-advice prompts (tax ≈ 0); **raw-count comparison as an explicit strawman**; Pythia scale baseline; lightweight human-advice reference; pretraining-only attribution baseline.

- **Expected findings.** The empathic/safety register is more prevalent per-document in SFT than Dolma specifically at ≥6 words, while short openers are generic web boilerplate. A compact template vocabulary recurring across subdomains and concentrating on few conversations; a non-trivial-but-bounded template tax, highest for emotional/relationship advice, ~0 for factual controls, specific to the register. **Honest headline: aligned small-model advice is template-*scaffolded*, not template-*copied*** — the first joint, normalized, two-layer measurement in a high-stakes domain.

- **Threats.** Substring exact-only (all numbers are lower bounds; report paraphrase-miss proxy); causal confound (Tulu is partly GPT-4-distilled, so phrasing lives in both — claim the weaker *prevalence + concentration* statement, not causation); unit mismatch (per-document normalization + unit-aligned variant); selection bias (pre-register + automatic mining + negative control); scale (Pythia control, soften title); 1/6 bias (de-cap); zero-Dolma → +∞ LPR (ε-smoothing + report zero-Dolma fraction separately).

- **Venue / effort.** Memorization/data-provenance workshop / BlackboxNLP / CLPsych + an **inline-SFT-conversation demo**. **~2.5–3.5 weeks, CPU-only;** ~2.2 GB full-Tulu, optional Anthropic key.

---

### 4.7 Developed proposal 6 — *Copy-vs-Compose Curves Across the Pythia Ladder*

**Verbatim Pretraining Coverage of Free-Form Answers as a Function of Scale on Identical Data.** *(Score 3.00 — the scale-control anchor)*

- **Research question.** Holding pretraining data and order fixed across Pythia 70M–1.4B on The Pile, how does verbatim-overlap coverage of a model's own free-form answers against its own corpus change with scale, **after stripping prompt-echo and degenerate-repetition artifacts**? Does corrected coverage rise monotonically, does the longest span grow, and does mean span-frequency fall?

- **Hypotheses.** H1 corrected coverage rises monotonically; H2 longest deduped span grows (monotone claim confirmatory only — <2 orders of magnitude is under-powered for "super-linear"); H3 mean span-frequency falls (rarer phrasings); **H4 artifact share (load-bearing):** a large fraction of the raw slope is echo + degeneracy — if the residual vanishes, the headline is "the apparent copy-up-with-scale curve on free-form completions is largely an artifact" (publishable negative result); H5 decoding sensitivity (steeper under greedy).

- **Method.** Pythia 70m/160m/410m/1b/1.4b (deduped), CPU, base-model path. Add 5 `GEN_MODELS` + 5 `PRETRAIN_INDEXES`→`v4_piletrain_llama` entries. **Artifact control:** prompt-echo strip + degeneracy-collapse (repeated-4-gram detection) producing raw and collapsed variants; metrics on collapsed. **Neutralize the 80-call cap** by truncating to a fixed 80-word window so the cap never binds (verified via `n_calls_used`). Position-union coverage (no double-counting). 50 prompts in 3 strata (factual / advice / instruction), greedy + sampled (3 seeds).

- **Experiments.** E0 pilot + artifact audit; E1 confirmatory curve (greedy, collapsed, windowed); **E2 artifact-share decomposition (the credibility experiment)** — regress coverage on scale with/without degeneracy + length covariates; E3 decoding control; E4 domain breakdown (advice phrasings, e.g. "I am sorry to hear that you" = 509 Pile hits); E5 (optional) Pythia-2.8b rung.

- **Metrics.** Verbatim coverage (position-union); longest deduped span; mean/median/frequency-capped span rarity; novelty = 1−coverage; degeneracy score; prompt-echo rate; **artifact share**; `n_calls_used` (proves cap never binds); a word-vs-BPE caveat metric (report coverage as a relative trend index, not absolute "fraction memorized").

- **Baselines.** Within-study 70m anchor (paired); shuffled-answer chance-level baseline; human-reference advice answers; raw-vs-collapsed; greedy-vs-sampled; optional OLMo-2-vs-Pythia cross-family sanity rung.

- **Expected findings.** A clean artifact-corrected monotone curve at the endpoints, but with E2 revealing that ~30–70% of the *raw* curve was echo + degeneracy and a smaller real residual surviving. Advice prompts show higher baseline coverage of stock empathic phrasings than factual prompts. Honest secondary outcomes: noisy/flat middle rungs, or a genuine null after correction — either is publishable, with the artifact-aware method as the contribution.

- **Threats.** Degeneracy + echo confound (mandatory preprocessing + E2 decomposition); cap × length × scale (80-word window); word-vs-BPE mismatch (relative trend + spot-check); narrow scale range (monotone claim only, optional 2.8b); "so what" (contribution is artifact-corrected free-form coverage vs. the true corpus + the advice domain — cite Carlini/Biderman, frame as replication-plus-extension); base-model validity (re-scope to "free-form completions of instruction-style prompts," not "instruction-following"); n=1 greedy cells (prompts as replication unit + sampled seeds).

- **Venue / effort.** BlackboxNLP / a memorization workshop / TMLR. **3–5 weeks, CPU-only;** core 70m–1.4b needs no extra resources; 2.8b rung optional (~16 GB RAM).

---

## 5. Flagship Worked Example — *How Original Is Machine Advice?*

**Provenance of LLM relationship/health answers.** This is the public-facing spine that the top-5 proposals all feed. It is fully specified below as a runnable pipeline.

### 5.1 Setup

- **Generator:** `allenai/OLMo-2-0425-1B-Instruct`, CPU, transformers, temp 0.7 / top_p 0.9 / max_new_tokens 400 (raised from 256 so trailing disclaimers survive). True-provenance model: its actual data is Dolma v1.7 + Tulu 3.
- **Pretraining corpus:** Dolma v1.7 via infini-gram `v4_dolma-v1_7_llama` (free API).
- **SFT corpora:** `allenai/tulu-3-sft-mixture` (all 6 shards, ~939k convos) + `OpenAssistant/oasst1`, local parquet via DuckDB, **role-scoped to assistant turns**.
- **Prompts:** the running domain — "how do I stop snoring," "I broke up with my girlfriend, what should I do," "I think I'm having a panic attack," "how do I get over a breakup."

### 5.2 Pipeline (per answer)

1. **Generate** the answer; persist via `trace.store.save`.
2. **Pretraining attribution** — `trace.ngrams.attribute('v4_dolma-v1_7_llama', answer, max_calls=80)` → maximal verbatim spans, each with Dolma count + sample source passages (`passages_for`, `maxnum=10`).
3. **Behavioral/disclaimer tagging** — locate empathy / validation / hedge / structure / disclaimer move-spans via the pre-registered variant-aware lexicon; the remaining attributed spans are the *topical* control population.
4. **SFT attribution (role-scoped)** — for each behavioral and topical span, parameterized assistant-only ILIKE over Tulu (native `messages` STRUCT) + oasst1; return total matches + up to 3 full multi-turn conversations + HF viewer links.
5. **Rarity-calibrated three-way label** — per span record `(Dolma_count, SFT_assistant_matches)`; band Dolma counts by quantile; label SFT-LEANING / COMMON / NOVEL.
6. **Source-domain bucketing** — `bucket_source()` maps each passage's `path`/`metadata` to {reddit (+subreddit), cc_en_head/middle/tail, falcon, c4, books, wiki, other, unknown}.
7. **Roll-up** — coverage, verbatim-overlap ratio, novelty rate, cross-layer ratio, source distribution; export JSON/MD/CSV.

### 5.3 Exact metrics

| Metric | Definition |
|---|---|
| **Behavioral-span coverage** | answer word-tokens inside any behavioral span / total answer word-tokens |
| **SFT-assistant recoverability** | fraction of behavioral spans with ≥1 assistant-turn match in Tulu+oasst1 |
| **Behavioral−topical gap** | per-answer paired difference in SFT recoverability (frequency-band-matched); Wilcoxon + bootstrap CI |
| **Verbatim-overlap ratio** | Σ word counts of Dolma-matched spans / total answer words (position-union) |
| **Novelty rate** | 1 − matched-span coverage |
| **Cross-layer ratio** | SFT-recoverable behavioral spans / Dolma-present behavioral spans |
| **Source distribution** | % of disclaimer/behavioral passages per domain bucket (reddit/cc_en_head/falcon/c4…) |
| **Three-way label split** | %SFT-LEANING / %COMMON / %NOVEL after Dolma-frequency banding |

### 5.4 Figure / table plan

- **Figure 1 (hero):** one fully annotated relationship answer ("I just broke up with my girlfriend…") with color-coded behavioral spans, each tagged `Dolma-count / SFT-assistant-match / class`. Expected: opener "It sounds like you're going through a difficult time" (EMPATHY, high in both → COMMON); "it's completely normal to feel this way" (VALIDATION; Dolma ≈1,480 / Tulu-assistant ≈7 → SFT-LEANING); "Here are a few things you can do: First,…" (STRUCTURE, ubiquitous, non-discriminating); "consider speaking to a professional" (DISCLAIMER → SFT-LEANING) — while breakup-specific topical content has near-zero SFT-assistant matches. The visual punchline: **the behavioral skeleton is SFT-shaped; the topical flesh is not.**
- **Figure 2:** stacked-bar "where your health advice comes from" — source-domain distribution for disclaimer/behavioral spans vs. body prose vs. Dolma base rate.
- **Figure 3:** Dolma-frequency vs. SFT-match calibration scatter, showing the naive collapse to "both" and the emergent SFT-leaning subset after banding.
- **Table 1:** per-disclaimer-phrase provenance (phrase | words | Dolma count | top source domains | top reddit subreddits | full-Tulu matches | oasst1 matches | # SFT template families | ownership verdict).
- **Table 2:** behavioral-vs-topical recoverability paired result, with domain × move-class separation and the pre-registered nulls (structure/hedge).

---

## 6. One-Week Pilot You Could Run Immediately

**Goal:** the smallest experiment that yields a real, defensible result with the *existing* tool plus two trivial patches.

**Day 1 — Freeze inputs.** Write 10 advice prompts (breakup, snoring, anxiety, loneliness, panic) + 10 factual-QA controls to `prompts.jsonl`. Commit a ~20-phrase behavioral/disclaimer lexicon (empathy / validation / disclaimer / structure). Set `PYTHONIOENCODING=utf-8` (verified necessary on this Windows box).

**Day 2 — Generate.** Run OLMo-2-1B on all 20 prompts (temp 0.7, max_new_tokens 400). ~10–15 min/answer on CPU; batch unattended.

**Day 3 — Pretraining attribution.** `trace.ngrams.attribute` over Dolma for every answer → verbatim spans + counts + source paths. ~32 s/answer.

**Day 4 — Two micro-patches + SFT search.** (a) Add role-scoped, *parameterized* assistant-only search to `parquet_search` (apostrophe-safe). (b) Query each behavioral phrase **directly** (bypass the rarest-term picker) over the already-cached Tulu shard + full oasst1. Record blob-vs-assistant counts (the role-scoping inflation factor — pre-verified "First," 7,338 → 2,238).

**Day 5 — Source bucketing.** Add the ~30-line `bucket_source()` helper; tabulate the source-domain distribution of behavioral/disclaimer spans vs. body spans.

**Day 6 — Analysis.** Compute, per answer: behavioral-span coverage, SFT-assistant recoverability, novelty rate, three-way (banded) label split, source distribution. Wilcoxon on behavioral-vs-topical recoverability across the 20 answers.

**Day 7 — Write the result.** Produce Figure 1 (one annotated answer) + one summary table.

**Real result the pilot yields:** a first measured estimate of (i) the behavioral-vs-topical SFT-recoverability gap, (ii) the role-scoping inflation artifact, and (iii) the advice-vs-factual source-domain contrast — enough to decide whether to commit to *Templated Empathy* (proposal 1) or *The Disclaimer Reflex* (proposal 2) as the full study. All on cached data, free API, CPU only. **Caveat to note in the pilot:** the cached Tulu is only 1/6 shards, so SFT counts are reported as the *relative* contrast (the bias is common-mode), not as absolute rates — the full study un-caps to all 6 shards.

---

## 7. Limitations & Validity Threats of the Whole Program

These apply across the portfolio and must be stated honestly in every write-up.

1. **Verbatim/substring overlap ≠ causal influence.** This is the deepest, program-wide threat. Presence of a phrasing in Dolma or Tulu shows *co-occurrence*, not that the model *learned* it there — OLMoTrace's own authors disclaim causality, and Grosse et al. (2023) show influence is often abstract and order-sensitive. **Mitigation:** frame every claim as overlap/membership/provenance-candidate; lead with *differential* contrasts (behavioral-vs-topical, support-vs-factual, with/without frequency control); position lexical provenance as a cheap candidate-generator for, not a replacement of, GPU-bound influence methods.

2. **Substring is exact-only; paraphrase is invisible.** ILIKE and n-gram matching miss "I'm sorry you're going through this" vs. "I am sorry to hear." **All SFT numbers are strict lower bounds** — state this prominently; optionally bound the miss with a cheap cached embedding model (e5-small / MiniLM).

3. **Tulu 1/6 sampling biases SFT counts low.** Scan all 6 shards (~2 GB, feasible) for any absolute claim; otherwise report only relative contrasts (common-mode bias).

4. **Tulu is partly distilled from GPT-4/web**, so the same phrasing legitimately lives in *both* layers — weakening any clean "SFT-owned" attribution. Claim the defensible weaker statement (higher per-document *prevalence* + *concentration* onto few conversations), never "caused by SFT."

5. **Frequency confounds source/stage signals.** Generic phrases are common everywhere; raw counts and source tiers track commonness, not stakes. **Mitigation:** corpus-size normalization (per-million-doc rates, LPR), Dolma-frequency banding, frequency-matched controls, CMH stratified tests — and report the null honestly if an effect dies after control.

6. **Sampling/ordering bias in infini-gram `search_docs`** (returns corpus-internal order, ≤10 of up-to-millions of hits), so source profiles are "mix among returned docs," not true mass. Pool over many spans; run the maxnum-stability check.

7. **Small, weak generators.** OLMo-2-1B and Pythia ≤1.4B are not deployed assistants. Scope all claims to "small open-data instruction-tuned models"; bracket with a second model, a temperature sweep, and (proxy-flagged) Claude — never over-generalize to frontier systems.

8. **Closed-model provenance is fundamentally proxy-only.** Claude/GPT/Gemini traces against Dolma are *plausible-web* surfaces, not actual provenance. Flag loudly (`is_proxy`); the ownership theses rest entirely on the true-provenance open models.

9. **Heuristic sensitivity.** Greedy longest-match fragments fixed phrases at the first OOV word; rarest-term selection has degraded to typos ("withotu"). **Mitigation:** query full/anchor phrases directly via `count`/`passages_for`; pre-register lexicons; report held-out-lexicon robustness (this is itself proposal 9/13).

10. **Domain sensitivity / ethics.** Crisis-adjacent prompts are research artifacts only, never deployed; the study object is *provenance of phrasing*, not advice quality; include an ethics/broader-impacts statement and avoid presenting generated advice as guidance.

11. **Reproducibility hygiene.** Pin the infini-gram index id (`v4_dolma-v1_7_llama`), the Tulu/oasst1 parquet commit, the shard set, seeds, and decoding params; the previously-exposed HF token must be **rotated before any run** (per project memory).

12. **Novelty ceiling.** Source-domain breakdowns risk reading as "a viz of metadata WIMBD/infini-gram already expose." The defensible contribution across the program is the **joint two-layer, role-scoped, frequency-/rarity-calibrated, precision-validated per-answer decomposition in an unstudied high-stakes domain** — an *analysis*, not a relabeling of corpus stats.

---

*Files referenced for implementation live under the existing tracer: `trace/ngrams.py` (attribution), `trace/infinigram.py` (`count`/`passages_for`), `trace/parquet_search.py` (DuckDB SFT search; add role-scoped parameterized queries + `_SHARD_CAP=None`), `trace/generate.py` (local generation), `trace/pipeline.py` (`_src_tag` → extend into `bucket_source()`), and `trace/store.py` (history/export). All proposed code changes are small, upstreamable patches to a working tool.*
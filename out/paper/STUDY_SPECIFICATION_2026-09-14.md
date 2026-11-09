# Prospective revision study specification

Status on 15 September 2026: **generation and analysis workflow implemented; prompt manipulation review and confirmatory execution pending**. No prompt in this specification is human-validated, and no confirmatory response has been generated. Matched exploratory eight-response development checks under frozen v2 are complete: all eight responses hit the 256-token cap, and six hit the 512-token cap while two reached EOS. The 256-token primary cap and 512-token longer-budget sensitivity remain proposed; neither has established natural-completion adequacy. Further budget evidence, six-checkpoint calibration, and actual human annotation timing remain pending.

## Research design

The prospective evaluation contains 40 scenario families: 20 health and 20 everyday. Every family has four variants crossing emotional framing (off/on) with request type (information/advice), for 160 evaluation prompts. A separate development split contains 10 families (five health and five everyday), also crossed four ways, for 40 prompts. All four variants of a family remain in one split.

Each family is stored as a shared first-person context, one optional emotional sentence, and two requests. Within a request type, the emotional prompt differs from the neutral prompt by exactly the added emotional sentence. The information and advice requests share the same context. Information requests explicitly ask for an explanation rather than recommendations in both framing variants. The draft excludes the 12 known construction prompts and emergency or crisis scenarios.

The primary grid retains SFT, DPO, and final Instruct checkpoints at 1B and 7B, with three logical seeds. This gives 160 prompts × 6 checkpoints × 3 seeds = 2,880 planned responses. The longer-budget sensitivity uses eight deterministically selected evaluation families, all four variants, all six checkpoints, one seed, and a 512-token cap, for 192 planned responses. Both evaluation modes require a separately completed human manipulation review before execution.

The frozen checkpoint commits are:

- 1B SFT: `0d85a3d037876ce6ac7d4311d994400fc66ac27f`
- 1B DPO: `c4b0485961ab24c2433b090f3b922f0913a9290f`
- 1B Instruct: `48d788eca847d4d7548f375ad03d3c9312f6139e`
- 7B SFT: `1de02c0175118a9de5854aec80a1f970e701e928`
- 7B DPO: `e34ea60adff2e575f4fe7569eaffd1b28509b6fd`
- 7B Instruct: `470b1fba1ae01581f270116362ee4aa1b97f4c84`

Model and tokenizer revisions are pinned to these exact commits. The official rendered serialization identity is identical across SFT, DPO, and Instruct within each size; it is also identical across sizes in the resolved artifacts. The protocol records this observed cross-size equality without imposing it as a requirement.

## Protocol, review, and execution identity

`output/revision/full_study/protocol_v2.json` is the active frozen protocol (`protocol_32f4c8538587e45cdf07e88ed9b1f77d`). It contains every planned response cell, prompt and source hashes, exact checkpoint and tokenizer revisions, official serialization identities, generation settings, deterministic RNG assignments, sensitivity-family selection, and expected grid counts. `protocol_v1.json` is preserved as a superseded pre-generation record; no response was generated under it.

`output/revision/full_study/human_prompt_review_v2.json` is an unfilled manual-review template bound to the v2 protocol, corpus hash, 40 family IDs, and every prompt hash. Its `PENDING_HUMAN_REVIEW` state cannot unlock evaluation. The runner accepts evaluation only when a human supplies reviewer provenance, changes the status to `COMPLETE_HUMAN_REVIEW`, and explicitly affirms all five checks for every family. Structural code checks do not count as human validation.

Each scientific `response_id` identifies a frozen prompt × checkpoint × seed × generation cell. The concrete response artifact identity is the tuple (`run_id`, `response_id`, `execution_profile_id`). The run ID binds the full expected response list, protocol hash, human-review snapshot hash when applicable, and runtime profile. That profile includes CPU or CUDA device class, exact float precision, Torch/Transformers/Python versions, CUDA runtime and accelerator identity when present, thread count, and the single-response generation rule. Resume and export reject mixed runtime profiles.

The per-response RNG seed is derived only from prompt ID and logical seed, then reset immediately before each single-response `generate` call. Response randomness therefore does not depend on checkpoint order, prompt order, subsets, or resume boundaries. The same prompt and logical seed intentionally receive matched RNG seeds across checkpoints.

Responses are checkpointed as one immutable JSON file per response. A durable append-only progress log records attempt starts, checkpoint load times, each completed response, generation time, output tokens, and attempt completion. Shutdown signals finish and save the current response before stopping. Resume validates every cached identity and skips only valid completed response IDs. Strict verification rejects missing responses, duplicate IDs, unexpected IDs, incomplete factorial families, shrunken evaluation grids, changed run configs, changed protocol hashes, missing or changed human-review snapshots, and mixed execution profiles.

## Frozen historical detector and prospective predictions

`output/revision/study_v1/detector_manifest_v1.json` transcribes the complete four-category detector in `experiments/raw_output_robustness.py`: lexicon, exemplars, case-insensitive substring matching, sentence splitter, e5 identity and prefixes, threshold 0.82, exact/embedding union and intersection, and full/64-token/128-token windows. The historical script did not pin embedding-model or tokenizer revisions; the manifest records those revisions as unrecorded rather than inferring them from the current cache.

The five-category definitions in `trace/behavior.py` and `experiments/build_notebook_v2.py` add `crisis_referral`. The manuscript appendix abbreviates parts of the executable inventory. These are recorded as provenance conflicts and are not merged into the four-category 700-response baseline. The recovered cache audit checks all 700 final-7B responses and finds zero exact-matching discrepancies with the historical source. This cache remains a historical diagnostic.

The full-run predictor accepts only a strictly complete 2,880-response evaluation run. It emits exactly 8,640 rows: one row per response for `full_response`, `prefix_64_tokens`, and `prefix_128_tokens`. Each row stores the pinned detector dependency identity, exact/embedding/union/intersection results for the original four categories, and detector-window tokenizer metadata. The metadata includes full-response token count, window limit, evaluated token count, whether the response is shorter than that window, and whether truncation occurred. `crisis_referral` remains explicitly unavailable. Full-response human labels may be compared only with `full_response` predictions; prefix rows are sensitivity analyses.

## Prospective human validation

The separate calibration companion in `experiments/calibration_study.py` constructs 96 development responses: all six checkpoints, four complete scenario families per checkpoint, four prompt variants, and seed 0. Each checkpoint has two health and two everyday families; the deterministic stage allocation covers all ten development families. Both model sizes use the same families at a given stage. Candidate planning is read-only. Freezing and execution are separate actions, and a frozen calibration cap must match the primary cap in its parent protocol. No empirical calibration manifest, complete calibration archive, or human calibration judgments exist yet. See `out/paper/CALIBRATION_PROTOCOL_2026-09-15.md`.

The provisional fresh-response plan samples 13 responses from each of 48 model × emotion × request × domain strata, for 624 random items. It must also cover relevant response lengths, as required by the original revision plan. The implemented and independently reviewed prospective sampling companion derives length bins from the complete verified calibration archive, allocates each cell's 13 selections over nonempty bins, and records exact inclusion probabilities as rational numbers. Its full structural round trip covers 2,880 responses and 8,640 predictions using explicitly synthetic test fixtures. Frozen plans must reconstruct their calibration evidence; response and prediction identities must match the actual verified evaluation run. The plan must be frozen after actual human guide calibration and timing, before examining held-out results. A separate detector-positive enrichment may add at most 120 full-response items, with up to 30 draws per original category and all overlaps retained. No prospective sampling plan or packet has been frozen. The 112-item historical diagnostic packet is not a held-out prospective validation set.

## Portable notebook and resources

`output/jupyter-notebook/advice_marker_revision_study.ipynb` is paired with `advice_marker_revision_study_bundle.zip`. The bundle contains the current local, potentially uncommitted source files; frozen v2 protocol; blank review template; detector manifest; and checkpoint-analysis implementation. The notebook checks the whole-archive SHA-256, validates every bundled file hash and size, extracts into a bundle-content-specific child directory without deleting a user-selected workspace, and never clones public `main`.

The notebook installs required libraries on Colab, inventories the GPU, optionally uses an explicit Google Drive cache, and supports the 256/512 development modes, fixed 1,024-token probe, and separate 96-response calibration. It requires the actual human calibration evidence to freeze and validate prospective sampling before the full evaluation, enforces human prompt review, verifies complete response exports, creates three detector windows, prepares the blinded validation packet, and invokes checkpoint analysis. All 12 execution/freezing/Drive controls default to off. All 21 code cells in the 31-cell notebook executed without errors on the local default path, and all 25 bundle files were hash-verified. This local check does not establish GPU execution or empirical study completion.

The current host has 8 logical CPU cores, 31.9 GiB RAM, no CUDA device, and PyTorch 2.13.0+cpu. Only the final 1B Instruct checkpoint is available locally. The eight-response development run is feasible locally; the 2,880-response evaluation is intended for a suitable GPU runtime after human review.

## Commands

Validate the active protocol and prepare the eight-response exploratory development run:

```text
python -m experiments.full_revision_study plan --protocol output/revision/full_study/protocol_v2.json --mode development-256 --family-ids dev_health_hiking_blister dev_everyday_closet --output-root output/revision/full_study --device cpu --torch-threads 4
```

Execute or resume that exact run from the local pinned 1B Instruct cache:

```text
python -m experiments.full_revision_study run --protocol output/revision/full_study/protocol_v2.json --mode development-256 --family-ids dev_health_hiking_blister dev_everyday_closet --output-root output/revision/full_study --device cpu --torch-threads 4 --local-files-only --execute
```

After a human completes the v2 review manifest, plan and execute the whole evaluation grid:

```text
python -m experiments.full_revision_study validate-human-review --protocol output/revision/full_study/protocol_v2.json --human-review PATH_TO_COMPLETED_REVIEW.json
python -m experiments.full_revision_study run --protocol output/revision/full_study/protocol_v2.json --mode evaluation --human-review PATH_TO_COMPLETED_REVIEW.json --output-root PATH_TO_DURABLE_CACHE --device cuda --execute
python -m experiments.full_revision_study verify --protocol output/revision/full_study/protocol_v2.json --run-dir PATH_TO_RUN_DIR
python -m experiments.full_revision_study export --protocol output/revision/full_study/protocol_v2.json --run-dir PATH_TO_RUN_DIR --output PATH_TO_RESPONSES.jsonl
python -m experiments.full_revision_study predict --protocol output/revision/full_study/protocol_v2.json --run-dir PATH_TO_RUN_DIR --output PATH_TO_ALL_WINDOWS_PREDICTIONS.jsonl
```

Build the portable notebook and local source bundle:

```text
python -m experiments.build_revision_notebook
```

Full execution remains pending human manipulation review, development cap-rate review, GPU access, response generation, prospective human annotation, and analysis of the resulting complete archives.

# Fixed 1,024-token development budget probe protocol

**Date:** 2026-09-15  
**Status:** implementation and nine mocked tests passed independent review; the real manifest was frozen and the fixed CPU run started on 15 September. Completion is recorded separately in the progress report and verified response archive.  
**Implementation:** `experiments/development_budget_probe.py`  
**Parent protocol:** `output/revision/full_study/protocol_v2.json`

## Motivation and scope

All eight matched 256-token development generations reached their cap. In the matched 512-token run, six of eight reached the cap and two reached EOS after 461 and 454 output tokens. The 512 result does not establish how much generation budget is sufficient for naturally completed answers. It motivates one bounded 1,024-token probe before any decision about changing the broader study protocol.

The probe is fixed to the same two complete development families:

- `dev_health_hiking_blister`
- `dev_everyday_closet`

Each family contributes all four emotional-framing × request-type variants. Every response uses the pinned `1b_instruct` checkpoint, logical seed 0, sampling enabled, temperature 0.7, top-p 0.9, and at most 1,024 new tokens. The grid therefore contains exactly eight responses. There is no CLI option to change the families, checkpoint, seed, or cap.

This is exploratory development evidence about completion behavior. It does not require the evaluation prompt-review gate, does not use evaluation families, and produces no human labels. It cannot estimate model-stage effects, detector validity, framing effects, request effects, or corpus-wide completion rates.

## Separation from frozen studies

The probe imports the frozen v2 core implementation without changing its four source-hashed files. It also does not change the separate 96-response calibration implementation. The calibration source remains SHA-256 `fd348ce184a986b8d298bee52c6e78ccfa431f9bf08824318349d1a2b9485862`; a read-only audit and a test verify that preservation. The calibration file is not a scientific or runtime dependency of the probe and therefore is not included in the probe manifest or run identity.

The probe binds the actual dependencies it uses:

- parent protocol ID and file SHA-256;
- the parent protocol's declared four source hashes;
- current `full_revision_study.py` and probe source hashes;
- prompt-corpus hash;
- pinned 1B Instruct model, tokenizer, model-content, serialization, and chat-template identities;
- exact ordered eight-cell grid and 1,024-token generation configuration;
- SHA-256 identities of the completed 256- and 512-token verification and response archives.

The frozen parent loader already rejects source hashes that differ from the protocol. The companion audit exposes the same check per file for review. Before binding the earlier evidence, the companion reconstructs the exact parent 256- and 512-token grid entries, validates every response against its expected entry, requires one shared execution profile across both archives, and checks that each verification record's export SHA-256 equals the corresponding response archive's current SHA-256.

## Identity and matched randomness

The probe constructs its grid with the frozen core identity function. The 1,024-token response IDs differ from the 256- and 512-token IDs because maximum generation length is part of the response identity. The deterministic RNG seed depends only on prompt ID and logical seed, so it remains matched across the three budgets.

Each response adds the probe manifest ID, probe runner version, parent protocol ID and hash, checkpoint alias, and exploratory classification. A concrete artifact is identified by the tuple of probe run ID, response ID, and execution-profile ID.

The manifest has a content-derived ID and is frozen separately from planning. The run ID additionally binds the exact manifest and parent snapshots, companion and core source hashes, the ordered eight response IDs, and the complete execution profile.

## Context and generation checks

After loading the exact pinned checkpoint and before generating any response, the runner renders and tokenizes every prompt with the pinned official tokenizer. It refuses to continue if input tokens plus the 1,024-token budget exceed `max_position_embeddings`. The durable preflight event records the observed context limit and input length for every prompt.

Generation uses the frozen single-response core. Each response is written atomically before the next begins. Progress events are flushed to disk and record load time, generation time, output tokens, and stop reason. Interrupt signals stop only after the current response has been saved.

Resume validates the manifest, parent snapshot, source hashes, run config, execution profile, and every cached response before loading a model. Unexpected, duplicate, misnamed, incomplete, or mixed-profile caches fail. A complete valid cache returns zero new responses. Export is allowed only after strict verification finds all eight unique expected responses.

## Commands

Planning is read-only and creates no manifest or run:

```text
python -m experiments.development_budget_probe plan --parent-protocol output/revision/full_study/protocol_v2.json
```

The unrelated frozen-source preservation audit is also read-only:

```text
python -m experiments.development_budget_probe audit-preservation --parent-protocol output/revision/full_study/protocol_v2.json
```

After review, freeze the one immutable probe manifest:

```text
python -m experiments.development_budget_probe freeze-manifest --parent-protocol output/revision/full_study/protocol_v2.json --output output/revision/development_budget_probe/development_1024_probe_manifest_v1.json
```

The run command remains a dry run without `--execute`:

```text
python -m experiments.development_budget_probe run --manifest output/revision/development_budget_probe/development_1024_probe_manifest_v1.json --parent-protocol output/revision/full_study/protocol_v2.json --output-root output/revision/development_budget_probe --device cpu --torch-threads 4 --local-files-only
```

An explicitly authorized local execution or resume adds `--execute`:

```text
python -m experiments.development_budget_probe run --manifest output/revision/development_budget_probe/development_1024_probe_manifest_v1.json --parent-protocol output/revision/full_study/protocol_v2.json --output-root output/revision/development_budget_probe --device cpu --torch-threads 4 --local-files-only --execute
```

Strict verification and complete-only export use:

```text
python -m experiments.development_budget_probe verify --manifest output/revision/development_budget_probe/development_1024_probe_manifest_v1.json --parent-protocol output/revision/full_study/protocol_v2.json --run-dir PATH_TO_PROBE_RUN
python -m experiments.development_budget_probe export --manifest output/revision/development_budget_probe/development_1024_probe_manifest_v1.json --parent-protocol output/revision/full_study/protocol_v2.json --run-dir PATH_TO_PROBE_RUN --output output/revision/development_budget_probe/development_1024_eight_responses.jsonl
```

## Interpretation limits

The probe covers two deliberately selected families, one checkpoint, and one seed. Its stop counts are descriptive. Even if every response reaches EOS below 1,024 tokens, eight responses cannot establish corpus-wide completion adequacy. If responses still reach the cap, the probe does not by itself identify a sufficient replacement budget.

Decoded text can be compared with the matched 512-token artifacts, but the earlier artifacts did not retain raw continuation token IDs. Any agreement statement must therefore be labeled decoded-text prefix agreement rather than original token-ID identity.

No CPU duration is projected from the 512-token run, and no inference is made about 7B or GPU throughput. The probe does not change the v2 primary budget or the 96-response calibration rules. Any later adoption of a different primary budget requires a deliberate superseding parent protocol and, if necessary, a separately versioned calibration design.

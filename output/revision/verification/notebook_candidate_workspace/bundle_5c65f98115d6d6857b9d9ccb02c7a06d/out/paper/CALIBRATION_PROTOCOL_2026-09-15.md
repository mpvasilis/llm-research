# Development checkpoint calibration protocol

**Date:** 2026-09-15  
**Status:** implementation ready; token-cap decision pending; no empirical calibration manifest frozen and no calibration generation run started by this work  
**Implementation:** `experiments/calibration_study.py`  
**Parent protocol:** `output/revision/full_study/protocol_v2.json`

## Purpose and boundary

This companion run is exploratory development work for refining the prospective human-labeling guide and estimating generation and annotation timing. Its responses and any judgments made from them must remain separate from the confirmatory checkpoint analysis. They cannot be presented as confirmatory evidence, added to the 2,880-response evaluation archive, or used to select a favorable detector result.

The companion accepts development prompts only. It reconstructs its grid from the frozen parent protocol and the frozen prompt corpus, rejects every evaluation row, and has no option for evaluation-family selection. It neither invokes nor relaxes the parent runner's human-review gate. No human labels are created or inferred by the software.

## Fixed grid construction

The grid contains 96 responses:

`6 checkpoints × 4 scenario families per checkpoint × 4 prompt variants × seed 0 = 96`.

The retained checkpoints are SFT, DPO, and Instruct for both 1B and 7B. For a given stage, both model sizes use the same four scenario families: two health and two everyday families. Every selected family contributes the complete emotional-framing by request-type crossing: neutral information, emotional information, neutral advice, and emotional advice.

Within each domain, the five development family IDs are sorted lexicographically. SFT takes indices `(0, 1)`, DPO takes `(2, 3)`, and Instruct takes `(4, 0)`:

| Stage | Health families | Everyday families |
|---|---|---|
| SFT | `dev_health_bedroom_light`; `dev_health_dry_eyes` | `dev_everyday_closet`; `dev_everyday_desk_cables` |
| DPO | `dev_health_hiking_blister`; `dev_health_hoarse_voice` | `dev_everyday_grocery_plan`; `dev_everyday_laundry_colors` |
| Instruct | `dev_health_sunscreen_labels`; `dev_health_bedroom_light` | `dev_everyday_reading_habit`; `dev_everyday_closet` |

This rotation covers all ten development families. Repeated index 0 at Instruct is intentional. No evaluation family is eligible.

## Token-cap decision

The generation cap must be chosen explicitly as either 256 or 512 new tokens. The current pilot evidence informs that decision, but this protocol does not choose a cap. The final calibration cap must equal the primary cap in the parent protocol used for the study. Running `plan` constructs a read-only candidate for either cap and does not write a manifest. Freezing is refused if the selected cap differs from the parent protocol's primary cap. The empirical manifest should be frozen only after the pilot decision is recorded and any superseding parent protocol has been frozen outside this tool.

The other generation settings remain inherited from the frozen core: sampling enabled, temperature 0.7, and top-p 0.9. The manifest records the chosen cap in every response identity.

## Identity and provenance

The calibration manifest has schema `revision-study-calibration-manifest-v1` and a content-derived `calibration_manifest_id`. It pins:

- the parent protocol ID and file SHA-256;
- the prompt-corpus version and SHA-256;
- all six model IDs, model revisions, tokenizer revisions, model hashes, serialization IDs, and chat-template hashes;
- the deterministic family allocation, prompt texts and IDs, conditions, logical seed, RNG seeds, and generation configuration;
- the complete ordered list of 96 response identities.

The run has schema `revision-study-calibration-run-v1`. Its content-derived run ID additionally binds the manifest hash, current calibration and core-runner source hashes, and the complete execution profile, including device, dtype, library versions, and thread count. The run directory stores immutable byte-for-byte snapshots of the calibration manifest and parent protocol.

Scientific response IDs continue to use the frozen core identity function so that prompt, checkpoint, seed, serialization, corpus, and decoding identities are checked by the same validator as the full study. The core generator therefore retains its own response schema and runner version. Each record is additionally marked with the calibration runner version, manifest ID, parent protocol ID and hash, checkpoint alias, and exploratory classification. A concrete artifact is identified by its calibration run ID and response ID.

## Operational workflow

Before a cap decision, inspect either candidate without writing an artifact:

```powershell
python experiments/calibration_study.py plan --parent-protocol output/revision/full_study/protocol_v2.json --max-new-tokens <256-or-512>
```

After the decision, freeze exactly one immutable manifest:

```powershell
python experiments/calibration_study.py freeze-manifest --parent-protocol output/revision/full_study/protocol_v2.json --max-new-tokens <256-or-512> --output <calibration-manifest.json>
```

The run command is a dry run unless `--execute` is present. A dry run validates the frozen manifest without creating a run or loading a model:

```powershell
python experiments/calibration_study.py run --manifest <calibration-manifest.json> --parent-protocol output/revision/full_study/protocol_v2.json
```

Generation requires explicit execution authorization:

```powershell
python experiments/calibration_study.py run --manifest <calibration-manifest.json> --parent-protocol output/revision/full_study/protocol_v2.json --local-files-only --execute
```

Each completed response is written atomically as its own JSON file before the next response begins. Progress events are durably appended. An interrupted run can be resumed only when the manifest, parent protocol, source hashes, execution profile, and cached response contents remain unchanged. Every cached response is validated before any model is loaded; a complete valid cache causes zero new generation calls.

Strict verification requires all 96 unique response files:

```powershell
python experiments/calibration_study.py verify --manifest <calibration-manifest.json> --parent-protocol output/revision/full_study/protocol_v2.json --run-dir <run-directory>
```

Export is refused unless the same complete verification succeeds:

```powershell
python experiments/calibration_study.py export --manifest <calibration-manifest.json> --parent-protocol output/revision/full_study/protocol_v2.json --run-dir <run-directory> --output <responses.jsonl>
```

## Permitted interpretation

Descriptive inspection may report generation time, output length, stopping reason, cap frequency, annotation time, and concrete ambiguities that motivate revisions to a human-labeling guide. The 96 responses are a deliberately selected development frame rather than a probability sample. They do not estimate evaluation-family performance, new-topic coverage, checkpoint effects, detector accuracy, or causal effects of training stage. Any guide revisions should be documented before confirmatory human annotation begins.

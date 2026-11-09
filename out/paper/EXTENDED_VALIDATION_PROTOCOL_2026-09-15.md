# Higher-budget prospective validation protocol

**Date:** 2026-09-15  
**Status:** implementation and synthetic contract tests only; no successor plan, evaluation archive, annotation packet, or human result exists

## Purpose and gate

`experiments/extended_prospective_validation.py` carries the existing prospective sampling design into the evidence-gated higher-budget study. It accepts only a frozen successor protocol produced by `experiments/extended_revision_study.py`. That loader reconstructs the protocol and its grids from the embedded evidence package before this module accepts a calibration run, sampling plan, evaluation run, or prediction archive.

The successor protocol must retain the accepted low-level study and runner schemas, declare `extended-revision-study-protocol-v1` as its extended schema and `extended-revision-study-orchestration-v1` as its orchestration identity, set a primary generation cap above 512 tokens, and set its sensitivity cap above the primary cap. Legacy v2 protocols, legacy calibration manifests, candidate protocols, mixed evidence, and rehashed protocol or grid substitutions are rejected.

No plan may be frozen until all of the following exist and validate together:

- the complete 96-response successor calibration run and its byte-equivalent export;
- the calibration cap equal to the successor primary evaluation cap;
- at least 60 distinct calibration responses reviewed by people using the bound annotation guide;
- positive timing evidence for distinct reviewed responses;
- explicit non-AI provenance, reviewer identifiers, and review time;
- hashes and identifiers tying that evidence to the successor protocol, calibration manifest, run, export, and current guide.

These are readiness requirements. This implementation creates no human labels or certification.

## Frozen sampling design

Length bins use only `generation.output_tokens` from the complete development calibration. Tertiles use nearest ranks at one third and two thirds. Duplicate cutpoints and a cutpoint at the observed maximum are removed rather than replaced with evaluation-informed boundaries, preserving the original tie rule.

The probability component crosses six checkpoints, two domains, emotional framing on or off, and information or advice requests: 48 outer cells of 60 evaluation responses each. It selects 13 responses per outer cell, for exactly 624. Every nonempty frozen length bin receives one item first; remaining slots use Hamilton largest-remainder allocation on residual capacity, with bin index resolving exact ties. Selection is deterministic from content-derived plan identity and named random substreams.

A separate diagnostic enrichment draws at most 30 union-positive full responses for each of the four historical detector categories and at most 120 unique responses overall. Enrichment does not support population estimates. For every selected item, the hidden sampling design records exact numerator and denominator for its random-component probability, all applicable category enrichment probabilities, the enrichment-union probability, and the combined panel probability. This includes items selected only through enrichment, whose chance of random selection remains nonzero.

## Evaluation archive and packet

Packet preparation requires exactly 2,880 verified evaluation responses and 8,640 verified detector predictions: full response, first 64 detector-tokenizer tokens, and first 128 tokens for every response. Prediction IDs and dependency identities are recomputed and validated. The response export must equal the verified run directory, all prediction rows must carry that run ID, and the run must retain the same six readiness inputs used to validate the supplied plan.

Only full-response predictions select enrichment items and compare with human labels. Prefix predictions remain detector-sensitivity measurements. The packet contains a blinded deterministic item order, a hidden response and sampling key, two blank independent annotation passes, and adjudication support through unchanged `validation_v4.py`. Existing packet directories and frozen plan files are never overwritten.

## Public commands

The successor preserves the established command names:

```text
python -m experiments.extended_prospective_validation preview-plan --protocol PROTOCOL --calibration-manifest MANIFEST --calibration-responses RESPONSES --calibration-run-dir RUN --annotation-guide GUIDE

python -m experiments.extended_prospective_validation freeze-plan --protocol PROTOCOL --calibration-manifest MANIFEST --calibration-responses RESPONSES --calibration-run-dir RUN --timing-evidence HUMAN_EVIDENCE --annotation-guide GUIDE --output PLAN

python -m experiments.extended_prospective_validation validate-plan --plan PLAN --protocol PROTOCOL --calibration-manifest MANIFEST --calibration-responses RESPONSES --calibration-run-dir RUN --timing-evidence HUMAN_EVIDENCE --annotation-guide GUIDE

python -m experiments.extended_prospective_validation prepare --plan PLAN --protocol PROTOCOL --responses EVALUATION_RESPONSES --predictions EVALUATION_PREDICTIONS --calibration-manifest MANIFEST --calibration-responses CALIBRATION_RESPONSES --calibration-run-dir CALIBRATION_RUN --timing-evidence HUMAN_EVIDENCE --evaluation-run-dir EVALUATION_RUN --annotation-guide GUIDE --out-dir PACKET
```

The output defaults are under `output/revision/extended_study/`. Preview is read-only. Freeze and prepare require explicit complete inputs and use atomic, no-overwrite writes.

On Windows, every successor validation input and derived artifact path is normalized through the shared extended-length path helper before file access, hashing, globbing, copying, replacement, or calls into the unchanged analyzer and annotation validator. User-visible paths are converted back to ordinary readable form. Extended-length prefixes are never included in scientific identities or provenance fields. Tests retain a nested artifact path longer than 300 characters and a relocated complete-package check rather than shortening the fixture around the platform limit.

## Limits

This protocol fixes a validation design; it does not make 1,024 tokens scientifically sufficient, certify prompt review, certify development evidence, run models, or produce annotations. The planned 624-item probability sample and up-to-120-item enrichment remain prospective until a valid successor plan is frozen before held-out evaluation inspection. Two completed human passes and adjudication are required before any final detector-validation claim.

# Extended higher-budget study protocol

Date: 2026-09-15

This successor preserves the frozen six checkpoints, the 160 evaluation and 40 development prompts, the 40/10 family split, the primary seeds 0/1/2, the sensitivity and development seed 0, the family-selection rule, and the stage-rotated 96-response calibration design. It changes only the generation budgets and the evidence-gated orchestration around the unchanged low-level generator.

No generation cap is selected in this document. Candidate planning requires two explicit integers: a primary cap greater than 512 and a sensitivity cap greater than the primary cap. Planning writes nothing. The candidate contains 2,880 primary evaluation responses, 192 pre-specified sensitivity responses, and two 40-response development grids. The calibration companion contains exactly 96 development responses, 16 per checkpoint, at the selected primary cap. A complete evaluation produces 8,640 detector rows across the full-response, 64-token-prefix, and 128-token-prefix windows.

Freezing is prohibited until the fixed 1,024-token probe has a complete verified 8/8 run and exact complete export. Freezing also requires a separate development budget-decision record whose selected caps, source parent, probe manifest, and probe export hash agree with the proposed successor. The decision is provenance for a development choice; it is not a human annotation. The frozen protocol copies its source parent, probe manifest, complete probe run, probe export, decision, and preceding matched 256/512 evidence into a relocatable package and validates their contents rather than relying on their original locations.

The protocol retains `revision-study-protocol-v1`, `revision-study-full-runner-v2`, and `revision-study-response-v2` because the low-level response generator and archive format are unchanged. `extended_schema_version` and `orchestration_version` identify the successor. Response and RNG identities are rebuilt by the unchanged core identity functions, so a response may intentionally have the same scientific ID as a fixed probe at matching prompt, checkpoint, seed, and cap. Protocol, calibration, readiness, and run identities prevent artifacts from different studies from being mixed.

Evaluation and sensitivity generation require all of the following before checkpoint loading: a genuine complete human prompt-review manifest; a frozen prospective sampling plan; its complete 96-response calibration manifest, run, and export; complete development evidence with real timing; and the annotation guide. The run stores immutable copies and content hashes of these artifacts. Resume and verification repeat the checks from the copies. Development-primary and development-longer runs remain exploratory and may subset only complete four-variant families.

At every loaded checkpoint, generation validates the official chat-template hash and serializes every pending prompt before producing a response. Both selected caps must fit the checkpoint context window. Progress records model-load time, context-preflight time, per-response generation time, output-token count, effective stop reason, EOS status, and cap status. Cached responses, their execution profile, the protocol snapshot, and all gate snapshots are validated before any model is loaded. A complete cache returns zero new responses without rewriting response files.

The successor validator also closes a metadata gap in the frozen low-level validator without changing that frozen source. It requires the top-level generation cap and generation config to match the frozen response identity, bounds output-token counts by that cap, and reconstructs the producer's stopping convention: EOS has priority, otherwise a cap-length output is marked `max_new_tokens`, otherwise the stop is `other_model_stop`. The same checks apply to copied 256/512 evidence, the fixed 1,024-token probe, calibration responses, cached responses, and exported runs.

The command-line interfaces are dry-run by default. Candidate planning, verification, and inspection never load generation checkpoints. The `run` commands require `--execute` before generation can begin. Freezing the protocol and calibration manifest are separate explicit commands; this implementation does not freeze a real protocol or choose a real cap.

On Windows, successor file operations normalize their inputs to the operating system's extended-length path form. This permits the immutable evidence and readiness snapshots to retain their descriptive nested layout even when a response-cache path exceeds the traditional 260-character limit. Serialized reports and scientific identities use ordinary display paths or path-independent content identities; the Windows filesystem prefix never becomes scientific provenance.

Implementation:

- `experiments/extended_revision_study.py`
- `experiments/extended_calibration_study.py`
- `experiments/extended_prospective_validation.py`
- `experiments/extended_sensitivity_analysis.py`

Outputs use `output/revision/extended_study/`; the earlier frozen `output/revision/` artifacts and source files remain unchanged.

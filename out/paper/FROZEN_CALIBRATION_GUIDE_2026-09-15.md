# Frozen calibration preparation companion

This companion carries the frozen 1,024-token primary protocol, its complete relative evidence tree, the predeclared 2,048-token sensitivity design, the exact 96-response calibration manifest, and blank prompt-review materials. The evidence retains the historical execution profiles for the completed 256/512/1,024-token probes, but no execution profile has been selected for future calibration or evaluation. It contains no calibration responses, human judgments, model weights, or evaluation-readiness claim.

Use it with the already verified source bundle `output/jupyter-notebook/advice_marker_extended_study_bundle.zip` (SHA-256 `d148bdb43c6545c0a02c5cb75e6b2683f0b56d3dfa31c2e0c25050b3031ed4cb`, bundle ID `extended_bundle_6550d81c61f8ffc57b4c12e9773248dc`). Extract both archives into the same workspace root. The protocol must remain beside its `e/` directory.

Set these notebook environment paths to the extracted files:

- `EXTENDED_PROTOCOL=<workspace>/output/revision/extended_study/full_study/protocol_v1.json`
- `EXTENDED_CALIBRATION_MANIFEST=<workspace>/output/revision/extended_study/calibration_study/manifest.json`

Keep `FREEZE_SUCCESSOR_PROTOCOL=False` and `FREEZE_CALIBRATION_MANIFEST=False`. The notebook loads the frozen protocol directly and uses the existing calibration manifest when calibration is explicitly enabled. `PRIMARY_CAP` and `SENSITIVITY_CAP` may remain `None`; the source-bundle notebook's generic pending-cap message describes its candidate/refreeze controls and does not invalidate the already frozen 1,024/2,048 protocol.

The blank review is `output/revision/extended_study/human_prompt_review_template_v1.json`, with its bound worksheet at `output/revision/extended_study/prompt_review_worksheet.md`. The named `v1` template is the intended frozen file; the worksheet's isolated reference to a “blank v2 file” is a wording typo and does not identify a different artifact. Preserve the template. Make a separate completed copy, record only genuine human judgments, and set `EXTENDED_HUMAN_REVIEW` only to that completed copy. Pointing `EXTENDED_HUMAN_REVIEW` at the blank template correctly fails validation.

All execution controls remain off by default. A suitable environment must provide the six exact pinned checkpoints before `RUN_CALIBRATION=True` is enabled. The execution profile is recorded when that run is created. Extracting or validating this companion performs no installation, download, model load, generation, human-field creation, protocol freeze, or manifest freeze.

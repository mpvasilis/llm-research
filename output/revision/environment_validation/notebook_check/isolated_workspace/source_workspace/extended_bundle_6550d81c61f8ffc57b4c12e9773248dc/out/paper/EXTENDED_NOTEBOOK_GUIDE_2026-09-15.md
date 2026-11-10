# Extended study notebook guide

The extended notebook is a portable preparation and execution workflow. Its default run validates the uploaded source bundle and the existing frozen parent/probe prerequisites, then reports every successor stage as pending. It does not select generation caps, install packages, access the network, load models, mount Drive, create human records, or freeze scientific artifacts.

Use `output/jupyter-notebook/advice_marker_extended_study.ipynb` with `advice_marker_extended_study_bundle.zip`. The notebook verifies the bundle hash and every member before importing the four successor modules.

## Required order

1. Finish and strictly verify all eight fixed 1,024-token probe responses. Export the exact verified run.
2. Review the probe evidence and create a separate budget-decision JSON record. Set `PRIMARY_CAP` and `SENSITIVITY_CAP` in the notebook to the same explicit integer values recorded in that decision. The primary cap must exceed 512 and the sensitivity cap must exceed the primary cap.
3. Set `PROBE_RUN_DIR`, `PROBE_EXPORT_PATH`, and `BUDGET_DECISION_PATH`. The fixed probe manifest is already bundled. Enable only `FREEZE_SUCCESSOR_PROTOCOL` to freeze the evidence-bound protocol.
4. Have a real reviewer complete the prompt review outside the notebook. `CREATE_HUMAN_REVIEW_FORM` only creates a blank form; it never supplies judgments.
5. Freeze and generate the exact 96-response calibration grid. Export it only after strict complete verification.
6. At least 60 distinct calibration responses must receive genuine human guide review. The timing evidence must include a nonempty timed subset and satisfy `extended_prospective_validation`.
7. Freeze the prospective plan before held-out evaluation inspection.
8. Run and strictly verify the complete 2,880-response primary grid and 192-response sensitivity grid using the readiness mapping embedded in the notebook.
9. Export 8,640 primary detector rows and 576 sensitivity detector rows with the pinned dependencies.
10. Run the unchanged primary analysis and the matched seed-zero sensitivity analysis. Prepare the prospective packet, complete two independent human passes, adjudicate disagreements if required, and score only after those human files exist.

Every mutation or expensive operation has its own `False` switch. Enable one stage at a time after its displayed prerequisites pass. Generation functions resume safely from individually validated cached responses.

For local validation of the delivered notebook, the builder executes every code cell with all controls disabled:

```powershell
python -m experiments.build_extended_revision_notebook
```

The builder writes its verification records only under `output/revision/extended_notebook/`. It refuses to include a partial probe run as completed evidence and never writes under `output/revision/extended_study/`.

The local all-controls-off validation checks bundle integrity, imports, path handling, and safe workflow branching. It does not install dependencies, validate a clean-environment installation, load models, or test a GPU run.

The local fixed probe is now complete and remains external to this source-only bundle because no budget decision or successor caps have been selected. Its verified inputs are:

- run: `output/revision/development_budget_probe/runs/budget_probe_run_9acdc0e5ce18effd34337b139ff53c24/`
- export: `output/revision/development_budget_probe/exports/development_1024_eight_responses.jsonl`
- verification: `output/revision/development_budget_probe/development_1024_verification_20260915.json` (SHA-256 `c673a94fb20f659a77c39a941e2d3612ae90c28060fa5b18077d54c963a54369`)

The companion `output/jupyter-notebook/advice_marker_completed_probe_bundle.zip` contains the fixed manifest, every file in that exact completed run directory, the export, and the final verification under their repository-relative paths. Extract it at the same root as the source bundle. The resulting run and export paths are exactly the two paths above. Its external manifest is `output/revision/extended_notebook/completed_probe_bundle_manifest.json`, which records every member hash and the compatible source-bundle hash.

Supply the extracted run and export together with the later explicit budget-decision JSON. The source builder's optional combined-package mode accepts those three inputs only as a complete unit and strictly validates all eight probe responses before adding them. The completed-probe companion contains no budget decision and does not authorize a cap or protocol freeze.

This bundle contains source and preparation artifacts, not model weights. A read-only local audit at `output/revision/verification/checkpoint_local_weight_files_audit_20260915.json` (SHA-256 `c23f82417a7338f89300d30386752c425f8a8cb102e891805a3000785f48340c`) found a complete weight file only for the pinned `1b_instruct` snapshot; the other five pinned snapshots lacked local weight files. Running the full study therefore requires suitable compute and access to the exact pinned weights. The audit file is informative and is not a bundle dependency.

# Revision annotation materials, version 4

The files in this directory are preparation materials, not completed human
validation.

`historical_final7b_response_frame_v4.jsonl` normalizes the 700 recovered
final-7B responses into the shared revision-study schema. Its 140 scenario
families are marked `historical_diagnostic`; 60 responses derived from the 12
construction prompts are explicitly flagged. This frame is not fresh, held-out,
or stagewise evidence.

`historical_final7b_frame_manifest_v4.json` records source and output hashes,
counts, scope, and the construction-prompt policy.

`historical_packet_v2/` is the provenance-complete historical diagnostic
starter packet. It joins the frame to 700 versioned full-response union
predictions whose embedding dependency is pinned to a resolved commit. The
packet contains 112 unique blinded items: 40 probability-sample items and 75
items selected by at least one detector-positive enrichment rule, with three
items in both components. Nine sampled items come from construction prompts and
remain flagged in the hidden provenance. Enriched rows are diagnostics and are
not pooled into population estimates.

`historical_packet/` is the earlier provisional packet made before the
embedding-model revision was pinned in the prediction identity. It is preserved
for provenance, explicitly marked provisional, and should not be assigned.

The v2 packet includes two independent 112-row blank templates. They contain no
labels, timestamps, annotator identities, metadata certifications, or AI
judgments. No `human_annotations_*` file or human `.meta.json` has been created.
Each human must initialize and certify their own pass.

The packet remains a one-checkpoint historical diagnostic. It is not fresh or
held out and cannot validate stagewise trajectories.

See `out/paper/ANNOTATION_GUIDE.md` for label definitions and commands.

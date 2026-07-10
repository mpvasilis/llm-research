# Blinded detector-validation protocol (v3)

The complete Colab notebook prepares either the existing 120-sentence sample
or, after full regeneration, the preferred response-level sample spanning
models, stages, and conditions. Two humans must label independently. The UI
shows only an item ID, the prompt when available, and the text to label.

The files are deliberately separated:

- `validation_items_v3.csv`: blinded text shown to annotators;
- `validation_key_v3.csv`: hidden predictions, stages, and conditions;
- `human_annotations_1_v3.csv` and `human_annotations_2_v3.csv`: independent
  per-annotator files;
- matching `.meta.json` files: explicit human/independence certification;
- `validation_disagreements_v3.csv`: created only after both passes;
- `adjudicated_annotations_v3.csv`: final labels and decision rules.

## Labels

- `empathy_opener` (reported as **empathy expression**): explicitly acknowledges
  or sympathizes with a user's emotional experience. Generic politeness and
  purely factual descriptions do not count.
- `validation`: normalizes, legitimizes, or affirms an emotion or reaction.
- `disclaimer` (reported as **professional referral**): recommends that the user
  consult a qualified professional. Merely mentioning clinicians, treatments,
  or professional roles in factual exposition does not count.
- `crisis_referral`: directs the user to a crisis-specific resource such as 988,
  emergency services, or a suicide-prevention hotline.
- `structure` (reported as **structuring language**): explicitly organizes the
  response through an enumeration or list preamble. A bold span or transition
  without an organizing function does not count.
- `none`: none of the above is expressed.

Multi-label annotation is allowed. Label what the sentence expresses, not what
the detector might have intended to find.

## Procedure

1. Prepare items in the complete Colab notebook. Use `response_stagewise` for
   the strongest validation; `sentence_120` is the deadline-minimum sample.
2. Annotator 1 sets `ANNOTATOR_ID=1`, provides a pseudonymous ID, certifies that
   they are human and independent, and completes the blinded UI.
3. A different person repeats the process with `ANNOTATOR_ID=2`. Neither opens
   `validation_key_v3.csv` or the other annotator's CSV.
4. Run the scoring cell. It computes category-level agreement and Cohen's
   kappa, detector precision/recall/F1 with 95% Wilson intervals, and
   per-condition/per-stage metrics. It refuses incomplete or same-ID passes.
5. Only after both passes, adjudicate exported disagreements. Record a final
   label and short decision rule, then rerun scoring.

For the response-level positive-enriched sample, use only rows marked
`random_stratified` for unweighted prevalence and cross-condition estimates.
The enriched rows improve precision measurement but are not a prevalence
sample.

The manuscript should report annotator independence, category-level agreement,
per-condition and per-stage error rates, confidence intervals, and the
adjudication procedure.

## Excluded evidence

The legacy 80-row `validation_sheet.csv` is a tool-assisted draft whose human
provenance is not established. The AI annotation files are a useful diagnostic
audit but cannot be described as human validation. Neither enters validation
v3 or the final human result.

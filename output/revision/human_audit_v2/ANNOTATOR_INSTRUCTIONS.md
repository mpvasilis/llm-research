# Annotator instructions: sentence audit v2

You will label single sentences taken from chatbot answers. Two people label
every sentence independently. Do not discuss items with the other annotator
until both files are complete.

## Files

- `audit_items_blinded.csv`: one row per sentence. Columns: `item_id`, the user
  `prompt`, the `previous_sentence` from the same answer (context only, blank
  for the first sentence) and the `sentence` to label.
- `annotator_1_labels.csv` or `annotator_2_labels.csv`: your own blank file.
  Fill one of them only.

Do not open `hidden_key/` or anything under `output/revision/free_analyses/`.
They hold detector outputs and model identities.

## What to label

Label the `sentence` only. Use the prompt and the previous sentence to
understand it, but do not credit the sentence with something that only the
context says. Put `1` or `0` in each of the four columns. More than one `1` is
allowed.

The definitions are those in `out/paper/ANNOTATION_GUIDE.md` (read it first):

| Column | Guide target(s) |
|---|---|
| `empathy` | `empathy_expression`: explicitly acknowledges or sympathizes with the user's emotional experience. Generic politeness, advice and factual talk about emotions do not count. |
| `validation` | `validation`: normalizes, legitimizes or affirms an emotion or reaction. Empathy alone does not count. |
| `referral_disclaimer` | `1` if the sentence meets `professional_referral` (recommends contacting, consulting or seeking help from a qualified professional) or `professional_disclaimer` (states the assistant's lack of professional status or limits of the answer as professional advice). A factual mention of professionals or treatments alone is `0`. |
| `structuring` | `structuring`: explicit organization through a list item, enumeration, heading, or a preamble such as "Here are three steps". Ordinary transitions without an organizing function are `0`. |

Use `notes` for hard cases (a short quote or reason). Leave no label cell
blank. The guide's response-level fields (opening position, professional
mention) are not used in this sentence audit.

## When you finish

Save the file as UTF-8 CSV with the same columns and row order, add your
pseudonymous annotator ID and the completion date to the first row's `notes`,
and hand it to the study lead. Scoring is done with
`python -m experiments.free_analyses.audit_score_labels annotator_1_labels.csv annotator_2_labels.csv`.

# Independent readability check

Status: **prepared, not completed**. Two readers unfamiliar with the project
must read the same manuscript version independently. No reader has been
assigned, no response has been supplied, and no human completion is certified.
This check addresses the revision plan's requirement that two unfamiliar
readers can independently identify the paper's question and findings. It does
not replace response annotation or scientific validation.

## Materials and order

Give each reader the PDF recorded in the packet manifest and only their own
blank reader form under `output/revision/readability_v1/`. Keep this protocol's
assessment guidance, the plain-language companion, the review-response matrix,
the other reader's answers, and prior reviewer comments away from readers
until they finish the independent first reading. Do not coach their answers.

The reader records their own pseudonymous identifier, date, relevant research
experience, prior familiarity with the project, and actual time spent. These
fields remain blank in the prepared forms. The first two questions capture the
reader's own account of the question and findings before the more specific
methodology questions. Preserve those initial answers if later questions
change their understanding.

Both readers must use the same PDF SHA-256. If the manuscript changes before
either reading begins, issue two new blank forms bound to the new version.
If a reading has already begun or finished, preserve its original file and
version binding; do not silently update the manifest beneath it.

## Author assessment after both forms are returned

Record each criterion separately for each reader as pending, clear, unclear,
or materially misunderstood, with the exact answer supporting the assessment.
Do not convert a blank form, an AI review, or an author's own reading into an
independent human result. Confirm that the two readers are different people
and that their answers were independent before declaring this requirement met.

The expected central question concerns how detection rules and the amount of
response text examined affect conclusions about advice-style language across
OLMo-2 post-training. Exact phrasing or memorized numerical values are not
required. The core findings should be recognizable in both readers' accounts:

1. The historical detector and output-window choices change some comparisons.
   The available response-level sensitivity evidence is limited to final 7B;
   the 120-sentence diagnostic uses AI labels. A point-estimate sign change
   does not establish two certain effects in opposite directions.
2. Under the historical rule, same-interface SFT-to-DPO validation gaps
   increase and structuring gaps contract. Base-to-SFT also changes the input
   interface, and most stagewise raw outputs are unavailable.
3. In the exploratory paired preference-data analysis, broad referral/
   disclaimer and structuring associations remain positive after conditioning
   on length, while both narrower-referral intervals include one. These are
   associations, not changes in preference probability or causal effects.

The reader should distinguish an operational marker and detector from a
human construct, keep the different datasets and their units separate, and
recognize the major scope limits. An answer that treats the AI diagnostic as
human validation, the conditional odds ratio as a causal effect, or the two
overlapping preference mixtures as independent replications identifies a
substantive explanation problem even if the prose otherwise seems clear.

## Completion and revision decisions

The requirement is met only after both independent forms and their version
provenance are present, both readers identify the same central question and
core takeaways, and substantive misunderstandings have been resolved in the
manuscript. Preserve disagreement and confusing passages as revision evidence.
There is no automatic score or checkbox that establishes comprehension.

Use recurring points of confusion to revise the relevant passage, table,
caption, or claim. If the change is substantive, obtain another independent
reading of the revised version. Report which version was assessed and what
changed; do not claim that an older reading validates a later manuscript.

The present paper remains a revision draft. Human annotation, the complete
prospective checkpoint archive, and its planned analyses are separate pending
requirements even if these readability checks eventually pass.

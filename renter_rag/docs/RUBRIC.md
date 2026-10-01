# Correctness rubric (for the two human raters)

The automatic check (`facts` in the question files) only asks "does the answer contain every key fact?". It can't tell
a correct paraphrase from a wrong one, so two people score every **answered** test question in
`results/rubric_test.csv`. Each rater scores alone, then we report the agreement and settle disagreements together.

| Grade | Meaning | Example (Q: notice for a rent increase?) |
|---|---|---|
| **Correct** | Every key fact is right and nothing important is missing or wrong. Citations point at a source that supports the answer. | "At least 90 days' notice, in the prescribed form [1]." |
| **Partly correct** | The core fact is right, but a condition or exception that matters is missing, OR one minor detail is wrong, OR the citation does not support the sentence. | "90 days' notice." (correct, but leaves out that the notice must use the prescribed form; acceptable only if the question didn't hinge on it) |
| **Incorrect** | A key fact is wrong (number, period, who is responsible, whether something is allowed), the answer would lead a renter to act wrongly, or it relies on a source that says something else. | "60 days' notice." |

Rules:
1. Judge against the **cited source and the Act**, not against your own memory of the law.
2. An answer that is right but would still mislead a renter (e.g. leaves out a deadline they must meet) is at most *partly correct*.
3. Mark a citation problem in `notes`, even if the text is right.
4. Don't look at the other rater's column until both are done.

Report: % correct / partly / incorrect for each rater, the % of questions where both agree, and Cohen's kappa
(`sklearn.metrics.cohen_kappa_score`). If an LLM is ever used as an automatic judge, check with the mentor first and declare it.

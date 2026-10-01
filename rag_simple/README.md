# Victorian Renter Rights Assistant: Group 99 (COSC2669 WIL Project)

A RAG chatbot that answers Victorian renting questions from Consumer Affairs Victoria (CAV) guidance, shows the source of every answer, and says "I don't know" when the evidence is weak. It comes with a test-driven evaluation and a dashboard.

## What the brief asks for, and where it is

| The brief asks for | Where it is in this project |
|---|---|
| A workable RAG pipeline (start from Walert) | `rag.py`: BM25 search (Walert's baseline) + answer generation with sources |
| A knowledge base for a chosen domain | `data/pages/`: 25 CAV renting pages → 156 passages (`build_kb.py`) |
| Use a free LLM (e.g. Ollama) | `rag.py --ollama` path uses `llama3.2:3b` locally |
| A robust evaluation framework, including % unanswered (Walert) | `evaluate.py`: answer rate / unanswered %, accuracy, refusal rate, source accuracy, NDCG@3 |
| Other dimensions (fairness, correctness of sources…) | Fairness: the same questions in plain, legal and slang wording. Source accuracy: did it cite the right passage? |
| Show value to a stakeholder | Renters get current, sourced answers. The no-RAG comparison shows a normal LLM giving the old "60 days" rule |
| A tangible output (demo / MVP) | `app.py`: Ask page + Dashboard |
| Ethics / standards | Says "I don't know" instead of guessing, shows sources, not legal advice, runs locally (privacy) |

## How to run

```bash
pip install -r requirements.txt
python build_kb.py        # 1. pages -> data/passages.json
python evaluate.py        # 2. tune threshold on dev, test on test -> results/
python app.py             # 3. open http://127.0.0.1:5000  (Ask + Dashboard)
```

With Ollama installed (`ollama pull llama3.2:3b`):

```bash
python evaluate.py --ollama     # LLM writes the answers + "LLM without RAG" comparison
python app.py --ollama
```

## The files (read them in this order)

1. **`build_kb.py`** cuts each CAV page at its headings into passages, and gives each one an ID like `CAV-RENTINC-05`.
2. **`rag.py`** answers one question: scope check → swap everyday words (landlord → rental provider) → BM25 finds the top 3 passages → confidence = score of the best passage → if confidence is below the threshold, say "I don't know" → otherwise write the answer and cite the passages.
3. **`evaluate.py`** runs all the test questions and computes the dashboard numbers. The threshold is chosen on the dev questions, and results are reported on the test questions.
4. **`app.py`** and **`templates/`** make up the web app.
5. **`data/dev_questions.yaml`** and **`data/test_questions.yaml`** hold the questions with known answers.

## How the threshold is chosen

A wrong answer with a source attached looks trustworthy, so it is **worse than no answer**. Each question is scored:

| | Answered correctly | Answered wrongly | Said "I don't know" |
|---|---|---|---|
| Question it should answer | +1 | −1 | 0 |
| Question it should NOT answer | — | −1 (answered at all) | +1 |

We try every threshold from 0 to 25 on the **dev** set and keep the one with the best average score. That's 9.5, shown in `results/threshold_curve_dev.png`. Then we test once on the **test** set.

## Results (offline, extractive answers; test set of 81 questions)

| Metric | Result |
|---|---|
| Answer rate (questions it should answer) | 82.6% (so 17.4% unanswered) |
| Accuracy on answered questions (all key facts present) | 54.4% |
| Said "I don't know" to questions it shouldn't answer | 83.3% (10 of 12) |
| Answers citing a correct source | 80.7% |
| Average sources per answer | 1.71 |
| Retrieval NDCG@3 | 0.59 (Walert's BM25 on its Known set: 0.49) |
| Fairness: accuracy for the same 12 questions | plain 58%, legal 67%, **slang 17%** |

## Key findings to present

1. **The "I don't know" gate works.** It refused 10 of 12 questions it should not answer (e.g. "bond in NSW?", "AFL grand final?").
2. **When it finds the right page, it doesn't always pick the right sentence.** Sources are correct 81% of the time, but only 54% of answers contain all the key facts. An LLM should fix this: run `--ollama`.
3. **Fairness problem: slang wording.** "im bit behind on rent, when can they tell me to get out" gets far worse results than the plain version, because keyword search needs matching words. Next step: better synonyms or embeddings.
4. **Why RAG adds value:** the law changed in Nov 2025 (rent-increase notice went from 60 to 90 days). Our answers come from the current CAV pages with dates. An LLM on its own may give the old rule (measured by `--ollama`).

## Questions a marker might ask

- *Why BM25?* Walert used it as the baseline, it is fast, it needs no GPU, and in Milestone 1 it did well on Walert's inferred questions.
- *What is the confidence score?* The BM25 score of the best passage: how strongly the question's words match the best passage.
- *Why a separate dev set?* If we tuned the threshold on the test questions, the test result would look better than it really is.
- *How do you know an answer is correct?* Each test question lists key facts (e.g. "90 days"). An answer is correct only if it contains all of them.
- *Limitations:* small question set (81), keyword search struggles with slang, CAV guidance only (not the full Act), and the extractive answers are clunky without the LLM.

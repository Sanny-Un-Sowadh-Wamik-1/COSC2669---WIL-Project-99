"""
STEP 3 - Test-driven evaluation

We have two sets of questions with known correct answers (data/*.yaml):

  dev_questions.yaml   24 questions - used ONLY to tune the confidence threshold
  test_questions.yaml  81 questions - used ONLY to report the final results

(Tuning and testing on different questions is like studying from practice
questions and then sitting a different exam. It stops us "overfitting".)

Question types:
  K  known       answer is written directly in one passage
  I  inferred    a real-life scenario that needs the rule applied
  O  out-of-KB   the system SHOULD say "I don't know" (e.g. "bond in NSW?")
  L  legal       a K question reworded in legal language      (fairness check)
  C  colloquial  a K question reworded in slang / simple English (fairness check)

Each answerable question lists:
  gold  : the passage IDs that contain the answer
  facts : the key facts a correct answer must contain, e.g. '90 days'

What we measure (these are the numbers on the dashboard):
  - answer rate            % of answerable questions the system answered
  - unanswered %           100 - answer rate (Walert's main metric)
  - accuracy on answered   % of answers containing ALL the key facts
  - refusal rate           % of out-of-KB questions where it correctly said "I don't know"
  - source accuracy        % of answers citing a correct (gold) passage
  - avg confidence         average confidence score when it answers
  - avg sources            average number of passages cited per answer
  - retrieval NDCG@3       did the search put the right passage near the top? (same as Walert)
  - fairness               accuracy for plain vs legal vs colloquial wording of the same question

Choosing the threshold (the "balance" between answering and refusing):
  answered correctly  +1      refused a question it should answer   0
  answered wrongly    -1      out-of-KB: refused +1, answered -1
  A wrong answer costs more than no answer, because a wrong answer with sources
  attached looks trustworthy. We try every threshold on the DEV set and keep the
  one with the highest average score.

Run:
  python evaluate.py              # offline (extractive answers)
  python evaluate.py --ollama     # answers written by the local LLM + a "no RAG" comparison
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ranx import Qrels, Run, evaluate as ranx_evaluate

from rag import RenterRAG, OLLAMA_URL, OLLAMA_MODEL

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------- loading questions

def load_questions(filename):
    """Read a question file. L and C questions copy gold/facts from the K question they reword."""
    questions = yaml.safe_load((HERE / "data" / filename).read_text(encoding="utf-8"))
    by_id = {q["qid"]: q for q in questions}
    for q in questions:
        if q.get("pair"):
            original = by_id[q["pair"]]
            q["gold"], q["facts"] = original["gold"], original["facts"]
        q.setdefault("gold", {})
        q.setdefault("facts", [])
    return questions


# ---------------------------------------------------------------- checking one answer

def check_answer(question, result):
    """Compare the system's result with the known answer for one question."""
    row = {
        "qid": question["qid"], "type": question["cat"], "question": question["q"],
        "confidence": result["confidence"], "answered_raw": result["answered"],
        "reason": result["reason"], "answer": result["answer"],
        "sources": " ".join(result["sources"]), "n_sources": len(result["sources"]),
        "retrieved": " ".join(result["retrieved"]),
    }
    if question["cat"] == "O":                 # should NOT be answered - nothing to check
        row["correct"] = None
        row["source_correct"] = None
        return row
    found = [bool(re.search(fact, result["answer"], re.IGNORECASE)) for fact in question["facts"]]
    row["correct"] = all(found)
    row["facts_found"] = sum(found) / len(found)
    row["source_correct"] = any(s in question["gold"] for s in result["sources"])
    return row


def run_questions(rag, questions):
    """Run every question once with threshold 0, keeping the confidence score.
    We apply the real threshold afterwards, so we can try many thresholds without
    re-running the system."""
    rag.threshold = 0.0
    return pd.DataFrame([check_answer(q, rag.answer(q["q"])) for q in questions])


# ---------------------------------------------------------------- applying a threshold

def apply_threshold(df, threshold):
    df = df.copy()
    df["answered"] = df["answered_raw"] & (df["confidence"] >= threshold)

    def score(row):
        if row["type"] == "O":
            return -1 if row["answered"] else 1
        if not row["answered"]:
            return 0
        return 1 if row["correct"] else -1

    df["score"] = df.apply(score, axis=1)
    return df


def percent(series):
    """Percentage of True values, or None if there is nothing to measure."""
    return round(100 * float(series.mean()), 1) if len(series) else None


def average(series):
    return round(float(series.mean()), 2) if len(series) else None


def metrics(df):
    """The dashboard numbers for a set of questions (after the threshold is applied)."""
    answerable = df[df["type"] != "O"]
    should_refuse = df[df["type"] == "O"]
    answered = answerable[answerable["answered"]]
    answer_rate = percent(answerable["answered"])
    return {
        "questions": len(df),
        "answer_rate": answer_rate,
        "unanswered_pct": None if answer_rate is None else round(100 - answer_rate, 1),
        "accuracy_on_answered": percent(answered["correct"].astype(bool)),
        "facts_found_on_answered": percent(answered["facts_found"]) if "facts_found" in answered else None,
        "refusal_rate_out_of_kb": percent(~should_refuse["answered"]),
        "source_accuracy": percent(answered["source_correct"].astype(bool)),
        "avg_confidence_when_answering": average(df[df["answered"]]["confidence"]),
        "avg_sources_per_answer": average(df[df["answered"]]["n_sources"]),
        "avg_score": round(float(df["score"].mean()), 3),
    }


def choose_threshold(dev_df):
    """Try thresholds from 0 to 25 in steps of 0.5 and keep the best average score."""
    rows = []
    for t in [x / 2 for x in range(0, 51)]:
        m = metrics(apply_threshold(dev_df, t))
        rows.append({"threshold": t, **m})
    curve = pd.DataFrame(rows).fillna(0)
    best = curve.loc[curve["avg_score"].idxmax(), "threshold"]    # first (lowest) best threshold
    return float(best), curve


# ---------------------------------------------------------------- retrieval quality (like Walert)

def retrieval_scores(rag, questions):
    """NDCG@3 and Recall@3 for the search step alone, using ranx like Walert's eval.py."""
    answerable = [q for q in questions if q["cat"] != "O"]
    qrels = Qrels({q["qid"]: q["gold"] for q in answerable})
    run = Run({q["qid"]: {p["id"]: s for p, s in rag.retrieve(q["q"], k=10)} for q in answerable})
    return {k: round(float(v), 3) for k, v in ranx_evaluate(qrels, run, ["ndcg@3", "recall@3"]).items()}


# ---------------------------------------------------------------- fairness

def fairness(df, questions):
    """Accuracy for the same 12 questions asked three ways."""
    pairs = {q["qid"]: q["pair"] for q in questions if q.get("pair")}
    plain_ids = sorted(set(pairs.values()))
    is_right = (df["answered"] & (df["correct"] == True)).set_axis(df["qid"])   # noqa: E712
    out = {"plain": percent(is_right[plain_ids])}
    for letter, name in [("L", "legal"), ("C", "colloquial")]:
        ids = [q for q in pairs if q.startswith(letter)]
        out[name] = percent(is_right[ids])
    return out


# ---------------------------------------------------------------- no-RAG comparison (needs Ollama)

def closed_book_accuracy(questions):
    """Ask the same LLM WITHOUT giving it our passages. This shows what RAG adds."""
    import requests
    right, stale, total = 0, 0, 0
    for q in questions:
        if q["cat"] not in ("K", "I"):
            continue
        reply = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": f"You help renters in Victoria, Australia. "
                                                     f"Answer in 1-3 sentences: {q['q']}"}]},
            timeout=300).json()["message"]["content"]
        total += 1
        right += all(re.search(f, reply, re.IGNORECASE) for f in q["facts"])
        # old rule: 60 days' notice for rent increases / sale (it is 90 days since Nov 2025)
        if q["qid"] in ("K01", "K15") and re.search(r"\b60 days\b", reply):
            stale += 1
    return {"accuracy": round(100 * right / total, 1), "questions": total,
            "used_old_60_day_rule": stale}


# ---------------------------------------------------------------- chart

def plot_threshold_curve(curve, chosen, path):
    curve = curve[curve["threshold"] <= 20]      # above 20 almost nothing is answered
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(curve["threshold"], curve["answer_rate"], label="Answer rate (answerable questions)")
    ax.plot(curve["threshold"], curve["accuracy_on_answered"], label="Accuracy on answered")
    ax.plot(curve["threshold"], curve["refusal_rate_out_of_kb"], label="Refusal rate (should-not-answer)")
    ax.axvline(chosen, color="grey", linestyle="--")
    ax.text(chosen + 0.3, 5, f"chosen = {chosen}", color="grey")
    ax.set_xlabel("Confidence threshold (BM25 score of the best passage)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Tuning the threshold on the dev set")
    ax.legend(fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true", help="use the local LLM to write answers")
    args = parser.parse_args()

    rag = RenterRAG(use_ollama=args.ollama)
    dev = load_questions("dev_questions.yaml")
    test = load_questions("test_questions.yaml")

    # 1. tune the threshold on the DEV questions
    dev_df = run_questions(rag, dev)
    threshold, curve = choose_threshold(dev_df)
    curve.to_csv(RESULTS / "threshold_curve_dev.csv", index=False)
    plot_threshold_curve(curve, threshold, RESULTS / "threshold_curve_dev.png")

    # 2. report on the TEST questions with that threshold fixed
    test_df = apply_threshold(run_questions(rag, test), threshold)
    test_df.to_csv(RESULTS / "test_results_per_question.csv", index=False)

    by_type = {t: metrics(g) for t, g in test_df.groupby("type")}
    summary = {
        "generator": f"Ollama {OLLAMA_MODEL}" if args.ollama else "extractive (no LLM)",
        "threshold": threshold,
        "test_overall": metrics(test_df),
        "test_by_type": by_type,
        "retrieval_test": retrieval_scores(rag, test),
        "fairness_accuracy": fairness(test_df, test),
        "test_curve": [{"threshold": t, **metrics(apply_threshold(test_df, t))}
                       for t in [x / 2 for x in range(0, 51)]],
    }
    if args.ollama:
        summary["no_rag_llm"] = closed_book_accuracy(test)

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    # 3. print a short report
    o = summary["test_overall"]
    print(f"\nGenerator: {summary['generator']}   threshold (tuned on dev): {threshold}")
    print(f"Answer rate {o['answer_rate']}%  |  unanswered {o['unanswered_pct']}%  |  "
          f"accuracy on answered {o['accuracy_on_answered']}%  |  "
          f"refused should-not-answer {o['refusal_rate_out_of_kb']}%")
    print(f"Source accuracy {o['source_accuracy']}%  |  avg confidence {o['avg_confidence_when_answering']}"
          f"  |  avg sources {o['avg_sources_per_answer']}")
    print(f"Retrieval: {summary['retrieval_test']}   Fairness (accuracy %): {summary['fairness_accuracy']}")
    if args.ollama:
        print(f"LLM without RAG: {summary['no_rag_llm']}")
    print(f"\nSaved results/ summary.json, test_results_per_question.csv, threshold_curve_dev.png")


if __name__ == "__main__":
    main()

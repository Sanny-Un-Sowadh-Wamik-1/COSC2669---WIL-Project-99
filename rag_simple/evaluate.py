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
  python evaluate.py              # offline (extractive answers)        -> results_offline/
  python evaluate.py --ollama     # answers written by the local LLM     -> results/
                                  #   + a "no RAG" comparison
Once both runs exist, it also makes the offline-vs-LLM table and chart (in results/).
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

from rag import RenterRAG, OLLAMA_URL, OLLAMA_MODEL, TOP_K

HERE = Path(__file__).parent
OFFLINE_DIR = HERE / "results_offline"   # python evaluate.py           saves here
LLM_DIR = HERE / "results"               # python evaluate.py --ollama  saves here


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

def check_answer(question, result, rag):
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
        row["supported"] = None
        return row
    found = [bool(re.search(fact, result["answer"], re.IGNORECASE)) for fact in question["facts"]]
    row["correct"] = all(found)
    row["facts_found"] = sum(found) / len(found)
    row["source_correct"] = any(s in question["gold"] for s in result["sources"])
    # Faithfulness ("backed by its sources"): the answer cites at least one passage, AND every key
    # fact that appears in the answer can also be found in the passages it cited.
    # So a reader who clicks the source can check the answer for themselves.
    cited_text = " ".join(rag.by_id[s]["text"] for s in result["sources"])
    facts_in_answer = [fact for fact, hit in zip(question["facts"], found) if hit]
    row["supported"] = bool(result["sources"]) and all(
        re.search(fact, cited_text, re.IGNORECASE) for fact in facts_in_answer)
    return row


def run_questions(rag, questions):
    """Run every question once with threshold 0, keeping the confidence score.
    We apply the real threshold afterwards, so we can try many thresholds without
    re-running the system."""
    rag.threshold = 0.0
    return pd.DataFrame([check_answer(q, rag.answer(q["q"]), rag) for q in questions])


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
        "supported_by_sources": percent(answered["supported"].astype(bool)),
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


def retrieval_depth(rag, questions):
    """How often is a correct passage among the top k search results?  (k = 1, 3, 5, 10)
    This is how we chose to give the generator the top 3 passages: more passages find the
    answer more often, but also give the generator more text that is not about the question."""
    answerable = [q for q in questions if q["cat"] != "O"]
    found = {k: 0 for k in (1, 3, 5, 10)}
    for q in answerable:
        ids = [p["id"] for p, _ in rag.retrieve(q["q"], k=10)]
        for k in found:
            found[k] += any(i in q["gold"] for i in ids[:k])
    return {str(k): round(100 * n / len(answerable), 1) for k, n in found.items()}


# ---------------------------------------------------------------- fairness

def fairness(df, questions):
    """Accuracy for the same 12 questions asked three ways."""
    pairs = {q["qid"]: q["pair"] for q in questions if q.get("pair")}
    plain_ids = sorted(set(pairs.values()))
    is_right = (df["answered"] & (df["correct"] == True)).set_axis(df["qid"])   # noqa: E712
    out = {"plain": percent(is_right[plain_ids]), "questions_per_wording": len(plain_ids)}
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
        # old rule: 60 days' notice (it is 90 days since Nov 2025). We check every question
        # whose correct answer needs "90 days", and count replies that say "60 days" instead.
        if any("90 days" in f for f in q["facts"]) and re.search(r"\b60 days\b", reply):
            stale += 1
    return {"accuracy": round(100 * right / total, 1), "correct": right, "questions": total,
            "used_old_60_day_rule": stale}


def rag_fully_correct(df):
    """The same measure as closed_book_accuracy, but WITH RAG, so the two compare fairly:
    out of all Known + Inferred test questions, how many got a correct answer
    (saying "I don't know" counts as not correct here)."""
    ki = df[df["type"].isin(["K", "I"])]
    right = int((ki["answered"] & (ki["correct"] == True)).sum())   # noqa: E712
    return {"accuracy": round(100 * right / len(ki), 1), "correct": right, "questions": len(ki)}


# ---------------------------------------------------------------- chart

def plot_threshold_curve(dev_df, chosen, path):
    """A simple line chart: the total score on the dev set at each threshold.
    Score = +1 right answer, -1 wrong answer, 0 "I don't know",
            +1 refusing an off-topic question, -1 answering one.
    The highest point is the threshold we chose."""
    thresholds = [x / 2 for x in range(0, 41)]            # 0, 0.5, 1 ... 20
    scores = [int(apply_threshold(dev_df, t)["score"].sum()) for t in thresholds]
    best = scores[thresholds.index(chosen)]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(thresholds, scores, color="#9fb7b2", linewidth=2, marker="o", markersize=4)
    ax.plot([chosen], [best], "o", color="#0e7c70", markersize=11)          # the chosen threshold
    ax.annotate(f"Best: threshold {chosen:g}, score {best}", (chosen, best),
                xytext=(0, 12), textcoords="offset points", ha="center",
                color="#0e7c70", fontsize=10, weight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Confidence threshold (higher = stricter, says \"I don't know\" more)")
    ax.set_ylabel("Total score on the dev set")
    ax.set_title("Which threshold gives the best score? (+1 right, -1 wrong)")
    ax.set_ylim(min(scores) - 2, best + 4)                  # room for the label above the best point
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))   # whole numbers only (scores are counts)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- offline vs LLM

# (label shown in the table/chart, where to find the number inside summary.json)
METRICS = [
    ("Answer rate %",                     ["test_overall", "answer_rate"]),
    ("Accuracy on answered %",            ["test_overall", "accuracy_on_answered"]),
    ("Said \"I don't know\" when it should %", ["test_overall", "refusal_rate_out_of_kb"]),
    ("Cited a correct source %",          ["test_overall", "source_accuracy"]),
    ("Answer backed by its sources %",    ["test_overall", "supported_by_sources"]),
    ("Fully correct (known + inferred) %", ["rag_fully_correct", "accuracy"]),
    ("Fairness: plain wording %",         ["fairness_accuracy", "plain"]),
    ("Fairness: legal wording %",         ["fairness_accuracy", "legal"]),
    ("Fairness: slang wording %",         ["fairness_accuracy", "colloquial"]),
]
OTHER = [   # not percentages, so they go in the table only
    ("Threshold (tuned on dev)",          ["threshold"]),
    ("Avg sources per answer",            ["test_overall", "avg_sources_per_answer"]),
    ("Retrieval NDCG@3",                  ["retrieval_test", "ndcg@3"]),
]


def get(summary, path):
    """Follow a list of keys into the summary, e.g. ["test_overall", "answer_rate"]."""
    value = summary
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def compare_runs():
    """Offline vs LLM side by side. Runs automatically once BOTH runs exist.
    Every number is read from the two summary.json files (nothing typed in by hand)."""
    offline_file, llm_file = OFFLINE_DIR / "summary.json", LLM_DIR / "summary.json"
    if not (offline_file.exists() and llm_file.exists()):
        return   # only one run so far - nothing to compare yet
    offline = json.loads(offline_file.read_text())
    llm = json.loads(llm_file.read_text())

    # ---- table
    lines = [f"| Metric | Offline ({offline['generator']}) | LLM with RAG ({llm['generator']}) |",
             "|---|---|---|"]
    for label, path in METRICS + OTHER:
        lines.append(f"| {label} | {get(offline, path)} | {get(llm, path)} |")
    if "no_rag_llm" in llm:
        n = llm["no_rag_llm"]
        r = llm["rag_fully_correct"]
        lines.append("")
        lines.append(f"RAG vs no RAG (same LLM, known + inferred questions): with RAG {r['correct']} of "
                     f"{r['questions']} fully correct ({r['accuracy']}%), without RAG {n['correct']} of "
                     f"{n['questions']} ({n['accuracy']}%). Without RAG it gave the old 60-day rule "
                     f"{n['used_old_60_day_rule']} time(s).")
    table = "\n".join(lines)
    (LLM_DIR / "comparison_table.md").write_text(table + "\n")
    print("\n" + table)

    # ---- chart: two bars per metric (offline, LLM)
    labels = [label.replace(" %", "") for label, _ in METRICS]
    off_values = [get(offline, p) or 0 for _, p in METRICS]
    llm_values = [get(llm, p) or 0 for _, p in METRICS]
    y = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([i - 0.2 for i in y], off_values, height=0.38, color="#9fb7b2", label="Offline (copied sentences)")
    ax.barh([i + 0.2 for i in y], llm_values, height=0.38, color="#0e7c70", label="LLM with RAG")
    for i in y:   # write the number at the end of each bar
        ax.text(off_values[i] + 1, i - 0.2, f"{off_values[i]:g}", va="center", fontsize=8)
        ax.text(llm_values[i] + 1, i + 0.2, f"{llm_values[i]:g}", va="center", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()                      # first metric at the top
    ax.set_xlim(0, 110)
    ax.set_xlabel("% of test questions")
    ax.set_title("Offline vs LLM on the same test questions")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(LLM_DIR / "comparison_chart.png", dpi=160)
    plt.close(fig)
    print("\nSaved results/comparison_table.md and results/comparison_chart.png")



# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true", help="use the local LLM to write answers")
    args = parser.parse_args()

    RESULTS = LLM_DIR if args.ollama else OFFLINE_DIR
    RESULTS.mkdir(exist_ok=True)
    rag = RenterRAG(use_ollama=args.ollama)
    dev = load_questions("dev_questions.yaml")
    test = load_questions("test_questions.yaml")

    # 1. tune the threshold on the DEV questions
    dev_df = run_questions(rag, dev)
    threshold, curve = choose_threshold(dev_df)
    curve.to_csv(RESULTS / "threshold_curve_dev.csv", index=False)
    plot_threshold_curve(dev_df, threshold, RESULTS / "threshold_curve_dev.png")

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
        "retrieval_depth": retrieval_depth(rag, test),
        "top_k_used": TOP_K,
        "fairness_accuracy": fairness(test_df, test),
        "rag_fully_correct": rag_fully_correct(test_df),
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
    print(f"Answer backed by its sources {o['supported_by_sources']}%  |  "
          f"correct passage in top 1/3/5/10: {summary['retrieval_depth']}")
    print(f"Retrieval: {summary['retrieval_test']}   Fairness (accuracy %): {summary['fairness_accuracy']}")
    if args.ollama:
        print(f"LLM without RAG: {summary['no_rag_llm']}")
    print(f"With RAG (same measure): {summary['rag_fully_correct']}")
    print(f"\nSaved {RESULTS.name}/ summary.json, test_results_per_question.csv, threshold_curve_dev.png")

    # 4. if both runs exist now, put them side by side
    compare_runs()


if __name__ == "__main__":
    main()

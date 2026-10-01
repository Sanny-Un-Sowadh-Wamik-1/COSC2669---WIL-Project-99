"""Experiment 2 + final evaluation.

On DEV (every decision is made here):
  a. every candidate LLM answers every dev question once, with all gates switched off;
  b. each candidate confidence signal is scored by AUROC (answerable vs unanswerable);
  c. the topic gate threshold is set just below the least on-topic answerable dev question;
  d. the answer threshold is swept and scored with the utility below; the best LLM + threshold
     becomes the operating point (results/operating_point.json).

On TEST (run once, with the operating point fixed):
  dashboard metrics, per-question results, a no-retrieval LLM baseline, error analysis,
  fairness by question style, and a blank rubric sheet for two human raters.

Utility (per question):            answered correctly +1 | partly 0 | wrongly -1 | abstained 0
                                    unanswerable: abstained +1 | answered -1
A wrong answer with a citation looks trustworthy, so it costs as much as a correct one earns.

Run:  python run_eval.py            (needs Ollama; generations are cached in cache/generations.json)
"""

import csv
import hashlib
import json
import sys

import yaml

from config import CACHE, QUESTIONS, RESULTS, operating_point
from metrics import auroc, facts_found, mean
from pipeline import CLOSED_BOOK_PROMPT, SYSTEM_PROMPT, RenterRAG, chat, format_sources, says_dont_know

LLMS = ["qwen2.5:3b", "llama3.2:3b", "gemma3:4b"]
SIGNALS = ["top_cosine", "margin", "top_bm25", "agreement"]
GEN_CACHE = CACHE / "generations.json"
_gen_cache = json.loads(GEN_CACHE.read_text()) if GEN_CACHE.exists() else {}


def cached_chat(model, system, user):
    key = f"{model}|" + hashlib.sha1((system + user).encode()).hexdigest()
    if key not in _gen_cache:
        _gen_cache[key] = chat(model, system, user)
        GEN_CACHE.write_text(json.dumps(_gen_cache, indent=0))
    return _gen_cache[key]


def load(split):
    return yaml.safe_load((QUESTIONS / f"{split}.yaml").read_text())


# ---------------------------------------------------------------------------- running
def run(rag, questions, model):
    """Answer every question with all gates open; record what each gate *would* decide."""
    rows = []
    for n, q in enumerate(questions, 1):
        print(f"\r  {model}: {n}/{len(questions)}", end="", flush=True)
        r = rag.prepare(q["q"])
        row = {"qid": q["qid"], "group": q["group"], "style": q.get("style", "-"), "question": q["q"],
               "gold": q.get("gold", {}), "facts": q.get("facts", []), "reference": q.get("answer", ""),
               "jurisdiction_refused": r["reason"] == "other jurisdiction", "signals": r["signals"],
               "retrieved": [c["id"] for c in r["retrieved"]],
               "retrieved_units": sorted({u for c in r["retrieved"] for u in c["units"]}),
               "context_facts": facts_found(" ".join(c["text"] for c in r["retrieved"]), q.get("facts", []))}
        if not row["jurisdiction_refused"]:
            prompt = f"Sources:\n{format_sources(r['retrieved'])}\n\nQuestion: {r['question']}"
            done = rag.finish(r, cached_chat(model, SYSTEM_PROMPT, prompt))
            row.update(llm_answered=done["answered"], llm_reason=done["reason"],
                       answer=done.get("raw_answer", done["answer"]),
                       sources=[c["citation"] for c in done["sources"]],
                       source_units=sorted({u for c in done["sources"] for u in c["units"]}))
        else:
            row.update(llm_answered=False, llm_reason="", answer="", sources=[], source_units=[])
        row["facts_score"] = facts_found(row["answer"], row["facts"]) if row["llm_answered"] else None
        rows.append(row)
    print()
    return rows


def decide(row, signal, threshold, topic_threshold):
    """Apply the gates to a pre-computed row. Returns (answered, reason)."""
    if row["jurisdiction_refused"]:
        return False, "other jurisdiction"
    if row["signals"]["top_cosine_all"] < topic_threshold:
        return False, "off topic"
    if row["signals"][signal] < threshold:
        return False, "low confidence"
    if not row["llm_answered"]:
        return False, row["llm_reason"]
    return True, "answered"


def label(row, answered):
    if row["group"] != "answerable":
        return "wrongly answered" if answered else "correct abstention"
    if not answered:
        return "false refusal"
    return {1.0: "correct"}.get(row["facts_score"], "partly correct" if row["facts_score"] > 0 else "incorrect")


UTILITY = {"correct": 1, "partly correct": 0, "incorrect": -1, "false refusal": 0,
           "correct abstention": 1, "wrongly answered": -1}


def summarise(rows, signal, threshold, topic_threshold):
    ans = [r for r in rows if r["group"] == "answerable"]
    una = [r for r in rows if r["group"] != "answerable"]
    out = []
    for r in rows:
        answered, reason = decide(r, signal, threshold, topic_threshold)
        out.append({**r, "answered": answered, "reason": reason, "label": label(r, answered)})
    answered_ans = [r for r in out if r["group"] == "answerable" and r["answered"]]
    answered_all = [r for r in out if r["answered"]]
    pct = lambda xs, n: round(100 * len(xs) / n, 1) if n else None  # noqa: E731
    return out, {
        "questions": len(rows), "answerable": len(ans), "unanswerable": len(una),
        "answer_rate": pct(answered_ans, len(ans)),
        "accuracy_on_answered": pct([r for r in answered_ans if r["label"] == "correct"], len(answered_ans)),
        "partly_correct_on_answered": pct([r for r in answered_ans if r["label"] == "partly correct"], len(answered_ans)),
        "correct_abstention_rate": pct([r for r in out if r["group"] != "answerable" and not r["answered"]], len(una)),
        "false_refusal_rate": pct([r for r in out if r["label"] == "false refusal"], len(ans)),
        "source_accuracy": pct([r for r in answered_ans if set(r["source_units"]) & set(r["gold"])], len(answered_ans)),
        "avg_sources_per_answer": round(mean([len(r["sources"]) for r in answered_all]) or 0, 2),
        "avg_confidence_answered": round(mean([r["signals"][signal] for r in answered_all]) or 0, 3),
        "utility": round(mean([UTILITY[r["label"]] for r in out]), 3),
    }


def thresholds_for(rows, signal):
    values = sorted({round(r["signals"][signal], 3) for r in rows})
    return [values[0] - 1e-6] + values


# ---------------------------------------------------------------------------- error analysis
def error_category(row, units):
    """Why did this question go wrong? (only called for wrong / partly / false refusal / wrongly answered)"""
    if row["group"] != "answerable":
        return "threshold: over-confident on unanswerable" if row["llm_answered"] else "gate passed, LLM still answered"
    best = max(row["gold"].values())
    gold_hit = any(row["gold"].get(u) == best for u in row["retrieved_units"])
    if not gold_hit:
        return "retrieval miss"
    full_text = " ".join(units[u]["text"] for u in row["gold"] if u in units and row["gold"][u] == best)
    if row["context_facts"] < 1 and facts_found(full_text, row["facts"]) > row["context_facts"]:
        return "chunking: fact cut off from the retrieved chunk"
    if row["label"] == "false refusal":
        if row["reason"] in ("off topic", "low confidence", "other jurisdiction"):
            return "threshold: refused although the answer was retrieved"
        return f"generation: {row['reason']}"
    return "generation: wrong or incomplete answer from correct context"


# ---------------------------------------------------------------------------- main
def main(models):
    RESULTS.mkdir(exist_ok=True)
    choice = json.loads((RESULTS / "retrieval_choice.json").read_text())["chosen"]
    settings = {**operating_point(), **choice, "threshold": -1e9, "topic_threshold": -1e9}
    if settings["embed_model"] == "-":
        settings["embed_model"] = "nomic-embed-text"
    rag = RenterRAG(settings)
    dev, test = load("dev"), load("test")

    # a-d. DEV
    dev_runs = {m: run(rag, dev, m) for m in models}
    any_rows = next(iter(dev_runs.values()))
    ans_rows = [r for r in any_rows if r["group"] == "answerable"]
    una_rows = [r for r in any_rows if r["group"] != "answerable" and not r["jurisdiction_refused"]]
    signal_auroc = {s: round(auroc([r["signals"][s] for r in ans_rows], [r["signals"][s] for r in una_rows]), 3)
                    for s in SIGNALS}
    signal = max(signal_auroc, key=signal_auroc.get)
    topic_threshold = round(min(r["signals"]["top_cosine_all"] for r in ans_rows) - 0.02, 3)
    print("confidence AUROC on dev:", signal_auroc, "-> using", signal, "| topic threshold", topic_threshold)

    curves, best = {}, None
    for model, rows in dev_runs.items():
        curve = []
        for t in thresholds_for(rows, signal):
            _, s = summarise(rows, signal, t, topic_threshold)
            curve.append({"threshold": round(t, 4), **s})
        curves[model] = curve
        top = max(curve, key=lambda c: (c["utility"], c["answer_rate"] or 0))
        print(f"  {model:14s} best dev utility {top['utility']:.3f} at threshold {top['threshold']:.3f} "
              f"(answer rate {top['answer_rate']}%, accuracy {top['accuracy_on_answered']}%)")
        if best is None or top["utility"] > best[1]["utility"]:
            best = (model, top)
    model, point = best
    op = {**choice, "embed_model": settings["embed_model"], "top_k": settings["top_k"], "llm": model,
          "confidence": signal, "threshold": point["threshold"], "topic_threshold": topic_threshold}
    (RESULTS / "operating_point.json").write_text(json.dumps(op, indent=2))
    print("operating point:", op)

    # TEST (once)
    test_rows = run(rag, test, model)
    test_out, test_summary = summarise(test_rows, signal, point["threshold"], topic_threshold)
    units = json.loads((CACHE / "units.json").read_text())
    for r in test_out:
        r["error"] = error_category(r, units) if r["label"] not in ("correct", "correct abstention") else ""
    by_group = {g: summarise([r for r in test_rows if r["group"] == g], signal, point["threshold"], topic_threshold)[1]
                for g in ("answerable", "near_miss", "off_topic")}
    by_style = {s: summarise([r for r in test_rows if r.get("style") == s], signal, point["threshold"], topic_threshold)[1]
                for s in ("plain", "scenario", "legal", "slang")}

    # no-retrieval baseline on TEST
    print("  closed-book baseline ...")
    closed = []
    for q in test:
        text = cached_chat(model, CLOSED_BOOK_PROMPT, q["q"])
        answered = not says_dont_know(text)
        score = facts_found(text, q.get("facts", [])) if answered and q["group"] == "answerable" else None
        closed.append({"qid": q["qid"], "group": q["group"], "answer": text, "answered": answered, "facts_score": score})
    c_ans = [c for c in closed if c["group"] == "answerable"]
    c_una = [c for c in closed if c["group"] != "answerable"]
    c_answered = [c for c in c_ans if c["answered"]]
    closed_summary = {
        "answer_rate": round(100 * len(c_answered) / len(c_ans), 1),
        "accuracy_on_answered": round(100 * sum(c["facts_score"] == 1 for c in c_answered) / max(len(c_answered), 1), 1),
        "correct_abstention_rate": round(100 * sum(not c["answered"] for c in c_una) / len(c_una), 1),
        "avg_sources_per_answer": 0.0,
    }

    errors = {}
    for r in test_out:
        if r["error"]:
            errors[r["error"]] = errors.get(r["error"], 0) + 1

    summary = {"operating_point": op, "signal_auroc_dev": signal_auroc, "dev_curves": curves,
               "dev_point": point, "test": test_summary, "test_by_group": by_group, "test_by_style": by_style,
               "closed_book_test": closed_summary, "errors_test": dict(sorted(errors.items(), key=lambda x: -x[1])),
               "retrieval": json.loads((RESULTS / "retrieval_choice.json").read_text())}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    write_per_question(test_out, closed, signal)
    plot_curve(curves[model], point, signal, model)
    print(json.dumps({"test": test_summary, "closed_book": closed_summary, "errors": summary["errors_test"]}, indent=2))


def write_per_question(rows, closed, signal):
    closed = {c["qid"]: c for c in closed}
    fields = ["qid", "group", "style", "question", "answered", "label", "reason", "error", "confidence",
              "answer", "sources", "gold", "reference", "closed_book_answer"]
    with (RESULTS / "test_per_question.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"qid": r["qid"], "group": r["group"], "style": r["style"], "question": r["question"],
                        "answered": r["answered"], "label": r["label"], "reason": r["reason"], "error": r["error"],
                        "confidence": round(r["signals"][signal], 3) if r["signals"] else "",
                        "answer": r["answer"], "sources": " | ".join(r["sources"]),
                        "gold": " ".join(r["gold"]), "reference": r["reference"],
                        "closed_book_answer": closed[r["qid"]]["answer"]})
    # rubric sheet for two human raters (Step 7): only answered questions need a correctness grade
    with (RESULTS / "rubric_test.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question", "system_answer", "sources", "reference_answer", "auto_label",
                    "rater1 (correct/partly/incorrect)", "rater2 (correct/partly/incorrect)", "notes"])
        for r in rows:
            if r["answered"]:
                w.writerow([r["qid"], r["question"], r["answer"], " | ".join(r["sources"]), r["reference"],
                            r["label"], "", "", ""])


def plot_curve(curve, point, signal, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [c["threshold"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
    ax.plot(t, [c["answer_rate"] or 0 for c in curve], label="answer rate (answerable)", color="#2563eb")
    ax.plot(t, [c["accuracy_on_answered"] or 0 for c in curve], label="accuracy on answered", color="#16a34a")
    ax.plot(t, [c["correct_abstention_rate"] or 0 for c in curve], label="correct abstention (unanswerable)", color="#9333ea")
    ax.axvline(point["threshold"], color="#64748b", linestyle="--", linewidth=1, label=f"chosen {point['threshold']:.3f}")
    ax.set_xlabel(f"confidence threshold ({signal})")
    ax.set_ylabel("%")
    ax.set_title(f"Threshold sweep on DEV ({model})")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "threshold_curve_dev.png")


if __name__ == "__main__":
    main(sys.argv[1:] or LLMS)

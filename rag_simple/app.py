"""
STEP 4 - The web app (prototype + dashboard)

Two pages:
  /            Ask a question -> see the answer, the confidence, and the passages it came from
  /dashboard   The evaluation results from evaluate.py, with a threshold slider

Run:  python app.py        then open  http://127.0.0.1:5000
(Run evaluate.py first so the dashboard has numbers and the tuned threshold.)
Add --ollama to let the local LLM write the answers.
"""

import argparse
import json
from pathlib import Path

from flask import Flask, render_template, request

from rag import RenterRAG, load_tuned_threshold

app = Flask(__name__, static_folder="results")   # lets the page show the chart from results/
SUMMARY_FILE = Path(__file__).parent / "results" / "summary.json"
rag = None   # created in __main__

EXAMPLES = [
    "How much notice does my landlord have to give before putting the rent up?",
    "Can my landlord charge me a pet bond?",
    "If my landlord won't fix an urgent repair, can I pay for it myself?",
    "Can I be evicted at the end of my lease for no reason?",
    "What is the maximum bond in New South Wales?",
]


@app.route("/", methods=["GET", "POST"])
def ask():
    result, cited = None, []
    if request.method == "POST":
        question = request.form.get("question", "").strip()[:500]
        if question:
            result = rag.answer(question)
            cited = [rag.by_id[s] for s in result["sources"]]
    return render_template("ask.html", result=result, cited=cited, examples=EXAMPLES,
                           threshold=rag.threshold)


@app.route("/dashboard")
def dashboard():
    if not SUMMARY_FILE.exists():
        return "Run  python evaluate.py  first.", 404
    summary = json.loads(SUMMARY_FILE.read_text())
    return render_template("dashboard.html", s=summary, curve_json=json.dumps(summary["test_curve"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    rag = RenterRAG(use_ollama=args.ollama, threshold=load_tuned_threshold())
    app.run(port=args.port)

"""
STEP 4 - The web app (prototype + dashboard)

Two pages:
  /            The chatbot: ask a question -> answer, confidence and the sources it came from
  /dashboard   The evaluation results from evaluate.py, with a threshold slider

Run:  python app.py        then open  http://127.0.0.1:5000
(Run evaluate.py first so the dashboard has numbers and the tuned threshold.)
Add --ollama to let the local LLM write the answers.
"""

import argparse
import json
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from rag import RenterRAG, load_tuned_threshold

app = Flask(__name__)
HERE = Path(__file__).parent
# The two evaluation runs the Dashboard can show
RUNS = {"offline": HERE / "results_offline",  # made by: python evaluate.py
        "llm": HERE / "results"}              # made by: python evaluate.py --ollama
rag = None   # created in __main__

EXAMPLES = [
    "How much notice does my landlord have to give before putting the rent up?",
    "Can my landlord charge me a pet bond?",
    "If my landlord won't fix an urgent repair, can I pay for it myself?",
    "Can I be evicted at the end of my lease for no reason?",
    "What is the maximum bond in New South Wales?",
]


@app.route("/")
def chat():
    """The chatbot page. The page itself is plain HTML; its JavaScript sends each
    message to /api/chat and shows the reply as a chat bubble."""
    return render_template("chat.html", examples=EXAMPLES, threshold=rag.threshold)


@app.route("/api/chat", methods=["POST"])
def chat_api():
    """Answer ONE message. Each message is answered on its own (no memory of earlier
    messages), so the chatbot behaves exactly like the system we evaluated."""
    question = request.get_json().get("message", "").strip()[:500]
    if not question:
        return jsonify({"error": "empty message"}), 400
    result = rag.answer(question)
    sources = [{"id": s, "heading": rag.by_id[s]["heading"], "url": rag.by_id[s]["url"]}
               for s in result["sources"]]
    return jsonify({"answer": result["answer"], "answered": result["answered"],
                    "confidence": result["confidence"], "sources": sources})


@app.route("/dashboard")
def dashboard():
    """Shows one run: /dashboard?run=offline or /dashboard?run=llm.
    Without ?run= it shows the LLM run if it exists, otherwise the offline run."""
    default = "llm" if (RUNS["llm"] / "summary.json").exists() else "offline"
    run = request.args.get("run", default)
    if run not in RUNS:
        abort(404)
    summary_file = RUNS[run] / "summary.json"
    if not summary_file.exists():
        return "No results for this run yet. Run  python evaluate.py  (and/or  python evaluate.py --ollama)  first.", 404
    summary = json.loads(summary_file.read_text())
    # buttons at the top of the page, one for each run that exists
    runs = {name: json.loads((folder / "summary.json").read_text())["generator"]
            for name, folder in RUNS.items() if (folder / "summary.json").exists()}
    # the side-by-side chart only exists once both runs have been done
    has_comparison = (RUNS["llm"] / "comparison_chart.png").exists()
    return render_template("dashboard.html", s=summary, run=run, runs=runs,
                           curve_json=json.dumps(summary["test_curve"]), has_comparison=has_comparison)


@app.route("/files/<run>/<path:filename>")
def result_file(run, filename):
    """Sends a chart picture from results/ or results_offline/ to the page."""
    if run not in RUNS:
        abort(404)
    return send_from_directory(RUNS[run], filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    # use the threshold tuned for the same kind of run (LLM or offline)
    folder = RUNS["llm"] if args.ollama else RUNS["offline"]
    rag = RenterRAG(use_ollama=args.ollama, threshold=load_tuned_threshold(folder))
    app.run(port=args.port)

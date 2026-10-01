"""Web app: Ask page (streams the answer as it is written) + Dashboard (precomputed results).

Run:  python app.py        then open http://127.0.0.1:5000

Kept deliberately light so it stays fast on a laptop: the index and embeddings are loaded once at
start-up, the LLM is warmed up in the background, answers stream token by token, the dashboard only
reads files written by run_eval.py, and the pages use no animation or front-end framework.
"""

import csv
import json
import threading

from flask import Flask, Response, abort, render_template, request, send_from_directory

from config import RESULTS, operating_point
from pipeline import RenterRAG, chat

app = Flask(__name__)
rag = RenterRAG()

EXAMPLES = [
    "How much notice do I get before a rent increase?",
    "My heater broke and the agent won't answer. What can I do?",
    "Can my landlord enter without telling me?",
    "Can I be evicted at the end of my lease for no reason?",
    "Can the agent charge me a fee to pay rent online?",
]


def _warm_up():
    try:
        chat(rag.settings["llm"], "Reply OK.", "OK", num_predict=1)
    except Exception as exc:  # Ollama not running: the Ask page will show the error instead
        print("LLM warm-up failed:", exc)


threading.Thread(target=_warm_up, daemon=True).start()


def public(chunk, snippet=320):
    return {"citation": chunk["citation"], "heading": chunk["heading"], "url": chunk["url"],
            "date": chunk["date"], "kind": chunk["kind"],
            "text": chunk["text"][:snippet] + ("…" if len(chunk["text"]) > snippet else "")}


@app.get("/")
def ask():
    return render_template("ask.html", examples=EXAMPLES, settings=rag.settings, page="ask")


@app.post("/api/ask")
def api_ask():
    question = (request.get_json(silent=True) or {}).get("question", "")

    def events():
        try:
            for kind, payload in rag.stream(question):
                if kind == "token":
                    yield json.dumps({"type": "token", "text": payload}) + "\n"
                else:
                    yield json.dumps({
                        "type": kind, "answered": payload["answered"], "reason": payload["reason"],
                        "answer": payload["answer"] if kind == "done" else "",
                        "confidence": payload["confidence"],
                        "sources": [public(c) for c in payload["sources"]],
                        "retrieved": [public(c, 160) for c in payload["retrieved"]] if kind == "done" else [],
                    }) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}. Is Ollama running?"}) + "\n"

    return Response(events(), mimetype="application/x-ndjson", headers={"X-Accel-Buffering": "no"})


_cache = {}


def _load(name, reader):
    path = RESULTS / name
    if not path.exists():
        return None
    stamp = path.stat().st_mtime
    if _cache.get(name, (None,))[0] != stamp:
        _cache[name] = (stamp, reader(path))
    return _cache[name][1]


@app.get("/dashboard")
def dashboard():
    summary = _load("summary.json", lambda p: json.loads(p.read_text()))
    rows = _load("test_per_question.csv", lambda p: list(csv.DictReader(p.open())))
    retrieval = _load("retrieval_dev.csv", lambda p: list(csv.DictReader(p.open())))
    group, outcome = request.args.get("group", ""), request.args.get("label", "")
    shown = [r for r in rows or [] if (not group or r["group"] == group) and (not outcome or r["label"] == outcome)]
    retrieval_top = sorted(retrieval or [], key=lambda r: -float(r["ndcg@5"]))
    return render_template("dashboard.html", s=summary, rows=shown, all_rows=rows or [], group=group,
                           label=outcome, retrieval=retrieval_top, op=operating_point(), page="dashboard")


@app.get("/results/<name>.png")
def result_image(name):
    if not (RESULTS / f"{name}.png").exists():
        abort(404)
    return send_from_directory(RESULTS, f"{name}.png", max_age=60)


@app.get("/about")
def about():
    return render_template("about.html", op=operating_point(), page="about")


if __name__ == "__main__":
    app.run(debug=False, threaded=True)

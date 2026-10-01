"""Web app: Ask page (streams the answer as it is written) + Dashboard (precomputed results).

Run:  python app.py        then open http://127.0.0.1:5050

Built to feel instant on a laptop:
  * the index, embeddings and BM25 tokens load from cache/ at start-up (~1 s);
  * both Ollama models (embedding + LLM) are warmed in the background, so the first question
    doesn't wait for a model to load;
  * the retrieved sources are sent to the browser as soon as they are found (~0.3 s), and the
    answer then streams in token by token;
  * repeated questions are answered from memory;
  * pages are server-rendered with no framework, web fonts or images to download; the dashboard
    only reads files written by run_eval.py and caches them until they change.
"""

import csv
import json
import os
import threading
import time
from collections import OrderedDict

from flask import Flask, Response, render_template, request

from config import RESULTS, ROOT, operating_point
from pipeline import RenterRAG, chat
from retrieval import embed

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 86400        # static files are versioned below
rag = RenterRAG()

EXAMPLES = [
    "How much notice do I get before a rent increase?",
    "My heater broke and the agent won't answer. What can I do?",
    "Can my landlord enter without telling me?",
    "Can I be evicted at the end of my lease for no reason?",
    "Can the agent charge me a fee to pay rent online?",
    "What must a rental have in the kitchen?",
]


def _warm_up():
    try:
        embed(["warm up"], rag.settings["embed_model"])
        rag.prepare(EXAMPLES[0])                          # touches BM25 + numpy paths once
        chat(rag.settings["llm"], "Reply OK.", "OK", num_predict=1)
    except Exception as exc:                            # Ollama not running: the Ask page shows the error
        print("Warm-up failed:", exc)


threading.Thread(target=_warm_up, daemon=True).start()


@app.context_processor
def asset_version():
    css = ROOT / "static" / "style.css"
    return {"css_version": int(css.stat().st_mtime)}


def public(chunk, snippet=360):
    text = chunk["text"]
    source = ("Act" if chunk["doc"] == "RTA" else "Regulations" if chunk["kind"] == "law" else "CAV guide")
    return {"citation": chunk["citation"], "heading": chunk["heading"], "url": chunk["url"], "date": chunk["date"],
            "source": source, "text": text[:snippet] + ("…" if len(text) > snippet else "")}


# ------------------------------------------------------------------------------------ ask
_answers = OrderedDict()            # question -> finished result, so repeats are instant
_MAX_CACHED = 300


def _key(question):
    return " ".join(question.lower().split())


@app.get("/")
def ask():
    return render_template("ask.html", examples=EXAMPLES, settings=rag.settings, page="ask")


@app.post("/api/ask")
def api_ask():
    question = (request.get_json(silent=True) or {}).get("question", "")
    start = time.perf_counter()
    ms = lambda: round((time.perf_counter() - start) * 1000)  # noqa: E731

    def done_event(result, cached=False):
        return json.dumps({"type": "done", "answered": result["answered"], "reason": result["reason"],
                           "answer": result["answer"], "confidence": result["confidence"],
                           "cited": [result["retrieved"].index(c) + 1 for c in result["sources"]],
                           "retrieved": [public(c) for c in result["retrieved"]], "ms": ms(),
                           "cached": cached}) + "\n"

    def events():
        key = _key(question)
        if key in _answers:
            _answers.move_to_end(key)
            yield done_event(_answers[key], cached=True)
            return
        try:
            for kind, payload in rag.stream(question):
                if kind == "token":
                    yield json.dumps({"type": "token", "text": payload}) + "\n"
                elif kind == "meta":
                    yield json.dumps({"type": "meta", "answered": payload["answered"], "reason": payload["reason"],
                                      "confidence": payload["confidence"], "ms": ms(),
                                      "retrieved": [public(c) for c in payload["retrieved"]]}) + "\n"
                else:
                    _answers[key] = payload
                    if len(_answers) > _MAX_CACHED:
                        _answers.popitem(last=False)
                    yield done_event(payload)
        except Exception as exc:
            yield json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}. Is Ollama running?"}) + "\n"

    return Response(events(), mimetype="application/x-ndjson",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})


# ------------------------------------------------------------------------------------ dashboard
_cache = {}


def _load(name, reader):
    path = RESULTS / name
    if not path.exists():
        return None
    stamp = path.stat().st_mtime
    if _cache.get(name, (None,))[0] != stamp:
        _cache[name] = (stamp, reader(path))
    return _cache[name][1]


def eval_progress():
    """Last progress line of a running run_eval.py, e.g. 'llama3.2:3b: 12/57'."""
    log = ROOT / "cache" / "eval_log.txt"
    if not log.exists():
        return ""
    lines = [x.strip() for x in log.read_text().replace("\r", "\n").splitlines() if x.strip()]
    return lines[-1] if lines else ""


def curve_svg(curve, chosen, width=760, height=260):
    """Threshold sweep as polylines (drawn by the browser: crisp, theme-aware, nothing to download)."""
    pad_l, pad_r, pad_t, pad_b = 40, 12, 12, 30
    xs = [c["threshold"] for c in curve if c["threshold"] > -1]
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    sx = lambda t: pad_l + (min(max(t, lo), hi) - lo) / ((hi - lo) or 1) * (width - pad_l - pad_r)  # noqa: E731
    sy = lambda v: pad_t + (1 - (v or 0) / 100) * (height - pad_t - pad_b)  # noqa: E731
    series = []
    for key, label, cls in (("answer_rate", "Answer rate", "s1"), ("accuracy_on_answered", "Accuracy on answered", "s2"),
                            ("correct_abstention_rate", "Correct abstention", "s3")):
        pts = " ".join(f"{sx(c['threshold']):.1f},{sy(c[key]):.1f}" for c in curve if c["threshold"] > -1)
        series.append({"points": pts, "label": label, "cls": cls})
    grid = [{"y": sy(v), "label": v} for v in (0, 25, 50, 75, 100)]
    ticks = [{"x": sx(lo + (hi - lo) * f), "label": f"{lo + (hi - lo) * f:.2f}"} for f in (0, .25, .5, .75, 1)]
    return {"w": width, "h": height, "series": series, "grid": grid, "ticks": ticks, "chosen_x": sx(chosen),
            "x0": pad_l, "x1": width - pad_r, "y1": height - pad_b}


@app.get("/dashboard")
def dashboard():
    summary = _load("summary.json", lambda p: json.loads(p.read_text()))
    rows = _load("test_per_question.csv", lambda p: list(csv.DictReader(p.open())))
    retrieval = _load("retrieval_dev.csv", lambda p: sorted(csv.DictReader(p.open()), key=lambda r: -float(r["ndcg@5"])))
    choice = _load("retrieval_choice.json", lambda p: json.loads(p.read_text()))
    group, outcome = request.args.get("group", ""), request.args.get("label", "")
    shown = [r for r in rows or [] if (not group or r["group"] == group) and (not outcome or r["label"] == outcome)]
    op = operating_point()
    chart = None
    if summary and op["llm"] in summary.get("dev_curves", {}):
        chart = curve_svg(summary["dev_curves"][op["llm"]], op["threshold"])
    best_per_llm = {m: max(c, key=lambda x: (x["utility"], x["answer_rate"] or 0))
                    for m, c in (summary or {}).get("dev_curves", {}).items()}
    return render_template("dashboard.html", s=summary, rows=shown, all_rows=rows or [], group=group, label=outcome,
                           retrieval=retrieval or [], choice=choice, progress=eval_progress(), op=op, chart=chart,
                           best_per_llm=best_per_llm, page="dashboard")


@app.get("/about")
def about():
    return render_template("about.html", op=operating_point(), page="about")


if __name__ == "__main__":
    # 5050 by default: port 5000 is taken by AirPlay Receiver on macOS
    app.run(port=int(os.environ.get("PORT", 5050)), debug=False, threaded=True)

# Victorian Renter Rights Assistant (v2): Group 99, COSC2669 WIL

A local RAG assistant that answers questions about **renting a home in Victoria** from the
**Residential Tenancies Act 1997 (Vic)**, its **Regulations** and **Consumer Affairs Victoria** guidance,
and cites the exact section. It declines questions about other states, off-topic questions and questions its sources
don't cover. It comes with a test-driven evaluation and a simple dashboard.

> General information only, not legal advice. See `templates/about.html` and `docs/METHODOLOGY.md` for the limits.

## Run it

Needs Python 3.10+ and [Ollama](https://ollama.com) running locally.

```bash
cd renter_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull mxbai-embed-large && ollama pull qwen2.5:3b     # + nomic-embed-text, llama3.2:3b to rerun the comparisons

python fetch_sources.py    # 1. download the authorised Act + Regulations (not committed, see sources.yaml)
python ingest.py           # 2. parse + chunk -> cache/ (about 10 s)
python run_retrieval.py    # 3. retrieval experiment on dev -> results/retrieval_*.csv (embeds once, then about 15 s)
python run_eval.py         # 4. LLM / confidence / threshold on dev, then test once -> results/ (slow: about 1 h on an 8 GB M1)
python app.py              # 5. http://127.0.0.1:5050  -> Ask, Dashboard, About
pytest -q                  # unit tests for metrics, chunking and gates
```

The app works as soon as steps 1–2 are done, using the defaults in `config.py`. Steps 3–4 replace those defaults with the
settings chosen by experiment (`results/operating_point.json`).

## What's where

| Step in our plan | File |
|---|---|
| 2 Knowledge base, sources, terms of use | `sources.yaml`, `fetch_sources.py`, `ingest.py`, `data/sources/cav/` |
| 2 Chunking comparison (section vs fixed) | `ingest.py`, `run_retrieval.py` |
| 3 Test set (answerable / near-miss / off-topic; dev vs held-out test) | `data/questions/dev.yaml`, `test.yaml` |
| 4 BM25 vs dense vs hybrid, 2 embedding models, k = 1/3/5/10 | `retrieval.py`, `run_retrieval.py`, `results/retrieval_dev.csv` |
| 5 Grounded prompt, citations, "I don't know", confidence | `pipeline.py` |
| 6 Threshold sweep on dev, validated once on test | `run_eval.py`, `results/threshold_curve_dev.png` |
| 7 Dashboard metrics, no-RAG baseline, rubric | `run_eval.py`, `results/summary.json`, `docs/RUBRIC.md`, `results/rubric_test.csv` |
| 8 Error analysis | `results/test_per_question.csv` (`error` column), dashboard |
| 9 Dashboard + demo | `app.py`, `templates/`, `static/style.css` |
| 10 Ethics and limitations | disclaimer on every page, `templates/about.html`, `docs/HANDOVER.md` |
| 11 Hand-over | `docs/HANDOVER.md`, `docs/ARCHITECTURE.md`, `docs/METHODOLOGY.md` |
| 12 AI-use log | `docs/AI_USE_LOG.md` |

`../rag_simple/` is the v1 prototype (BM25 over CAV pages only), kept as the baseline.

## Design in one paragraph

Authorised Word versions of the legislation are parsed paragraph by paragraph, keeping section numbers, Parts and Divisions.
Each section (or CAV heading) is one chunk. Chunks are embedded with `mxbai-embed-large`. A question first passes a
jurisdiction check and a topic gate (similarity to the whole knowledge base), then retrieves the top 5 chunks. Provisions for
rooming houses, caravan parks and SDA are demoted unless the question mentions them. If confidence is under a threshold tuned
on the dev set, the assistant says "I don't know". Otherwise a local LLM (temperature 0) answers only from the numbered
sources and must cite them; uncited answers are withheld. Every choice was made on dev and checked once on test. The
evidence is in `docs/METHODOLOGY.md`.

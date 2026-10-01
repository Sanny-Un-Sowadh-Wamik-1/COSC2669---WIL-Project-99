"""Shared settings. Values that are *chosen by experiment* live in results/operating_point.json
(written by run_eval.py) and override the defaults here."""

import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SOURCES = DATA / "sources"
CACHE = ROOT / "cache"            # chunks + embeddings, rebuilt by ingest.py (not committed)
RESULTS = ROOT / "results"
QUESTIONS = DATA / "questions"

OLLAMA = "http://localhost:11434"

# Embedding models compared in run_retrieval.py. Each needs its own query / document prefixes.
EMBED_MODELS = {
    "nomic-embed-text": {"query": "search_query: ", "doc": "search_document: "},
    "mxbai-embed-large": {"query": "Represent this sentence for searching relevant passages: ", "doc": ""},
}

CHUNKINGS = ["section", "fixed"]   # see ingest.py
FIXED_WORDS, FIXED_OVERLAP = 200, 50
SECTION_MAX_WORDS = 300

# Defaults, replaced by results/operating_point.json once the experiments have been run.
DEFAULTS = {
    "chunking": "section",
    "retriever": "hybrid",
    "embed_model": "nomic-embed-text",
    "top_k": 5,
    "confidence": "top_cosine",
    "threshold": 0.55,
    "topic_threshold": 0.45,
    "llm": "qwen2.5:3b",
}


def operating_point():
    path = RESULTS / "operating_point.json"
    settings = dict(DEFAULTS)
    if path.exists():
        settings.update(json.loads(path.read_text()))
    return settings

"""Three retrievers over the same chunks: BM25 (keywords), dense (embeddings), and hybrid (both).

Every retriever returns [(chunk_index, score)] best first. The dense cosine of each chunk is also
available through `cosines()`, because cosine is on a fixed 0-1 scale and therefore usable as a
confidence signal, whereas a BM25 score depends on how long the question is.
"""

import hashlib
import json
import re

import numpy as np
import requests
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from config import CACHE, EMBED_MODELS, OLLAMA

STOPWORDS = set("""a an and are as at be by can could do does for from has have how i if in is it
its me my of on or our should so than that the their them then there they this to was we what
when where which who will with would you your""".split())

# Everyday words -> the words the Act and CAV use since the 2021 reforms. Applied to BM25 queries only.
SYNONYMS = {
    "landlord": "rental provider", "landlords": "rental provider", "landlady": "rental provider",
    "owner": "rental provider", "agent": "rental provider agent", "tenant": "renter", "tenants": "renter",
    "lease": "rental agreement", "leases": "rental agreement", "evict": "notice to vacate",
    "evicted": "notice to vacate", "eviction": "notice to vacate", "kicked": "notice to vacate",
    "deposit": "bond", "tribunal": "VCAT",
}

_stem = PorterStemmer().stem

# The Act repeats most rules for rooming houses, caravan parks, Part 4A sites and specialist
# disability accommodation (SDA), often word for word. Most users rent an ordinary house or flat,
# so unless the question mentions one of these, chunks about them are demoted (not removed).
OTHER_TENURE = re.compile(r"rooming house|caravan|\bsite\b|site agreement|site tenant|Part 4A|\bSDA\b|"
                          r"specialist disability|residency right|\bresidents?\b", re.IGNORECASE)
OTHER_TENURE_QUERY = re.compile(r"rooming|caravan|residential park|\bsite\b|\bSDA\b|disability|"
                                r"boarding|movable dwelling", re.IGNORECASE)
TENURE_PENALTY = 0.2


def is_other_tenure(chunk):
    if chunk["doc"] == "RHS" or "Schedule 6" in chunk["part"]:
        return True
    if chunk["doc"] == "RTA":
        return bool(re.match(r"Part (3|4|4A|12A|14)\b", chunk["part"])) or bool(OTHER_TENURE.search(chunk["heading"]))
    if chunk["doc"] == "RTR":
        return bool(OTHER_TENURE.search(chunk["heading"]))
    return False


def tokenize(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in STOPWORDS]


def expand(question):
    return " ".join(SYNONYMS.get(w.lower().strip("?,.!'"), w) for w in question.split())


def chunk_text(chunk):
    return f"{chunk['citation']}. {chunk['heading']}. {chunk['text']}"


# ---------------------------------------------------------------------------- embeddings
def embed(texts, model, batch=32):
    vectors = []
    for i in range(0, len(texts), batch):
        reply = requests.post(f"{OLLAMA}/api/embed", json={"model": model, "input": texts[i:i + batch],
                                                           "keep_alive": "30m"}, timeout=600)
        reply.raise_for_status()
        vectors.extend(reply.json()["embeddings"])
    matrix = np.asarray(vectors, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def chunk_embeddings(chunks, model, chunking):
    """Embeddings for every chunk, cached on disk and keyed by a hash of the chunk texts."""
    texts = [EMBED_MODELS[model]["doc"] + chunk_text(c) for c in chunks]
    digest = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]
    path = CACHE / f"emb_{model.split(':')[0]}_{chunking}_{digest}.npy"
    if path.exists():
        return np.load(path)
    print(f"  embedding {len(texts)} chunks with {model} ({chunking}) ...", flush=True)
    matrix = embed(texts, model)
    np.save(path, matrix)
    return matrix


# ---------------------------------------------------------------------------- the index
class Index:
    def __init__(self, chunking, embed_model):
        self.chunking, self.embed_model = chunking, embed_model
        self.chunks = json.loads((CACHE / f"chunks_{chunking}.json").read_text())
        self.bm25 = BM25Okapi([tokenize(chunk_text(c)) for c in self.chunks])
        self.vectors = chunk_embeddings(self.chunks, embed_model, chunking) if embed_model else None
        self._qcache = {}

    def query_vector(self, question):
        if question not in self._qcache:
            prefix = EMBED_MODELS[self.embed_model]["query"]
            self._qcache[question] = embed([prefix + question], self.embed_model)[0]
        return self._qcache[question]

    def cosines(self, question):
        return self.vectors @ self.query_vector(question)

    def bm25_scores(self, question):
        return self.bm25.get_scores(tokenize(expand(question)))

    def search(self, question, method="hybrid", k=5, tenure_filter=True):
        scores = self.scores(question, method)
        if tenure_filter and not OTHER_TENURE_QUERY.search(question):
            scores = np.where(self.other_tenure, scores - TENURE_PENALTY * np.abs(scores), scores)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    @property
    def other_tenure(self):
        if not hasattr(self, "_other_tenure"):
            self._other_tenure = np.array([is_other_tenure(c) for c in self.chunks])
        return self._other_tenure

    def scores(self, question, method):
        if method == "bm25":
            scores = self.bm25_scores(question)
        elif method == "dense":
            scores = self.cosines(question)
        elif method == "hybrid":                  # reciprocal rank fusion (Cormack et al., 2009)
            scores = rrf([self.bm25_scores(question), self.cosines(question)])
        elif method == "hybrid_wsum":             # weighted sum of min-max normalised scores
            scores = 0.5 * minmax(self.bm25_scores(question)) + 0.5 * minmax(self.cosines(question))
        else:
            raise ValueError(method)
        return np.asarray(scores, dtype=np.float64)


def rrf(score_lists, k=60):
    fused = np.zeros_like(score_lists[0], dtype=np.float64)
    for scores in score_lists:
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused


def minmax(x):
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return (x - x.min()) / span if span else np.zeros_like(x)

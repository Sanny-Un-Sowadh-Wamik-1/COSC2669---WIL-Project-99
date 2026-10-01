"""
STEP 2 - The RAG system (Retrieval-Augmented Generation)

How one question is answered:

  1. Scope check  : is the question about another state (NSW, QLD...)? Our
                    knowledge base only covers Victoria, so we say "I don't know".
  2. Clean-up     : swap everyday words for the words CAV uses
                    (landlord -> rental provider, lease -> rental agreement).
  3. Retrieve     : BM25 search finds the 3 passages that best match the question.
                    BM25 is the same keyword search Walert used as a baseline.
  4. Confidence   : the BM25 score of the best passage. High score = the words of
                    the question are strongly matched by a passage.
  5. Decide       : if confidence < THRESHOLD -> answer "I don't know".
                    A wrong answer with sources attached LOOKS trustworthy, so it is
                    worse than no answer. The threshold is tuned in evaluate.py.
  6. Generate     : write the answer from those 3 passages only, and cite them.
                    - with Ollama running: a local LLM writes the answer
                    - without Ollama: we copy the 2 best-matching sentences (extractive)

Try it:  python rag.py "Can my landlord charge a pet bond?"
"""

import json
import re
import sys
from pathlib import Path

import requests
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

PASSAGES_FILE = Path(__file__).parent / "data" / "passages.json"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

DEFAULT_THRESHOLD = 8.0      # replaced by the tuned value in results/summary.json (see evaluate.py)
TOP_K = 3                    # how many passages we give to the generator

I_DONT_KNOW = ("I don't know. I couldn't find this in the Victorian renting guidance I have. "
               "Please contact Consumer Affairs Victoria (1300 55 81 81) or Tenants Victoria.")

# Common words that carry no meaning for search ("the", "can", "my"...).
STOPWORDS = set("""a an and are as at be by can could do does for from has have how i if in is it
its me my of on or our should so than that the their them then there they this to was we what
when where which who will with would you your""".split())

# Everyday word -> the word Consumer Affairs Victoria uses since the 2021 law changes.
SYNONYMS = {
    "landlord": "rental provider",
    "landlords": "rental provider",
    "owner": "rental provider",
    "tenant": "renter",
    "tenants": "renter",
    "lease": "rental agreement",
    "evict": "notice to vacate",
    "evicted": "notice to vacate",
    "eviction": "notice to vacate",
    "deposit": "bond",
    "tribunal": "VCAT",
}

# Our knowledge base only covers Victoria.
OTHER_STATES = re.compile(r"\b(nsw|new south wales|queensland|qld|tasmania|western australia|"
                          r"south australia|northern territory|ACT)\b", re.IGNORECASE)

stemmer = PorterStemmer()


def tokenize(text):
    """Lower-case, keep words and numbers, drop stopwords, and stem
    (so 'increases', 'increased' and 'increase' all become 'increas')."""
    words = re.findall(r"[a-z0-9$]+", text.lower())
    return [stemmer.stem(w) for w in words if w not in STOPWORDS]


def replace_synonyms(question):
    words = question.split()
    return " ".join(SYNONYMS.get(w.lower().strip("?,.!"), w) for w in words)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]


class RenterRAG:
    def __init__(self, use_ollama=False, threshold=DEFAULT_THRESHOLD):
        self.passages = json.loads(PASSAGES_FILE.read_text(encoding="utf-8"))
        self.by_id = {p["id"]: p for p in self.passages}
        # We index the page title + heading + text, so "Pet bonds" in a heading also counts.
        documents = [tokenize(f"{p['page_title']} {p['heading']} {p['text']}") for p in self.passages]
        self.bm25 = BM25Okapi(documents)
        self.use_ollama = use_ollama
        self.threshold = threshold

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, question, k=TOP_K):
        """Return the k best passages as (passage, score) pairs, best first."""
        query = tokenize(replace_synonyms(question))
        scores = self.bm25.get_scores(query)
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.passages[i], float(scores[i])) for i in best]

    # ---------------------------------------------------------------- generation
    def extractive_answer(self, question, passages):
        """No LLM: pick the 2 sentences that share the most words with the question."""
        query_words = set(tokenize(replace_synonyms(question)))
        candidates = []
        for rank, p in enumerate(passages):
            for sentence in split_sentences(p["text"]):
                overlap = len(query_words & set(tokenize(sentence)))
                candidates.append((overlap, -rank, sentence, p["id"]))
        candidates.sort(reverse=True)                    # most overlap first, then best passage
        chosen = candidates[:2]
        return " ".join(f"{sentence} [{pid}]" for _, _, sentence, pid in chosen)

    def llm_answer(self, question, passages):
        """With Ollama: give the LLM ONLY our passages and ask it to cite them."""
        context = "\n\n".join(f"[{p['id']}] {p['heading']}: {p['text']}" for p in passages)
        prompt = (
            "Answer the renter's question using ONLY the passages below. "
            "After each sentence put the passage id in square brackets, e.g. [CAV-PETS-06]. "
            "Keep it to 1-3 sentences and copy numbers exactly. "
            "If the passages do not answer the question, reply exactly: I don't know.\n\n"
            f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        reply = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "user", "content": prompt}],
        }, timeout=300)
        return reply.json()["message"]["content"].strip()

    # ---------------------------------------------------------------- the whole pipeline
    def answer(self, question):
        """Answer one question. Returns a dict so the app and the evaluation can use it."""
        result = {"question": question, "answer": I_DONT_KNOW, "answered": False,
                  "confidence": 0.0, "sources": [], "retrieved": [], "reason": ""}

        # 1. scope check
        if OTHER_STATES.search(question):
            result["reason"] = "other state"
            return result

        # 2-3. retrieve the top passages
        top = self.retrieve(question)
        result["retrieved"] = [p["id"] for p, _ in top]

        # 4-5. confidence = BM25 score of the best passage; too low -> I don't know
        result["confidence"] = round(top[0][1], 2)
        if result["confidence"] < self.threshold:
            result["reason"] = "low confidence"
            return result

        # 6. generate the answer from the passages
        passages = [p for p, _ in top]
        if self.use_ollama:
            text = self.llm_answer(question, passages)
        else:
            text = self.extractive_answer(question, passages)
        if text.lower().startswith("i don't know"):
            result["reason"] = "LLM said it doesn't know"
            return result

        # keep only citations that point at passages we actually gave it
        cited = re.findall(r"\[(CAV-[A-Z]+-\d+)\]", text)
        result.update(answer=text, answered=True,
                      sources=[c for c in dict.fromkeys(cited) if c in result["retrieved"]])
        return result


def load_tuned_threshold():
    """Use the threshold chosen by evaluate.py if it has been run."""
    summary = Path(__file__).parent / "results" / "summary.json"
    if summary.exists():
        return json.loads(summary.read_text())["threshold"]
    return DEFAULT_THRESHOLD


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much notice do I get before a rent increase?"
    rag = RenterRAG(threshold=load_tuned_threshold())
    out = rag.answer(question)
    print(json.dumps(out, indent=2))

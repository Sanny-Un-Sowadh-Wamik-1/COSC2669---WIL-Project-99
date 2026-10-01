"""Experiment 1: which chunking + retriever + embedding model finds the right law?

Compares, on the answerable DEV questions:
    chunking  : section | fixed (200 words, 50 overlap)
    retriever : bm25 | dense | hybrid (RRF) | hybrid_wsum
    embedding : nomic-embed-text | mxbai-embed-large
    tenure    : with / without demoting rooming-house / caravan / SDA provisions
and reports recall@k, hit@k, MRR and nDCG@k for k = 1, 3, 5, 10.

The best configuration is chosen on DEV by nDCG@5 (ties -> hit@5), then the chosen configuration
and the BM25 baseline are run once on TEST for the report.

Run:  python run_retrieval.py       -> results/retrieval_dev.csv, results/retrieval_test.csv,
                                       results/retrieval_choice.json
"""

import csv
import json

import yaml

from config import CHUNKINGS, EMBED_MODELS, QUESTIONS, RESULTS
from metrics import hit_at_k, mean, mrr, ndcg_at_k, recall_at_k
from retrieval import Index

KS = (1, 3, 5, 10)


def load(split):
    return yaml.safe_load((QUESTIONS / f"{split}.yaml").read_text())


def evaluate(index, questions, method, tenure):
    rows = []
    for q in questions:
        ranked = [index.chunks[i]["units"] for i, _ in index.search(q["q"], method, k=max(KS), tenure_filter=tenure)]
        gold = q["gold"]
        row = {"qid": q["qid"], "mrr": mrr(ranked, gold)}
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(ranked, gold, k)
            row[f"hit@{k}"] = hit_at_k(ranked, gold, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        rows.append(row)
    keys = [k for k in rows[0] if k != "qid"]
    return {k: round(mean([r[k] for r in rows]), 3) for k in keys}, rows


def configurations():
    for chunking in CHUNKINGS:
        yield chunking, "bm25", None
        for model in EMBED_MODELS:
            for method in ("dense", "hybrid", "hybrid_wsum"):
                yield chunking, method, model


def main():
    RESULTS.mkdir(exist_ok=True)
    dev = [q for q in load("dev") if q["group"] == "answerable"]
    test = [q for q in load("test") if q["group"] == "answerable"]
    indexes, table = {}, []
    for chunking, method, model in configurations():
        key = (chunking, model or "nomic-embed-text")
        if key not in indexes:
            indexes[key] = Index(chunking, key[1])
        for tenure in (False, True):
            scores, _ = evaluate(indexes[key], dev, method, tenure)
            table.append({"chunking": chunking, "retriever": method, "embed_model": model or "-",
                          "tenure_filter": tenure, **scores})
            print(f"{chunking:8s} {method:12s} {model or '-':18s} tenure={tenure!s:5s} "
                  f"hit@5={scores['hit@5']:.3f} ndcg@5={scores['ndcg@5']:.3f} mrr={scores['mrr']:.3f}")

    write_csv(RESULTS / "retrieval_dev.csv", table)
    best = max(table, key=lambda r: (r["ndcg@5"], r["hit@5"]))
    choice = {k: best[k] for k in ("chunking", "retriever", "embed_model", "tenure_filter")}
    print("\nChosen on DEV:", choice)

    # Report the chosen configuration (and the BM25 baseline) once on TEST.
    test_rows = []
    baseline = {"chunking": "section", "retriever": "bm25", "embed_model": "-", "tenure_filter": False}
    for name, cfg in (("bm25 baseline (v1)", baseline), ("chosen", choice)):
        model = cfg["embed_model"] if cfg["embed_model"] != "-" else "nomic-embed-text"
        scores, per_q = evaluate(indexes[(cfg["chunking"], model)], test, cfg["retriever"], cfg["tenure_filter"])
        test_rows.append({"system": name, **cfg, **scores})
    write_csv(RESULTS / "retrieval_test.csv", test_rows)
    (RESULTS / "retrieval_choice.json").write_text(json.dumps({"chosen": choice, "dev": best,
                                                                "test": test_rows}, indent=2))
    for r in test_rows:
        print(f"TEST {r['system']:20s} hit@5={r['hit@5']:.3f} ndcg@5={r['ndcg@5']:.3f} mrr={r['mrr']:.3f}")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

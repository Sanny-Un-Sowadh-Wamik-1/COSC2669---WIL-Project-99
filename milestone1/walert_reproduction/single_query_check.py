from pathlib import Path
import ast
import pandas as pd


# Configuration

QUESTION_ID = "W02Q01"
TOP_K = 3


# Locating the repository and local Walert

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parents[1]

WALERT_DIR = (
    PROJECT_ROOT
    / "external"
    / "walert"
    / "quantitative_eval"
)

DATA_DIR = WALERT_DIR / "data"
TARGET_DIR = WALERT_DIR / "target"


# Checking that Walert exists locally

if not WALERT_DIR.exists():
    raise FileNotFoundError(
        "Local Walert repository was not found.\n"
        "Clone it into external/walert before running this script."
    )


# Loading stored Walert RAG results

summary_file = TARGET_DIR / "summaries" / "falcon_dense_eval.csv"

results = pd.read_csv(summary_file)

question_result = results[
    results["question_id"] == QUESTION_ID
]

if question_result.empty:
    raise ValueError(
        f"Question {QUESTION_ID} was not found in the Walert results."
    )

row = question_result.iloc[0]


# Extracting top-three retrieved passages

top3 = ast.literal_eval(row["top3"])

top3 = top3[:TOP_K]


# Loading Walert relevance judgments

qrels = pd.read_csv(
    DATA_DIR / "qrels.txt",
    sep=r"\s+",
    names=[
        "question_id",
        "zero",
        "passage_id",
        "relevance",
    ],
)


# Result

print("=" * 70)
print("WALERT SINGLE-QUERY TEST")
print("=" * 70)

print("\nQuestion ID:")
print(row["question_id"])

print("\nQuestion:")
print(row["question_content"])

print("\nTop 3 dense retrieval results:")
print("-" * 70)

relevant_count = 0

for rank, passage_id in enumerate(top3, start=1):

    relevance_match = qrels[
        (qrels["question_id"] == QUESTION_ID)
        & (qrels["passage_id"] == passage_id)
    ]

    if relevance_match.empty:
        relevance = 0
    else:
        relevance = int(
            relevance_match.iloc[0]["relevance"]
        )

    if relevance > 0:
        relevant_count += 1

    print(
        f"Rank {rank}: "
        f"{passage_id}, relevance = {relevance}"
    )


print(
    f"\nRelevant passages in top {TOP_K}: "
    f"{relevant_count}/{TOP_K}"
)


# Showing the stored generated answer

print("\nWalert generated answer from falcon_dense_eval.csv:")
print("-" * 70)
print(row["answer_top3_voice"])
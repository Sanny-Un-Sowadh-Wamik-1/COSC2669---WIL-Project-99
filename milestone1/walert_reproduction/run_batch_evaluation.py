
# Note : This script we produced does not reimplement Walert's evaluation metrics. It calls the original Walert eval.py.

from pathlib import Path
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

WALERT_RETRIEVAL = (
    PROJECT_ROOT
    / "external"
    / "walert"
    / "quantitative_eval"
    / "src"
    / "retrieval"
)

EVAL_SCRIPT = WALERT_RETRIEVAL / "eval.py"

QRELS = "../../data/qrels.txt"
INTENT_RUN = "../../target/runs/walert-intent.txt"
BM25_RUN = "../../target/runs/rag-bm25.txt"
DENSE_RUN = "../../target/runs/rag-dense-faiss.txt"


def run_evaluation(question_type):
    print(f"\n{'=' * 60}")
    print(f"Walert {question_type.capitalize()} Question Evaluation")
    print("=" * 60)

    subprocess.run(
        [
            sys.executable,
            "eval.py",
            question_type,
            QRELS,
            INTENT_RUN,
            BM25_RUN,
            DENSE_RUN,
        ],
        cwd=WALERT_RETRIEVAL,
        check=True,
    )


if __name__ == "__main__":
    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(
            "Walert eval.py was not found. "
            "Clone Walert into external/walert first."
        )

    run_evaluation("known")
    run_evaluation("inferred")
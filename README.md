## Group ID

*99*

## Project Links

•⁠  ⁠*Trello:* https://trello.com/b/6ovWpGaa/cosc2669-wil-project-99
•⁠  ⁠*GitHub:* https://github.com/Sanny-Un-Sowadh-Wamik-1/COSC2669---WIL-Project-99
•⁠  ⁠*Walert Repository:* https://github.com/rmit-ir/walert

## Tentative Project Aim

To develop and quantitatively evaluate a Test-Driven Retrieval-Augmented Generation (RAG) system for a selected domain, using Walert as an initial reproducible baseline.

	As a renter in Victoria, I want to check what the current law says about my situation so that I can hold my ground when facing an agent, a rental provider, or a dispute process instead of guessing.

Our aim is to build a Retrieval-Augmented Generation (RAG) system that makes Victorian renting law accessible to renters, with guidance from Consumer Affairs Victoria, and every answer attributed to the section it came from. Alongside the prototype, we will develop an evaluation framework measuring answer correctness, source attribution accuracy, and the system's ability to decline questions outside its knowledge base.

## Group Members

| Student ID | Name | Milestone 1 Responsibility |
|---|---|---|
| s4130359 | Sanny Un Sowadh Wamik | Issues and bottlenecks |
| s4089296 | Haswanth Lagadapati | Walert reproduction and evaluation evidence |
| s4092642 | Vivan Marwah | Progress so far |
| s4021994 | Twaritha Yepuri | Project aim, three-week plan and Walert evidence |

## Milestone 1 Progress

For Milestone 1, the team reproduced and inspected the supplied Walert retrieval evaluation as an initial baseline before adapting the Test-Driven RAG approach to the Victorian Renter Rights Assistant domain.

The current reproduction work includes:

•⁠  ⁠validation of the Walert evaluation environment;
•⁠  ⁠inspection of a single existing Walert query and its retrieved passages;
•⁠  ⁠batch retrieval evaluation for Walert's Known question set; and
•⁠  ⁠batch retrieval evaluation for Walert's Inferred question set.

The reproduction uses Walert's supplied relevance judgments, stored retrieval runs and original evaluation script. The Group 99 helper scripts do not reimplement Walert's evaluation metrics.

## Preliminary Walert Retrieval Results

| Question Set | Retrieval Method | NDCG@1 | NDCG@3 | NDCG@5 |
|---|---|---:|---:|---:|
| Known | Walert Intent | 0.6429 | 0.3017 | 0.2180 |
| Known | BM25 | 0.5119 | 0.4912 | 0.4733 |
| Known | Dense FAISS | *0.6905* | *0.6119* | *0.5812* |
| Inferred | Walert Intent | 0.0833 | 0.0391 | 0.0391 |
| Inferred | BM25 | 0.1667 | *0.2566* | *0.3291* |
| Inferred | Dense FAISS | *0.2500* | 0.2380 | 0.2380 |

For the Known question set, Dense FAISS achieved the highest NDCG score at all three evaluated cut-offs.

For the Inferred question set, Dense FAISS achieved the highest NDCG@1 score, while BM25 achieved the highest NDCG@3 and NDCG@5 scores.

These results provide an initial quantitative retrieval baseline for the team's later Test-Driven RAG experiments.

## Repository Structure

⁠ text
COSC2669---WIL-Project-99/
├── README.md
├── .gitignore
└── milestone1/
    └── walert_reproduction/
        ├── single_query_check.py
        ├── run_batch_evaluation.py
        └── evidence/
            ├── A1_environment.png
            ├── A2_single_query.png
            ├── A3a_known_evaluation.png
            └── A3b_inferred_evaluation.png
 ⁠

The original Walert repository is used locally under ⁠ external/walert/ ⁠ and is excluded from the Group 99 GitHub repository through ⁠ .gitignore ⁠.

## Walert Reproduction

### Environment Setup

Create a Python 3.9 Conda environment:

⁠ bash
conda create -n walert39 python=3.9 -y
conda activate walert39
 ⁠

Install the packages used for the retrieval evaluation:

⁠ bash
pip install numpy==1.26.4 pandas==2.0.3 ranx==0.3.18
 ⁠

The environment used for the Milestone 1 reproduction was validated with:

⁠ text
Python 3.9.25
NumPy 1.26.4
pandas 2.0.3
ranx 0.3.18
 ⁠

### Clone the Walert Baseline

From the Group 99 repository root:

⁠ bash
mkdir -p external
git clone https://github.com/rmit-ir/walert.git external/walert
 ⁠

To reproduce the same Walert version used for Milestone 1:

⁠ bash
git -C external/walert checkout <WALERT_COMMIT_HASH>
 ⁠

	⁠Replace ⁠ <WALERT_COMMIT_HASH> ⁠ with the exact Walert commit used for the reproduction.

### Run the Single-Query Inspection

From the Group 99 repository root:

⁠ bash
python milestone1/walert_reproduction/single_query_check.py
 ⁠

This helper inspects an existing Walert question and its supplied retrieval results.
The current example uses:

⁠ text
Question ID: W02Q01
Question: Can you describe the range of electives that are on offer?
 ⁠

The top three Dense retrieval passages were:

⁠ text
P09
P07
P08
 ⁠

Each had a supplied relevance judgment of ⁠ 2 ⁠.

The stored generated answer displayed by the helper is read from Walert's existing ⁠ falcon_dense_eval.csv ⁠. The helper does not rerun Falcon locally.

### Run the Batch Retrieval Evaluation

From the Group 99 repository root:

⁠ bash
python milestone1/walert_reproduction/run_batch_evaluation.py
 ⁠

The Group 99 helper executes Walert's original ⁠ eval.py ⁠ for both:

⁠ text
Known questions
Inferred questions
 ⁠

and reports:

⁠ text
NDCG@1
NDCG@3
NDCG@5
 ⁠

The helper does not implement NDCG itself.

## Milestone 1 Evidence

The following evidence is stored under:

⁠ text
milestone1/walert_reproduction/evidence/
 ⁠

•⁠  ⁠⁠ A1_environment.png ⁠ - validation of the Walert evaluation environment.
•⁠  ⁠⁠ A2_single_query.png ⁠ - inspection of one existing Walert query and its retrieved passages.
•⁠  ⁠⁠ A3a_known_evaluation.png ⁠ - batch retrieval evaluation for the Known question set.
•⁠  ⁠⁠ A3b_inferred_evaluation.png ⁠ - batch retrieval evaluation for the Inferred question set.

The report appendix contains the corresponding screenshots and short interpretations of these results.

## Reproduction Scope

The Milestone 1 work reproduces the supplied Walert retrieval evaluation using Walert's existing:

•⁠  ⁠relevance judgments;
•⁠  ⁠stored retrieval runs;
•⁠  ⁠test question sets; and
•⁠  ⁠retrieval evaluation script.

It does not claim to reproduce the complete Walert RAG pipeline. The team did not rebuild the Walert indexes or rerun Falcon generation as part of this preliminary milestone.

## Next Project Stage

The next stage will adapt the Test-Driven RAG approach to the *Victorian Renter Rights Assistant* domain by:

•⁠  ⁠identifying authoritative Victorian tenancy information sources;
•⁠  ⁠constructing a domain-specific knowledge base;
•⁠  ⁠developing renter-focused evaluation questions;
•⁠  ⁠comparing retrieval and generation configurations; and
•⁠  ⁠evaluating the system quantitatively against defined test cases.

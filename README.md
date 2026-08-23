can someone commit this in git hub new readme version : # COSC2669 WIL Project 99

## Group ID

*99*

## Project Links

•⁠  ⁠*Trello:* https://trello.com/b/6ovWpGaa/cosc2669-wil-project-99
•⁠  ⁠*GitHub:* https://github.com/Sanny-Un-Sowadh-Wamik-1/COSC2669---WIL-Project-99
•⁠  ⁠*Walert Repository:* https://github.com/rmit-ir/walert

## Tentative Project Aim

To develop and quantitatively evaluate a Test-Driven Retrieval-Augmented Generation (RAG) system for a selected domain, using Walert as an initial reproducible baseline.

	⁠The domain-specific aim will be updated once the team confirms the final project topic.

## Group Members

| Student ID | Name | Milestone 1 Responsibility |
|---|---|---|
| s4130359 | Sanny Un Sowadh Wamik | Issues and bottlenecks |
| s4089296 | Haswanth Lagadapati | Walert reproduction and evaluation evidence |
| s4092642 | Vivan Marwah | Progress so far |
| s4021994 | Twaritha Yepuri | Project aim, three-week plan and Walert evidence |

## Milestone 1 Progress

For Milestone 1, the team began by reproducing and inspecting the supplied Walert retrieval evaluation as preliminary project work.

The current reproduction work includes:

•⁠  ⁠validation of the Walert evaluation environment;
•⁠  ⁠inspection of a single existing Walert query and its retrieved passages;
•⁠  ⁠batch retrieval evaluation for Walert's Known question set; and
•⁠  ⁠batch retrieval evaluation for Walert's Inferred question set.

The reproduction uses Walert's supplied relevance judgments, stored retrieval runs and original evaluation script. The Group 99 helper scripts do not reimplement Walert's evaluation metrics.

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

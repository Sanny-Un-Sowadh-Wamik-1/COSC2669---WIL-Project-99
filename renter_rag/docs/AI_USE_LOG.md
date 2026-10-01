# AI-use log

Every interaction with an AI tool on this part of the project. Your declaration must match this log.
**The "Decision" column is for Sanny to complete after reviewing each output.** A blank means it hasn't been reviewed yet.

| # | Date | Tool | Purpose | What the AI produced | Decision (accepted / revised / rejected, and why) |
|---|---|---|---|---|---|
| 1 | 2026-10-01 | Claude Code (Claude Opus 5.5) | Gap analysis of `rag_simple` v1 against the 12-step plan | Table of what was done, partly done and missing; risks (synonym-list leakage, unverified 60→90-day claim) | |
| 2 | 2026-10-01 | Claude Code | Find and download the current Victorian tenancy legislation | Located authorised versions (RTA v114, Regs v009, Rooming House Regs v002), checked copyright terms of both sites, wrote `sources.yaml` + `fetch_sources.py` | |
| 3 | 2026-10-01 | Claude Code | Parse the legislation and implement two chunking strategies | `ingest.py` (section vs 200-word fixed window) | |
| 4 | 2026-10-01 | Claude Code | Extend the test set | Added Act/Regulation gold units to the v1 questions, 20 new legislation questions (N01–N20), 5 near-miss and 9 off-topic / prompt-injection questions; re-split dev/test. **The legislation gold labels and reference answers were AI-drafted from the Act text and must be checked by a teammate (Step 3).** | |
| 5 | 2026-10-01 | Claude Code | Retrieval comparison | `retrieval.py`, `run_retrieval.py`, `metrics.py` + unit tests; ran 28 configurations | |
| 6 | 2026-10-01 | Claude Code | Grounded generation, confidence signals, threshold tuning, evaluation | `pipeline.py` (prompt, gates), `run_eval.py` (3 LLMs, 4 confidence signals, dev sweep, test, closed-book baseline, error categories) | |
| 7 | 2026-10-01 | Claude Code | Web app and dashboard | Flask app, templates, CSS | |
| 8 | 2026-10-01 | Claude Code | Documentation | README, ARCHITECTURE, RUBRIC, METHODOLOGY, HANDOVER, this log | |

The local LLMs used *inside* the system (qwen2.5:3b, llama3.2:3b, gemma3:4b via Ollama) are part of the product being evaluated,
not AI assistance with the assignment. No LLM was used as a judge of correctness: correctness is a key-fact check plus the human rubric.

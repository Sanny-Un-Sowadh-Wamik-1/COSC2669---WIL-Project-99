# Architecture

```mermaid
flowchart LR
    subgraph Offline["Offline (once)"]
        A1["legislation.vic.gov.au<br/>RTA 1997 v114 · Regs 2021 v009<br/>Rooming House Regs 2023 v002"] -->|fetch_sources.py| A2[".docx"]
        A3["consumer.vic.gov.au<br/>25 CAV renting pages"] --> A4[".md with source header"]
        A2 & A4 -->|"ingest.py<br/>section chunks (1 per s / reg / heading)"| B["1,509 chunks<br/>+ unit ids (RTA-s44 …)"]
        B -->|"mxbai-embed-large<br/>via Ollama"| C[("embeddings .npy")]
        B --> D[("BM25 index")]
    end

    subgraph Online["Per question (app.py → pipeline.py)"]
        Q["Renter's question"] --> G1{"Other state?"}
        G1 -- yes --> R1["Decline: Victorian law only"]
        G1 -- no --> E["Embed question"]
        E --> G2{"Max similarity to KB<br/>≥ topic threshold?"}
        G2 -- no --> R2["Decline: off-topic"]
        G2 -- yes --> RET["Dense retrieval top-5<br/>(rooming house / caravan / SDA demoted)"]
        RET --> G3{"Confidence<br/>≥ threshold?"}
        G3 -- no --> R3["I don't know + where to get help"]
        G3 -- yes --> LLM["Local LLM, temperature 0<br/>answer ONLY from numbered sources, cite [n]"]
        LLM --> G4{"Said 'I don't know'<br/>or cited nothing?"}
        G4 -- yes --> R3
        G4 -- no --> OUT["Answer + cited sections<br/>(streamed to the browser)"]
    end
    C -. cosine .-> RET
    C -. cosine .-> G2
```

| Component | File | Notes |
|---|---|---|
| Source download | `fetch_sources.py`, `sources.yaml` | URLs, versions, access dates, terms of use |
| Parsing + chunking | `ingest.py` | Reads the authorised Word files paragraph by paragraph; keeps section numbers, Part and Division; skips side notes, transitional provisions and forms |
| Retrieval | `retrieval.py` | BM25, dense, hybrid (RRF and weighted sum), embedding cache, tenure demotion |
| Pipeline | `pipeline.py` | Gates, prompt, citation check, streaming |
| Experiments | `run_retrieval.py`, `run_eval.py` | Dev for choices, test once for reporting |
| App | `app.py`, `templates/`, `static/style.css` | Flask, server-rendered, no JS framework, no animation |

**Speed.** On an 8 GB M1 MacBook Air: start-up about 2 s (cached embeddings), retrieval and gates under 0.1 s.
The first answer token arrives in about 2–4 s once the LLM is warm, and the app warms it at start-up.
Off-topic and other-state questions are refused instantly, without calling the LLM.

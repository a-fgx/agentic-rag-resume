# PDD.md
## Project: Agentic RAG Resume — Phase 2 (MongoDB Atlas RAG)

---

## 1. Overview

This project is a CLI-based AI agent that answers questions about Alain Feigneux's professional background. It demonstrates practical experience building and integrating production-grade AI/LLM-powered systems.

The agent:

- Uses **LangGraph** to orchestrate a 5-node agent workflow
- Implements **RAG** with **MongoDB Atlas Vector Search** (`$vectorSearch`)
- Supports **prompt routing** across three response modes
- Enforces **structured outputs** via Pydantic
- Provides **traceability** (chunk-level citations) and **latency metrics**
- Supports optional **text-to-speech** via ElevenLabs
- Runs entirely via CLI

---

## 2. Objectives

The system demonstrates competence in:

- Agent workflow orchestration (LangGraph)
- Vector database integration (MongoDB Atlas)
- Prompt orchestration and routing
- Structured output validation (Pydantic)
- Observability discipline (per-node timings, traceable chunk citations)

---

## 3. Success Criteria

A reviewer can:

### 1. Run ingestion
```bash
uv run python -m app.ingest
```

### 2. Run a query
```bash
uv run rag "How do you handle latency spikes?" --mode incident
```

### 3. Observe
- Structured JSON output
- Mode-specific behavior (qa / executive / incident)
- Retrieved chunk citations (e.g. `resume.md#2`)
- Per-node timing metrics (`retrieve`, `llm`, `total`)

If these are present and coherent, the proof is successful.

---

## 4. Scope

### In Scope (Phase 2)
- Markdown knowledge ingestion with stable chunk IDs
- Paragraph-based chunking (~200 words per chunk)
- Batch embedding via Gemini (`text-embedding-004`) or OpenAI
- MongoDB Atlas `$vectorSearch` index (cosine similarity, 768 dims)
- LangGraph workflow orchestration (5 nodes, conditional edge)
- Mode routing (qa, executive, incident)
- Structured JSON output with Pydantic validation
- Chunk-level citations (`resume.md#2`, `projects.md#0`, ...)
- Per-node timing metrics
- Optional TTS via ElevenLabs (conditional routing)
- 66 mocked tests — no API keys required for CI

### Out of Scope
- Web interface
- Authentication
- Streaming responses
- Advanced evaluation framework

---

## 5. Repository Structure

```
agentic-rag-resume/
├── knowledge/
│   ├── resume.md
│   ├── projects.md
│   └── stories.md
│
├── prompts/
│   ├── system.txt
│   ├── qa.txt
│   ├── executive.txt
│   └── incident.txt
│
├── app/
│   ├── __init__.py
│   ├── schemas.py       # Pydantic models + GraphState TypedDict
│   ├── ingest.py        # Ingestion: chunk → embed → upsert → index
│   └── graph.py         # LangGraph: 5 nodes + conditional edge
│
├── tests/
│   ├── conftest.py
│   ├── test_m0_ingest.py
│   ├── test_m1_graph.py
│   ├── test_m2_output.py
│   └── test_voice.py
│
├── cli.py
├── pyproject.toml
├── .env.example
└── PDD.md
```

---

## 6. High-Level Architecture

```
CLI
  │
  ▼
LangGraph Workflow
  ├── Retrieve Node    (MongoDB Atlas $vectorSearch)
  ├── Route Node       (mode → prompt template)
  ├── Generate Node    (LLM + Pydantic structured output)
  ├── Finalize Node    (chunk citations + total timing)
  └── Speak Node       (ElevenLabs TTS — conditional)

Flow:
START → retrieve → route → generate → finalize → END
                                          ↓ (voice=True)
                                        speak → END
```

---

## 7. Data Model

### Knowledge Chunks (MongoDB Atlas)

Each document stored in the `chunks` collection:

```json
{
  "_id": "<ObjectId>",
  "chunk_id": "resume.md#2",
  "source": "resume.md",
  "text": "...",
  "embedding": [768 floats]
}
```

### Runtime State (LangGraph `GraphState`)

```python
class GraphState(TypedDict, total=False):
    mode: str                     # qa | executive | incident
    question: str
    k: int                        # top-k chunks (default: 5)
    voice: bool                   # TTS flag
    retrieved_chunks: list[dict]  # from MongoDB.aggregate()
    prompt_template: str
    answer: dict[str, Any]
    citations: list[str]          # e.g. ["resume.md#2", "projects.md#0"]
    timings_ms: dict[str, float]  # retrieve, llm, tts?, total
```

---

## 8. Vector Database Design

### Embedding Model
- Default: `text-embedding-004` (Google Gemini, 768 dimensions)
- Alternative: `text-embedding-3-large` (OpenAI, 1536 dimensions)
- Configurable via `EMBEDDING_MODEL` environment variable

### Similarity
- Cosine similarity via MongoDB Atlas `$vectorSearch`

### Retrieval
- Default top-k: 5 (configurable via `--k` CLI flag)
- Returns: `chunk_id`, `source`, `text`, `vectorSearchScore`

### Vector Index Definition (Atlas)

```json
{
  "name": "vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 768,
        "similarity": "cosine"
      }
    ]
  }
}
```

---

## 9. LangGraph Workflow Design

### Node 1: Retrieve

**Input:** `question`, `k`
**Output:** `retrieved_chunks`, `timings_ms.retrieve`

Responsibilities:
- Embed question with `EMBEDDING_MODEL`
- Run MongoDB Atlas `$vectorSearch` aggregation pipeline
- Return top-k chunks with scores and chunk IDs

### Node 2: Route

**Input:** `mode`
**Output:** `mode`, `prompt_template`

Responsibilities:
- Validate mode (`qa` | `executive` | `incident`), default to `qa`
- Load matching prompt template from `prompts/<mode>.txt`

### Node 3: Generate

**Input:** `question`, `retrieved_chunks`, `prompt_template`
**Output:** `answer`, `timings_ms.llm`

Responsibilities:
- Build context string: each chunk tagged with `[chunk_id]`
- Use `.replace()` (not `.format()`) to safely inject `{context}` and `{question}`
- Call LLM with `response_format={"type": "json_object"}`
- Validate response against `Answer` Pydantic schema (fail fast if invalid)

### Node 4: Finalize

**Input:** `retrieved_chunks`, `timings_ms`
**Output:** `citations`, `timings_ms.total`

Responsibilities:
- Extract `chunk_id` from each retrieved chunk → `citations`
- Compute `total = sum(all timings)`

### Node 5: Speak *(conditional)*

**Input:** `answer.short_answer`, `timings_ms`
**Output:** `timings_ms.tts`

Responsibilities:
- Only reached when `voice=True` (conditional edge from `finalize`)
- Convert `short_answer` to audio via ElevenLabs `text_to_speech.convert`
- Play audio inline

---

## 10. Ingestion Pipeline (`app/ingest.py`)

```
load_markdown_files()   → list of {source, text}
  ↓
chunk_document()        → list of {chunk_id, source, text}   (~200 words/chunk)
  ↓
embed_text()            → list of float[]                     (batched, 100/call)
  ↓
upsert_chunks()         → MongoDB insert_many                 (drops + recreates)
  ↓
ensure_vector_index()   → Atlas vectorSearch index creation   (idempotent)
```

---

## 11. Output Schema

### Answer Model

```python
class Answer(BaseModel):
    short_answer: str
    technical_plan: list[str]
    risks: list[str]
    business_impact: list[str]
    follow_up_question: Optional[str] = None
```

### CLI Output Envelope

```json
{
  "mode": "incident",
  "question": "How did you handle production latency spikes?",
  "answer": {
    "short_answer": "...",
    "technical_plan": ["..."],
    "risks": ["..."],
    "business_impact": ["..."],
    "follow_up_question": "..."
  },
  "citations": ["resume.md#2", "stories.md#1", "projects.md#0"],
  "timings_ms": {
    "retrieve": 45.2,
    "llm": 820.4,
    "total": 865.6
  }
}
```

---

## 12. Prompt Strategy

### `system.txt`
- Enterprise tone, concise and operational
- Grounded strictly in provided context
- Strict JSON output enforcement

### `qa.txt`
- Factual, direct answers
- Context-only grounding

### `executive.txt`
- 60-second board-level framing
- Business impact first, evidence-grounded

### `incident.txt`
- SRE-style: diagnosis → mitigation → prevention
- Technical plan required

---

## 13. CLI Specification

```bash
uv run rag "<question>" [--mode qa|executive|incident] [--k N] [--voice]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `qa` | Response mode |
| `--k` | `5` | Number of chunks to retrieve |
| `--voice` | `False` | Speak `short_answer` via ElevenLabs |

Behavior:
- Prints structured JSON to stdout
- Non-zero exit on fatal errors

---

## 14. Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes* | — | Google Gemini API key |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key (alternative to Gemini) |
| `CHAT_MODEL` | No | `gemini-2.0-flash` | LLM model |
| `EMBEDDING_MODEL` | No | `text-embedding-004` | Embedding model |
| `MONGODB_URI` | Yes | — | MongoDB Atlas connection string |
| `MONGODB_DB` | No | `ragresume` | Database name |
| `MONGODB_COLLECTION` | No | `chunks` | Collection name |
| `ELEVENLABS_API_KEY` | No | — | ElevenLabs key (only for `--voice`) |
| `ELEVENLABS_VOICE_ID` | No | Rachel | ElevenLabs voice ID |

*Either `GEMINI_API_KEY` or `OPENAI_API_KEY` is required.

---

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Hallucination | Strict context-only grounding in prompts |
| Weak retrieval | Paragraph-based chunking + configurable top-k |
| JSON parsing failure | Pydantic validation — fails fast with clear error |
| High latency | Configurable `--k`; use `gemini-2.0-flash` |
| Atlas index not ready | 60-second build time documented; graceful error |

---

## 16. Milestones

| Milestone | Description | Status |
|---|---|---|
| M0 | Ingestion pipeline: chunk → embed → MongoDB upsert | Done |
| M1 | LangGraph workflow: retrieve → route → generate → finalize | Done |
| M2 | Structured output + chunk citations + per-node timings | Done |
| M2+ | ElevenLabs TTS with conditional routing | Done |
| M3 | Grounding verification node (optional) | Backlog |

---

## 17. Definition of Done

The prototype clearly demonstrates:

- Agent workflow orchestration via LangGraph (5 nodes, conditional edge)
- MongoDB Atlas Vector Search integration (`$vectorSearch`)
- Knowledge ingestion pipeline (chunk → embed → upsert → index)
- Prompt routing logic (3 modes)
- Structured and validated outputs (Pydantic)
- Traceable chunk-level citations
- Per-node latency observability
- 66 tests — all mocked, no API keys required for CI
- GitHub Actions CI pipeline

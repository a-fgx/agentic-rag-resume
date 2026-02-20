# Agentic RAG Resume

[![Tests](https://github.com/a-fgx/agentic-rag-resume/actions/workflows/tests.yml/badge.svg)](https://github.com/a-fgx/agentic-rag-resume/actions/workflows/tests.yml)

A CLI agent that answers questions about Alain Feigneux's professional background using **LangGraph**, **MongoDB Atlas Vector Search**, **Gemini**, **ElevenLabs TTS**, and **Pydantic**.

---

## Architecture

```
CLI → [retrieve] → [route] → [generate] → [finalize] → JSON output
                                               ↓ (--voice)
                                          [speak: ElevenLabs TTS]
```

| Node | Responsibility |
|---|---|
| `retrieve` | Embeds the question and fetches the top-k chunks from MongoDB Atlas `$vectorSearch` |
| `route` | Validates the mode and loads the matching prompt template |
| `generate` | Calls the LLM with JSON mode, validates the response with Pydantic |
| `finalize` | Extracts chunk citations and computes total timing |
| `speak` | *(optional)* Converts `short_answer` to speech via ElevenLabs TTS |

The `speak` node is reached via a **conditional edge** from `finalize` — only when `--voice` is passed. This demonstrates LangGraph's conditional routing.

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required variables:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key ([get one free](https://aistudio.google.com)) |
| `CHAT_MODEL` | LLM model (default: `gemini-2.0-flash`) |
| `EMBEDDING_MODEL` | Embedding model (default: `text-embedding-004`) |
| `MONGODB_URI` | MongoDB Atlas connection string |
| `MONGODB_DB` | Database name (default: `ragresume`) |
| `MONGODB_COLLECTION` | Collection name (default: `chunks`) |
| `ELEVENLABS_API_KEY` | Your ElevenLabs API key — only needed for `--voice` |
| `ELEVENLABS_VOICE_ID` | *(optional)* Your cloned voice ID — defaults to "Rachel" |

### 3. Ingest knowledge into MongoDB

Chunk and embed the knowledge base, then upsert to MongoDB Atlas:

```bash
uv run python -m app.ingest
```

This will:
1. Load all `knowledge/*.md` files
2. Split them into ~200-word chunks with stable IDs (e.g. `resume.md#0`)
3. Embed each chunk using Gemini (or OpenAI)
4. Upsert all chunks + embeddings into MongoDB
5. Create the `vector_index` Atlas vector search index

> **Note:** After running ingestion, wait ~60 seconds for Atlas to build the vector index before querying.

---

## Usage

```bash
# Standard Q&A
uv run rag "What is your experience with CI/CD?"

# Executive mode — board-level framing
uv run rag "What are your key achievements?" --mode executive

# Incident mode — SRE-style response
uv run rag "How did you handle production latency spikes?" --mode incident

# Adjust number of retrieved chunks (default: 5)
uv run rag "What did you do at Nexthink?" --k 3

# With voice — speaks the short_answer aloud via ElevenLabs
uv run rag "How did you reduce release cycle time?" --voice
uv run rag "How did you handle incidents?" --mode incident --voice
```

### Example output

```json
{
  "mode": "incident",
  "question": "How did you handle production latency spikes?",
  "answer": {
    "short_answer": "Latency spikes were detected early via Datadog dashboards...",
    "technical_plan": ["Implemented Datadog alerting on P99 latency", "..."],
    "risks": ["Recurrence under peak load if thresholds are not tuned"],
    "business_impact": ["Reduced mean time to detection", "Limited user impact"],
    "follow_up_question": "What is the current P99 latency baseline?"
  },
  "citations": ["resume.md#2", "stories.md#1", "projects.md#0"],
  "timings_ms": {
    "retrieve": 45.2,
    "llm": 843.5,
    "total": 888.7
  }
}
```

---

## Running tests

```bash
uv run pytest
```

All 66 tests run without any API key (all external calls are mocked).

---

## Voice cloning (use your own voice)

ElevenLabs supports instant voice cloning — no extra tools needed:

1. Go to [elevenlabs.io](https://elevenlabs.io) → **Voice Lab** → **Add a new voice** → **Instant Voice Clone**
2. Upload 1–3 minutes of clean audio of yourself speaking
3. Copy the resulting **Voice ID**
4. Add to your `.env`: `ELEVENLABS_VOICE_ID=your-cloned-voice-id`

Requires at least a Creator plan on ElevenLabs.

---

## Project structure

```
agentic-rag-resume/
├── knowledge/          # Markdown knowledge base
│   ├── resume.md
│   ├── projects.md
│   └── stories.md
├── prompts/            # Prompt templates per mode
│   ├── system.txt
│   ├── qa.txt
│   ├── executive.txt
│   └── incident.txt
├── app/
│   ├── schemas.py      # Pydantic models + GraphState TypedDict
│   ├── graph.py        # LangGraph workflow (5 nodes + conditional edge)
│   └── ingest.py       # MongoDB ingestion pipeline (chunk → embed → upsert)
├── tests/
│   ├── conftest.py
│   ├── test_m0_ingest.py  # Ingestion pipeline tests (chunking, embedding, upsert)
│   ├── test_m1_graph.py   # Node-level + integration tests
│   ├── test_m2_output.py  # Schema + output + CLI tests
│   └── test_voice.py      # ElevenLabs TTS + conditional routing tests
├── cli.py
├── pyproject.toml
└── .env.example
```

---

## Alternative: OpenAI instead of Gemini

Set these variables in `.env` (and remove `GEMINI_API_KEY`):

```
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
```

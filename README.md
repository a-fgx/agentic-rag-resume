# Agentic RAG Resume

A CLI agent that answers questions about Alain Feigneux's professional background using **LangGraph**, **Gemini**, **ElevenLabs TTS**, and **Pydantic**.

---

## Architecture

```
CLI → [load_context] → [route] → [generate] → [finalize] → JSON output
                                                    ↓ (--voice)
                                               [speak: ElevenLabs TTS]
```

| Node | Responsibility |
|---|---|
| `load_context` | Reads all `knowledge/*.md` files into a single context string |
| `route` | Validates the mode and loads the matching prompt template |
| `generate` | Calls the LLM with JSON mode, validates the response with Pydantic |
| `finalize` | Computes total timing and assembles the final envelope |
| `speak` | *(optional)* Converts `short_answer` to speech via ElevenLabs TTS |

The `speak` node is reached via a **conditional edge** from `finalize` — only when `--voice` is passed. This demonstrates LangGraph's conditional routing.

**Phase 2** (planned) will replace `load_context` with a MongoDB Atlas Vector Search retrieval node for chunk-level RAG.

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
| `ELEVENLABS_API_KEY` | Your ElevenLabs API key — only needed for `--voice` |
| `ELEVENLABS_VOICE_ID` | *(optional)* Your cloned voice ID — defaults to "Rachel" |

---

## Usage

```bash
# Standard Q&A
uv run rag "What is your experience with CI/CD?"

# Executive mode — board-level framing
uv run rag "What are your key achievements?" --mode executive

# Incident mode — SRE-style response
uv run rag "How did you handle production latency spikes?" --mode incident

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
  "citations": ["resume.md", "projects.md", "stories.md"],
  "timings_ms": {
    "load_context": 1.2,
    "llm": 843.5,
    "total": 844.7
  }
}
```

---

## Running tests

```bash
uv run pytest
```

All 44 tests run without any API key (all external calls are mocked).

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
│   └── graph.py        # LangGraph workflow (5 nodes + conditional edge)
├── tests/
│   ├── conftest.py
│   ├── test_m1_graph.py   # Node-level + integration tests
│   ├── test_m2_output.py  # Schema + output + CLI tests
│   └── test_voice.py      # ElevenLabs TTS + conditional routing tests
├── cli.py
└── pyproject.toml
```

---

## Phase 2 — MongoDB Atlas RAG (planned)

Phase 2 will add chunk-level retrieval:

1. Uncomment `pymongo>=4.7.0` in the `rag-phase2` dependency group:
   ```bash
   uv sync --group rag-phase2
   ```
2. Add `MONGODB_URI`, `MONGODB_DB`, `MONGODB_COLLECTION` to `.env`
3. Run ingestion: `uv run python -m app.ingest`
4. The `load_context` node becomes a `retrieve` node using `$vectorSearch`
5. Citations will reference specific chunks (e.g. `resume.md#3`) instead of file names

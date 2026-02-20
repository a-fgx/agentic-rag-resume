# PDD.md  
## Project: Parloa LangGraph LLM Proof

---

## 1. Overview

This project is a minimal technical prototype designed to demonstrate practical experience building and integrating AI/LLM-powered systems.

The prototype will:

- Use LangGraph to orchestrate an agent workflow  
- Implement RAG with a vector database (FAISS)  
- Support prompt routing across multiple modes  
- Enforce structured outputs via Pydantic  
- Provide traceability (citations) and latency metrics  
- Run entirely via CLI (no UI, no audio)

This is a technical proof, not a production system.

---

## 2. Objectives

The system must prove competence in:

- Agent workflow orchestration  
- Vector database integration  
- Prompt orchestration and routing  
- Structured output validation  
- Observability discipline (timings, traceable evidence)

---

## 3. Success Criteria

A reviewer can:

### 1. Run ingestion

```bash
python -m app.ingest
```

### 2. Run a query
```bash
python cli.py "How do you handle latency spikes?" --mode incident
```

### 3. Observe
- Structured JSON output
- Mode-specific behavior
- Retrieved chunk citations
- Retrieval and generation timings

If these are present and coherent, the proof is successful.

### 4. Scope

In Scope (v0)
- Markdown knowledge ingestion
- Chunking and embeddings
- FAISS local vector index
- Retrieval (top-k)
- LangGraph workflow orchestration
- Mode routing (qa, executive, incident)
- Structured JSON output (Pydantic)
- Citations (chunk IDs)
- Basic timing metrics

Out of Scope (v0)
- Audio (STT / TTS)
- Web interface
- Authentication
- Distributed infrastructure
- Advanced evaluation framework

### 5. Repository Structure
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
│   ├── schemas.py
│   ├── ingest.py
│   ├── rag.py
│   └── graph.py
│
├── store/
│   ├── faiss.index
│   └── chunks.jsonl
│
├── cli.py
├── requirements.txt
├── README.md
└── PDD.md

### 6. High-Level Architecture
CLI
  │
  ▼
LangGraph Workflow
  ├── Retrieve Node (FAISS)
  ├── Route Node (mode selection)
  ├── Generate Node (LLM + structured output)
  └── Finalize Node (citations + timings)

  Flow:
  START → retrieve → route → generate → finalize → END

### 7. Data Model

Knowledge Chunks

Each chunk stored in chunks.jsonl contains:
- chunk_id (stable identifier)
- source (filename)
- text (chunk content)

Runtime State (LangGraph)

State keys:
- mode
- question
- retrieved_chunks
- answer
- citations
- timings_ms


### 8. Vector Database Design

Embeddings
- Default model: text-embedding-3-large
- Configurable via environment variable

Similarity
- L2 normalization + inner product (cosine-like similarity)

Retrieval
- Default top-k: 5
- Include scores and chunk IDs

Artifacts produced:
- faiss.index
- chunks.jsonl

### 9. LangGraph Workflow Design

Node 1: Retrieve

Input: question
Output: retrieved_chunks, retrieval timing

Responsibilities:
- Embed query
- Perform FAISS search
- Attach scores and chunk IDs

⸻

Node 2: Route

Input: mode
Output: selected prompt template

Responsibilities:
- Validate mode
- Map to:
- qa.txt
- executive.txt
- incident.txt

⸻

Node 3: Generate

Input:
- question
- retrieved_chunks
- selected prompt

Output:
- structured Answer object
- LLM timing

Responsibilities:
- Build prompt with context
- Call LLM
- Parse JSON via Pydantic
- Fail if schema invalid

⸻

Node 4: Finalize

Input:
- answer
- retrieved_chunks
- timings

Output:
- Final JSON envelope

Responsibilities:
- Select top 3 citations
- Compute total timing
- Return structured result

### 10. Output Schema

Answer Model
- short_answer: string
- technical_plan: list[string]
- risks: list[string]
- business_impact: list[string]
- follow_up_question: optional string

CLI Output Envelope
{
  "mode": "incident",
  "question": "...",
  "answer": { ... },
  "citations": ["resume.md#12", "projects.md#4"],
  "timings_ms": {
    "retrieve": 52.1,
    "llm": 820.4,
    "total": 903.8
  }
}

### 11. Prompt Strategy

system.txt
- Enterprise tone
- Concise and operational
- Grounded in provided context
- Strict JSON output

qa.txt
- Factual answers
- Context-only responses

executive.txt
- 60-second board-level framing
- Still grounded in evidence

incident.txt
- SRE-style
- Diagnosis
- Mitigation
- Prevention

### 12. CLI Specification

Command:
```bash
python cli.py "<question>" --mode qa|executive|incident --k 5
```

Behavior:
- Fail if index missing
- Print structured JSON to stdout
- Non-zero exit on fatal errors


### 13. Configuration

Environment variables:
- OPENAI_API_KEY (required)
- EMBEDDING_MODEL (optional)
- CHAT_MODEL (optional)
- TOP_K (optional)

### 14. Risks and Mitigation
Risk
Mitigation
Hallucination
Strict grounding + context-only rule
Weak retrieval
Improve chunking, increase top-k
JSON parsing failure
Pydantic validation
High latency
Use smaller chat model

### 15. Milestones

M0
Ingestion and FAISS working

M1
LangGraph workflow operational

M2
Structured output + citations + timings

M3 (optional)
Grounding verification node
Switch to Qdrant for more production-like architecture


### 16. Definition of Done

The prototype clearly demonstrates:
- Agent workflow orchestration via LangGraph
- Vector database integration
- Prompt routing logic
- Structured and validated outputs
- Traceable citations
- Basic observability discipline

No UI required.
No audio required.
Reproducible, inspectable proof of LLM system integration capability.



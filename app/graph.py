from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.play import play
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import ValidationError

from app.schemas import Answer, GraphState, OutputEnvelope

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
VALID_MODES = {"qa", "executive", "incident"}

# ---------------------------------------------------------------------------
# Lazy-initialised LLM / embedding client
# Gemini (via OpenAI-compatible API) when GEMINI_API_KEY is set, else OpenAI.
# ---------------------------------------------------------------------------
_client: OpenAI | None = None

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            _client = OpenAI(
                api_key=gemini_key, base_url=GEMINI_BASE_URL, max_retries=10
            )
        else:
            _client = OpenAI(max_retries=10)  # falls back to OPENAI_API_KEY
    return _client


# ---------------------------------------------------------------------------
# Lazy-initialised MongoDB collection
# ---------------------------------------------------------------------------
_mongo_collection = None


def _get_mongo_collection():
    global _mongo_collection
    if _mongo_collection is None:
        from pymongo import MongoClient

        mongo_client = MongoClient(os.getenv("MONGODB_URI"))
        _mongo_collection = mongo_client[os.getenv("MONGODB_DB", "ragresume")][
            os.getenv("MONGODB_COLLECTION", "chunks")
        ]
    return _mongo_collection


# ---------------------------------------------------------------------------
# Lazy-initialised ElevenLabs client
# ---------------------------------------------------------------------------
_el_client: ElevenLabs | None = None


def _get_el_client() -> ElevenLabs:
    global _el_client
    if _el_client is None:
        _el_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    return _el_client


# ---------------------------------------------------------------------------
# Node 1: retrieve
# ---------------------------------------------------------------------------

def retrieve_node(state: GraphState) -> GraphState:
    """Embed the question and retrieve the top-k chunks from MongoDB Atlas."""
    t0 = time.perf_counter()

    question = state.get("question", "")
    k = state.get("k", 5)
    embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

    client = _get_client()
    resp = client.embeddings.create(model=embed_model, input=[question])
    query_vec = resp.data[0].embedding

    collection = _get_mongo_collection()
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_vec,
                "numCandidates": k * 10,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_id": 1,
                "source": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    retrieved = list(collection.aggregate(pipeline))

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "retrieved_chunks": retrieved,
        "timings_ms": {**state.get("timings_ms", {}), "retrieve": elapsed},
    }


# ---------------------------------------------------------------------------
# Node 2: route
# ---------------------------------------------------------------------------

def route_node(state: GraphState) -> GraphState:
    """Validate mode and load the corresponding prompt template."""
    mode = state.get("mode", "qa")
    if mode not in VALID_MODES:
        mode = "qa"

    template = (PROMPTS_DIR / f"{mode}.txt").read_text(encoding="utf-8")
    return {"mode": mode, "prompt_template": template}


# ---------------------------------------------------------------------------
# Node 3: generate
# ---------------------------------------------------------------------------

def generate_node(state: GraphState) -> GraphState:
    """Build prompt from retrieved chunks, call LLM, validate with Pydantic."""
    t0 = time.perf_counter()

    chunks = state.get("retrieved_chunks", [])
    question = state.get("question", "")

    # Build context from retrieved chunks, tagging each with its chunk_id
    context = "\n\n---\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c in chunks
    )

    # Use .replace() instead of .format() — context may contain literal { } chars
    user_prompt = (
        state["prompt_template"]
        .replace("{context}", context)
        .replace("{question}", question)
    )

    system_prompt = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")

    client = _get_client()
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "gemini-2.0-flash"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_json = response.choices[0].message.content
    answer_dict = json.loads(raw_json)

    # Validate schema — raises ValidationError if the LLM response is malformed
    Answer.model_validate(answer_dict)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "answer": answer_dict,
        "timings_ms": {**state.get("timings_ms", {}), "llm": elapsed},
    }


# ---------------------------------------------------------------------------
# Node 4: finalize
# ---------------------------------------------------------------------------

def finalize_node(state: GraphState) -> GraphState:
    """Set citations from retrieved chunks and compute total timing."""
    chunks = state.get("retrieved_chunks", [])
    citations = [c["chunk_id"] for c in chunks]

    timings = {**state.get("timings_ms", {})}
    timings["total"] = round(sum(timings.values()), 1)
    return {"citations": citations, "timings_ms": timings}


# ---------------------------------------------------------------------------
# Node 5: speak  (only reached when voice=True — conditional edge)
# ---------------------------------------------------------------------------

# Default to "Rachel" voice; override with ELEVENLABS_VOICE_ID in .env
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


def speak_node(state: GraphState) -> GraphState:
    """Convert short_answer to speech and play it via ElevenLabs TTS."""
    t0 = time.perf_counter()

    text = state.get("answer", {}).get("short_answer", "")
    if not text:
        return {}

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID)
    client = _get_el_client()

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"),
    )
    play(audio)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {"timings_ms": {**state.get("timings_ms", {}), "tts": elapsed}}


# ---------------------------------------------------------------------------
# Conditional routing: finalize → speak (if voice) or END
# ---------------------------------------------------------------------------

def _route_after_finalize(state: GraphState) -> str:
    return "speak" if state.get("voice") else END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("route", route_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("speak", speak_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "route")
    workflow.add_edge("route", "generate")
    workflow.add_edge("generate", "finalize")
    workflow.add_conditional_edges("finalize", _route_after_finalize, {"speak": "speak", END: END})
    workflow.add_edge("speak", END)

    return workflow.compile()


_graph = build_graph()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_graph(
    question: str, mode: str = "qa", voice: bool = False, k: int = 5
) -> dict:
    """Run the full LangGraph workflow and return a serialisable OutputEnvelope dict."""
    initial_state: GraphState = {
        "question": question,
        "mode": mode,
        "voice": voice,
        "k": k,
        "retrieved_chunks": [],
        "prompt_template": "",
        "answer": {},
        "citations": [],
        "timings_ms": {},
    }

    final_state = _graph.invoke(initial_state)

    envelope = OutputEnvelope(
        mode=final_state["mode"],
        question=final_state["question"],
        answer=Answer.model_validate(final_state["answer"]),
        citations=final_state["citations"],
        timings_ms=final_state["timings_ms"],
    )
    return envelope.model_dump()

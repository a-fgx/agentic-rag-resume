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

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
VALID_MODES = {"qa", "executive", "incident"}

# ---------------------------------------------------------------------------
# Lazy-initialised LLM client — Gemini (via OpenAI-compatible API) or OpenAI
# Set GEMINI_API_KEY to use Gemini; otherwise falls back to OPENAI_API_KEY.
# ---------------------------------------------------------------------------
_client: OpenAI | None = None

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            _client = OpenAI(api_key=gemini_key, base_url=GEMINI_BASE_URL)
        else:
            _client = OpenAI()  # falls back to OPENAI_API_KEY
    return _client


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
# Node 1: load_context
# ---------------------------------------------------------------------------

def load_context_node(state: GraphState) -> GraphState:
    """Load all knowledge markdown files and concatenate into a single context string."""
    t0 = time.perf_counter()

    parts: list[str] = []
    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        parts.append(f"=== {md_file.name} ===\n\n{content}")

    context = "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(parts) + "\n\n"
    citations = [f.name for f in sorted(KNOWLEDGE_DIR.glob("*.md"))]

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "context": context,
        "citations": citations,
        "timings_ms": {**state.get("timings_ms", {}), "load_context": elapsed},
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
    """Build prompt, call LLM with JSON mode, validate output with Pydantic."""
    t0 = time.perf_counter()

    context = state.get("context", "")
    question = state.get("question", "")

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
    """Compute total timing."""
    timings = {**state.get("timings_ms", {})}
    timings["total"] = round(sum(timings.values()), 1)
    return {"timings_ms": timings}


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
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("route", route_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("speak", speak_node)

    workflow.set_entry_point("load_context")
    workflow.add_edge("load_context", "route")
    workflow.add_edge("route", "generate")
    workflow.add_edge("generate", "finalize")
    workflow.add_conditional_edges("finalize", _route_after_finalize, {"speak": "speak", END: END})
    workflow.add_edge("speak", END)

    return workflow.compile()


_graph = build_graph()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_graph(question: str, mode: str = "qa", voice: bool = False) -> dict:
    """Run the full LangGraph workflow and return a serialisable OutputEnvelope dict."""
    initial_state: GraphState = {
        "question": question,
        "mode": mode,
        "voice": voice,
        "context": "",
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

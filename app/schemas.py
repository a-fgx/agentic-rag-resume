from __future__ import annotations

from typing import Any, Optional, TypedDict

from pydantic import BaseModel


class GraphState(TypedDict, total=False):
    """LangGraph state — all fields optional to allow partial node updates."""

    mode: str
    question: str
    k: int                            # Number of chunks to retrieve (default: 5)
    voice: bool                       # If True, speak short_answer via ElevenLabs TTS
    retrieved_chunks: list[dict]      # Chunks returned by retrieve_node
    prompt_template: str
    answer: dict[str, Any]
    citations: list[str]              # Chunk IDs (e.g. "resume.md#3")
    timings_ms: dict[str, float]


class Answer(BaseModel):
    short_answer: str
    technical_plan: list[str]
    risks: list[str]
    business_impact: list[str]
    follow_up_question: Optional[str] = None


class OutputEnvelope(BaseModel):
    mode: str
    question: str
    answer: Answer
    citations: list[str]
    timings_ms: dict[str, float]

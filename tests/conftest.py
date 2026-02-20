"""Shared pytest fixtures."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

VALID_ANSWER_DICT = {
    "short_answer": "Alain reduced release cycle time by 30% using GitHub Actions.",
    "technical_plan": ["Introduced GitOps workflows", "Automated release gates"],
    "risks": ["Pipeline misconfiguration could block deployments"],
    "business_impact": ["30% reduction in post-release incidents"],
    "follow_up_question": "What monitoring is in place for the pipeline?",
}

VALID_ANSWER_JSON = json.dumps(VALID_ANSWER_DICT)


@pytest.fixture
def valid_answer_dict():
    return VALID_ANSWER_DICT.copy()


@pytest.fixture
def valid_answer_json():
    return VALID_ANSWER_JSON


# ---------------------------------------------------------------------------
# Phase 2 — sample chunks (simulated MongoDB retrieve results)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "resume.md#0",
            "source": "resume.md",
            "text": "Alain worked at Nexthink for 15 years.",
            "score": 0.95,
        },
        {
            "chunk_id": "resume.md#1",
            "source": "resume.md",
            "text": "He reduced release cycle time by 30%.",
            "score": 0.88,
        },
        {
            "chunk_id": "projects.md#0",
            "source": "projects.md",
            "text": "GitHub Actions release pipelines.",
            "score": 0.82,
        },
    ]


@pytest.fixture
def fake_embedding():
    """768-dim unit vector (matches Gemini text-embedding-004 dimensions)."""
    dim = 768
    return [1.0 / dim**0.5] * dim


# ---------------------------------------------------------------------------
# Mock OpenAI chat completion
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chat_response(valid_answer_json):
    """A MagicMock that looks like an OpenAI ChatCompletion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = valid_answer_json
    return mock_response


@pytest.fixture
def mock_embed_response(fake_embedding):
    """A MagicMock that looks like an OpenAI Embeddings response."""
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding)]
    return mock_response


# ---------------------------------------------------------------------------
# Minimal GraphState for node unit tests
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state():
    return {
        "question": "How did you reduce release cycle time?",
        "mode": "qa",
        "k": 5,
        "retrieved_chunks": [],
        "prompt_template": "",
        "answer": {},
        "citations": [],
        "timings_ms": {},
    }

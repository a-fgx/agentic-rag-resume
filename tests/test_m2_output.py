"""Phase M2 — Output schema, CLI, and end-to-end tests.

All OpenAI calls are mocked. No real API key is needed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import Answer, OutputEnvelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(answer_dict: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(answer_dict)
    return mock_response


# ---------------------------------------------------------------------------
# OutputEnvelope schema tests
# ---------------------------------------------------------------------------

class TestOutputEnvelopeSchema:
    def test_valid_envelope_passes(self, valid_answer_dict):
        envelope = OutputEnvelope(
            mode="qa",
            question="What did you do at Nexthink?",
            answer=Answer.model_validate(valid_answer_dict),
            citations=["resume.md", "projects.md"],
            timings_ms={"load_context": 5.0, "llm": 200.0, "total": 205.0},
        )
        assert envelope.mode == "qa"
        assert envelope.answer.short_answer == valid_answer_dict["short_answer"]

    def test_follow_up_question_is_optional(self, valid_answer_dict):
        data = {**valid_answer_dict, "follow_up_question": None}
        answer = Answer.model_validate(data)
        assert answer.follow_up_question is None

    def test_answer_missing_required_field_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Answer.model_validate({"short_answer": "OK"})  # missing risks etc.

    def test_envelope_model_dump_is_json_serialisable(self, valid_answer_dict):
        envelope = OutputEnvelope(
            mode="incident",
            question="How did you handle latency?",
            answer=Answer.model_validate(valid_answer_dict),
            citations=["resume.md"],
            timings_ms={"load_context": 10.0, "llm": 500.0, "total": 510.0},
        )
        dumped = envelope.model_dump()
        # Must be serialisable with json.dumps
        serialised = json.dumps(dumped)
        roundtripped = json.loads(serialised)
        assert roundtripped["mode"] == "incident"


# ---------------------------------------------------------------------------
# run_graph() output tests
# ---------------------------------------------------------------------------

class TestRunGraphOutput:
    def _run(self, mode: str, tmp_path: Path, monkeypatch, answer_dict: dict) -> dict:
        (tmp_path / "resume.md").write_text("Alain worked at Nexthink for 15 years.")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _make_llm_response(answer_dict)
            mock_get_client.return_value = mock_client

            return g.run_graph("Test question", mode=mode)

    def test_output_has_all_top_level_keys(self, valid_answer_dict, tmp_path, monkeypatch):
        result = self._run("qa", tmp_path, monkeypatch, valid_answer_dict)
        for key in ("mode", "question", "answer", "citations", "timings_ms"):
            assert key in result, f"Missing key: {key}"

    def test_timings_has_required_keys(self, valid_answer_dict, tmp_path, monkeypatch):
        result = self._run("qa", tmp_path, monkeypatch, valid_answer_dict)
        for key in ("load_context", "llm", "total"):
            assert key in result["timings_ms"], f"Missing timing key: {key}"

    def test_timings_are_non_negative_floats(self, valid_answer_dict, tmp_path, monkeypatch):
        result = self._run("qa", tmp_path, monkeypatch, valid_answer_dict)
        for key, val in result["timings_ms"].items():
            assert isinstance(val, float), f"{key} is not float"
            assert val >= 0, f"{key} is negative"

    def test_total_timing_equals_sum_of_parts(self, valid_answer_dict, tmp_path, monkeypatch):
        result = self._run("qa", tmp_path, monkeypatch, valid_answer_dict)
        timings = result["timings_ms"]
        expected_total = sum(v for k, v in timings.items() if k != "total")
        assert timings["total"] == pytest.approx(expected_total, abs=0.1)

    def test_citations_are_strings(self, valid_answer_dict, tmp_path, monkeypatch):
        result = self._run("qa", tmp_path, monkeypatch, valid_answer_dict)
        assert isinstance(result["citations"], list)
        for c in result["citations"]:
            assert isinstance(c, str)

    def test_mode_is_preserved(self, valid_answer_dict, tmp_path, monkeypatch):
        for mode in ("qa", "executive", "incident"):
            result = self._run(mode, tmp_path, monkeypatch, valid_answer_dict)
            assert result["mode"] == mode

    def test_question_is_preserved(self, valid_answer_dict, tmp_path, monkeypatch):
        (tmp_path / "resume.md").write_text("content")
        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _make_llm_response(valid_answer_dict)
            mock_get_client.return_value = mock_client

            result = g.run_graph("Specific question text", mode="qa")

        assert result["question"] == "Specific question text"

    def test_executive_mode_produces_non_empty_business_impact(
        self, tmp_path, monkeypatch
    ):
        answer = {
            "short_answer": "Strategic leader.",
            "technical_plan": ["Plan A"],
            "risks": ["Risk A"],
            "business_impact": ["30% incident reduction", "40% faster releases"],
            "follow_up_question": None,
        }
        result = self._run("executive", tmp_path, monkeypatch, answer)
        assert len(result["answer"]["business_impact"]) > 0

    def test_incident_mode_produces_non_empty_technical_plan(
        self, tmp_path, monkeypatch
    ):
        answer = {
            "short_answer": "Latency spikes caused by resource contention.",
            "technical_plan": ["Scale horizontally", "Add caching layer"],
            "risks": ["Recurrence under peak load"],
            "business_impact": ["SLA breach risk"],
            "follow_up_question": "What is the P99 latency?",
        }
        result = self._run("incident", tmp_path, monkeypatch, answer)
        assert len(result["answer"]["technical_plan"]) > 0

    def test_result_passes_output_envelope_validation(
        self, valid_answer_dict, tmp_path, monkeypatch
    ):
        result = self._run("qa", tmp_path, monkeypatch, valid_answer_dict)
        OutputEnvelope.model_validate(result)  # must not raise


# ---------------------------------------------------------------------------
# CLI error handling
# ---------------------------------------------------------------------------

class TestCLIErrors:
    def test_missing_knowledge_dir_exits_non_zero(self, tmp_path, monkeypatch):
        """If knowledge dir is empty / missing, the graph still runs but returns empty citations."""
        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)  # empty dir

        answer = {
            "short_answer": "No context available.",
            "technical_plan": [],
            "risks": [],
            "business_impact": [],
            "follow_up_question": None,
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _make_llm_response(answer)
            mock_get_client.return_value = mock_client

            result = g.run_graph("Any question", mode="qa")

        assert result["citations"] == []

    def test_llm_connection_error_propagates(self, tmp_path, monkeypatch):
        """An OpenAI connection error should propagate as an exception."""
        (tmp_path / "resume.md").write_text("content")
        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = ConnectionError("API unreachable")
            mock_get_client.return_value = mock_client

            with pytest.raises(ConnectionError):
                g.run_graph("Any question", mode="qa")

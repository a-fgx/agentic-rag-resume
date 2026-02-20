"""Phase M1 — LangGraph workflow tests.

All OpenAI calls are mocked. No real API key is needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import Answer, OutputEnvelope


# ---------------------------------------------------------------------------
# Node: load_context_node
# ---------------------------------------------------------------------------

class TestLoadContextNode:
    def test_loads_markdown_files(self, base_state, tmp_path, monkeypatch):
        """load_context_node reads all *.md files from KNOWLEDGE_DIR."""
        (tmp_path / "resume.md").write_text("# Resume\n\nAlain Feigneux.")
        (tmp_path / "projects.md").write_text("# Projects\n\nGitHub Actions.")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        from app.graph import load_context_node

        result = load_context_node(base_state)

        assert "resume.md" in result["context"]
        assert "projects.md" in result["context"]
        assert "Alain Feigneux" in result["context"]
        assert "GitHub Actions" in result["context"]

    def test_citations_are_filenames(self, base_state, tmp_path, monkeypatch):
        """Citations should list the loaded markdown filenames."""
        (tmp_path / "resume.md").write_text("Resume content.")
        (tmp_path / "stories.md").write_text("Stories content.")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        from app.graph import load_context_node

        result = load_context_node(base_state)

        assert set(result["citations"]) == {"resume.md", "stories.md"}

    def test_load_context_timing(self, base_state, tmp_path, monkeypatch):
        """load_context_node records load_context timing in timings_ms."""
        (tmp_path / "resume.md").write_text("content")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        from app.graph import load_context_node

        result = load_context_node(base_state)

        assert "load_context" in result["timings_ms"]
        assert result["timings_ms"]["load_context"] >= 0

    def test_missing_knowledge_dir_raises(self, base_state, tmp_path, monkeypatch):
        """load_context_node with empty dir produces empty context."""
        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)  # tmp_path is empty

        from app.graph import load_context_node

        result = load_context_node(base_state)
        # No files → parts list is empty → context is whitespace only
        assert result["citations"] == []


# ---------------------------------------------------------------------------
# Node: route_node
# ---------------------------------------------------------------------------

class TestRouteNode:
    def test_route_qa(self, base_state):
        from app.graph import route_node

        state = {**base_state, "mode": "qa"}
        result = route_node(state)

        assert result["mode"] == "qa"
        assert "{context}" in result["prompt_template"]
        assert "{question}" in result["prompt_template"]

    def test_route_executive(self, base_state):
        from app.graph import route_node

        state = {**base_state, "mode": "executive"}
        result = route_node(state)

        assert result["mode"] == "executive"
        assert "executive" in result["prompt_template"].lower() or "board" in result["prompt_template"].lower()

    def test_route_incident(self, base_state):
        from app.graph import route_node

        state = {**base_state, "mode": "incident"}
        result = route_node(state)

        assert result["mode"] == "incident"
        # incident.txt should contain SRE-related language
        assert any(word in result["prompt_template"].lower() for word in ["sre", "incident", "mitigation"])

    def test_invalid_mode_falls_back_to_qa(self, base_state):
        from app.graph import route_node

        state = {**base_state, "mode": "unknown_mode"}
        result = route_node(state)

        assert result["mode"] == "qa"

    def test_missing_mode_defaults_to_qa(self, base_state):
        from app.graph import route_node

        state = {k: v for k, v in base_state.items() if k != "mode"}
        result = route_node(state)

        assert result["mode"] == "qa"


# ---------------------------------------------------------------------------
# Node: generate_node
# ---------------------------------------------------------------------------

class TestGenerateNode:
    def test_valid_llm_response_produces_answer(self, base_state, mock_chat_response, valid_answer_dict):
        """generate_node returns a valid answer dict when LLM responds correctly."""
        state = {
            **base_state,
            "context": "Alain worked at Nexthink.",
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON.",
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_chat_response
            mock_get_client.return_value = mock_client

            from app.graph import generate_node

            result = generate_node(state)

        assert "answer" in result
        Answer.model_validate(result["answer"])  # must not raise
        assert result["answer"]["short_answer"] == valid_answer_dict["short_answer"]

    def test_llm_timing_is_recorded(self, base_state, mock_chat_response):
        state = {
            **base_state,
            "context": "context",
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON.",
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_chat_response
            mock_get_client.return_value = mock_client

            from app.graph import generate_node

            result = generate_node(state)

        assert "llm" in result["timings_ms"]
        assert result["timings_ms"]["llm"] >= 0

    def test_invalid_schema_raises_validation_error(self, base_state):
        """generate_node raises ValidationError when LLM returns wrong schema."""
        from pydantic import ValidationError

        bad_json = json.dumps({"wrong_field": "bad response"})
        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = bad_json

        state = {
            **base_state,
            "context": "context",
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON.",
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = bad_response
            mock_get_client.return_value = mock_client

            from app.graph import generate_node

            with pytest.raises(ValidationError):
                generate_node(state)

    def test_curly_braces_in_context_do_not_break_prompt(self, base_state, mock_chat_response):
        """Context with { } chars must not cause a KeyError (use .replace, not .format)."""
        state = {
            **base_state,
            "context": "Here is some JSON: {'key': 'value'} and code {var}",
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON.",
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_chat_response
            mock_get_client.return_value = mock_client

            from app.graph import generate_node

            # Should not raise KeyError or IndexError
            result = generate_node(state)
            assert "answer" in result


# ---------------------------------------------------------------------------
# Node: finalize_node
# ---------------------------------------------------------------------------

class TestFinalizeNode:
    def test_total_timing_is_sum(self, base_state):
        from app.graph import finalize_node

        state = {**base_state, "timings_ms": {"load_context": 10.0, "llm": 200.0}}
        result = finalize_node(state)

        assert result["timings_ms"]["total"] == pytest.approx(210.0)

    def test_all_timing_keys_present(self, base_state):
        from app.graph import finalize_node

        state = {**base_state, "timings_ms": {"load_context": 5.0, "llm": 100.0}}
        result = finalize_node(state)

        assert {"load_context", "llm", "total"} <= result["timings_ms"].keys()

    def test_empty_timings_produces_zero_total(self, base_state):
        from app.graph import finalize_node

        state = {**base_state, "timings_ms": {}}
        result = finalize_node(state)

        assert result["timings_ms"]["total"] == 0.0


# ---------------------------------------------------------------------------
# Full graph integration (all external calls mocked)
# ---------------------------------------------------------------------------

class TestFullGraphIntegration:
    def test_graph_invoke_produces_valid_envelope(
        self, base_state, mock_chat_response, tmp_path, monkeypatch
    ):
        """End-to-end graph run with mocked LLM and tmp knowledge dir."""
        (tmp_path / "resume.md").write_text("Alain worked at Nexthink for 15 years.")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_chat_response
            mock_get_client.return_value = mock_client

            final_state = g._graph.invoke({
                **base_state,
                "mode": "qa",
            })

        # Must be valid OutputEnvelope
        envelope = OutputEnvelope(
            mode=final_state["mode"],
            question=final_state["question"],
            answer=Answer.model_validate(final_state["answer"]),
            citations=final_state["citations"],
            timings_ms=final_state["timings_ms"],
        )
        assert envelope.mode == "qa"
        assert envelope.question == base_state["question"]
        assert isinstance(envelope.citations, list)
        assert "total" in envelope.timings_ms

    def test_run_graph_returns_serialisable_dict(
        self, mock_chat_response, tmp_path, monkeypatch
    ):
        """run_graph() returns a plain dict that passes OutputEnvelope.model_validate."""
        (tmp_path / "resume.md").write_text("Alain worked at Nexthink.")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_chat_response
            mock_get_client.return_value = mock_client

            result = g.run_graph("What did Alain do at Nexthink?", mode="qa")

        assert isinstance(result, dict)
        OutputEnvelope.model_validate(result)

        # All top-level keys must be present
        for key in ("mode", "question", "answer", "citations", "timings_ms"):
            assert key in result

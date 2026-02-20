"""Phase M1 — LangGraph workflow tests.

All OpenAI and MongoDB calls are mocked. No API key or Atlas cluster needed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.schemas import Answer, OutputEnvelope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(mock_chat_response, fake_embedding):
    """Returns a mock OpenAI client handling both embeddings and chat completions."""
    mock_embed = MagicMock()
    mock_embed.data = [MagicMock(embedding=fake_embedding)]

    mock = MagicMock()
    mock.embeddings.create.return_value = mock_embed
    mock.chat.completions.create.return_value = mock_chat_response
    return mock


def _mock_collection(chunks):
    """Returns a mock MongoDB collection that returns chunks from aggregate()."""
    mock = MagicMock()
    mock.aggregate.return_value = iter(chunks)
    return mock


# ---------------------------------------------------------------------------
# Node: retrieve_node
# ---------------------------------------------------------------------------

class TestRetrieveNode:
    def test_returns_retrieved_chunks(self, base_state, sample_chunks, fake_embedding):
        mock_embed = MagicMock()
        mock_embed.data = [MagicMock(embedding=fake_embedding)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_embed

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=_mock_collection(sample_chunks)):
            from app.graph import retrieve_node
            result = retrieve_node(base_state)

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == len(sample_chunks)

    def test_chunks_have_required_keys(self, base_state, sample_chunks, fake_embedding):
        mock_embed = MagicMock()
        mock_embed.data = [MagicMock(embedding=fake_embedding)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_embed

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=_mock_collection(sample_chunks)):
            from app.graph import retrieve_node
            result = retrieve_node(base_state)

        for chunk in result["retrieved_chunks"]:
            assert {"chunk_id", "source", "text"} <= chunk.keys()

    def test_retrieve_timing_is_recorded(self, base_state, sample_chunks, fake_embedding):
        mock_embed = MagicMock()
        mock_embed.data = [MagicMock(embedding=fake_embedding)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_embed

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=_mock_collection(sample_chunks)):
            from app.graph import retrieve_node
            result = retrieve_node(base_state)

        assert "retrieve" in result["timings_ms"]
        assert result["timings_ms"]["retrieve"] >= 0

    def test_uses_k_from_state(self, base_state, sample_chunks, fake_embedding):
        """retrieve_node passes state['k'] as the pipeline limit."""
        mock_embed = MagicMock()
        mock_embed.data = [MagicMock(embedding=fake_embedding)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_embed
        mock_coll = _mock_collection(sample_chunks)

        state = {**base_state, "k": 3}

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=mock_coll):
            from app.graph import retrieve_node
            retrieve_node(state)

        pipeline_arg = mock_coll.aggregate.call_args[0][0]
        assert pipeline_arg[0]["$vectorSearch"]["limit"] == 3

    def test_default_k_is_5(self, fake_embedding):
        """When k is absent from state, retrieve_node defaults to 5."""
        mock_embed = MagicMock()
        mock_embed.data = [MagicMock(embedding=fake_embedding)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_embed
        mock_coll = _mock_collection([])

        state = {"question": "test question", "timings_ms": {}}

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=mock_coll):
            from app.graph import retrieve_node
            retrieve_node(state)

        pipeline_arg = mock_coll.aggregate.call_args[0][0]
        assert pipeline_arg[0]["$vectorSearch"]["limit"] == 5


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
    def test_valid_llm_response_produces_answer(
        self, base_state, mock_chat_response, valid_answer_dict, sample_chunks
    ):
        state = {
            **base_state,
            "retrieved_chunks": sample_chunks,
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

    def test_llm_timing_is_recorded(self, base_state, mock_chat_response, sample_chunks):
        state = {
            **base_state,
            "retrieved_chunks": sample_chunks,
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

    def test_invalid_schema_raises_validation_error(self, base_state, sample_chunks):
        from pydantic import ValidationError

        bad_json = json.dumps({"wrong_field": "bad response"})
        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = bad_json

        state = {
            **base_state,
            "retrieved_chunks": sample_chunks,
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON.",
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = bad_response
            mock_get_client.return_value = mock_client

            from app.graph import generate_node
            with pytest.raises(ValidationError):
                generate_node(state)

    def test_curly_braces_in_chunk_text_do_not_break_prompt(
        self, base_state, mock_chat_response
    ):
        """Chunk text with { } chars must not cause a KeyError (use .replace, not .format)."""
        chunks_with_braces = [
            {
                "chunk_id": "resume.md#0",
                "source": "resume.md",
                "text": "Here is some JSON: {'key': 'value'} and code {var}",
                "score": 0.9,
            }
        ]
        state = {
            **base_state,
            "retrieved_chunks": chunks_with_braces,
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON.",
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_chat_response
            mock_get_client.return_value = mock_client

            from app.graph import generate_node
            result = generate_node(state)
            assert "answer" in result

    def test_chunk_ids_appear_in_prompt(self, base_state, mock_chat_response, sample_chunks):
        """generate_node tags each chunk with its chunk_id in the context."""
        state = {
            **base_state,
            "retrieved_chunks": sample_chunks,
            "prompt_template": "Context:\n{context}\n\nQuestion: {question}\n\nReturn JSON.",
        }

        with patch("app.graph._get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_chat_response
            mock_get_client.return_value = mock_client

            from app.graph import generate_node
            generate_node(state)

        # Verify chunk_ids appear in the user prompt sent to the LLM
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        user_message = call_kwargs["messages"][1]["content"]
        assert "resume.md#0" in user_message
        assert "projects.md#0" in user_message


# ---------------------------------------------------------------------------
# Node: finalize_node
# ---------------------------------------------------------------------------

class TestFinalizeNode:
    def test_total_timing_is_sum(self, base_state):
        from app.graph import finalize_node

        state = {**base_state, "timings_ms": {"retrieve": 10.0, "llm": 200.0}}
        result = finalize_node(state)

        assert result["timings_ms"]["total"] == pytest.approx(210.0)

    def test_all_timing_keys_present(self, base_state):
        from app.graph import finalize_node

        state = {**base_state, "timings_ms": {"retrieve": 5.0, "llm": 100.0}}
        result = finalize_node(state)

        assert {"retrieve", "llm", "total"} <= result["timings_ms"].keys()

    def test_empty_timings_produces_zero_total(self, base_state):
        from app.graph import finalize_node

        state = {**base_state, "timings_ms": {}}
        result = finalize_node(state)

        assert result["timings_ms"]["total"] == 0.0

    def test_citations_come_from_retrieved_chunks(self, base_state, sample_chunks):
        from app.graph import finalize_node

        state = {**base_state, "retrieved_chunks": sample_chunks, "timings_ms": {}}
        result = finalize_node(state)

        assert result["citations"] == [c["chunk_id"] for c in sample_chunks]

    def test_empty_chunks_gives_empty_citations(self, base_state):
        from app.graph import finalize_node

        state = {**base_state, "retrieved_chunks": [], "timings_ms": {}}
        result = finalize_node(state)

        assert result["citations"] == []


# ---------------------------------------------------------------------------
# Full graph integration (all external calls mocked)
# ---------------------------------------------------------------------------

class TestFullGraphIntegration:
    def test_graph_invoke_produces_valid_envelope(
        self, base_state, mock_chat_response, sample_chunks, fake_embedding
    ):
        """End-to-end graph run with mocked embeddings, MongoDB, and LLM."""
        mock_client = _mock_client(mock_chat_response, fake_embedding)

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=_mock_collection(sample_chunks)):
            import app.graph as g
            final_state = g._graph.invoke({**base_state, "mode": "qa"})

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
        self, mock_chat_response, sample_chunks, fake_embedding
    ):
        """run_graph() returns a plain dict that passes OutputEnvelope.model_validate."""
        mock_client = _mock_client(mock_chat_response, fake_embedding)

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=_mock_collection(sample_chunks)):
            import app.graph as g
            result = g.run_graph("What did Alain do at Nexthink?", mode="qa")

        assert isinstance(result, dict)
        OutputEnvelope.model_validate(result)

        for key in ("mode", "question", "answer", "citations", "timings_ms"):
            assert key in result

    def test_citations_use_chunk_id_format(
        self, mock_chat_response, sample_chunks, fake_embedding
    ):
        """Citations in the envelope must be chunk IDs, not plain filenames."""
        mock_client = _mock_client(mock_chat_response, fake_embedding)

        with patch("app.graph._get_client", return_value=mock_client), \
             patch("app.graph._get_mongo_collection", return_value=_mock_collection(sample_chunks)):
            import app.graph as g
            result = g.run_graph("Question", mode="qa")

        import re
        for citation in result["citations"]:
            assert re.match(r"^[\w.]+#\d+$", citation), f"Bad citation format: {citation}"

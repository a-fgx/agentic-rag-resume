"""Voice / TTS tests — ElevenLabs is fully mocked, no API key needed."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# speak_node unit tests
# ---------------------------------------------------------------------------

class TestSpeakNode:
    def _state_with_answer(self, short_answer: str = "Alain reduced incidents by 30%.") -> dict:
        return {
            "voice": True,
            "answer": {
                "short_answer": short_answer,
                "technical_plan": [],
                "risks": [],
                "business_impact": [],
                "follow_up_question": None,
            },
            "timings_ms": {"retrieve": 5.0, "llm": 200.0, "total": 205.0},
        }

    def test_speak_node_calls_tts(self):
        """speak_node calls ElevenLabs TTS with the short_answer text."""
        state = self._state_with_answer("Alain reduced incidents by 30%.")

        with patch("app.graph._get_el_client") as mock_get_el, \
             patch("app.graph.play") as mock_play:
            mock_el = MagicMock()
            mock_el.text_to_speech.convert.return_value = iter([b"audio_bytes"])
            mock_get_el.return_value = mock_el

            from app.graph import speak_node
            speak_node(state)

        mock_el.text_to_speech.convert.assert_called_once()
        call_kwargs = mock_el.text_to_speech.convert.call_args
        assert "Alain reduced incidents by 30%." in str(call_kwargs)
        mock_play.assert_called_once()

    def test_speak_node_records_tts_timing(self):
        """speak_node adds 'tts' key to timings_ms."""
        state = self._state_with_answer()

        with patch("app.graph._get_el_client") as mock_get_el, \
             patch("app.graph.play"):
            mock_el = MagicMock()
            mock_el.text_to_speech.convert.return_value = iter([b"audio"])
            mock_get_el.return_value = mock_el

            from app.graph import speak_node
            result = speak_node(state)

        assert "tts" in result.get("timings_ms", {})
        assert result["timings_ms"]["tts"] >= 0

    def test_speak_node_skips_when_no_text(self):
        """speak_node returns empty dict without calling TTS if short_answer is empty."""
        state = {**self._state_with_answer(), "answer": {"short_answer": ""}}

        with patch("app.graph._get_el_client") as mock_get_el, \
             patch("app.graph.play") as mock_play:
            from app.graph import speak_node
            result = speak_node(state)

        mock_get_el.assert_not_called()
        mock_play.assert_not_called()
        assert result == {}


# ---------------------------------------------------------------------------
# Conditional edge routing tests
# ---------------------------------------------------------------------------

class TestConditionalVoiceRouting:
    def test_voice_false_routes_to_end(self):
        """With voice=False, _route_after_finalize returns END."""
        from langgraph.graph import END
        from app.graph import _route_after_finalize

        assert _route_after_finalize({"voice": False}) == END

    def test_voice_true_routes_to_speak(self):
        """With voice=True, _route_after_finalize returns 'speak'."""
        from app.graph import _route_after_finalize

        assert _route_after_finalize({"voice": True}) == "speak"

    def test_voice_missing_routes_to_end(self):
        """With voice key absent, defaults to END."""
        from langgraph.graph import END
        from app.graph import _route_after_finalize

        assert _route_after_finalize({}) == END


# ---------------------------------------------------------------------------
# Full graph integration — voice=True path (all mocked)
# ---------------------------------------------------------------------------

class TestFullGraphWithVoice:
    def test_graph_calls_speak_when_voice_true(
        self, mock_chat_response, tmp_path, monkeypatch
    ):
        """Full graph with voice=True reaches speak_node and calls TTS."""
        (tmp_path / "resume.md").write_text("Alain worked at Nexthink.")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        mock_coll = MagicMock()
        mock_coll.aggregate.return_value = iter([
            {"chunk_id": "resume.md#0", "source": "resume.md", "text": "Alain worked at Nexthink.", "score": 0.9}
        ])

        with patch("app.graph._get_client") as mock_get_client, \
             patch("app.graph._get_mongo_collection", return_value=mock_coll), \
             patch("app.graph._get_el_client") as mock_get_el, \
             patch("app.graph.play") as mock_play:

            mock_llm = MagicMock()
            mock_llm.chat.completions.create.return_value = mock_chat_response
            
            mock_embed = MagicMock()
            mock_embed.data = [MagicMock(embedding=[0.1]*768)]
            mock_llm.embeddings.create.return_value = mock_embed
            
            mock_get_client.return_value = mock_llm

            mock_el = MagicMock()
            mock_el.text_to_speech.convert.return_value = iter([b"audio"])
            mock_get_el.return_value = mock_el

            g.run_graph("What did Alain do?", mode="qa", voice=True)

        mock_play.assert_called_once()

    def test_graph_skips_speak_when_voice_false(
        self, mock_chat_response, tmp_path, monkeypatch
    ):
        """Full graph with voice=False must NOT call TTS."""
        (tmp_path / "resume.md").write_text("Alain worked at Nexthink.")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        mock_coll = MagicMock()
        mock_coll.aggregate.return_value = iter([
            {"chunk_id": "resume.md#0", "source": "resume.md", "text": "Alain worked at Nexthink.", "score": 0.9}
        ])

        with patch("app.graph._get_client") as mock_get_client, \
             patch("app.graph._get_mongo_collection", return_value=mock_coll), \
             patch("app.graph.play") as mock_play:

            mock_llm = MagicMock()
            mock_llm.chat.completions.create.return_value = mock_chat_response

            mock_embed = MagicMock()
            mock_embed.data = [MagicMock(embedding=[0.1]*768)]
            mock_llm.embeddings.create.return_value = mock_embed

            mock_get_client.return_value = mock_llm

            g.run_graph("What did Alain do?", mode="qa", voice=False)

        mock_play.assert_not_called()

    def test_tts_timing_present_when_voice_true(
        self, mock_chat_response, tmp_path, monkeypatch
    ):
        """Output timings_ms includes 'tts' key when voice=True."""
        (tmp_path / "resume.md").write_text("content")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        mock_coll = MagicMock()
        mock_coll.aggregate.return_value = iter([
            {"chunk_id": "resume.md#0", "source": "resume.md", "text": "content", "score": 0.9}
        ])

        with patch("app.graph._get_client") as mock_get_client, \
             patch("app.graph._get_mongo_collection", return_value=mock_coll), \
             patch("app.graph._get_el_client") as mock_get_el, \
             patch("app.graph.play"):

            mock_llm = MagicMock()
            mock_llm.chat.completions.create.return_value = mock_chat_response

            mock_embed = MagicMock()
            mock_embed.data = [MagicMock(embedding=[0.1]*768)]
            mock_llm.embeddings.create.return_value = mock_embed

            mock_get_client.return_value = mock_llm

            mock_el = MagicMock()
            mock_el.text_to_speech.convert.return_value = iter([b"audio"])
            mock_get_el.return_value = mock_el

            result = g.run_graph("Question", mode="qa", voice=True)

        assert "tts" in result["timings_ms"]

    def test_tts_timing_absent_when_voice_false(
        self, mock_chat_response, tmp_path, monkeypatch
    ):
        """Output timings_ms must NOT include 'tts' when voice=False."""
        (tmp_path / "resume.md").write_text("content")

        import app.graph as g
        monkeypatch.setattr(g, "KNOWLEDGE_DIR", tmp_path)

        mock_coll = MagicMock()
        mock_coll.aggregate.return_value = iter([
            {"chunk_id": "resume.md#0", "source": "resume.md", "text": "content", "score": 0.9}
        ])

        with patch("app.graph._get_client") as mock_get_client, \
             patch("app.graph._get_mongo_collection", return_value=mock_coll):
            mock_llm = MagicMock()
            mock_llm.chat.completions.create.return_value = mock_chat_response

            mock_embed = MagicMock()
            mock_embed.data = [MagicMock(embedding=[0.1]*768)]
            mock_llm.embeddings.create.return_value = mock_embed

            mock_get_client.return_value = mock_llm

            result = g.run_graph("Question", mode="qa", voice=False)

        assert "tts" not in result["timings_ms"]

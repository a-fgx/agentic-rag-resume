"""Phase M0 — Ingestion pipeline tests.

All OpenAI and MongoDB calls are mocked. No API key or Atlas cluster needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from app.ingest import chunk_document, embed_text, ensure_vector_index, upsert_chunks


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

class TestChunkDocument:
    def test_single_short_paragraph_becomes_one_chunk(self):
        text = "Alain worked at Nexthink for 15 years."
        chunks = chunk_document("resume.md", text)
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "resume.md#0"
        assert chunks[0]["source"] == "resume.md"
        assert chunks[0]["text"] == text

    def test_chunk_ids_are_sequential(self):
        # Three paragraphs of ~100 words each → should split into 2 chunks (max_words=200)
        para = " ".join(["word"] * 100)
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = chunk_document("resume.md", text, max_words=200)
        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] == "resume.md#0"
        assert chunks[1]["chunk_id"] == "resume.md#1"

    def test_chunk_id_format(self):
        para = " ".join(["word"] * 50)
        text = "\n\n".join([para] * 6)
        chunks = chunk_document("projects.md", text, max_words=100)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"projects.md#{i}"

    def test_source_is_preserved_in_all_chunks(self):
        para = " ".join(["word"] * 100)
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = chunk_document("stories.md", text, max_words=150)
        for chunk in chunks:
            assert chunk["source"] == "stories.md"

    def test_empty_paragraphs_are_skipped(self):
        text = "First paragraph.\n\n\n\n\nSecond paragraph."
        chunks = chunk_document("resume.md", text)
        assert len(chunks) == 1  # short enough to stay in one chunk
        assert "First paragraph." in chunks[0]["text"]
        assert "Second paragraph." in chunks[0]["text"]

    def test_returns_list_of_dicts_with_required_keys(self):
        chunks = chunk_document("resume.md", "Some text.")
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert {"chunk_id", "source", "text"} <= chunk.keys()


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:
    def _make_mock_client(self, embedding: list[float]) -> MagicMock:
        mock_client = MagicMock()

        def side_effect(model, input):
            mock_response = MagicMock()
            # Return one embedding per input text
            mock_response.data = [MagicMock(embedding=embedding) for _ in input]
            return mock_response

        mock_client.embeddings.create.side_effect = side_effect
        return mock_client

    def test_returns_list_of_lists(self):
        vec = [0.1, 0.2, 0.3]
        mock_client = self._make_mock_client(vec)

        result = embed_text(["text A", "text B"], mock_client, model="test-model")

        assert isinstance(result, list)
        assert len(result) == 2
        for item in result:
            assert isinstance(item, list)
            assert item == vec

    def test_embedding_content_matches_mock(self):
        vec = [1.0, 0.0, 0.0]
        mock_client = self._make_mock_client(vec)

        result = embed_text(["hello"], mock_client, model="test-model")

        assert result[0] == vec

    def test_batches_calls_for_large_input(self):
        """250 texts with BATCH_SIZE=100 → 3 API calls."""
        vec = [0.1]
        mock_client = self._make_mock_client(vec)

        texts = ["text"] * 250
        result = embed_text(texts, mock_client, model="test-model")

        # 250 / 100 = 3 calls (ceil)
        assert mock_client.embeddings.create.call_count == 3
        assert len(result) == 250

    def test_single_text_makes_one_call(self):
        vec = [0.5, 0.5]
        mock_client = self._make_mock_client(vec)

        embed_text(["single text"], mock_client)

        mock_client.embeddings.create.assert_called_once()

    def test_empty_input_returns_empty_list(self):
        mock_client = MagicMock()
        result = embed_text([], mock_client)
        assert result == []
        mock_client.embeddings.create.assert_not_called()


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------

class TestUpsertChunks:
    def test_drops_and_inserts(self):
        mock_collection = MagicMock()
        chunks = [
            {"chunk_id": "resume.md#0", "source": "resume.md", "text": "Text A"},
            {"chunk_id": "resume.md#1", "source": "resume.md", "text": "Text B"},
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        upsert_chunks(chunks, embeddings, mock_collection)

        mock_collection.drop.assert_called_once()
        mock_collection.insert_many.assert_called_once()

    def test_documents_include_embedding_key(self):
        mock_collection = MagicMock()
        chunks = [{"chunk_id": "resume.md#0", "source": "resume.md", "text": "Text"}]
        embeddings = [[0.1, 0.2, 0.3]]

        upsert_chunks(chunks, embeddings, mock_collection)

        inserted_docs = mock_collection.insert_many.call_args[0][0]
        assert len(inserted_docs) == 1
        assert inserted_docs[0]["embedding"] == [0.1, 0.2, 0.3]
        assert inserted_docs[0]["chunk_id"] == "resume.md#0"

    def test_empty_chunks_skips_insert(self):
        mock_collection = MagicMock()
        upsert_chunks([], [], mock_collection)
        mock_collection.drop.assert_called_once()
        mock_collection.insert_many.assert_not_called()


# ---------------------------------------------------------------------------
# ensure_vector_index
# ---------------------------------------------------------------------------

class TestEnsureVectorIndex:
    def test_calls_create_search_index(self):
        mock_collection = MagicMock()
        ensure_vector_index(mock_collection, num_dimensions=768)
        mock_collection.create_search_index.assert_called_once()

    def test_index_definition_has_correct_fields(self):
        mock_collection = MagicMock()
        ensure_vector_index(mock_collection, num_dimensions=768)

        call_arg = mock_collection.create_search_index.call_args[0][0]
        assert call_arg["name"] == "vector_index"
        assert call_arg["type"] == "vectorSearch"
        fields = call_arg["definition"]["fields"]
        assert len(fields) == 1
        assert fields[0]["type"] == "vector"
        assert fields[0]["path"] == "embedding"
        assert fields[0]["numDimensions"] == 768
        assert fields[0]["similarity"] == "cosine"

    def test_already_exists_does_not_raise(self):
        mock_collection = MagicMock()
        mock_collection.create_search_index.side_effect = Exception("already exists")

        # Must not propagate the exception
        ensure_vector_index(mock_collection, num_dimensions=768)

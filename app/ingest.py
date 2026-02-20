"""Phase 2 — Knowledge ingestion script.

Chunks markdown files, embeds them with Gemini (or OpenAI), and upserts
into MongoDB Atlas for vector search.

Usage:
    uv run python -m app.ingest
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Step 1 — Load markdown files
# ---------------------------------------------------------------------------

def load_markdown_files(knowledge_dir: Path) -> list[dict[str, str]]:
    """Return [{"source": "resume.md", "text": "..."}, ...] for all *.md files."""
    docs = []
    for md_file in sorted(knowledge_dir.glob("*.md")):
        docs.append({
            "source": md_file.name,
            "text": md_file.read_text(encoding="utf-8").strip(),
        })
    return docs


# ---------------------------------------------------------------------------
# Step 2 — Chunk documents
# ---------------------------------------------------------------------------

def chunk_document(source: str, text: str, max_words: int = 200) -> list[dict[str, str]]:
    """Split on double newline, accumulate paragraphs until ~max_words.

    Returns:
        [{"chunk_id": "resume.md#0", "source": "resume.md", "text": "..."}, ...]
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict[str, str]] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > max_words and current_parts:
            chunks.append({
                "chunk_id": f"{source}#{len(chunks)}",
                "source": source,
                "text": "\n\n".join(current_parts),
            })
            current_parts = []
            current_words = 0
        current_parts.append(para)
        current_words += para_words

    if current_parts:
        chunks.append({
            "chunk_id": f"{source}#{len(chunks)}",
            "source": source,
            "text": "\n\n".join(current_parts),
        })

    return chunks


# ---------------------------------------------------------------------------
# Step 3 — Embed texts
# ---------------------------------------------------------------------------

def embed_text(
    texts: list[str], client: OpenAI, model: str = EMBED_MODEL
) -> list[list[float]]:
    """Batch-embed texts (BATCH_SIZE per API call). Returns list of float lists."""
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


# ---------------------------------------------------------------------------
# Step 4 — Upsert to MongoDB
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunks: list[dict], embeddings: list[list[float]], collection
) -> None:
    """Drop and recreate the collection, then insert all chunks with embeddings."""
    collection.drop()
    docs = [
        {**chunk, "embedding": emb}
        for chunk, emb in zip(chunks, embeddings)
    ]
    if docs:
        collection.insert_many(docs)


# ---------------------------------------------------------------------------
# Step 5 — Create / ensure vector index
# ---------------------------------------------------------------------------

def ensure_vector_index(collection, num_dimensions: int = 768) -> None:
    """Create Atlas vectorSearch index named 'vector_index'. Skip if already exists."""
    try:
        collection.create_search_index({
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": num_dimensions,
                        "similarity": "cosine",
                    }
                ]
            },
        })
    except Exception:
        # Index already exists or cluster doesn't support it — skip silently
        pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Load → chunk → embed → upsert → create index."""
    from pymongo import MongoClient

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        client = OpenAI(
            api_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=10,
        )
    else:
        client = OpenAI(max_retries=10)

    mongo_client = MongoClient(os.getenv("MONGODB_URI"))
    db = mongo_client[os.getenv("MONGODB_DB", "ragresume")]
    collection = db[os.getenv("MONGODB_COLLECTION", "chunks")]

    model = os.getenv("EMBEDDING_MODEL", EMBED_MODEL)

    print("Loading knowledge files...")
    docs = load_markdown_files(KNOWLEDGE_DIR)
    print(f"  Loaded {len(docs)} file(s).")

    print("Chunking documents...")
    all_chunks: list[dict] = []
    for doc in docs:
        chunks = chunk_document(doc["source"], doc["text"])
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunk(s)")

    print(f"Embedding {len(all_chunks)} chunk(s) with model={model}...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_text(texts, client, model=model)

    print("Upserting to MongoDB...")
    upsert_chunks(all_chunks, embeddings, collection)
    print(f"  Inserted {len(all_chunks)} document(s).")

    num_dim = len(embeddings[0]) if embeddings else 768
    print(f"Creating vector index (dim={num_dim})...")
    ensure_vector_index(collection, num_dimensions=num_dim)
    print("Done. Wait ~60 s for Atlas to build the index before querying.")


if __name__ == "__main__":
    main()

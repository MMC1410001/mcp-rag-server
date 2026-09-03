"""ChromaDB vector store wrapper."""

import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings

from embeddings import embed, embed_one

# Persist data in ./chroma_db relative to this file
_DB_PATH = str(Path(__file__).parent / "chroma_db")
_COLLECTION_NAME = "rag_documents"


def _get_collection():
    client = chromadb.PersistentClient(path=_DB_PATH)
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[str], metadata) -> int:
    """
    Embed and store chunks. Returns count added.

    `metadata` accepts either:
      - dict: same metadata applied to every chunk (legacy behavior)
      - list[dict]: per-chunk metadata, must have same length as `chunks`
    """
    if not chunks:
        return 0

    if isinstance(metadata, list):
        if len(metadata) != len(chunks):
            raise ValueError(
                f"metadata list length ({len(metadata)}) does not match chunks ({len(chunks)})"
            )
        metadatas = [m.copy() for m in metadata]
    elif isinstance(metadata, dict):
        metadatas = [metadata.copy() for _ in chunks]
    else:
        raise TypeError(f"metadata must be dict or list[dict], got {type(metadata).__name__}")

    collection = _get_collection()
    embeddings = embed(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query: str, n_results: int = 5) -> list[dict]:
    """
    Semantic search. Returns list of:
    {"text": str, "score": float, "metadata": dict}
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []

    n_results = min(n_results, count)
    query_embedding = embed_one(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text": doc,
            "score": round(1 - dist, 4),  # cosine similarity
            "metadata": meta,
        })
    return output


def list_documents() -> list[dict]:
    """Return unique documents (by source_url) currently indexed."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    seen = {}
    for meta in results["metadatas"]:
        url = meta.get("source_url", "unknown")
        if url not in seen:
            seen[url] = {
                "source_url": url,
                "filename": meta.get("filename", ""),
                "doc_type": meta.get("doc_type", ""),
            }
    return list(seen.values())


def delete_document(source_url: str) -> int:
    """Delete all chunks for a given source URL. Returns deleted count."""
    collection = _get_collection()
    results = collection.get(where={"source_url": source_url}, include=[])
    ids = results["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)

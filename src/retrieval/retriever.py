"""Similarity search over the indexed chunks.

Thin retrieval layer over the vector store: it owns the default top-k policy so
callers (the MCP tools, the demo entry point) do not each hard-code it.
"""

from src.utils import helpers
from src.vectordb import vector_store


def retrieve(query: str, n_results: int | None = None) -> list[dict]:
    """Return the chunks most similar to `query`, best match first."""
    if n_results is None:
        n_results = helpers.get("retrieval", "n_results")
    return vector_store.search(query, n_results=n_results)


def retrieve_context(question: str, n_chunks: int | None = None) -> list[dict]:
    """Retrieve chunks intended as LLM grounding context for `question`."""
    if n_chunks is None:
        n_chunks = helpers.get("retrieval", "n_context_chunks")
    return vector_store.search(question, n_results=n_chunks)

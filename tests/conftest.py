"""Shared pytest fixtures for the mcp-rag-server test suite.

Strategy:
- ChromaDB runs in a per-test temp directory (no shared state, no cleanup needed).
- Embedding model calls are stubbed with a deterministic hash-based vector so the
  sentence-transformers model never has to load in CI.
- The Anthropic client is stubbed so no network calls or API keys are needed.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Make the project root importable so `src.*` resolves regardless of where
# pytest is run from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Deterministic embedding stub — no model download required in CI
# ---------------------------------------------------------------------------

EMBED_DIM = 384  # matches sentence-transformers/all-MiniLM-L6-v2


def _deterministic_vector(text: str) -> list[float]:
    """Hash the text into a stable float vector. Same input → same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat the digest until we have enough bytes; map each byte to a [-1, 1] float.
    needed = EMBED_DIM
    repeats = (needed // len(digest)) + 1
    raw = (digest * repeats)[:needed]
    return [(b - 128) / 128.0 for b in raw]


@pytest.fixture
def fake_embed(monkeypatch):
    """Patch embeddings.embed / embed_one to deterministic stubs."""
    from src.embeddings import embedder as embeddings
    from src.vectordb import vector_store

    def _embed(texts: list[str]) -> list[list[float]]:
        return [_deterministic_vector(t) for t in texts]

    def _embed_one(text: str) -> list[float]:
        return _deterministic_vector(text)

    monkeypatch.setattr(embeddings, "embed", _embed)
    monkeypatch.setattr(embeddings, "embed_one", _embed_one)
    # vector_store imports these at module load via `from ...embedder import ...`,
    # so we also patch the names bound there.
    monkeypatch.setattr(vector_store, "embed", _embed)
    monkeypatch.setattr(vector_store, "embed_one", _embed_one)


# ---------------------------------------------------------------------------
# Isolated ChromaDB per test
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_db(tmp_path, monkeypatch, fake_embed):
    """Point vector_store at a fresh ChromaDB under tmp_path. Yields the module."""
    from src.vectordb import vector_store

    db_dir = tmp_path / "chroma_db"
    monkeypatch.setattr(vector_store, "_DB_PATH", str(db_dir))
    return vector_store


# ---------------------------------------------------------------------------
# Anthropic client stub
# ---------------------------------------------------------------------------


class _StubMessage:
    def __init__(self, text: str):
        self.content = [SimpleNamespace(text=text)]


class _StubMessages:
    def __init__(self, response_text: str = "Stubbed answer."):
        self._response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _StubMessage(self._response_text)


class StubAnthropicClient:
    """Drop-in replacement for anthropic.Anthropic in tests."""

    def __init__(self, response_text: str = "Stubbed answer."):
        self.messages = _StubMessages(response_text)


@pytest.fixture
def stub_anthropic():
    return StubAnthropicClient()

"""Sentence-transformers embedding wrapper (no API key required).

The heavy `sentence_transformers` dependency is imported lazily so that this
module can be imported in environments (e.g. CI test runs) that stub `embed`
and `embed_one` without needing the model installed.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def get_model():
    # all-MiniLM-L6-v2: ~80MB, fast, good quality for semantic search
    from sentence_transformers import SentenceTransformer  # lazy import
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]

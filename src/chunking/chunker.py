"""Split extracted document text into overlapping chunks for embedding."""

from src.utils import helpers


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks by word count.

    Defaults come from config.yaml (chunking.chunk_size / chunking.overlap) and
    match the previous hard-coded values of 500 and 50.
    """
    if chunk_size is None:
        chunk_size = helpers.get("chunking", "chunk_size")
    if overlap is None:
        overlap = helpers.get("chunking", "overlap")

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(words):
            break
        start = end - overlap
    return chunks

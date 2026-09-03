"""Document loading, parsing, and chunking from Google Drive links."""

import re
import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

import gdown
import httpx

logger = logging.getLogger(__name__)


def _gdrive_file_id(url: str) -> Optional[str]:
    """Extract file ID from various Google Drive URL formats."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _gdrive_folder_id(url: str) -> Optional[str]:
    match = re.search(r"/drive/folders/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def _is_gdrive_folder(url: str) -> bool:
    return "/drive/folders/" in url


def download_from_gdrive(url: str, dest_dir: str) -> tuple[str, str]:
    """Download a file from Google Drive. Returns (local_path, filename)."""
    file_id = _gdrive_file_id(url)
    if not file_id:
        raise ValueError(f"Could not extract file ID from URL: {url}")

    dest_path = os.path.join(dest_dir, file_id)

    output = gdown.download(url, dest_path, quiet=True, fuzzy=True)
    if output is None:
        raise RuntimeError("gdown failed to download the file. Ensure the file is publicly shared.")

    actual_path = output if os.path.exists(output) else dest_path
    return actual_path, os.path.basename(actual_path)


def download_folder_from_gdrive(url: str, dest_dir: str) -> list[str]:
    """Download all files from a Google Drive folder. Returns list of local paths."""
    folder_id = _gdrive_folder_id(url)
    if not folder_id:
        raise ValueError(f"Could not extract folder ID from URL: {url}")

    paths = gdown.download_folder(id=folder_id, output=dest_dir, quiet=True, use_cookies=False)
    if not paths:
        raise RuntimeError("gdown failed to download the folder. Ensure the folder is publicly shared.")
    return paths


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by word count."""
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


def parse_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(para.text for para in doc.paragraphs)


def parse_image_with_claude(path: str, client) -> str:
    """Use Claude vision to extract content description from an image."""
    import base64
    from pathlib import Path

    suffix = Path(path).suffix.lower()
    media_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = media_map.get(suffix, "image/png")

    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_data},
                    },
                    {"type": "text", "text": "Describe all text, diagrams, and key information in this image in detail."},
                ],
            }
        ],
    )
    return response.content[0].text


def _parse_file(local_path: str, client) -> tuple[str, str]:
    """Parse a single file. Returns (text, doc_type)."""
    ext = Path(local_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(local_path), "pdf"
    elif ext in (".docx", ".doc"):
        return parse_docx(local_path), "docx"
    elif ext in (".txt", ".md"):
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), "text"
    elif ext in (".png", ".jpg", ".jpeg", ".drawing"):
        return parse_image_with_claude(local_path, client), "image"
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_document(url: str, client) -> tuple[list[str], list[dict], dict]:
    """
    Download and parse a Google Drive file or folder.

    Returns:
        chunks:           list[str]   — text chunks ready for embedding
        chunk_metadatas:  list[dict]  — per-chunk metadata (parallel to chunks),
                                        each containing source_url, filename, doc_type
        summary:          dict        — aggregate metadata for the ingest response
                                        (filename, doc_type, file_count, skipped_files)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        if _is_gdrive_folder(url):
            paths = download_folder_from_gdrive(url, tmp_dir)
            all_chunks: list[str] = []
            all_metadatas: list[dict] = []
            processed_files: list[str] = []
            skipped_files: list[dict] = []

            for path in paths:
                filename = os.path.basename(path)
                try:
                    text, doc_type = _parse_file(path, client)
                except ValueError as e:
                    # Unsupported file type — record and skip
                    reason = f"unsupported file type ({Path(path).suffix or 'no extension'})"
                    skipped_files.append({"filename": filename, "reason": reason})
                    logger.warning("Skipping %s: %s", filename, reason)
                    continue
                except Exception as e:
                    # Parse failure (corrupt PDF, image API failure, etc.) — record and skip
                    reason = f"parse error: {type(e).__name__}: {e}"
                    skipped_files.append({"filename": filename, "reason": reason})
                    logger.warning("Skipping %s: %s", filename, reason)
                    continue

                file_chunks = chunk_text(text)
                if not file_chunks:
                    skipped_files.append({"filename": filename, "reason": "no extractable text"})
                    logger.warning("Skipping %s: no extractable text", filename)
                    continue

                per_file_meta = {
                    "source_url": url,
                    "filename": filename,
                    "doc_type": doc_type,
                    "parent": "gdrive_folder",
                }
                all_chunks.extend(file_chunks)
                all_metadatas.extend([per_file_meta.copy() for _ in file_chunks])
                processed_files.append(filename)

            if not all_chunks:
                raise RuntimeError(
                    f"No supported files could be indexed from the folder. "
                    f"Skipped: {skipped_files}"
                )

            summary = {
                "source_url": url,
                "filename": "gdrive_folder",
                "doc_type": "folder",
                "file_count": len(processed_files),
                "processed_files": processed_files,
                "skipped_files": skipped_files,
            }
            return all_chunks, all_metadatas, summary
        else:
            local_path, filename = download_from_gdrive(url, tmp_dir)
            text, doc_type = _parse_file(local_path, client)
            chunks = chunk_text(text)
            if not chunks:
                raise RuntimeError(f"No extractable text from {filename}")
            per_chunk_meta = {
                "source_url": url,
                "filename": filename,
                "doc_type": doc_type,
            }
            chunk_metadatas = [per_chunk_meta.copy() for _ in chunks]
            summary = {
                "source_url": url,
                "filename": filename,
                "doc_type": doc_type,
                "file_count": 1,
                "skipped_files": [],
            }
            return chunks, chunk_metadatas, summary

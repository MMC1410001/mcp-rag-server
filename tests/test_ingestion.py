"""Unit tests for ingestion.py — URL parsing, chunking, parsing dispatch."""

from __future__ import annotations

import pytest

from src.ingestion import loader as ingestion


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


class TestGdriveUrlParsing:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://drive.google.com/file/d/ABC123_xyz/view", "ABC123_xyz"),
            ("https://drive.google.com/open?id=XYZ-456", "XYZ-456"),
            ("https://drive.google.com/d/SHORT_ID", "SHORT_ID"),
        ],
    )
    def test_extracts_file_id_from_various_formats(self, url, expected):
        assert ingestion._gdrive_file_id(url) == expected

    def test_returns_none_for_unparseable_url(self):
        assert ingestion._gdrive_file_id("https://example.com/no-id-here") is None

    def test_extracts_folder_id(self):
        url = "https://drive.google.com/drive/folders/FOLDER_ABC?usp=sharing"
        assert ingestion._gdrive_folder_id(url) == "FOLDER_ABC"

    def test_is_gdrive_folder_detects_correctly(self):
        assert ingestion._is_gdrive_folder(
            "https://drive.google.com/drive/folders/abc"
        )
        assert not ingestion._is_gdrive_folder(
            "https://drive.google.com/file/d/abc/view"
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_produces_single_chunk(self):
        chunks = ingestion.chunk_text("hello world", chunk_size=500, overlap=50)
        assert chunks == ["hello world"]

    def test_empty_string_returns_no_chunks(self):
        assert ingestion.chunk_text("") == []
        assert ingestion.chunk_text("   ") == []

    def test_long_text_is_split_with_overlap(self):
        words = [f"w{i}" for i in range(120)]
        text = " ".join(words)
        chunks = ingestion.chunk_text(text, chunk_size=50, overlap=10)
        # 120 words, 50 per chunk, 10 overlap → step=40
        # chunk1: words[0:50], chunk2: words[40:90], chunk3: words[80:120]
        assert len(chunks) == 3
        assert chunks[0].split()[0] == "w0"
        assert chunks[1].split()[0] == "w40"
        assert chunks[2].split()[-1] == "w119"

    def test_overlap_actually_overlaps(self):
        text = " ".join(f"w{i}" for i in range(100))
        chunks = ingestion.chunk_text(text, chunk_size=40, overlap=10)
        # chunk1 ends at w39, chunk2 starts at w30 → overlap on w30..w39
        last_words_c1 = chunks[0].split()[-10:]
        first_words_c2 = chunks[1].split()[:10]
        assert last_words_c1 == first_words_c2


# ---------------------------------------------------------------------------
# _parse_file dispatch
# ---------------------------------------------------------------------------


class TestParseFileDispatch:
    def test_unsupported_extension_raises_value_error(self, tmp_path):
        bad = tmp_path / "diagram.drawio"
        bad.write_text("<xml/>")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestion._parse_file(str(bad), client=None)

    def test_txt_file_is_read_directly(self, tmp_path):
        f = tmp_path / "note.txt"
        f.write_text("hello from a text file")
        text, doc_type = ingestion._parse_file(str(f), client=None)
        assert "hello from a text file" in text
        assert doc_type == "text"

    def test_md_file_is_read_directly(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# heading\n\nbody")
        text, doc_type = ingestion._parse_file(str(f), client=None)
        assert "heading" in text
        assert doc_type == "text"


# ---------------------------------------------------------------------------
# load_document — folder path (the part that was bug-prone)
# ---------------------------------------------------------------------------


def _make_txt(path, content):
    p = path
    p.write_text(content)
    return str(p)


class TestLoadDocumentFolder:
    """Folder ingestion is where D1/D2/D3 lived. Cover each explicitly."""

    def test_returns_three_tuple_signature(self, tmp_path, monkeypatch):
        f1 = _make_txt(tmp_path / "file_a.txt", "alpha bravo charlie")
        monkeypatch.setattr(
            ingestion, "download_folder_from_gdrive", lambda url, dest: [f1]
        )
        url = "https://drive.google.com/drive/folders/FAKE"

        result = ingestion.load_document(url, client=None)
        assert len(result) == 3, "load_document must return (chunks, metadatas, summary)"
        chunks, metadatas, summary = result
        assert isinstance(chunks, list)
        assert isinstance(metadatas, list)
        assert isinstance(summary, dict)

    def test_chunks_and_metadatas_have_matching_length(self, tmp_path, monkeypatch):
        f1 = _make_txt(tmp_path / "file_a.txt", "a " * 200)
        f2 = _make_txt(tmp_path / "file_b.txt", "b " * 200)
        monkeypatch.setattr(
            ingestion, "download_folder_from_gdrive", lambda url, dest: [f1, f2]
        )

        chunks, metadatas, summary = ingestion.load_document(
            "https://drive.google.com/drive/folders/FAKE", client=None
        )
        assert len(chunks) == len(metadatas)
        assert summary["file_count"] == 2

    def test_all_unsupported_raises_runtime_error(self, tmp_path, monkeypatch):
        bad = tmp_path / "img.drawio"
        bad.write_text("<xml/>")
        monkeypatch.setattr(
            ingestion, "download_folder_from_gdrive", lambda url, dest: [str(bad)]
        )

        with pytest.raises(RuntimeError, match="No supported files"):
            ingestion.load_document(
                "https://drive.google.com/drive/folders/FAKE", client=None
            )


class TestLoadDocumentSingleFile:
    def test_single_file_returns_three_tuple(self, tmp_path, monkeypatch):
        f = _make_txt(tmp_path / "single.txt", "lorem ipsum dolor sit amet")
        monkeypatch.setattr(
            ingestion,
            "download_from_gdrive",
            lambda url, dest: (f, "single.txt"),
        )

        chunks, metadatas, summary = ingestion.load_document(
            "https://drive.google.com/file/d/FAKE/view", client=None
        )
        assert chunks
        assert len(chunks) == len(metadatas)
        assert summary["filename"] == "single.txt"
        assert summary["doc_type"] == "text"
        assert summary["file_count"] == 1
        # Every chunk's metadata should match the file
        for m in metadatas:
            assert m["filename"] == "single.txt"
            assert m["doc_type"] == "text"

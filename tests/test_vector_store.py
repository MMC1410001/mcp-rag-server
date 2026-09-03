"""Unit tests for vector_store — covers both legacy (dict) and new (list[dict]) metadata."""

from __future__ import annotations

import pytest


class TestAddChunksMetadataShape:
    def test_dict_metadata_applied_to_all_chunks(self, isolated_db):
        meta = {"source_url": "u1", "filename": "a.txt", "doc_type": "text"}
        count = isolated_db.add_chunks(["x", "y", "z"], meta)
        assert count == 3
        # All chunks should carry the same filename
        results = isolated_db.search("x", n_results=3)
        for r in results:
            assert r["metadata"]["filename"] == "a.txt"

    def test_list_metadata_per_chunk(self, isolated_db):
        chunks = ["alpha text one", "beta text two", "gamma text three"]
        metas = [
            {"source_url": "u1", "filename": "alpha.txt", "doc_type": "text"},
            {"source_url": "u1", "filename": "beta.txt", "doc_type": "text"},
            {"source_url": "u1", "filename": "gamma.txt", "doc_type": "text"},
        ]
        count = isolated_db.add_chunks(chunks, metas)
        assert count == 3
        results = isolated_db.search("alpha text one", n_results=3)
        filenames = {r["metadata"]["filename"] for r in results}
        # All three filenames should be retrievable
        assert {"alpha.txt", "beta.txt", "gamma.txt"} == filenames

    def test_list_length_mismatch_raises(self, isolated_db):
        with pytest.raises(ValueError, match="length"):
            isolated_db.add_chunks(["a", "b"], [{"filename": "x"}])

    def test_invalid_metadata_type_raises(self, isolated_db):
        with pytest.raises(TypeError):
            isolated_db.add_chunks(["a"], "not a dict")

    def test_empty_chunks_returns_zero(self, isolated_db):
        assert isolated_db.add_chunks([], {"filename": "x"}) == 0


class TestSearch:
    def test_empty_collection_returns_empty_list(self, isolated_db):
        assert isolated_db.search("anything") == []

    def test_search_returns_text_score_metadata(self, isolated_db):
        isolated_db.add_chunks(
            ["the quick brown fox"],
            {"source_url": "u", "filename": "f.txt", "doc_type": "text"},
        )
        results = isolated_db.search("the quick brown fox", n_results=1)
        assert len(results) == 1
        r = results[0]
        assert set(r.keys()) == {"text", "score", "metadata"}
        assert r["text"] == "the quick brown fox"
        assert 0.0 <= r["score"] <= 1.0


class TestListDocuments:
    def test_dedups_by_source_url(self, isolated_db):
        url = "https://example.com/folder"
        meta = {"source_url": url, "filename": "f1.docx", "doc_type": "docx"}
        # Add 10 chunks all sharing the same source_url
        isolated_db.add_chunks([f"chunk {i}" for i in range(10)], meta)

        docs = isolated_db.list_documents()
        assert len(docs) == 1
        assert docs[0]["source_url"] == url

    def test_multiple_sources_listed_separately(self, isolated_db):
        isolated_db.add_chunks(
            ["a"], {"source_url": "u1", "filename": "f1", "doc_type": "text"}
        )
        isolated_db.add_chunks(
            ["b"], {"source_url": "u2", "filename": "f2", "doc_type": "text"}
        )
        docs = isolated_db.list_documents()
        urls = {d["source_url"] for d in docs}
        assert urls == {"u1", "u2"}


class TestDeleteDocument:
    def test_delete_removes_only_matching_source(self, isolated_db):
        isolated_db.add_chunks(
            ["keep me"], {"source_url": "u1", "filename": "k", "doc_type": "text"}
        )
        isolated_db.add_chunks(
            ["delete me 1", "delete me 2"],
            {"source_url": "u2", "filename": "d", "doc_type": "text"},
        )

        deleted = isolated_db.delete_document("u2")
        assert deleted == 2

        remaining = isolated_db.list_documents()
        assert len(remaining) == 1
        assert remaining[0]["source_url"] == "u1"

    def test_delete_nonexistent_returns_zero(self, isolated_db):
        assert isolated_db.delete_document("does-not-exist") == 0

"""Regression tests pinned to the specific defects fixed during QA review.

Defect | Description
-------|------------
D1     | Folder ingestion only caught ValueError, so a corrupt PDF or any other
       | exception in _parse_file would crash the *entire* batch. After the fix,
       | per-file failures must be isolated and the rest of the folder must
       | still ingest.
D2     | All chunks in a folder shared filename="gdrive_folder". After the fix
       | each chunk must carry the *actual* filename it came from.
D3     | Skipped files vanished silently with no log or audit. After the fix
       | the load_document summary must include a `skipped_files` list with
       | filename + reason for each skip.

If any of these tests fail in the future, one of the QA-found bugs has regressed.
"""

from __future__ import annotations

import logging

import pytest

import ingestion


def _make_txt(path, content):
    path.write_text(content)
    return str(path)


# ---------------------------------------------------------------------------
# D1 — per-file error isolation
# ---------------------------------------------------------------------------


class TestD1_PerFileErrorIsolation:
    def test_one_corrupt_file_does_not_abort_batch(self, tmp_path, monkeypatch):
        good = _make_txt(tmp_path / "good.txt", "this is good content " * 20)
        bad = tmp_path / "corrupt.txt"
        bad.write_text("placeholder")

        # Simulate a runtime parse failure on 'bad' (e.g. corrupt PDF, image API error).
        original_parse = ingestion._parse_file

        def parsing(path, client):
            if path.endswith("corrupt.txt"):
                raise RuntimeError("simulated corruption")
            return original_parse(path, client)

        monkeypatch.setattr(ingestion, "_parse_file", parsing)
        monkeypatch.setattr(
            ingestion,
            "download_folder_from_gdrive",
            lambda url, dest: [good, str(bad)],
        )

        chunks, metadatas, summary = ingestion.load_document(
            "https://drive.google.com/drive/folders/FAKE", client=None
        )
        # Good file went through
        assert len(chunks) >= 1
        assert summary["file_count"] == 1
        # Bad file was recorded, not silently dropped
        skipped_names = [s["filename"] for s in summary["skipped_files"]]
        assert "corrupt.txt" in skipped_names

    def test_unsupported_type_is_skipped_not_raised(self, tmp_path, monkeypatch):
        good = _make_txt(tmp_path / "ok.txt", "good content " * 20)
        bad = tmp_path / "diagram.drawio"
        bad.write_text("<xml/>")

        monkeypatch.setattr(
            ingestion,
            "download_folder_from_gdrive",
            lambda url, dest: [good, str(bad)],
        )
        # Should NOT raise — drawio gets skipped, txt gets ingested
        chunks, metadatas, summary = ingestion.load_document(
            "https://drive.google.com/drive/folders/FAKE", client=None
        )
        assert summary["file_count"] == 1
        assert any(
            "drawio" in s["reason"] for s in summary["skipped_files"]
        ), "Skipped reason should identify the unsupported extension"


# ---------------------------------------------------------------------------
# D2 — per-file source attribution
# ---------------------------------------------------------------------------


class TestD2_PerFileSourceAttribution:
    def test_each_chunk_carries_its_originating_filename(
        self, tmp_path, monkeypatch
    ):
        f1 = _make_txt(tmp_path / "spec.txt", "specification body " * 30)
        f2 = _make_txt(tmp_path / "journey.txt", "user journey body " * 30)

        monkeypatch.setattr(
            ingestion, "download_folder_from_gdrive", lambda url, dest: [f1, f2]
        )

        chunks, metadatas, summary = ingestion.load_document(
            "https://drive.google.com/drive/folders/FAKE", client=None
        )

        filenames = {m["filename"] for m in metadatas}
        # Regression assertion: must contain the real filenames, NOT the legacy
        # bucket-name "gdrive_folder".
        assert "spec.txt" in filenames
        assert "journey.txt" in filenames
        assert "gdrive_folder" not in filenames, (
            "REGRESSION D2: chunks reverted to using the folder bucket name "
            "instead of per-file filenames."
        )

    def test_single_file_chunks_carry_real_filename(self, tmp_path, monkeypatch):
        f = _make_txt(tmp_path / "single.txt", "lone content " * 30)
        monkeypatch.setattr(
            ingestion,
            "download_from_gdrive",
            lambda url, dest: (f, "single.txt"),
        )

        chunks, metadatas, summary = ingestion.load_document(
            "https://drive.google.com/file/d/FAKE/view", client=None
        )
        assert all(m["filename"] == "single.txt" for m in metadatas)


# ---------------------------------------------------------------------------
# D3 — skip auditability (log + structured response)
# ---------------------------------------------------------------------------


class TestD3_SkipAuditability:
    def test_skipped_files_in_summary_have_filename_and_reason(
        self, tmp_path, monkeypatch
    ):
        good = _make_txt(tmp_path / "ok.txt", "good " * 30)
        bad = tmp_path / "weird.drawio"
        bad.write_text("<xml/>")

        monkeypatch.setattr(
            ingestion,
            "download_folder_from_gdrive",
            lambda url, dest: [good, str(bad)],
        )

        _, _, summary = ingestion.load_document(
            "https://drive.google.com/drive/folders/FAKE", client=None
        )

        assert "skipped_files" in summary
        assert len(summary["skipped_files"]) == 1
        entry = summary["skipped_files"][0]
        assert entry["filename"] == "weird.drawio"
        assert "reason" in entry and entry["reason"]

    def test_skip_emits_warning_log(self, tmp_path, monkeypatch, caplog):
        good = _make_txt(tmp_path / "ok.txt", "good " * 30)
        bad = tmp_path / "weird.drawio"
        bad.write_text("<xml/>")

        monkeypatch.setattr(
            ingestion,
            "download_folder_from_gdrive",
            lambda url, dest: [good, str(bad)],
        )

        with caplog.at_level(logging.WARNING, logger="ingestion"):
            ingestion.load_document(
                "https://drive.google.com/drive/folders/FAKE", client=None
            )

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("weird.drawio" in r.getMessage() for r in warnings), (
            "REGRESSION D3: skipped files no longer emit a warning log entry."
        )

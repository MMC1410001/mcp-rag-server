"""Unit tests for generator.py — answer formatting, no-context fallback, source dedup."""

from __future__ import annotations

import generator


class TestAnswerNoContext:
    def test_returns_safe_default_when_no_chunks(self, stub_anthropic):
        result = generator.answer("anything", [], stub_anthropic)
        assert result["sources"] == []
        assert "no relevant" in result["answer"].lower()
        # Claude should NOT be called when there is no context
        assert stub_anthropic.messages.calls == []


class TestAnswerWithContext:
    def _make_chunk(self, text, filename="report.pdf"):
        return {
            "text": text,
            "score": 0.9,
            "metadata": {"source_url": "u1", "filename": filename, "doc_type": "pdf"},
        }

    def test_calls_claude_with_context_block(self, stub_anthropic):
        chunks = [self._make_chunk("alpha content")]
        generator.answer("question?", chunks, stub_anthropic)

        assert len(stub_anthropic.messages.calls) == 1
        call = stub_anthropic.messages.calls[0]
        user_msg = call["messages"][0]["content"]
        assert "alpha content" in user_msg
        assert "[Source: report.pdf]" in user_msg
        assert "question?" in user_msg

    def test_sources_are_deduplicated(self, stub_anthropic):
        chunks = [
            self._make_chunk("a", "file1.pdf"),
            self._make_chunk("b", "file1.pdf"),
            self._make_chunk("c", "file2.pdf"),
        ]
        result = generator.answer("q", chunks, stub_anthropic)
        assert set(result["sources"]) == {"file1.pdf", "file2.pdf"}

    def test_unknown_filename_falls_back_gracefully(self, stub_anthropic):
        chunks = [{"text": "orphan text", "score": 0.5, "metadata": {}}]
        result = generator.answer("q", chunks, stub_anthropic)
        assert "unknown" in result["sources"]

    def test_returns_stubbed_answer_text(self, stub_anthropic):
        chunks = [self._make_chunk("alpha")]
        stub_anthropic.messages._response_text = "Custom stub response"
        result = generator.answer("q", chunks, stub_anthropic)
        assert result["answer"] == "Custom stub response"

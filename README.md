# MCP RAG Server

A retrieval-augmented generation system exposed as an **MCP (Model Context Protocol) server**.
Point it at a document or an entire Google Drive folder; it parses, chunks, embeds and indexes
the content locally, then answers questions against it with citations.

Because it speaks MCP over stdio, any MCP-capable client (Claude Desktop, Claude Code, or your
own) can use it as a tool — the knowledge base becomes something the model can query directly.

## What it does

```
 Drive URL / local file
          │
          ▼
   ingestion.py ──── PDF · DOCX · TXT · MD · PNG/JPG (via Claude vision)
          │           chunking with configurable size + overlap
          ▼
   embeddings.py ─── sentence-transformers all-MiniLM-L6-v2 (runs locally, no API cost)
          │
          ▼
  vector_store.py ── ChromaDB, cosine similarity, persisted to disk
          │
          ▼
   generator.py ──── retrieve top-k, then Claude answers grounded in the chunks
          │
          ▼
     server.py ───── exposes the pipeline as five MCP tools
```

## MCP tools

| Tool | Purpose |
|---|---|
| `ingest_document` | Parse and index a file or an entire folder |
| `search_documents` | Semantic search, returns matching chunks with scores |
| `ask_question` | Full RAG — retrieve, then generate a grounded answer with sources |
| `list_documents` | What is currently in the knowledge base |
| `delete_document` | Remove a document and all of its chunks |

## Design notes

- **Embeddings run locally.** `all-MiniLM-L6-v2` means indexing a large folder costs nothing
  and needs no network round-trip per chunk. Only generation calls the API.
- **Folder ingestion is fault-tolerant.** An unsupported or corrupt file does not abort the
  batch — it is recorded in `skipped_files` with a reason and the rest continue.
- **Per-chunk metadata.** `add_chunks` takes either one dict for the whole batch or a parallel
  list, so a folder ingest attributes every chunk to the file it came from and citations stay
  accurate.
- **Images are documents too.** PNG/JPG go through Claude's vision API, so screenshots and
  scanned pages are searchable alongside text.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY
python demo.py              # indexes samples/ and asks a question
```

## Using it as an MCP server

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["/absolute/path/to/mcp-rag-server/server.py"],
      "env": { "ANTHROPIC_API_KEY": "your-key" }
    }
  }
}
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # no API key needed — the Anthropic client is stubbed
```

Covers ingestion (chunking, overlap, parser dispatch, folder fault-tolerance), the vector store
(add/search/list/delete, metadata handling), the generator (prompt construction, empty-context
behaviour), and a regression suite pinning previously-fixed bugs.

## Bringing your own documents

See [`samples/README.md`](samples/README.md) for accepted inputs and URL formats.

## Tech stack

Python · MCP SDK · Anthropic SDK · ChromaDB · sentence-transformers · pypdf · python-docx ·
Pillow · gdown · pytest

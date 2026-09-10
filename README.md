# MCP RAG Server

A retrieval-augmented generation system exposed as an **MCP (Model Context Protocol) server**.
Point it at a document or an entire Google Drive folder; it parses, chunks, embeds and indexes
the content locally, then answers questions against it with citations.

Because it speaks MCP over stdio, any MCP-capable client (Claude Desktop, Claude Code, or your
own) can use it as a tool, so the knowledge base becomes something the model can query directly.

## What it does

```
 Drive URL / local file
          │
          ▼
 src/ingestion/loader.py ──── PDF · DOCX · TXT · MD · PNG/JPG (via Claude vision)
          │
          ▼
 src/chunking/chunker.py ──── configurable size + overlap (config.yaml)
          │
          ▼
 src/embeddings/embedder.py ─ sentence-transformers all-MiniLM-L6-v2, runs locally
          │
          ▼
 src/vectordb/vector_store.py ─ ChromaDB, cosine similarity, persisted to disk
          │
          ▼
 src/retrieval/retriever.py ── top-k similarity search
          │
          ▼
 src/llm/llm_client.py ─────── Claude answers, grounded in the retrieved chunks
   + src/prompts/
          │
          ▼
 src/api/routes.py ─────────── exposes the pipeline as five MCP tools
          │
          ▼
        main.py ────────────── entry point
```

## Project structure

```
mcp-rag-server/
├── README.md
├── requirements.txt
├── .env.example            copy to .env and add ANTHROPIC_API_KEY
├── .gitignore
├── config.yaml             chunk size, model, DB path, top-k, logging
├── src/
│   ├── ingestion/          download and parse PDFs, DOCX, TXT, MD, images
│   ├── chunking/           split text into overlapping chunks
│   ├── embeddings/         convert chunks into vectors
│   ├── vectordb/           ChromaDB operations
│   ├── retrieval/          similarity search
│   ├── prompts/            prompt templates
│   ├── llm/                Claude API calls
│   ├── api/                MCP tool routes
│   └── utils/              config loading, paths, logging
├── tests/                  unit and regression tests
├── logs/                   app.log
└── main.py                 entry point
```

## MCP tools

| Tool | Purpose |
|---|---|
| `ingest_document` | Parse and index a file or an entire folder |
| `search_documents` | Semantic search, returns matching chunks with scores |
| `ask_question` | Full RAG: retrieve, then generate a grounded answer with sources |
| `list_documents` | What is currently in the knowledge base |
| `delete_document` | Remove a document and all of its chunks |

## Design notes

- **Embeddings run locally.** `all-MiniLM-L6-v2` means indexing a large folder costs nothing
  and needs no network round-trip per chunk. Only generation calls the API.
- **Folder ingestion is fault-tolerant.** An unsupported or corrupt file does not abort the
  batch. It is recorded in `skipped_files` with a reason and the rest continue.
- **Per-chunk metadata.** `add_chunks` takes either one dict for the whole batch or a parallel
  list, so a folder ingest attributes every chunk to the file it came from and citations stay
  accurate.
- **Images are documents too.** PNG/JPG go through Claude's vision API, so screenshots and
  scanned pages are searchable alongside text.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY
python main.py --demo       # indexes samples/ and asks a question
python main.py              # run the MCP server over stdio
```

## Configuration

`config.yaml` holds chunk size and overlap, the embedding model, the ChromaDB path and
collection, retrieval top-k, the Claude model and token limit, and logging. Every value
defaults to what the code used before, so the pipeline behaves the same if you leave it alone.

## Using it as an MCP server

```jsonc
// claude_desktop_config.json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["/absolute/path/to/mcp-rag-server/main.py"],
      "env": { "ANTHROPIC_API_KEY": "your-key" }
    }
  }
}
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # no API key needed, the Anthropic client is stubbed
```

Covers ingestion (chunking, overlap, parser dispatch, folder fault-tolerance), the vector store
(add/search/list/delete, metadata handling), the generator (prompt construction, empty-context
behaviour), and a regression suite pinning previously-fixed bugs.

## Bringing your own documents

See [`samples/README.md`](samples/README.md) for accepted inputs and URL formats.

## Tech stack

Python · MCP SDK · Anthropic SDK · ChromaDB · sentence-transformers · pypdf · python-docx ·
Pillow · gdown · pytest

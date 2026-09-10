"""Entry point for the RAG MCP server.

    cp .env.example .env      # then fill in ANTHROPIC_API_KEY
    pip install -r requirements.txt
    python main.py            # run the MCP server over stdio
    python main.py --demo     # index the bundled sample and ask one question

The MCP server exposes the pipeline as five tools. The demo drives the same
pipeline directly so you can see it work without wiring up an MCP client.
"""

import asyncio
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.chunking.chunker import chunk_text
from src.llm import llm_client
from src.retrieval import retriever
from src.utils import helpers
from src.vectordb import vector_store

load_dotenv()

SAMPLE = Path(__file__).parent / "samples" / "orbital-mechanics-primer.md"
QUESTION = "When does a bi-elliptic transfer beat a Hohmann transfer on propellant?"


def run_demo() -> None:
    """Index the bundled sample document, then ask a question about it."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"Indexing {SAMPLE.name} ...")
    chunks = chunk_text(SAMPLE.read_text(encoding="utf-8"))
    metadata = {
        "source_url": SAMPLE.name,
        "filename": SAMPLE.name,
        "doc_type": "md",
    }
    indexed = vector_store.add_chunks(chunks, metadata)
    print(f"Indexed {indexed} chunks.\n")

    print(f"Q: {QUESTION}")
    hits = retriever.retrieve(QUESTION, n_results=3)
    result = llm_client.answer(QUESTION, hits, client)
    print(f"A: {result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")


def main() -> None:
    helpers.setup_logging()
    if "--demo" in sys.argv:
        run_demo()
        return
    from src.api.routes import serve  # imported lazily so --demo needs no API key at import

    asyncio.run(serve())


if __name__ == "__main__":
    main()

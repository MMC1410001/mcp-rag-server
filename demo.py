"""End-to-end demo: index the bundled sample document, then ask a question about it.

    cp .env.example .env      # then fill in ANTHROPIC_API_KEY
    pip install -r requirements.txt
    python demo.py

The MCP server (server.py) exposes this same pipeline as five tools. This script just
drives it directly so you can see it work without wiring up an MCP client.
"""

import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

import generator
import ingestion
import vector_store

load_dotenv()

SAMPLE = Path(__file__).parent / "samples" / "orbital-mechanics-primer.md"
QUESTION = "When does a bi-elliptic transfer beat a Hohmann transfer on propellant?"


def main() -> None:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"Indexing {SAMPLE.name} ...")
    chunks = ingestion.chunk_text(SAMPLE.read_text(encoding="utf-8"))
    metadata = {
        "source_url": SAMPLE.name,
        "filename": SAMPLE.name,
        "doc_type": "md",
    }
    indexed = vector_store.add_chunks(chunks, metadata)
    print(f"Indexed {indexed} chunks.\n")

    print(f"Q: {QUESTION}")
    hits = vector_store.search(QUESTION, n_results=3)
    result = generator.answer(QUESTION, hits, client)
    print(f"A: {result['answer']}")
    print(f"\nSources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()

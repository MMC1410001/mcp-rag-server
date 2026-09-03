"""Claude API answer generation with retrieved context."""

import anthropic

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided document context.

Rules:
- Answer only from the provided context chunks
- If the context doesn't contain enough information, say so explicitly
- Cite the source filename when referencing specific information
- Be concise and accurate"""


def answer(question: str, context_chunks: list[dict], client: anthropic.Anthropic) -> dict:
    """
    Generate an answer using Claude with RAG context.
    Returns {"answer": str, "sources": list[str]}
    """
    if not context_chunks:
        return {
            "answer": "No relevant documents found in the knowledge base. Please ingest documents first.",
            "sources": [],
        }

    context_text = "\n\n---\n\n".join(
        f"[Source: {chunk['metadata'].get('filename', 'unknown')}]\n{chunk['text']}"
        for chunk in context_chunks
    )

    sources = list({
        chunk["metadata"].get("filename", "unknown")
        for chunk in context_chunks
    })

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            }
        ],
    )

    return {
        "answer": response.content[0].text,
        "sources": sources,
    }

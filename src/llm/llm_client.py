"""Claude API answer generation with retrieved context."""

import anthropic

from src.prompts import prompt_templates
from src.utils import helpers

SYSTEM_PROMPT = prompt_templates.SYSTEM_PROMPT


def answer(question: str, context_chunks: list[dict], client: anthropic.Anthropic) -> dict:
    """
    Generate an answer using Claude with RAG context.
    Returns {"answer": str, "sources": list[str]}
    """
    if not context_chunks:
        return {
            "answer": prompt_templates.NO_CONTEXT_ANSWER,
            "sources": [],
        }

    context_text = prompt_templates.CONTEXT_SEPARATOR.join(
        prompt_templates.CONTEXT_CHUNK_TEMPLATE.format(
            filename=chunk["metadata"].get("filename", "unknown"), text=chunk["text"]
        )
        for chunk in context_chunks
    )

    sources = list({
        chunk["metadata"].get("filename", "unknown")
        for chunk in context_chunks
    })

    response = client.messages.create(
        model=helpers.get("llm", "model"),
        max_tokens=helpers.get("llm", "max_tokens"),
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": prompt_templates.USER_TEMPLATE.format(
                    context=context_text, question=question
                ),
            }
        ],
    )

    return {
        "answer": response.content[0].text,
        "sources": sources,
    }

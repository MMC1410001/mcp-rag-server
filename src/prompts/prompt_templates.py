"""Prompt templates used by the generation layer."""

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based strictly on the provided document context.

Rules:
- Answer only from the provided context chunks
- If the context doesn't contain enough information, say so explicitly
- Cite the source filename when referencing specific information
- Be concise and accurate"""

CONTEXT_CHUNK_TEMPLATE = "[Source: {filename}]\n{text}"

CONTEXT_SEPARATOR = "\n\n---\n\n"

USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"

IMAGE_EXTRACTION_PROMPT = (
    "Describe all text, diagrams, and key information in this image in detail."
)

NO_CONTEXT_ANSWER = (
    "No relevant documents found in the knowledge base. Please ingest documents first."
)

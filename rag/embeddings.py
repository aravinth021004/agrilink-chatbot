"""
Document loading, chunking, and embedding for RAG pipeline.
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")


def load_and_chunk_documents(chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Load the knowledge base markdown file and split it into chunks.

    Args:
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Overlap between consecutive chunks to preserve context.

    Returns:
        List of Document objects (chunks).
    """
    knowledge_file = os.path.join(KNOWLEDGE_DIR, "agrilink-knowledge-base.md")

    if not os.path.exists(knowledge_file):
        raise FileNotFoundError(f"Knowledge base not found: {knowledge_file}")

    loader = TextLoader(knowledge_file, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n---\n", "\n## ", "\n### ", "\n\n", "\n", " "],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = "agrilink-knowledge-base"

    print(f"✅ Loaded knowledge base: {len(chunks)} chunks from {knowledge_file}")
    return chunks

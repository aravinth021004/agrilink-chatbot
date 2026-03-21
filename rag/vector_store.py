"""
ChromaDB vector store for document embeddings.
"""

import os
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

from .embeddings import load_and_chunk_documents


# Persist directory for ChromaDB
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# Embedding model (runs locally, no API cost)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Singleton
_vector_store = None
_embedding_function = None


def get_embedding_function():
    """Get or create the sentence transformer embedding function."""
    global _embedding_function
    if _embedding_function is None:
        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}...")
        _embedding_function = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
    return _embedding_function


def initialize_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Initialize or load the ChromaDB vector store.

    If the store already exists on disk and force_rebuild is False,
    it will be loaded from disk. Otherwise, documents are loaded,
    chunked, embedded, and stored.

    Args:
        force_rebuild: If True, rebuild the vector store from scratch.

    Returns:
        Chroma vector store instance.
    """
    global _vector_store

    if _vector_store is not None and not force_rebuild:
        return _vector_store

    embedding_fn = get_embedding_function()

    # Check if we have an existing persisted store
    if os.path.exists(PERSIST_DIR) and not force_rebuild:
        print(f"📂 Loading existing vector store from {PERSIST_DIR}...")
        _vector_store = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_fn,
            collection_name="agrilink_knowledge",
        )

        # Verify it has documents
        count = _vector_store._collection.count()
        if count > 0:
            print(f"✅ Vector store loaded: {count} documents")
            return _vector_store
        else:
            print("⚠️ Vector store is empty, rebuilding...")

    # Build from scratch
    print("🔨 Building vector store from knowledge base...")
    chunks = load_and_chunk_documents()

    _vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=PERSIST_DIR,
        collection_name="agrilink_knowledge",
    )

    print(f"✅ Vector store built and persisted: {len(chunks)} chunks")
    return _vector_store


def similarity_search(query: str, k: int = 4) -> list:
    """
    Search the vector store for chunks most relevant to the query.

    Args:
        query: The user's question.
        k: Number of top results to return.

    Returns:
        List of relevant Document objects.
    """
    store = initialize_vector_store()
    results = store.similarity_search(query, k=k)
    return results

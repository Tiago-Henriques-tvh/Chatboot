"""Embeddings and Qdrant vector-store singletons.

Heavy models and network clients are created lazily via ``lru_cache`` so that
importing this module (and anything that depends on it) stays cheap and unit
testable.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached HuggingFace embedding model instance."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        cache_folder=settings.hf_cache_dir,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_client():
    """Return a cached Qdrant client."""
    from qdrant_client import QdrantClient

    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=30.0,
    )


def ensure_collection() -> None:
    """Create the target collection if it does not already exist."""
    from qdrant_client.http.models import Distance, VectorParams

    client = get_client()
    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )


def get_vectorstore():
    """Return a LangChain Qdrant vector store bound to the training collection."""
    from langchain_qdrant import QdrantVectorStore

    ensure_collection()
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.collection_name,
        embedding=get_embeddings(),
    )

"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Values are read (case-insensitively) from environment variables and an
    optional local ``.env`` file. Defaults are chosen so the app can also run
    outside Docker (e.g. ``localhost`` Qdrant) for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Storage / ingestion
    data_path: str = "./data"
    hf_cache_dir: str | None = None

    # Qdrant vector database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "training_knowledge"

    # Ollama LLM
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:3b"
    request_timeout: float = 120.0
    temperature: float = 0.2

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Retrieval / chunking
    top_k: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()

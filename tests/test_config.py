"""Tests for configuration loading."""

from __future__ import annotations

from app.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.qdrant_port == 6333
    assert s.collection_name == "training_knowledge"
    assert s.embedding_dim == 384
    assert s.model_name == "qwen2.5:3b"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "qwen2.5:1.5b")
    monkeypatch.setenv("QDRANT_PORT", "7000")
    s = Settings(_env_file=None)
    assert s.model_name == "qwen2.5:1.5b"
    assert s.qdrant_port == 7000

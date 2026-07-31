"""Pytest fixtures with lightweight stubs for heavy optional dependencies.

These stubs let the test-suite exercise our own logic (tools, ingestion,
FastAPI endpoints, config) without installing torch, sentence-transformers,
Ollama, or a running Qdrant instance.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_doc():
    def _make(content: str, source: str = "notes.md"):
        return types.SimpleNamespace(page_content=content, metadata={"source": source})

    return _make


class _FakeStore:
    """In-memory stand-in for a LangChain vector store."""

    def __init__(self, docs=None):
        self.docs = docs or []
        self.added = []

    def similarity_search(self, query, k=5):
        return self.docs[:k]

    def add_documents(self, docs, ids=None):
        self.added.append((list(docs), list(ids or [])))
        return ids or []


@pytest.fixture
def fake_store():
    return _FakeStore


@pytest.fixture(autouse=True)
def _stub_duckduckgo(monkeypatch):
    """Provide a fake ``duckduckgo_search`` module so search_web is testable."""
    mod = types.ModuleType("duckduckgo_search")

    class _DDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results=5):
            return [
                {
                    "title": f"Result for {query}",
                    "body": "example body",
                    "href": "https://example.com",
                }
            ]

    mod.DDGS = _DDGS
    monkeypatch.setitem(sys.modules, "duckduckgo_search", mod)
    return mod

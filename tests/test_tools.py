"""Tests for the ReAct tools."""

from __future__ import annotations

from app import tools


def test_search_training_database_formats_results(monkeypatch, fake_store, fake_doc):
    store = fake_store([fake_doc("weekly mileage 40km", "run.md")])
    monkeypatch.setattr(tools, "get_vectorstore", lambda: store)

    out = tools.search_training_database.invoke({"query": "mileage"})

    assert "weekly mileage 40km" in out
    assert "run.md" in out


def test_search_training_database_empty(monkeypatch, fake_store):
    monkeypatch.setattr(tools, "get_vectorstore", lambda: fake_store([]))
    out = tools.search_training_database.invoke({"query": "nothing"})
    assert "No matching documents" in out


def test_search_training_database_handles_error(monkeypatch):
    def boom():
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(tools, "get_vectorstore", boom)
    out = tools.search_training_database.invoke({"query": "x"})
    assert "Error querying the training database" in out


def test_search_web_returns_results():
    out = tools.search_web.invoke({"query": "marathon record"})
    assert "Result for marathon record" in out
    assert "https://example.com" in out


def test_generate_training_plan_dedupes_and_wraps(monkeypatch, fake_store, fake_doc):
    dup = fake_doc("long run 20km", "log.md")
    store = fake_store([dup, dup])
    monkeypatch.setattr(tools, "get_vectorstore", lambda: store)

    out = tools.generate_training_plan.invoke({"goal": "12-week half marathon"})

    assert "GOAL: 12-week half marathon" in out
    assert out.count("long run 20km") == 1
    assert "INSTRUCTIONS" in out


def test_all_tools_registered():
    names = {t.name for t in tools.ALL_TOOLS}
    assert names == {
        "search_training_database",
        "search_web",
        "generate_training_plan",
    }

"""Tests for ingestion helpers."""

from __future__ import annotations

from pathlib import Path

from app import ingest


def test_discover_filters_supported(tmp_path: Path):
    (tmp_path / "a.md").write_text("hi", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "d.csv").write_text("skip", encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "e.md").write_text("deep", encoding="utf-8")

    found = ingest._discover(tmp_path)
    names = {p.name for p in found}

    assert names == {"a.md", "b.txt", "c.pdf", "e.md"}


def test_deterministic_id_stable():
    a = ingest._deterministic_id("text", "src.md", 0)
    b = ingest._deterministic_id("text", "src.md", 0)
    c = ingest._deterministic_id("text", "src.md", 1)
    assert a == b
    assert a != c


def test_ingest_missing_dir_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    try:
        ingest.ingest(str(missing))
    except FileNotFoundError as exc:
        assert "not found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")


def test_ingest_empty_dir_returns_zero(tmp_path: Path):
    assert ingest.ingest(str(tmp_path)) == 0


def test_plan_to_text_flattens_workouts():
    plan = {
        "type": "marathon",
        "race": "Berlin Marathon 2025",
        "workouts": [
            {"day": "Monday", "workout": "10km easy run"},
            {"day": "Tuesday", "workout": "intervals 6x800m"},
        ],
    }
    text, meta = ingest._plan_to_text(plan)
    assert "Training plan type: marathon" in text
    assert "Race: Berlin Marathon 2025" in text
    assert "- Monday: 10km easy run" in text
    assert "- Tuesday: intervals 6x800m" in text
    assert meta == {"type": "marathon", "race": "Berlin Marathon 2025"}


def test_plan_to_text_workouts_as_mapping():
    plan = {"type": "paracanoe", "workouts": {"Sat": "long paddle 12km"}}
    text, meta = ingest._plan_to_text(plan)
    assert "- Sat: long paddle 12km" in text
    assert meta["type"] == "paracanoe"


def test_plan_to_text_unknown_shape_keeps_json():
    text, meta = ingest._plan_to_text({"foo": "bar"})
    assert "foo" in text and "bar" in text
    assert meta == {}


def test_load_json_single_and_list(tmp_path: Path):
    single = tmp_path / "plan.json"
    single.write_text(
        '{"type": "short distance", "workouts": [{"day": "Wed", "workout": "sprints"}]}',
        encoding="utf-8",
    )
    docs = ingest._load_json(single)
    assert len(docs) == 1
    assert "sprints" in docs[0].page_content
    assert docs[0].metadata["type"] == "short distance"

    multi = tmp_path / "plans.json"
    multi.write_text(
        '[{"type": "marathon", "workouts": []}, {"type": "5k", "workouts": []}]',
        encoding="utf-8",
    )
    docs = ingest._load_json(multi)
    assert len(docs) == 2
    assert {d.metadata["type"] for d in docs} == {"marathon", "5k"}


def test_json_is_supported_suffix():
    assert ".json" in ingest.SUPPORTED_SUFFIXES

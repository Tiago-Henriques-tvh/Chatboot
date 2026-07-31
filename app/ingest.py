"""Ingestion CLI: recursively load documents and embed them into Qdrant.

Run inside the container:

    python -m app.ingest

Supported formats: ``.pdf``, ``.txt``, ``.md``, ``.json``. Documents are
chunked, tagged with their source path and a content hash, then upserted into
the Qdrant ``training_knowledge`` collection. Re-running is idempotent for
unchanged files because chunk IDs are derived deterministically from their
content.

JSON files are treated as **training plans**. Each plan is flattened into
readable prose (so the embedding model can reason over it) and tagged with
``type``/``race`` metadata. A JSON file may contain a single plan object or a
list of plan objects.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from .config import settings
from .vectorstore import ensure_collection, get_vectorstore

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".json"}
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _plan_to_text(plan: dict) -> tuple[str, dict]:
    """Flatten one training-plan dict into readable text plus extracted metadata.

    Expected (flexible) shape::

        {
          "type": "marathon" | "short distance" | "paracanoe" | ...,
          "race": "Berlin Marathon 2025",        # optional
          "workouts": [
            {"day": "Monday", "workout": "10km easy run"},
            ...
          ]
        }

    Unknown keys are still rendered so no information is silently dropped.
    """
    lines: list[str] = []
    meta: dict[str, str] = {}

    ptype = plan.get("type") or plan.get("discipline") or plan.get("sport")
    if ptype:
        lines.append(f"Training plan type: {ptype}")
        meta["type"] = str(ptype)

    for key in ("race", "event", "goal", "name", "title"):
        if plan.get(key):
            lines.append(f"{key.capitalize()}: {plan[key]}")
            meta.setdefault("race", str(plan[key]))
            break

    workouts = plan.get("workouts")
    if isinstance(workouts, dict):
        nested_list = next(
            (v for v in workouts.values() if isinstance(v, list)), None
        )
        if nested_list is not None:
            workouts = nested_list
        else:
            workouts = [{"day": k, "workout": v} for k, v in workouts.items()]

    if isinstance(workouts, list):
        lines.append("")
        lines.append("Workouts:")
        for i, w in enumerate(workouts, start=1):
            if isinstance(w, dict):
                day = w.get("day") or w.get("date") or f"Session {i}"
                desc = (
                    w.get("workout")
                    or w.get("description")
                    or w.get("session")
                    or ""
                )
                extra = {
                    k: v
                    for k, v in w.items()
                    if k not in ("day", "date", "workout", "description", "session")
                }
                extra_str = (
                    " (" + ", ".join(f"{k}: {v}" for k, v in extra.items()) + ")"
                    if extra
                    else ""
                )
                lines.append(f"- {day}: {desc}{extra_str}")
            else:
                lines.append(f"- {w}")

    text = "\n".join(lines).strip()
    if not text:
        # Unrecognised shape: keep the raw JSON so nothing is lost.
        text = json.dumps(plan, ensure_ascii=False, indent=2)
    return text, meta


def _load_json(path: Path):
    """Load a JSON training-plan file into one or more LangChain documents."""
    from langchain_core.documents import Document

    raw = json.loads(path.read_text(encoding="utf-8"))
    plans = raw if isinstance(raw, list) else [raw]

    docs = []
    for plan in plans:
        if isinstance(plan, dict):
            text, meta = _plan_to_text(plan)
        else:
            text, meta = json.dumps(plan, ensure_ascii=False), {}
        docs.append(Document(page_content=text, metadata=dict(meta)))
    return docs


def _load_file(path: Path):
    """Load a single file into a list of LangChain documents."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()
    if suffix == ".json":
        return _load_json(path)
    # .txt and .md
    from langchain_community.document_loaders import TextLoader

    return TextLoader(str(path), encoding="utf-8").load()


def _discover(data_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


def _deterministic_id(text: str, source: str, index: int) -> str:
    digest = hashlib.sha256(f"{source}:{index}:{text}".encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_NAMESPACE, digest))


def ingest(data_path: str | None = None) -> int:
    """Ingest all supported files under ``data_path``.

    Returns the number of chunks written. Splitting and embedding happen in
    this function so it can be imported and tested directly.
    """
    data_dir = Path(data_path or settings.data_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = _discover(data_dir)
    if not files:
        print(f"No supported documents found in {data_dir}. Nothing to ingest.")
        return 0

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    ensure_collection()
    store = get_vectorstore()

    total = 0
    for path in files:
        print(f"Loading {path} ...")
        try:
            raw_docs = _load_file(path)
        except Exception as exc:  # pragma: no cover - loader/runtime dependent
            print(f"  ! Skipped ({exc})")
            continue

        chunks = splitter.split_documents(raw_docs)
        rel = str(path.relative_to(data_dir))
        ids = []
        for i, chunk in enumerate(chunks):
            chunk.metadata = {**(chunk.metadata or {}), "source": rel}
            ids.append(_deterministic_id(chunk.page_content, rel, i))

        if chunks:
            store.add_documents(chunks, ids=ids)
            total += len(chunks)
            print(f"  + {len(chunks)} chunks embedded")

    print(f"\nDone. {total} chunks written to '{settings.collection_name}'.")
    return total


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()

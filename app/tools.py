"""ReAct tools exposed to the LangGraph agent.

Three tools are registered:

1. ``search_training_database`` - vector similarity search over Qdrant.
2. ``search_web`` - live DuckDuckGo web search.
3. ``generate_training_plan`` - builds a progressive athletic schedule using
   retrieved historical training context.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .config import settings
from .vectorstore import get_vectorstore


def _format_docs(docs) -> str:
    if not docs:
        return "No matching documents were found in the local knowledge base."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


@tool
def search_training_database(query: str) -> str:
    """Search the user's personal notes, workout logs and documents.

    Use this whenever the question refers to the user's own data such as
    training history, mileage, personal notes, or previously ingested
    documents. Input should be a focused natural-language query.
    """
    try:
        store = get_vectorstore()
        docs = store.similarity_search(query, k=settings.top_k)
        return _format_docs(docs)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return f"Error querying the training database: {exc}"


@tool
def search_web(query: str) -> str:
    """Search the public web via DuckDuckGo for current information.

    Use this when the answer depends on general or up-to-date information that
    is unlikely to be in the user's personal knowledge base.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=settings.top_k))
        if not results:
            return "No web results were found."
        return "\n\n".join(
            f"[{i}] {r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
            for i, r in enumerate(results, start=1)
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return f"Error performing web search: {exc}"


@tool
def generate_training_plan(goal: str) -> str:
    """Generate a progressive athletic training plan for a stated goal.

    Retrieves the user's historical training volume, workout logs and recovery
    notes from the local knowledge base and returns them as structured context
    the agent should use to compile a week-by-week progressive schedule.

    ``goal`` should describe the target event, e.g. "12-week half marathon".
    """
    try:
        store = get_vectorstore()
        queries = [
            f"{goal} training history and weekly mileage",
            "recent workout logs long runs intervals",
            "recovery rest days injuries fatigue notes",
        ]
        seen: set[str] = set()
        collected = []
        for q in queries:
            for doc in store.similarity_search(q, k=settings.top_k):
                key = doc.page_content.strip()[:200]
                if key in seen:
                    continue
                seen.add(key)
                collected.append(doc)

        context = _format_docs(collected)
        return (
            f"GOAL: {goal}\n\n"
            "HISTORICAL TRAINING CONTEXT (from the user's logs):\n"
            f"{context}\n\n"
            "INSTRUCTIONS: Using the context above, produce a progressive, "
            "week-by-week schedule. Ramp weekly volume gradually (roughly the "
            "10% rule), include recovery/de-load weeks, and tailor intensity to "
            "the athlete's demonstrated fitness. If the context is sparse, state "
            "the assumptions you are making."
        )
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return f"Error generating training plan: {exc}"


ALL_TOOLS = [search_training_database, search_web, generate_training_plan]

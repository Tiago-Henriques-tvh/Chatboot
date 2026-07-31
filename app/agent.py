"""LangGraph ReAct agent wired to a local Ollama model."""

from __future__ import annotations

from functools import lru_cache

from .config import settings
from .tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are a privacy-focused, locally hosted athletic and knowledge "
    "assistant. You have access to three tools:\n"
    "- search_training_database: the user's personal notes, workout logs and "
    "documents.\n"
    "- search_web: live web information via DuckDuckGo.\n"
    "- generate_training_plan: retrieves historical training data to help you "
    "build progressive race schedules.\n\n"
    "Decide autonomously which tool(s) to use. Prefer the local training "
    "database for anything about the user's own data. Use the web only for "
    "general or current information. Answer directly when no tool is needed. "
    "Always cite which source (local notes vs web) your answer relied on."
)


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached ChatOllama instance."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.model_name,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        # langchain-ollama accepts a client-level timeout via `client_kwargs`.
        client_kwargs={"timeout": settings.request_timeout},
    )


@lru_cache(maxsize=1)
def get_agent():
    """Build and cache the LangGraph ReAct agent."""
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        model=get_llm(),
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )


def build_messages(prompt: str) -> dict:
    """Wrap a user prompt into the agent's expected input state."""
    return {"messages": [{"role": "user", "content": prompt}]}


async def run_agent(prompt: str) -> str:
    """Run the agent to completion and return the final assistant message."""
    agent = get_agent()
    result = await agent.ainvoke(build_messages(prompt))
    messages = result.get("messages", [])
    if not messages:
        return ""
    return getattr(messages[-1], "content", "") or ""


async def stream_agent_tokens(prompt: str):
    """Yield assistant token strings as the agent produces them.

    Uses LangGraph's ``messages`` stream mode which emits ``(chunk, metadata)``
    tuples where ``chunk`` is an LLM message chunk. Only assistant-generated
    text (not tool output) is forwarded to the client.
    """
    agent = get_agent()
    async for chunk, metadata in agent.astream(
        build_messages(prompt), stream_mode="messages"
    ):
        node = metadata.get("langgraph_node") if metadata else None
        content = getattr(chunk, "content", None)
        if content and node != "tools":
            yield content

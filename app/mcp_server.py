"""FastMCP server exposing the agent's tools over the Model Context Protocol.

The same three capabilities available to the ReAct agent are registered as MCP
tools so that native MCP clients (Claude Desktop, Cursor, etc.) can call them
directly. The server is mounted onto the FastAPI app under ``/mcp`` in
``app.main``.
"""

from __future__ import annotations

from fastmcp import FastMCP

from .agent import run_agent
from .tools import (
    generate_training_plan,
    search_training_database,
    search_web,
)

mcp = FastMCP("agentic-rag-mcp")


@mcp.tool
def training_database_search(query: str) -> str:
    """Vector similarity search over the user's local training knowledge base."""
    return search_training_database.invoke({"query": query})


@mcp.tool
def web_search(query: str) -> str:
    """Live DuckDuckGo web search for current information."""
    return search_web.invoke({"query": query})


@mcp.tool
def training_plan(goal: str) -> str:
    """Assemble progressive training-plan context for a stated athletic goal."""
    return generate_training_plan.invoke({"goal": goal})


@mcp.tool
async def agent_chat(prompt: str) -> str:
    """Run the full autonomous ReAct agent and return its final answer."""
    return await run_agent(prompt)

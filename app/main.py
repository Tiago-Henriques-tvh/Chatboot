"""FastAPI application: REST endpoints, SSE streaming, MCP mount and web UI."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import run_agent, stream_agent_tokens
from .config import settings
from .mcp_server import mcp
from .tools import (
    generate_training_plan,
    search_training_database,
    search_web,
)

STATIC_DIR = Path(__file__).parent / "static"

# FastMCP exposes an ASGI app we can mount directly onto FastAPI.
mcp_app = mcp.http_app(path="/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Delegate to FastMCP's lifespan so its session manager starts/stops.
    async with mcp_app.lifespan(app):
        yield


app = FastAPI(
    title="Local Agentic RAG Backend & FastMCP Server",
    version="0.1.0",
    description=(
        "Privacy-focused, 100% locally hosted agentic RAG gateway. Exposes an "
        "SSE streaming REST API plus a native Model Context Protocol interface."
    ),
    lifespan=lifespan,
)

app.mount("/mcp-server", mcp_app)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt for the agent.")


class ChatResponse(BaseModel):
    response: str


class ToolCallRequest(BaseModel):
    tool: str = Field(..., description="Tool name to invoke.")
    input: dict = Field(default_factory=dict, description="Tool arguments.")


_REST_TOOLS = {
    "search_training_database": (search_training_database, "query"),
    "search_web": (search_web, "query"),
    "generate_training_plan": (generate_training_plan, "goal"),
}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> str:
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Agentic RAG Backend</h1><p>See <a href='/docs'>/docs</a>.</p>"


@app.get("/health")
async def health() -> dict:
    """Readiness probe: reports Qdrant and Ollama reachability."""
    status = {"status": "ok", "qdrant": "unknown", "ollama": "unknown"}

    try:
        from .vectorstore import get_client

        get_client().get_collections()
        status["qdrant"] = "up"
    except Exception as exc:  # pragma: no cover - infra dependent
        status["qdrant"] = f"down: {exc}"
        status["status"] = "degraded"

    try:
        import httpx

        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{settings.ollama_base_url}/api/tags")
            status["ollama"] = "up" if resp.status_code == 200 else "down"
    except Exception as exc:  # pragma: no cover - infra dependent
        status["ollama"] = f"down: {exc}"
        status["status"] = "degraded"

    return status


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Synchronous chat: runs the agent to completion and returns the answer."""
    answer = await run_agent(request.prompt)
    return ChatResponse(response=answer)


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events stream of ``data: {"token": "..."}`` chunks."""

    async def event_generator():
        try:
            async for token in stream_agent_tokens(request.prompt):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as exc:  # pragma: no cover - runtime dependent
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/mcp/manifest", include_in_schema=True)
async def mcp_manifest() -> dict:
    """Lightweight REST manifest describing the available MCP tools."""
    tools = await mcp.get_tools()
    return {
        "name": mcp.name,
        "protocol": "mcp",
        "endpoint": "/mcp-server",
        "tools": [
            {"name": name, "description": (tool.description or "").strip()}
            for name, tool in tools.items()
        ],
    }


@app.post("/mcp/call")
async def mcp_call(request: ToolCallRequest) -> dict:
    """REST convenience endpoint to invoke a single tool by name."""
    entry = _REST_TOOLS.get(request.tool)
    if entry is None:
        return {
            "error": f"Unknown tool '{request.tool}'.",
            "available": sorted(_REST_TOOLS),
        }
    tool, arg_name = entry
    value = request.input.get(arg_name) or request.input.get("query") or ""
    result = tool.invoke({arg_name: value})
    return {"tool": request.tool, "result": result}

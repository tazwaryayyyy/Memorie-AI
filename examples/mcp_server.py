"""
Memoire MCP Server
==================
Exposes Memoire as a Model Context Protocol (MCP) tool server so that any
MCP-compatible AI agent (Claude Desktop, Cursor, Zed, etc.) can call:

  - memoire_remember(content)   → store a memory
  - memoire_recall(query, k)    → semantic search
  - memoire_forget(id)          → delete a memory
  - memoire_status()            → database stats

Protocol: stdio (JSON-RPC 2.0 over stdin/stdout as per the MCP spec)

Install & run:
    pip install mcp          # Anthropic's MCP Python SDK
    cargo build --release
    python examples/mcp_server.py

Add to Claude Desktop (~/Library/Application Support/Claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "memoire": {
          "command": "python",
          "args": ["/path/to/memoire/examples/mcp_server.py"],
          "env": { "MEMOIRE_DB": "/path/to/your/agent.db" }
        }
      }
    }
"""

import os
import sys
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "bindings" / "python"))

try:
    import mcp.server.stdio
    from mcp.server import Server
    from mcp.types import Tool, TextContent
except ImportError:
    sys.exit(
        "mcp package not found.\n"
        "Install it:  pip install mcp\n"
        "Then re-run: python examples/mcp_server.py\n"
    )

from memoire import Memoire, MemoireError, MemoryPolicy  # type: ignore[import]  # noqa: E402  # pylint: disable=import-error,wrong-import-position

# ─── Config ──────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("MEMOIRE_DB", str(
    Path.home() / ".memoire" / "agent.db"))
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ─── MCP server ──────────────────────────────────────────────────────────────

server = Server("memoire")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="save_lesson",
            description=(
                "Store a lesson, decision, or corrective fact as a persistent memory. "
                "Use after any notable failure, fix, or architectural decision so future "
                "sessions can recall and trust it. Content is quality-scored at ingestion "
                "— high-actionability lessons reach FOLLOW threshold faster."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The lesson to store. Be specific: include what "
                                       "went wrong, what the fix was, and why it matters.",
                    }
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="get_lessons",
            description=(
                "Retrieve relevant past lessons for a query and apply the trust policy. "
                "Returns only memories that should influence agent behaviour — FOLLOW "
                "(trust \u2265 0.75, inject as hard context) or HINT (trust \u2265 0.45, inject as "
                "soft suggestion). Low-trust memories are silently suppressed. "
                "Use this at the start of a task to prime the agent's context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of the current task.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Candidates to retrieve before policy filtering (default: 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memoire_remember",
            description=(
                "Store text as a persistent, searchable memory. "
                "Use this to record decisions, bug fixes, architectural choices, "
                "session notes, or any context you want to recall in future sessions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text to store. Can be any length — "
                                       "it will be chunked automatically.",
                    }
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="memoire_recall",
            description=(
                "Search stored memories using semantic similarity. "
                "Returns the most relevant past memories for the given query. "
                "Use this at the start of a session to recall relevant context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="memoire_forget",
            description="Delete a specific memory by its id (from memoire_recall results).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Memory id to delete."}
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="memoire_status",
            description="Return database statistics: total memories, database path and size.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        with Memoire(DB_PATH) as m:
            # ── High-level trust-aware tools ─────────────────────────────────
            if name == "save_lesson":
                content = arguments.get("content", "").strip()
                if not content:
                    return [TextContent(type="text", text="Error: content is empty.")]
                n = m.remember(content)
                total = m.count()
                return [TextContent(
                    type="text",
                    text=(
                        f"Lesson stored ({n} chunk(s)). Total memories: {total}.\n"
                        "Trust starts low — it will grow each time this lesson helps "
                        "an agent succeed."
                    ),
                )]

            elif name == "get_lessons":
                query = arguments.get("query", "").strip()
                if not query:
                    return [TextContent(type="text", text="Error: query is empty.")]
                top_k = int(arguments.get("top_k", 5))
                memories = m.recall(query, top_k=top_k)
                if not memories:
                    return [TextContent(type="text", text="No relevant memories found.")]
                policy = MemoryPolicy()
                decisions = policy.evaluate(memories)
                context = policy.inject_context(decisions)
                follow = [d for d in decisions if d.action == "follow"]
                hint = [d for d in decisions if d.action == "hint"]
                ignore = [d for d in decisions if d.action == "ignore"]
                summary = (
                    f"Policy applied to {len(memories)} candidate(s): "
                    f"{len(follow)} FOLLOW, {len(hint)} HINT, {len(ignore)} IGNORE.\n"
                )
                if not follow and not hint:
                    return [TextContent(
                        type="text",
                        text=summary + "No memories meet the trust threshold for injection.",
                    )]
                return [TextContent(type="text", text=summary + context)]

            # ── Low-level passthrough tools ───────────────────────────────────
            elif name == "memoire_remember":
                content = arguments.get("content", "").strip()
                if not content:
                    return [TextContent(type="text", text="Error: content is empty.")]
                n = m.remember(content)
                total = m.count()
                return [TextContent(
                    type="text",
                    text=f"Stored {n} chunk(s). Total memories: {total}.",
                )]

            elif name == "memoire_recall":
                query = arguments.get("query", "").strip()
                if not query:
                    return [TextContent(type="text", text="Error: query is empty.")]
                top_k = int(arguments.get("top_k", 5))
                results = m.recall(query, top_k=top_k)
                if not results:
                    return [TextContent(type="text", text="No relevant memories found.")]
                lines = [f"Found {len(results)} relevant memory/memories:\n"]
                for r in results:
                    lines.append(
                        f"[id={r.id} score={r.score:.3f}] {r.content}")
                return [TextContent(type="text", text="\n".join(lines))]

            elif name == "memoire_forget":
                mid = int(arguments["id"])
                deleted = m.forget(mid)
                msg = f"Deleted memory id={mid}." if deleted else f"Memory id={mid} not found."
                return [TextContent(type="text", text=msg)]

            elif name == "memoire_status":
                count = m.count()
                db = Path(DB_PATH)
                size = f"{db.stat().st_size / 1024:.1f} KB" if db.exists() else "new"
                return [TextContent(
                    type="text",
                    text=f"DB: {DB_PATH}\nSize: {size}\nStored chunks: {count}",
                )]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name!r}")]

    except MemoireError as e:
        return [TextContent(type="text", text=f"Memoire error: {e}")]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

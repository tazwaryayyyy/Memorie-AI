"""
Memoire × OpenAI Agent
=======================
Shows how to wire Memoire into an OpenAI-compatible coding agent using
function calling. Works with OpenAI, Groq, Mistral, or any provider
that follows the OpenAI tools spec.

Install:
    pip install openai
    cargo build --release

Set:
    export OPENAI_API_KEY=sk-...    (or GROQ_API_KEY= / MISTRAL_API_KEY= etc.)
    export OPENAI_BASE_URL=...      (optional — for Groq/Mistral/local endpoints)

Run:
    python examples/openai_agent.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "bindings" / "python"))

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Install openai:  pip install openai")

from memoire import Memoire  # noqa: E402

# ─── Config ──────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("MEMOIRE_DB", "./openai_agent.db")
MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI()   # reads OPENAI_API_KEY from env

# ─── Tool definitions (sent to the model) ────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memoire_remember",
            "description": (
                "Persist important information to long-term memory. "
                "Call this after every significant discovery, fix, or decision "
                "so that the information is available in future sessions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store.",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memoire_recall",
            "description": (
                "Search long-term memory for information relevant to the current task. "
                "Call this at the start of a new task before writing any code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural language description of what you are looking for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of memories to retrieve (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a senior software engineering assistant with access to
long-term persistent memory via two tools:

- memoire_recall:   search past memories before starting any task
- memoire_remember: save important findings, fixes, and decisions after each task

Your workflow:
1. When given a new task, ALWAYS call memoire_recall first to check what you know.
2. Complete the task using the retrieved context.
3. After completing a task, call memoire_remember to record what you learned.

This makes you progressively smarter across sessions."""


# ─── Tool execution ──────────────────────────────────────────────────────────

def execute_tool(name: str, args: dict, mem: Memoire) -> str:
    if name == "memoire_remember":
        n = mem.remember(args["content"])
        return f"Stored {n} chunk(s). Total memories: {mem.count()}."

    elif name == "memoire_recall":
        results = mem.recall(args["query"], top_k=args.get("top_k", 5))
        if not results:
            return "No relevant memories found."
        lines = [f"[{r.score:.3f}] {r.content}" for r in results]
        return "\n".join(lines)

    return f"Unknown tool: {name}"


# ─── Agent loop ──────────────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """Run one turn of the agent with persistent memory."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    with Memoire(DB_PATH) as mem:
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # If the model wants to call tools
            if msg.tool_calls:
                messages.append(msg)
                for call in msg.tool_calls:
                    args   = json.loads(call.function.arguments)
                    result = execute_tool(call.function.name, args, mem)
                    print(f"  [tool] {call.function.name}({args!r}) → {result[:80]}")
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": call.id,
                        "content":      result,
                    })
                # Continue the loop so the model can respond after tool results
                continue

            # No more tool calls — return the final text
            return msg.content or ""


# ─── Demo ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tasks = [
        "I just fixed a critical race condition in the payment service: two concurrent "
        "requests with the same idempotency key could both pass the uniqueness check "
        "before either committed. Fixed by wrapping the check in a serialisable transaction.",

        "Help me think about what we know about the payment service and what risks remain.",

        "I also discovered that the auth middleware was not validating the JWT issuer claim, "
        "meaning forged tokens from any issuer could authenticate. Patched it and added tests.",

        "What security issues have we fixed so far? Summarise what we know.",
    ]

    for task in tasks:
        print(f"\n{'='*70}")
        print(f"USER: {task[:100]}{'…' if len(task) > 100 else ''}")
        print(f"{'='*70}")
        answer = run_agent(task)
        print(f"AGENT: {answer}")

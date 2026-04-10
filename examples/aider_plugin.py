"""
Memoire × Aider Integration
============================
Gives Aider persistent semantic memory across sessions.

Aider calls this plugin at the start of every chat turn via --pre-message-hook
and at the end of every accepted diff via --post-message-hook.

Setup:
    1. Build Memoire:
          cargo build --release

    2. Install the Python binding (from repo root):
          pip install -e bindings/python/

    3. Add to your ~/.aider.conf.yml:
          pre-message-hook:  python examples/aider_plugin.py recall
          post-message-hook: python examples/aider_plugin.py remember

    OR call it manually inside an Aider session:
          /run python examples/aider_plugin.py recall "how did I fix auth?"
          /run python examples/aider_plugin.py remember "Fixed the JWT bug today"

Usage (standalone):
    python examples/aider_plugin.py remember "Fixed pagination off-by-one"
    python examples/aider_plugin.py recall   "pagination bugs"
    python examples/aider_plugin.py status
"""

import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "bindings" / "python"))

from memoire import Memoire, Memory  # noqa: E402

# ─── Config ──────────────────────────────────────────────────────────────────

DB_PATH   = os.environ.get("MEMOIRE_DB", Path.home() / ".memoire" / "aider.db")
TOP_K     = int(os.environ.get("MEMOIRE_TOP_K", "5"))
MIN_SCORE = float(os.environ.get("MEMOIRE_MIN_SCORE", "0.45"))

# ─── Helpers ─────────────────────────────────────────────────────────────────

def ensure_db_dir():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def format_recall(memories: list[Memory], query: str) -> str:
    """Format recall results as a block Aider can prepend to the system prompt."""
    if not memories:
        return ""
    lines = [
        f"# Memoire: relevant past context for '{query}'",
        "",
    ]
    for r in memories:
        lines.append(f"[{r.score:.2f}] {r.content}")
    lines.append("")
    return "\n".join(lines)


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_remember(content: str) -> None:
    """Store content in Memoire."""
    ensure_db_dir()
    if not content.strip():
        print("[memoire] Empty content — nothing stored.", file=sys.stderr)
        return
    with Memoire(str(DB_PATH)) as m:
        n = m.remember(content)
        total = m.count()
    print(f"[memoire] Stored {n} chunk(s). Total memories: {total}.")


def cmd_recall(query: str) -> None:
    """
    Print relevant past memories to stdout.
    Aider can capture this output and inject it into the system prompt.
    """
    ensure_db_dir()
    with Memoire(str(DB_PATH)) as m:
        if m.count() == 0:
            return
        results = [r for r in m.recall(query, top_k=TOP_K) if r.score >= MIN_SCORE]

    if results:
        print(format_recall(results, query))


def cmd_status() -> None:
    """Show Memoire database status."""
    ensure_db_dir()
    with Memoire(str(DB_PATH)) as m:
        count = m.count()
    db = Path(DB_PATH)
    size = f"{db.stat().st_size / 1024:.1f} KB" if db.exists() else "not created"
    print(f"[memoire] DB: {DB_PATH}")
    print(f"[memoire] Size: {size}")
    print(f"[memoire] Stored chunks: {count}")


def cmd_forget_last() -> None:
    """Delete the most recently stored memory."""
    ensure_db_dir()
    with Memoire(str(DB_PATH)) as m:
        # Recall with a broad query to surface recent items
        results = m.recall(".", top_k=1)
        if not results:
            print("[memoire] No memories to forget.")
            return
        deleted = m.forget(results[0].id)
        print(f"[memoire] Deleted id={results[0].id}: {deleted}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    command = args[0].lower()
    rest    = " ".join(args[1:])

    if command == "remember":
        if not rest:
            # Read from stdin (useful for piping git log, etc.)
            rest = sys.stdin.read()
        cmd_remember(rest)

    elif command == "recall":
        if not rest:
            print("[memoire] recall requires a query string.", file=sys.stderr)
            sys.exit(1)
        cmd_recall(rest)

    elif command in ("status", "info"):
        cmd_status()

    elif command == "forget-last":
        cmd_forget_last()

    else:
        print(f"[memoire] Unknown command: {command!r}", file=sys.stderr)
        print("  Commands: remember, recall, status, forget-last", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

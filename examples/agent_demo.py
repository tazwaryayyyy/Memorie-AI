"""
Memoire Python integration demo — simulates an AI coding agent
using Memoire for persistent semantic memory across sessions.

Usage:
    python examples/agent_demo.py

Requirements:
    Compile the library first: cargo build --release
"""

import ctypes
import json
import os
import sys
from pathlib import Path


# ─── Load library ─────────────────────────────────────────────────────────────

def _load() -> ctypes.CDLL:
    root = Path(__file__).parent.parent
    for name in ("libmemoire.so", "libmemoire.dylib", "memoire.dll"):
        p = root / "target" / "release" / name
        if p.exists():
            return ctypes.CDLL(str(p))
    sys.exit(
        "ERROR: compiled library not found.\n"
        "Run:  cargo build --release"
    )


lib = _load()

# ─── Function signatures ──────────────────────────────────────────────────────

lib.memoire_new.argtypes      = [ctypes.c_char_p]
lib.memoire_new.restype       = ctypes.c_void_p

lib.memoire_free.argtypes     = [ctypes.c_void_p]
lib.memoire_free.restype      = None

lib.memoire_remember.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.memoire_remember.restype  = ctypes.c_int

lib.memoire_recall.argtypes   = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.memoire_recall.restype    = ctypes.c_char_p

lib.memoire_forget.argtypes   = [ctypes.c_void_p, ctypes.c_int64]
lib.memoire_forget.restype    = ctypes.c_int

lib.memoire_count.argtypes    = [ctypes.c_void_p]
lib.memoire_count.restype     = ctypes.c_int64

lib.memoire_clear.argtypes    = [ctypes.c_void_p]
lib.memoire_clear.restype     = ctypes.c_int

lib.memoire_free_string.argtypes = [ctypes.c_char_p]
lib.memoire_free_string.restype  = None


# ─── Python wrapper ───────────────────────────────────────────────────────────

class Memoire:
    """
    Pythonic wrapper around the Memoire C FFI.

        with Memoire("agent.db") as m:
            m.remember("Fixed the auth bug today")
            for r in m.recall("auth issues", top_k=3):
                print(f"[{r['score']:.3f}] {r['content']}")
    """

    def __init__(self, db_path: str = "agent_memory.db"):
        self._h = lib.memoire_new(db_path.encode())
        if not self._h:
            raise RuntimeError(f"Failed to open Memoire at {db_path!r}")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._h:
            lib.memoire_free(self._h)
            self._h = None

    def remember(self, content: str) -> int:
        n = lib.memoire_remember(self._h, content.encode())
        if n < 0:
            raise RuntimeError("memoire_remember() failed")
        return n

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        raw = lib.memoire_recall(self._h, query.encode(), top_k)
        if not raw:
            raise RuntimeError("memoire_recall() failed")
        try:
            return json.loads(raw.decode())
        finally:
            lib.memoire_free_string(raw)

    def forget(self, memory_id: int) -> bool:
        r = lib.memoire_forget(self._h, memory_id)
        if r < 0:
            raise RuntimeError("memoire_forget() failed")
        return r == 1

    def count(self) -> int:
        n = lib.memoire_count(self._h)
        if n < 0:
            raise RuntimeError("memoire_count() failed")
        return n

    def clear(self):
        if lib.memoire_clear(self._h) < 0:
            raise RuntimeError("memoire_clear() failed")


# ─── Demo ─────────────────────────────────────────────────────────────────────

DB = "demo_agent.db"

SESSIONS = [
    [
        "Fixed critical bug where JWT tokens were not validated against the issuer "
        "claim, allowing forged tokens from any issuer to authenticate.",

        "Replaced bcrypt with Argon2id for password hashing. Added migration flag "
        "to re-hash passwords on next login.",

        "/api/reset-password endpoint had no rate limiting. Added 5 req/hour per IP "
        "using Redis sorted sets.",
    ],
    [
        "Found N+1 query in the user dashboard — each widget made a separate DB call. "
        "Consolidated into a single JOIN. Load time dropped from 4.2s to 380ms.",

        "Added SQLite WAL mode and connection pooling (max 20). Previously each "
        "request opened a fresh connection.",

        "Profiled the image resize pipeline. Switched from Pillow to libvips. "
        "Throughput improved 6x.",
    ],
]

QUERIES = [
    "what security vulnerabilities did I fix?",
    "how did I make the database faster?",
    "rate limiting and API protection",
    "image processing and file handling",
    "authentication and token validation",
]


def banner(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    # Start fresh
    if os.path.exists(DB):
        os.remove(DB)

    with Memoire(DB) as m:
        # Store sessions
        for i, notes in enumerate(SESSIONS, 1):
            banner(f"SESSION {i} — storing {len(notes)} observations")
            for note in notes:
                chunks = m.remember(note)
                print(f"  ✓ ({chunks} chunk) {note[:70]}…")
            print(f"\n  Total in store: {m.count()}")

        # Query
        banner("AGENT RECALL — querying memory before starting work")
        for q in QUERIES:
            print(f"\n  ▶ \"{q}\"")
            results = m.recall(q, top_k=2)
            if not results:
                print("    (no relevant memories)")
            for r in results:
                print(f"    [{r['score']:.4f}] {r['content'][:90]}…")

        # Demo forget
        banner("FORGET DEMO")
        all_results = m.recall("anything", top_k=1)
        if all_results:
            target_id = all_results[0]["id"]
            deleted = m.forget(target_id)
            print(f"  Deleted memory id={target_id}: {deleted}")
            print(f"  Remaining count: {m.count()}")

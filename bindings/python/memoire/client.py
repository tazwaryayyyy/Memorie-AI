"""
Core Python client for Memoire. Wraps the C FFI via ctypes.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


class MemoireError(Exception):
    """Raised when the Memoire library returns an error."""


@dataclass
class Memory:
    """A stored memory chunk with its similarity score."""
    id: int
    content: str
    score: float
    created_at: int

    def __repr__(self) -> str:
        preview = self.content[:60] + "…" if len(self.content) > 60 else self.content
        return f"Memory(id={self.id}, score={self.score:.4f}, content={preview!r})"


# ─── Library loader ───────────────────────────────────────────────────────────

def _find_lib() -> str:
    """Search for the compiled shared library."""
    # Allow override via environment variable
    env = os.environ.get("MEMOIRE_LIB")
    if env:
        return env

    names = {
        "linux":  "libmemoire.so",
        "darwin": "libmemoire.dylib",
        "win32":  "memoire.dll",
    }
    platform = sys.platform
    lib_name = names.get(platform)
    if lib_name is None:
        raise MemoireError(f"Unsupported platform: {platform!r}")

    # Walk up from this file to find the repo root and the Cargo output
    candidates = [
        Path(__file__).parent.parent.parent.parent / "target" / "release" / lib_name,
        Path(lib_name),  # system path / LD_LIBRARY_PATH
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise MemoireError(
        f"Could not find {lib_name}.\n"
        "Build it first:  cargo build --release\n"
        "Or set MEMOIRE_LIB=/path/to/libmemoire.so"
    )


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_find_lib())

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

    return lib


_lib: Optional[ctypes.CDLL] = None

def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_lib()
    return _lib


# ─── Main class ───────────────────────────────────────────────────────────────

class Memoire:
    """
    Local-first semantic memory engine for AI coding agents.

    Thread safety: each Memoire instance is NOT thread-safe. Create one instance
    per thread, or protect concurrent access with a threading.Lock.

    Context manager:
        with Memoire("agent.db") as m:
            m.remember("...")
            results = m.recall("...", top_k=5)
    """

    def __init__(self, db_path: str = "./memoire.db") -> None:
        """
        Open or create a Memoire database at `db_path`.

        Args:
            db_path: Path to the SQLite database file.
                     Pass ':memory:' for a non-persistent in-memory store.

        Raises:
            MemoireError: if the library fails to initialise.
        """
        lib = _get_lib()
        self._lib = lib
        self._handle = lib.memoire_new(db_path.encode("utf-8"))
        if not self._handle:
            raise MemoireError(
                f"Failed to initialise Memoire at {db_path!r}.\n"
                "Check that the database path is writable and the library is built."
            )

    # ─── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "Memoire":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        closed = self._handle is None
        return f"Memoire({'closed' if closed else f'count={self.count()}'})"

    # ─── Core API ─────────────────────────────────────────────────────────────

    def remember(self, content: str) -> int:
        """
        Chunk, embed, and store `content` as searchable memory.

        Args:
            content: Any text — session notes, code review feedback,
                     bug descriptions, architectural decisions, etc.

        Returns:
            Number of chunks stored (≥ 1).

        Raises:
            MemoireError: on library failure.
            TypeError: if content is not a string.
        """
        self._check_open()
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        n = self._lib.memoire_remember(self._handle, content.encode("utf-8"))
        if n < 0:
            raise MemoireError("remember() failed internally")
        return n

    def recall(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Return the `top_k` memories most semantically similar to `query`.

        Args:
            query: Natural language question or keyword phrase.
            top_k: Maximum number of results to return (default 5).

        Returns:
            List of Memory objects, sorted by score descending (0–1).

        Raises:
            MemoireError: on library failure.
        """
        self._check_open()
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        raw = self._lib.memoire_recall(
            self._handle, query.encode("utf-8"), ctypes.c_int(top_k)
        )
        if not raw:
            raise MemoireError("recall() failed internally")

        try:
            data = json.loads(raw.decode("utf-8"))
        finally:
            self._lib.memoire_free_string(raw)

        return [
            Memory(
                id=item["id"],
                content=item["content"],
                score=item["score"],
                created_at=item["created_at"],
            )
            for item in data
        ]

    def forget(self, memory_id: int) -> bool:
        """
        Delete a memory by its id (from Memory.id).

        Returns:
            True if the memory existed and was deleted, False if not found.
        """
        self._check_open()
        r = self._lib.memoire_forget(self._handle, ctypes.c_int64(memory_id))
        if r < 0:
            raise MemoireError(f"forget({memory_id}) failed internally")
        return r == 1

    def count(self) -> int:
        """Total number of stored memory chunks."""
        self._check_open()
        n = self._lib.memoire_count(self._handle)
        if n < 0:
            raise MemoireError("count() failed internally")
        return int(n)

    def clear(self) -> None:
        """
        Erase ALL memories. This cannot be undone.

        Consider exporting first if you want a backup.
        """
        self._check_open()
        if self._lib.memoire_clear(self._handle) < 0:
            raise MemoireError("clear() failed internally")

    def close(self) -> None:
        """Release the native handle. Called automatically by the context manager."""
        if self._handle:
            self._lib.memoire_free(self._handle)
            self._handle = None

    # ─── Convenience helpers ──────────────────────────────────────────────────

    def remember_lines(self, text: str) -> int:
        """
        Store each non-empty line of `text` as a separate memory.
        Useful for importing bullet-point session notes.

        Returns the total number of chunks stored.
        """
        total = 0
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                total += self.remember(line)
        return total

    def recall_one(self, query: str) -> Optional[Memory]:
        """Return the single most similar memory, or None if the store is empty."""
        results = self.recall(query, top_k=1)
        return results[0] if results else None

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _check_open(self) -> None:
        if self._handle is None:
            raise MemoireError("This Memoire instance has been closed.")

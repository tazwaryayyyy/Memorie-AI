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
    """A stored memory chunk with its similarity score, trust level, and uncertainty."""
    id: int
    content: str
    score: float
    trust: float
    uncertainty: float
    state: str
    created_at: int

    def __repr__(self) -> str:
        preview = self.content[:60] + \
            "…" if len(self.content) > 60 else self.content
        return (
            f"Memory(id={self.id}, score={self.score:.3f}, "
            f"trust={self.trust:.3f}, uncertainty={self.uncertainty:.3f}, "
            f"state={self.state!r}, content={preview!r})"
        )


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
    repo_root = Path(__file__).parent.parent.parent.parent
    candidates = [
        repo_root / "target" / "release" / lib_name,
        repo_root / "target" / "debug" / lib_name,
        Path(__file__).parent / lib_name,  # installed alongside the package
        Path(lib_name),                    # LD_LIBRARY_PATH / PATH / cwd
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    raise MemoireError(
        f"Could not find {lib_name}.\n"
        "Build it first:  cargo build --release\n"
        f"Or set MEMOIRE_LIB=/full/path/to/{lib_name}"
    )


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_find_lib())

    lib.memoire_new.argtypes = [ctypes.c_char_p]
    lib.memoire_new.restype = ctypes.c_void_p

    lib.memoire_free.argtypes = [ctypes.c_void_p]
    lib.memoire_free.restype = None

    lib.memoire_remember.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.memoire_remember.restype = ctypes.c_int

    lib.memoire_recall.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.memoire_recall.restype = ctypes.c_char_p

    lib.memoire_forget.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lib.memoire_forget.restype = ctypes.c_int

    lib.memoire_count.argtypes = [ctypes.c_void_p]
    lib.memoire_count.restype = ctypes.c_int64

    lib.memoire_clear.argtypes = [ctypes.c_void_p]
    lib.memoire_clear.restype = ctypes.c_int

    lib.memoire_free_string.argtypes = [ctypes.c_char_p]
    lib.memoire_free_string.restype = None

    lib.memoire_reinforce_if_used.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int]
    lib.memoire_reinforce_if_used.restype = ctypes.c_int

    lib.memoire_penalize_if_used.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
        ctypes.c_float,
    ]
    lib.memoire_penalize_if_used.restype = ctypes.c_char_p

    return lib


_lib_cache: list = []


def _get_lib() -> ctypes.CDLL:
    if not _lib_cache:
        _lib_cache.append(_load_lib())
    return _lib_cache[0]


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
            raise TypeError(
                f"content must be str, got {type(content).__name__}")
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
                trust=item.get("trust", 0.0),                uncertainty=item.get("uncertainty", 0.5),                state=item.get("state", "active"),
                created_at=item["created_at"],
            )
            for item in data
        ]

    def reinforce_if_used(
        self,
        memory_id: int,
        agent_output: str,
        task_succeeded: bool,
    ) -> bool:
        """
        Conditionally reinforce a memory based on actual use.

        Fires only when `task_succeeded` is True AND the token overlap
        between the memory content and `agent_output` exceeds 15%.

        Returns:
            True if reinforcement was applied.
        """
        self._check_open()
        r = self._lib.memoire_reinforce_if_used(
            self._handle,
            ctypes.c_int64(memory_id),
            agent_output.encode("utf-8"),
            ctypes.c_int(1 if task_succeeded else 0),
        )
        if r < 0:
            raise MemoireError(
                f"reinforce_if_used({memory_id}) failed internally")
        return r == 1

    def penalize_if_used(
        self,
        memory_ids: List[int],
        failure_severity: float = 1.0,
    ) -> List[dict]:
        """
        Penalize memories that contributed to a failed task outcome.

        `failure_severity` ∈ [0.0, 1.0] scales the penalty:

        * 1.0 — full penalty (bad guidance led directly to failure)
        * 0.5 — moderate (wrong direction but partially useful)
        * 0.0 — no-op

        Only pass memory ids that **actually influenced the decision** (i.e.
        those with action FOLLOW or HINT from ``MemoryPolicy``). Memories that
        were retrieved but ignored should not be penalized.

        Args:
            memory_ids:       List of Memory.id values to penalize.
            failure_severity: Scaling factor ∈ [0.0, 1.0]. Default 1.0.

        Returns:
            List of dicts:
            ``[{"id": int, "trust_before": float, "trust_after": float,
               "uncertainty_after": float}, ...]``
        """
        self._check_open()
        if not memory_ids:
            return []
        arr_type = ctypes.c_int64 * len(memory_ids)
        arr = arr_type(*memory_ids)
        raw = self._lib.memoire_penalize_if_used(
            self._handle,
            arr,
            ctypes.c_int(len(memory_ids)),
            ctypes.c_float(failure_severity),
        )
        if not raw:
            raise MemoireError("penalize_if_used failed internally")
        try:
            return json.loads(raw.decode("utf-8"))
        finally:
            self._lib.memoire_free_string(raw)

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


# ─── Agent Decision Policy ────────────────────────────────────────────────────

@dataclass
class PolicyDecision:
    """The policy verdict for a single recalled memory."""
    memory: Memory
    action: str   # "follow" | "hint" | "ignore"
    reason: str
    # ready-to-inject string, or None when ignored
    prompt_injection: Optional[str]


class MemoryPolicy:
    """
    Thin agent-side policy layer that turns trust scores into actionable decisions.

    Args:
        strict: When True, raises the bar for FOLLOW and HINT actions.
                Use for security-critical or high-consequence tasks.

    Thresholds:
        strict=False (default): FOLLOW >= 0.70, HINT >= 0.40
        strict=True:            FOLLOW >= 0.80, HINT >= 0.50

    Usage::

        policy = MemoryPolicy()                  # default
        policy = MemoryPolicy(strict=True)       # security tasks

        decisions = policy.evaluate(memories)
        for d in decisions:
            if d.action == "follow":
                system_prompt += d.prompt_injection
            elif d.action == "hint":
                system_prompt += d.prompt_injection
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        if strict:
            self.FOLLOW_THRESHOLD: float = 0.80
            self.HINT_THRESHOLD: float = 0.50
        else:
            self.FOLLOW_THRESHOLD = 0.70
            self.HINT_THRESHOLD = 0.40

    def evaluate(self, memories: List[Memory]) -> List[PolicyDecision]:
        """
        Evaluate a list of recalled memories and return policy decisions.

        Args:
            memories: List of Memory objects from Memoire.recall().

        Returns:
            List of PolicyDecision, one per memory, in the same order.
        """
        decisions = []
        for mem in memories:
            action, reason, injection = self._decide(mem)
            decisions.append(PolicyDecision(
                memory=mem,
                action=action,
                reason=reason,
                prompt_injection=injection,
            ))
        return decisions

    def _decide(self, mem: Memory) -> tuple:
        trust = mem.trust
        state = mem.state

        if trust >= self.FOLLOW_THRESHOLD:
            reason = f"trust={trust:.2f} {state} reinforced"
            injection = f"[MEMORY - HIGH TRUST]: {mem.content}"
            return "follow", reason, injection

        if trust >= self.HINT_THRESHOLD:
            reason = f"trust={trust:.2f} {state} low-confidence"
            injection = f"[MEMORY - HINT ONLY, verify before acting]: {mem.content}"
            return "hint", reason, injection

        # Below hint threshold — classify why
        if state == "shadow":
            reason = f"trust={trust:.2f} shadow unreinforced"
        elif trust < 0.20:
            reason = f"trust={trust:.2f} very low trust"
        else:
            reason = f"trust={trust:.2f} below hint threshold"

        return "ignore", reason, None

    def inject_context(self, decisions: List[PolicyDecision]) -> str:
        """
        Build a ready-to-prepend context block from all follow/hint decisions.

        Returns an empty string if no memories pass the threshold.
        """
        lines = []
        for d in decisions:
            if d.prompt_injection is not None:
                lines.append(d.prompt_injection)
        return "\n".join(lines)

"""
Core Python client for Memoire. Wraps the PyO3 native extension.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

# Import the compiled PyO3 native module
try:
    from .memoire import Memoire as _NativeMemoire, Memory as _NativeMemory, MemoireError
except ImportError:
    # If the native module is not compiled yet, import placeholder exceptions
    class MemoireError(Exception):
        """Raised when the Memoire library returns an error."""
    _NativeMemoire = None
    _NativeMemory = None

# Re-export the native Memory class
Memory = _NativeMemory


def _get_lib():
    """
    Backward-compatible native extension probe.

    Returns:
        The loaded native Memoire extension class when available.

    Raises:
        MemoireError: if the native extension is unavailable.
    """
    if _NativeMemoire is None:
        raise MemoireError(
            "Native compiled memoire extension is not installed.\n"
            "Please run `pip install .` or `maturin dev` in bindings/python."
        )
    return _NativeMemoire


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

    def __init__(self, db_path: str = "./memoire.db", namespace: str = "default") -> None:
        """
        Open or create a Memoire database at `db_path` scoped to `namespace`.

        Args:
            db_path: Path to the SQLite database file.
                     Pass ':memory:' for a non-persistent in-memory store.
            namespace: The isolation namespace for memories.

        Raises:
            MemoireError: if the library fails to initialise.
        """
        if _NativeMemoire is None:
            raise MemoireError(
                "Native compiled memoire extension is not installed.\n"
                "Please run `pip install .` or `maturin dev` in bindings/python."
            )
        try:
            self._impl = _NativeMemoire(db_path, namespace)
        except Exception as e:
            raise MemoireError(f"Failed to initialize Memoire: {e}") from e

    def __enter__(self) -> Memoire:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._impl is None:
            return "Memoire(closed)"
        return f"Memoire(count={self.count()})"

    def remember(self, content: str) -> int:
        """
        Chunk, embed, and store `content` as searchable memory.

        Args:
            content: Any text.

        Returns:
            Number of chunks stored (≥ 1).
        """
        self._check_open()
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        if not content.strip():
            return 0
        try:
            return self._impl.remember(content)
        except Exception as e:
            raise MemoireError(f"remember() failed: {e}") from e

    def recall(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Return the `top_k` memories most semantically similar to `query`.

        Args:
            query: Natural language question or keyword phrase.
            top_k: Maximum number of results to return (default 5).

        Returns:
            List of Memory objects, sorted by score descending (0–1).
        """
        self._check_open()
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        try:
            return self._impl.recall(query, top_k)
        except Exception as e:
            raise MemoireError(f"recall() failed: {e}") from e

    def reinforce_if_used(
        self,
        memory_id: int,
        agent_output: str,
        task_succeeded: bool,
    ) -> bool:
        """
        Conditionally reinforce a memory based on actual use.
        """
        self._check_open()
        try:
            return self._impl.reinforce_if_used(memory_id, agent_output, task_succeeded)
        except Exception as e:
            raise MemoireError(f"reinforce_if_used() failed: {e}") from e

    def penalize_if_used(
        self,
        memory_ids: List[int],
        failure_severity: float = 1.0,
    ) -> List[dict]:
        """
        Penalize memories that contributed to a failed task outcome.
        """
        self._check_open()
        if not memory_ids:
            return []
        try:
            res_str = self._impl.penalize_if_used(memory_ids, failure_severity)
            return json.loads(res_str)
        except Exception as e:
            raise MemoireError(f"penalize_if_used() failed: {e}") from e

    def forget(self, memory_id: int) -> bool:
        """
        Delete a memory by its id (from Memory.id).
        """
        self._check_open()
        try:
            return self._impl.forget(memory_id)
        except Exception as e:
            raise MemoireError(f"forget() failed: {e}") from e

    def resolve_contradictions(self, memory_id: int) -> bool:
        """
        Resolve contradictions for a specific memory id.
        """
        self._check_open()
        try:
            return self._impl.resolve_contradictions(memory_id)
        except Exception as e:
            raise MemoireError(f"resolve_contradictions() failed: {e}") from e

    def count(self) -> int:
        """Total number of stored memory chunks."""
        self._check_open()
        try:
            return self._impl.count()
        except Exception as e:
            raise MemoireError(f"count() failed: {e}") from e

    def clear(self) -> None:
        """Erase ALL memories. This cannot be undone."""
        self._check_open()
        try:
            self._impl.clear()
        except Exception as e:
            raise MemoireError(f"clear() failed: {e}") from e

    def export_namespace(self) -> dict:
        """Export the current namespace's non-archived memories as a JSON snapshot."""
        self._check_open()
        try:
            res_str = self._impl.export_namespace()
            return json.loads(res_str)
        except Exception as e:
            raise MemoireError(f"export_namespace() failed: {e}") from e

    def import_namespace(self, snapshot: dict) -> int:
        """Import memories from a namespace snapshot dict."""
        self._check_open()
        try:
            snapshot_str = json.dumps(snapshot)
            return self._impl.import_namespace(snapshot_str)
        except Exception as e:
            raise MemoireError(f"import_namespace() failed: {e}") from e

    def close(self) -> None:
        """Release the native instance."""
        self._impl = None

    def _check_open(self) -> None:
        if self._impl is None:
            raise MemoireError("This Memoire instance has been closed.")

    # ─── Convenience helpers ──────────────────────────────────────────────────

    def remember_lines(self, text: str) -> int:
        """
        Store each non-empty line of `text` as a separate memory.
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


# Re-export MemoryPolicy and PolicyDecision
@dataclass
class PolicyDecision:
    """The policy verdict for a single recalled memory."""
    memory: Memory
    action: str   # "follow" | "hint" | "ignore"
    reason: str
    prompt_injection: Optional[str]


class MemoryPolicy:
    """
    Thin agent-side policy layer that turns trust scores into actionable decisions.
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

        if state == "shadow":
            reason = f"trust={trust:.2f} shadow unreinforced"
        elif trust < 0.20:
            reason = f"trust={trust:.2f} very low trust"
        else:
            reason = f"trust={trust:.2f} below hint threshold"

        return "ignore", reason, None

    def inject_context(self, decisions: List[PolicyDecision]) -> str:
        lines = []
        for d in decisions:
            if d.prompt_injection is not None:
                lines.append(d.prompt_injection)
        return "\n".join(lines)

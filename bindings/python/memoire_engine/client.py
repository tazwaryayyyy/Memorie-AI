"""
Core Python client for Memoire. Wraps the PyO3 native extension.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Import the compiled PyO3 native module (memoire_engine.memoire_engine)
_NATIVE_IMPORT_ERROR: Optional[BaseException] = None
try:
    from .memoire_engine import (
        FOLLOW_THRESHOLD as _FOLLOW_THRESHOLD,
        HINT_THRESHOLD as _HINT_THRESHOLD,
        STRICT_FOLLOW_THRESHOLD as _STRICT_FOLLOW_THRESHOLD,
        STRICT_HINT_THRESHOLD as _STRICT_HINT_THRESHOLD,
        Memoire as _NativeMemoire,
        Memory as _NativeMemory,
        MemoireError,
    )
except ImportError as _exc:
    # If the native module is not compiled yet, import placeholder exceptions.
    # Keep the original error: it also covers a present-but-unloadable extension.
    _NATIVE_IMPORT_ERROR = _exc

    class MemoireError(Exception):
        """Raised when the Memoire library returns an error."""
    _NativeMemoire = None
    _NativeMemory = None
    _FOLLOW_THRESHOLD = _HINT_THRESHOLD = None
    _STRICT_FOLLOW_THRESHOLD = _STRICT_HINT_THRESHOLD = None

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
            "The native memoire_engine extension could not be imported "
            f"({_NATIVE_IMPORT_ERROR!r}).\n"
            "Run `maturin develop --release` in bindings/python inside the "
            "active virtualenv (needs a Rust toolchain)."
        )
    return _NativeMemoire


# Float32 trust values widened to Python floats can sit a hair below the decimal
# threshold they equal (float32(0.7) < 0.7), so policy comparisons allow this slack.
_TRUST_EPS = 1e-6


def policy_thresholds(strict: bool = False) -> Tuple[float, float]:
    """
    Return ``(follow, hint)`` trust thresholds.

    The values are defined once, in Rust (``memoire::quality::{FOLLOW,HINT}_THRESHOLD``
    and the ``STRICT_*`` pair), and read from the native extension here, so
    ``MemoryPolicy`` and the MCP server cannot drift from the engine.

    Raises:
        MemoireError: if the native extension is unavailable.
    """
    if _FOLLOW_THRESHOLD is None:
        _get_lib()  # raises MemoireError with build instructions
        raise MemoireError("native policy thresholds are unavailable")
    if strict:
        return _STRICT_FOLLOW_THRESHOLD, _STRICT_HINT_THRESHOLD
    return _FOLLOW_THRESHOLD, _HINT_THRESHOLD


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

    def __init__(
        self,
        db_path: str = "./memoire.db",
        namespace: str = "default",
        *,
        admission: Optional[str] = None,
        contradiction: Optional[str] = None,
        trial_cap: Optional[int] = None,
        quality_gate_threshold: Optional[float] = None,
        trial_min_cos: Optional[float] = None,
        verified_min_cos: Optional[float] = None,
    ) -> None:
        """
        Open or create a Memoire database at `db_path` scoped to `namespace`.

        Args:
            db_path: Path to the SQLite database file.
                     Pass ':memory:' for a non-persistent in-memory store.
            namespace: The isolation namespace for memories.
            admission: ``"admit_all"`` (default: every non-duplicate chunk enters as
                an unverified ``trial`` memory) or ``"quality_gate"`` (legacy centroid
                gate: low-scoring chunks are stored hidden as ``shadow``).
            contradiction: ``"nli"`` (default), ``"legacy"`` (cosine + polarity only)
                or ``"off"`` (never supersede or archive).
            trial_cap: Maximum unverified memories one recall may return (default 1).
                With no feedback history every memory is unverified, so a default
                recall returns at most one result; raise this for no-feedback use.
            quality_gate_threshold: Ingest importance at or above which
                ``admission="quality_gate"`` admits a chunk as ``trial`` (default 0.50);
                below it the chunk is stored hidden as ``shadow``. Values outside
                [0.0, 1.0] (and NaN) raise ``MemoireError``. Ignored under ``admission="admit_all"``. Calibrate it on a
                dev split, not on the data you report results on.
            trial_min_cos: Minimum cosine similarity for an *unverified* memory to be offered
                (default 0.0: no floor). Values outside [0.0, 1.0] (and NaN) raise ``MemoireError``.
            verified_min_cos: Minimum cosine similarity for a *verified* memory (one with any
                feedback) to be offered (default 0.0: no floor, so an unrelated row that once
                earned feedback can still surface for a new topic). Same validation. Neither
                floor applies to ``include_hidden=True``. Calibrate on a dev split.

        Raises:
            MemoireError: if the library fails to initialise.
        """
        if _NativeMemoire is None:
            raise MemoireError(
                "Native compiled memoire extension is not installed.\n"
                "Please run `pip install .` or `maturin dev` in bindings/python."
            )
        try:
            options = {}
            if admission is not None:
                options["admission"] = admission
            if contradiction is not None:
                options["contradiction"] = contradiction
            if trial_cap is not None:
                options["trial_cap"] = trial_cap
            if quality_gate_threshold is not None:
                options["quality_gate_threshold"] = float(quality_gate_threshold)
            if trial_min_cos is not None:
                options["trial_min_cos"] = float(trial_min_cos)
            if verified_min_cos is not None:
                options["verified_min_cos"] = float(verified_min_cos)
            self._impl = _NativeMemoire(db_path, namespace, **options)
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
            Number of chunks newly stored. This can be 0 for blank input, for
            exact-duplicate chunks, or for chunks rejected at ingestion. Use
            :meth:`remember_ids` if you need the ids of what was stored.
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

    def remember_ids(self, content: str) -> List[int]:
        """
        Like :meth:`remember`, but return the ids of the chunks actually stored.

        The ids are what :meth:`reinforce_if_used`, :meth:`penalize_if_used`,
        :meth:`forget` and :meth:`resolve_contradictions` take.

        Args:
            content: Any text.

        Returns:
            Row ids in chunk order. The list can be shorter than the number of
            chunks, or empty, when a chunk is an exact duplicate of an existing
            memory (the existing id is not returned) or is rejected at
            ingestion. Blank input returns ``[]``.
        """
        self._check_open()
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        if not content.strip():
            return []
        try:
            return list(self._impl.remember_ids(content))
        except Exception as e:
            raise MemoireError(f"remember_ids() failed: {e}") from e

    def recall(self, query: str, top_k: int = 5, include_hidden: bool = False) -> List[Memory]:
        """
        Return the `top_k` memories most semantically similar to `query`.

        Args:
            query: Natural language question or keyword phrase.
            top_k: Maximum number of results to return (default 5).
            include_hidden: Also return hidden rows (``shadow``, ``probationary``,
                ``quarantined``; ``exposure == "hidden"``) and skip the trial cap.
                An operator / debugging view; the default returns live rows only.

        Returns:
            List of Memory objects, sorted by score descending (0–1).
        """
        self._check_open()
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        try:
            return self._impl.recall(query, top_k, include_hidden)
        except Exception as e:
            raise MemoireError(f"recall() failed: {e}") from e

    def recall_mmr(self, query: str, top_k: int = 5, mmr_lambda: float = 0.5) -> List[Memory]:
        """
        Recall with Maximal Marginal Relevance reranking to reduce near-duplicates.

        Fetches ``max(3 * top_k, 20)`` live candidates (the default trial cap applies to
        them, so with no feedback history at most ``trial_cap`` unverified rows are
        candidates) and reranks them for relevance versus diversity.

        Args:
            query: Natural language question or keyword phrase.
            top_k: Maximum number of results to return (default 5).
            mmr_lambda: ``1.0`` is pure relevance (same order as :meth:`recall`), ``0.0``
                is pure diversity; default 0.5. Must be within [0.0, 1.0].
        """
        self._check_open()
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not math.isfinite(mmr_lambda) or not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda must be within [0.0, 1.0], got {mmr_lambda}")
        try:
            return self._impl.recall_mmr(query, top_k, mmr_lambda)
        except Exception as e:
            raise MemoireError(f"recall_mmr() failed: {e}") from e

    def reinforce_if_used(
        self,
        memory_id: int,
        agent_output: str,
        task_succeeded: bool,
        session_id: Optional[str] = None,
    ) -> bool:
        """
        Conditionally reinforce a memory based on actual use.
        """
        self._check_open()
        try:
            return self._impl.reinforce_if_used(memory_id, agent_output, task_succeeded, session_id)
        except Exception as e:
            raise MemoireError(f"reinforce_if_used() failed: {e}") from e

    def reinforce_declared(self, memory_id: int, session_id: Optional[str] = None) -> bool:
        """
        Reinforce a memory the agent *declared* it used, with no text-overlap gate.

        The caller asserts both the declaration and the task success. Returns False
        for an unknown or quarantined id, or when ``session_id`` equals the memory's
        last reinforcing session (one vote per memory per session).
        """
        self._check_open()
        try:
            return self._impl.reinforce_declared(memory_id, session_id)
        except Exception as e:
            raise MemoireError(f"reinforce_declared() failed: {e}") from e

    def unquarantine(self, memory_id: int) -> bool:
        """
        Operator action: return a quarantined memory to service.

        Clears its failure evidence (otherwise the next failure would re-quarantine
        it). Returns False if ``memory_id`` is not a quarantined memory.
        """
        self._check_open()
        try:
            return self._impl.unquarantine(memory_id)
        except Exception as e:
            raise MemoireError(f"unquarantine() failed: {e}") from e

    def promote_shadow(self, memory_id: int) -> bool:
        """Promote a hidden ``shadow`` / ``probationary`` memory to the live set."""
        self._check_open()
        try:
            return self._impl.promote_shadow(memory_id)
        except Exception as e:
            raise MemoireError(f"promote_shadow() failed: {e}") from e

    def recall_plain(self, query: str, top_k: int = 5) -> List[Memory]:
        """
        Plain cosine top-k over every stored memory.

        No trust, no state filter, no trial cap and no database writes: the
        plain-vector-memory baseline for evaluation. Results have
        ``exposure == "plain"``.
        """
        self._check_open()
        try:
            return self._impl.recall_plain(query, top_k)
        except Exception as e:
            raise MemoireError(f"recall_plain() failed: {e}") from e

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

    def recall_explained(
        self, query: str, top_k: int = 5, include_hidden: bool = False
    ) -> List[object]:
        """
        Return recalled memories with attribution signals explaining their ranking.

        ``include_hidden`` has the same meaning as in :meth:`recall`.
        """
        self._check_open()
        if not isinstance(query, str):
            raise TypeError(f"query must be str, got {type(query).__name__}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        try:
            return self._impl.recall_explained(query, top_k, include_hidden)
        except Exception as e:
            raise MemoireError(f"recall_explained() failed: {e}") from e

    def recompute_prototypes(self) -> None:
        """Recompute semantic scoring prototypes for the current embedder."""
        self._check_open()
        try:
            self._impl.recompute_prototypes()
        except Exception as e:
            raise MemoireError(f"recompute_prototypes() failed: {e}") from e
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

    def import_namespace(self, snapshot: dict, preserve_trust: bool = False) -> int:
        """Import memories from a namespace snapshot dict.

        By default only each memory's ``content`` is used and trust metadata
        (trust_ema, reinforcement_count, importance_base, confidence, created_at)
        is ignored, because a snapshot is untrusted input. Pass
        ``preserve_trust=True`` only to restore a snapshot you exported yourself;
        values are then clamped to sane bounds, which is not verification.
        """
        self._check_open()
        try:
            snapshot_str = json.dumps(snapshot)
            return self._impl.import_namespace(snapshot_str, preserve_trust)
        except Exception as e:
            raise MemoireError(f"import_namespace() failed: {e}") from e

    def start_background_maintenance(self, interval_secs: int, threshold: int) -> None:
        """
        Start scheduled background maintenance.
        - interval_secs: Interval in seconds for time-based maintenance.
        - threshold: Number of insertions that triggers maintenance.
        """
        self._check_open()
        try:
            self._impl.start_background_maintenance(interval_secs, threshold)
        except Exception as e:
            raise MemoireError(f"start_background_maintenance() failed: {e}") from e

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
        # Defined once in Rust and read from the native extension (see policy_thresholds).
        self.FOLLOW_THRESHOLD: float
        self.HINT_THRESHOLD: float
        self.FOLLOW_THRESHOLD, self.HINT_THRESHOLD = policy_thresholds(strict)

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

        if trust + _TRUST_EPS >= self.FOLLOW_THRESHOLD:
            reason = f"trust={trust:.2f} {state} reinforced"
            injection = f"[MEMORY - HIGH TRUST]: {mem.content}"
            return "follow", reason, injection

        if trust + _TRUST_EPS >= self.HINT_THRESHOLD:
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


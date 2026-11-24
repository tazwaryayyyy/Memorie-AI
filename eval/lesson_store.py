"""Thin adapter between the eval loop and a `memoire_engine.Memoire`-shaped object.

Everything the eval needs from Memoire goes through here, so tests can substitute a
fake that records calls. Only methods verified in bindings/python/memoire_engine/client.py
are used: remember_ids, recall, recall_plain, reinforce_declared, penalize_if_used, count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from .tasks.base import Lesson

FAILURE_SEVERITY = 1.0  # "direct failure" per Memoire.penalize_if_used docs
NAMESPACE = "eval"


class LessonIngestError(RuntimeError):
    pass


@dataclass
class Retrieved:
    key: str
    kind: str
    id: int
    text: str
    score: float
    trust: float
    exposure: str
    state: str
    cos: Optional[float] = None  # raw cosine(query, lesson); `score` is the engine's BLENDED rank score


def make_memoire(
    contradiction: str = "off",
    trial_cap: int = 1,
    admission: str = "admit_all",
    quality_gate_threshold: Optional[float] = None,
    trial_min_cos: Optional[float] = None,
    verified_min_cos: Optional[float] = None,
) -> Any:
    """Fresh in-memory Memoire. Imported lazily so offline tests never need the native module.
    Optional knobs are passed only when set, so an unset knob means the engine default."""
    from memoire_engine import Memoire

    options: Dict[str, Any] = {}
    if quality_gate_threshold is not None:
        options["quality_gate_threshold"] = quality_gate_threshold
    if trial_min_cos is not None:
        options["trial_min_cos"] = trial_min_cos
    if verified_min_cos is not None:
        options["verified_min_cos"] = verified_min_cos
    return Memoire(
        ":memory:",
        NAMESPACE,
        admission=admission,
        contradiction=contradiction,
        trial_cap=trial_cap,
        **options,
    )


class LessonStore:
    def __init__(self, memoire: Any) -> None:
        self._m = memoire
        self._by_id: Dict[int, Lesson] = {}
        self._id_by_key: Dict[str, int] = {}
        self.shadowed: Set[str] = set()  # lessons the ingest gate stored hidden (quality_gate only)

    def hidden_keys(self) -> List[str]:
        """Keys of lessons currently hidden (shadow or quarantined). Uses the operator view
        (`include_hidden=True`), so call it sparingly and only when needed."""
        total = len(self._by_id)
        if total == 0:
            return []
        rows = self._m.recall("lesson", top_k=total, include_hidden=True)
        return sorted(self._by_id[m.id].key for m in rows if str(m.exposure) == "hidden")

    def ingest(self, lesson: Lesson, track_shadow: bool = False) -> int:
        ids = list(self._m.remember_ids(lesson.text))
        if len(ids) != 1:
            raise LessonIngestError(
                f"lesson {lesson.key!r} produced {len(ids)} chunks/ids ({ids}); lessons must be a single "
                "non-duplicate chunk so feedback attributes to exactly one row"
            )
        self._by_id[ids[0]] = lesson
        self._id_by_key[lesson.key] = ids[0]
        if track_shadow and lesson.key in self.hidden_keys():
            self.shadowed.add(lesson.key)
        return ids[0]

    def count(self) -> int:
        return int(self._m.count())

    def _wrap(self, memories: Sequence[Any]) -> List[Retrieved]:
        out: List[Retrieved] = []
        for mem in memories:
            lesson = self._by_id.get(mem.id)
            if lesson is None:
                raise LessonIngestError(f"recalled id {mem.id} is not a known lesson")
            out.append(
                Retrieved(
                    key=lesson.key,
                    kind=lesson.kind,
                    id=mem.id,
                    text=lesson.text,
                    score=float(mem.score),
                    trust=float(mem.trust),
                    exposure=str(mem.exposure),
                    state=str(mem.state),
                )
            )
        return out

    def _plain_cosines(self, query: str) -> Dict[int, float]:
        """id -> raw cosine for every lesson, via the read-only `recall_plain` (no trust, no writes)."""
        total = len(self._by_id)
        return {m.id: float(m.score) for m in self._m.recall_plain(query, top_k=total)} if total else {}

    def recall(self, query: str, top_k: int) -> List[Retrieved]:
        rows = self._wrap(self._m.recall(query, top_k=top_k))
        if rows:
            cosines = self._plain_cosines(query)
            for row in rows:
                row.cos = cosines.get(row.id)
        return rows

    def recall_plain(self, query: str, top_k: int) -> List[Retrieved]:
        rows = self._wrap(self._m.recall_plain(query, top_k=top_k))
        for row in rows:
            row.cos = row.score  # plain recall's score IS the cosine
        return rows

    def reinforce(self, key: str, session_id: str) -> bool:
        return bool(self._m.reinforce_declared(self._id_by_key[key], session_id))

    def penalize(self, keys: Sequence[str], severity: float = FAILURE_SEVERITY) -> List[dict]:
        ids = [self._id_by_key[k] for k in keys]
        by_id = {v: k for k, v in self._id_by_key.items()}
        outcomes = self._m.penalize_if_used(ids, severity)
        return [
            {
                "key": by_id.get(o.get("id")),
                "trust_before": o.get("trust_before"),
                "trust_after": o.get("trust_after"),
            }
            for o in outcomes
        ]

    def key_for_index(self, shown: Sequence[Retrieved], index: int) -> str:
        return shown[index - 1].key

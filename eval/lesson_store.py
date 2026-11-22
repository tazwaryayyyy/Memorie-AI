"""Thin adapter between the eval loop and a `memoire_engine.Memoire`-shaped object.

Everything the eval needs from Memoire goes through here, so tests can substitute a
fake that records calls. Only methods verified in bindings/python/memoire_engine/client.py
are used: remember_ids, recall, recall_plain, reinforce_declared, penalize_if_used, count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

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


def make_memoire(contradiction: str = "off", trial_cap: int = 1, admission: str = "admit_all") -> Any:
    """Fresh in-memory Memoire. Imported lazily so offline tests never need the native module."""
    from memoire_engine import Memoire

    return Memoire(
        ":memory:",
        NAMESPACE,
        admission=admission,
        contradiction=contradiction,
        trial_cap=trial_cap,
    )


class LessonStore:
    def __init__(self, memoire: Any) -> None:
        self._m = memoire
        self._by_id: Dict[int, Lesson] = {}
        self._id_by_key: Dict[str, int] = {}

    def ingest(self, lesson: Lesson) -> int:
        ids = list(self._m.remember_ids(lesson.text))
        if len(ids) != 1:
            raise LessonIngestError(
                f"lesson {lesson.key!r} produced {len(ids)} chunks/ids ({ids}); lessons must be a single "
                "non-duplicate chunk so feedback attributes to exactly one row"
            )
        self._by_id[ids[0]] = lesson
        self._id_by_key[lesson.key] = ids[0]
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

    def recall(self, query: str, top_k: int) -> List[Retrieved]:
        return self._wrap(self._m.recall(query, top_k=top_k))

    def recall_plain(self, query: str, top_k: int) -> List[Retrieved]:
        return self._wrap(self._m.recall_plain(query, top_k=top_k))

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

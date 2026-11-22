"""Arm definitions: what each arm retrieves and which feedback it is allowed to write.

A  no_memory         no store, no notes section.
B  plain_vector      LessonStore.recall_plain: cosine top-k, no trust, no writes.
C0 memoire_static    LessonStore.recall (trust-ranked, trial-capped), never any feedback.
C1 memoire_feedback  like C0, plus outcome-attributed feedback on DECLARED notes only:
                     task passed -> reinforce_declared; task failed -> penalize_if_used.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from .lesson_store import FAILURE_SEVERITY, LessonStore, Retrieved
from .tasks.base import Task

ARM_A = "no_memory"
ARM_B = "plain_vector"
ARM_C0 = "memoire_static"
ARM_C1 = "memoire_feedback"
ALL_ARMS = (ARM_A, ARM_B, ARM_C0, ARM_C1)
_ALIASES = {"A": ARM_A, "B": ARM_B, "C0": ARM_C0, "C1": ARM_C1}


def normalize_arm(name: str) -> str:
    arm = _ALIASES.get(name.strip().upper(), name.strip())
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of A,B,C0,C1 or {ALL_ARMS}")
    return arm


def uses_memory(arm: str) -> bool:
    return arm != ARM_A


def has_feedback(arm: str) -> bool:
    return arm == ARM_C1


def retrieve(arm: str, store: Optional[LessonStore], task: Task, top_k: int) -> List[Retrieved]:
    if arm == ARM_A:
        return []
    assert store is not None, f"arm {arm} needs a LessonStore"
    if arm == ARM_B:
        return store.recall_plain(task.spec, top_k)
    return store.recall(task.spec, top_k)


def apply_feedback(
    arm: str,
    store: Optional[LessonStore],
    shown: Sequence[Retrieved],
    used_indices: Sequence[int],
    passed: bool,
    session_id: str,
) -> List[dict]:
    """Write feedback for C1 only; returns replayable event records (lesson keys, not row ids)."""
    if not has_feedback(arm) or not used_indices:
        return []
    assert store is not None
    keys = [store.key_for_index(shown, i) for i in used_indices]
    if passed:
        return [{"op": "reinforce", "key": k, "applied": store.reinforce(k, session_id)} for k in keys]
    return [{"op": "penalize", "keys": keys, "severity": FAILURE_SEVERITY, "outcomes": store.penalize(keys)}]


def replay_feedback(store: LessonStore, events: Sequence[dict], session_id: str) -> None:
    """Re-apply recorded C1 feedback when resuming, so the store matches an uninterrupted run."""
    for event in events:
        if event["op"] == "reinforce":
            store.reinforce(event["key"], session_id)
        elif event["op"] == "penalize":
            store.penalize(event["keys"], event.get("severity", FAILURE_SEVERITY))
        else:
            raise ValueError(f"unknown feedback op {event['op']!r}")

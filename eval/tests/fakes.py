"""Deterministic stand-ins: a call-recording Memoire fake and a scripted LLM (no network)."""
from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Sequence

from eval.llm import LLMResponse
from eval.tasks.base import Task

WRITE_METHODS = ("remember_ids", "reinforce_declared", "penalize_if_used", "forget")


def _words(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


class FakeMemoire:
    """Word-overlap 'embedding'. Mirrors the client.Memoire surface the harness uses."""

    def __init__(self, events: Optional[list] = None) -> None:
        self.rows: Dict[int, str] = {}
        self.succ: Dict[int, int] = {}
        self.fail: Dict[int, int] = {}
        self.calls: List[tuple] = []
        self.events = events if events is not None else []
        self._next = 1

    def _log(self, name: str, *args) -> None:
        self.calls.append((name, args))
        self.events.append((name, args))

    def remember_ids(self, text: str) -> List[int]:
        self._log("remember_ids", text)
        if text in self.rows.values():
            return []
        i, self._next = self._next, self._next + 1
        self.rows[i] = text
        self.succ[i] = self.fail[i] = 0
        return [i]

    def count(self) -> int:
        return len(self.rows)

    def _rank(self, query: str, top_k: int, exposure: str):
        q = _words(query)
        scored = sorted(
            ((len(q & _words(t)) / (1 + len(_words(t))), i) for i, t in self.rows.items()),
            key=lambda x: (-x[0], x[1]),
        )
        return [
            SimpleNamespace(id=i, content=self.rows[i], score=s, trust=0.5, exposure=exposure, state="trial")
            for s, i in scored[:top_k]
        ]

    def recall(self, query: str, top_k: int = 5, include_hidden: bool = False):
        self._log("recall", query, top_k)
        return self._rank(query, top_k, "trial")

    def recall_plain(self, query: str, top_k: int = 5):
        self._log("recall_plain", query, top_k)
        return self._rank(query, top_k, "plain")

    def reinforce_declared(self, memory_id: int, session_id: Optional[str] = None) -> bool:
        self._log("reinforce_declared", memory_id, session_id)
        self.succ[memory_id] += 1
        return True

    def penalize_if_used(self, memory_ids: Sequence[int], failure_severity: float = 1.0):
        self._log("penalize_if_used", tuple(memory_ids), failure_severity)
        out = []
        for i in memory_ids:
            self.fail[i] += 1
            out.append({"id": i, "trust_before": 0.5, "trust_after": 0.4, "uncertainty_after": 0.8})
        return out

    def writes_after_ingest(self) -> List[tuple]:
        return [c for c in self.calls if c[0] in WRITE_METHODS and c[0] != "remember_ids"]

    def state(self):
        return dict(self.succ), dict(self.fail)


def code_block(code: str) -> str:
    return f"Here is the fix.\n```python\n{code}```\n"


class ScriptedLLM:
    """Answers with a task's reference/naive solution. `choose(task) -> 'ref'|'naive'|'nocode'`;
    `declare(n_shown) -> list[int]` is the USED_MEMORY line (None = omit the line)."""

    def __init__(
        self,
        tasks: Sequence[Task],
        choose: Callable[[Task], str] = lambda t: "ref",
        declare: Callable[[int], Optional[List[int]]] = lambda n: [1] if n else [],
        events: Optional[list] = None,
    ) -> None:
        self.tasks = list(tasks)
        self.choose, self.declare = choose, declare
        self.total_requests = 0
        self.prompts: List[list] = []
        self.seeds: List[Optional[int]] = []
        self.events = events if events is not None else []

    def complete(self, messages, seed=None) -> LLMResponse:
        self.total_requests += 1
        self.prompts.append(messages)
        self.seeds.append(seed)
        user = messages[-1]["content"]
        task = next(t for t in self.tasks if t.spec in user)
        self.events.append(("llm", task.task_id))
        mode = self.choose(task)
        text = "I could not do it." if mode == "nocode" else code_block(task.reference if mode == "ref" else task.naive)
        n_shown = len(re.findall(r"^\[M\d+\]", user, re.MULTILINE))
        if "USED_MEMORY" in user:
            used = self.declare(n_shown)
            if used is not None:
                text += f"USED_MEMORY: [{', '.join(str(u) for u in used)}]\n"
        return LLMResponse(text=text, prompt_tokens=100, completion_tokens=50, latency_s=0.01,
                           attempts=1, model_returned="scripted-1", finish_reason="stop")

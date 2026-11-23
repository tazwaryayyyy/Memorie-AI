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


class EngineModel:
    """Python mirror of the Rust exposure / quarantine rules, for timeline reasoning only.

    Mirrors src/quality.rs (posterior_trust Beta(2,2), should_quarantine `2f > 3s + 2`,
    meets_follow `3s >= 8 + 7f`) and src/store.rs (`unverified` = s+f == 0, `exposure_label`,
    `select_with_trial_cap`, final score = 0.75 sim + 0.20 trust). Recency and decay are ignored.
    eval/tests/test_native_integration.py asserts the same facts against the real engine; if that
    test ever disagrees with this class, this class is wrong.
    """

    def __init__(self, sims: dict, trial_cap: int = 1, k: int = 3, verified_min_cos: float = 0.0,
                 trial_min_cos: float = 0.0) -> None:
        self.sims, self.trial_cap, self.k = dict(sims), trial_cap, k
        self.verified_min_cos, self.trial_min_cos = verified_min_cos, trial_min_cos
        self.s: dict = {}
        self.f: dict = {}
        self.hidden: set = set()

    def add(self, key: str) -> None:
        self.s[key], self.f[key] = 0, 0

    @staticmethod
    def posterior(s: int, f: int) -> float:
        return (2 + s) / (4 + s + f)

    def exposure(self, key: str) -> str:
        if key in self.hidden:
            return "hidden"
        s, f = self.s[key], self.f[key]
        if s + f == 0:
            return "trial"
        return "follow" if 3 * s >= 8 + 7 * f else "hint"

    def reinforce(self, key: str) -> None:
        self.s[key] += 1

    def penalize(self, keys) -> None:
        for key in keys:
            if key in self.hidden:
                continue
            self.f[key] += 1
            if 2 * self.f[key] > 3 * self.s[key] + 2:
                self.hidden.add(key)

    def recall(self) -> list:
        live = [key for key in self.s if key not in self.hidden]
        score = {key: 0.75 * self.sims.get(key, 0.0) + 0.20 * self.posterior(self.s[key], self.f[key]) for key in live}
        verified = sorted(
            (k for k in live if self.s[k] + self.f[k] > 0 and self.sims.get(k, 0.0) >= self.verified_min_cos),
            key=lambda k: -score[k],
        )
        trial = sorted(
            (k for k in live if self.s[k] + self.f[k] == 0 and self.sims.get(k, 0.0) >= self.trial_min_cos),
            key=lambda k: -score[k],
        )
        max_trial = min(self.trial_cap, self.k)
        n_trial = min(len(trial), max_trial) if not verified else (0 if self.k < 2 else min(len(trial), max_trial, self.k - 1))
        n_verified = min(len(verified), max(self.k - n_trial, 0))
        return sorted(verified[:n_verified] + trial[:n_trial], key=lambda k: -score[k])


class EngineBackedFake:
    """Memoire-shaped fake whose recall/feedback follow EngineModel (verified rows bypass the trial
    cap, hidden after quarantine) with a Jaccard word-overlap similarity. Lets runner-level tests
    play out multi-session feedback timelines offline."""

    def __init__(self, trial_cap: int = 1, verified_min_cos: float = 0.0, trial_min_cos: float = 0.0,
                 bonus=lambda text: 0.0) -> None:
        self.rows: dict = {}
        self.bonus = bonus
        self.model = EngineModel({}, trial_cap=trial_cap, verified_min_cos=verified_min_cos,
                                 trial_min_cos=trial_min_cos)
        self.calls: list = []
        self._next = 1

    def _sim(self, query: str, text: str) -> float:
        a, b = _words(query), _words(text)
        return (len(a & b) / len(a | b) if a | b else 0.0) + self.bonus(text)

    def remember_ids(self, text: str) -> list:
        if text in self.rows.values():
            return []
        i, self._next = self._next, self._next + 1
        self.rows[i] = text
        self.model.add(i)
        return [i]

    def count(self) -> int:
        return len(self.rows)

    def _memory(self, i: int, exposure: str) -> SimpleNamespace:
        m = self.model
        return SimpleNamespace(id=i, content=self.rows[i], score=m.sims[i], trust=m.posterior(m.s[i], m.f[i]),
                               exposure=exposure, state="quarantined" if i in m.hidden else "trial")

    def recall(self, query: str, top_k: int = 5, include_hidden: bool = False):
        self.calls.append(("recall", query, top_k, include_hidden))
        for i, text in self.rows.items():
            self.model.sims[i] = self._sim(query, text)
        self.model.k = top_k
        if include_hidden:
            ordered = sorted(self.rows, key=lambda i: -self.model.sims[i])[:top_k]
            return [self._memory(i, "hidden" if i in self.model.hidden else self.model.exposure(i)) for i in ordered]
        return [self._memory(i, self.model.exposure(i)) for i in self.model.recall()]

    def recall_plain(self, query: str, top_k: int = 5):
        self.calls.append(("recall_plain", query, top_k))
        for i, text in self.rows.items():
            self.model.sims[i] = self._sim(query, text)
        ordered = sorted(self.rows, key=lambda i: -self.model.sims[i])[:top_k]
        return [self._memory(i, "plain") for i in ordered]

    def reinforce_declared(self, memory_id: int, session_id=None) -> bool:
        self.calls.append(("reinforce_declared", memory_id, session_id))
        self.model.reinforce(memory_id)
        return True

    def penalize_if_used(self, memory_ids, failure_severity: float = 1.0):
        self.calls.append(("penalize_if_used", tuple(memory_ids), failure_severity))
        self.model.penalize(list(memory_ids))
        return [{"id": i, "trust_before": 0.5, "trust_after": 0.4} for i in memory_ids]


class GateFake(FakeMemoire):
    """FakeMemoire where lessons whose text is in `shadow_texts` are stored hidden (quality_gate)."""

    def __init__(self, shadow_texts) -> None:
        super().__init__()
        self.shadow_texts = set(shadow_texts)

    def recall(self, query: str, top_k: int = 5, include_hidden: bool = False):
        self._log("recall", query, top_k)
        live = [m for m in self._rank(query, len(self.rows), "trial") if m.content not in self.shadow_texts]
        if include_hidden:
            everything = self._rank(query, len(self.rows), "trial")
            for m in everything:
                if m.content in self.shadow_texts:
                    m.exposure = "hidden"
            return everything[:top_k]
        return live[:top_k]

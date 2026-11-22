"""Task / lesson data model and template expansion for the eval task suite."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

LESSON_KINDS = ("gold", "vague", "wrong")  # in the bank from the start
POISON_KIND = "poison"  # ingested right before `Family.poison_session`


@dataclass(frozen=True)
class Task:
    family: str
    session: int
    split: str
    spec: str
    solution_stub: str      # agent-visible solution.py
    visible_tests: str      # agent-visible test_visible.py (unittest)
    hidden_tests: str       # grading-only test_hidden.py (unittest)
    entrypoints: Tuple[str, ...]
    reference: str          # full solution.py: must pass visible + hidden
    naive: str              # full solution.py: must pass visible, fail hidden

    @property
    def task_id(self) -> str:
        return f"{self.family}/s{self.session}"


@dataclass(frozen=True)
class Lesson:
    key: str      # "<family>:<kind>", stable across runs
    family: str
    kind: str     # gold | vague | wrong | poison
    text: str


@dataclass(frozen=True)
class Family:
    name: str
    split: str
    tasks: Tuple[Task, ...]        # ordered by session, 1-based
    lessons: Tuple[Lesson, ...]    # gold, vague, wrong, poison
    poison_session: int = 2

    @property
    def bank_lessons(self) -> Tuple[Lesson, ...]:
        return tuple(l for l in self.lessons if l.kind in LESSON_KINDS)

    @property
    def poison(self) -> Lesson:
        return next(l for l in self.lessons if l.kind == POISON_KIND)


_PLACEHOLDER = re.compile(r"@@([A-Za-z0-9_]+)@@")


def expand(template: str, params: Mapping[str, str]) -> str:
    """Replace ``@@key@@`` with ``params[key]``; unknown keys are an error."""

    def sub(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in params:
            raise KeyError(f"unresolved placeholder @@{key}@@")
        return str(params[key])

    return _PLACEHOLDER.sub(sub, template)


def build_family(
    name: str,
    split: str,
    templates: Mapping[str, str],
    variants: Sequence[Mapping[str, str]],
    lessons: Mapping[str, str],
    poison_session: int = 2,
) -> Family:
    """Expand one family template over its per-session parameter sets."""
    tasks: List[Task] = []
    for index, params in enumerate(variants, start=1):
        entry = expand(templates["entrypoints"], params).split()
        tasks.append(
            Task(
                family=name,
                session=index,
                split=split,
                spec=expand(templates["spec"], params).strip(),
                solution_stub=expand(templates["stub"], params).strip() + "\n",
                visible_tests=expand(templates["visible"], params).strip() + "\n",
                hidden_tests=expand(templates["hidden"], params).strip() + "\n",
                entrypoints=tuple(entry),
                reference=expand(templates["reference"], params).strip() + "\n",
                naive=expand(templates["naive"], params).strip() + "\n",
            )
        )
    poison_params = variants[poison_session - 1]
    out: List[Lesson] = []
    for kind in (*LESSON_KINDS, POISON_KIND):
        text = expand(lessons[kind], poison_params if kind == POISON_KIND else {})
        out.append(Lesson(key=f"{name}:{kind}", family=name, kind=kind, text=" ".join(text.split())))
    return Family(name=name, split=split, tasks=tuple(tasks), lessons=tuple(out), poison_session=poison_session)


def canonical_hash(obj: object) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def families_hash(families: Sequence[Family]) -> Dict[str, str]:
    """Separate content hashes for tasks and lessons (both enter the run fingerprint)."""
    tasks = [
        [t.task_id, t.spec, t.solution_stub, t.visible_tests, t.hidden_tests, list(t.entrypoints)]
        for f in families
        for t in f.tasks
    ]
    lessons = [[l.key, l.text] for f in families for l in f.lessons]
    return {"tasks": canonical_hash(tasks), "lessons": canonical_hash(lessons)}

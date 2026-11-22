"""Run loop: schedule, per-arm isolation, JSONL checkpoint/resume with C1 feedback replay."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import HARNESS_VERSION
from .arms import apply_feedback, has_feedback, replay_feedback, retrieve, uses_memory
from .lesson_store import LessonStore
from .prompt import build_messages, parse_response, prompt_template_hash_input
from .sandbox import grade
from .tasks.base import Family, Task, canonical_hash, families_hash

RunKey = Tuple[str, str, int, int]  # (arm, family, session, seed)


class BudgetExhausted(RuntimeError):
    pass


class FingerprintMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSettings:
    split: str
    families: Tuple[str, ...]
    sessions: Tuple[int, ...]
    model: str
    base_url: str
    temperature: float
    top_k: int
    trial_cap: int
    contradiction: str
    max_tokens: int
    send_api_seed: bool = False

    def fingerprint(self, families: Sequence[Family]) -> str:
        hashes = families_hash(families)
        return canonical_hash(
            {
                "harness": HARNESS_VERSION,
                "settings": self.__dict__,
                "tasks": hashes["tasks"],
                "lessons": hashes["lessons"],
                "prompt": canonical_hash(prompt_template_hash_input()),
            }
        )


def select_families(all_families: Sequence[Family], names: Optional[Sequence[str]]) -> List[Family]:
    if not names:
        return list(all_families)
    known = {f.name: f for f in all_families}
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(f"unknown families {missing}; known: {sorted(known)}")
    return [known[n] for n in names]


class ResultsFile:
    """Append-only JSONL. A torn trailing line (crash mid-write) is dropped and truncated."""

    def __init__(self, path: str, fingerprint: str) -> None:
        self.path, self.fingerprint = path, fingerprint
        self._records: Dict[RunKey, dict] = {}
        self.ordered: List[dict] = []
        if os.path.exists(path):
            self._load()

    def _load(self) -> None:
        with open(self.path, "rb") as fh:
            raw = fh.read()
        keep, offset = 0, 0
        lines = raw.split(b"\n")
        for i, line in enumerate(lines):
            is_last = i == len(lines) - 1
            end = offset + len(line) + (0 if is_last else 1)
            if line.strip():
                try:
                    record = json.loads(line.decode("utf-8"))
                except ValueError:
                    if all(not rest.strip() for rest in lines[i + 1:]):
                        break  # torn tail
                    raise ValueError(f"{self.path}: corrupt record on line {i + 1}") from None
                if is_last and not raw.endswith(b"\n"):
                    break  # complete JSON but unterminated: treat as torn to stay line-aligned
                self._ingest(record)
            keep = end
            offset = end
        if keep != len(raw):
            with open(self.path, "r+b") as fh:
                fh.truncate(keep)

    def _ingest(self, record: dict) -> None:
        if record.get("fingerprint") != self.fingerprint:
            raise FingerprintMismatch(
                f"{self.path} was written with different settings/tasks/lessons/prompt "
                f"(fingerprint {str(record.get('fingerprint'))[:12]} != {self.fingerprint[:12]}). "
                "Use a new --out file."
            )
        key = (record["arm"], record["family"], record["session"], record["seed"])
        if key not in self._records:
            self._records[key] = record
            self.ordered.append(record)

    def has(self, key: RunKey) -> bool:
        return key in self._records

    def get(self, key: RunKey) -> Optional[dict]:
        return self._records.get(key)

    def append(self, record: dict) -> None:
        record["fingerprint"] = self.fingerprint
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "ab") as fh:
            fh.write(line.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        self._ingest(record)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_sequence(
    *,
    settings: RunSettings,
    arm: str,
    seed: int,
    families: Sequence[Family],
    llm: Any,
    results: ResultsFile,
    store_factory: Callable[[], Any],
    diag: Optional[Mapping[str, dict]] = None,
    max_calls: int = 0,
    log: Callable[[str], None] = print,
) -> Dict[str, int]:
    """One (arm, seed) pass over the schedule. Fresh store per call; completed runs are skipped
    (and, for C1, their recorded feedback is replayed so the store matches an uninterrupted run)."""
    diag = diag or {}
    store: Optional[LessonStore] = None
    if uses_memory(arm):
        store = LessonStore(store_factory())
        for fam in families:
            for lesson in fam.bank_lessons:
                store.ingest(lesson)
    stats = {"ran": 0, "skipped": 0, "passed": 0}

    for fam in families:
        for task in fam.tasks:
            if task.session not in settings.sessions:
                continue
            if store is not None and task.session == fam.poison_session:
                store.ingest(fam.poison)
            key: RunKey = (arm, fam.name, task.session, seed)
            session_id = f"{fam.name}-s{task.session}-seed{seed}"

            done = results.get(key)
            if done is not None:
                if has_feedback(arm):
                    assert store is not None
                    replay_feedback(store, done.get("feedback_events", []), session_id)
                stats["skipped"] += 1
                continue
            if max_calls and getattr(llm, "total_requests", 0) >= max_calls:
                raise BudgetExhausted(f"--max-calls {max_calls} reached; rerun the same command to resume")

            shown = retrieve(arm, store, task, settings.top_k)
            messages = build_messages(task, [r.text for r in shown] if uses_memory(arm) else None)
            response = llm.complete(messages, seed=seed)
            parsed = parse_response(response.text, task, len(shown), uses_memory(arm))

            if parsed.code is None:
                grading: dict = {"passed": False, "visible": None, "hidden": None, "reason": "no_code_block"}
                events: List[dict] = []
                feedback_note = "skipped: no code block (format failure is not evidence about the notes)"
            else:
                grading = grade(task, parsed.code).to_json()
                used = parsed.used_indices
                events = apply_feedback(arm, store, shown, used, grading["passed"], session_id)
                feedback_note = None
            passed = bool(grading["passed"])
            stats["ran"] += 1
            stats["passed"] += int(passed)

            record = {
                "schema": 1,
                "ts": _now(),
                "arm": arm,
                "family": fam.name,
                "session": task.session,
                "seed": seed,
                "split": settings.split,
                "model": settings.model,
                "model_returned": response.model_returned,
                "temperature": settings.temperature,
                "send_api_seed": settings.send_api_seed,
                "top_k": settings.top_k,
                "trial_cap": settings.trial_cap,
                "contradiction": settings.contradiction,
                "prompt": messages,
                "response": response.text,
                "finish_reason": response.finish_reason,
                "parsed": parsed.to_json(),
                "used_keys": [shown[i - 1].key for i in parsed.used_indices],
                "retrieved": [
                    {
                        "key": r.key, "kind": r.kind, "id": r.id, "score": r.score, "trust": r.trust,
                        "exposure": r.exposure, "state": r.state,
                        "exemplar_cos": (diag.get(f"lesson:{r.key}") or {}).get("max_cos"),
                        "quality_score": ((diag.get(f"lesson:{r.key}") or {}).get("quality") or {}).get("importance_base"),
                    }
                    for r in shown
                ],
                "spec_exemplar_cos": (diag.get(f"spec:{task.task_id}") or {}).get("max_cos"),
                "grading": grading,
                "passed": passed,
                "feedback_events": events,
                "feedback_note": feedback_note,
                "usage": {"prompt_tokens": response.prompt_tokens, "completion_tokens": response.completion_tokens},
                "latency_s": round(response.latency_s, 3),
                "attempts": response.attempts,
            }
            results.append(record)
            log(f"[{arm} seed={seed}] {task.task_id}: {'PASS' if passed else 'FAIL'}"
                f" (tokens in/out {response.prompt_tokens}/{response.completion_tokens}, {response.latency_s:.1f}s)")
    return stats

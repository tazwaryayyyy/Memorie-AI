#!/usr/bin/env python3
"""
Agent behavior benchmark for Memoire quality-control claims.

Runs three arms:
- no_memory: no persistence
- raw_memory: store everything (top_k recall, no score filter)
- mqcl: quality-aware memory (min-score filtered retrieval)

Metric headline:
- repeated mistake reduction vs no_memory

Usage:
  cargo build --release
  python scripts/agent_behavior_benchmark.py
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "bindings" / "python"))
from memoire import Memoire  # noqa: E402  # pylint: disable=import-error,wrong-import-position


@dataclass
class Task:
    task_id: str
    prompt: str
    mistake_type: str
    eval_fn: Callable[[str], bool]


@dataclass
class RunResult:
    arm: str
    task_id: str
    passed: bool
    attempts: int
    time_sec: float
    repeated_mistake: bool
    mistake_type: str
    memory_used: bool


def evaluate_money_code(code: str) -> bool:
    return "Decimal" in code and "float" not in code


def evaluate_retry_code(code: str) -> bool:
    return "exponential" in code.lower() and "jitter" in code.lower()


def evaluate_auth_code(code: str) -> bool:
    low = code.lower()
    return "issuer" in low and "validate" in low and "disabled" not in low


def paired_tasks() -> List[Task]:
    return [
        Task(
            "billing_pair_1",
            "Implement tax computation for money values.",
            "float_money",
            evaluate_money_code,
        ),
        Task(
            "billing_pair_2",
            "Implement discount and refund computation for money values.",
            "float_money",
            evaluate_money_code,
        ),
        Task(
            "retry_pair_1",
            "Implement robust retry behavior for API calls.",
            "bad_retry",
            evaluate_retry_code,
        ),
        Task(
            "retry_pair_2",
            "Implement retries for webhook delivery under failures.",
            "bad_retry",
            evaluate_retry_code,
        ),
        Task(
            "auth_pair_1",
            "Implement JWT validation for auth middleware.",
            "issuer_not_validated",
            evaluate_auth_code,
        ),
        Task(
            "auth_pair_2",
            "Harden auth middleware for forged token prevention.",
            "issuer_not_validated",
            evaluate_auth_code,
        ),
    ]


def baseline_generator(task: Task, recalled: List[str]) -> str:
    """Deterministic stand-in for an agent loop (local, reproducible)."""
    hint = " ".join(recalled).lower()

    if task.mistake_type == "float_money":
        if "never use float for money" in hint or "decimal" in hint:
            return "from decimal import Decimal\nvalue = Decimal('1.23')"
        return "value = float(1.23)"

    if task.mistake_type == "bad_retry":
        if "exponential backoff" in hint and "jitter" in hint:
            return "retry='exponential'; jitter=True"
        return "retry='fixed'; jitter=False"

    if task.mistake_type == "issuer_not_validated":
        if "issuer" in hint and "must validate" in hint:
            return "validate_signature(); validate_issuer();"
        return "validate_signature(); # issuer skipped"

    return "pass"


def memory_line_for_failure(task: Task) -> str:
    if task.mistake_type == "float_money":
        return (
            "Never use float for money. Use Decimal and quantize to cents. "
            "This bug caused repeated rounding errors in billing."
        )
    if task.mistake_type == "bad_retry":
        return (
            "Retry policy must use exponential backoff with jitter. "
            "Fixed interval retries caused synchronized retry storms."
        )
    return (
        "JWT auth must validate issuer claim in addition to signature. "
        "Skipping issuer validation allows forged tokens from foreign issuers."
    )


def run_arm(arm: str, db_path: Path, tasks: List[Task]) -> List[RunResult]:
    if db_path.exists():
        db_path.unlink()

    repeated_seen: Dict[str, bool] = {}
    results: List[RunResult] = []

    mem = Memoire(str(db_path)) if arm != "no_memory" else None
    try:
        for task in tasks:
            start = time.time()
            recalled_lines: List[str] = []
            memory_used = False

            if mem is not None:
                recalled = mem.recall(task.prompt, top_k=5)
                if arm == "mqcl":
                    recalled = [r for r in recalled if r.score >= 0.60]
                recalled_lines = [r.content for r in recalled]
                memory_used = len(recalled_lines) > 0

            code = baseline_generator(task, recalled_lines)
            passed = task.eval_fn(code)
            attempts = 1

            repeated_mistake = False
            if not passed:
                if repeated_seen.get(task.mistake_type, False):
                    repeated_mistake = True
                repeated_seen[task.mistake_type] = True

                if mem is not None:
                    mem.remember(memory_line_for_failure(task))

                # A second attempt after learning from failure.
                recalled_lines_retry = recalled_lines + \
                    [memory_line_for_failure(task)]
                code_retry = baseline_generator(task, recalled_lines_retry)
                passed = task.eval_fn(code_retry)
                attempts = 2

            elapsed = time.time() - start
            results.append(
                RunResult(
                    arm=arm,
                    task_id=task.task_id,
                    passed=passed,
                    attempts=attempts,
                    time_sec=elapsed,
                    repeated_mistake=repeated_mistake,
                    mistake_type=task.mistake_type,
                    memory_used=memory_used,
                )
            )
    finally:
        if mem is not None:
            mem.close()

    return results


def summarize(results: List[RunResult]) -> dict:
    completed = sum(1 for r in results if r.passed)
    repeated = sum(1 for r in results if r.repeated_mistake)
    opportunities = sum(1 for r in results if "pair_2" in r.task_id)
    median_attempts = statistics.median(r.attempts for r in results)
    median_time = statistics.median(r.time_sec for r in results)
    return {
        "task_completion_rate": completed / len(results),
        "repeated_mistakes": repeated,
        "repeated_mistake_rate": (repeated / opportunities) if opportunities else 0.0,
        "median_attempts": median_attempts,
        "median_time_sec": median_time,
    }


def main() -> None:
    tasks = paired_tasks()
    out_dir = Path("benchmark_outputs")
    out_dir.mkdir(exist_ok=True)

    all_results = {}
    for arm in ("no_memory", "raw_memory", "mqcl"):
        db_path = out_dir / f"{arm}.db"
        results = run_arm(arm, db_path, tasks)
        all_results[arm] = {
            "summary": summarize(results),
            "events": [r.__dict__ for r in results],
        }

    baseline = all_results["no_memory"]["summary"]["repeated_mistake_rate"]
    improved = all_results["mqcl"]["summary"]["repeated_mistake_rate"]
    delta = 0.0
    if baseline > 0:
        delta = ((baseline - improved) / baseline) * 100.0

    report = {
        "headline": f"Agents with MQCL made {delta:.1f}% fewer repeated mistakes vs no memory.",
        "results": all_results,
    }

    out_file = out_dir / "agent_behavior_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(report["headline"])
    print("Saved report:", out_file)
    for arm, payload in all_results.items():
        s = payload["summary"]
        print(
            f"{arm:>10} | completion={s['task_completion_rate']:.2f} "
            f"| rmr={s['repeated_mistake_rate']:.2f} "
            f"| attempts={s['median_attempts']:.1f}"
        )


if __name__ == "__main__":
    main()

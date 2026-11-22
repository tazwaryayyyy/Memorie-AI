"""Run agent-visible and hidden tests for a candidate solution.

This is process isolation, NOT a security sandbox: candidate code runs as the
current OS user and can touch the filesystem and network. The harness only
guarantees a scrubbed environment (no API keys), a throw-away working
directory, no bytecode writes and a hard timeout.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

from .tasks.base import Task

TIMEOUT_S = 10
MAX_OUTPUT_CHARS = 4000

# Variables copied from the parent environment. Everything else (notably every
# *_API_KEY / *_TOKEN / *_SECRET) is dropped. Windows needs SYSTEMROOT to start Python.
_PASSTHROUGH = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT")


def scrubbed_env(workdir: str) -> Dict[str, str]:
    env = {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONHASHSEED": "0",
            "HOME": workdir,
            "USERPROFILE": workdir,
            "TMPDIR": workdir,
            "TEMP": workdir,
            "TMP": workdir,
        }
    )
    return env


@dataclass
class TestOutcome:
    # not a pytest test class
    __test__ = False
    exit_code: Optional[int]
    timed_out: bool
    output: str

    @property
    def ok(self) -> bool:
        return (not self.timed_out) and self.exit_code == 0


@dataclass
class GradeResult:
    visible: TestOutcome
    hidden: TestOutcome

    @property
    def passed(self) -> bool:
        return self.visible.ok and self.hidden.ok

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "visible": asdict(self.visible),
            "hidden": asdict(self.hidden),
        }


def _tail(text: str) -> str:
    return text if len(text) <= MAX_OUTPUT_CHARS else "...[truncated]...\n" + text[-MAX_OUTPUT_CHARS:]


def run_unittest(workdir: str, module: str, timeout: int = TIMEOUT_S) -> TestOutcome:
    cmd = [sys.executable, "-B", "-s", "-X", "utf8", "-m", "unittest", "-q", module]
    try:
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            env=scrubbed_env(workdir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return TestOutcome(exit_code=None, timed_out=True, output=_tail(partial))
    return TestOutcome(exit_code=proc.returncode, timed_out=False, output=_tail(proc.stdout or ""))


def grade(task: Task, solution_code: str, timeout: int = TIMEOUT_S) -> GradeResult:
    """Write the micro-repo to a temp dir and run visible, then hidden tests."""
    with tempfile.TemporaryDirectory(prefix="memoire_eval_", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        (root / "solution.py").write_text(solution_code, encoding="utf-8")
        (root / "test_visible.py").write_text(task.visible_tests, encoding="utf-8")
        (root / "test_hidden.py").write_text(task.hidden_tests, encoding="utf-8")
        visible = run_unittest(tmp, "test_visible", timeout)
        hidden = run_unittest(tmp, "test_hidden", timeout)
    return GradeResult(visible=visible, hidden=hidden)

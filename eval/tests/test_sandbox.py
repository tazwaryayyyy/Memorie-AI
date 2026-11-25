import os
import tempfile
from pathlib import Path

from eval.sandbox import run_unittest, scrubbed_env
from eval.tasks.base import Task


def _task(hidden: str) -> Task:
    visible = "import unittest\nclass V(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n"
    return Task("f", 1, "dev", "spec", "", visible, hidden, ("f",), "", "")


def _run(files: dict, module: str, timeout=10):
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in files.items():
            Path(tmp, name).write_text(body, encoding="utf-8")
        outcome = run_unittest(tmp, module, timeout)
        return outcome, sorted(p.name for p in Path(tmp).iterdir())


def test_secrets_are_scrubbed_from_the_child_environment(monkeypatch):
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MY_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "super-secret")
    env = scrubbed_env("/tmp/x")
    assert not [k for k in env if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET"))]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"

    probe = (
        "import os, unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_env(self):\n"
        "        bad = [k for k in os.environ if any(s in k.upper() for s in ('KEY','TOKEN','SECRET'))]\n"
        "        self.assertEqual(bad, [])\n"
    )
    outcome, _ = _run({"test_probe.py": probe}, "test_probe")
    assert outcome.ok, outcome.output


def test_timeout_is_reported_not_raised():
    spin = "import unittest\nclass T(unittest.TestCase):\n    def test_spin(self):\n        while True:\n            pass\n"
    outcome, _ = _run({"test_spin.py": spin}, "test_spin", timeout=1)
    assert outcome.timed_out and outcome.exit_code is None and not outcome.ok


def test_no_bytecode_is_written():
    mod = "VALUE = 1\n"
    t = "import unittest, mod\nclass T(unittest.TestCase):\n    def test_v(self):\n        self.assertEqual(mod.VALUE, 1)\n"
    outcome, listing = _run({"mod.py": mod, "test_mod.py": t}, "test_mod")
    assert outcome.ok, outcome.output
    assert "__pycache__" not in listing


def test_failing_hidden_test_gives_nonzero_exit_and_output():
    outcome, _ = _run({"test_bad.py": "import unittest\nclass T(unittest.TestCase):\n    def test_x(self):\n        self.assertEqual(1, 2)\n"}, "test_bad")
    assert not outcome.ok and outcome.exit_code not in (0, None) and "AssertionError" in outcome.output

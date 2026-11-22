"""Task-suite contract, executed for real in subprocesses (no network, no native module)."""
import re

import pytest

from eval.sandbox import grade
from eval.tasks import load_split

DEV = load_split("dev")
TEST = load_split("test")
ALL_TASKS = [t for f in DEV + TEST for t in f.tasks]
ALL_FAMILIES = DEV + TEST

# Words tied to the five remaining prototype exemplars (payment, bcrypt, parameterized SQL,
# input validation, closing handles) and the consequence exemplars.
BANNED = ("bcrypt", "password", "parameterized", "payment", "sanitiz", "database", "sql", "file handle",
          "breach", "outage", "validated", "authentication")


def test_split_shape():
    assert len(DEV) == 5 and len(TEST) >= 8
    for fam in ALL_FAMILIES:
        assert [t.session for t in fam.tasks] == [1, 2, 3]
        assert {t.split for t in fam.tasks} == {fam.split}
    assert not {f.name for f in DEV} & {f.name for f in TEST}


def test_lessons_are_well_formed_and_do_not_echo_prototype_topics():
    keys = set()
    for fam in ALL_FAMILIES:
        assert [l.kind for l in fam.lessons] == ["gold", "vague", "wrong", "poison"]
        for lesson in fam.lessons:
            assert lesson.key not in keys
            keys.add(lesson.key)
            assert "@@" not in lesson.text and "\n" not in lesson.text
            assert len(lesson.text.split()) <= 70, lesson.key  # must stay a single ingest chunk
            low = lesson.text.lower()
            assert not [w for w in BANNED if w in low], (lesson.key, [w for w in BANNED if w in low])
            assert not re.match(r"(always|never|rule:|mandatory:)", low), lesson.key
    for fam in ALL_FAMILIES:
        assert fam.poison.text != fam.lessons[0].text


def test_poison_names_the_session_two_entrypoint():
    for fam in ALL_FAMILIES:
        s2 = fam.tasks[fam.poison_session - 1]
        assert s2.entrypoints[0] in fam.poison.text, fam.name


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.task_id)
def test_reference_passes_naive_passes_visible_only_stub_fails(task):
    ref = grade(task, task.reference)
    assert ref.visible.ok, ref.visible.output
    assert ref.hidden.ok, ref.hidden.output
    naive = grade(task, task.naive)
    assert naive.visible.ok, f"naive must pass the visible tests:\n{naive.visible.output}"
    assert not naive.hidden.ok and not naive.hidden.timed_out, "naive must fail hidden tests by assertion"
    stub = grade(task, task.solution_stub)
    assert not stub.visible.ok and not stub.hidden.ok


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.task_id)
def test_entrypoints_are_defined_in_stub_and_tests_do_not_leak(task):
    for name in task.entrypoints:
        assert re.search(rf"def {name}\(", task.solution_stub)
    assert "@@" not in task.spec + task.solution_stub + task.visible_tests + task.hidden_tests
    # hidden test class must not be recoverable from agent-visible material
    assert "Hidden" not in task.spec + task.solution_stub + task.visible_tests

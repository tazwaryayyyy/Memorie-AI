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
          "breach", "outage", "validat", "authentication", "credential", "encrypt", "leak", "production",
          "data loss", "vulnerab")


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


# ---- Step 8c-2: convention-family fairness guards ---------------------------------------------

# Tokens that carry the convention. They must be in the gold lesson, must NOT be in the wrong
# lesson, and must NOT be recoverable from anything the agent sees without memory.
CONVENTION = {
    "wire_timestamp": (["Z suffix", "three fractional digits"], ["+00:00"]),
    "lock_ordering": (["modulo 16"], []),
    "fee_rounding": (["up to the next cent"], ["half-up"]),
    "cache_key_format": (["kv3:", "%25", "%3A"], ["cache:v1"]),
    "http_status": (["422", "503"], ["504"]),
    "money_wire": (["amount_minor"], ["amount as a string"]),
    "idempotency_key": (["idem_", "32 hex"], ["full hex digest"]),
    "retry_backoff": (["half a second", "30 seconds"], ["60 seconds"]),
    "cursor_ties": (["id ascending"], ["both descending"]),
    "semver_bump": (["major version is 0"], ["always increments the major"]),
    "slug_rules": (["40 characters", "untitled"], ["no length limit"]),
    "event_tiebreak": (["system first"], ["alphabetically"]),
    "histogram_bucket": (["upper-inclusive"], ["lower-inclusive"]),
}
LEAK_TOKENS = {
    "wire_timestamp": ["%f", ".000", "[:3]", "// 1000"],
    "lock_ordering": ["% 16", "%16", "shard"],
    "fee_rounding": ["CEILING", "ROUND_UP", "per line,", "ceil"],
    "cache_key_format": ["kv3", "%25", "%3A"],
    "http_status": ["422", "503"],
    "money_wire": ["amount_minor", "minor unit"],
    "idempotency_key": ["idem_", "sha256", "[:32]"],
    "retry_backoff": ["0.5", "** ", "30.0"],
    "cursor_ties": ["ascending", "id asc"],
    "semver_bump": ["0.5.0", "pre-1.0 policy"],
    "slug_rules": ["untitled", "[:40]", "40"],
    "event_tiebreak": ["priority", "system first"],
    "histogram_bucket": ["bisect", "inclusive"],
}
ARM_A_VISIBLE = lambda t: "\n".join((t.spec, t.solution_stub, t.visible_tests))


def test_every_family_has_convention_metadata():
    assert {f.name for f in ALL_FAMILIES} == set(CONVENTION) == set(LEAK_TOKENS)


def test_every_spec_declares_an_undocumented_convention_without_stating_it():
    for task in ALL_TASKS:
        flat = " ".join(task.spec.split())
        assert "Convention note:" in flat and "not documented in this" in flat, task.task_id


def test_gold_states_the_convention_and_wrong_states_the_alternative():
    for fam in ALL_FAMILIES:
        gold = fam.lessons[0].text
        wrong = fam.lessons[2].text
        must, must_not = CONVENTION[fam.name]
        assert all(tok in gold for tok in must), (fam.name, [t for t in must if t not in gold])
        assert not any(tok in wrong for tok in must), fam.name
        assert all(tok in wrong or tok in fam.poison.text for tok in must_not), (fam.name, must_not)
        assert not any(tok in gold for tok in must_not), fam.name


def test_the_convention_cannot_be_read_off_agent_visible_material():
    for task in ALL_TASKS:
        text = ARM_A_VISIBLE(task)
        leaked = [t for t in LEAK_TOKENS[task.family] if t in text]
        assert not leaked, (task.task_id, leaked)


def test_visible_tests_do_not_discriminate_the_reference_from_the_naive_solution():
    # Same as the contract test, stated as the fairness property: both pass visible, only one hidden.
    for fam in ALL_FAMILIES:
        task = fam.tasks[0]
        assert grade(task, task.reference).visible.ok and grade(task, task.naive).visible.ok

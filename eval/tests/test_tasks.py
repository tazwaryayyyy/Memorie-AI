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
        assert [t.session for t in fam.tasks] == [1, 2, 3, 4]
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


# ---- Step 8c-3: single-choice convention families -------------------------------------------

from eval.tasks.design import DESIGN  # noqa: E402

# Tokens that carry the gold choice. They must be in the gold lesson, must NOT be in the wrong
# lesson, and must NOT be recoverable from anything the agent sees without memory. `must_not`
# tokens belong to the alternative the wrong lesson states.
CONVENTION = {
    "wire_timestamp": (["literal Z"], ["+00:00"]),
    "lock_ordering": (["lower account id first"], ["higher account id first"]),
    "fee_rounding": (["nearest even cent"], ["ROUND_HALF_UP"]),
    "cache_key_format": (["only the tenant is lowercased"], ["every part lowercased"]),
    "http_status": (["422"], ["400"]),
    "money_wire": (["integer count of minor units"], ["decimal strings in the major unit"]),
    "percentile_method": (["nearest-rank"], ["linear interpolation"]),
    "retry_backoff": (["doubles the wait"], ["grows the wait linearly"]),
    "cursor_ties": (["id ascending"], ["both descending"]),
    "semver_bump": (["minor number for a breaking change"], ["next major number"]),
    "slug_separator": (["single underscore"], ["single hyphen"]),
    "event_tiebreak": (["order they arrived"], ["by seq ascending"]),
    "histogram_bucket": (["lower-inclusive"], ["upper-inclusive"]),
}
LEAK_TOKENS = {
    "wire_timestamp": ["literal Z", "ends with Z"],
    "lock_ordering": ["sorted(", "lower account id", "ascending", "descending"],
    "fee_rounding": ["HALF_EVEN", "HALF_UP", "banker", "nearest even"],
    "cache_key_format": ["only the tenant", "tenant.lower"],
    "http_status": ["422"],
    "money_wire": ["minor unit", "cents"],
    "percentile_method": ["nearest", "ceil", "interpol"],
    "retry_backoff": ["doubl", "2.0 **", "** "],
    "cursor_ties": ["ascending", "id asc"],
    "semver_bump": ["0.5.0", "minor number"],
    "slug_separator": ["underscore", "creme_brulee"],
    "event_tiebreak": ["arrival", "stable"],
    "histogram_bucket": ["bisect", "lower-inclusive", "upper-inclusive"],
}
ARM_A_VISIBLE = lambda t: "\n".join((t.spec, t.solution_stub, t.visible_tests))


def test_every_family_has_convention_and_design_metadata():
    assert {f.name for f in ALL_FAMILIES} == set(CONVENTION) == set(LEAK_TOKENS) == set(DESIGN)
    for fam in ALL_FAMILIES:
        d = DESIGN[fam.name]
        assert d["split"] == fam.split and d["gold"] in d["options"] and 2 <= len(d["options"]) <= 3, fam.name


def test_test_split_gold_choices_follow_the_preregistered_parity_rule():
    for index, fam in enumerate(TEST, start=1):
        d = DESIGN[fam.name]
        assert d["position"] == index, fam.name
        assert d["gold_is"] == ("common" if index % 2 == 1 else "less-common"), fam.name


def test_every_spec_declares_an_undocumented_convention_without_stating_it():
    for task in ALL_TASKS:
        flat = " ".join(task.spec.split())
        assert "Convention note:" in flat and "not documented in this" in flat, task.task_id


def test_gold_states_the_choice_and_wrong_states_the_alternative():
    for fam in ALL_FAMILIES:
        gold, wrong = fam.lessons[0].text, fam.lessons[2].text
        must, must_not = CONVENTION[fam.name]
        assert all(tok in gold for tok in must), (fam.name, [t for t in must if t not in gold])
        assert not any(tok in wrong for tok in must), fam.name
        assert all(tok in wrong for tok in must_not), (fam.name, must_not)
        assert not any(tok in gold for tok in must_not), fam.name


def test_the_choice_cannot_be_read_off_agent_visible_material():
    for task in ALL_TASKS:
        text = ARM_A_VISIBLE(task)
        leaked = [t for t in LEAK_TOKENS[task.family] if t in text]
        assert not leaked, (task.task_id, leaked)


@pytest.mark.parametrize("task", [t for t in ALL_TASKS if t.alternates], ids=lambda t: t.task_id)
def test_every_other_legitimate_option_passes_visible_and_fails_hidden(task):
    for code in task.alternates:
        result = grade(task, code)
        assert result.visible.ok, f"visible tests must be neutral across options:\n{result.visible.output}"
        assert not result.hidden.ok and not result.hidden.timed_out


def test_three_option_families_cover_every_option_with_naive_plus_alternates():
    three = [n for n, d in DESIGN.items() if len(d["options"]) == 3]
    assert set(three) == {"lock_ordering", "cache_key_format"}
    for name in three:
        fam = next(f for f in ALL_FAMILIES if f.name == name)
        assert all(t.alternates for t in fam.tasks), name

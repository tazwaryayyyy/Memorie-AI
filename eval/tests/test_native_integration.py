"""Runs only where the native extension is importable (`maturin develop --release`).
Pins the engine behaviours the harness relies on. Needs the embedder model cached/downloadable."""
import pytest

pytest.importorskip("memoire_engine.memoire_engine", reason="native extension not built")

from eval.lesson_store import LessonStore, make_memoire  # noqa: E402
from eval.tasks import load_split  # noqa: E402

FAM = load_split("dev")[0]


@pytest.fixture()
def store():
    s = LessonStore(make_memoire(contradiction="off", trial_cap=1))
    for lesson in FAM.bank_lessons:
        s.ingest(lesson)
    return s


def test_lessons_ingest_as_single_chunks(store):
    assert store.count() == 3


def test_plain_recall_returns_k_with_plain_exposure_and_changes_nothing(store):
    query = FAM.tasks[0].spec
    first = store.recall_plain(query, 3)
    second = store.recall_plain(query, 3)
    assert len(first) == 3 and {r.exposure for r in first} == {"plain"}
    assert [(r.key, r.score) for r in first] == [(r.key, r.score) for r in second]


def test_default_recall_is_trial_capped_to_one_unverified_row(store):
    got = store.recall(FAM.tasks[0].spec, 3)
    assert len(got) == 1 and got[0].exposure == "trial"


def test_reinforce_one_vote_per_session_and_penalize_lowers_trust(store):
    key = FAM.bank_lessons[0].key
    assert store.reinforce(key, "sess-1") is True
    assert store.reinforce(key, "sess-1") is False
    out = store.penalize([FAM.bank_lessons[1].key])
    assert out[0]["trust_after"] < out[0]["trust_before"]


# ---- step 8c-3: facts the poison/feedback timelines (test_poison_timeline.py) rely on -----------

def test_quality_scores_are_readable_through_export_and_independent_of_ingest_order():
    from eval.diagnostics import quality_scores

    texts = {l.key: l.text for l in FAM.lessons}
    forward = quality_scores(lambda: make_memoire(contradiction="off", trial_cap=1), texts)
    backward = quality_scores(lambda: make_memoire(contradiction="off", trial_cap=1), dict(reversed(list(texts.items()))))
    assert forward == backward
    assert all(v is not None and 0.0 <= v["importance_base"] <= 1.0 for v in forward.values())


def test_poison_needs_two_failures_to_be_hidden_and_keeps_a_slot_after_the_first():
    store = LessonStore(make_memoire(contradiction="off", trial_cap=1))
    for lesson in FAM.bank_lessons:
        store.ingest(lesson)
    store.ingest(FAM.poison)
    query, key = FAM.tasks[1].spec, FAM.poison.key

    first = store.penalize([key])
    assert first[0]["trust_after"] < first[0]["trust_before"]
    got = store.recall(query, 3)
    assert key in [r.key for r in got]                               # verified rows bypass trial_cap
    assert [r.exposure for r in got if r.key == key] == ["hint"]     # one failure: not quarantined

    store.penalize([key])                                            # second failure -> 2f > 3s + 2
    assert key not in [r.key for r in store.recall(query, 3)]


def test_verified_rows_have_no_similarity_floor():
    other = load_split("dev")[1]
    store = LessonStore(make_memoire(contradiction="off", trial_cap=1))
    for lesson in FAM.bank_lessons + other.bank_lessons:
        store.ingest(lesson)
    unrelated = other.bank_lessons[0].key
    assert store.reinforce(unrelated, "sess-x") is True
    got = store.recall(FAM.tasks[0].spec, 3)
    assert unrelated in [r.key for r in got]


# ---- step 8c-4: relevance floors in the real engine -----------------------------------------------

def _two_family_store(**opts):
    other = load_split("dev")[1]
    store = LessonStore(make_memoire(contradiction="off", trial_cap=4, **opts))
    for lesson in FAM.bank_lessons + other.bank_lessons:
        store.ingest(lesson)
    store.reinforce(other.bank_lessons[0].key, "sess-x")       # an unrelated row becomes verified
    return store, other.bank_lessons[0].key


def test_verified_floor_drops_an_unrelated_verified_row_in_the_real_engine():
    query = FAM.tasks[0].spec
    store, unrelated = _two_family_store()
    assert unrelated in [r.key for r in store.recall(query, 6)]          # default: no floor
    store, unrelated = _two_family_store(verified_min_cos=0.99)
    assert unrelated not in [r.key for r in store.recall(query, 6)]


def test_trial_floor_empties_the_trial_pool_but_not_verified_rows():
    query = FAM.tasks[0].spec
    store, unrelated = _two_family_store(trial_min_cos=0.99)
    got = store.recall(query, 6)
    assert [r.key for r in got] == [unrelated] and all(r.exposure != "trial" for r in got)


def test_quality_gate_mode_reports_hidden_lessons_through_the_store():
    store = LessonStore(make_memoire(contradiction="off", trial_cap=4, admission="quality_gate",
                                     quality_gate_threshold=1.0))
    for lesson in FAM.bank_lessons:
        store.ingest(lesson, track_shadow=True)
    assert store.shadowed == {l.key for l in FAM.bank_lessons}
    assert store.recall(FAM.tasks[0].spec, 3) == []

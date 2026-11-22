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

import json

import pytest

from eval.arms import ARM_A, ARM_C1
from eval.runner import BudgetExhausted, FingerprintMismatch, ResultsFile, RunSettings, run_sequence, select_families
from eval.tasks import load_split
from eval.tests.fakes import FakeMemoire, ScriptedLLM
from eval.tests.test_arms_and_runner import run, settings

DEV = load_split("dev")
FAMS = select_families(DEV, ["wire_timestamp", "fee_rounding"])
FAIL_FIRST = lambda t: "naive" if t.session == 1 else "ref"


def test_resume_skips_completed_runs_and_never_duplicates(tmp_path):
    path = str(tmp_path / "r.jsonl")
    st = settings(FAMS)
    tasks = [t for f in FAMS for t in f.tasks]
    llm1 = ScriptedLLM(tasks)
    results = ResultsFile(path, st.fingerprint(FAMS))
    with pytest.raises(BudgetExhausted):
        run_sequence(settings=st, arm=ARM_A, seed=0, families=FAMS, llm=llm1, results=results,
                     store_factory=FakeMemoire, max_calls=4, log=lambda s: None)
    assert llm1.total_requests == 4 and len(open(path).read().splitlines()) == 4

    llm2 = ScriptedLLM(tasks)
    stats = run_sequence(settings=st, arm=ARM_A, seed=0, families=FAMS, llm=llm2,
                         results=ResultsFile(path, st.fingerprint(FAMS)), store_factory=FakeMemoire,
                         log=lambda s: None)
    assert stats["skipped"] == 4 and stats["ran"] == 2 and llm2.total_requests == 2
    keys = [(r["arm"], r["family"], r["session"], r["seed"]) for r in map(json.loads, open(path))]
    assert len(keys) == len(set(keys)) == 6

    llm3 = ScriptedLLM(tasks)
    stats = run_sequence(settings=st, arm=ARM_A, seed=0, families=FAMS, llm=llm3,
                         results=ResultsFile(path, st.fingerprint(FAMS)), store_factory=FakeMemoire,
                         log=lambda s: None)
    assert stats["ran"] == 0 and llm3.total_requests == 0


def test_c1_resume_replays_feedback_so_final_store_matches_an_uninterrupted_run(tmp_path):
    declare = lambda n: [1, 2]
    _, uninterrupted, _, _ = run(ARM_C1, tmp_path, families=FAMS, choose=FAIL_FIRST, declare=declare, name="u.jsonl")

    path = str(tmp_path / "i.jsonl")
    st = settings(FAMS)
    tasks = [t for f in FAMS for t in f.tasks]
    first = FakeMemoire()
    with pytest.raises(BudgetExhausted):
        run_sequence(settings=st, arm=ARM_C1, seed=0, families=FAMS, llm=ScriptedLLM(tasks, FAIL_FIRST, declare),
                     results=ResultsFile(path, st.fingerprint(FAMS)), store_factory=lambda: first, max_calls=4,
                     log=lambda s: None)
    resumed = FakeMemoire()  # a new process: empty store, rebuilt from the log
    llm = ScriptedLLM(tasks, FAIL_FIRST, declare)
    run_sequence(settings=st, arm=ARM_C1, seed=0, families=FAMS, llm=llm,
                 results=ResultsFile(path, st.fingerprint(FAMS)), store_factory=lambda: resumed,
                 log=lambda s: None)
    assert llm.total_requests == 2
    assert resumed.state() == uninterrupted.state()
    assert resumed.rows == uninterrupted.rows


def test_torn_trailing_line_is_dropped_and_file_is_realigned(tmp_path):
    path = tmp_path / "t.jsonl"
    st = settings(FAMS)
    _, _, _, results = run(ARM_A, tmp_path, families=FAMS[:1], name="t.jsonl")
    good = path.read_text().splitlines()
    assert len(good) == 3
    path.write_text("\n".join(good) + "\n" + good[0][: len(good[0]) // 2])  # crash mid-write
    reloaded = ResultsFile(str(path), results.fingerprint)
    assert len(reloaded.ordered) == 3 and path.read_text().endswith("\n") and len(path.read_text().splitlines()) == 3
    reloaded.append({**good_record(good[0]), "seed": 99})
    assert len(ResultsFile(str(path), results.fingerprint).ordered) == 4


def good_record(line):
    rec = json.loads(line)
    rec.pop("fingerprint", None)
    return rec


def test_complete_json_without_newline_is_treated_as_torn(tmp_path):
    _, _, _, results = run(ARM_A, tmp_path, families=FAMS[:1], name="n.jsonl")
    path = tmp_path / "n.jsonl"
    path.write_text(path.read_text().rstrip("\n"))
    assert len(ResultsFile(str(path), results.fingerprint).ordered) == 2


def test_corruption_in_the_middle_is_an_error(tmp_path):
    _, _, _, results = run(ARM_A, tmp_path, families=FAMS[:1], name="m.jsonl")
    path = tmp_path / "m.jsonl"
    lines = path.read_text().splitlines()
    lines[1] = "{broken"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        ResultsFile(str(path), results.fingerprint)


def test_changed_model_or_tasks_refuse_to_mix_into_one_file(tmp_path):
    _, _, _, results = run(ARM_A, tmp_path, families=FAMS[:1], name="f.jsonl")
    other = settings(FAMS[:1], model="another-model").fingerprint(FAMS[:1])
    with pytest.raises(FingerprintMismatch):
        ResultsFile(str(tmp_path / "f.jsonl"), other)
    assert settings(FAMS[:1], top_k=5).fingerprint(FAMS[:1]) != results.fingerprint
    assert settings(FAMS[:1], trial_cap=3).fingerprint(FAMS[:1]) != results.fingerprint

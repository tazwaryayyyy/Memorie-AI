"""step 8d-1: `--no-poison` (poison-harm counterfactual). No network, no native module."""
import json

import pytest

from eval.arms import ARM_A, ARM_B, ARM_C1
from eval.run_eval import main
from eval.runner import ResultsFile, RunSettings, run_sequence, select_families
from eval.tasks import load_split
from eval.tests.fakes import EngineBackedFake, FakeMemoire, ScriptedLLM
from eval.tests.test_cli import BASE, _capture_client

DEV = load_split("dev")
POISON_BONUS = lambda text: 0.5 if "`" in text else 0.0  # poison texts name the s2 function in backticks


def _settings(fams, **kw):
    base = dict(split="dev", families=tuple(f.name for f in fams), sessions=(1, 2, 3, 4), model="scripted-1",
                base_url="http://x", temperature=0.0, top_k=3, trial_cap=1, contradiction="off", max_tokens=100,
                store_scope="family")
    base.update(kw)
    return RunSettings(**base)


def _episode(tmp_path, arm, no_poison, name, names=("wire_timestamp", "http_status")):
    fams = select_families(DEV, list(names))
    st = _settings(fams, no_poison=no_poison)
    tasks = [t for f in fams for t in f.tasks]
    llm = ScriptedLLM(tasks, choose=lambda t: "naive", declare=lambda n: list(range(1, n + 1)))
    results = ResultsFile(str(tmp_path / name), st.fingerprint(fams))
    stores = []

    def factory():
        stores.append(EngineBackedFake(bonus=POISON_BONUS))
        return stores[-1]

    run_sequence(settings=st, arm=arm, seed=0, families=fams, llm=llm, results=results, store_factory=factory,
                 log=lambda s: None)
    return fams, stores, results, llm


def _ingested(stores):
    return [text for s in stores for text in s.rows.values()]


# ---- fingerprint -------------------------------------------------------------------------------------


def test_default_no_poison_stays_out_of_the_fingerprint_settings_so_old_files_resume():
    fams = select_families(DEV, ["wire_timestamp"])
    default, flagged = _settings(fams), _settings(fams, no_poison=True)
    assert "no_poison" not in default.fingerprint_settings()
    assert flagged.fingerprint_settings()["no_poison"] is True
    assert default.fingerprint(fams) != flagged.fingerprint(fams)
    assert default.fingerprint(fams) == _settings(fams, no_poison=False).fingerprint(fams)


# ---- run loop ----------------------------------------------------------------------------------------


def test_no_poison_never_ingests_the_poison_lesson_and_control_does(tmp_path):
    fams, stores, results, _ = _episode(tmp_path, ARM_B, True, "np.jsonl")
    poison_texts = {f.poison.text for f in fams}
    assert not poison_texts & set(_ingested(stores))
    assert all(r["no_poison"] is True for r in results.ordered)
    assert not any(x["kind"] == "poison" for r in results.ordered for x in r["retrieved"])

    fams, stores, control, _ = _episode(tmp_path, ARM_B, False, "ctl.jsonl")
    assert poison_texts <= set(_ingested(stores))
    assert all(r["no_poison"] is False for r in control.ordered)
    assert any(x["kind"] == "poison" for r in control.ordered if r["session"] >= 2 for x in r["retrieved"])


def test_session_one_prompts_are_identical_with_and_without_poison(tmp_path):
    _, _, np_res, _ = _episode(tmp_path, ARM_B, True, "np.jsonl")
    _, _, ctl_res, _ = _episode(tmp_path, ARM_B, False, "ctl.jsonl")
    s1 = lambda res: {r["family"]: r["prompt"] for r in res.ordered if r["session"] == 1}
    assert s1(np_res) == s1(ctl_res) and s1(np_res)
    # from poison_session on the stores differ, so at least one prompt must differ
    later = lambda res: {(r["family"], r["session"]): r["prompt"] for r in res.ordered if r["session"] >= 2}
    assert later(np_res) != later(ctl_res)


def test_no_poison_runs_resume_without_error_for_the_feedback_arm(tmp_path):
    fams, _, results, llm = _episode(tmp_path, ARM_C1, True, "c1.jsonl")
    first = len(results.ordered)
    st = _settings(fams, no_poison=True)
    again = ResultsFile(str(tmp_path / "c1.jsonl"), st.fingerprint(fams))
    stats = run_sequence(settings=st, arm=ARM_C1, seed=0, families=fams, llm=llm, results=again,
                         store_factory=lambda: EngineBackedFake(bonus=POISON_BONUS), log=lambda s: None)
    assert stats["ran"] == 0 and stats["skipped"] == first == len(again.ordered)


def test_a_file_written_with_poison_cannot_be_resumed_with_no_poison(tmp_path):
    fams, _, _, _ = _episode(tmp_path, ARM_B, False, "mix.jsonl")
    with pytest.raises(Exception) as exc:
        ResultsFile(str(tmp_path / "mix.jsonl"), _settings(fams, no_poison=True).fingerprint(fams))
    assert "different settings" in str(exc.value)


# ---- CLI ---------------------------------------------------------------------------------------------


def test_cli_no_poison_flag_is_recorded_and_rejected_for_arm_a(tmp_path, monkeypatch, capsys):
    _capture_client(monkeypatch)
    monkeypatch.setattr("eval.lesson_store.make_memoire", lambda **kw: FakeMemoire())
    out = str(tmp_path / "np.jsonl")
    assert main(BASE + [out, "--arms", "B", "--families", "wire_timestamp", "--no-poison"]) == 0
    recs = [json.loads(line) for line in open(out)]
    assert recs and {r["no_poison"] for r in recs} == {True}

    out2 = str(tmp_path / "default.jsonl")
    assert main(BASE + [out2, "--arms", "B", "--families", "wire_timestamp"]) == 0
    assert {json.loads(line)["no_poison"] for line in open(out2)} == {False}

    capsys.readouterr()
    assert main(BASE + [str(tmp_path / "a.jsonl"), "--arms", "A,B", "--no-poison"]) == 2
    assert "Arm A has none" in capsys.readouterr().err

    assert main(["summarize", "--results", out]) == 0
    assert "[no-poison]" in capsys.readouterr().out

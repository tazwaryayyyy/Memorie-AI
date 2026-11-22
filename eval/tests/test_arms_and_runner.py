import pytest

from eval.arms import ARM_A, ARM_B, ARM_C0, ARM_C1, normalize_arm
from eval.runner import ResultsFile, RunSettings, run_sequence, select_families
from eval.tasks import load_split
from eval.tests.fakes import FakeMemoire, ScriptedLLM

DEV = load_split("dev")


def settings(families, sessions=(1, 2, 3), **kw):
    base = dict(split="dev", families=tuple(f.name for f in families), sessions=sessions, model="scripted-1",
                base_url="http://x", temperature=0.0, top_k=3, trial_cap=1, contradiction="off", max_tokens=100)
    base.update(kw)
    return RunSettings(**base)


def run(arm, tmp_path, families=None, choose=lambda t: "ref", declare=lambda n: [1] if n else [], seed=0,
        fake=None, events=None, name="r.jsonl", max_calls=0, results=None, llm=None):
    families = families or select_families(DEV, ["loop_closures"])
    st = settings(families)
    fake = fake or FakeMemoire(events)
    tasks = [t for f in families for t in f.tasks]
    llm = llm or ScriptedLLM(tasks, choose, declare, events)
    results = results or ResultsFile(str(tmp_path / name), st.fingerprint(families))
    stats = run_sequence(settings=st, arm=arm, seed=seed, families=families, llm=llm, results=results,
                         store_factory=lambda: fake, max_calls=max_calls, log=lambda s: None)
    return stats, fake, llm, results


def test_arm_aliases():
    assert [normalize_arm(a) for a in ("A", "b", "C0", "c1", "plain_vector")] == [ARM_A, ARM_B, ARM_C0, ARM_C1, ARM_B]
    with pytest.raises(ValueError):
        normalize_arm("D")


def test_arm_a_touches_no_memory_and_prompt_has_no_notes(tmp_path):
    def boom():
        raise AssertionError("Arm A must not construct a memory store")

    fams = select_families(DEV, ["loop_closures"])
    st = settings(fams)
    tasks = [t for f in fams for t in f.tasks]
    llm = ScriptedLLM(tasks)
    results = ResultsFile(str(tmp_path / "a.jsonl"), st.fingerprint(fams))
    run_sequence(settings=st, arm=ARM_A, seed=0, families=fams, llm=llm, results=results,
                 store_factory=boom, log=lambda s: None)
    assert all("USED_MEMORY" not in p[-1]["content"] for p in llm.prompts)
    rec = results.ordered[0]
    assert rec["retrieved"] == [] and rec["parsed"]["used_parse_ok"] is False and rec["passed"] is True


def test_arm_b_is_plain_cosine_and_side_effect_free(tmp_path):
    _, fake, _, results = run(ARM_B, tmp_path, choose=lambda t: "naive")
    names = {c[0] for c in fake.calls}
    assert names <= {"remember_ids", "recall_plain"}, names
    assert fake.writes_after_ingest() == []
    assert fake.state() == ({i: 0 for i in fake.rows}, {i: 0 for i in fake.rows})
    assert all(r["retrieved"] and all(x["exposure"] == "plain" for x in r["retrieved"]) for r in results.ordered)


def test_arm_c0_uses_recall_and_never_writes_feedback_even_when_notes_are_declared(tmp_path):
    _, fake, _, results = run(ARM_C0, tmp_path, choose=lambda t: "naive", declare=lambda n: [1, 2])
    names = {c[0] for c in fake.calls}
    assert names <= {"remember_ids", "recall"}, names
    assert all(r["feedback_events"] == [] for r in results.ordered)
    assert results.ordered[0]["used_keys"]  # declarations are still recorded


def test_arm_c1_reinforces_declared_on_pass_and_penalizes_declared_on_fail(tmp_path):
    choose = lambda t: "naive" if t.session == 1 else "ref"
    _, fake, _, results = run(ARM_C1, tmp_path, choose=choose, declare=lambda n: [1, 3])
    s1, s2, s3 = results.ordered
    assert s1["passed"] is False and s2["passed"] and s3["passed"]
    pen = [c for c in fake.calls if c[0] == "penalize_if_used"]
    rei = [c for c in fake.calls if c[0] == "reinforce_declared"]
    assert len(pen) == 1 and len(pen[0][1][0]) == 2 and pen[0][1][1] == 1.0
    assert len(rei) == 4  # two declared notes x two passing sessions
    assert {c[1][1] for c in rei} == {"loop_closures-s2-seed0", "loop_closures-s3-seed0"}
    assert s1["feedback_events"][0]["op"] == "penalize" and s2["feedback_events"][0]["op"] == "reinforce"
    shown_keys = [r["key"] for r in s1["retrieved"]]
    assert s1["used_keys"] == [shown_keys[0], shown_keys[2]]


def test_c1_does_nothing_when_nothing_declared_or_index_invalid_or_no_code(tmp_path):
    _, fake, _, _ = run(ARM_C1, tmp_path, choose=lambda t: "naive", declare=lambda n: [], name="a.jsonl")
    assert fake.writes_after_ingest() == []
    _, fake, _, _ = run(ARM_C1, tmp_path, choose=lambda t: "naive", declare=lambda n: [9], name="b.jsonl")
    assert fake.writes_after_ingest() == []
    _, fake, _, results = run(ARM_C1, tmp_path, choose=lambda t: "nocode", declare=lambda n: [1], name="c.jsonl")
    assert fake.writes_after_ingest() == []
    assert results.ordered[0]["grading"]["reason"] == "no_code_block" and results.ordered[0]["feedback_note"]


def test_missing_used_memory_line_is_recorded_and_causes_no_feedback(tmp_path):
    _, fake, _, results = run(ARM_C1, tmp_path, declare=lambda n: None)
    assert fake.writes_after_ingest() == [] and results.ordered[0]["parsed"]["used_parse_ok"] is False


def test_hidden_test_grading_and_record_fields(tmp_path):
    _, _, _, results = run(ARM_A, tmp_path, choose=lambda t: "naive")
    rec = results.ordered[0]
    assert rec["passed"] is False
    assert rec["grading"]["visible"]["exit_code"] == 0 and rec["grading"]["hidden"]["exit_code"] != 0
    assert "AssertionError" in rec["grading"]["hidden"]["output"]
    for field in ("prompt", "response", "usage", "latency_s", "model", "model_returned", "parsed", "fingerprint",
                  "spec_exemplar_cos", "retrieved", "seed", "temperature"):
        assert field in rec
    assert rec["usage"] == {"prompt_tokens": 100, "completion_tokens": 50}


def test_poison_enters_the_store_immediately_before_session_two_only(tmp_path):
    events = []
    fams = select_families(DEV, ["loop_closures"])
    run(ARM_C0, tmp_path, families=fams, events=events)
    fam = fams[0]
    timeline = []
    for name, args in events:
        if name == "remember_ids":
            timeline.append(("ingest", args[0]))
        elif name == "llm":
            timeline.append(("run", args))
    bank = [("ingest", l.text) for l in fam.bank_lessons]
    assert timeline[:3] == bank
    assert timeline[3:] == [("run", "loop_closures/s1"), ("ingest", fam.poison.text),
                            ("run", "loop_closures/s2"), ("run", "loop_closures/s3")]


def test_each_run_gets_a_fresh_store(tmp_path):
    first = FakeMemoire()
    run(ARM_C1, tmp_path, fake=first, declare=lambda n: [1], name="x.jsonl")
    second = FakeMemoire()
    run(ARM_C1, tmp_path, fake=second, seed=1, declare=lambda n: [1], name="y.jsonl")
    assert second.state() == first.state() and len(second.rows) == len(first.rows)


def test_seed_is_forwarded_and_part_of_the_key(tmp_path):
    fams = select_families(DEV, ["loop_closures"])
    _, _, llm, results = run(ARM_A, tmp_path, families=fams, seed=7)
    assert set(llm.seeds) == {7} and all(r["seed"] == 7 for r in results.ordered)

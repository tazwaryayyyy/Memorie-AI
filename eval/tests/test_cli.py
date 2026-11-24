import json

import pytest

from eval.arms import ARM_A
from eval.run_eval import main
from eval.runner import ResultsFile, run_sequence, select_families
from eval.tasks import load_split
from eval.tests.fakes import FakeMemoire, ScriptedLLM
from eval.tests.test_arms_and_runner import settings

BASE = ["run", "--split", "dev", "--model", "gemini-3.1-flash-lite", "--out"]


def test_dry_run_needs_no_key_and_makes_no_calls(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(BASE + [str(tmp_path / "o.jsonl"), "--dry-run", "--arms", "A"]) == 0
    out = capsys.readouterr().out
    assert "planned calls: 20" in out and not (tmp_path / "o.jsonl").exists()


def test_floating_latest_model_is_rejected(tmp_path, capsys):
    assert main(["run", "--split", "dev", "--model", "gemini-flash-latest", "--out", str(tmp_path / "o"), "--dry-run"]) == 2


def test_test_split_is_locked_without_the_explicit_flag(tmp_path, capsys):
    argv = ["run", "--split", "test", "--model", "m-1", "--out", str(tmp_path / "o"), "--dry-run"]
    assert main(argv) == 2 and "locked" in capsys.readouterr().err
    assert main(argv + ["--allow-test-split"]) == 0


def test_missing_api_key_is_a_clean_error(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(BASE + [str(tmp_path / "o.jsonl")]) == 2 and "GEMINI_API_KEY" in capsys.readouterr().err


def test_unknown_arm_or_family_is_a_clean_error(tmp_path, capsys):
    assert main(BASE + [str(tmp_path / "o"), "--dry-run", "--arms", "Z"]) == 2
    assert main(BASE + [str(tmp_path / "o"), "--dry-run", "--families", "nope"]) == 2


def _fixture(tmp_path, choose):
    fams = select_families(load_split("dev"), None)
    st = settings(fams)
    path = str(tmp_path / "s.jsonl")
    tasks = [t for f in fams for t in f.tasks]
    run_sequence(settings=st, arm=ARM_A, seed=0, families=fams, llm=ScriptedLLM(tasks, choose),
                 results=ResultsFile(path, st.fingerprint(fams)), store_factory=FakeMemoire, log=lambda s: None)
    return path


def test_summarize_flags_ceiling_and_in_range(tmp_path, capsys):
    path = _fixture(tmp_path, lambda t: "ref")
    assert main(["summarize", "--results", path]) == 0
    out = capsys.readouterr().out
    assert "15/15" in out and "OUT OF RANGE" in out and "Harden dev tasks" in out
    # 5 of 15 pass -> in range
    (tmp_path / "second").mkdir()
    path = _fixture(tmp_path / "second", lambda t: "ref" if t.family == "http_status" or (t.session == 3 and t.family in ("lock_ordering", "wire_timestamp")) else "naive")
    main(["summarize", "--results", path])
    assert "IN RANGE" in capsys.readouterr().out


# ---- step 8c-1b: HTTP seed is opt-in; max_tokens default leaves thinking headroom -------------

from eval.run_eval import build_parser  # noqa: E402
from eval.tests.test_arms_and_runner import settings as _settings_for  # noqa: E402


def _capture_client(monkeypatch):
    made = []

    class FakeClient:
        def __init__(self, base_url, model, api_key, **kw):
            self.kw, self.total_requests = kw, 0
            made.append(self)
            self._llm = None

        def complete(self, messages, seed=None):
            fams = select_families(load_split("dev"), ["wire_timestamp"])
            self._llm = self._llm or ScriptedLLM([t for f in fams for t in f.tasks])
            self.total_requests += 1
            return self._llm.complete(messages, seed)

    monkeypatch.setattr("eval.run_eval.OpenAICompatClient", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    return made


def test_cli_defaults_do_not_send_api_seed_and_keep_large_max_tokens(tmp_path, monkeypatch):
    made = _capture_client(monkeypatch)
    out = str(tmp_path / "d.jsonl")
    assert main(BASE + [out, "--families", "wire_timestamp", "--seeds", "5"]) == 0
    assert made[0].kw["send_api_seed"] is False and made[0].kw["max_tokens"] == 4096
    rec = json.loads(open(out).readline())
    assert rec["seed"] == 5 and rec["send_api_seed"] is False  # seed still keys the run locally


def test_cli_opt_in_flag_reaches_the_client_and_the_record(tmp_path, monkeypatch):
    made = _capture_client(monkeypatch)
    out = str(tmp_path / "o.jsonl")
    assert main(BASE + [out, "--families", "wire_timestamp", "--send-api-seed"]) == 0
    assert made[0].kw["send_api_seed"] is True
    assert json.loads(open(out).readline())["send_api_seed"] is True


def test_send_api_seed_is_part_of_the_run_fingerprint(tmp_path, monkeypatch):
    made = _capture_client(monkeypatch)
    out = str(tmp_path / "f.jsonl")
    assert main(BASE + [out, "--families", "wire_timestamp"]) == 0
    assert main(BASE + [out, "--families", "wire_timestamp", "--send-api-seed"]) == 3  # refuses to mix
    fams = select_families(load_split("dev"), ["wire_timestamp"])
    assert _settings_for(fams).fingerprint(fams) != _settings_for(fams, send_api_seed=True).fingerprint(fams)


def test_old_flag_is_gone_and_low_max_tokens_warns(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(BASE + ["x", "--no-seed-param"])
    assert build_parser().parse_args(BASE + ["x"]).send_api_seed is False
    assert build_parser().parse_args(BASE + ["x"]).max_tokens == 4096
    assert main(BASE + [str(tmp_path / "w.jsonl"), "--dry-run", "--max-tokens", "512"]) == 0
    assert "--max-tokens 512 < 2048" in capsys.readouterr().err


# ---- step 8c-3: diagnose quality scores, --max-retries, summarize --detail -----------------------

from eval.tests.test_diagnostics import FakeExport  # noqa: E402


class _Q(FakeExport):
    def __init__(self):
        super().__init__({})

    def export_namespace(self):
        return {"memories": [{"content": t, "importance_base": 0.3 + (len(t) % 40) / 100, "confidence": 0.6}
                             for t in self.rows.values()]}


def test_diagnose_prints_and_writes_quality_scores_for_every_dev_lesson(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("eval.lesson_store.make_memoire", lambda **kw: _Q())
    out = tmp_path / "d.json"
    assert main(["diagnose", "--split", "dev", "--quality-threshold", "0.5", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "importance_base" in text and ("admit" in text or "shadow" in text)
    for kind in ("gold", "vague", "wrong", "poison"):
        assert f"\n{kind} " in text
    assert "admitted at tau" in text and "fitting dev wording" in text
    data = json.loads(out.read_text())
    lesson_keys = [k for k in data if k.startswith("lesson:")]
    assert len(lesson_keys) == 20  # 5 families x gold/vague/wrong/poison
    assert all(data[k]["quality"]["importance_base"] is not None for k in lesson_keys)
    assert not any("quality" in data[k] for k in data if k.startswith("spec:"))
    assert set(data["_quality_summary"]) == {"gold", "vague", "wrong", "poison"}
    assert data["_quality_pooled"]["n"] == 20 and "p10" in data["_quality_pooled"]
    assert "_relevance_separation" in data
    assert "pooled over all 20 lessons" in text and "relevance separation" in text


def test_run_records_carry_the_quality_score_of_retrieved_lessons(tmp_path, monkeypatch):
    from eval.tests.fakes import FakeMemoire
    made = _capture_client(monkeypatch)
    monkeypatch.setattr("eval.lesson_store.make_memoire", lambda **kw: FakeMemoire())
    fams = select_families(load_split("dev"), ["wire_timestamp"])
    diag = {f"lesson:{l.key}": {"max_cos": 0.1, "quality": {"importance_base": 0.42}} for f in fams for l in f.lessons}
    dpath = tmp_path / "diag.json"
    dpath.write_text(json.dumps(diag))
    out = str(tmp_path / "c0.jsonl")
    assert main(BASE + [out, "--arms", "C0", "--families", "wire_timestamp", "--diagnostics-json", str(dpath)]) == 0
    recs = [json.loads(l) for l in open(out)]
    retrieved = [r for rec in recs for r in rec["retrieved"]]
    assert retrieved and all(r["quality_score"] == 0.42 and r["exemplar_cos"] == 0.1 for r in retrieved)


def test_max_retries_flag_reaches_the_client(tmp_path, monkeypatch):
    made = _capture_client(monkeypatch)
    assert main(BASE + [str(tmp_path / "r.jsonl"), "--families", "wire_timestamp", "--max-retries", "12"]) == 0
    assert made[0].kw["max_retries"] == 12
    assert build_parser().parse_args(BASE + ["x"]).max_retries == 8


def test_summarize_detail_shows_visible_hidden_status_and_failing_tests(tmp_path, capsys):
    path = _fixture(tmp_path, lambda t: "naive" if t.family == "http_status" else "ref")
    assert main(["summarize", "--results", path, "--detail"]) == 0
    out = capsys.readouterr().out
    assert "http_status/s1 seed=0 FAIL vis=ok hid=FAIL" in out
    assert "wire_timestamp/s1 seed=0 PASS vis=ok hid=ok" in out
    assert "failed=['test_bad_input_status'" in out or "failed=[" in out


# ---- step 8c-4: retrieval floors, admission and store scope flags ----------------------------

def test_retrieval_flags_reach_the_memoire_constructor_and_the_records(tmp_path, monkeypatch):
    from eval.tests.fakes import FakeMemoire
    _capture_client(monkeypatch)
    seen = []
    monkeypatch.setattr("eval.lesson_store.make_memoire", lambda **kw: (seen.append(kw), FakeMemoire())[1])
    out = str(tmp_path / "f.jsonl")
    argv = BASE + [out, "--arms", "C0", "--families", "wire_timestamp", "--verified-min-cos", "0.3",
                   "--trial-min-cos", "0.2", "--store-scope", "family", "--admission", "quality_gate",
                   "--quality-gate-threshold", "0.1", "--trial-cap", "2"]
    assert main(argv) == 0
    assert seen and seen[0] == {"contradiction": "off", "trial_cap": 2, "admission": "quality_gate",
                                "quality_gate_threshold": 0.1, "trial_min_cos": 0.2, "verified_min_cos": 0.3}
    rec = json.loads(open(out).readline())
    assert (rec["verified_min_cos"], rec["trial_min_cos"], rec["store_scope"], rec["admission"]) == (0.3, 0.2, "family", "quality_gate")


def test_defaults_leave_engine_defaults_untouched_and_run_four_sessions(tmp_path, monkeypatch):
    from eval.tests.fakes import FakeMemoire
    _capture_client(monkeypatch)
    seen = []
    monkeypatch.setattr("eval.lesson_store.make_memoire", lambda **kw: (seen.append(kw), FakeMemoire())[1])
    out = str(tmp_path / "d.jsonl")
    assert main(BASE + [out, "--arms", "C0", "--families", "wire_timestamp"]) == 0
    assert seen[0]["verified_min_cos"] is None and seen[0]["trial_min_cos"] is None and seen[0]["admission"] == "admit_all"
    assert [json.loads(l)["session"] for l in open(out)] == [1, 2, 3, 4]


@pytest.mark.parametrize("flag", ["--verified-min-cos", "--trial-min-cos", "--quality-gate-threshold"])
@pytest.mark.parametrize("bad", ["1.5", "-0.1"])
def test_out_of_range_floors_are_rejected_by_the_parser(flag, bad):
    with pytest.raises(SystemExit):
        build_parser().parse_args(BASE + ["x", flag, bad])


def test_prompt_policy_flag_reaches_the_records_and_defaults_to_plain(tmp_path, monkeypatch):
    _capture_client(monkeypatch)
    monkeypatch.setattr("eval.lesson_store.make_memoire", lambda **kw: FakeMemoire())
    for policy, arm, expected in (("tagged", "C0", "tagged"), ("tagged", "B", "plain"), (None, "C1", "plain")):
        out = str(tmp_path / f"{policy}-{arm}.jsonl")
        argv = BASE + [out, "--arms", arm, "--families", "wire_timestamp"] + (["--prompt-policy", policy] if policy else [])
        assert main(argv) == 0
        recs = [json.loads(line) for line in open(out)]
        assert recs and {r["prompt_policy"] for r in recs} == {expected}, (policy, arm)
    with pytest.raises(SystemExit):
        main(BASE + [str(tmp_path / "bad.jsonl"), "--arms", "C0", "--prompt-policy", "fancy"])

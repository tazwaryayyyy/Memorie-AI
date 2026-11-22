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
    assert "planned calls: 15" in out and not (tmp_path / "o.jsonl").exists()


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

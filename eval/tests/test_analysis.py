"""step 8d-1: pre-registered analysis, exercised on synthetic JSONL (no network, no native module, no test-split run)."""
import itertools
import json
import random
from fractions import Fraction as F
from types import SimpleNamespace

import pytest

from eval import analysis as an
from eval.run_eval import main
from eval.tasks import load_split

FAM8 = [SimpleNamespace(name=f"fam{i}", tasks=tuple(SimpleNamespace(session=s) for s in (1, 2, 3, 4)), poison_session=2)
        for i in range(8)]

SPEC = {
    "A": dict(arm="no_memory", trial_cap=1, prompt_policy="plain", no_poison=False),
    "B": dict(arm="plain_vector", trial_cap=1, prompt_policy="plain", no_poison=False),
    "B-nopoison": dict(arm="plain_vector", trial_cap=1, prompt_policy="plain", no_poison=True),
    "C0-t1": dict(arm="memoire_static", trial_cap=1, prompt_policy="tagged", no_poison=False),
    "C1-t1": dict(arm="memoire_feedback", trial_cap=1, prompt_policy="tagged", no_poison=False),
    "C0-t3": dict(arm="memoire_static", trial_cap=3, prompt_policy="tagged", no_poison=False),
    "C1-t3": dict(arm="memoire_feedback", trial_cap=3, prompt_policy="tagged", no_poison=False),
}


@pytest.fixture(autouse=True)
def eight_families(monkeypatch):
    monkeypatch.setattr("eval.analysis.load_split", lambda split: FAM8)


def rec(label, fam, session, seed, passed, **over):
    r = {"arm": SPEC[label]["arm"], "family": fam, "session": session, "seed": seed, "split": "test",
         "model": "m-1", "model_returned": "m-1", "temperature": 0.0, "top_k": 3, "admission": "admit_all",
         "store_scope": "family", "trial_cap": SPEC[label]["trial_cap"], "prompt_policy": SPEC[label]["prompt_policy"],
         "no_poison": SPEC[label]["no_poison"], "passed": passed, "parsed": {"code_found": True},
         "finish_reason": "stop", "used_keys": [], "retrieved": [], "fingerprint": f"fp-{label}",
         "prompt": [{"role": "user", "content": f"{fam}-s{session}"}], "_source": f"{label}.jsonl"}
    r.update(over)
    return r


def cond(label, patterns, seed=0, **over):
    """patterns: one 4-char P/F string per family (or a single string used for every family)."""
    if isinstance(patterns, str):
        patterns = [patterns] * len(FAM8)
    out = []
    for f, pat in zip(FAM8, patterns):
        for s, ch in zip((1, 2, 3, 4), pat):
            out.append(rec(label, f.name, s, seed, ch == "P", **over))
    return out


def write(tmp_path, label, records, name=None):
    path = tmp_path / (name or f"{label}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            r = {k: v for k, v in r.items() if k != "_source"}
            fh.write(json.dumps(r) + "\n")
    return str(path)


def cells_of(*groups):
    return an.build_cells([r for g in groups for r in g])


def primary_of(a, b, c1):
    cells = cells_of(cond("A", a), cond("B", b), cond("C1-t1", c1))
    gate = an.gate(cells, FAM8)
    prim = an.contrast(cells, FAM8, "C1-t1", "B", "s1-s4", primary=True)
    return prim, gate, an.verdict(prim, gate)


# ---- exact sign-flip ---------------------------------------------------------------------------------


@pytest.mark.parametrize("pos,neg,zero,expected", [
    (8, 0, 0, 2 / 256), (7, 1, 0, 18 / 256), (6, 2, 0, 74 / 256), (7, 0, 1, 2 / 128), (6, 1, 1, 16 / 128),
    (6, 0, 2, 2 / 64), (0, 0, 8, 1.0),
])
def test_exact_sign_flip_matches_reference_for_equal_magnitudes(pos, neg, zero, expected):
    deltas = [F(1, 4)] * pos + [F(-1, 4)] * neg + [F(0)] * zero
    assert an.exact_sign_flip_p(deltas) == pytest.approx(expected)


def test_exact_sign_flip_uses_magnitudes_not_just_counts_and_matches_brute_force():
    # seven +0.25 and one -0.0625: only 4 of 256 sign patterns reach |sum| >= observed
    assert an.exact_sign_flip_p([F(1, 4)] * 7 + [F(-1, 16)]) == pytest.approx(4 / 256)
    rng = random.Random(7)
    for _ in range(25):
        deltas = [F(rng.randint(-6, 6), 8) for _ in range(rng.randint(1, 8))]
        nz = [d for d in deltas if d != 0]
        obs = abs(sum(nz))
        hits = sum(1 for signs in itertools.product((1, -1), repeat=len(nz))
                   if abs(sum(s * abs(d) for s, d in zip(signs, nz))) >= obs)
        expected = hits / 2 ** len(nz) if nz else 1.0
        assert an.exact_sign_flip_p(deltas) == pytest.approx(expected)


def test_seven_of_eight_clears_one_sided_but_not_two_sided_at_alpha_05():
    two_sided = an.exact_sign_flip_p([F(1, 4)] * 7 + [F(-1, 4)])
    assert two_sided == pytest.approx(0.0703125) and two_sided > an.ALPHA
    one_sided = two_sided / 2  # what a one-sided test would have said (symmetric null)
    assert one_sided == pytest.approx(0.03515625) and one_sided < an.ALPHA


def test_bootstrap_is_deterministic_and_seeded_per_contrast(monkeypatch):
    d = [F(1, 4), F(0), F(-1, 4), F(1, 2), F(1, 4), F(0), F(1, 4), F(1, 2)]
    assert an.bootstrap_ci(d, "x") == an.bootstrap_ci(d, "x")
    lo, hi = an.bootstrap_ci(d, "x")
    assert lo <= float(sum(d, F(0)) / len(d)) <= hi
    # endpoints on discrete data can coincide across streams, so check the seeding itself
    seen = []
    real = an.random.Random
    monkeypatch.setattr(an.random, "Random", lambda seed: seen.append(seed) or real(seed))
    an.bootstrap_ci(d, "C1-t1 - B [s1-s4]")
    assert seen == [f"{an.BOOTSTRAP_SEED}:C1-t1 - B [s1-s4]"]


# ---- labels and registration -------------------------------------------------------------------------


@pytest.mark.parametrize("label", list(SPEC))
def test_condition_label_roundtrips_the_registered_conditions(label):
    assert an.condition_label(rec(label, "fam0", 1, 0, True)) == label


def test_unregistered_variants_get_distinct_labels_and_a_warning():
    plain = rec("C1-t1", "fam0", 1, 0, True, prompt_policy="plain")
    assert an.condition_label(plain) == "C1-t1-plain"
    cells = cells_of(cond("B", "PPPP"), cond("C1-t1", "PPPP", prompt_policy="plain"))
    assert any("unregistered" in w for w in an.integrity(cells, FAM8, "test")["warnings"])


def test_registration_is_one_primary_and_everything_else_descriptive():
    assert PRIMARY_NOT_IN_SECONDARY()
    assert set(an.PRIMARY[:2]) <= set(an.REGISTERED)
    for a, b, w in an.SECONDARY:
        assert a in an.REGISTERED and b in an.REGISTERED and w in an.WINDOWS


def PRIMARY_NOT_IN_SECONDARY():
    return an.PRIMARY not in an.SECONDARY


def test_only_the_primary_contrast_ever_carries_a_p_value():
    groups = [cond(l, "PPFF" if i % 2 else "PFFP") for i, l in enumerate(SPEC)]
    cells = cells_of(*groups)
    secondary = [an.contrast(cells, FAM8, a, b, w) for a, b, w in an.SECONDARY]
    assert all(c["status"] == "ok" and c["p_exact_two_sided"] is None for c in secondary)
    assert an.contrast(cells, FAM8, *an.PRIMARY[:2], an.PRIMARY[2], primary=True)["p_exact_two_sided"] is not None


# ---- verdict logic -----------------------------------------------------------------------------------


def test_verdict_supported_worse_and_not_distinguishable():
    prim, gate, verdict = primary_of("FFFP", "PPFF", "PPPP")
    assert gate["passed"] is True and prim["p_exact_two_sided"] == pytest.approx(2 / 256)
    assert verdict.startswith("supported")
    assert primary_of("FFFP", "PPFF", "FFFF")[2].startswith("worse")
    # 7 families better, 1 worse by the same amount: two-sided p = 0.0703 -> not distinguishable
    c1 = ["PPPP"] * 7 + ["FFFF"]
    prim, _, verdict = primary_of("FFFP", "PPFF", c1)
    assert prim["tally"] == {"a_better": 7, "b_better": 1, "tied": 0}
    assert prim["p_exact_two_sided"] == pytest.approx(0.0703125)
    assert verdict.startswith("not_distinguishable")


def test_gate_failure_blocks_inference_but_still_reports_the_effect():
    prim, gate, verdict = primary_of("PPPF", "PPFF", "PPPP")  # A at 0.75 is outside [0.15, 0.45]
    assert gate["passed"] is False and gate["a_in_range"] is False
    assert verdict.startswith("gate_not_passed") and prim["status"] == "ok"
    _, gate2, verdict2 = primary_of("FFFP", "FFFP", "PPPP")  # B - A = 0 < 0.15
    assert gate2["b_minus_a_ok"] is False and verdict2.startswith("gate_not_passed")


def test_missing_cells_are_reported_never_silently_dropped():
    c1 = cond("C1-t1", "PPPP")
    del c1[5]
    cells = cells_of(cond("A", "FFFP"), cond("B", "PPFF"), c1)
    prim = an.contrast(cells, FAM8, "C1-t1", "B", "s1-s4", primary=True)
    assert prim["status"].startswith("incomplete") and prim["p_exact_two_sided"] is None
    assert an.verdict(prim, an.gate(cells, FAM8)) == "incomplete"
    integ = an.integrity(cells, FAM8, "test")
    assert not integ["ok"] and any("C1-t1" in f and "missing" in f for f in integ["failures"])


def test_contrasts_with_absent_conditions_are_skipped_not_failed():
    cells = cells_of(cond("B", "PPFF"))
    assert an.contrast(cells, FAM8, "C1-t3", "B", "s1-s4")["status"].startswith("skipped")


# ---- windows, seeds, poison, s1 identity -------------------------------------------------------------


def test_windows_pool_the_right_sessions_and_seeds():
    cells = cells_of(cond("B", "PPFF", seed=0), cond("B", "FFPP", seed=1))
    # 8 families x 2 seeds x window sessions cells; seed 0 passes s1-s2, seed 1 passes s3-s4
    for window, expected in (("s1-s4", (32, 64)), ("s1-s2", (16, 32)), ("s3-s4", (16, 32)), ("s1", (8, 16))):
        r = an.pooled_rate(cells, "B", FAM8, [0, 1], window)
        assert (r["passed"], r["n"]) == expected, window


def test_poison_metrics_count_exposure_failure_and_declared_use():
    recs = []
    for f in FAM8:
        recs.append(rec("B", f.name, 1, 0, True))
        recs.append(rec("B", f.name, 2, 0, False, retrieved=[{"kind": "poison"}, {"kind": "gold"}],
                        used_keys=[f"{f.name}:poison"]))
        recs.append(rec("B", f.name, 3, 0, True, retrieved=[{"kind": "poison"}], used_keys=[f"{f.name}:gold"]))
        recs.append(rec("B", f.name, 4, 0, True, retrieved=[{"kind": "gold"}]))
    m = an.poison_metrics(cells_of(recs), "B", FAM8, [0])
    assert m["cells_at_or_after_poison_session"] == 24 and m["poison_shown"] == 16
    assert m["exposure_rate"] == pytest.approx(16 / 24) and m["failed_given_shown"] == pytest.approx(8 / 16)
    assert m["failed_and_poison_declared"] == 8
    assert m["last_session_cells"] == 8 and m["last_session_poison_shown"] == 0


def test_no_poison_condition_must_never_show_poison_and_s1_prompts_must_match():
    clean = cells_of(cond("B", "PPFF"), cond("B-nopoison", "PPFF"))
    assert an.integrity(clean, FAM8, "test")["ok"]
    ident = an.s1_identity(clean, FAM8)
    assert ident == {"pairs": 8, "prompts_identical": 8, "outcome_flips": 0}

    leaky = cond("B-nopoison", "PPFF")
    leaky[5]["retrieved"] = [{"kind": "poison"}]
    bad = an.integrity(cells_of(cond("B", "PPFF"), leaky), FAM8, "test")
    assert any("showed a poison" in f for f in bad["failures"])

    skew = cond("B-nopoison", "PPFF")
    skew[0]["prompt"] = [{"role": "user", "content": "different"}]
    skew[0]["passed"] = False
    cells = cells_of(cond("B", "PPFF"), skew)
    assert an.s1_identity(cells, FAM8) == {"pairs": 8, "prompts_identical": 7, "outcome_flips": 1}
    assert any("s1 prompts differ" in f for f in an.integrity(cells, FAM8, "test")["failures"])


def test_integrity_flags_wrong_split_scope_and_mixed_model():
    cells = cells_of(cond("B", "PPFF", split="dev"), cond("C1-t1", "PPFF", store_scope="shared", model_returned="m-2"))
    failures = " | ".join(an.integrity(cells, FAM8, "test")["failures"])
    assert "split" in failures and "store_scope" in failures and "model_returned" in failures


def test_truncation_and_missing_code_are_counted_and_warned():
    recs = cond("B", "PPPP")
    for r in recs[:4]:
        r["finish_reason"] = "length"
    recs[4]["parsed"] = {"code_found": False}
    integ = an.integrity(cells_of(recs), FAM8, "test")
    assert integ["format_failures"]["B"] == {"no_code_block": 1, "truncated": 4, "n": 32}
    assert any("no code block or were truncated" in w for w in integ["warnings"])


# ---- preflight (seed-count rule) ---------------------------------------------------------------------


def _pre(tmp_path, flips_b, flips_c1):
    def seeded(label, flips):
        s0 = cond(label, "PPFF")
        s1 = cond(label, "PPFF", seed=1)
        for r in s1[:flips]:
            r["passed"] = not r["passed"]
        return s0 + s1
    return [write(tmp_path, "B", seeded("B", flips_b)), write(tmp_path, "C1-t1", seeded("C1-t1", flips_c1))]


def test_preflight_rule_is_one_seed_at_or_below_one_flip_else_two(tmp_path):
    assert an.preflight(_pre(tmp_path, 0, 0), "dev")["decision"] == "ONE seed on test"
    assert an.preflight(_pre(tmp_path, 1, 0), "dev")["decision"] == "ONE seed on test"
    two = an.preflight(_pre(tmp_path, 1, 1), "dev")
    assert two["flips"] == 2 and two["decision"] == "TWO seeds on test"


def test_preflight_ignores_other_conditions_in_the_input_and_says_so(tmp_path, capsys):
    extra = write(tmp_path, "A", cond("A", "FFFP") + cond("A", "FFFP", seed=1))
    result = an.preflight([*_pre(tmp_path, 0, 0), extra], "dev")
    assert result["ignored_conditions"] == ["A"] and result["decision"] == "ONE seed on test"
    assert main(["analyze", "--split", "dev", "--preflight", "--results", *_pre(tmp_path, 0, 0), extra]) == 0
    assert "ignored other conditions" in capsys.readouterr().out


def test_preflight_demands_b_and_c1_with_complete_seeds_zero_and_one(tmp_path):
    only_b = [_pre(tmp_path, 0, 0)[0]]
    with pytest.raises(an.AnalysisError):
        an.preflight(only_b, "dev")
    one_seed = [write(tmp_path, "B", cond("B", "PPFF")), write(tmp_path, "C1-t1", cond("C1-t1", "PPFF"))]
    with pytest.raises(an.AnalysisError):
        an.preflight(one_seed, "dev")


# ---- files -------------------------------------------------------------------------------------------


def test_duplicate_cells_across_files_are_an_error_naming_both_files(tmp_path):
    a = write(tmp_path, "B", cond("B", "PPFF"), "one.jsonl")
    b = write(tmp_path, "B", cond("B", "PPFF"), "two.jsonl")
    with pytest.raises(an.AnalysisError) as exc:
        an.build_cells(an.load_records([a, b]))
    assert "one.jsonl" in str(exc.value) and "two.jsonl" in str(exc.value)


def test_torn_trailing_line_is_ignored_but_a_corrupt_middle_line_is_not(tmp_path):
    good = write(tmp_path, "B", cond("B", "PPFF"))
    with open(good, "a", encoding="utf-8") as fh:
        fh.write('{"arm": "plain_ve')
    assert len(an.load_records([good])) == 32
    path = tmp_path / "bad.jsonl"
    path.write_text('{"x": 1}\nnot json\n{"x": 2}\n', encoding="utf-8")
    with pytest.raises(an.AnalysisError):
        an.load_records([str(path)])


# ---- end to end through the CLI ----------------------------------------------------------------------


def _all_files(tmp_path):
    pats = {"A": "FFFP", "B": "PPFF", "B-nopoison": "PPFF", "C0-t1": "PFFF", "C1-t1": "PPFF", "C0-t3": "PPFF",
            "C1-t3": "PPFP"}
    return [write(tmp_path, l, cond(l, p)) for l, p in pats.items()]


def test_analyze_cli_writes_a_deterministic_json_report_and_prints_the_verdict(tmp_path, capsys):
    files = _all_files(tmp_path)
    out1, out2 = str(tmp_path / "r1.json"), str(tmp_path / "r2.json")
    assert main(["analyze", "--split", "test", "--results", *files, "--out", out1]) == 0
    printed = capsys.readouterr().out
    assert "PRIMARY C1-t1 - B [s1-s4]" in printed and "VERDICT:" in printed and "integrity: OK" in printed
    assert main(["analyze", "--split", "test", "--results", *files, "--out", out2]) == 0
    assert open(out1).read() == open(out2).read()
    report = json.load(open(out1))
    assert report["primary"]["tally"] == {"a_better": 0, "b_better": 0, "tied": 8}
    assert report["primary"]["p_exact_two_sided"] == 1.0
    assert report["primary_verdict"].startswith("not_distinguishable")
    assert report["integrity"]["registered_present"] == list(an.REGISTERED)
    assert all(c["p_exact_two_sided"] is None for c in report["secondary"])
    assert report["constants"]["primary"] == list(an.PRIMARY)


def test_analyze_cli_exits_nonzero_on_integrity_failure_and_2_on_bad_input(tmp_path, capsys):
    broken = cond("C1-t1", "PPPF")
    del broken[3]
    files = [write(tmp_path, "A", cond("A", "FFFP")), write(tmp_path, "B", cond("B", "PPFF")),
             write(tmp_path, "C1-t1", broken)]
    assert main(["analyze", "--split", "test", "--results", *files]) == 1
    assert "integrity: FAILED" in capsys.readouterr().out
    dup = write(tmp_path, "B", cond("B", "PPFF"), "dup.jsonl")
    assert main(["analyze", "--split", "test", "--results", files[1], dup]) == 2
    assert "duplicate cell" in capsys.readouterr().err


def test_analyze_cli_preflight_prints_the_decision(tmp_path, capsys):
    assert main(["analyze", "--split", "dev", "--preflight", "--results", *_pre(tmp_path, 1, 0)]) == 0
    out = capsys.readouterr().out
    assert "1 flips in 64 paired cells" in out and "DECISION: ONE seed on test" in out


# ---- the real test split: shape only, nothing is run -------------------------------------------------


def test_real_test_split_has_the_registered_shape(monkeypatch):
    monkeypatch.undo()
    fams = load_split("test")
    assert len(fams) == 8 and all(len(f.tasks) == 4 for f in fams)
    assert all(f.poison_session == 2 and len(f.bank_lessons) == 3 and f.poison.kind == "poison" for f in fams)
    assert sum(len(f.tasks) for f in fams) == 32


def test_analysis_reads_records_written_by_the_real_run_loop(tmp_path, monkeypatch):
    """Schema guard: records come from run_sequence (EngineBackedFake + ScriptedLLM), not from the helpers above."""
    from eval.arms import ARM_A, ARM_B, ARM_C1
    from eval.runner import ResultsFile, RunSettings, run_sequence
    from eval.tests.fakes import EngineBackedFake, ScriptedLLM

    monkeypatch.undo()  # use the real dev split instead of the synthetic 8 families
    fams = load_split("dev")
    tasks = [t for f in fams for t in f.tasks]
    paths = []
    for name, arm, no_poison, policy in (("A", ARM_A, False, "plain"), ("B", ARM_B, False, "plain"),
                                         ("Bnp", ARM_B, True, "plain"), ("C1", ARM_C1, False, "tagged")):
        st = RunSettings(split="dev", families=tuple(f.name for f in fams), sessions=(1, 2, 3, 4), model="scripted-1",
                         base_url="http://x", temperature=0.0, top_k=3, trial_cap=1, contradiction="off",
                         max_tokens=100, store_scope="family", prompt_policy=policy, no_poison=no_poison)
        path = str(tmp_path / f"{name}.jsonl")
        llm = ScriptedLLM(tasks, choose=lambda t: "naive", declare=lambda n: list(range(1, n + 1)))
        run_sequence(settings=st, arm=arm, seed=0, families=fams, llm=llm,
                     results=ResultsFile(path, st.fingerprint(fams)),
                     store_factory=lambda: EngineBackedFake(bonus=lambda text: 0.5 if "`" in text else 0.0),
                     log=lambda s: None)
        paths.append(path)
    report = an.analyze(paths, "dev")
    assert report["integrity"]["ok"], report["integrity"]["failures"]
    assert report["integrity"]["registered_present"] == ["A", "B", "B-nopoison", "C1-t1"]
    assert report["primary"]["status"] == "ok" and report["primary"]["n_families"] == 5
    assert report["s1_identity_B_vs_B_nopoison"] == {"pairs": 5, "prompts_identical": 5, "outcome_flips": 0}
    assert report["conditions"]["B-nopoison"]["poison"]["poison_shown"] == 0
    assert report["conditions"]["B"]["poison"]["poison_shown"] > 0
    assert all(c["p_exact_two_sided"] is None for c in report["secondary"])

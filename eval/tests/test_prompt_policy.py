"""step 8c-5: --prompt-policy tagged, and the raw-cosine diagnostic (no network, no native module)."""
import pytest

from eval.arms import ARM_A, ARM_B, ARM_C0, ARM_C1, effective_prompt_policy, prompt_notes
from eval.lesson_store import Retrieved
from eval.prompt import (NOTES_HEADER, NOTES_HEADER_TAGGED, POLICY_PLAIN, POLICY_TAGGED, Note, build_messages,
                         prompt_template_hash_input, tag_for)
from eval.run_eval import _shown
from eval.runner import ResultsFile, RunSettings, run_sequence, select_families
from eval.tasks import load_split
from eval.tests.fakes import EngineBackedFake, ScriptedLLM

DEV = load_split("dev")
TASK = DEV[0].tasks[0]


def _user(notes, policy):
    return build_messages(TASK, notes, policy)[1]["content"]


def _note_lines(prompt):
    """Only the numbered note lines; the tagged legend legitimately mentions the tag formats."""
    return [ln for ln in prompt.splitlines() if ln.startswith("[M")]


# ---- tag semantics ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exposure,trust,expected",
    [
        ("trial", 0.5, "[UNTESTED]"),
        ("hint", 0.6, "[TRUST 0.60: worked more often than it failed]"),
        ("follow", 0.8, "[TRUST 0.80: worked more often than it failed]"),
        ("hint", 0.4, "[TRUST 0.40: failed more often than it worked]"),
        ("hint", 0.5, "[TRUST 0.50: mixed record]"),
        ("plain", 0.9, None),
        ("hidden", 0.1, None),
        ("hint", None, None),
    ],
)
def test_tag_depends_only_on_engine_exposure_and_trust(exposure, trust, expected):
    assert tag_for(Note("any text", trust, exposure)) == expected


def test_tags_never_depend_on_lesson_text_and_never_name_a_lesson_kind():
    a, b = Note("use the 3-arg form", 0.4, "hint"), Note("totally different words", 0.4, "hint")
    assert tag_for(a) == tag_for(b)
    every = [tag_for(Note("x", t, e)) for e in ("trial", "hint", "follow") for t in (0.3, 0.5, 0.7)]
    legend_addition = NOTES_HEADER_TAGGED[len(NOTES_HEADER):]  # the inherited header already says "may be wrong"
    blob = " ".join(t for t in every if t).lower() + legend_addition.lower()
    for word in ("gold", "wrong", "poison", "vague", "correct", "verified"):
        assert word not in blob, word


def test_trust_below_half_means_more_failures_than_successes_under_the_engine_posterior():
    # (2+s)/(4+s+f) < 0.5  <=>  s < f ; the wording in tag_for relies on this identity.
    for s in range(0, 6):
        for f in range(0, 6):
            if s + f == 0:
                continue
            trust = (2 + s) / (4 + s + f)
            tag = tag_for(Note("x", trust, "hint"))
            assert ("failed more often" in tag) == (f > s) and ("worked more often" in tag) == (s > f), (s, f, tag)


# ---- prompt rendering -------------------------------------------------------------------------------


def test_plain_policy_is_byte_identical_for_strings_and_notes():
    notes = [Note("first", 0.4, "hint"), Note("second", 0.5, "trial")]
    legacy = _user(["first", "second"], POLICY_PLAIN)
    assert _user(notes, POLICY_PLAIN) == legacy == build_messages(TASK, ["first", "second"])[1]["content"]
    assert NOTES_HEADER in legacy and "[UNTESTED]" not in legacy and "TRUST" not in legacy


def test_tagged_policy_prefixes_each_note_and_keeps_the_numbering_the_parser_counts():
    user = _user([Note("first", 0.4, "hint"), Note("second", 0.5, "trial"), Note("third", 0.5, "plain")], POLICY_TAGGED)
    assert "[M1] [TRUST 0.40: failed more often than it worked] first" in user
    assert "[M2] [UNTESTED] second" in user
    assert "[M3] third" in user  # no outcome semantics -> no tag
    assert NOTES_HEADER_TAGGED in user and "USED_MEMORY" in user


def test_tagged_policy_leaves_bare_strings_alone_and_unknown_policy_is_an_error():
    assert _note_lines(_user(["bare"], POLICY_TAGGED)) == ["[M1] bare"]
    with pytest.raises(ValueError):
        build_messages(TASK, ["x"], "fancy")


def test_arm_a_prompt_has_no_notes_under_either_policy():
    assert "Notes from earlier sessions" not in _user(None, POLICY_TAGGED)


# ---- which arms get tags ----------------------------------------------------------------------------


def _r(text="t", trust=0.4, exposure="hint"):
    return Retrieved(key="f:gold", kind="gold", text=text, id=1, score=0.5, trust=trust, exposure=exposure,
                     state="trial")


def test_only_the_trust_ranked_arms_get_tagged_notes():
    shown = [_r()]
    assert prompt_notes(ARM_A, shown, POLICY_TAGGED) is None
    assert prompt_notes(ARM_B, shown, POLICY_TAGGED) == ["t"]
    assert prompt_notes(ARM_C0, shown, POLICY_PLAIN) == ["t"] and prompt_notes(ARM_C1, shown, POLICY_PLAIN) == ["t"]
    for arm in (ARM_C0, ARM_C1):
        assert prompt_notes(arm, shown, POLICY_TAGGED) == [Note("t", 0.4, "hint")]
    assert [effective_prompt_policy(a, POLICY_TAGGED) for a in (ARM_A, ARM_B, ARM_C0, ARM_C1)] == [
        POLICY_PLAIN, POLICY_PLAIN, POLICY_TAGGED, POLICY_TAGGED]


# ---- end to end through the run loop ----------------------------------------------------------------


def _settings(fams, **kw):
    base = dict(split="dev", families=tuple(f.name for f in fams), sessions=(1, 2, 3, 4), model="scripted-1",
                base_url="http://x", temperature=0.0, top_k=3, trial_cap=1, contradiction="off", max_tokens=100)
    base.update(kw)
    return RunSettings(**base)


def _episode(tmp_path, arm, policy, name):
    fams = select_families(DEV, ["wire_timestamp"])
    st = _settings(fams, prompt_policy=policy)
    tasks = [t for f in fams for t in f.tasks]
    llm = ScriptedLLM(tasks, choose=lambda t: "naive", declare=lambda n: list(range(1, n + 1)))
    results = ResultsFile(str(tmp_path / name), st.fingerprint(fams))
    run_sequence(settings=st, arm=arm, seed=0, families=fams, llm=llm, results=results,
                 store_factory=lambda: EngineBackedFake(), log=lambda s: None)
    return [p[-1]["content"] for p in llm.prompts], results


def test_c1_tagged_shows_the_penalty_it_caused_while_c1_plain_shows_nothing(tmp_path):
    tagged, rec_t = _episode(tmp_path, ARM_C1, POLICY_TAGGED, "t.jsonl")
    plain, rec_p = _episode(tmp_path, ARM_C1, POLICY_PLAIN, "p.jsonl")
    notes = lambda p: " ".join(_note_lines(p))
    assert "[UNTESTED]" in notes(tagged[0]) and "failed more often" not in notes(tagged[0])  # s1: no history yet
    assert "failed more often than it worked" in notes(tagged[1])  # s1 failed with the note declared -> penalized
    assert all("[TRUST" not in notes(p) and "[UNTESTED]" not in notes(p) for p in plain)
    assert all(NOTES_HEADER_TAGGED not in p for p in plain)
    assert {r["prompt_policy"] for r in rec_t.ordered} == {"tagged"} and {r["prompt_policy"] for r in rec_p.ordered} == {"plain"}
    # retrieval is identical, only the rendering differs (same seeds, same fake, same scripted answers)
    keys = lambda res: [[x["key"] for x in r["retrieved"]] for r in res.ordered]
    assert keys(rec_t) == keys(rec_p)


def test_c0_tagged_is_a_null_control_every_note_is_untested(tmp_path):
    tagged, _ = _episode(tmp_path, ARM_C0, POLICY_TAGGED, "c0.jsonl")
    lines = [ln for p in tagged for ln in _note_lines(p)]
    assert lines and all(ln.startswith(("[M1] [UNTESTED]", "[M2] [UNTESTED]", "[M3] [UNTESTED]")) for ln in lines)


def test_arm_b_is_untouched_by_the_tagged_policy_and_records_plain(tmp_path):
    tagged, results = _episode(tmp_path, ARM_B, POLICY_TAGGED, "b.jsonl")
    assert all(not any(("[TRUST" in ln or "[UNTESTED]" in ln) for ln in _note_lines(p)) for p in tagged)
    assert all(NOTES_HEADER_TAGGED not in p for p in tagged)
    assert {r["prompt_policy"] for r in results.ordered} == {"plain"}


def test_prompt_policy_is_part_of_the_run_fingerprint_and_validated():
    fams = select_families(DEV, ["wire_timestamp"])
    assert _settings(fams).fingerprint(fams) != _settings(fams, prompt_policy="tagged").fingerprint(fams)
    assert prompt_template_hash_input("plain") != prompt_template_hash_input("tagged")
    with pytest.raises(ValueError):
        _settings(fams, prompt_policy="fancy")


# ---- raw cosine diagnostic --------------------------------------------------------------------------


def test_records_carry_raw_cosine_next_to_the_blended_score(tmp_path):
    _, rec_c = _episode(tmp_path, ARM_C1, POLICY_PLAIN, "c.jsonl")
    _, rec_b = _episode(tmp_path, ARM_B, POLICY_PLAIN, "b2.jsonl")
    for res in (rec_c, rec_b):
        assert all(x["cos"] is not None for r in res.ordered for x in r["retrieved"])
    assert all(x["cos"] == x["score"] for r in rec_b.ordered for x in r["retrieved"])  # plain recall: score is cosine


def test_summarize_detail_helper_prints_kind_exposure_trust_and_cosine():
    rec = {"retrieved": [{"key": "fee_rounding:gold", "exposure": "hint", "trust": 0.6, "cos": 0.6123},
                         {"key": "fee_rounding:wrong", "exposure": "trial", "trust": 0.5, "cos": None}]}
    assert _shown(rec) == "gold:hint:0.60:0.612,wrong:trial:0.50:?"
    assert _shown({"retrieved": []}) == "-"

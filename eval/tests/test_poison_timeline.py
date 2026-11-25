"""How many sessions does feedback need? Timelines under the engine's real rules.

Rules (src/quality.rs, src/store.rs): any feedback makes a row "verified" (exempt from trial_cap,
and with no similarity floor); a row is hidden once 2f > 3s + 2 (two failures when s == 0); it reaches
`follow` once 3s >= 8 + 7f (three successes when f == 0). These tests use EngineModel, a mirror that
the native test cross-checks.
"""
from eval.tests.fakes import EngineModel

SIMS = {"G": 0.80, "V": 0.60, "W": 0.70, "P": 0.90}


def _model(**kw):
    m = EngineModel(SIMS, **kw)
    for key in ("G", "V", "W"):
        m.add(key)
    return m


def test_poison_arriving_at_session_two_is_hidden_only_after_session_three():
    m = _model()
    m.add("P")                       # ingested immediately before s2
    assert m.exposure("P") == "trial" and "P" in m.recall()          # s2 retrieval: P offered
    m.penalize(["P"])                # s2 run failed, P declared -> f = 1
    assert m.exposure("P") == "hint" and "P" in m.recall()           # s3 retrieval: still offered, now as verified
    m.penalize(["P"])                # s3 run failed -> f = 2 -> hidden, but no session remains
    assert m.exposure("P") == "hidden"
    assert "P" not in m.recall()     # a 4th session would be the first one without P


def test_poison_present_from_session_one_is_hidden_before_session_three():
    m = _model()
    m.add("P")
    m.penalize(["P"])                # s1 failed
    assert "P" in m.recall()         # s2 retrieval still offers it
    m.penalize(["P"])                # s2 failed
    assert "P" not in m.recall()     # s3 retrieval: suppressed


def test_a_wrong_lesson_used_in_sessions_one_and_two_is_hidden_for_session_three():
    m = _model()
    m.penalize(["W"])
    m.penalize(["W"])
    assert "W" not in m.recall()


def test_a_lesson_with_one_success_still_needs_three_failures_to_be_hidden():
    m = _model()
    m.reinforce("W")
    m.penalize(["W"]); m.penalize(["W"])
    assert m.exposure("W") == "hint"      # 2f = 4 > 3*1 + 2 = 5 is false
    m.penalize(["W"])
    assert m.exposure("W") == "hidden"    # 6 > 5


def test_gold_reaches_follow_only_after_three_reinforcements_so_not_within_three_sessions():
    m = _model()
    seen = []
    for _session in (1, 2, 3):
        seen.append(m.exposure("G"))      # label at the START of the session
        m.reinforce("G")                  # that session passed with G declared
    assert seen == ["trial", "hint", "hint"]
    assert m.exposure("G") == "follow"    # visible only to a 4th session


def test_a_once_failed_lesson_keeps_a_guaranteed_slot_while_untried_ones_stay_capped():
    m = _model(trial_cap=1, k=3)
    m.add("P")
    m.penalize(["P"])
    got = m.recall()
    assert "P" in got                                  # verified rows are not capped
    assert sum(1 for g in got if m.exposure(g) == "trial") == 1


def test_verified_rows_have_no_similarity_floor():
    m = _model()
    m.sims["U"] = 0.0                                  # an unrelated lesson from another family
    m.add("U"); m.reinforce("U")
    assert "U" in m.recall()


def test_four_sessions_are_the_minimum_to_observe_both_mechanisms_after_a_session_two_poison():
    m = _model()
    m.add("P")
    follows, poison_offered = [], []
    for session in (2, 3, 4):
        poison_offered.append("P" in m.recall())
        follows.append(m.exposure("G"))
        if poison_offered[-1]:
            m.penalize(["P"])
        m.reinforce("G")
    assert poison_offered == [True, True, False]      # P is suppressed in the 4th session only


# ---- verified relevance floor (src/quality.rs ScoringConfig::verified_min_cos) --------------------

def test_without_a_floor_an_unrelated_verified_row_takes_the_only_slot_at_k_equals_one():
    m = EngineModel({"T": 0.9, "U": 0.0}, trial_cap=1, k=1)
    m.add("T"); m.add("U"); m.reinforce("U")
    assert m.recall() == ["U"]


def test_a_verified_floor_gives_the_only_slot_back_to_the_relevant_trial_row():
    m = EngineModel({"T": 0.9, "U": 0.0}, trial_cap=1, k=1, verified_min_cos=0.3)
    m.add("T"); m.add("U"); m.reinforce("U")
    assert m.recall() == ["T"]


def test_at_the_eval_settings_k3_cap1_unrelated_verified_rows_add_noise_not_displacement():
    sims = {"T": 0.9, "U1": 0.05, "U2": 0.04}
    open_ = EngineModel(sims, trial_cap=1, k=3)
    floored = EngineModel(sims, trial_cap=1, k=3, verified_min_cos=0.3)
    for m in (open_, floored):
        for key in sims:
            m.add(key)
        m.reinforce("U1"); m.reinforce("U2")
    assert "T" in open_.recall() and set(open_.recall()) == {"T", "U1", "U2"}   # noise fills the free slots
    assert floored.recall() == ["T"]


def test_a_floor_never_hides_a_relevant_verified_row():
    m = EngineModel({"G": 0.6, "U": 0.05}, trial_cap=1, k=3, verified_min_cos=0.3)
    m.add("G"); m.add("U"); m.reinforce("G"); m.reinforce("U")
    assert m.recall() == ["G"]


def test_a_trial_floor_keeps_unrelated_unverified_rows_out_of_the_trial_slot():
    sims = {"T": 0.9, "U": 0.05}
    open_ = EngineModel(sims, trial_cap=2, k=3)
    floored = EngineModel(sims, trial_cap=2, k=3, trial_min_cos=0.3)
    for m in (open_, floored):
        m.add("T"); m.add("U")
    assert set(open_.recall()) == {"T", "U"}
    assert floored.recall() == ["T"]

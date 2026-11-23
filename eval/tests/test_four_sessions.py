"""4-session episodes through the real run loop, backed by engine-rule fakes (no network, no native)."""
import json

import pytest

from eval.arms import ARM_C0, ARM_C1
from eval.runner import ResultsFile, RunSettings, run_sequence, select_families
from eval.tasks import load_split
from eval.tests.fakes import EngineBackedFake, GateFake, ScriptedLLM

DEV = load_split("dev")


def _settings(families, **kw):
    base = dict(split="dev", families=tuple(f.name for f in families), sessions=(1, 2, 3, 4), model="scripted-1",
                base_url="http://x", temperature=0.0, top_k=3, trial_cap=1, contradiction="off", max_tokens=100)
    base.update(kw)
    return RunSettings(**base)


POISON_BONUS = lambda text: 0.5 if "`" in text else 0.0  # poison texts name the s2 function in backticks


def _poisoned():
    return EngineBackedFake(bonus=POISON_BONUS)


def _run(tmp_path, names, arm=ARM_C1, choose=lambda t: "naive", declare_all=True, factory=None, name="r.jsonl", **kw):
    fams = select_families(DEV, names)
    st = _settings(fams, **kw)
    tasks = [t for f in fams for t in f.tasks]
    declare = (lambda n: list(range(1, n + 1))) if declare_all else (lambda n: [])
    llm = ScriptedLLM(tasks, choose, declare)
    results = ResultsFile(str(tmp_path / name), st.fingerprint(fams))
    made = []

    def make():
        made.append(factory() if factory else _poisoned())
        return made[-1]

    run_sequence(settings=st, arm=arm, seed=0, families=fams, llm=llm, results=results, store_factory=make,
                 log=lambda s: None)
    return fams, results, made


def _offered(results, family):
    recs = sorted((r for r in results.ordered if r["family"] == family), key=lambda r: r["session"])
    return [[x["key"] for x in r["retrieved"]] for r in recs]


def test_default_episode_has_four_sessions_per_family(tmp_path):
    fams, results, _ = _run(tmp_path, ["wire_timestamp"])
    assert [r["session"] for r in results.ordered] == [1, 2, 3, 4]


def test_a_session_two_poison_is_offered_at_s2_and_s3_and_suppressed_at_s4(tmp_path):
    fams, results, made = _run(tmp_path, ["wire_timestamp"])
    poison = fams[0].poison.key
    s1, s2, s3, s4 = _offered(results, "wire_timestamp")
    assert poison not in s1                       # not ingested yet
    assert poison in s2 and poison in s3          # trial at s2, verified "hint" at s3
    assert poison not in s4                       # f == 2 after s3 -> hidden for s4
    pid = next(i for i, t in made[0].rows.items() if t == fams[0].poison.text)
    assert made[0].model.f[pid] == 2 and pid in made[0].model.hidden
    exposures = {r["session"]: [x["exposure"] for x in r["retrieved"] if x["key"] == poison]
                 for r in results.ordered}
    assert exposures[2] == ["trial"] and exposures[3] == ["hint"] and exposures[4] == []


def test_the_same_episode_cannot_suppress_the_poison_with_three_sessions(tmp_path):
    fams, results, made = _run(tmp_path, ["wire_timestamp"], sessions=(1, 2, 3))
    poison = fams[0].poison.key
    s1, s2, s3 = _offered(results, "wire_timestamp")
    assert poison in s2 and poison in s3
    pid = next(i for i, t in made[0].rows.items() if t == fams[0].poison.text)
    assert made[0].model.f[pid] == 2              # hidden only after the last session has run


def test_gold_reaches_follow_only_at_the_fourth_session_when_it_passes_every_time(tmp_path):
    fams, results, _ = _run(tmp_path, ["wire_timestamp"], choose=lambda t: "ref")
    gold = fams[0].lessons[0].key
    seen = {r["session"]: [x["exposure"] for x in r["retrieved"] if x["key"] == gold] for r in results.ordered}
    assert seen[1] == ["trial"] and seen[2] == ["hint"] and seen[3] == ["hint"] and seen[4] == ["follow"]


def _foreign(results, family, exposures=None):
    return [(r["session"], x["key"], x["exposure"]) for r in results.ordered if r["family"] == family
            for x in r["retrieved"] if not x["key"].startswith(f"{family}:")
            and (exposures is None or x["exposure"] in exposures)]


NAMES = ["wire_timestamp", "lock_ordering"]


def test_shared_store_without_floors_leaks_other_families_lessons_through_both_pools(tmp_path):
    _, results, made = _run(tmp_path, NAMES, name="shared.jsonl", store_scope="shared")
    leaked = _foreign(results, "wire_timestamp") + _foreign(results, "lock_ordering")
    assert leaked
    assert {e for _, _, e in leaked} & {"trial"}, "trial_min_cos = 0 hands trial slots to unrelated lessons"
    # ...and feedback then lands on lessons that never belonged to the task (here: before their own s1):
    lock_gold = next(i for i, t in made[0].rows.items() if t == select_families(DEV, ["lock_ordering"])[0].lessons[0].text)
    first_lock_run = min(r["session"] for r in results.ordered if r["family"] == "lock_ordering")
    assert first_lock_run == 1 and made[0].model.f[lock_gold] + made[0].model.s[lock_gold] >= 1


def test_a_verified_floor_alone_stops_verified_spill_but_not_the_trial_leak(tmp_path):
    factory = lambda: EngineBackedFake(verified_min_cos=0.5, bonus=POISON_BONUS)
    _, results, _ = _run(tmp_path, NAMES, name="v.jsonl", factory=factory, verified_min_cos=0.5)
    verified_foreign = _foreign(results, "wire_timestamp", {"hint", "follow"}) + _foreign(results, "lock_ordering", {"hint", "follow"})
    trial_foreign = _foreign(results, "wire_timestamp", {"trial"}) + _foreign(results, "lock_ordering", {"trial"})
    assert verified_foreign == [] and trial_foreign


def test_both_floors_together_remove_the_cross_family_leak(tmp_path):
    factory = lambda: EngineBackedFake(verified_min_cos=0.5, trial_min_cos=0.5, bonus=POISON_BONUS)
    _, results, _ = _run(tmp_path, NAMES, name="both.jsonl", factory=factory, verified_min_cos=0.5, trial_min_cos=0.5)
    assert _foreign(results, "wire_timestamp") == [] and _foreign(results, "lock_ordering") == []
    assert all(r["verified_min_cos"] == 0.5 and r["trial_min_cos"] == 0.5 for r in results.ordered)


def test_family_scope_isolates_families_completely(tmp_path):
    _, results, _ = _run(tmp_path, NAMES, name="family.jsonl", store_scope="family")
    assert _foreign(results, "wire_timestamp") == [] and _foreign(results, "lock_ordering") == []
    assert all(r["store_scope"] == "family" for r in results.ordered)


def test_quality_gate_runs_record_which_lessons_the_gate_hid_and_never_offer_them(tmp_path):
    fam = select_families(DEV, ["wire_timestamp"])[0]
    shadow = {fam.lessons[0].text, fam.lessons[3].text}             # gate hides gold and poison
    fams, results, _ = _run(tmp_path, ["wire_timestamp"], arm=ARM_C0, declare_all=False,
                            factory=lambda: GateFake(shadow), admission="quality_gate", quality_gate_threshold=0.3)
    expected = sorted([fam.lessons[0].key, fam.lessons[3].key])
    last = max(results.ordered, key=lambda r: r["session"])
    assert last["shadowed_keys"] == expected and last["admission"] == "quality_gate"
    first = min(results.ordered, key=lambda r: r["session"])
    assert first["shadowed_keys"] == [fam.lessons[0].key]            # poison not ingested before s2
    assert not any(x["key"] in expected for r in results.ordered for x in r["retrieved"])


def test_admit_all_runs_never_probe_the_hidden_view(tmp_path):
    _, results, made = _run(tmp_path, ["wire_timestamp"], arm=ARM_C0, declare_all=False)
    assert not any(c[0] == "recall" and c[3] for c in made[0].calls)
    assert all(r["shadowed_keys"] == [] for r in results.ordered)


def test_settings_that_change_retrieval_are_part_of_the_fingerprint():
    fams = select_families(DEV, ["wire_timestamp"])
    base = _settings(fams).fingerprint(fams)
    for change in (dict(verified_min_cos=0.3), dict(trial_min_cos=0.3), dict(store_scope="family"),
                   dict(admission="quality_gate"), dict(quality_gate_threshold=0.2), dict(sessions=(1, 2, 3))):
        assert _settings(fams, **change).fingerprint(fams) != base, change

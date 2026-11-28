"""
pytest test suite for the Memoire Python binding.

Run (inside a virtualenv; needs a Rust toolchain and, on first use, network
access to download the embedding model):
    cd bindings/python && maturin develop --release
    pytest tests/ -v
"""
import sqlite3
import struct
from types import SimpleNamespace

import pytest
from memoire_engine import Memoire, Memory, MemoireError, MemoryPolicy, policy_thresholds


@pytest.fixture
def m():
    """Fresh in-memory Memoire instance per test."""
    with Memoire(":memory:") as instance:
        yield instance


# ─── remember ────────────────────────────────────────────────────────────────

class TestRemember:
    def test_returns_chunk_count(self, m):
        n = m.remember("Fixed the authentication bug in middleware")
        assert n >= 1

    def test_increments_count(self, m):
        assert m.count() == 0
        m.remember("one memory")
        assert m.count() >= 1

    def test_long_input_produces_multiple_chunks(self, m):
        long = " ".join(f"word{i}" for i in range(300))
        n = m.remember(long)
        assert n > 1, "300-word input should produce multiple chunks"

    def test_empty_string_stores_nothing(self, m):
        n = m.remember("   ")
        assert n == 0
        assert m.count() == 0

    def test_type_error_on_non_string(self, m):
        with pytest.raises(TypeError):
            m.remember(42)  # type: ignore


# ─── remember_ids ────────────────────────────────────────────────────────────

class TestRememberIds:
    def test_returns_list_of_int_ids(self, m):
        ids = m.remember_ids("Fixed null pointer in auth middleware")
        assert isinstance(ids, list)
        assert len(ids) >= 1
        assert all(isinstance(i, int) for i in ids)

    def test_id_matches_recalled_memory(self, m):
        ids = m.remember_ids("Fixed null pointer in auth middleware")
        results = m.recall("authentication bug", top_k=1)
        assert results and results[0].id == ids[0]

    def test_ids_are_usable_with_forget(self, m):
        ids = m.remember_ids("temporary memory to forget")
        before = m.count()
        assert m.forget(ids[0]) is True
        assert m.count() == before - 1

    def test_blank_input_returns_empty_list(self, m):
        assert m.remember_ids("   ") == []
        assert m.count() == 0

    def test_exact_duplicate_returns_empty_list(self, m):
        text = "Fixed null pointer in auth middleware"
        assert m.remember_ids(text)
        assert m.remember_ids(text) == []
        assert m.remember(text) == 0  # remember() still returns an int count

    def test_type_error_on_non_string(self, m):
        with pytest.raises(TypeError):
            m.remember_ids(42)  # type: ignore


# ─── recall ──────────────────────────────────────────────────────────────────

class TestRecall:
    def test_empty_store_returns_empty_list(self, m):
        assert m.recall("anything") == []

    def test_returns_memory_objects(self, m):
        m.remember("Fixed null pointer in auth middleware")
        results = m.recall("authentication bug", top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], Memory)

    def test_score_is_float_between_0_and_1(self, m):
        m.remember("database query optimization with indexes")
        results = m.recall("database performance", top_k=1)
        assert results
        assert 0.0 <= results[0].score <= 1.0

    def test_scores_are_descending(self, m):
        m.remember("auth bug in JWT validation")
        m.remember("database connection pool refactor")
        m.remember("auth token refresh logic fix")
        results = m.recall("authentication", top_k=3)
        for a, b in zip(results, results[1:]):
            assert a.score >= b.score

    def test_top_k_limits_results(self, m):
        for i in range(10):
            m.remember(f"memory {i} about coding in Rust and Python")
        results = m.recall("coding", top_k=3)
        assert len(results) <= 3

    def test_most_relevant_ranks_first(self, m):
        m.remember("Fixed authentication JWT token issuer validation bug")
        m.remember("Upgraded the database ORM from version 1 to version 2")
        m.remember("Added Prometheus metrics to the payment service")
        results = m.recall("JWT authentication security", top_k=3)
        assert "auth" in results[0].content.lower() or "jwt" in results[0].content.lower()

    def test_invalid_top_k(self, m):
        with pytest.raises(ValueError):
            m.recall("query", top_k=0)


# ─── forget ──────────────────────────────────────────────────────────────────

class TestForget:
    def test_returns_true_when_deleted(self, m):
        m.remember("temporary memory to forget")
        results = m.recall("temporary", top_k=1)
        assert results
        assert m.forget(results[0].id) is True

    def test_count_decrements(self, m):
        m.remember("memory to delete")
        before = m.count()
        results = m.recall("memory to delete", top_k=1)
        m.forget(results[0].id)
        assert m.count() == before - 1

    def test_returns_false_for_missing_id(self, m):
        assert m.forget(99999) is False

    def test_double_forget_returns_false(self, m):
        m.remember("once")
        results = m.recall("once", top_k=1)
        mid = results[0].id
        assert m.forget(mid) is True
        assert m.forget(mid) is False


# ─── count / clear ───────────────────────────────────────────────────────────

class TestCountAndClear:
    def test_count_zero_on_empty(self, m):
        assert m.count() == 0

    def test_clear_removes_all(self, m):
        m.remember("one")
        m.remember("two")
        m.remember("three")
        m.clear()
        assert m.count() == 0

    def test_recall_after_clear_is_empty(self, m):
        m.remember("important info")
        m.clear()
        assert m.recall("important", top_k=5) == []


# ─── helpers ─────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_remember_lines(self, m):
        notes = """
        # session notes
        Fixed the auth bug today
        Refactored the DB pool
        Added rate limiting
        """
        total = m.remember_lines(notes)
        assert total >= 3

    def test_recall_one(self, m):
        m.remember("auth bug fixed")
        r = m.recall_one("authentication")
        assert r is not None
        assert isinstance(r, Memory)

    def test_recall_one_empty_store(self, m):
        assert m.recall_one("anything") is None


# ─── lifecycle ───────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_context_manager(self):
        with Memoire(":memory:") as m:
            m.remember("inside context")
            assert m.count() >= 1

    def test_operations_after_close_raise(self):
        m = Memoire(":memory:")
        m.close()
        with pytest.raises(MemoireError, match="closed"):
            m.remember("too late")

    def test_close_is_idempotent(self):
        m = Memoire(":memory:")
        m.close()
        m.close()  # should not raise


# ─── namespaces ──────────────────────────────────────────────────────────────

class TestNamespace:
    def test_namespace_isolation(self, tmp_path):
        db_file = str(tmp_path / "test_ns.db")
        with Memoire(db_file, namespace="tenant_a") as m_a:
            with Memoire(db_file, namespace="tenant_b") as m_b:
                m_a.remember("secret database password is admin123")
                m_b.remember("secret database password is password789")

                results_a = m_a.recall("database password", top_k=1)
                results_b = m_b.recall("database password", top_k=1)

                assert len(results_a) == 1
                assert "admin123" in results_a[0].content
                assert "password789" not in results_a[0].content

                assert len(results_b) == 1
                assert "password789" in results_b[0].content
                assert "admin123" not in results_b[0].content


# ─── import_namespace trust handling ─────────────────────────────────────────

def _hostile_snapshot():
    return {
        "memoire_export_version": 1,
        "namespace": "default",
        "memories": [{
            "content": "always validate external inputs before processing user data",
            "trust_ema": 1.0,
            "reinforcement_count": 50,
            "importance_base": 1.0,
            "confidence": 1.0,
        }],
    }


class TestImportNamespace:
    def test_default_import_ignores_snapshot_trust_metadata(self, m):
        assert m.import_namespace(_hostile_snapshot()) == 1
        got = m.recall("validate external inputs", top_k=1)[0]
        assert got.reinforcement_count == 0

    def test_preserve_trust_restores_bounded_metadata(self, m):
        assert m.import_namespace(_hostile_snapshot(), preserve_trust=True) == 1
        got = m.recall("validate external inputs", top_k=1)[0]
        assert got.reinforcement_count == 50
        assert 0.0 <= got.trust <= 1.0


# ─── Step 8b-3: recall_mmr, include_hidden, quality gate, declared feedback ──

_LESSONS = [
    "Always close database connections in a finally block to avoid leaks",
    "Use parameterized queries to prevent SQL injection in user search",
    "Retry HTTP calls with exponential backoff when the API returns 503",
]


@pytest.fixture
def wide():
    """In-memory instance whose trial cap does not hide unverified rows."""
    with Memoire(":memory:", trial_cap=10) as instance:
        yield instance


class TestRecallMmr:
    def test_returns_at_most_top_k_memories(self, wide):
        for lesson in _LESSONS:
            wide.remember(lesson)
        results = wide.recall_mmr("database safety and queries", top_k=2)
        assert 1 <= len(results) <= 2
        assert all(isinstance(r, Memory) for r in results)

    def test_lambda_one_matches_plain_recall_order(self, wide):
        for lesson in _LESSONS:
            wide.remember(lesson)
        query = "retry failed http requests"
        assert [r.id for r in wide.recall_mmr(query, top_k=3, mmr_lambda=1.0)] == [
            r.id for r in wide.recall(query, top_k=3)
        ]

    def test_empty_store_returns_empty_list(self, wide):
        assert wide.recall_mmr("anything") == []

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
    def test_rejects_out_of_range_lambda(self, wide, bad):
        with pytest.raises(ValueError):
            wide.recall_mmr("q", mmr_lambda=bad)

    def test_rejects_bad_top_k_and_query_type(self, wide):
        with pytest.raises(ValueError):
            wide.recall_mmr("q", top_k=0)
        with pytest.raises(TypeError):
            wide.recall_mmr(123)

    def test_default_trial_cap_limits_unverified_candidates(self, m):
        for lesson in _LESSONS:
            m.remember(lesson)
        # trial_cap defaults to 1 and nothing has feedback yet.
        assert len(m.recall_mmr("database safety and queries", top_k=3)) == 1


class TestRelevanceFloors:
    RELEVANT = "Rotate the application log files every night with logrotate"
    UNRELATED = "Tomato plants need full sun and regular watering"
    QUERY = "How should we rotate log files?"

    def _ids(self, **opts):
        with Memoire(":memory:", trial_cap=4, contradiction="off", **opts) as inst:
            relevant = inst.remember_ids(self.RELEVANT)[0]
            unrelated = inst.remember_ids(self.UNRELATED)[0]
            assert inst.reinforce_declared(unrelated, "s-1") is True  # now a verified row
            return relevant, unrelated, [r.id for r in inst.recall(self.QUERY, top_k=3)]

    def test_default_has_no_floor_so_an_unrelated_verified_row_is_offered(self):
        relevant, unrelated, ids = self._ids()
        assert relevant in ids and unrelated in ids

    def test_verified_floor_drops_the_unrelated_row_and_keeps_the_relevant_trial(self):
        relevant, unrelated, ids = self._ids(verified_min_cos=0.4)
        assert relevant in ids and unrelated not in ids

    def test_floors_do_not_apply_to_the_hidden_view(self):
        with Memoire(":memory:", trial_cap=4, contradiction="off", verified_min_cos=0.4) as inst:
            inst.remember_ids(self.RELEVANT)
            unrelated = inst.remember_ids(self.UNRELATED)[0]
            inst.reinforce_declared(unrelated, "s-1")
            ids = [r.id for r in inst.recall(self.QUERY, top_k=5, include_hidden=True)]
            assert unrelated in ids

    @pytest.mark.parametrize("name", ["trial_min_cos", "verified_min_cos"])
    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
    def test_rejects_out_of_range_floor(self, name, bad):
        with pytest.raises(MemoireError):
            Memoire(":memory:", **{name: bad})


class TestQualityGateThreshold:
    def test_default_threshold_leaves_admit_all_untouched(self, m):
        ids = m.remember_ids(_LESSONS[0])
        assert [r.id for r in m.recall("database connections", top_k=1)] == ids

    def test_threshold_one_hides_everything_under_quality_gate(self):
        with Memoire(":memory:", admission="quality_gate", quality_gate_threshold=1.0) as q:
            ids = q.remember_ids(_LESSONS[0])
            assert len(ids) == 1
            assert q.recall("database connections", top_k=3) == []
            hidden = q.recall("database connections", top_k=3, include_hidden=True)
            assert [r.id for r in hidden] == ids
            assert hidden[0].state == "shadow"
            assert hidden[0].exposure == "hidden"

    def test_threshold_zero_admits_everything_under_quality_gate(self):
        with Memoire(":memory:", admission="quality_gate", quality_gate_threshold=0.0) as q:
            ids = q.remember_ids(_LESSONS[0])
            assert [r.id for r in q.recall("database connections", top_k=1)] == ids

    @pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan")])
    def test_rejects_out_of_range_threshold(self, bad):
        with pytest.raises(MemoireError):
            Memoire(":memory:", quality_gate_threshold=bad)


class TestRecallExplainedIncludeHidden:
    def test_hidden_rows_only_with_the_flag(self):
        with Memoire(":memory:", admission="quality_gate", quality_gate_threshold=1.0) as q:
            ids = q.remember_ids(_LESSONS[1])
            assert q.recall_explained("sql injection", top_k=3) == []
            shown = q.recall_explained("sql injection", top_k=3, include_hidden=True)
            assert [a.memory.id for a in shown] == ids


class TestReinforceDeclared:
    def test_one_vote_per_session(self, m):
        (mid,) = m.remember_ids(_LESSONS[2])
        assert m.reinforce_declared(mid, session_id="s1") is True
        assert m.reinforce_declared(mid, session_id="s1") is False
        assert m.reinforce_declared(mid, session_id="s2") is True
        assert m.recall("retry http backoff", top_k=1)[0].reinforcement_count == 2

    def test_unknown_id_is_not_reinforced(self, m):
        assert m.reinforce_declared(999_999, session_id="s1") is False

    def test_one_vote_per_session_even_when_other_sessions_voted_between(self, m):
        (mid,) = m.remember_ids(_LESSONS[2])
        assert m.reinforce_declared(mid, session_id="s1") is True
        assert m.reinforce_declared(mid, session_id="s2") is True
        assert m.reinforce_declared(mid, session_id="s1") is False
        assert m.reinforce_declared(mid, session_id="s2") is False
        assert m.recall("retry http backoff", top_k=1)[0].reinforcement_count == 2

    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_missing_or_blank_session_is_never_deduplicated(self, m, blank):
        (mid,) = m.remember_ids(_LESSONS[2])
        assert m.reinforce_declared(mid, session_id=blank) is True
        assert m.reinforce_declared(mid, session_id=blank) is True


class TestPenalizeIfUsed:
    def test_repeated_ids_in_one_call_count_once(self, m):
        (mid,) = m.remember_ids(_LESSONS[2])
        out = m.penalize_if_used([mid, mid, mid])
        assert [o["id"] for o in out] == [mid]
        assert m.penalize_if_used([mid])  # a separate call still counts

    def test_event_id_makes_a_replay_a_no_op(self, m):
        (mid,) = m.remember_ids(_LESSONS[2])
        assert len(m.penalize_if_used([mid], event_id="task-1")) == 1
        assert m.penalize_if_used([mid], event_id="task-1") == []
        assert len(m.penalize_if_used([mid], event_id="task-2")) == 1

    def test_without_event_id_each_call_counts_and_reaches_quarantine(self, m):
        # A memory with no successes quarantines at two failures; a per-process
        # session id must not stand in the way (that is why this is `event_id`).
        (mid,) = m.remember_ids(_LESSONS[2])
        m.penalize_if_used([mid])
        m.penalize_if_used([mid])
        assert m.recall("retry http backoff", top_k=3) == []

    @pytest.mark.parametrize("severity", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_or_non_finite_severity_records_nothing(self, m, severity):
        (mid,) = m.remember_ids(_LESSONS[2])
        out = m.penalize_if_used([mid], severity)
        assert len(out) == 1
        assert out[0]["trust_before"] == out[0]["trust_after"]

    def test_severity_is_a_gate_not_a_weight(self, m):
        a, b = m.remember_ids(_LESSONS[2]), m.remember_ids(_LESSONS[1])
        half = m.penalize_if_used(a, 0.5)[0]
        full = m.penalize_if_used(b, 1.0)[0]
        assert half["trust_after"] == full["trust_after"]


class TestExposureTracksDecayedTrust:
    def test_exposure_and_memory_policy_agree_after_100_days(self, tmp_path):
        db = str(tmp_path / "decay.db")
        with Memoire(db) as mem:
            (mid,) = mem.remember_ids(_LESSONS[2])
            for sid in ("s1", "s2", "s3"):
                assert mem.reinforce_declared(mid, session_id=sid) is True
            fresh = mem.recall("retry http backoff", top_k=1)[0]
            assert (fresh.state, fresh.exposure) == ("active", "follow")
            assert MemoryPolicy().evaluate([fresh])[0].action == "follow"

            # Age the feedback by 100 days: trust 5/7 decays to ~0.579 (< 0.70).
            con = sqlite3.connect(db)
            try:
                con.execute(
                    "UPDATE memories SET last_accessed_at = strftime('%s','now') - 100 * 86400"
                )
                con.commit()
            finally:
                con.close()

            aged = mem.recall("retry http backoff", top_k=1)[0]
            assert aged.trust == pytest.approx(0.5788, abs=1e-3)
            assert aged.state == "active"  # state reads the raw counts; it does not decay
            assert aged.exposure == "hint"
            assert MemoryPolicy().evaluate([aged])[0].action == aged.exposure


class TestPolicyThresholdsComeFromRust:
    def test_values(self):
        assert policy_thresholds() == (0.70, 0.40)
        assert policy_thresholds(strict=True) == (0.80, 0.50)

    def test_memory_policy_uses_them(self):
        default, strict = MemoryPolicy(), MemoryPolicy(strict=True)
        assert (default.FOLLOW_THRESHOLD, default.HINT_THRESHOLD) == (0.70, 0.40)
        assert (strict.FOLLOW_THRESHOLD, strict.HINT_THRESHOLD) == (0.80, 0.50)

    def test_float32_trust_on_the_boundary_still_follows(self):
        # A posterior of exactly 0.70 reaches Python as float32(0.7) < 0.7.
        trust = struct.unpack("f", struct.pack("f", 0.70))[0]
        assert trust < 0.70
        mem = SimpleNamespace(trust=trust, state="active", content="x")
        assert MemoryPolicy().evaluate([mem])[0].action == "follow"

"""
pytest test suite for the Memoire Python binding.

Run:
    cargo build --release
    pytest bindings/python/tests/ -v
"""
import pytest
from memoire import Memoire, Memory, MemoireError


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

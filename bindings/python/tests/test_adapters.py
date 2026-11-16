"""
Tests for memoire_engine.adapters.

They never open a real database or load the embedding model: the adapters'
``Memoire`` dependency is replaced with a small fake. Framework-specific tests
are skipped when that framework is not installed; the optional-import tests run
in a subprocess so they can hide the frameworks without disturbing this process.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import List

import pytest

from memoire_engine import adapters


def _memory(id_: int, content: str, trust: float, score: float = 0.9, state: str = "active"):
    return SimpleNamespace(
        id=id_, content=content, trust=trust, score=score, uncertainty=0.2, state=state
    )


class _FakeMemoire:
    """Stands in for memoire_engine.client.Memoire inside the adapters module."""

    memories: List[SimpleNamespace] = []
    opened: List[dict] = []
    closed = 0

    def __init__(self, db_path: str, namespace: str = "default") -> None:
        type(self).opened.append({"db_path": db_path, "namespace": namespace, "recalls": []})

    def __enter__(self):
        return self

    def __exit__(self, *_):
        type(self).closed += 1

    def recall(self, query: str, top_k: int = 5):
        type(self).opened[-1]["recalls"].append((query, top_k))
        return list(type(self).memories)


@pytest.fixture
def fake_memoire(monkeypatch):
    _FakeMemoire.memories = [
        _memory(1, "high trust memory", trust=0.95),
        _memory(2, "hint memory", trust=0.55, score=0.6),
        _memory(3, "ignored memory", trust=0.10, score=0.5),
    ]
    _FakeMemoire.opened = []
    _FakeMemoire.closed = 0
    monkeypatch.setattr(adapters, "Memoire", _FakeMemoire)
    return _FakeMemoire


# ─── LangChain ────────────────────────────────────────────────────────────────

class TestMemoireRetriever:
    @pytest.fixture(autouse=True)
    def _need_langchain_core(self):
        pytest.importorskip("langchain_core")

    def test_is_a_langchain_base_retriever(self):
        from langchain_core.retrievers import BaseRetriever

        assert issubclass(adapters.MemoireRetriever, BaseRetriever)
        assert isinstance(adapters.MemoireRetriever(db_path="x.db"), BaseRetriever)

    def test_invoke_applies_policy_and_maps_metadata(self, fake_memoire):
        from langchain_core.documents import Document

        docs = adapters.MemoireRetriever(db_path="x.db").invoke("billing")

        assert all(isinstance(d, Document) for d in docs)
        assert [d.page_content for d in docs] == ["high trust memory", "hint memory"]
        assert [d.metadata["action"] for d in docs] == ["follow", "hint"]
        assert docs[0].metadata == {
            "memory_id": 1,
            "score": 0.9,
            "trust": 0.95,
            "uncertainty": 0.2,
            "action": "follow",
            "state": "active",
        }

    def test_constructor_arguments_reach_memoire(self, fake_memoire):
        retriever = adapters.MemoireRetriever(
            "agent.db", namespace="tenant_a", top_k=3  # db_path stays positional
        )
        retriever.invoke("billing")

        assert fake_memoire.opened == [
            {"db_path": "agent.db", "namespace": "tenant_a", "recalls": [("billing", 3)]}
        ]
        assert fake_memoire.closed == 1, "the database must be closed after each call"

    def test_min_score_filters_before_policy(self, fake_memoire):
        docs = adapters.MemoireRetriever(db_path="x.db", min_score=0.8).invoke("q")
        assert [d.page_content for d in docs] == ["high trust memory"]

    def test_strict_policy_changes_the_decision(self, fake_memoire):
        fake_memoire.memories = [_memory(1, "borderline", trust=0.75)]
        default = adapters.MemoireRetriever(db_path="x.db").invoke("q")
        strict = adapters.MemoireRetriever(db_path="x.db", strict_policy=True).invoke("q")
        assert default[0].metadata["action"] == "follow"  # 0.75 >= 0.70
        assert strict[0].metadata["action"] == "hint"  # 0.75 < 0.80

    def test_ainvoke(self, fake_memoire):
        docs = asyncio.run(adapters.MemoireRetriever(db_path="x.db").ainvoke("q"))
        assert [d.page_content for d in docs] == ["high trust memory", "hint memory"]

    def test_composes_in_an_lcel_pipe(self, fake_memoire):
        from langchain_core.runnables import RunnableLambda

        chain = adapters.MemoireRetriever(db_path="x.db") | RunnableLambda(
            lambda docs: [d.page_content for d in docs]
        )
        assert chain.invoke("q") == ["high trust memory", "hint memory"]


# ─── LlamaIndex ───────────────────────────────────────────────────────────────

class TestMemoireIndex:
    @pytest.fixture(autouse=True)
    def _need_llama_index(self):
        pytest.importorskip("llama_index.core")

    def test_retrieve_nodes_applies_policy(self, fake_memoire):
        index = adapters.MemoireIndex(db_path="x.db", namespace="tenant_a", top_k=4)
        nodes = index.retrieve_nodes("billing")

        assert [n.node.get_content() for n in nodes] == ["high trust memory", "hint memory"]
        assert [n.node.metadata["action"] for n in nodes] == ["follow", "hint"]
        assert fake_memoire.opened[0]["namespace"] == "tenant_a"
        assert fake_memoire.opened[0]["recalls"] == [("billing", 4)]

    def test_as_retriever_returns_nodes(self, fake_memoire):
        nodes = adapters.MemoireIndex(db_path="x.db").as_retriever(top_k=2).retrieve("q")
        assert len(nodes) == 2

    def test_retriever_accepts_a_query_bundle(self, fake_memoire):
        from llama_index.core.schema import QueryBundle

        retriever = adapters.MemoireIndex(db_path="x.db").as_retriever()
        assert len(retriever.retrieve(QueryBundle("billing"))) == 2
        assert fake_memoire.opened[0]["recalls"][0][0] == "billing"

    def test_as_query_engine_passes_a_plain_string_to_recall(self, fake_memoire):
        # Regression: query engines call retrieve() with a QueryBundle and the
        # real Memoire.recall() raises TypeError for anything but str.
        from llama_index.core.llms import MockLLM

        class _StrictMemoire(_FakeMemoire):
            def recall(self, query, top_k=5):
                if not isinstance(query, str):
                    raise TypeError(f"query must be str, got {type(query).__name__}")
                return super().recall(query, top_k)

        adapters.Memoire = _StrictMemoire  # fake_memoire's monkeypatch restores it
        engine = adapters.MemoireIndex(db_path="x.db").as_query_engine(llm=MockLLM())
        response = engine.query("billing precision?")
        assert "high trust memory" in str(response)


# ─── Optional imports (subprocess so frameworks can be hidden) ────────────────

_BLOCKER = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked!r}

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("blocked for test: " + name)

    sys.meta_path.insert(0, _Block())
    """
)


def _run(blocked, body: str) -> subprocess.CompletedProcess:
    code = _BLOCKER.format(blocked=tuple(blocked)) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )


def test_package_import_pulls_in_no_framework():
    proc = _run(
        (),
        """
        import sys
        import memoire_engine
        leaked = [m for m in sys.modules if m.split(".")[0] in ("langchain", "langchain_core", "llama_index")]
        assert not leaked, leaked
        print("ok")
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_imports_and_fails_clearly_with_neither_framework():
    proc = _run(
        ("langchain", "langchain_core", "llama_index"),
        """
        import memoire_engine
        from memoire_engine import adapters

        assert adapters._HAS_LANGCHAIN_CORE is False
        for factory, needle in ((adapters.MemoireRetriever, "langchain-core"),
                                (adapters.MemoireIndex, "llama-index-core")):
            try:
                factory(db_path="x.db")
            except ImportError as exc:
                assert needle in str(exc), str(exc)
            else:
                raise SystemExit(factory.__name__ + " did not raise ImportError")
        print("ok")
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_memoire_index_works_with_only_llama_index_core():
    pytest.importorskip("llama_index.core")
    proc = _run(
        ("langchain", "langchain_core"),
        """
        from memoire_engine import adapters

        assert adapters._HAS_LANGCHAIN_CORE is False
        index = adapters.MemoireIndex(db_path="x.db")  # opens nothing until retrieval
        assert index.as_retriever(top_k=2) is not None
        print("ok")
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_memoire_retriever_works_with_only_langchain_core():
    pytest.importorskip("langchain_core")
    proc = _run(
        ("langchain", "llama_index"),
        """
        from langchain_core.retrievers import BaseRetriever
        from memoire_engine import adapters

        assert adapters._HAS_LANGCHAIN_CORE is True
        assert isinstance(adapters.MemoireRetriever(db_path="x.db"), BaseRetriever)
        print("ok")
        """,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"

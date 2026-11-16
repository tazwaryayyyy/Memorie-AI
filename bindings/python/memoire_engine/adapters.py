"""
Zero-config adapters for popular agent frameworks.

LangChain
---------
    from memoire_engine.adapters import MemoireRetriever
    retriever = MemoireRetriever(db_path="agent.db")
    docs = retriever.invoke("billing precision bug")

LlamaIndex
----------
    from memoire_engine.adapters import MemoireIndex
    index = MemoireIndex(db_path="agent.db")
    engine = index.as_query_engine()
    response = engine.query("what did we learn about billing?")

Both adapters apply MemoryPolicy internally — only FOLLOW/HINT memories
are surfaced to the framework. IGNORE-ranked memories are filtered out
before the framework sees them, so low-trust noise never reaches the LLM.

Requirements (install only what you use; neither is needed to import this module):
    pip install langchain-core     # for MemoireRetriever
    pip install llama-index-core   # for MemoireIndex
"""
from __future__ import annotations

from typing import Any, List, Optional

from .client import Memoire, Memory, MemoryPolicy

# ─── LangChain ────────────────────────────────────────────────────────────────
#
# ``langchain-core`` is optional. When it is installed, ``MemoireRetriever`` is a
# real ``langchain_core.retrievers.BaseRetriever`` (a pydantic model that is also a
# Runnable: ``invoke``/``ainvoke``, ``retriever | prompt | llm``, callbacks).
# When it is not, the class still exists so ``import memoire_engine.adapters``
# never fails, but constructing it raises ``ImportError`` with install advice.

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever as _LangChainBase

    _HAS_LANGCHAIN_CORE = True
except ImportError:  # pragma: no cover - exercised in a subprocess test
    CallbackManagerForRetrieverRun = Any  # type: ignore[assignment,misc]
    Document = Any  # type: ignore[assignment,misc]
    _LangChainBase = object  # type: ignore[assignment,misc]
    _HAS_LANGCHAIN_CORE = False


def _require_langchain_core() -> None:
    if not _HAS_LANGCHAIN_CORE:
        raise ImportError(
            "langchain-core is not installed.\n"
            "Install it:  pip install langchain-core"
        )


class MemoireRetriever(_LangChainBase):  # type: ignore[misc,valid-type]
    """
    LangChain retriever backed by Memoire.

    Subclasses ``langchain_core.retrievers.BaseRetriever`` and applies
    MemoryPolicy so only trustworthy memories (FOLLOW/HINT) reach the chain.
    Each call opens the Memoire database, recalls, and closes it again, so one
    retriever instance is safe to use from several threads.

    Usage::

        from memoire_engine.adapters import MemoireRetriever

        retriever = MemoireRetriever(db_path="agent.db", top_k=5)
        docs = retriever.invoke("how do we handle billing precision?")

        # inside a chain:  retriever | prompt | llm

    Each returned ``Document`` carries ``memory_id``, ``score``, ``trust``,
    ``uncertainty``, ``action`` and ``state`` in its metadata.
    """

    db_path: str = "./memoire.db"
    namespace: str = "default"
    top_k: int = 5
    min_score: float = 0.0
    strict_policy: bool = False

    def __init__(
        self,
        db_path: str = "./memoire.db",
        namespace: str = "default",
        top_k: int = 5,
        min_score: float = 0.0,
        strict_policy: bool = False,
        **kwargs: Any,
    ) -> None:
        _require_langchain_core()
        # BaseRetriever is a pydantic model: fields must be passed by keyword.
        super().__init__(
            db_path=db_path,
            namespace=namespace,
            top_k=top_k,
            min_score=min_score,
            strict_policy=strict_policy,
            **kwargs,
        )

    # ``run_manager`` must stay in this signature: BaseRetriever inspects it to
    # decide whether to pass the callback manager through.
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        policy = MemoryPolicy(strict=self.strict_policy)
        with Memoire(self.db_path, namespace=self.namespace) as m:
            memories: List[Memory] = m.recall(query, top_k=self.top_k)
        if self.min_score > 0.0:
            memories = [mem for mem in memories if mem.score >= self.min_score]

        return [
            Document(
                page_content=d.memory.content,
                metadata={
                    "memory_id": d.memory.id,
                    "score": d.memory.score,
                    "trust": d.memory.trust,
                    "uncertainty": d.memory.uncertainty,
                    "action": d.action,
                    "state": d.memory.state,
                },
            )
            for d in policy.evaluate(memories)
            if d.action in ("follow", "hint")
        ]

    # Async: BaseRetriever's default ``_aget_relevant_documents`` runs the method
    # above in an executor thread, which is correct for this blocking native call.


# ─── LlamaIndex ───────────────────────────────────────────────────────────────

def _require_llama_index() -> Any:
    try:
        # ``llama_index`` alone is a namespace package, so import the real module.
        import llama_index.core  # noqa: F401
        return llama_index.core
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex is not installed.\n"
            "Install it:  pip install llama-index-core"
        ) from exc


class MemoireIndex:
    """
    LlamaIndex-compatible index backed by Memoire.

    Wraps a Memoire database as a LlamaIndex ``BaseIndex``-compatible object,
    so it can be used anywhere LlamaIndex expects an index — ``as_retriever()``,
    ``as_query_engine()``, sub-question decomposition, etc.

    MemoryPolicy is applied at retrieval time. IGNORE-ranked memories are
    excluded before LlamaIndex processes the node list.

    Usage::

        from memoire_engine.adapters import MemoireIndex
        from llama_index.llms.openai import OpenAI

        index = MemoireIndex(db_path="agent.db")
        engine = index.as_query_engine(llm=OpenAI())
        response = engine.query("what patterns caused billing regressions?")

    Or as a retriever inside a query pipeline::

        retriever = index.as_retriever(top_k=5)
        nodes = retriever.retrieve("billing precision bug")
    """

    def __init__(
        self,
        db_path: str = "./memoire.db",
        namespace: str = "default",
        top_k: int = 5,
        strict_policy: bool = False,
    ) -> None:
        _require_llama_index()
        self._db_path = db_path
        self._namespace = namespace
        self._top_k = top_k
        self._policy = MemoryPolicy(strict=strict_policy)

    def retrieve_nodes(self, query: str, top_k: Optional[int] = None) -> list:
        """Return LlamaIndex ``NodeWithScore`` objects, policy-filtered."""
        from llama_index.core.schema import (  # type: ignore[import-not-found]  # pylint: disable=import-error
            NodeWithScore,
            TextNode,
        )

        k = top_k or self._top_k
        with Memoire(self._db_path, namespace=self._namespace) as m:
            memories: List[Memory] = m.recall(query, top_k=k)
            decisions = self._policy.evaluate(memories)
            nodes = []
            for d in decisions:
                if d.action not in ("follow", "hint"):
                    continue
                node = TextNode(
                    text=d.memory.content,
                    metadata={
                        "memory_id": d.memory.id,
                        "trust": d.memory.trust,
                        "uncertainty": d.memory.uncertainty,
                        "action": d.action,
                        "state": d.memory.state,
                    },
                )
                nodes.append(NodeWithScore(node=node, score=d.memory.score))
            return nodes

    def as_retriever(self, top_k: Optional[int] = None) -> "_MemoireRetrieverWrapper":
        """Return a LlamaIndex-compatible retriever object."""
        return _MemoireRetrieverWrapper(self, top_k=top_k or self._top_k)

    def as_query_engine(self, llm: Any = None, **kwargs: Any) -> Any:
        """
        Return a LlamaIndex ``RetrieverQueryEngine`` using this index.

        Pass ``llm=`` to override the default LLM. Extra kwargs are forwarded
        to ``RetrieverQueryEngine.from_args()``.
        """
        from llama_index.core.query_engine import RetrieverQueryEngine  # type: ignore[import-not-found][import-not-found]  # pylint: disable=import-error

        retriever = self.as_retriever()
        build_kwargs: dict = {}
        if llm is not None:
            build_kwargs["llm"] = llm
        build_kwargs.update(kwargs)
        return RetrieverQueryEngine.from_args(retriever, **build_kwargs)


class _MemoireRetrieverWrapper:
    """
    Thin LlamaIndex ``BaseRetriever``-compatible wrapper around ``MemoireIndex``.
    Not intended for direct instantiation — use ``MemoireIndex.as_retriever()``.
    """

    def __init__(self, index: MemoireIndex, top_k: int = 5) -> None:
        self._index = index
        self._top_k = top_k

    def retrieve(self, query: Any) -> list:
        # LlamaIndex query engines call retrieve() with a ``QueryBundle``, but
        # Memoire.recall() only accepts ``str``.
        query_str = query if isinstance(query, str) else query.query_str
        return self._index.retrieve_nodes(query_str, top_k=self._top_k)

    async def aretrieve(self, query: Any) -> list:
        return self.retrieve(query)

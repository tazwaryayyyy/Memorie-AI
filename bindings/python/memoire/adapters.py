"""
Zero-config adapters for popular agent frameworks.

LangChain
---------
    from memoire.adapters import MemoireRetriever
    retriever = MemoireRetriever(db_path="agent.db")
    docs = retriever.get_relevant_documents("billing precision bug")

LlamaIndex
----------
    from memoire.adapters import MemoireIndex
    index = MemoireIndex(db_path="agent.db")
    engine = index.as_query_engine()
    response = engine.query("what did we learn about billing?")

Both adapters apply MemoryPolicy internally — only FOLLOW/HINT memories
are surfaced to the framework. IGNORE-ranked memories are filtered out
before the framework sees them, so low-trust noise never reaches the LLM.

Requirements (install only what you use):
    pip install langchain          # for MemoireRetriever
    pip install llama-index-core   # for MemoireIndex
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from .client import Memoire, Memory, MemoryPolicy

if TYPE_CHECKING:
    # Avoid hard import at module load time — adapters are opt-in.
    pass


# ─── LangChain ────────────────────────────────────────────────────────────────

def _require_langchain() -> Any:
    try:
        import langchain  # noqa: F401
        return langchain
    except ImportError as exc:
        raise ImportError(
            "LangChain is not installed.\n"
            "Install it:  pip install langchain langchain-core"
        ) from exc


class MemoireRetriever:
    """
    LangChain-compatible retriever backed by Memoire.

    Subclasses ``langchain_core.retrievers.BaseRetriever`` and applies
    MemoryPolicy so only trustworthy memories reach the chain.

    Usage::

        from memoire.adapters import MemoireRetriever
        from langchain.chains import RetrievalQA
        from langchain_openai import ChatOpenAI

        retriever = MemoireRetriever(db_path="agent.db", top_k=5)
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            retriever=retriever,
        )
        answer = qa.invoke({"query": "how do we handle billing precision?"})
    """

    def __init__(
        self,
        db_path: str = "./memoire.db",
        top_k: int = 5,
        min_score: float = 0.0,
        strict_policy: bool = False,
    ) -> None:
        _require_langchain()
        self._db_path = db_path
        self._top_k = top_k
        self._min_score = min_score
        self._policy = MemoryPolicy(strict=strict_policy)

    # langchain_core.retrievers.BaseRetriever interface
    # type: ignore[override]
    def get_relevant_documents(self, query: str) -> list:
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.schema import Document  # type: ignore[no-redef]

        with Memoire(self._db_path) as m:
            memories: List[Memory] = m.recall(query, top_k=self._top_k)
            if self._min_score > 0.0:
                memories = [
                    mem for mem in memories if mem.score >= self._min_score]

            decisions = self._policy.evaluate(memories)
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
                for d in decisions
                if d.action in ("follow", "hint")
            ]

    # Async variant (LangChain LCEL compatibility)
    # type: ignore[override]
    async def aget_relevant_documents(self, query: str) -> list:
        return self.get_relevant_documents(query)

    # Allow use as a callable in LCEL pipes: retriever | prompt | llm
    def __call__(self, query: str) -> list:
        return self.get_relevant_documents(query)


# ─── LlamaIndex ───────────────────────────────────────────────────────────────

def _require_llama_index() -> Any:
    try:
        import llama_index  # noqa: F401
        return llama_index
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

        from memoire.adapters import MemoireIndex
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
        top_k: int = 5,
        strict_policy: bool = False,
    ) -> None:
        _require_llama_index()
        self._db_path = db_path
        self._top_k = top_k
        self._policy = MemoryPolicy(strict=strict_policy)

    def retrieve_nodes(self, query: str, top_k: Optional[int] = None) -> list:
        """Return LlamaIndex ``NodeWithScore`` objects, policy-filtered."""
        from llama_index.core.schema import (  # type: ignore[import-not-found]  # pylint: disable=import-error
            NodeWithScore,
            TextNode,
        )

        k = top_k or self._top_k
        with Memoire(self._db_path) as m:
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

    def retrieve(self, query: str) -> list:
        return self._index.retrieve_nodes(query, top_k=self._top_k)

    async def aretrieve(self, query: str) -> list:
        return self.retrieve(query)

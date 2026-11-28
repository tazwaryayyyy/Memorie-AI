"""
memoire_engine — Python bindings for the Memoire semantic memory engine.

Install from PyPI (prebuilt wheels for Linux x86_64, macOS arm64 and Windows
x86_64 only; there is no sdist):
    pip install memoire-engine

Other platforms, or development: install from a source checkout (needs a Rust
toolchain), inside a virtualenv:
    cd bindings/python && maturin develop --release

The import name is ``memoire_engine``. The unrelated PyPI project ``memoire`` is
a different package.

Usage:
    from memoire_engine import Memoire

    with Memoire("agent.db") as m:
        m.remember("Fixed the JWT issuer validation bug in auth middleware")
        results = m.recall("authentication security issues", top_k=3)
        for r in results:
            print(f"[{r.score:.3f}] {r.content}")
"""

from .client import (
    Memoire,
    Memory,
    MemoireError,
    MemoryPolicy,
    PolicyDecision,
    policy_thresholds,
)

__all__ = [
    "Memoire",
    "Memory",
    "MemoireError",
    "MemoryPolicy",
    "PolicyDecision",
    "policy_thresholds",
    # Framework adapters (opt-in — requires langchain-core or llama-index-core)
    # from memoire_engine.adapters import MemoireRetriever, MemoireIndex
]
__version__ = "0.1.0"

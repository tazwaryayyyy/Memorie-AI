"""
memoire — Python bindings for the Memoire semantic memory engine.

Install:
    pip install memoire   (once published to PyPI)

Or use directly from the source tree — just build the Rust library first:
    cargo build --release

Usage:
    from memoire import Memoire

    with Memoire("agent.db") as m:
        m.remember("Fixed the JWT issuer validation bug in auth middleware")
        results = m.recall("authentication security issues", top_k=3)
        for r in results:
            print(f"[{r.score:.3f}] {r.content}")
"""

from .client import Memoire, Memory, MemoireError

__all__ = ["Memoire", "Memory", "MemoireError"]
__version__ = "0.1.0"

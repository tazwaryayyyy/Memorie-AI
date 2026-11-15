# memoire-engine (Python)

Python bindings for the Memoire semantic memory engine, built as a PyO3
extension with maturin. Import name: `memoire_engine`.

The PyPI distribution name `memoire-engine` is **not published yet**. The
unrelated PyPI project `memoire` is a different package; do not install it
expecting this one.

Install from a source checkout (needs a Rust toolchain), inside a virtualenv:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
cd bindings/python
maturin develop --release
```

```python
from memoire_engine import Memoire

with Memoire("agent.db") as m:
    m.remember("Fixed the JWT issuer validation bug in auth middleware")
    for r in m.recall("authentication security issues", top_k=3):
        print(f"[{r.score:.3f}] {r.content}")
```

The first real `remember()`/`recall()` downloads the embedding model (about
23 MB) unless it is already in the Hugging Face cache.

`import_namespace(snapshot)` ignores trust metadata in the snapshot by default.
Pass `preserve_trust=True` only to restore a snapshot you exported yourself.

Tests: `cd bindings/python && pytest tests/` (needs the extension built as above).
`MEMOIRE_LIB` is not used by this package. See `docs/RELEASE.md` for releases.

# memoire-engine (Python)

Python bindings for the Memoire semantic memory engine, built as a PyO3
extension with maturin. Import name: `memoire_engine`.

`memoire-engine` `0.1.0` is on PyPI as prebuilt `cp310-abi3` wheels for Linux `x86_64`
(`manylinux_2_28`), macOS `arm64` and Windows `x86_64` (Python 3.10+). There is no sdist. The
import name is `memoire_engine`; the unrelated PyPI project `memoire` is a different package, so
do not install it expecting this one.

```bash
pip install memoire-engine
```

On any other platform (for example Intel macOS, Linux `aarch64` or musl), or to develop, build
from a source checkout (needs a Rust toolchain; targets without a prebuilt ONNX Runtime also need
`ORT_LIB_LOCATION`, see `docs/RELEASE.md`), inside a virtualenv:

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
87 MiB) unless it is already in the Hugging Face cache.

`import_namespace(snapshot)` ignores trust metadata in the snapshot by default.
Pass `preserve_trust=True` only to restore a snapshot you exported yourself.

Tests: `cd bindings/python && pytest tests/` (needs the extension built as above).
`MEMOIRE_LIB` is not used by this package. See `docs/RELEASE.md` for releases.

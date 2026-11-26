#!/usr/bin/env python3
"""Smoke test for a *wheel-installed* memoire-engine + memoire-mcp. Stdlib only.

Run it inside a fresh virtualenv, from a directory that is NOT the repository checkout:
    python smoke_installed.py
It exercises the native extension (statically linked ONNX Runtime) with a real embedding, so the
first run downloads the ~87 MiB model unless FASTEMBED_CACHE_DIR / HF_HOME already holds it.
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util as iu
import os
import pathlib
import shutil
import sys
import tempfile
import time


def main() -> int:
    import memoire_engine

    where = pathlib.Path(memoire_engine.__file__).resolve()
    print("memoire_engine from:", where)
    assert "site-packages" in where.parts, f"imported from a source tree, not an installed wheel: {where}"
    ver = md.version("memoire-engine")
    assert memoire_engine.__version__ == ver, (memoire_engine.__version__, ver)
    import memoire_engine.memoire_engine  # noqa: F401  (compiled extension must load)
    print("memoire-engine", ver, "| memoire-mcp", md.version("memoire-mcp"))

    # No generic top-level modules, and the old import name is gone.
    assert iu.find_spec("server") is None and iu.find_spec("memoire") is None

    import memoire_mcp.server as s

    health = s.startup_health_check()
    print("mcp health:", health)
    assert health["ok"], health
    assert shutil.which("memoire-mcp"), "console script memoire-mcp is not on PATH"

    from memoire_engine import Memoire

    text = "Never use float for money. Use Decimal for billing calculations."
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ.setdefault("FASTEMBED_CACHE_DIR", os.path.join(tmp, "fastembed_cache"))
        db = os.path.join(tmp, "smoke.db")
        last: Exception | None = None
        for attempt in range(3):  # first call may download the model
            try:
                with Memoire(db) as m:
                    ids = m.remember_ids(text)
                    assert ids, "remember_ids returned no ids"
                    got = m.recall("billing precision", top_k=3)
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                print(f"attempt {attempt + 1} failed: {exc!r}")
                time.sleep(10)
        else:
            raise SystemExit(f"remember/recall failed 3 times: {last!r}")
    assert got, "recall returned nothing"
    assert "Decimal" in got[0].content, got[0].content
    print("recall ok:", got[0].content, "| state:", getattr(got[0], "state", "?"))
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

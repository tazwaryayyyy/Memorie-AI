"""
pytest configuration for the Memoire Python test suite.
Ensures the compiled library can be found before any test runs.
"""
import os
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    """Set MEMOIRE_LIB if not already set, searching for the release build."""
    if os.environ.get("MEMOIRE_LIB"):
        return  # already set — trust the caller

    repo_root = Path(__file__).parent.parent.parent.parent
    candidates = {
        "linux":  repo_root / "target" / "release" / "libmemoire.so",
        "darwin": repo_root / "target" / "release" / "libmemoire.dylib",
        "win32":  repo_root / "target" / "release" / "memoire.dll",
    }
    lib = candidates.get(sys.platform)
    if lib and lib.exists():
        os.environ["MEMOIRE_LIB"] = str(lib)
    else:
        pytest.exit(
            "\nCould not find compiled libmemoire.\n"
            "Run:  cargo build --release\n"
            "Then: pytest bindings/python/tests/\n",
            returncode=1,
        )

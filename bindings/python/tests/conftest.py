"""
pytest configuration for the Memoire Python binding tests.

The tests need the compiled PyO3 extension (memoire_engine.memoire_engine).
Build it into the active virtualenv first:

    cd bindings/python && maturin develop --release
"""
import pytest


def pytest_configure(config):
    """Fail fast with build instructions if the native extension is missing."""
    from memoire_engine import client

    if client._NativeMemoire is None:
        pytest.exit(
            "\nThe native memoire_engine extension is not importable "
            f"({client._NATIVE_IMPORT_ERROR!r}).\n"
            "Run, inside your virtualenv:  cd bindings/python && maturin develop --release\n"
            "Then:                         pytest tests/\n",
            returncode=1,
        )

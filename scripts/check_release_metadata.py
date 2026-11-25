#!/usr/bin/env python3
"""Static release-metadata consistency checks. Stdlib only, Python >= 3.11.

Run from anywhere:  python scripts/check_release_metadata.py [--tag vX.Y.Z]

Checks (no network, no build):
  * Cargo.toml, bindings/python/pyproject.toml, mcp-server/pyproject.toml share one version
  * memoire_engine.__version__ matches, when it is a plain string literal
  * the memoire-engine pin in mcp-server/pyproject.toml admits that version
  * mcp-server/server.json agrees with the memoire-mcp package (name, version, PyPI identifier,
    description length) and the README carries the registry ownership marker
  * LICENSE / THIRD_PARTY_NOTICES.md copies shipped inside the Python packages are identical
  * with --tag: the tag is exactly "v" + that version
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MCP_NAME = "io.github.tazwaryayyyy/memoire-mcp"

errors: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def toml(rel: str) -> dict:
    return tomllib.loads((ROOT / rel).read_text(encoding="utf-8"))


def vtuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="git tag being released, e.g. v0.1.0")
    args = ap.parse_args()

    cargo = toml("Cargo.toml")["package"]["version"]
    engine = toml("bindings/python/pyproject.toml")["project"]["version"]
    mcp_proj = toml("mcp-server/pyproject.toml")["project"]
    mcp = mcp_proj["version"]
    versions = {"Cargo.toml": cargo, "bindings/python/pyproject.toml": engine,
                "mcp-server/pyproject.toml": mcp}
    if len(set(versions.values())) != 1:
        fail(f"version mismatch: {versions}")
    version = cargo
    print(f"version: {version}")

    init = ROOT / "bindings/python/memoire_engine/__init__.py"
    if init.is_file():
        m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init.read_text(encoding="utf-8"), re.M)
        if m is None:
            print("NOTE: __version__ is not a string literal in __init__.py; "
                  "the installed-wheel smoke test compares it with package metadata instead.")
        elif m.group(1) != version:
            fail(f"memoire_engine.__version__ is {m.group(1)!r}, expected {version!r}")

    pin = next((d for d in mcp_proj.get("dependencies", []) if d.replace("_", "-").startswith("memoire-engine")), None)
    if pin is None:
        fail("mcp-server/pyproject.toml has no memoire-engine dependency")
    else:
        m = re.search(r">=\s*(\d+\.\d+\.\d+)\s*,\s*<\s*(\d+\.\d+\.\d+)", pin)
        if m is None:
            fail(f"cannot parse memoire-engine pin {pin!r} (expected '>=X.Y.Z,<A.B.C')")
        elif not (vtuple(m.group(1)) <= vtuple(engine) < vtuple(m.group(2))):
            fail(f"memoire-engine pin {pin!r} does not admit engine version {engine}")

    sj_path = ROOT / "mcp-server/server.json"
    if not sj_path.is_file():
        fail("mcp-server/server.json is missing")
    else:
        sj = json.loads(sj_path.read_text(encoding="utf-8"))
        if sj.get("name") != MCP_NAME:
            fail(f"server.json name {sj.get('name')!r} != {MCP_NAME!r}")
        if len(sj.get("description", "")) > 100 or not sj.get("description"):
            fail("server.json description must be 1..100 characters")
        if sj.get("version") != mcp:
            fail(f"server.json version {sj.get('version')!r} != memoire-mcp version {mcp!r}")
        pk = [p for p in sj.get("packages", []) if p.get("registryType") == "pypi"]
        if len(pk) != 1:
            fail("server.json needs exactly one pypi package")
        else:
            if pk[0].get("identifier") != mcp_proj["name"]:
                fail(f"server.json pypi identifier {pk[0].get('identifier')!r} != {mcp_proj['name']!r}")
            if pk[0].get("version") != mcp:
                fail(f"server.json package version {pk[0].get('version')!r} != {mcp!r}")
            if pk[0].get("transport", {}).get("type") != "stdio":
                fail("server.json package transport must be stdio")
    readme = (ROOT / "mcp-server/README.md").read_text(encoding="utf-8")
    if f"mcp-name: {MCP_NAME}" not in readme:
        fail(f"mcp-server/README.md lacks the marker 'mcp-name: {MCP_NAME}'")

    for a, b in [("LICENSE", "bindings/python/LICENSE"),
                 ("LICENSE", "mcp-server/LICENSE"),
                 ("THIRD_PARTY_NOTICES.md", "bindings/python/memoire_engine/THIRD_PARTY_NOTICES.md")]:
        pa, pb = ROOT / a, ROOT / b
        if not pa.is_file() or not pb.is_file():
            fail(f"missing {a if not pa.is_file() else b}")
        elif pa.read_bytes().replace(b"\r\n", b"\n") != pb.read_bytes().replace(b"\r\n", b"\n"):
            fail(f"{b} differs from {a} (copy it again)")

    if args.tag is not None and args.tag != f"v{version}":
        fail(f"tag {args.tag!r} != 'v{version}'")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("release metadata OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

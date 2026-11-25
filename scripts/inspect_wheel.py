#!/usr/bin/env python3
"""Inspect built wheels before they are uploaded anywhere. Stdlib only.

  python scripts/inspect_wheel.py dist/*.whl --max-mb 90 --require LICENSE
         [--require THIRD_PARTY_NOTICES.md] [--metadata-contains "mcp-name: ..."]

Globs are expanded here so the same command works in PowerShell. Fails when a size limit is
exceeded, a --require name is absent from the archive, METADATA lacks a required string, or a
generic top-level module (server, memoire, tests, ...) would be installed.
"""
from __future__ import annotations

import argparse
import glob
import sys
import zipfile
from pathlib import PurePosixPath

NATIVE = (".so", ".pyd", ".dll", ".dylib")
FORBIDDEN_TOP = {"server", "memoire", "tests", "test", "eval", "scripts", "docs", "src"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wheels", nargs="+")
    ap.add_argument("--max-mb", type=float, default=90.0, help="PyPI's default limit is 100 MB")
    ap.add_argument("--require", action="append", default=[], help="archive member basename")
    ap.add_argument("--metadata-contains", action="append", default=[])
    args = ap.parse_args()

    paths = sorted({p for pat in args.wheels for p in glob.glob(pat)})
    if not paths:
        print("no wheels matched", args.wheels)
        return 1
    bad = False
    for path in paths:
        size_mb = __import__("os").path.getsize(path) / 1e6
        print(f"\n== {path}  ({size_mb:.1f} MB)")
        if size_mb > args.max_mb:
            print(f"FAIL: larger than {args.max_mb} MB")
            bad = True
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            tops = {PurePosixPath(n).parts[0] for n in names if "/" in n}
            for n in names:
                if n.endswith(NATIVE):
                    print(f"  native: {n}  ({z.getinfo(n).file_size / 1e6:.1f} MB)")
            libs = [t for t in tops if t.endswith(".libs")]
            for t in libs:
                print(f"  WARNING: bundled shared libraries in {t}/: list them in THIRD_PARTY_NOTICES.md")
            clash = tops & FORBIDDEN_TOP
            if clash:
                print(f"FAIL: generic top-level package(s) would be installed: {sorted(clash)}")
                bad = True
            base = {PurePosixPath(n).name for n in names}
            for req in args.require:
                if req not in base:
                    print(f"FAIL: {req} not found in wheel")
                    bad = True
                else:
                    print(f"  ok: {req}")
            meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
            text = z.read(meta).decode("utf-8", "replace") if meta else ""
            for s in args.metadata_contains:
                if s not in text:
                    print(f"FAIL: METADATA lacks {s!r}")
                    bad = True
                else:
                    print(f"  ok: METADATA contains {s!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

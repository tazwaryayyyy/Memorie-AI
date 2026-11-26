#!/usr/bin/env python3
"""Regenerate the Rust dependency inventory in THIRD_PARTY_NOTICES.md. Stdlib only.

    python scripts/generate_third_party_notices.py                  # rewrite both copies
    python scripts/generate_third_party_notices.py --check          # exit 1 if either is stale
    python scripts/generate_third_party_notices.py --metadata-json F  # use saved `cargo metadata` output

Runs `cargo metadata --locked --all-features --format-version 1`. `--all-features` matters: the
`pyo3` feature (used to build the Python wheel), `wasm` and `native` are all optional, and the
default feature set alone leaves out the whole PyO3 dependency family.

Only the text between the BEGIN/END markers is rewritten. The two copies (repository root and
bindings/python/memoire_engine/, which ships inside the wheel) are always written together.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["THIRD_PARTY_NOTICES.md", "bindings/python/memoire_engine/THIRD_PARTY_NOTICES.md"]
BEGIN = "<!-- BEGIN GENERATED INVENTORY (python scripts/generate_third_party_notices.py) -->"
END = "<!-- END GENERATED INVENTORY -->"
NO_SPDX = "No SPDX expression declared (see the crate's license file)"


def normalize(expr: str) -> str:
    """Equivalent spellings share a group: 'MIT/Apache-2.0' == 'Apache-2.0 OR MIT'."""
    e = re.sub(r"\s*/\s*", " OR ", expr.strip())
    if " AND " in e or "(" in e:
        return e
    return " OR ".join(sorted(p.strip() for p in e.split(" OR ")))


def load_packages(metadata_json: str | None) -> list[dict]:
    if metadata_json:
        data = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    else:
        out = subprocess.run(
            ["cargo", "metadata", "--locked", "--all-features", "--format-version", "1"],
            cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
        ).stdout
        data = json.loads(out)
    # Registry/git packages only: the workspace's own crate has no `source`.
    return [p for p in data["packages"] if p.get("source")]


def render(pkgs: list[dict]) -> str:
    groups: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    for p in pkgs:
        groups[normalize(p["license"]) if p.get("license") else NO_SPDX].add((p["name"], p["version"]))
    total = sum(len(v) for v in groups.values())
    lines = [BEGIN, "",
             f"{total} packages in {len(groups)} license groups, from `cargo metadata --locked "
             "--all-features` (the resolved `Cargo.lock`). Grouped by license expression; equivalent "
             "spellings (`MIT/Apache-2.0`, `Apache-2.0 OR MIT`) share a group.", ""]
    for expr, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines += [f"### {expr} ({len(members)})", ""]
        lines += [f"- `{n} {v}`" for n, v in sorted(members)]
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    nl = "\r\n" if "\r\n" in text else "\n"
    t = text.replace("\r\n", "\n")
    if t.count(BEGIN) != 1 or t.count(END) != 1 or t.index(BEGIN) > t.index(END):
        raise SystemExit("inventory markers missing or duplicated in notices file")
    a, b = t.index(BEGIN), t.index(END) + len(END)
    return (t[:a] + block + t[b:]).replace("\n", nl)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--metadata-json")
    args = ap.parse_args()
    pkgs = load_packages(args.metadata_json)
    missing = [f"{p['name']} {p['version']}" for p in pkgs if not p.get("license")]
    if missing:
        print("WARNING: no SPDX license expression declared for:", ", ".join(sorted(missing)))
    block = render(pkgs)
    stale = False
    for rel in TARGETS:
        path = ROOT / rel
        old = path.read_bytes().decode("utf-8")
        new = splice(old, block)
        if new.replace("\r\n", "\n") == old.replace("\r\n", "\n"):
            print(f"up to date: {rel}")
        elif args.check:
            print(f"STALE: {rel}")
            stale = True
        else:
            path.write_bytes(new.encode("utf-8"))
            print(f"rewrote:   {rel}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())

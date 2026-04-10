#!/usr/bin/env python3
"""
Parse Criterion benchmark results and emit a Markdown table.

Usage:
    cargo bench
    python scripts/benchmark_report.py > docs/BENCHMARKS.md
"""

import json
import sys
from pathlib import Path


def parse_criterion_dir(criterion_dir: Path) -> list[dict]:
    rows = []
    for estimates_file in sorted(criterion_dir.rglob("estimates.json")):
        # Path: target/criterion/<group>/<bench>/new/estimates.json
        parts   = estimates_file.parts
        try:
            group = parts[-4]
            bench = parts[-3]
        except IndexError:
            continue

        with open(estimates_file) as f:
            data = json.load(f)

        mean_ns  = data.get("mean", {}).get("point_estimate", 0)
        std_ns   = data.get("std_dev", {}).get("point_estimate", 0)

        rows.append({
            "group": group,
            "bench": bench,
            "mean_us": mean_ns / 1_000,
            "std_us":  std_ns  / 1_000,
        })

    return rows


def fmt(us: float) -> str:
    if us >= 1_000_000:
        return f"{us/1_000_000:.2f} s"
    if us >= 1_000:
        return f"{us/1_000:.2f} ms"
    return f"{us:.1f} µs"


def main():
    criterion_dir = Path("target") / "criterion"
    if not criterion_dir.exists():
        print("No criterion output found. Run: cargo bench", file=sys.stderr)
        sys.exit(1)

    rows = parse_criterion_dir(criterion_dir)
    if not rows:
        print("No benchmark data found.", file=sys.stderr)
        sys.exit(1)

    current_group = None
    print("# Memoire Benchmark Results\n")
    print("Generated from `cargo bench`. All timings are wall-clock means.\n")
    print("| Benchmark | Parameter | Mean | Std Dev |")
    print("|-----------|-----------|------|---------|")

    for r in rows:
        if r["group"] != current_group:
            current_group = r["group"]

        print(f"| {r['group']} | {r['bench']} | {fmt(r['mean_us'])} | ±{fmt(r['std_us'])} |")

    print()
    print("> *CPU: measured on a mid-range laptop. YMMV.*")
    print("> *`remember()` time is dominated by the ONNX embedding inference.*")
    print("> *`recall()` is a full cosine scan — add HNSW for 100k+ memories.*")


if __name__ == "__main__":
    main()

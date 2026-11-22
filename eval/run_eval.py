"""CLI:  python -m eval.run_eval {run,summarize,diagnose} ...   (see eval/README.md)"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

from .arms import ALL_ARMS, ARM_A, normalize_arm, uses_memory
from .diagnostics import exemplar_cosines
from .llm import DEFAULT_BASE_URL, MIN_RECOMMENDED_MAX_TOKENS, LLMError, OpenAICompatClient
from .prompt import build_messages
from .runner import (BudgetExhausted, FingerprintMismatch, ResultsFile, RunSettings, run_sequence,
                     select_families)
from .tasks import load_split

ARM_A_RANGE = (0.1, 0.7)


def _csv(value: Optional[str]) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _settings(args, families) -> RunSettings:
    sessions = tuple(int(s) for s in _csv(args.sessions)) or (1, 2, 3)
    return RunSettings(
        split=args.split, families=tuple(f.name for f in families), sessions=sessions, model=args.model,
        base_url=args.base_url, temperature=args.temperature, top_k=args.top_k, trial_cap=args.trial_cap,
        contradiction=args.contradiction, max_tokens=args.max_tokens, send_api_seed=args.send_api_seed,
    )


def cmd_run(args) -> int:
    if args.model.endswith("-latest") or ":latest" in args.model:
        print("error: pin an explicit model id; floating '-latest' tags are not reproducible", file=sys.stderr)
        return 2
    if args.split == "test" and not args.allow_test_split:
        print("error: the held-out test split is locked; pass --allow-test-split only after dev "
              "calibration is frozen", file=sys.stderr)
        return 2
    if args.max_tokens < MIN_RECOMMENDED_MAX_TOKENS:
        print(f"warning: --max-tokens {args.max_tokens} < {MIN_RECOMMENDED_MAX_TOKENS}; thinking models may "
              "truncate the code block (check finish_reason=length in `summarize`)", file=sys.stderr)
    try:
        arms = [normalize_arm(a) for a in _csv(args.arms)]
        families = select_families(load_split(args.split), _csv(args.families))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    seeds = [int(s) for s in _csv(args.seeds)]
    settings = _settings(args, families)

    if args.dry_run:
        return _dry_run(settings, arms, seeds, families, args)

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"error: environment variable {args.api_key_env} is not set", file=sys.stderr)
        return 2
    diag = None
    if args.diagnostics_json:
        with open(args.diagnostics_json, encoding="utf-8") as fh:
            diag = json.load(fh)
    try:
        results = ResultsFile(args.out, settings.fingerprint(families))
    except (FingerprintMismatch, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    llm = OpenAICompatClient(args.base_url, args.model, api_key, temperature=args.temperature,
                             max_tokens=args.max_tokens, rpm=args.rpm, send_api_seed=args.send_api_seed)

    from .lesson_store import make_memoire

    def factory():
        return make_memoire(contradiction=args.contradiction, trial_cap=args.trial_cap)

    try:
        for seed in seeds:
            for arm in arms:
                stats = run_sequence(settings=settings, arm=arm, seed=seed, families=families, llm=llm,
                                     results=results, store_factory=factory, diag=diag,
                                     max_calls=args.max_calls)
                print(f"== {arm} seed={seed}: ran {stats['ran']}, skipped {stats['skipped']}, "
                      f"passed(new) {stats['passed']}")
    except BudgetExhausted as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 4
    except (LLMError, FingerprintMismatch) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"results: {args.out}   (HTTP requests this process: {llm.total_requests})")
    return 0


def _dry_run(settings, arms, seeds, families, args) -> int:
    results = ResultsFile(args.out, settings.fingerprint(families)) if os.path.exists(args.out) else None
    sizes: List[int] = []
    planned = skipped = 0
    for arm in arms:
        for seed in seeds:
            for fam in families:
                for task in fam.tasks:
                    if task.session not in settings.sessions:
                        continue
                    if results and results.has((arm, fam.name, task.session, seed)):
                        skipped += 1
                        continue
                    planned += 1
                    notes = None if not uses_memory(arm) else [l.text for l in fam.bank_lessons][: settings.top_k]
                    sizes.append(sum(len(m["content"]) for m in build_messages(task, notes)) // 4)
    avg = sum(sizes) / len(sizes) if sizes else 0
    print(f"planned calls: {planned} (already recorded: {skipped}); est. avg input tokens/call: {avg:.0f} "
          "(chars/4, rough)")
    if avg:
        print(f"TPM-bound RPM ceiling for a 14K-effective-TPM model: {int(14000 // avg)}; you asked for --rpm {args.rpm}")
    if args.max_calls and planned > args.max_calls:
        print(f"note: --max-calls {args.max_calls} < planned {planned}; the run will stop and be resumable")
    return 0


def _load_records(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def cmd_summarize(args) -> int:
    records = _load_records(args.results)
    if not records:
        print("no records")
        return 1
    by_arm: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)
    for arm in [a for a in ALL_ARMS if a in by_arm]:
        recs = by_arm[arm]
        n, k = len(recs), sum(r["passed"] for r in recs)
        no_code = sum(1 for r in recs if not r["parsed"]["code_found"])
        out_tok = sum((r["usage"]["completion_tokens"] or 0) for r in recs)
        cut = sum(1 for r in recs if r.get("finish_reason") == "length")
        print(f"\n{arm}: {k}/{n} passed ({k / n:.2f}); no-code-block: {no_code}; "
              f"truncated (finish_reason=length): {cut}; completion tokens: {out_tok}")
        fam: Dict[str, List[bool]] = defaultdict(list)
        for r in sorted(recs, key=lambda r: (r["family"], r["session"])):
            fam[r["family"]].append(r["passed"])
        for name, flags in fam.items():
            print(f"  {name:28s} " + " ".join("P" if f else "F" for f in flags))
    if ARM_A in by_arm:
        recs = by_arm[ARM_A]
        rate = sum(r["passed"] for r in recs) / len(recs)
        lo, hi = ARM_A_RANGE
        verdict = "IN RANGE" if lo <= rate <= hi else "OUT OF RANGE"
        print(f"\nArm A ceiling check: pass rate {rate:.2f} vs target [{lo}, {hi}] -> {verdict} "
              f"(n={len(recs)}; one temperature-0 sample per task, so this is a coarse screen)")
        if rate > hi:
            print("  Too easy: Arm A already near ceiling; memory cannot show a gain. Harden dev tasks.")
        elif rate < lo:
            print("  Too hard/noisy: check no-code-block counts and hidden-test outputs before concluding.")
    return 0


def cmd_diagnose(args) -> int:
    from .lesson_store import make_memoire

    families = load_split(args.split)
    texts: Dict[str, str] = {}
    for fam in families:
        for lesson in fam.lessons:
            texts[f"lesson:{lesson.key}"] = lesson.text
        for task in fam.tasks:
            texts[f"spec:{task.task_id}"] = task.spec
    table = exemplar_cosines(lambda: make_memoire(contradiction="off", trial_cap=1), texts)
    rows = sorted(table.items(), key=lambda kv: -(kv[1]["max_cos"] or 0))
    print(f"{'max_cos':>8}  group_max(c/a/r)         text")
    for key, info in rows:
        g = info["group_max"]
        flag = "  <-- above warn" if (info["max_cos"] or 0) >= args.warn_above else ""
        print(f"{info['max_cos'] or 0:8.3f}  {g['consequence']:.2f}/{g['actionability']:.2f}/{g['reusability']:.2f}"
              f"   {key}{flag}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(table, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m eval.run_eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run arms against the LLM (checkpointed, resumable)")
    r.add_argument("--split", choices=("dev", "test"), required=True)
    r.add_argument("--arms", default="A", help="comma list of A,B,C0,C1 (default: A)")
    r.add_argument("--seeds", default="0")
    r.add_argument("--families", help="comma list; default all in split (part of the run fingerprint)")
    r.add_argument("--sessions", help="comma list; default 1,2,3 (part of the run fingerprint)")
    r.add_argument("--model", required=True, help="explicit model id; '-latest' tags are rejected")
    r.add_argument("--base-url", default=DEFAULT_BASE_URL)
    r.add_argument("--api-key-env", default="GEMINI_API_KEY")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-tokens", type=int, default=4096,
                   help="completion cap; thinking models may count reasoning tokens in it (warns below 2048)")
    r.add_argument("--rpm", type=float, default=13)
    r.add_argument("--top-k", type=int, default=3)
    r.add_argument("--trial-cap", type=int, default=1, help="Memoire ScoringConfig.trial_cap (product default 1)")
    r.add_argument("--contradiction", choices=("off", "legacy", "nli"), default="off")
    r.add_argument("--max-calls", type=int, default=0, help="stop after this many HTTP requests (0 = no cap)")
    r.add_argument("--out", required=True, help="JSONL results/checkpoint file")
    r.add_argument("--diagnostics-json", help="output of `diagnose --out`, embedded per record")
    r.add_argument("--send-api-seed", action="store_true",
                   help="also send --seeds value as the HTTP `seed` field (OpenAI/Ollama-style endpoints only; "
                        "Google's compat endpoint rejects it with HTTP 400). Default: not sent")
    r.add_argument("--dry-run", action="store_true", help="plan only; no API calls, no key needed")
    r.add_argument("--allow-test-split", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="pass rates per arm/family from a results file")
    s.add_argument("--results", required=True)
    s.set_defaults(func=cmd_summarize)

    d = sub.add_parser("diagnose", help="exemplar-cosine table for lessons and task specs (needs native build)")
    d.add_argument("--split", choices=("dev", "test"), required=True)
    d.add_argument("--warn-above", type=float, default=0.50)
    d.add_argument("--out")
    d.set_defaults(func=cmd_diagnose)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""CLI:  python -m eval.run_eval {run,summarize,diagnose} ...   (see eval/README.md)"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

from .arms import ALL_ARMS, ARM_A, normalize_arm, uses_memory
from .diagnostics import TAU_GRID, exemplar_cosines, quality_scores, summarize_quality
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
                             max_tokens=args.max_tokens, rpm=args.rpm, send_api_seed=args.send_api_seed,
                             max_retries=args.max_retries)

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


def _failed_tests(output: str) -> List[str]:
    names = re.findall(r"^(?:FAIL|ERROR): (\S+)", output or "", re.MULTILINE)
    return sorted(set(names))


def _status(outcome: Optional[dict]) -> str:
    if outcome is None:
        return "-"
    if outcome.get("timed_out"):
        return "TIMEOUT"
    return "ok" if outcome.get("exit_code") == 0 else "FAIL"


def _print_detail(records: List[dict]) -> None:
    print("per-run detail (visible/hidden; failing test names come from the last 4000 chars of output):")
    for r in sorted(records, key=lambda r: (r["arm"], r["family"], r["session"], r["seed"])):
        g = r["grading"]
        failed = _failed_tests((g.get("hidden") or {}).get("output", ""))
        used = ",".join(r.get("used_keys") or []) or "-"
        print(f"  {r['arm']:16s} {r['family']}/s{r['session']} seed={r['seed']} "
              f"{'PASS' if r['passed'] else 'FAIL'} vis={_status(g.get('visible'))} hid={_status(g.get('hidden'))} "
              f"finish={r.get('finish_reason')} failed={failed or '-'} used={used}")


def cmd_summarize(args) -> int:
    records = _load_records(args.results)
    if not records:
        print("no records")
        return 1
    by_arm: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_arm[r["arm"]].append(r)
    if args.detail:
        _print_detail(records)
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
    factory = lambda: make_memoire(contradiction="off", trial_cap=1)
    table = exemplar_cosines(factory, texts)
    lesson_texts = {k: v for k, v in texts.items() if k.startswith("lesson:")}
    scores = quality_scores(factory, lesson_texts)
    for key, info in table.items():
        if key in scores:
            info["quality"] = scores[key]
    summary = summarize_quality(lesson_texts, scores)
    tau = args.quality_threshold
    rows = sorted(table.items(), key=lambda kv: -(kv[1]["max_cos"] or 0))
    print(f"{'max_cos':>8}  group_max(c/a/r)  {'q(importance_base)':>18}  text")
    for key, info in rows:
        g = info["group_max"]
        flag = "  <-- cosine above warn" if (info["max_cos"] or 0) >= args.warn_above else ""
        q = (info.get("quality") or {}).get("importance_base")
        qcol = "-" if q is None and key.startswith("spec:") else ("UNSCORED" if q is None else f"{q:.3f} {'admit' if q >= tau else 'shadow'}")
        print(f"{info['max_cos'] or 0:8.3f}  {g['consequence']:.2f}/{g['actionability']:.2f}/{g['reusability']:.2f}"
              f"  {qcol:>18}  {key}{flag}")
    print("\nquality_score = stored importance_base under the engine's default ScoringWeights "
          "(src/quality.rs: actionability 3/7, consequence 5/14, reusability 3/14; novelty and evidence weigh 0), "
          f"so it does not depend on ingestion order. 'admit' means q >= tau_q={tau:.2f} under admission=quality_gate.")
    print(f"{'kind':8s} {'n':>2} {'min':>6} {'mean':>6} {'max':>6}   admitted at tau = " + " ".join(f"{t:.2f}" for t in TAU_GRID))
    for kind in ("gold", "vague", "wrong", "poison"):
        if kind in summary:
            m = summary[kind]
            print(f"{kind:8s} {m['n']:>2} {m['min']:6.3f} {m['mean']:6.3f} {m['max']:6.3f}   " + "      ".join(
                str(m["admitted"][f"{t:.2f}"]) for t in TAU_GRID))
    print("caution: the gate sees text features only, never correctness. A tau_q that admits gold but drops wrong "
          "on dev would be fitting dev wording, not a property of the gate.")
    if args.out:
        table["_quality_summary"] = summary
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
    r.add_argument("--max-calls", type=int, default=0,
                   help="stop after this many HTTP requests, retries included (0 = no cap); leave headroom for 5xx retries")
    r.add_argument("--max-retries", type=int, default=8,
                   help="retries per call on 408/429/5xx and network errors, exponential backoff with jitter")
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
    s.add_argument("--detail", action="store_true", help="per-run visible/hidden status and failing test names")
    s.set_defaults(func=cmd_summarize)

    d = sub.add_parser("diagnose", help="exemplar-cosine table for lessons and task specs (needs native build)")
    d.add_argument("--split", choices=("dev", "test"), required=True)
    d.add_argument("--warn-above", type=float, default=0.50)
    d.add_argument("--quality-threshold", type=float, default=0.50,
                   help="tau_q shown in the would-admit column (engine default 0.50); reporting only")
    d.add_argument("--out")
    d.set_defaults(func=cmd_diagnose)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Pre-registered analysis for Step 8d (see eval/PREREGISTRATION_8d.md). Stdlib only, deterministic.

Everything the pre-registration fixes lives in the constants below. Change one and the analysis no longer
matches the registered plan (the report records ANALYSIS_VERSION and the constants it used).

Design rules enforced here, not just documented:
  * the unit of analysis is the FAMILY (sessions inside a family re-skin one choice, so tasks are not
    independent); per-family scores are paired across conditions;
  * exactly ONE contrast (PRIMARY) gets an inference (exact two-sided paired sign-flip p); every other contrast
    is descriptive (effect, bootstrap interval, family tally) and carries `p_exact_two_sided = None`;
  * a contrast is computed only on cells that exist in BOTH conditions for every registered family/session/seed;
    anything missing is reported as incomplete, never silently dropped.
"""
from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from math import lcm
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .tasks import load_split

ANALYSIS_VERSION = "8d-1"
ALPHA = 0.05  # two-sided, applied to PRIMARY only
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261124  # fixed; per-contrast stream is seeded from "<seed>:<contrast>:<window>"
GATE_A_RANGE = (0.15, 0.45)  # eval/README.md: Arm A is bounded by guessing, so this is a screen
GATE_MIN_B_MINUS_A = Fraction(3, 20)  # 0.15
PREFLIGHT_MAX_FLIPS = 1  # pooled over B and C1-t1; <= this many flips -> one seed on test, else two
FORMAT_FAILURE_WARN_SHARE = 0.05
MAX_FAMILIES_EXACT = 20

WINDOWS: Dict[str, Tuple[int, ...]] = {
    "s1-s4": (1, 2, 3, 4),
    "s1-s2": (1, 2),
    "s3-s4": (3, 4),
    "s2-s4": (2, 3, 4),
    "s1": (1,),
}
REGISTERED: Tuple[str, ...] = ("A", "B", "B-nopoison", "C0-t1", "C1-t1", "C0-t3", "C1-t3")

# (a, b, window): effect = accuracy(a) - accuracy(b), paired by family.
PRIMARY = ("C1-t1", "B", "s1-s4")
SECONDARY: Tuple[Tuple[str, str, str], ...] = (
    ("C1-t1", "A", "s1-s4"),
    ("C1-t1", "C0-t1", "s1-s4"),   # isolates feedback (same tagged wrapper, same trial_cap)
    ("C1-t1", "C0-t1", "s3-s4"),   # dev-informed mechanism window: feedback cannot act before s2
    ("C1-t1", "B", "s3-s4"),
    ("C0-t1", "B", "s1-s4"),
    ("C0-t1", "B", "s1-s2"),
    ("C0-t3", "C0-t1", "s1-s2"),   # exposure effect of trial_cap (dev: s1..s2 gap looked like exposure)
    ("C0-t3", "B", "s1-s4"),
    ("C1-t3", "B", "s1-s4"),
    ("C1-t3", "C0-t3", "s1-s4"),
    ("C1-t3", "C0-t3", "s3-s4"),
    ("B-nopoison", "B", "s2-s4"),  # poison harm on plain vector: positive = poison hurts B
    ("B-nopoison", "B", "s1-s4"),
)

ARM_NAMES = {"no_memory": "A", "plain_vector": "B", "memoire_static": "C0", "memoire_feedback": "C1"}


class AnalysisError(ValueError):
    pass


# ------------------------------------------------------------------ loading

def condition_label(rec: Mapping[str, Any]) -> str:
    """Condition = arm + the settings that define it. Registered conditions use the tagged prompt for C0/C1."""
    arm = rec["arm"]
    if arm not in ARM_NAMES:
        raise AnalysisError(f"unknown arm {arm!r}")
    base = ARM_NAMES[arm]
    if arm in ("memoire_static", "memoire_feedback"):
        base += f"-t{rec['trial_cap']}"
        if rec.get("prompt_policy", "plain") != "tagged":
            base += "-plain"
    if rec.get("no_poison"):
        base += "-nopoison"
    return base


def load_records(paths: Sequence[str]) -> List[dict]:
    out: List[dict] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            lines = [ln for ln in fh.read().split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except ValueError:
                if i == len(lines) - 1:
                    continue  # torn trailing line from a crash; the run loop would drop it too
                raise AnalysisError(f"{path}: corrupt record on line {i + 1}") from None
            rec["_source"] = path
            out.append(rec)
    return out


Cells = Dict[str, Dict[Tuple[str, int, int], dict]]


def build_cells(records: Iterable[dict]) -> Cells:
    cells: Cells = defaultdict(dict)
    for rec in records:
        key = (rec["family"], rec["session"], rec["seed"])
        label = condition_label(rec)
        if key in cells[label]:
            raise AnalysisError(
                f"duplicate cell {label} {key}: {cells[label][key]['_source']} and {rec['_source']}")
        cells[label][key] = rec
    return cells


# ------------------------------------------------------------------ statistics

def exact_sign_flip_p(deltas: Sequence[Fraction]) -> float:
    """Exact two-sided paired sign-flip p-value on the SUM of per-family differences.

    Under H0 each non-zero difference is equally likely to carry either sign; p is the share of the 2^n sign
    patterns whose |sum| is at least the observed |sum|. Uses the actual magnitudes (not just counts) and exact
    integer arithmetic; zero differences carry no information and are dropped.
    """
    nonzero = [d for d in deltas if d != 0]
    if not nonzero:
        return 1.0
    if len(nonzero) > MAX_FAMILIES_EXACT:
        raise AnalysisError(f"{len(nonzero)} non-zero families; exact enumeration is capped at {MAX_FAMILIES_EXACT}")
    scale = lcm(*(d.denominator for d in nonzero))
    mags = [abs(int(d * scale)) for d in nonzero]
    observed = abs(sum(int(d * scale) for d in nonzero))
    dist: Dict[int, int] = {0: 1}
    for m in mags:
        nxt: Dict[int, int] = defaultdict(int)
        for total, count in dist.items():
            nxt[total + m] += count
            nxt[total - m] += count
        dist = nxt
    extreme = sum(count for total, count in dist.items() if abs(total) >= observed)
    return extreme / 2 ** len(nonzero)


def bootstrap_ci(deltas: Sequence[Fraction], stream: str) -> Tuple[float, float]:
    """Percentile bootstrap over FAMILIES. Anti-conservative at n=8 (simulated type-I ~0.10 vs 0.05 nominal);
    reported as an interval, never used as the decision test."""
    rng = random.Random(f"{BOOTSTRAP_SEED}:{stream}")
    values = [float(d) for d in deltas]
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP_RESAMPLES))
    return means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]


# ------------------------------------------------------------------ cells -> scores

def _expected_cells(families) -> List[Tuple[str, int]]:
    return [(f.name, t.session) for f in families for t in f.tasks]


def full_seeds(cells: Cells, label: str, families) -> List[int]:
    """Seeds for which `label` has every registered (family, session) cell."""
    expected = _expected_cells(families)
    seeds = sorted({k[2] for k in cells.get(label, {})})
    return [s for s in seeds if all((f, ss, s) in cells[label] for f, ss in expected)]


def missing_cells(cells: Cells, label: str, families, seeds: Sequence[int]) -> List[Tuple[str, int, int]]:
    return [(f, ss, s) for s in seeds for f, ss in _expected_cells(families) if (f, ss, s) not in cells.get(label, {})]


def family_scores(cells: Cells, label: str, families, seeds: Sequence[int], window: str) -> Dict[str, Fraction]:
    sessions = WINDOWS[window]
    scores: Dict[str, Fraction] = {}
    for fam in families:
        passes = total = 0
        for seed in seeds:
            for session in sessions:
                rec = cells[label].get((fam.name, session, seed))
                if rec is None:
                    continue
                passes += int(bool(rec["passed"]))
                total += 1
        if total:
            scores[fam.name] = Fraction(passes, total)
    return scores


def contrast(cells: Cells, families, a: str, b: str, window: str, *, primary: bool = False) -> dict:
    name = f"{a} - {b} [{window}]"
    out: Dict[str, Any] = {"name": name, "a": a, "b": b, "window": window,
                           "inference": "primary" if primary else "descriptive", "p_exact_two_sided": None}
    for label in (a, b):
        if label not in cells:
            out["status"] = f"skipped: condition {label} has no records"
            return out
    seeds = sorted(set(full_seeds(cells, a, families)) & set(full_seeds(cells, b, families)))
    if not seeds:
        out["status"] = "incomplete: no seed has every registered family/session cell in both conditions"
        return out
    sa, sb = family_scores(cells, a, families, seeds, window), family_scores(cells, b, families, seeds, window)
    deltas = {f: sa[f] - sb[f] for f in sa if f in sb}
    values = list(deltas.values())
    mean = sum(values, Fraction(0)) / len(values)
    lo, hi = bootstrap_ci(values, name)
    out.update({
        "status": "ok", "seeds": seeds, "n_families": len(values),
        "delta_mean": float(mean), "delta_mean_exact": str(mean),
        "ci95_family_bootstrap": [lo, hi],
        "tally": {"a_better": sum(1 for d in values if d > 0), "b_better": sum(1 for d in values if d < 0),
                  "tied": sum(1 for d in values if d == 0)},
        "per_family_delta": {f: str(d) for f, d in sorted(deltas.items())},
    })
    if primary:
        out["p_exact_two_sided"] = exact_sign_flip_p(values)
    return out


def pooled_rate(cells: Cells, label: str, families, seeds: Sequence[int], window: str) -> Optional[dict]:
    sessions, names = WINDOWS[window], {f.name for f in families}
    recs = [r for (fam, ses, seed), r in cells.get(label, {}).items()
            if fam in names and ses in sessions and seed in seeds]
    if not recs:
        return None
    passed = sum(1 for r in recs if r["passed"])
    return {"passed": passed, "n": len(recs), "rate": passed / len(recs)}


# ------------------------------------------------------------------ poison, flips, integrity

def poison_metrics(cells: Cells, label: str, families, seeds: Sequence[int]) -> dict:
    """Descriptive, self-report-based: `used_keys` is what the model declared, not a causal attribution."""
    n = shown = failed_shown = declared_failed = last_n = last_shown = 0
    for fam in families:
        last = max(t.session for t in fam.tasks)
        for seed in seeds:
            for task in fam.tasks:
                if task.session < fam.poison_session:
                    continue
                rec = cells.get(label, {}).get((fam.name, task.session, seed))
                if rec is None:
                    continue
                is_shown = any(r.get("kind") == "poison" for r in rec.get("retrieved") or [])
                declared = any(str(k).endswith(":poison") for k in rec.get("used_keys") or [])
                failed = not rec["passed"]
                n += 1
                shown += is_shown
                failed_shown += is_shown and failed
                declared_failed += declared and failed
                if task.session == last:
                    last_n += 1
                    last_shown += is_shown
    rate = lambda num, den: (num / den) if den else None
    return {"cells_at_or_after_poison_session": n, "poison_shown": shown,
            "exposure_rate": rate(shown, n), "failed_given_shown": rate(failed_shown, shown),
            "failed_and_poison_declared": declared_failed, "failed_and_poison_declared_rate": rate(declared_failed, n),
            "last_session_cells": last_n, "last_session_poison_shown": last_shown,
            "last_session_exposure_rate": rate(last_shown, last_n)}


def seed_flips(cells: Cells, label: str, families) -> dict:
    seeds = full_seeds(cells, label, families)
    pairs = flips = 0
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            for fam, ses in _expected_cells(families):
                pairs += 1
                flips += int(bool(cells[label][(fam, ses, s1)]["passed"]) != bool(cells[label][(fam, ses, s2)]["passed"]))
    return {"seeds": seeds, "paired_cells": pairs, "flips": flips}


def s1_identity(cells: Cells, families) -> Optional[dict]:
    """B and B-nopoison hold identical stores at s1, so their s1 prompts must be byte-identical; any outcome
    difference there is pure API noise for the same input."""
    if "B" not in cells or "B-nopoison" not in cells:
        return None
    pairs = identical = flips = 0
    for fam in families:
        for seed in sorted(set(full_seeds(cells, "B", families)) & set(full_seeds(cells, "B-nopoison", families))):
            a, b = cells["B"][(fam.name, 1, seed)], cells["B-nopoison"][(fam.name, 1, seed)]
            pairs += 1
            identical += int(a["prompt"] == b["prompt"])
            flips += int(bool(a["passed"]) != bool(b["passed"]))
    return {"pairs": pairs, "prompts_identical": identical, "outcome_flips": flips}


def integrity(cells: Cells, families, split: str) -> dict:
    failures: List[str] = []
    warnings: List[str] = []
    present = [l for l in REGISTERED if l in cells]
    extra = sorted(l for l in cells if l not in REGISTERED)
    if extra:
        warnings.append(f"unregistered conditions present (excluded from registered contrasts): {extra}")
    all_recs = [r for l in cells for r in cells[l].values()]
    for field, want in (("split", split), ("store_scope", "family"), ("admission", "admit_all")):
        bad = sorted({str(r.get(field)) for r in all_recs if r.get(field) != want})
        if bad:
            failures.append(f"{field} must be {want!r} for every record; found {bad}")
    for field in ("model", "model_returned", "temperature", "top_k"):
        seen = sorted({str(r.get(field)) for l in present for r in cells[l].values()})
        if len(seen) > 1:
            failures.append(f"{field} is not constant across registered conditions: {seen}")
    by_file: Dict[str, set] = defaultdict(set)
    for r in all_recs:
        by_file[r["_source"]].add(r.get("fingerprint"))
    for path, fps in sorted(by_file.items()):
        if len(fps) > 1:
            failures.append(f"{path} mixes {len(fps)} run fingerprints")
    seeds_union = sorted({s for l in present for s in full_seeds(cells, l, families)})
    for label in present:
        gaps = missing_cells(cells, label, families, seeds_union) if seeds_union else []
        if gaps:
            failures.append(f"{label}: {len(gaps)} missing cell(s) for seeds {seeds_union}, e.g. {gaps[:3]}")
        recs = list(cells[label].values())
        bad_format = sum(1 for r in recs if not r["parsed"]["code_found"] or r.get("finish_reason") == "length")
        if recs and bad_format / len(recs) > FORMAT_FAILURE_WARN_SHARE:
            warnings.append(f"{label}: {bad_format}/{len(recs)} runs had no code block or were truncated "
                            f"(counted as failures)")
    if "B-nopoison" in cells:
        leak = poison_metrics(cells, "B-nopoison", families, full_seeds(cells, "B-nopoison", families))
        if leak["poison_shown"]:
            failures.append(f"B-nopoison showed a poison lesson in {leak['poison_shown']} prompt(s)")
    ident = s1_identity(cells, families)
    if ident and ident["prompts_identical"] != ident["pairs"]:
        failures.append(f"B and B-nopoison s1 prompts differ in {ident['pairs'] - ident['prompts_identical']} "
                        f"of {ident['pairs']} pairs (stores must be identical at s1)")
    return {"ok": not failures, "failures": failures, "warnings": warnings, "registered_present": present,
            "format_failures": {l: {"no_code_block": sum(1 for r in cells[l].values() if not r["parsed"]["code_found"]),
                                    "truncated": sum(1 for r in cells[l].values() if r.get("finish_reason") == "length"),
                                    "n": len(cells[l])} for l in sorted(cells)}}


# ------------------------------------------------------------------ report

def _git(*args: str) -> Optional[str]:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def gate(cells: Cells, families) -> dict:
    a = contrast(cells, families, "B", "A", "s1-s4")
    seeds_a = full_seeds(cells, "A", families) if "A" in cells else []
    rate = pooled_rate(cells, "A", families, seeds_a, "s1-s4") if seeds_a else None
    if rate is None or a.get("status") != "ok":
        return {"passed": None, "reason": "needs complete A and B"}
    in_range = GATE_A_RANGE[0] <= rate["rate"] <= GATE_A_RANGE[1]
    ok_gap = Fraction(a["delta_mean_exact"]) >= GATE_MIN_B_MINUS_A
    return {"passed": bool(in_range and ok_gap), "a_rate": rate["rate"], "a_range": list(GATE_A_RANGE),
            "a_in_range": in_range, "b_minus_a": a["delta_mean"], "b_minus_a_min": float(GATE_MIN_B_MINUS_A),
            "b_minus_a_ok": ok_gap}


def verdict(primary: dict, gate_result: dict) -> str:
    if primary.get("status") != "ok":
        return "incomplete"
    if gate_result.get("passed") is not True:
        return "gate_not_passed: effect reported descriptively, no inference"
    p, d = primary["p_exact_two_sided"], primary["delta_mean"]
    if p < ALPHA and d > 0:
        return "supported (this bank, this model)"
    if p < ALPHA and d < 0:
        return "worse (this bank, this model)"
    return "not_distinguishable at this sample size"


def analyze(paths: Sequence[str], split: str) -> dict:
    families = load_split(split)
    cells = build_cells(load_records(paths))
    present = [l for l in REGISTERED if l in cells]
    seeds_union = sorted({s for l in present for s in full_seeds(cells, l, families)})
    gate_result = gate(cells, families)
    primary = contrast(cells, families, *PRIMARY[:2], PRIMARY[2], primary=True)
    conditions = {}
    for label in sorted(cells):
        seeds = full_seeds(cells, label, families)
        conditions[label] = {
            "seeds_complete": seeds,
            "windows": {w: pooled_rate(cells, label, families, seeds, w) for w in WINDOWS},
            "per_family_s1_s4": {f: str(v) for f, v in sorted(family_scores(cells, label, families, seeds, "s1-s4").items())},
            "seed_flips": seed_flips(cells, label, families),
            "poison": poison_metrics(cells, label, families, seeds) if label != "A" else None,
        }
    return {
        "analysis_version": ANALYSIS_VERSION, "split": split, "inputs": sorted(paths),
        "constants": {"alpha_two_sided": ALPHA, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                      "bootstrap_seed": BOOTSTRAP_SEED, "gate_a_range": list(GATE_A_RANGE),
                      "gate_min_b_minus_a": float(GATE_MIN_B_MINUS_A), "primary": list(PRIMARY)},
        "provenance": {"python": platform.python_version(), "git_head": _git("rev-parse", "HEAD"),
                       "git_tag_at_head": _git("describe", "--tags", "--exact-match")},
        "integrity": integrity(cells, families, split),
        "seeds_in_registered_conditions": seeds_union,
        "conditions": conditions,
        "gate": gate_result,
        "primary": primary,
        "primary_verdict": verdict(primary, gate_result),
        "secondary": [contrast(cells, families, a, b, w) for a, b, w in SECONDARY],
        "s1_identity_B_vs_B_nopoison": s1_identity(cells, families),
    }


def preflight(paths: Sequence[str], split: str) -> dict:
    """Seed-count decision from dev reruns: B and C1-t1 at seeds {0,1}; pooled flips <= PREFLIGHT_MAX_FLIPS -> 1 seed."""
    families = load_split(split)
    cells = build_cells(load_records(paths))
    labels = ["B", "C1-t1"]
    absent = [l for l in labels if l not in cells]
    if absent:
        raise AnalysisError(f"preflight needs the conditions B and C1-t1 (tagged); missing {absent}; found {sorted(cells)}")
    ignored = sorted(set(cells) - set(labels))
    detail, pairs, flips = {}, 0, 0
    for label in labels:
        info = seed_flips(cells, label, families)
        if info["seeds"] != [0, 1]:
            raise AnalysisError(f"{label}: need complete seeds [0, 1]; complete seeds found {info['seeds']}")
        detail[label] = info
        pairs += info["paired_cells"]
        flips += info["flips"]
    return {"analysis_version": ANALYSIS_VERSION, "split": split, "per_condition": detail, "paired_cells": pairs,
            "ignored_conditions": ignored, "flips": flips, "max_flips_for_one_seed": PREFLIGHT_MAX_FLIPS,
            "decision": "ONE seed on test" if flips <= PREFLIGHT_MAX_FLIPS else "TWO seeds on test"}


def render(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    w = lines.append
    w(f"Step 8d analysis {report['analysis_version']}  split={report['split']}  "
      f"head={report['provenance']['git_head']}  tag={report['provenance']['git_tag_at_head']}")
    integ = report["integrity"]
    w(f"integrity: {'OK' if integ['ok'] else 'FAILED'}")
    for item in integ["failures"]:
        w(f"  FAILURE: {item}")
    for item in integ["warnings"]:
        w(f"  warning: {item}")
    w("\ncondition        seeds   s1-s4        s1-s2        s3-s4        flips   poison-exposure(s>=2)")
    for label, c in report["conditions"].items():
        def cell(win):
            r = c["windows"][win]
            return f"{r['passed']}/{r['n']} ({r['rate']:.2f})" if r else "-"
        sf, po = c["seed_flips"], c["poison"]
        flips = f"{sf['flips']}/{sf['paired_cells']}" if sf["paired_cells"] else "-"
        exp = "-" if not po or po["exposure_rate"] is None else f"{po['exposure_rate']:.2f}"
        w(f"{label:16s} {str(c['seeds_complete']):7s} {cell('s1-s4'):12s} {cell('s1-s2'):12s} {cell('s3-s4'):12s} "
          f"{flips:7s} {exp}")
    g = report["gate"]
    w(f"\ngate: {g}")
    p = report["primary"]
    w(f"\nPRIMARY {p['name']}: status={p['status']}")
    if p["status"] == "ok":
        w(f"  delta={p['delta_mean']:+.3f}  95% family-bootstrap [{p['ci95_family_bootstrap'][0]:+.3f}, "
          f"{p['ci95_family_bootstrap'][1]:+.3f}] (anti-conservative at n=8)  exact two-sided p={p['p_exact_two_sided']:.4f}  "
          f"families C1-better/B-better/tied={p['tally']['a_better']}/{p['tally']['b_better']}/{p['tally']['tied']}")
    w(f"  VERDICT: {report['primary_verdict']}")
    w("\nsecondary (descriptive only, no p-values by design):")
    for c in report["secondary"]:
        if c["status"] != "ok":
            w(f"  {c['name']}: {c['status']}")
            continue
        w(f"  {c['name']:34s} delta={c['delta_mean']:+.3f} CI [{c['ci95_family_bootstrap'][0]:+.3f}, "
          f"{c['ci95_family_bootstrap'][1]:+.3f}]  better/worse/tied={c['tally']['a_better']}/{c['tally']['b_better']}/"
          f"{c['tally']['tied']}")
    ident = report["s1_identity_B_vs_B_nopoison"]
    if ident:
        w(f"\nB vs B-nopoison at s1 (identical prompts expected): identical {ident['prompts_identical']}/{ident['pairs']}, "
          f"outcome flips {ident['outcome_flips']}/{ident['pairs']}  <- API noise floor on the test split")
    return "\n".join(lines)


def cmd_analyze(args) -> int:
    try:
        if args.preflight:
            result = preflight(args.results, args.split)
            print(f"preflight ({args.split}): {result['flips']} flips in {result['paired_cells']} paired cells "
                  f"(rule: <= {result['max_flips_for_one_seed']} -> one seed)")
            for label, info in result["per_condition"].items():
                print(f"  {label}: {info['flips']}/{info['paired_cells']} flips between seeds {info['seeds']}")
            if result["ignored_conditions"]:
                print(f"  (ignored other conditions in the input: {result['ignored_conditions']})")
            print(f"DECISION: {result['decision']}")
        else:
            result = analyze(args.results, args.split)
            print(render(result))
    except AnalysisError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"wrote {args.out}")
    if not args.preflight and not result["integrity"]["ok"]:
        return 1
    return 0

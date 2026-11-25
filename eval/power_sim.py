"""Power of the family-paired comparison at the sample sizes in play (Step 8d planning aid).  Not part of the analysis.

    python -m eval.power_sim            # prints the table quoted in eval/PREREGISTRATION_8d.md

Assumptions (all of them are guesses; change them and re-read the table, do not treat it as a result):
  * family success probability for the baseline B ~ Beta(0.7, 0.3): mostly all-or-none families, mean 0.70;
  * the comparison arm's probability = B's + delta + N(0, 0.2^2), clipped to [0, 1] (so the REALIZED mean effect is
    smaller than delta near the ceiling; the table reports the realized value). The "null" row (delta=None) is the
    exact null instead: identical family probabilities, only the Bernoulli draws differ;
  * 4 sessions per family, independent Bernoulli draws given the family probability; `reps` replicates multiply
    the draws and only help if reruns are genuinely stochastic, so reps > 1 rows are upper bounds.
Uses the same exact paired sign-flip as the real analysis, so the two cannot drift apart.
"""
from __future__ import annotations

import random
import sys
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .analysis import ALPHA, exact_sign_flip_p

SEED = 20261124
TAU = 0.2
SESSIONS = 4
BOOT = 400


def _family_deltas(rng: random.Random, n_fam: int, delta: Optional[float], reps: int) -> List[Fraction]:
    out = []
    n = SESSIONS * reps
    for _ in range(n_fam):
        pb = rng.betavariate(0.7, 0.3)
        pc = pb if delta is None else min(1.0, max(0.0, pb + delta + rng.gauss(0, TAU)))
        b = sum(rng.random() < pb for _ in range(n))
        c = sum(rng.random() < pc for _ in range(n))
        out.append(Fraction(c, n) - Fraction(b, n))
    return out


def _boot_excludes_zero(rng: random.Random, deltas: List[Fraction]) -> Tuple[bool, float]:
    vals = [float(d) for d in deltas]
    n = len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(BOOT))
    lo, hi = means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]
    return (lo > 0 or hi < 0), hi - lo


def realized_effect(delta: Optional[float], draws: int = 100_000) -> float:
    if delta is None:
        return 0.0
    rng = random.Random(f"{SEED}:realized:{delta}")
    return sum(min(1.0, max(0.0, (pb := rng.betavariate(0.7, 0.3)) + delta + rng.gauss(0, TAU))) - pb
               for _ in range(draws)) / draws


def simulate(n_fam: int, delta: Optional[float], reps: int, trials: int = 400) -> Dict[str, float]:
    rng = random.Random(f"{SEED}:{n_fam}:{delta}:{reps}")
    exact = boot = 0
    widths = []
    for _ in range(trials):
        deltas = _family_deltas(rng, n_fam, delta, reps)
        exact += exact_sign_flip_p(deltas) < ALPHA
        excludes, width = _boot_excludes_zero(rng, deltas)
        boot += excludes
        widths.append(width)
    return {"power_exact": exact / trials, "power_or_fpr_bootstrap": boot / trials,
            "mean_ci_width": sum(widths) / len(widths)}


def main(trials: int = 400) -> int:
    print("paired C1-B, family unit, two-sided alpha=0.05; on the null row both columns are false-positive rates")
    print("fams reps nominal_delta realized_effect | exact-test power | bootstrap CI excludes 0 | mean CI width")
    for n_fam in (8, 16):
        for reps in (1, 3):
            for delta in (None, 0.10, 0.20, 0.30):
                r = simulate(n_fam, delta, reps, trials)
                label = "null" if delta is None else f"{delta:.2f}"
                print(f"{n_fam:4d} {reps:4d} {label:>13s} {realized_effect(delta):15.3f} | {r['power_exact']:16.2f} | "
                      f"{r['power_or_fpr_bootstrap']:24.2f} | {r['mean_ci_width']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 400))

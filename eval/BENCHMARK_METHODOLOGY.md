# Benchmark methodology and results: Step 8d held-out run

Status: **final for the Step 8d run.** The numbers below are transcribed from the `python -m eval.run_eval analyze --split test`
output of the run against tag `prereg-8d`; nothing is recomputed or adjusted. The plan this reports against is
`eval/PREREGISTRATION_8d.md` (unchanged by this document).

## 0. Summary

- **The pre-registered gate was not passed**, so the primary endpoint (`C1-t1` vs `B`) yields **no inference**. Plain-vector memory beat no
  memory by 0.03 (needed at least 0.15). The effect is reported descriptively: `C1-t1` 0.34 vs `B` 0.41, difference −0.06,
  95% family-bootstrap interval [−0.250, +0.125], exact two-sided p = 0.75.
- **No accuracy-improvement claim is made for `trial_cap=1`, or for any other configuration.** This follows the commitment in
  `eval/PREREGISTRATION_8d.md` section 3 and is repeated in `README.md`.
- The dev pilot's plain-vector advantage over no memory (14/20 vs 8/20, 5 families) **did not replicate** on the 8 held-out families
  (13/32 vs 12/32).
- Exploratory observations (section 6) point to a poison effect on plain vector, a first-session bottleneck at `trial_cap=1`, and a
  larger feedback effect when exposure is matched (`trial_cap=3`). They are descriptive and uncorrected, several lean on a single
  family, and none is evidence on its own. They would need replication on new families under a new pre-registration.

## 1. Provenance

| Item | Value |
|---|---|
| Pre-registration tag / commit | `prereg-8d` / `10e0f8fb55e1066bc2a5ba4e9fe04599a4356006` (the tag on the remote resolves to this SHA and is the tip of `fix/p0-stabilization`) |
| Analysis version | `8d-1` (`ANALYSIS_VERSION` in the tagged `eval/analysis.py`) |
| Model | `gemini-3.1-flash-lite` (as filled in the tagged pre-registration, section 2) |
| `model_returned` | constant across all records (integrity check passed); exact string: `gemini-3.1-flash-lite` |
| Seeds | one: seed 0. Dev preflight: 0 flips in 40 paired cells (B 0/20, C1-t1 0/20), so the pre-registered rule selected one seed |
| Settings (all conditions) | temperature 0.0, `top_k` 3, `--store-scope family`, `admit_all`, `contradiction off`, `--max-tokens 4096`, `--rpm 13`, no API seed; `--prompt-policy tagged` for C0/C1, `plain` for A/B |
| Split | `test`: 8 families x 4 sessions = 32 tasks per condition |
| Run dates (UTC) | 2026-11-25 01:46:46–02:06:04 |
| Integrity | `OK`: 0 no-code-block, 0 truncated; `B` vs `B-nopoison` at s1: prompts identical 8/8, outcome flips 0/8 (API noise floor on identical input) |
| Deviations from the pre-registration | none reported to the drafter; the runs were executed against the tag. The pre-run tracked-file check (pre-registration section 8) is the owner's attestation |
| Raw files | git-ignored (`eval/results/`); SHA-256 below |
| Report JSON | `eval/reports/test_report.json` (versioned copy; source `eval/results/test_report.json` is git-ignored); SHA-256: `BD982A7A18AA12C076CD2D665B787E9FADD5F6654C34E06EC1DA1EBDDEFADC7D` |

Raw result files (SHA-256, from `Get-FileHash`):

| File | SHA-256 |
|---|---|
| `eval/results/test_A.jsonl` | `E5719BB430E7EFCC7C92AD5A012B700E230119F21402E09C322943536021C127` |
| `eval/results/test_B.jsonl` | `2D5D9E669ED639008029307EFC71B584DDC29031844DF1D04D983EF9660E450A` |
| `eval/results/test_B_nopoison.jsonl` | `488C13F3A86F74E4A678B724A1CB5FA71F7E9DC03FA75914F5518950AED46081` |
| `eval/results/test_C0_t1.jsonl` | `E3C9A1710A3E874A13EDECD390217D426EE26DA47CC6C406FCA0BF9AF5C1B7C8` |
| `eval/results/test_C0_t3.jsonl` | `42406E6B46984E39CDAF79CA8D9C4BA0CD8B3499670175996C9879652B6E9D78` |
| `eval/results/test_C1_t1.jsonl` | `CAF181E569DCB31EC0C6186AC24B91E4EA6A6C7AE0ED1B05416DFBC4A2CCB51D` |
| `eval/results/test_C1_t3.jsonl` | `E0AA5EB04598FE8C7D644B5FAD26945FF0C5D2858506176440836E713131DB7D` |

## 2. What was measured

Cross-session pass rate of one model on 8 held-out families x 4 sessions, with a bank of conflicting lessons (gold, vague, wrong)
and a poison lesson that enters before session 2. Seven conditions (flags in pre-registration section 2). Lessons contain the answer
and were written by the project, and the poison lesson is wrong by design. This measures retrieval and trust under conflicting
lessons; it is not a measure of general coding ability, of other models, or of the prevalence of bad lessons in real use.

## 3. Primary result (pre-registered)

| Item | Value |
|---|---|
| Gate: Arm A pooled pass rate in [0.15, 0.45] | **met** (0.375) |
| Gate: `B` − `A` ≥ 0.15 | **not met** (+0.031, i.e. 13/32 vs 12/32) |
| Gate | **not passed** |
| `C1-t1` − `B`, s1–s4 | -0.0625 (11/32 = 0.34 vs 13/32 = 0.41) |
| 95% family-bootstrap interval (anti-conservative at n = 8) | [-0.250, +0.125] |
| Exact two-sided sign-flip p, as printed | 0.7500 (**not an inference** under the gate rule) |
| Families `C1-t1` better / `B` better / tied | 1 / 3 / 4 |
| `primary_verdict` | `gate_not_passed: effect reported descriptively, no inference` |

Reading, per the pre-registered commitment: no accuracy-improvement claim is made. An interval that includes zero is not evidence
of no effect, and a gate failure means this bank and prompt, with this model, did not show that plain-vector lessons help at all
(0.41 vs 0.38), so the benchmark could not discriminate between memory designs on its own terms. The dev pilot had shown a large gap
(B 14/20 vs A 8/20); either that gap was specific to the 5 dev families or to small-sample noise. This run cannot tell which.

## 4. Conditions

Pooled pass rate over s1–s4 (32 tasks) and by window (16 tasks each):

| Label | Arm | s1–s4 | s1–s2 | s3–s4 |
|---|---|---|---|---|
| `A` | no_memory | 12/32 (0.38) | 5/16 (0.31) | 7/16 (0.44) |
| `B` | plain_vector | 13/32 (0.41) | 8/16 (0.50) | 5/16 (0.31) |
| `B-nopoison` | plain_vector, no poison ingested | 17/32 (0.53) | 10/16 (0.62) | 7/16 (0.44) |
| `C0-t1` | memoire_static, trial_cap 1 | 8/32 (0.25) | 2/16 (0.12) | 6/16 (0.38) |
| `C0-t3` | memoire_static, trial_cap 3 | 11/32 (0.34) | 6/16 (0.38) | 5/16 (0.31) |
| `C1-t1` | memoire_feedback, trial_cap 1 | 11/32 (0.34) | 2/16 (0.12) | 9/16 (0.56) |
| `C1-t3` | memoire_feedback, trial_cap 3 | 18/32 (0.56) | 8/16 (0.50) | 10/16 (0.62) |

Two facts the table shows without any contrast: `C0-t1` has the lowest pooled rate (0.25), and `C1-t3` has the highest (0.56). Both `trial_cap=1`
conditions scored 2/16 in sessions 1–2, the lowest of the seven.

## 5. Poison metrics (descriptive; `used_keys` is the model's self-report, not causal attribution)

Cells at or after the poison session: 8 families x s2–s4 = 24 per condition. Counts in parentheses are derived from the reported rates.
`B-nopoison` ingested no poison and the integrity check (no poison ever shown) passed.

| Condition | Poison shown (s2–s4) | Failed given shown | Failed with poison declared | Poison shown in s4 |
|---|---|---|---|---|
| `B` | 0.75 (18/24) | 0.667 (12/18) | 7/24 | 0.625 (5/8) |
| `C0-t1` | 0.50 (12/24) | 1.000 (12/12) | 12/24 | 0.375 (3/8) |
| `C0-t3` | 0.75 (18/24) | 0.722 (13/18) | 7/24 | 0.625 (5/8) |
| `C1-t1` | 0.83 (20/24) | 0.600 (12/20) | 11/24 | 0.50 (4/8) |
| `C1-t3` | 0.71 (17/24) | 0.412 (7/17) | 7/24 | 0.50 (4/8) |

Poison was still present in the s4 prompt in 5 of 8 `B` and `C0-t3` families, 3 of 8 `C0-t1` families, and 4 of 8 families for
each `C1` condition. For `C1`, the demotion mechanism (two attributed failures) had not removed it from half the families by
the last session.

## 6. Secondary contrasts (13 pre-registered; descriptive only, no p-values by design)

Effect = accuracy(first) − accuracy(second), paired by family. Interval = percentile bootstrap over 8 families, which
rejects roughly 10–12% of the time under a true null at this size (`python -m eval.power_sim`), so it is optimistic. Tally = first better / second
better / tied.

| # | Contrast | Window | Effect | 95% interval | Tally |
|---|---|---|---|---|---|
| 1 | `C1-t1` − `A` | s1-s4 | -0.031 | [-0.312, +0.219] | 2/3/3 |
| 2 | `C1-t1` − `C0-t1` | s1-s4 | +0.094 | [-0.188, +0.312] | 4/1/3 |
| 3 | `C1-t1` − `C0-t1` | s3-s4 | +0.188 | [-0.250, +0.625] | 3/1/4 |
| 4 | `C1-t1` − `B` | s3-s4 | +0.250 | [+0.062, +0.500] | 3/0/5 |
| 5 | `C0-t1` − `B` | s1-s4 | -0.156 | [-0.500, +0.188] | 2/4/2 |
| 6 | `C0-t1` − `B` | s1-s2 | -0.375 | [-0.688, -0.062] | 1/5/2 |
| 7 | `C0-t3` − `C0-t1` | s1-s2 | +0.250 | [+0.000, +0.438] | 5/1/2 |
| 8 | `C0-t3` − `B` | s1-s4 | -0.062 | [-0.219, +0.062] | 1/2/5 |
| 9 | `C1-t3` − `B` | s1-s4 | +0.156 | [-0.031, +0.375] | 3/1/4 |
| 10 | `C1-t3` − `C0-t3` | s1-s4 | +0.219 | [+0.031, +0.438] | 5/1/2 |
| 11 | `C1-t3` − `C0-t3` | s3-s4 | +0.312 | [+0.000, +0.625] | 4/1/3 |
| 12 | `B-nopoison` − `B` | s2-s4 | +0.167 | [-0.042, +0.458] | 3/1/4 |
| 13 | `B-nopoison` − `B` | s1-s4 | +0.125 | [-0.031, +0.344] | 3/1/4 |

Multiplicity: 13 contrasts were registered. Three intervals exclude zero (#4, #6, #10) and two touch it (#7, #11); with a ~0.10–0.12 per-contrast false-positive
rate and shared conditions, a few exclusions are expected by chance. None is treated as an inference.

## 7. Exploratory findings

Everything in this section is exploratory, non-inferential, and subject to replication. Pooled rates in section 4 and the per-family
counts below are facts about this run; the interpretations are hypotheses.

1. **Poison on plain vector.** `B-nopoison` 0.53 vs `B` 0.41 (#13: +0.125 [−0.031, +0.344], 3/1/4; #12 on s2–s4: +0.167 [−0.042, +0.458]).
   The estimate points toward poison harm and both intervals include zero. `B` showed poison in 18/24 s2–s4 prompts and failed 12 of those 18.
   Part of the gap is displacement (with poison `B` shows the top 3 of 4 lessons, without it all 3). **One family dominates:** `retry_backoff`
   (A 4/4, `B-nopoison` 4/4, `B` 1/4) contributes +0.75 of the +1.0 family-sum, i.e. 75% of the net difference (derived from the per-family counts).
2. **`trial_cap=1` first-session bottleneck.** `C0-t1` − `B` in s1–s2 is −0.375 (#6, 1/5/2) and `C0-t3` − `C0-t1` in s1–s2 is +0.25 (#7, lower bound 0.000, 5/1/2).
   This is the pattern predicted from the code before the run (with no feedback every row is unverified, so `trial_cap=1` shows at most one note;
   `README.md` Known Limitation 1) and agrees with the dev pilot. It is a property of the conservative default, not evidence about trust.
3. **Exposure-matched feedback.** `C1-t3` 0.56 vs `C0-t3` 0.34 (#10: +0.219 [+0.031, +0.438], 5/1/2; s3–s4 #11: +0.312 [0.000, +0.625]) and vs `B` 0.41
   (#9: +0.156 [−0.031, +0.375]). At `trial_cap=1` the same contrast is smaller and includes zero (#2: +0.094 [−0.188, +0.312]). **`retry_backoff` again carries a
   large share** (`C1-t3` 4/4 vs `B` 1/4 and `C0-t3` 1/4): +0.75 is 60% of the `C1-t3` − `B` family-sum and 43% of the `C1-t3` − `C0-t3` family-sum (derived). In that
   family no-memory already scored 4/4, so part of this result is a memory condition avoiding harm, not gaining from a lesson. No contrast of `C1-t3` against `A` was registered.
4. **The dev-informed s3–s4 window.** `C1-t1` scored 9/16 in s3–s4 against 5/16 for `B` (#4: +0.250 [+0.062, +0.500], 3/0/5) and 6/16 for `C0-t1` (#3: +0.188 [−0.250, +0.625], 3/1/4).
   The direction matches the dev pilot's hypothesis, and the same condition scored 2/16 in s1–s2, which is why the registered overall endpoint is negative. The window was chosen
   after seeing dev data and the interval is uncorrected, so this is a hypothesis for replication. `A` scored 7/16 in s3–s4, above `B` (5/16).
5. **Per-family floor and ceiling facts** (counts out of 4, from the report): `event_tiebreak` 0/4 in all seven conditions, including no memory, so it
   contributes nothing to any contrast; `cursor_ties` A 1/4, `C0-t1` 3/4, every other condition 0/4; `retry_backoff` A 4/4, `B-nopoison` 4/4, `C1-t3` 4/4,
   `B` 1/4, `C0-t3` 1/4, `C1-t1` 1/4, `C0-t1` 0/4. The test split was not edited after the run, and these families were not examined further.

## 8. What the data do and do not support

| Statement | Status |
|---|---|
| `C1` at `trial_cap=1` improves pass rate over plain vector | **Not supported** (gate not passed; descriptive difference −0.06) |
| Memoire in any configuration improves pass rate over no memory | **Not shown** (`C1-t1` − `A` −0.03; `B` − `A` +0.03). `C1-t3` pooled 0.56 vs `A` 0.38 is unregistered and descriptive only |
| Plain-vector lessons help on this bank | **Not shown** here; the dev pilot's gap did not replicate |
| The poison lesson hurts plain vector | Suggestive only; intervals include zero; one family drives the difference |
| Outcome feedback helps when exposure is matched (`trial_cap=3`) | Exploratory; one interval excludes zero (uncorrected), one family contributes 43–60% |
| `trial_cap=3` should replace the default `1` | **Not justified** by this run (post-hoc comparison on the same 8 families) |

Permitted positioning: Memoire is a local, zero-LLM-cost, MCP-shared memory engine with a trust mechanism that demotes notes that fail; whether that
improves agent accuracy has been tested once, with a negative-or-inconclusive result, and is stated that way.

## 9. Threats to validity

- One model, one window, one seed (dev preflight and the s1 identical-input check both showed 0 flips, so noise at temperature 0 looked small), one bank author. Lessons contain the answer
  and the poison is wrong by design, so poison harm here is susceptibility to adversarial-by-design lessons, not a rate of bad lessons in practice.
- n = 8 families: the exact test has very little power (about 0.02 at a realized effect of 0.14 with one replicate, `eval/PREREGISTRATION_8d.md` section 4), and the bootstrap interval is optimistic.
- Gate failure: the bank did not separate memory from no memory for this model, which limits what any contrast on it can show.
- `event_tiebreak` is a floor (0/4 everywhere) and `retry_backoff` is a ceiling for no memory (4/4); a few families carry most of the signal.
- Forking paths disclosed in pre-registration section 10 (tagged wording, family scope, choice of primary arm, the `s3-s4` window, dev-calibrated gold wording); `trial_cap=3` and the
  exposure-matched contrasts were pre-registered as secondary, but their emphasis here follows the results.
- Feedback in `C1` is driven by the model's self-reported `USED_MEMORY`; wrong or missing declarations feed wrong trust.
- Heuristic engine components (trust posterior, tag wording, `MemoryPolicy` thresholds) are configurable heuristics, not validated constants.

## 10. Follow-up (proposals, not commitments)

- Any claim about `trial_cap=3`, or about a changed default, needs a **new pre-registration on new, unseen families**; the 8 test families have now been used. Authoring and
  freezing them before any model output is the cost, not API budget.
- A replication needs **more families, not more replicates**: at temperature 0 the dev preflight and the s1 identical-input check showed 0 flips, so reruns add no information.
  With one replicate `eval/power_sim.py` gives about 0.41 exact-test power at 16 families for a realized effect near 0.14 (about 0.19 at 0.10); the 0.77 in the pre-registration
  table assumes stochastic reruns and should not be relied on. A design with more than 20 non-tied families would also need the cap in `exact_sign_flip_p`
  (`MAX_FAMILIES_EXACT = 20`) revisited; the enumeration itself is cheap. The primary (overall accuracy or a post-feedback window) must be fixed in advance.
- No engine change is justified by this result. `src/` is untouched by Steps 8d-1 and 8d-2.

## 11. Deviations from the pre-registration

None reported to the drafter. (`eval/PREREGISTRATION_8d.md`, `eval/analysis.py` and `src/` are unchanged by this document; verify with
`git diff --quiet prereg-8d -- eval/PREREGISTRATION_8d.md eval/analysis.py src`.)

## 12. Reproduction

```powershell
git checkout prereg-8d
python -m pytest eval/tests
# runs: eval/PREREGISTRATION_8d.md section 9; analysis (reads files only, no model calls):
python -m eval.run_eval analyze --split test --results (Get-ChildItem eval/results/test_*.jsonl | ForEach-Object FullName) --out eval/results/test_report.json
# verify raw files against section 1:
Get-FileHash eval/results/test_*.jsonl -Algorithm SHA256 | Format-Table Hash, Path
```

Re-running `analyze` reproduces every statistic here (the bootstrap is seeded); only the `provenance` block (git head, Python version) differs.

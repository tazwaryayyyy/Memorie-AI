# Benchmark methodology (skeleton; filled after the Step 8d runs)

Do not hand-edit numbers here. Every figure comes from `eval/results/test_report.json` (produced by
`python -m eval.run_eval analyze`) and is quoted with the report's path and SHA-256. Sections marked **TBD** stay empty until
the runs are finished. The plan this document reports against is `eval/PREREGISTRATION_8d.md`.

## 1. Provenance

| Item | Value |
|---|---|
| Pre-registration tag / commit | `prereg-8d` / **TBD** (`git rev-parse "prereg-8d^{commit}"`) |
| Harness commit used for the runs | **TBD** (must have no tracked diff from the tag) |
| Model id and `model_returned` | **TBD** |
| Run dates (UTC) and seeds | **TBD** |
| Dev preflight outcome | flips **TBD** / 40 -> **TBD** seed(s) |
| Raw files and SHA-256 | **TBD** (`eval/results/test_*.jsonl`, `test_report.json`) |
| Deviations from the plan | none / see section 7 |

## 2. What was measured

Cross-session pass rate of one model on 8 held-out families x 4 sessions (32 tasks per condition and seed), with a bank of
conflicting lessons (gold, vague, wrong) and a poison lesson that enters before session 2. Conditions: A, B, B-nopoison,
C0-t1, C1-t1, C0-t3, C1-t3 (definitions and flags: pre-registration section 2). Lessons contain the answer and were written by the
project; this measures retrieval and trust under conflict, not general coding ability.

## 3. Primary result

**TBD.** Quote verbatim from the report: gate status, `Delta = accuracy(C1-t1) - accuracy(B)` over s1-s4, family-bootstrap interval
(flagged anti-conservative at n = 8), exact two-sided sign-flip p, family tally, and `primary_verdict`.
Then one sentence stating the permitted claim for that verdict (pre-registration section 3), and nothing broader.

## 4. Secondary and descriptive results

**TBD.** Table of the pre-registered secondary contrasts (no p-values by design) and per-condition pass rates by window.
Label the `s3-s4` window as dev-informed exploratory. Report the exposure-matched pair (`t3`) next to the `t1` pair so the
`trial_cap` cost is visible separately from any trust effect.

## 5. Poison

**TBD.** Per condition: exposure rate, failure rate given poison shown, failures with poison declared, last-session exposure.
Poison harm = `accuracy(B-nopoison) - accuracy(B)` over s2-s4 (includes displacement from top-3 of 4 vs 3 of 3).
State the s1 identical-prompt noise floor (flips / pairs) next to it. Declared use is self-report, not causal attribution.

## 6. Threats to validity

- Single model, single window, single bank author; lessons contain the answer.
- n = 8 families: the exact test has little power and the percentile bootstrap is optimistic (pre-registration section 4).
- Forking paths disclosed in pre-registration section 10 (tagged wording, family scope, primary arm choice, `s3-s4`, dev-calibrated gold wording).
- API nondeterminism at temperature 0; no API seed; possible silent model updates (checked via `model_returned`).
- Self-reported note usage (`USED_MEMORY`) drives feedback in C1; wrong or missing declarations feed wrong trust.
- Heuristic engine components (trust posterior, tag wording, `MemoryPolicy` thresholds) are configurable heuristics, not validated constants.

## 7. Deviations from the pre-registration

**TBD.** Copy the append-only table from pre-registration section 11, or write "none".

## 8. Reproduction

```powershell
git checkout prereg-8d
python -m pytest eval/tests
# runs: pre-registration section 9; analysis:
python -m eval.run_eval analyze --split test --results (Get-ChildItem eval/results/test_*.jsonl | ForEach-Object FullName) --out eval/results/test_report.json
```

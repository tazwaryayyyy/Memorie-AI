# eval/ - hermetic real-LLM evaluation harness (Step 8c)

Not part of the shipped library. Answers one question honestly: does Memoire's
trust-ranked, feedback-updated memory improve a real coding agent over no memory
and over plain vector memory? Nothing here is evidence of that until the arms
have been run on the held-out split.

## Arms
| id | name | retrieval | writes |
|----|------|-----------|--------|
| A  | `no_memory` | none (no notes section, no `USED_MEMORY` instruction) | none |
| B  | `plain_vector` | `recall_plain` cosine top-k | none |
| C0 | `memoire_static` | `recall` (trust-ranked, `trial_cap`) | none |
| C1 | `memoire_feedback` | `recall` | declared notes only: pass -> `reinforce_declared`, fail -> `penalize_if_used` |

Retrieval query = the task spec. By default notes are shown identically in B/C0/C1 as bare text
(`--prompt-policy plain`); `--prompt-policy tagged` adds engine outcome tags to C0/C1 only (Step 8c-5
below). `contradiction` defaults to `off` so the frozen bank is not rewritten.

## Task suite (single-choice convention families, Steps 8c-3/8c-4)
Each family asks for ordinary, generally-correct code and leaves exactly ONE choice open, a
genuine 2-3 way engineering decision (`eval/tasks/design.py` lists the dimension, the options and
which is gold). The spec fixes everything else and says the team's choice "is not documented in
this repository". The visible tests accept every legitimate option; the hidden tests check one.
Gold lessons state the choice, `wrong` lessons state the common alternative, `vague` lessons say
nothing usable, `poison` is a `wrong` variant naming the session-2 function. Because visible tests
cannot tell gold from wrong, only outcome feedback (C1) can separate them across sessions.

History: step 8c-1 used textbook traps (Arm A 0.93, too easy); step 8c-2 bundled 2-3 independent
conventions per task (Arm A 0.00: the chance of guessing all of them was ~0, and one hidden test
wrongly required both locks to be taken before an error check). Now one choice per family.

| split | family | open choice | gold |
|-------|--------|-------------|------|
| dev | wire_timestamp | UTC designator | literal `Z` (not `+00:00`) |
| dev | lock_ordering | global lock order | lower id first (not argument order, not higher first) |
| dev | fee_rounding | exact half-cent ties | to even (`ROUND_HALF_EVEN`), not half-up |
| dev | cache_key_format | which key parts are case-folded | tenant only (not none, not all) |
| dev | http_status | status for `ValueError` | 422, not 400 |
| test | money_wire | unit of `amount` | integer minor units, not decimal string |
| test | percentile_method | percentile between ranks | nearest-rank, not linear interpolation |
| test | retry_backoff | growth of the wait | doubling, not linear |
| test | cursor_ties | order of rows tied on ts | id ascending, not descending |
| test | semver_bump | breaking change while major is 0 | minor bump, not major |
| test | slug_separator | word separator | underscore, not hyphen |
| test | event_tiebreak | events with equal ts | arrival order, not by seq |
| test | histogram_bucket | value equal to an edge | next bucket up, not its own |

### How the gold choices were picked (read before trusting any result)
- **Dev gold choices were calibrated on dev model behaviour** (the 8c-2 pilot's failure counts showed
  the model's defaults: sorted-by-id locks, 400/504 statuses). Lock ordering is therefore a
  near-control (the default is right; the wrong lesson can only hurt), wire_timestamp is a coin flip,
  and the other three are expected to fail without memory. These are unverified priors, not results.
- **Test gold choices follow a rule fixed before any model output on that split exists:** odd position =
  the choice most common in industry practice, even position = a less common but legitimate one
  (`DESIGN[...]["gold_is"]`, enforced by `eval/tests/test_tasks.py`). Nothing on `test` was tuned.
- Every non-gold option is implemented (`naive` + `alternates`) and must pass visible and fail hidden,
  which is what "visible tests are neutral across options" means here.
- Arm A is bounded by guessing, not model skill, so the [0.15, 0.45] target is a screen. With 5 families
  x 4 near-identical sessions, results arrive in steps of 0.2 (all four sessions of a family usually agree).
- Gold lessons contain the answer, so B/C0/C1 mostly test retrieval plus trust under conflict. This is
  not evidence about general bug avoidance. Process isolation only: candidate code runs as your OS user.

### Sessions: 4 per family (Step 8c-4)
A row with any feedback is *verified*: it bypasses `trial_cap` and is hidden when `2f > 3s + 2` (two
failures with no success); it reaches `follow` when `3s >= 8 + 7f` (three successes). Three sessions
could not show either mechanism inside an episode, so every family now has `s1..s4`
(dev 5 x 4 = 20 tasks, test 8 x 4 = 32; `s4` re-skins the same choice). With the poison ingested
before `s2`: offered as a `trial` at s2, as a verified `hint` at s3 (f=1), hidden at s4 (f=2 after
s3). Gold is `trial` at s1, `hint` at s2 and s3, `follow` at s4. `test_poison_timeline.py` (rules
mirror) and `test_four_sessions.py` (the real run loop) pin this; the native tests check the engine.

### Cross-family leakage and the relevance floors (src/quality.rs `ScoringConfig`)
Before 8c-4 only `trial_min_cos` existed (default 0.0, not reachable from Python) and verified rows had
no floor. In a store shared by several families this leaks in two ways, both reproduced offline in
`test_four_sessions.py`: (1) a verified row from another family is offered whatever its similarity;
(2) once a family's own unverified lessons are used up, `trial_min_cos = 0` hands the trial slot to an
unrelated family's lesson, and feedback then lands on lessons before their own first session.
At the eval settings (`top_k=3, trial_cap=1`) this is prompt noise and misattributed feedback, not
displacement of the relevant trial row; displacement happens at `top_k=1` or a larger `trial_cap`.
- New `verified_min_cos` (default 0.0 = previous behaviour); `trial_min_cos` is now exposed to Python too.
  Neither applies to `include_hidden=True`. Flags: `--verified-min-cos`, `--trial-min-cos`.
- `--store-scope family` (a fresh store per family) is the isolation control; default `shared`.
- Label-blind floor rule: `diagnose` prints the lowest own-family and highest other-family cosine over
  all task specs and lessons (family membership only, never correctness). If the gap is positive, the
  floor is its midpoint, derived on dev and applied unchanged to test; if not, there is no clean floor
  and `--store-scope family` is the only clean control.

### C1 and the quality gate: pre-registered rule
All memory arms (B, C0, C1) ingest with `--admission admit_all`; no tau_q is tuned, because the
hypothesis is outcome-attributed trust, not an ingest gate. On the dev bank every lesson scores below
0.30 (engine default tau_q = 0.50 would hide all 20: the cold-start deadlock). The scores also track how
the bank was written, not whether a lesson is right: mean importance is gold 0.178 > vague 0.157 > wrong
0.126 > poison 0.108, and the two lowest-scoring lessons are a `wrong` (0.019) and a `poison` (0.025).
So any threshold, even a label-blind pooled percentile such as p10, hides only bad lessons and would hand
whichever arm uses the gate a win from authoring style. Do not run a gate arm on this bank; if one is
run anyway, apply the same tau_q to B, C0 and C1, and report `shadowed_keys` (recorded per run) by kind
as a confound, not a result.

## Step 8c-5: ranking facts, prompt policy, and the pre-seeded bank

### What ranks the notes (read before interpreting a `score=` in a log)
`Store::search_filtered` (src/store.rs) sorts by `final_score = 0.75*cos + 0.20*trust + 0.05*recency`;
`effective_weight` is output-only and plays no part. `trust = (2+s)/(4+s+f)`, so one success is 0.60, one
failure 0.40, and the `score` printed in a record is already this blend, **not** the cosine. Consequences:
- A `trust=0.40` row beats a `trust=0.60` row only if its raw cosine is higher by more than
  `0.20*0.20/0.75 = 0.053` (recency is equal inside one run). The pilot logs `0.671/0.40` over `0.658/0.60`
  and `0.614/0.40` over `0.612/0.60` therefore imply cosine gaps of about 0.07 and 0.06, not 0.002 (derived
  by arithmetic from those logged scores, recency assumed equal). Nothing there shows the sort ignoring trust;
  it shows trust being a weak term: one outcome moves the score by only 0.02, less than ordinary cosine
  spread between paraphrased lessons. Whether that weight should be larger is an open design question, to be
  settled by an ablation, not by tuning on the dev pilot.
- Records now carry `cos` (raw cosine, from the read-only `recall_plain`) next to `score`, and
  `summarize --detail` prints `shown=kind:exposure:trust:cos` per note in prompt order, so this can be read
  off a run instead of inferred. Runs recorded before this change have no `cos` (`?`).
- `RecallAttribution.rank_score` (src/lib.rs `recall_explained`) is a display composite that does not drive
  ordering; it is not touched here (public field, MCP-visible; separate step).

### `--prompt-policy tagged`
C0/C1 notes are prefixed with a tag derived **only** from the engine's `(exposure, trust)`: `[UNTESTED]`
(no outcome recorded), `[TRUST 0.60: worked more often than it failed]`, `[TRUST 0.40: failed more often than
it worked]`, `[TRUST 0.50: mixed record]` (`prompt.tag_for`; the wording is exact because `trust > 0.5` iff
`s > f`). Tags never use lesson text or kind. It deliberately does not call `MemoryPolicy.evaluate`: its
default bands put trust 0.40, 0.50 and 0.60 all in `hint`, which would hide the difference being tested. The
word "verified" is avoided because one success is weak evidence. A and B are unaffected (B has no trust to show).
- C0 never writes feedback, so under `tagged` every C0 note reads `[UNTESTED]`: C0-tagged is a **null control**
  for the wrapper text itself, not a treatment.
- The clean contrast is **C1-plain vs C1-tagged** on the same seeds (retrieval is identical, only rendering
  differs). C0-plain vs C1-tagged would confound feedback-driven retrieval with presentation.
- `prompt_policy` enters the run fingerprint, so use a new `--out` per policy; JSONL files recorded before this
  step no longer resume (they still summarize).

### Why the pre-seeded bank favours B at s1 (and what to run about it)
The bank at s1 is gold + vague + wrong for the family (poison is ingested before s2, not at s1). With every
row unverified, `select_with_trial_cap` returns at most `trial_cap=1` rows, so C0/C1 show **one** note at s1,
the top-cosine one, which is gold only if gold happens to be nearest. B (`top_k=3`) shows the three nearest
rows, i.e. the whole family bank (exact with `--store-scope family`; in a shared store it is the family bank
whenever `diagnose` reports positive own/other cosine separation). B therefore always sees gold at s1 and
C0/C1 only sometimes, which says nothing about trust. Check it with `summarize --detail` (`shown=` at s1).
- Available now, no code: run C0/C1 with `--trial-cap 3` (exposure-matched to B at s1).
- Not implemented, recommended next (needs an owner decision, since the harness supplies the lesson):
  **S1** empty store, gold ingested only after the first failure; **S2** poison incumbent from the start,
  gold ingested after the first failure. They test learning and poison recovery without a 3-note lottery, but
  the oracle-written gold lesson must be labelled as such in any report. Keep the current bank as the
  conflict-discrimination condition; do not replace it.

## Step 8d: pre-registered held-out run and analysis

Plan, decision rule, budget and freeze procedure: `eval/PREREGISTRATION_8d.md`. Method, results and caveats of the held-out run:
`eval/BENCHMARK_METHODOLOGY.md` (primary endpoint not met; no accuracy-improvement claim).
- `--no-poison` (memory arms only; rejected for A): the family's poison lesson is never ingested. Pre-registered for B only, as the
  poison-harm counterfactual. B and B-nopoison hold identical stores at s1, so their s1 prompts must be byte-identical; `analyze`
  checks that. It is part of the run fingerprint only when set, so older result files keep resuming.
- `python -m eval.run_eval analyze --split {dev,test} --results FILES... [--out report.json]` reads results files only (no model calls).
  Conditions are read from the records (arm, `trial_cap`, `prompt_policy`, `no_poison`), compared paired by family, and exactly one
  contrast (C1-t1 vs B over s1-s4) gets an inference (exact two-sided sign-flip). Everything else is descriptive by construction.
  Exit code 1 means an integrity check failed; 2 means unreadable or duplicated input.
- `analyze --preflight --split dev --results FILES...` applies the seed-count rule to dev reruns of B and C1-tagged at seeds 0 and 1
  (pooled flips <= 1 of 40 -> one seed on test, else two).
- `python -m eval.power_sim` prints the planning table quoted in the pre-registration (assumptions are in the file; it is not an analysis).

## Checkpointing
`--out file.jsonl` is append-only; key = `(arm, family, session, seed)`. Re-running the same command
skips recorded runs; for C1 their feedback is replayed into a fresh store first. A torn last line is
dropped. Every record carries a fingerprint of model, base URL, temperature, top-k, trial cap,
contradiction mode, families, sessions, task text, lesson text and prompt template; a mismatch aborts
(use a new `--out`).

## Commands (PowerShell, repo root, venv active)
    pytest eval/tests
    python -m eval.run_eval run --split dev --arms A --model <pinned-id> --rpm 13 --max-calls 25 --out eval/results/dev_A.jsonl --dry-run
    $env:GEMINI_API_KEY = "<key>"
    python -m eval.run_eval run --split dev --arms A --model <pinned-id> --rpm 13 --max-calls 25 --out eval/results/dev_A.jsonl
    python -m eval.run_eval summarize --results eval/results/dev_A.jsonl --detail
    python -m eval.run_eval diagnose --split dev --out eval/results/dev_exemplar_cos.json   # needs native build

`--seeds` only keys/replicates runs locally. The HTTP `seed` field is NOT sent unless you pass
`--send-api-seed` (Google's OpenAI-compat endpoint rejects it with HTTP 400; OpenAI/Ollama-style
endpoints accept it). The flag is part of the run fingerprint. Default `--max-tokens` is 4096 because
thinking models can count reasoning tokens against it.

`--max-calls` counts HTTP attempts including 408/429/5xx retries (each call retries up to `--max-retries`
=8 times with exponential backoff and jitter), so leave headroom: a 15-call pilot needs about 40.
`diagnose` also writes each lesson's ingest `importance_base` (quality score) for tau_q selection; the
gate sees text features only, so do not fit tau_q to separate gold from wrong.

`diagnose` also prints the label-blind relevance separation (own-family vs other-family cosine).

Exit codes: 0 ok, 2 usage/config, 3 LLM/fingerprint error, 4 `--max-calls` reached (resumable).

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

Retrieval query = the task spec. Notes are shown identically in B/C0/C1 (no `MemoryPolicy`
gating yet). `contradiction` defaults to `off` so the frozen bank is not rewritten.

## Task suite (single-choice convention families, Step 8c-3)
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
  x 3 near-identical sessions, results arrive in steps of 0.2: expect 0, 3 or 6 passes per family pair.
- Gold lessons contain the answer, so B/C0/C1 mostly test retrieval plus trust under conflict. This is
  not evidence about general bug avoidance. Process isolation only: candidate code runs as your OS user.

### How many sessions does feedback need? (from src/store.rs, src/quality.rs)
A row with any feedback is *verified*: it bypasses `trial_cap` and has **no similarity floor**; it is
hidden when `2f > 3s + 2` (two failures with no success) and reaches `follow` when `3s >= 8 + 7f`
(three successes). `eval/tests/test_poison_timeline.py` encodes the timelines; the native test checks
them against the real engine.
- Poison ingested before session 2 gets f=1 after s2 and f=2 after s3: it is hidden only for a
  4th session, so **a 3-session family cannot show poison suppression**. It stays on offer at s3 as a
  verified `hint`, and a once-failed row keeps its slot while untried rows are capped.
- Poison present from session 1 is hidden before s3. A `wrong` lesson used and failed in s1 and s2 likewise.
- Gold is `trial` at s1, `hint` at s2 and s3, and `follow` only at a 4th session.
- Verified rows from earlier families can fill the verified slots at later families' s1.

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

Exit codes: 0 ok, 2 usage/config, 3 LLM/fingerprint error, 4 `--max-calls` reached (resumable).

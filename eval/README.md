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

## Task suite (convention families, Step 8c-2)
Textbook traps (mutable defaults, `ROUND_HALF_UP`, ...) were dropped: a strong model already knows
them, so Arm A sat at ceiling (0.93 on gemini-3.1-flash-lite, dev). Every family now asks for
ordinary, generally-correct code, and the spec says the team has an established convention for one
aspect that "is not documented in this repository". The visible tests accept every plausible
convention; the hidden tests check exactly one. Pretraining cannot reveal it; only a recalled lesson
or luck can.

- `dev`: 5 families x 3 sessions: wire_timestamp, lock_ordering, fee_rounding, cache_key_format,
  http_status. `test`: 8 families x 3 sessions, locked behind `--allow-test-split`: money_wire,
  idempotency_key, retry_backoff, cursor_ties, semver_bump, slug_rules, event_tiebreak, histogram_bucket.
- Each task: stub `solution.py`, agent-visible `test_visible.py`, grading-only `test_hidden.py`
  (stdlib `unittest`, 10 s timeout, scrubbed env, temp dir). The naive solution (the most common
  default) passes visible and fails hidden by assertion.
- Bank per family: `gold` (states the convention), `vague`, `wrong` (states the common alternative),
  all ingested up front; `poison` (a `wrong` variant naming the session-2 function) is ingested
  immediately before session 2. Visible tests cannot tell gold from wrong, so only outcome feedback
  (C1) can separate them across sessions. That is the hypothesis under test.
- Lessons are single-chunk and avoid the wording of the Rust prototype exemplars (word-level guard in
  `eval/tests/test_tasks.py`; the embedding check is `diagnose`).
- Sessions in a family re-skin the same convention (names, nouns, values): "same rule, new wording".

### What this design does and does not measure
- It measures whether a memory system delivers a project convention the model cannot know, and
  whether trust feedback demotes a confidently wrong convention. It does NOT measure general bug
  avoidance, and a gain here is not evidence about textbook coding errors.
- Arm A is bounded by guessing, not by model ability: some conventions have a small choice set
  (histogram edge side, 422 vs 400), others are effectively unguessable (cache key layout,
  idempotency digest). Per-family Arm A rates are therefore set by design. Treat the aggregate
  [0.15, 0.55] target as a screen, and read per-family rates.
- The gold lesson contains the answer, so B/C0/C1 largely test retrieval plus trust under conflict.
- Process isolation only: candidate code runs as your OS user.

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
    python -m eval.run_eval summarize --results eval/results/dev_A.jsonl
    python -m eval.run_eval diagnose --split dev --out eval/results/dev_exemplar_cos.json   # needs native build

`--seeds` only keys/replicates runs locally. The HTTP `seed` field is NOT sent unless you pass
`--send-api-seed` (Google's OpenAI-compat endpoint rejects it with HTTP 400; OpenAI/Ollama-style
endpoints accept it). The flag is part of the run fingerprint. Default `--max-tokens` is 4096 because
thinking models can count reasoning tokens against it.

Exit codes: 0 ok, 2 usage/config, 3 LLM/fingerprint error, 4 `--max-calls` reached (resumable).

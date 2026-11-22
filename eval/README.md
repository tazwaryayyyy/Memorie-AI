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

## Task suite
- `dev`: 5 families x 3 sessions. `test`: 8 families x 3 sessions (locked behind `--allow-test-split`).
- Each task: stub `solution.py`, agent-visible `test_visible.py`, grading-only `test_hidden.py`
  (stdlib `unittest`, 10 s timeout, scrubbed env, temp dir). Naive solutions pass visible, fail hidden.
- Sessions in a family re-skin the same trap (names/domain/values). They measure "same trap, new
  wording", not transfer to structurally new bugs.
- Bank per family: `gold`, `vague`, `wrong` ingested up front; `poison` is ingested immediately
  before session 2. Lessons are single-chunk and avoid the wording of the Rust prototype exemplars.
- Process isolation only: candidate code runs as your OS user. Use a throwaway account/container
  if that matters to you.

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

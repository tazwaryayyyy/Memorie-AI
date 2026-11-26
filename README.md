# Memoire

> Local-first semantic memory engine and MCP server for AI coding agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/tazwaryayyyy/Memorie-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/tazwaryayyyy/Memorie-AI/actions)

> **Status: pre-release (`0.1.0`, alpha). Nothing is published yet.**
> `memoire-engine` (crates.io and PyPI) and `memoire-mcp` (PyPI) are the intended
> package names but are **not published**. Install from a source checkout only. The
> unrelated PyPI project `memoire` is a different package; do not install it expecting
> this one.
>
> The central idea (trust that rises or falls with task outcomes) is **not yet
> empirically validated**. A pre-registered held-out run (Step 8d) did not show an
> accuracy improvement: see [Evidence so far](#evidence-so-far) and
> [Known Limitations](#known-limitations).

Memoire stores short lessons from agent runs in a local SQLite file, retrieves them by
embedding similarity, and keeps a per-memory trust value that you can move up or down
by reporting whether a recalled lesson helped. Several agents or tools can share one
database through namespaces.

## Why

AI coding agents repeat mistakes across sessions:

```text
Task 1 -> use float for money -> tests fail
Task 2 -> use float for money again -> tests fail again
```

The intended loop:

```text
Task 1 fails  -> remember("Never use float for money. Use Decimal.")
Task 2 starts -> recall("billing precision") -> trust-gated context for the agent
Task 2 ends   -> reinforce_if_used(...) or penalize_if_used(...)
```

Whether this loop improves a real coding agent was tested once (Step 8d: one model, 8
held-out task families). The pre-registered primary comparison did not show an
improvement, so **no accuracy-improvement claim is made** (see
[Evidence so far](#evidence-so-far)). As of Step
8b-1 a fresh lesson enters as an unverified `trial` memory and is offered in a bounded
trial slot (see Trust and Policy and Known Limitations), verified by deterministic engine
tests only.

## What It Does

- **Storage**: SQLite (bundled via `rusqlite`), one local file, no hosted service.
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim) via `fastembed` + ONNX Runtime, run
  locally on CPU. The model is downloaded on first use (see [Model cache](#model-cache-and-offline-use)).
- **Deduplication**: BLAKE3 fingerprints; exact-content duplicates are dropped.
- **Quality scoring at ingest**: three semantic features (actionability, consequence,
  reusability: cosine to fixed centroids) weighted `3/7`, `5/14`, `3/14`. `novelty` and
  `evidence` are still computed and stored as metadata but weigh 0. With the default
  `AdmissionMode::AdmitAll` the score does not gate exposure; `QualityGate` restores the
  `>= 0.50` live / else hidden `shadow` split. Because the weights now sum over centroid
  cosines only, realistic lessons often score below 0.50, so `QualityGate` needs its own
  calibration before use.
- **Trust**: a Beta posterior `(2 + s) / (4 + s + f)` over attributed reinforcements and
  failures, independent of ingest scores, time-regressed toward 0.5. Constants are hand-set.
- **Contradiction handling**: when a new memory is close to an existing one (cosine
  `>= 0.80`) and the two are judged contradictory, the loser is hidden: archived if an ONNX
  NLI model decided, quarantined (recoverable) if only the lexical heuristic did.
- **Shadow corroboration** (`AdmissionMode::QualityGate` only): a chunk that independently
  restates a hidden `shadow` row (cosine `>= 0.92`, no contradiction, from a different
  `remember` call) promotes that row to `trial` instead of inserting a duplicate, and the
  promoted id is returned. It records no reinforcement. Exact replays do not count.
- **MMR recall**, optional cross-encoder reranking, **namespaces**, JSON
  **export/import**.
- **Interfaces**: Rust library, Python SDK (PyO3), MCP server, CLI, C FFI,
  experimental local HTTP server.

## Support Matrix

| Component | Path | v0.1 status |
|---|---|---|
| Rust core (`memoire-engine`, lib name `memoire`) | `src/` | **Supported** (not published) |
| Python SDK (`memoire-engine`, `import memoire_engine`) | `bindings/python/` | **Supported** (not published) |
| MCP server (`memoire-mcp`, `python -m memoire_mcp`) | `mcp-server/` | **Supported** (not published) |
| CLI (`memoire-cli`) | `src/bin/cli.rs` | Secondary: tested, built from source |
| C FFI (`libmemoire`, `include/memoire.h`) | `src/ffi.rs` | Secondary: header/ABI parity is tested; the Node and Go wrappers on top of it are not maintained. See [docs/FFI_GUIDE.md](docs/FFI_GUIDE.md) |
| HTTP server (`memoire-server`) | `src/bin/memoire_server.rs` | **Experimental, local-only** (loopback bind, no option to change it) |
| Dashboard (Next.js) | `dashboard/` | **Experimental, local-only** |
| Node.js binding | `bindings/node/` | **Unmaintained / postponed** |
| Go binding | `bindings/go/` | **Unmaintained / postponed** |
| WASM | `--features wasm` | Only the pure-Rust `quality` module and chunker build; no store or embedder; `quality::now_ts()` uses `SystemTime::now()` and is expected to panic at runtime on `wasm32-unknown-unknown`. **Postponed.** |

## Build and Install (from source)

Requirements: a Rust toolchain and a C linker (MSVC on Windows). No minimum
supported Rust version is declared in `Cargo.toml`; the maintainer's last full local
verification used Rust 1.100.0 on Windows x86_64 (MSVC). The build downloads a
prebuilt ONNX Runtime (see [docs/RELEASE.md](docs/RELEASE.md)), so the **build** needs
network access unless `ORT_LIB_LOCATION` is set.

```bash
git clone https://github.com/tazwaryayyyy/Memorie-AI
cd Memorie-AI
cargo build --release
```

| Output | Path |
|---|---|
| CLI | `target/release/memoire-cli` (`.exe` on Windows) |
| HTTP server (experimental) | `target/release/memoire-server` |
| Shared library (Linux / macOS / Windows) | `target/release/libmemoire.so` / `libmemoire.dylib` / `memoire.dll` |

### Python SDK

Python >= 3.10, inside a virtualenv:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install maturin
cd bindings/python
maturin develop --release
```

```python
from memoire_engine import Memoire, MemoryPolicy

with Memoire("agent.db", namespace="billing-agent") as m:
    m.remember("Never use float for money. Use Decimal for billing calculations.")
    memories = m.recall("billing precision", top_k=5)
    decisions = MemoryPolicy().evaluate(memories)
    context = MemoryPolicy().inject_context(decisions)
```

`remember()` and `remember_ids()` differ:

| Call | Returns |
|---|---|
| Python `Memoire.remember(text)` | `int`: number of chunks stored. `0` for duplicates or rejected chunks. |
| Python `Memoire.remember_ids(text)` | `list[int]`: ids of stored chunks (`[]` for duplicates or rejected chunks). Use this when you need ids for `reinforce_if_used` / `penalize_if_used`. |
| Rust `Memoire::remember(&str)` | `Result<Vec<i64>>` (ids) |
| C FFI `memoire_remember` | `int` (`>= 0`, `-1` on error) |

### Rust

Not on crates.io. Use a path or git dependency; the library target is named `memoire`:

```rust
use memoire::Memoire;

let m = Memoire::new("agent.db")?;
let ids = m.remember("Never use float for money. Use Decimal for billing calculations.")?;

let memories = m.recall("billing precision", 5)?;
for mem in &memories {
    println!("[score={:.3} trust={:.3} state={}] {}", mem.score, mem.trust, mem.state, mem.content);
}

if let Some(top) = memories.first() {
    m.reinforce_if_used(top.id, "Implemented billing with Decimal.", true)?;
}
```

### CLI

Global flags (`--db`, `--ns`/`--namespace`) must come **before** the subcommand; unknown
or misplaced flags are rejected. `--top` is bounded to `1..=1000`.

```bash
memoire-cli --db agent.db remember "Never use float for money."
memoire-cli --db agent.db recall "billing precision" --top 3
memoire-cli --db agent.db --ns billing-agent export --output billing.json
memoire-cli --db agent.db --ns billing-agent import billing.json            # content only, trust reset
memoire-cli --db agent.db --ns billing-agent import billing.json --preserve-trust  # only for your own exports
```

## Model Cache and Offline Use

The default embedder (`all-MiniLM-L6-v2`, about 87 MiB, roughly 91 MB) is **downloaded from Hugging Face
on first use**. Memoire is therefore **not offline on a cold start**; it needs no
network once the model is cached.

Cache location is decided by the pinned `fastembed` (`=5.13.2`) crate, in this order:

1. `$HF_HOME`, if set
2. else `$FASTEMBED_CACHE_DIR`, if set
3. else `./.fastembed_cache`, **relative to the current working directory**

So with neither variable set, every working directory (and every MCP workspace) gets its
own copy and its own download. Set `FASTEMBED_CACHE_DIR` or `HF_HOME` to share one cache.

`memoire-cli cache-models` constructs the default embedder once, which populates that
cache. It does not fetch the optional reranker (`BAAI/bge-reranker-base`, downloaded when
`with_reranker()` / `recall_reranked` is first used) or an NLI model (you supply the
ONNX file path yourself via `with_nli_model`).

## Trust and Policy

Each recalled `Memory` carries `score`, `trust`, `uncertainty`, `state` and `exposure`.
States: `trial` and `active` are live; `shadow`, `probationary` and `quarantined` are
hidden from default recall. `exposure` is `follow` (trust >= 0.70), `hint`, `trial`
(no feedback yet), `hidden`, or `plain` (`recall_plain`).

- `score` (ranking) = `0.75 * cosine + 0.20 * trust + 0.05 * recency_bonus`.
- `trust` is a Beta posterior over attributed feedback: `(2 + s) / (4 + s + f)` with
  `s = reinforcement_count` and `f = failure_count` (a prior of four pseudo-observations
  at 0.5), regressed toward 0.5 by `exp(-decay_rate * days_since_last_feedback)` (default
  `0.01`). `uncertainty = 4 / (4 + s + f)`. It does not depend on any ingest-time score:
  every new memory starts at exactly `0.5`.
- State transitions use integer rules on `(s, f)`: `trial -> active` when `3s >= 8 + 7f`
  (trust >= 0.70, e.g. three successes and no failures); any live state `-> quarantined`
  when `2f > 3s + 2` (trust < 0.40, e.g. two failures and no successes). Quarantined rows
  are excluded from default recall and from export, take no further feedback, and are
  restored by the operator with `unquarantine` (CLI: `memoire-cli unquarantine <ID>`),
  which clears their failure count.
- Feedback: `reinforce_if_used` requires text attribution (the memory's distinctive tokens
  contained in the output at `containment_threshold`, default 0.35, or cosine >= 0.75);
  `reinforce_declared` skips the text gate for callers that know which memories the agent
  used. Both count one vote per memory per session id. `penalize_if_used` records one
  failure per id and no longer rewrites `importance_base` or `confidence`.

These constants (the prior strength, the thresholds, `trial_cap`) are hand-chosen defaults,
not fitted or validated.

`memoire_engine.MemoryPolicy` turns `trust` into an action:

| Policy | FOLLOW | HINT | IGNORE |
|---|---|---|---|
| Default (`MemoryPolicy()`) | `trust >= 0.70` | `trust >= 0.40` | `trust < 0.40` |
| Strict (`MemoryPolicy(strict=True)`) | `trust >= 0.80` | `trust >= 0.50` | `trust < 0.50` |

`inject_context()` only emits text for FOLLOW and HINT decisions. The four thresholds are
defined once in Rust (`memoire::quality::{FOLLOW,HINT,STRICT_FOLLOW,STRICT_HINT}_THRESHOLD`)
and read from the native extension by `MemoryPolicy`
(`memoire_engine.policy_thresholds(strict=False)`) and the MCP server. Comparisons allow
`1e-6` of slack because a float32 trust of exactly `0.70` reaches Python as slightly
below `0.70`.

## Core API

### Rust

```rust
let mut m = Memoire::new("agent.db")?;

let ids       = m.remember("lesson text")?;                 // Vec<i64>
let results   = m.recall("query", 5)?;
let diverse   = m.recall_mmr("query", 5, 0.5)?;
let reranked  = m.recall_reranked("query", 5)?;             // needs .with_reranker()
let explained = m.recall_explained("query", 5)?;            // Vec<RecallAttribution>

m.reinforce_if_used(ids[0], "agent output", true)?;         // -> bool
m.penalize_if_used(&[ids[0]], 1.0)?;                        // -> Vec<PenaltyOutcome>
m.recompute_prototypes()?;
m.forget(ids[0])?;

// Runs maintenance_pass() every 300 s or after 10 inserts. Uses the ambient Tokio
// runtime if there is one, otherwise a private one-worker runtime. Stopped on Drop.
m.start_background_maintenance(300, 10);

let snapshot = m.export_namespace()?;
let target   = Memoire::new_ns("backup.db", "billing-agent")?;
target.import_namespace(&snapshot)?;   // ImportTrust::Reset by default
```

### Python

```python
with Memoire("agent.db") as m:
    n         = m.remember("lesson text")                   # int (chunk count)
    ids       = m.remember_ids("lesson text")               # list[int]
    memories  = m.recall("query", top_k=5)
    explained = m.recall_explained("query", top_k=5)
    ok        = m.reinforce_if_used(memories[0].id, "output", True)
    outcomes  = m.penalize_if_used([memories[0].id], failure_severity=1.0)
    m.recompute_prototypes()
    m.start_background_maintenance(interval_secs=300, threshold=10)
    deleted   = m.forget(memories[0].id)
    snapshot  = m.export_namespace()
    m.import_namespace(snapshot)                            # preserve_trust=False by default
```

`recall_mmr(query, top_k=5, mmr_lambda=0.5)` is wrapped by the high-level class. It
reranks the live candidates returned by the default recall, so the default `trial_cap` of
`1` applies to them: with no feedback history it can return at most one result. Pass
`trial_cap=` to `Memoire(...)` to widen that. `recall_reranked` exists on the native class
(`memoire_engine.memoire_engine.Memoire`) but is **not wrapped**; use Rust or the native
class if you need it. `recall(..., include_hidden=True)` returns hidden rows too, and
`reinforce_declared(memory_id, session_id=None)` records a success with no text gate.

Thread safety: the native class serializes calls behind a mutex, but the wrapper class
documents itself as not thread-safe. Use one instance per thread.

For C/FFI consumers: [docs/FFI_GUIDE.md](docs/FFI_GUIDE.md).

## Contradiction Detection

Run after each `remember()` for each new chunk, against live `trial`/`active` and hidden
`shadow` rows in the same namespace (`NliChecker` in `src/quality.rs`):

1. Cosine similarity `< nli_cosine_threshold` (default `0.80`) -> treated as unrelated.
2. If an ONNX NLI model was supplied (`Memoire::with_nli_model(path)`), its label is
   authoritative. Without one, or if inference errors, a lexical fallback is used: the two
   texts must have opposing polarity (from a small keyword lexicon) **and** at least one
   negation term (from the same lexicon, plus `not` / `no`) must appear in exactly one of
   them. Opposing polarity without that asymmetry is `Neutral`.
3. The side with more feedback evidence (`compute_gravity_radius`: posterior trust,
   reinforcement count, age) wins; between two unverified memories the newer one wins.
   Ingest importance is not consulted.
4. The loser is hidden. Only a model-backed verdict **archives** it (`archived = 1`,
   `superseded_by` set; maintenance deletes archived rows after seven days). A lexical-only
   or `Legacy` verdict **quarantines** it instead: hidden from default recall, never
   deleted, restored with `unquarantine` (CLI: `memoire-cli unquarantine <ID>`).

Caveats: the lexical fallback is a keyword heuristic and will produce false positives and
false negatives. If the new memory's polarity is neutral the check is skipped entirely, so an NLI
model never sees that pair. Terms are matched on whitespace-delimited tokens, so one followed
by punctuation (`disabled.`) is not matched. The winner rule compares the incoming memory's
ingest importance (a centroid score, often well below 0.50 with MiniLM) with that threshold,
so the existing memory usually wins. There is no automated test against a real NLI model in
this repository.

```rust
use memoire::quality::{ContradictionMode, ScoringConfig};
// Off: never supersede or archive. Legacy: cosine + stored polarity only. Nli: default.
let config = ScoringConfig { contradiction_mode: ContradictionMode::Off, ..ScoringConfig::default() };
let m = Memoire::new("agent.db")?.with_scoring_config(config);
```

## Namespaces

Namespaces partition one SQLite file; reads, writes, feedback (`reinforce_if_used`,
`penalize_if_used`), `forget`, `clear` and maintenance are scoped to the handle's
namespace:

```rust
let agent_a = Memoire::new_ns("shared.db", "agent-a")?;
let agent_b = Memoire::new_ns("shared.db", "agent-b")?;
agent_a.remember("JWT tokens expire after 15 minutes.")?;
assert!(agent_b.recall("JWT", 5)?.is_empty());
```

This is logical isolation inside one file, not a security boundary against someone who can
read the file. See [SECURITY.md](SECURITY.md).

## Export / Import

The snapshot is JSON containing `content` and metadata (`trust_ema` (legacy, ignored),
`reinforcement_count`, `importance_base`, `confidence`, `created_at`). Quarantined memories
are not exported. Embeddings are recomputed on import.
**By default, import keeps only `content` and resets trust** (`ImportTrust::Reset`,
`preserve_trust=False`, no `--preserve-trust`). Restoring metadata is opt-in and clamped
to bounds; "bounded" does not mean "verified", so use it only for snapshots you exported.

## MCP Server

Source checkout, from the repository root. Details and the full environment in
[mcp-server/README.md](mcp-server/README.md):

```bash
cd mcp-server
uv sync --locked --extra dev   # builds the native extension via maturin (needs Rust)
uv run memoire-mcp
```

Client configuration (any MCP stdio client; point at the virtualenv's script):

```json
{
  "mcpServers": {
    "memoire": {
      "command": "/ABSOLUTE/PATH/TO/.venv/bin/memoire-mcp",
      "env": {
        "MEMOIRE_DB_PATH": "/ABSOLUTE/PATH/TO/memoire.db",
        "FASTEMBED_CACHE_DIR": "/ABSOLUTE/PATH/TO/.fastembed_cache"
      }
    }
  }
}
```

(Windows: `...\.venv\Scripts\memoire-mcp.exe`.) Copy-paste setup for Claude Code
(`claude mcp add ...`) and Cursor (`mcp.json`) is in [mcp-server/README.md](mcp-server/README.md).
Setting `FASTEMBED_CACHE_DIR` (or `HF_HOME`)
avoids a per-workspace model download; the first tool call that needs the embedder may
take long enough to hit a client tool timeout while the model downloads.

Database location precedence: `MEMOIRE_DB_PATH`, then `MEMOIRE_DB`, then
`<workspace>/.memoire_path` (must resolve inside the workspace), then
`<workspace>/memoire.db`.

Tools: `memoire_health`, `memoire_remember`, `memoire_recall`, `memoire_recall_explained`,
`memoire_reinforce`, `memoire_penalize`, `memoire_batch_feedback`,
`memoire_resolve_conflicts`, `memoire_forget`, `memoire_count`, `memoire_status`,
`memoire_export`, `memoire_identify_gaps`, `memoire_namespace_health`. `memoire_clear`
and `memoire_import` are listed only when `MEMOIRE_ENABLE_ADMIN_TOOLS=1`. Tools other than
`memoire_health` accept a `namespace` parameter.

`memoire_recall` and `memoire_recall_explained` have no Python-side filter or over-fetch:
the engine's default recall returns only live (`trial` / `active`) rows, and
`include_shadow=true` is passed through as `include_hidden`, which also returns `shadow`,
`probationary` and `quarantined` rows and skips the trial cap (an operator view). Results
carry `exposure` (`follow` / `hint` / `trial` / `hidden`); `memoire_remember` returns
`memory_ids`. Feedback tools pass a per-process session id (`session_id` in
`memoire_status`), so the engine refuses to reinforce a memory twice in a row from the same
server process. The engine only compares against the *most recent* reinforcing session, so
this is not a per-(memory, session) vote count across interleaved sessions.

`MEMOIRE_TRIAL_CAP` (non-negative integer; an invalid value stops the server with exit
code `2`) overrides the engine's `trial_cap`. The engine default is `1`: with no feedback
history every memory is unverified, so a recall returns at most one result. Raise it if
agents need several results before any feedback has been recorded.
`memoire_identify_gaps` (`min_trust`, default now the engine's HINT threshold `0.40`) and
`memoire_namespace_health` (`trust_threshold` in its output) read that threshold from the
engine instead of hardcoding `0.45`.

## Experimental: HTTP Server and Dashboard

Local development conveniences, not network services. `memoire-server` binds `127.0.0.1`
only (no option to change it), default port `6779` (`MEMOIRE_SERVER_PORT`).

```bash
./target/release/memoire-server
cd dashboard && npm install && npm run dev   # binds 127.0.0.1, http://127.0.0.1:3000
```

Authentication, path restrictions and admin gating are described in
[SECURITY.md](SECURITY.md). `POST` routes need a JSON `Content-Type`:

```bash
curl -H "Content-Type: application/json" -H "Authorization: Bearer $MEMOIRE_API_TOKEN" \
     http://127.0.0.1:6779/recall \
     -d '{"db":"agent.db","ns":"default","query":"billing","k":5}'
```

(Omit the `Authorization` header if `MEMOIRE_API_TOKEN` is unset.) The `db` path must
resolve inside `MEMOIRE_ALLOWED_PATHS` (default: the server's working directory).

## Evidence so far

One pre-registered, held-out comparison has been run (Step 8d; plan in
[`eval/PREREGISTRATION_8d.md`](eval/PREREGISTRATION_8d.md), frozen at git tag `prereg-8d`;
method, tables, hashes and caveats in
[`eval/BENCHMARK_METHODOLOGY.md`](eval/BENCHMARK_METHODOLOGY.md)). Model
`gemini-3.1-flash-lite`, temperature 0, 8 task families x 4 sessions (32 tasks per
condition), one seed, conflicting lessons plus one deliberately wrong "poison" lesson.

| Condition | Pass rate |
|---|---|
| no memory (`A`) | 12/32 (0.38) |
| plain vector, top 3 (`B`) | 13/32 (0.41) |
| plain vector, no poison ingested (`B-nopoison`) | 17/32 (0.53) |
| Memoire static, `trial_cap` 1 (`C0-t1`) | 8/32 (0.25) |
| Memoire with outcome feedback, `trial_cap` 1, the default (`C1-t1`) | 11/32 (0.34) |
| Memoire static, `trial_cap` 3 (`C0-t3`) | 11/32 (0.34) |
| Memoire with outcome feedback, `trial_cap` 3 (`C1-t3`) | 18/32 (0.56) |

- **The primary endpoint was not met.** The pre-registered gate (plain-vector memory must
  beat no memory by at least 0.15) failed: 13/32 vs 12/32. Under the pre-registration no
  inference is drawn from the primary comparison; reported descriptively, `C1-t1` vs `B`
  is -0.0625 (95% family-bootstrap interval [-0.250, +0.125], exact two-sided p = 0.75, which
  is not an inference under the gate rule).
- **No accuracy-improvement claim is made** for the default `trial_cap=1` or for any other
  configuration. The dev pilot's gap between plain vector and no memory (14/20 vs 8/20, 5
  families) did not replicate.
- **Exploratory, uncorrected, one family drives several differences** (`retry_backoff`):
  removing the poison lesson helped plain vector; `trial_cap=1` scored lowest in the first
  two sessions (2/16 for both `C0` and `C1`); with exposure matched (`trial_cap=3`) outcome
  feedback scored higher than static recall. These are hypotheses for a replication on new
  families under a new pre-registration, not results. This run does not justify changing
  the `trial_cap` default.
- The comparison covers one model and lessons written by this project; the poison lesson
  is wrong by design. With 8 families the statistical test has very little power.

## Known Limitations

1. **Cold start (changed in Step 8b-1; effect on real agents unmeasured).** New lessons
   used to be stored `shadow` by a centroid quality gate with no promotion path and were
   suppressed before feedback could reach them. By default they now enter as unverified
   `trial` memories (`AdmissionMode::AdmitAll`) with a neutral trust of exactly `0.5`,
   and one recall may return at most `ScoringConfig::trial_cap` (default `1`) of them so
   outcome feedback can act. Consequences to be aware of: with no feedback history every
   memory is unverified, so a default recall returns **at most one result**; raise
   `trial_cap` (or pass `trial_cap=` in Python) for no-feedback use. Hidden `shadow` rows
   (only under `AdmissionMode::QualityGate`) are promoted by `Store::promote_shadow`;
   ingest does not call it yet. This is verified by deterministic engine tests only, not
   by any real-agent run. On the Step 8d held-out run both `trial_cap=1` conditions scored
   2/16 in sessions 1 and 2, the lowest of the seven conditions (exploratory; see
   [Evidence so far](#evidence-so-far)).
2. **Real-LLM evidence is one inconclusive-to-negative run.** `scripts/agent_behavior_benchmark.py`
   is a deterministic synthetic smoke test (it prints `[SYNTHETIC SMOKE TEST, not an LLM
   benchmark]`) and `examples/brutal_moment_demo.py` is a scripted demo; neither is evidence
   that Memoire improves a real coding agent. The only real-LLM evaluation is the Step 8d
   held-out run (one model, 8 families, one seed): its primary endpoint was not met and no
   accuracy-improvement claim is made (see [Evidence so far](#evidence-so-far)). It has not
   been replicated and does not cover other models or real-world lesson quality.
3. **Heuristic scoring.** Quality scoring (three semantic centroid features; the `evidence`
   keyword and `novelty` features are stored but no longer scored), the trust formula, the
   policy thresholds and the lexical contradiction fallback are heuristics with hand-set
   constants. They are configurable, not validated or calibrated.
4. **Not offline on cold start** (model download), and the default cache directory is
   relative to the working directory (see [Model cache](#model-cache-and-offline-use)).
5. **Not published.** No registry package exists yet. `release.yml` is set up to build wheels
   for Linux x86_64 (`manylinux_2_28`), macOS arm64 and Windows x86_64, but that pipeline has
   not been run to completion; `manylinux` compatibility of the statically linked ONNX Runtime
   is unverified. macOS x86_64, Linux aarch64 and musl have no wheel and no tested build path.
   `THIRD_PARTY_NOTICES.md` inventories every locked Rust crate and names the non-crate
   components, but does not reproduce license texts; ONNX Runtime's own third-party notices
   are referenced, not bundled.
6. **Python wrapper gap** (`recall_reranked`) and per-query `Memoire` construction in
   `memoire_engine.adapters` (reloads the embedder on each retrieval).
7. Other known debt: `resolve_contradictions_for_id` skips new memories whose polarity is
   neutral; there is no automated test with a real NLI model; the one-vote-per-session rule
   only compares against the most recent reinforcing session; some source comments contain
   mojibake.

## Tests and CI

Local:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test                                   # lib, bins, tests/*.rs, doc-tests
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
(cd bindings/python && pytest -q)            # needs `maturin develop --release` first
(cd mcp-server && uv sync --locked --extra dev && uv run pytest -q)
(cd dashboard && npx tsc --noEmit && npm test)
```

What GitHub Actions (`.github/workflows/ci.yml`) is **configured to run**, on pushes to
`main` and `fix/p0-stabilization`, on pull requests targeting `main`, and on manual dispatch.
This describes the workflow file, not a pass status; check the Actions tab for the latest run:

- Rust, on ubuntu / macOS / windows: `cargo fmt --check`, `clippy -D warnings`,
  `cargo package --list`, debug and release builds, `cargo test --lib`,
  `cargo test --bins` (CLI and `memoire-server` tests), `cargo test --test
  integration_test`, `--test oracle --test import_security --test feedback_errors
  --test ort_smoke --test ffi_header_parity --test namespace_feedback`, and doc-tests.
- Ubuntu: `cargo package --locked` (builds the packaged crate; uploads nothing) and a check
  that the packaged file list contains the migrations, header, license and notices.
- Ubuntu: release shared-library build plus exported-symbol check.
- Ubuntu: `cargo check --target wasm32-unknown-unknown --no-default-features --features wasm`
  (compile only).
- Ubuntu, Python 3.10 and 3.13: `maturin develop --release`, `bindings/python/tests`, and (3.13
  only) the offline `eval/tests`.
- Ubuntu, Python 3.11: MCP server tests (`uv sync --locked --extra dev`, `pytest`).
- Ubuntu, Node 22: dashboard `tsc --noEmit` and `npm test`.
- Ubuntu, Python 3.11: clean-venv install smoke of `bindings/python` and `mcp-server`
  (from the source tree, not from built wheels).
- Ubuntu: `scripts/check_release_metadata.py` (matching versions, `server.json`, license copies,
  and that `THIRD_PARTY_NOTICES.md` lists exactly the packages in `Cargo.lock`).

CI does **not** run any benchmark or LLM evaluation. The release workflow
(`.github/workflows/release.yml`) additionally builds wheels and smoke-tests them in clean
virtualenvs; see [docs/RELEASE.md](docs/RELEASE.md). Treat that pipeline as unproven until a
manual dispatch has completed on all three platforms.

## More Detail

- [Architecture](docs/ARCHITECTURE.md)
- [FFI guide](docs/FFI_GUIDE.md)
- [Release notes procedure](docs/RELEASE.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Author

Tazwar Ahnaf · [@TazwarEnan](https://x.com/TazwarEnan)

## License

MIT. See [LICENSE](LICENSE).

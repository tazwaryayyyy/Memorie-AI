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
> empirically validated**. See [Known Limitations](#known-limitations).

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

Whether this loop improves a real coding agent has **not been measured**. Whether a
freshly stored lesson is even surfaced is currently affected by a known cold-start
problem (see Limitations).

## What It Does

- **Storage**: SQLite (bundled via `rusqlite`), one local file, no hosted service.
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim) via `fastembed` + ONNX Runtime, run
  locally on CPU. The model is downloaded on first use (see [Model cache](#model-cache-and-offline-use)).
- **Deduplication**: BLAKE3 fingerprints; exact-content duplicates are dropped.
- **Quality scoring at ingest**: five heuristic features (actionability, consequence,
  novelty, reusability, evidence) combined with fixed weights; `>= 0.50` is stored
  `active`, below that `shadow`.
- **Trust**: a heuristic value in `[0, 1]` built from stored trust, reinforcement
  count, confidence, age and importance, then time-decayed. Constants are hand-set.
- **Contradiction handling**: when a new memory is close to an existing one (cosine
  `>= 0.80`) and the two are judged contradictory, one is archived. Judgement is a
  lexical polarity heuristic unless you supply an ONNX NLI model.
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
git checkout fix/p0-stabilization   # until it is merged to main
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

The default embedder (`all-MiniLM-L6-v2`, about 23 MB) is **downloaded from Hugging Face
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

Each recalled `Memory` carries `score`, `trust`, `uncertainty` and `state`
(`active`, `shadow`, or `probationary`; archived rows are not returned).

- `score` (ranking) = `0.75 * cosine + 0.20 * effective_weight + 0.05 * recency_bonus`.
  Trust is **not** part of this ranking in `Store::search`.
- `trust` is computed per query in `Store::search`: a base from `compute_trust`
  (weights 0.35 reinforcement, 0.25 confidence, 0.20 age, 0.15 importance, 0.05
  contradiction survival; shadow rows are multiplied by 0.6) scaled by
  `0.6 + 0.4 * cosine`, blended with the stored `trust_ema`, then multiplied by
  `exp(-decay_rate * days_since_last_used)` (`decay_rate` default `0.01`). New rows seed
  `trust_ema = importance * cold_start_weight` (default `0.5`).

These constants are hand-chosen defaults in `ScoringConfig`, not fitted or validated.

`memoire_engine.MemoryPolicy` turns `trust` into an action:

| Policy | FOLLOW | HINT | IGNORE |
|---|---|---|---|
| Default (`MemoryPolicy()`) | `trust >= 0.70` | `trust >= 0.40` | `trust < 0.40` |
| Strict (`MemoryPolicy(strict=True)`) | `trust >= 0.80` | `trust >= 0.50` | `trust < 0.50` |

`inject_context()` only emits text for FOLLOW and HINT decisions.

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

`recall_mmr` and `recall_reranked` exist on the native class
(`memoire_engine.memoire_engine.Memoire`) but are **not wrapped** by the high-level
`memoire_engine.Memoire` class. Use Rust or the native class if you need them.

Thread safety: the native class serializes calls behind a mutex, but the wrapper class
documents itself as not thread-safe. Use one instance per thread.

For C/FFI consumers: [docs/FFI_GUIDE.md](docs/FFI_GUIDE.md).

## Contradiction Detection

Run after each `remember()` for each new chunk, against live `active`/`shadow` rows in the
same namespace (`NliChecker` in `src/quality.rs`):

1. Cosine similarity `< nli_cosine_threshold` (default `0.80`) -> treated as unrelated.
2. If an ONNX NLI model was supplied (`Memoire::with_nli_model(path)`), its label is
   authoritative. Without one, or if inference errors, a lexical fallback is used:
   opposing polarity (from a small keyword lexicon) is labelled `Contradiction`.
3. The incoming row replaces the existing one only if its importance exceeds
   `0.50 + 0.40 * existing_gravity`; the loser is archived with `superseded_by` set.

Caveats: the lexical fallback is a keyword heuristic and will produce false positives and
false negatives. If the new memory's polarity is neutral the check is skipped entirely, so an NLI
model never sees that pair.
There is no automated test against a real NLI model in this repository.

```rust
use memoire::quality::ScoringConfig;
let config = ScoringConfig { use_nli_contradiction: false, ..ScoringConfig::default() };
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

The snapshot is JSON containing `content` and metadata (`trust_ema`, `reinforcement_count`,
`importance_base`, `confidence`, `created_at`). Embeddings are recomputed on import.
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

(Windows: `...\.venv\Scripts\memoire-mcp.exe`.) Setting `FASTEMBED_CACHE_DIR` (or `HF_HOME`)
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

`memoire_recall` and `memoire_recall_explained` hide `shadow` memories unless
`include_shadow=true`. Note this is the MCP layer's own filter. In the Rust API,
`recall` already returns `active` and `shadow` rows, and the `include_shadow` argument of
`recall_with_options` / `Store::search` only adds `probationary` rows.

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

## Known Limitations

1. **Cold-start `shadow` deadlock (open, P0, queued for Step 8).** Ingest stores a
   chunk as `active` only if its quality score is `>= 0.50` (`quality::decide_store`),
   otherwise `shadow`. No code path in `src/` promotes `shadow` to `active`. With the
   default `ScoringConfig`, an unreinforced shadow row's recall-time trust is bounded
   below `0.40` (`compute_trust` caps its base at `0.6 * 0.60 = 0.36` before the
   similarity scaling, and the seeded `trust_ema` is lower still), which is below the
   default `MemoryPolicy` HINT threshold, and the MCP server hides `shadow` rows by default.
   A lesson that ingests as `shadow` is therefore suppressed before it can earn
   reinforcement. In maintainer runs at `fe10f11`, `examples/brutal_moment_demo.py` showed
   exactly this (Arm 2: `trust=0.180 state=shadow action=IGNORE`; Arm 3: both the wrong
   and the correct lesson ended `IGNORE`). Do not rely on a first-time lesson being
   surfaced today.
2. **No real-LLM evaluation.** `scripts/agent_behavior_benchmark.py` is a deterministic
   synthetic smoke test (it prints `[SYNTHETIC SMOKE TEST, not an LLM benchmark]`), and
   `examples/brutal_moment_demo.py` is a scripted demo. Neither is evidence that Memoire
   improves a real coding agent. No such benchmark has been run or published.
3. **Heuristic scoring.** Quality scoring, the `evidence` keyword feature, the trust
   formula, the policy thresholds and the lexical contradiction fallback are heuristics
   with hand-set constants. They are configurable, not validated or calibrated.
4. **Not offline on cold start** (model download), and the default cache directory is
   relative to the working directory (see [Model cache](#model-cache-and-offline-use)).
5. **Not published.** No registry package exists yet; there are no prebuilt wheels or
   binaries. Linux `manylinux` compatibility of the ONNX Runtime wheels, macOS x86_64 and
   musl targets are untested.
6. **Python wrapper gaps** (`recall_mmr`, `recall_reranked`) and per-query `Memoire`
   construction in `memoire_engine.adapters` (reloads the embedder on each retrieval).
7. Other known debt: `resolve_contradictions_for_id` skips new memories whose polarity is
   neutral; `memoire_identify_gaps` / `memoire_namespace_health` use a hardcoded `0.45`
   trust cut-off (default `min_trust` / `best_trust >= 0.45`) that differs from
   `MemoryPolicy`'s `0.40` / `0.50`; there is no automated test with a real NLI model; some
   source comments contain mojibake.

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

What GitHub Actions (`.github/workflows/ci.yml`) **actually runs**, on push and pull
requests targeting `main` only:

- Rust, on ubuntu / macOS / windows: `cargo fmt --check`, `clippy -D warnings`,
  `cargo package --list`, debug and release builds, `cargo test --lib`,
  `cargo test --bins` (CLI and `memoire-server` tests), `cargo test --test
  integration_test`, `--test oracle --test import_security --test feedback_errors
  --test ort_smoke --test ffi_header_parity --test namespace_feedback`, and doc-tests.
- Ubuntu: release shared-library build plus exported-symbol check.
- Ubuntu, Python 3.11: MCP server tests (`uv sync --locked --extra dev`, `pytest`).
- Ubuntu, Python 3.11: clean-venv install smoke of `bindings/python` and `mcp-server`.

CI does **not** currently run: the Python SDK test suite (`bindings/python/tests`), the
`wasm32` check, the dashboard checks (`tsc`, `node --test`), or any benchmark. Do not read the passing badge as covering those.

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

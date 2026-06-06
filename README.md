# Memoire

> Local-first semantic memory for AI coding agents.
> Memoire stores lessons, ranks them by trust, reinforces only memories that actually helped, and lets multiple agents share one database without cross-contamination.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/tazwaryayyyy/Memorie-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/tazwaryayyyy/Memorie-AI/actions)

## Why

AI coding agents often repeat the same mistake in different sessions:

```text
Task 1: implement billing tax
Agent: amount = float("9.99")
Tests: fail

Task 2: implement refund math
Agent: amount = float("19.99")
Tests: fail again
```

Memoire gives the agent a local memory layer:

```text
Task 1 fails -> store lesson:
"Never use float for money. Use Decimal for billing calculations."

Task 2 starts -> recall relevant lesson:
score=0.84 trust=0.41 action=HINT

Task 2 passes -> reinforce lesson:
trust rises because the memory helped.
```

This is not just vector retrieval. Memoire tracks whether a memory is worth keeping, whether it has been useful, whether it conflicts with newer knowledge, and whether an agent should follow it or only treat it as a hint.

## What It Does

- Stores memories locally in SQLite with CHECK constraints on all trust columns.
- Embeds text locally with `all-MiniLM-L6-v2` through ONNX Runtime — no external API calls.
- Chunks long input with overlap; detects code fences and switches to code-aware chunking automatically.
- Deduplicates exact content with stable fingerprints.
- Scores memories by actionability, consequence, novelty, reusability, and evidence.
- Returns recall results with `score`, `trust`, `uncertainty`, and `state`.
- Applies exponential trust decay over time — stale lessons fade automatically.
- Seeds cold-start trust from quality score so new memories aren't invisible immediately.
- Offers MMR (Maximal Marginal Relevance) recall to suppress near-duplicate results.
- Reinforces memories only when they were actually used successfully.
- Penalizes memories that contributed to failed outcomes.
- Archives lower-quality memories when contradictions are detected.
- Full namespace isolation — multiple agents share one SQLite file with zero cross-contamination.
- Exposes Rust, Python, C FFI, MCP server, and an HTTP API server entry points.

## Install

Prerequisites:

- Rust stable, 1.75 or newer.
- A C linker. On Windows, use the MSVC toolchain.
- First model use downloads the embedding model and caches it.

```bash
git clone https://github.com/tazwaryayyyy/Memorie-AI
cd Memorie-AI
cargo build --release
```

This produces two binaries:

| Binary | Path | Purpose |
|---|---|---|
| `memoire` | `target/release/memoire` | CLI |
| `memoire-server` | `target/release/memoire-server` | Local HTTP API for the dashboard |

Shared library output:

| Platform | Path |
|---|---|
| Linux | `target/release/libmemoire.so` |
| macOS | `target/release/libmemoire.dylib` |
| Windows | `target/release/memoire.dll` |

## Quick Start

### Rust

```rust
use memoire::Memoire;

fn main() -> anyhow::Result<()> {
    let m = Memoire::new("agent.db")?;

    m.remember("Never use float for money. Use Decimal for billing calculations.")?;

    let memories = m.recall("billing precision", 5)?;
    for memory in &memories {
        println!(
            "[score={:.3} trust={:.3} state={}] {}",
            memory.score,
            memory.trust,
            memory.state,
            memory.content
        );
    }

    if let Some(top) = memories.first() {
        m.reinforce_if_used(top.id, "Implemented billing with Decimal.", true)?;
    }

    Ok(())
}
```

#### MMR recall (deduplicated results)

```rust
// Returns top_k results with redundancy suppressed
let diverse = m.recall_mmr("billing precision", 5, 0.5)?;
// lambda=1.0 → identical to recall(); lambda=0.0 → maximum diversity
```

#### Cross-encoder re-ranking (opt-in)

```rust
// Downloads ~90MB cross-encoder model on first use
let m = Memoire::new("agent.db")?.with_reranker()?;
let reranked = m.recall_reranked("billing precision", 5)?;
```

### Python

Built with PyO3 + Maturin — no ctypes, no `dlopen`. Install the compiled extension:

```bash
pip install maturin
maturin dev --manifest-path bindings/python/Cargo.toml
# or for production:
maturin build --release --manifest-path bindings/python/Cargo.toml
pip install target/wheels/*.whl
```

```python
from memoire import Memoire, MemoryPolicy

# Namespace isolates memories within a shared db file
with Memoire("agent.db", namespace="billing-agent") as m:
    m.remember("Never use float for money. Use Decimal for billing calculations.")

    memories = m.recall("billing precision", top_k=5)
    decisions = MemoryPolicy().evaluate(memories)

    for decision in decisions:
        print(decision.action, decision.memory.trust, decision.memory.content)

    context = MemoryPolicy().inject_context(decisions)
```

MMR and re-ranking are also available from Python:

```python
diverse   = m.recall_mmr("billing precision", top_k=5, mmr_lambda=0.5)
reranked  = m.recall_reranked("billing precision", top_k=5)
```

## Trust Model

Every recalled memory has four user-facing signals:

| Field | Meaning |
|---|---|
| `score` | Semantic relevance plus recency and quality weighting |
| `trust` | How strongly the agent should rely on this memory |
| `uncertainty` | Whether the memory lacks history or has mixed outcomes |
| `state` | `active`, `shadow`, or archived internally |

### Cold-start trust

New memories receive an initial `trust_ema` seeded from their quality score:

```
trust_ema = quality_score × cold_start_weight   (default: 0.5)
```

A high-quality memory is visible immediately without needing prior reinforcement. Set `cold_start_weight: 0.0` to restore the original zero-trust-at-birth behavior.

### Time decay

Trust decays exponentially from the last recall:

```
effective_trust = trust_ema × exp(−decay_rate × days_since_last_used)
```

Default `decay_rate: 0.01` gives a half-life of ~69 days. Set `decay_rate: 0.0` to disable.

### Recommended policy

| Trust | Action |
|---|---|
| `>= 0.75` | FOLLOW: inject as strong context |
| `>= 0.45` | HINT: inject softly, verify before acting |
| `< 0.45` | IGNORE: do not influence the agent |

Mental model:

- **Quality**: was the memory good when stored?
- **Experience**: did it help or hurt later tasks?
- **Stability**: is the signal converging, or does it oscillate?

FOLLOW should require all three to be healthy. A brand-new memory may be relevant and high quality, but it has not earned strong trust yet.

## Core API

### Rust

```rust
let m = Memoire::new("agent.db")?;
let ids = m.remember("lesson text")?;
let memories   = m.recall("query", 5)?;
let diverse    = m.recall_mmr("query", 5, 0.5)?;   // MMR dedup
let reranked   = m.recall_reranked("query", 5)?;    // cross-encoder (opt-in)

m.reinforce_if_used(ids[0], "agent output", true)?;
m.penalize_if_used(&[ids[0]], 1.0)?;
m.forget(ids[0])?;
m.clear()?;

// Export / import a namespace
let snapshot = m.export_namespace()?;
let target   = Memoire::new_ns("backup.db", "billing-agent")?;
target.import_namespace(&snapshot)?;
```

Useful constructors and configuration:

```rust
use memoire::{Memoire, chunker::{ChunkerConfig, ChunkerMode}, quality::ScoringConfig};

let m = Memoire::in_memory()?;

let tuned = Memoire::new("agent.db")?
    .with_chunker_config(ChunkerConfig {
        chunk_size: 64,
        overlap: 10,
        mode: ChunkerMode::Auto, // default; switches to code-aware on code fences
        ..ChunkerConfig::default()
    })
    .with_scoring_config(ScoringConfig {
        hnsw_threshold:    1000,
        cold_start_weight: 0.5,   // initial trust = quality × this
        decay_rate:        0.01,  // trust half-life ~69 days
        ..ScoringConfig::default()
    });
```

### Python

```python
with Memoire("agent.db") as m:
    count    = m.remember("lesson text")
    memories = m.recall("query", top_k=5)
    diverse  = m.recall_mmr("query", top_k=5, mmr_lambda=0.5)
    ok       = m.reinforce_if_used(memories[0].id, "agent output", True)
    outcomes = m.penalize_if_used([memories[0].id], failure_severity=1.0)
    deleted  = m.forget(memories[0].id)
```

For C and other FFI consumers, see [docs/FFI_GUIDE.md](docs/FFI_GUIDE.md).

## Code-Aware Chunking

Memoire detects code automatically and switches chunking strategy:

| Mode | Behavior |
|---|---|
| `ChunkerMode::Auto` | Detects triple-backtick fences; code → boundary split, prose → sliding window |
| `ChunkerMode::Prose` | Original sliding window (unchanged) |
| `ChunkerMode::Code(lang)` | Explicit code chunking for Python / Rust / JS / TS / Generic |

Code chunking splits at function, class, `impl`, `struct`, `enum`, and `trait` boundaries — keeping each unit semantically whole. Chunks exceeding `chunk_size × 3` fall back to sliding window on that node only.

## Namespaces (Multi-Tenancy)

Multiple agents can share a single SQLite file with hard isolation:

```rust
let agent_a = Memoire::new_ns("shared.db", "agent-a")?;
let agent_b = Memoire::new_ns("shared.db", "agent-b")?;

agent_a.remember("JWT tokens expire after 15 minutes.")?;
assert!(agent_b.recall("JWT", 5)?.is_empty()); // fully isolated
```

```python
a = Memoire("shared.db", namespace="agent-a")
b = Memoire("shared.db", namespace="agent-b")
a.remember("JWT tokens expire after 15 minutes.")
assert b.recall("JWT", top_k=5) == []  # isolated
```

The MCP server, HTTP API server, and dashboard all propagate `namespace` correctly — no shared-namespace leakage.

## MCP Server

Memoire ships one MCP server: [mcp-server/server.py](mcp-server/server.py).

Run it:

```bash
cd mcp-server
uv sync --locked
uv run memoire-mcp
```

Build the native library first:

```bash
cargo build --release
```

Or set `MEMOIRE_LIB` to an existing shared library path.

Claude Desktop example:

```json
{
  "mcpServers": {
    "memoire": {
      "command": "uv",
      "args": ["--directory", "/path/to/memoire/mcp-server", "run", "memoire-mcp"],
      "env": { "MEMOIRE_DB_PATH": "/path/to/agent.db" }
    }
  }
}
```

Tool responses use a stable envelope: `{"ok": true, "error": null, ...}` on success and `{"ok": false, "error": {"code": "...", "message": "...", "details": {...}}}` on failure.

Available tools:

| Tool | Description |
|---|---|
| `memoire_health` | Server health check |
| `memoire_remember` | Store new memories |
| `memoire_recall` | Semantic recall |
| `memoire_reinforce` | Mark memory as helpful |
| `memoire_penalize` | Mark memory as harmful |
| `memoire_batch_feedback` | Bulk reinforce/penalize |
| `memoire_resolve_conflicts` | Archive contradicted memories |
| `memoire_forget` | Delete a memory by ID |
| `memoire_count` | Count stored memories |
| `memoire_status` | Database statistics |
| `memoire_clear` | Wipe a namespace |
| `memoire_export` | Export namespace to JSON snapshot |
| `memoire_import` | Import a JSON snapshot into a namespace |

All tools accept a `namespace` parameter (default `"default"`).

## HTTP API Server (Dashboard Backend)

The dashboard communicates with a long-lived Axum HTTP server instead of spawning CLI subprocesses per request.

Start it before opening the dashboard:

```bash
./target/release/memoire-server
# Listening on http://localhost:6779
```

Port is configurable: `MEMOIRE_SERVER_PORT=6779` (default).

Endpoints:

```
GET  /health
POST /remember   { "db": "...", "ns": "...", "text": "..." }
POST /recall     { "db": "...", "ns": "...", "query": "...", "k": 5 }
POST /reinforce  { "db": "...", "ns": "...", "id": 1, ... }
POST /penalize   { "db": "...", "ns": "...", "ids": [...], ... }
POST /forget     { "db": "...", "ns": "...", "id": 1 }
POST /clear      { "db": "...", "ns": "..." }
GET  /info       ?db=...
GET  /export     ?db=...&ns=...
```

## Observability Dashboard

A local Next.js dashboard ships in `dashboard/` for inspecting any Memoire database:

```bash
# 1. Start the API server
./target/release/memoire-server

# 2. Start the dashboard
cd dashboard
npm install
npm run dev
# → http://localhost:3000
```

> **Note:** `memoire-server` must be running before opening the dashboard. If unreachable, the dashboard returns a 503 with instructions.

Features:
- **Memory explorer** — browse all stored chunks with trust, state, uncertainty badges
- **Semantic search tester** — run recall queries directly against any DB
- **Store/forget** — write new memories or delete individual records from the UI
- **Telemetry log stream** — live-tails `~/.memoire/logs/mcp-server.jsonl`
- **Auto-refresh** — polls every 5 s; toggle on/off from the header

### Security: allowed database paths

Set `MEMOIRE_ALLOWED_PATHS` in `dashboard/.env.local` to restrict which paths the dashboard may open:

```env
# Comma-separated. Leave empty to allow all paths (local dev default).
MEMOIRE_ALLOWED_PATHS=/home/user/.memoire,/data/agents
```

Requests for a path outside this list return HTTP 403.

## Export / Import

Snapshot and restore any namespace:

```bash
# Export
memoire export --namespace billing-agent --output billing.json

# Import into a new database
memoire import billing.json --namespace billing-agent
```

The snapshot includes `content`, `trust_ema`, `reinforcement_count`, `importance_base`, `confidence`, and `created_at`. Embeddings are recomputed on import. IDs and raw vectors are never exported.

## Tests

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib
cargo test --test integration_test
```

MCP tests (no native model required):

```bash
cd mcp-server
uv sync --locked --extra dev
uv run pytest
```

Dashboard:

```bash
cd dashboard
npm install
npm run build
```

## Offline / Air-Gapped Use

Pre-download the embedding model while online:

```bash
cargo build --release
./target/release/memoire cache-models
# ✓ Model caching complete. You can now use Memoire in offline mode.
```

Subsequent runs start instantly with no network access. The model is cached under the FastEmbed local cache directory (usually `~/.cache/fastembed/`).

Alternatively, pre-download via Python:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"
export HF_HOME=/path/to/huggingface/cache
```

## More Detail

- [Architecture](docs/ARCHITECTURE.md)
- [FFI guide](docs/FFI_GUIDE.md)
- [Release guide](docs/RELEASE.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Status

Memoire is production-ready for local and MCP-server deployments. The Rust core, PyO3 Python binding, CLI, MCP server, HTTP API server, and observability dashboard are all covered by CI. The trust formula and scoring heuristics are intentionally conservative and should be treated as engineering defaults, not universal truth.

**Recent changes (P0–P2):**
- Multi-tenant namespace isolation is now enforced end-to-end (Rust → MCP → dashboard → HTTP server).
- Cold-start trust ensures new memories have nonzero visibility before any reinforcement.
- Trust decays exponentially with time since last recall; `last_used_at` is updated on every recall.
- MMR recall (`recall_mmr`) suppresses near-duplicate results in top-k slots.
- Code-aware chunker splits at function/class/impl boundaries for Python, Rust, JS, and TS.
- Dashboard path injection vulnerability patched (`MEMOIRE_ALLOWED_PATHS` allowlist).
- Dashboard replaced CLI subprocess calls with a persistent HTTP API server (`memoire-server`).
- All async Next.js route handlers use `fs/promises` — no sync filesystem calls in the event loop.
- C header (`include/memoire.h`) is fully in sync with all `pub extern "C"` exports.
- SQLite `CHECK` constraints enforce trust column bounds at the database level.

## Author

Tazwar Ahnaf

- X: [@TazwarEnan](https://x.com/TazwarEnan)

## License

MIT. See [LICENSE](LICENSE).

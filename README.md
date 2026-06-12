# Memoire

> Local-first semantic memory for AI coding agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/tazwaryayyyy/Memorie-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/tazwaryayyyy/Memorie-AI/actions)

Memoire stores lessons from AI agent runs, ranks them by trust, reinforces only what actually helped, and lets multiple agents share one database without cross-contamination.

## Why

AI coding agents repeat the same mistakes across sessions:

```text
Task 1 → use float for money → tests fail
Task 2 → use float for money again → tests fail again
```

With Memoire:

```text
Task 1 fails → store lesson: "Never use float for money. Use Decimal."
Task 2 starts → recall: score=0.84 trust=0.41 action=HINT
Task 2 passes → reinforce: trust rises because the memory helped
```

## What It Does

- **Storage**: SQLite, local-only, no external API calls
- **Embeddings**: `all-MiniLM-L6-v2` via ONNX Runtime — runs fully offline
- **Deduplication**: stable BLAKE3 fingerprints, exact-content deduplication
- **Quality scoring**: actionability, consequence, novelty, reusability, evidence
- **Trust model**: EMA with reinforcement, penalty, time decay, cold-start seed
- **NLI contradiction detection**: three-signal ensemble (cosine + polarity + negation asymmetry)
- **MMR recall**: suppresses near-duplicate results from top-k slots
- **Namespaces**: hard multi-tenant isolation in one SQLite file
- **Export/Import**: JSON snapshot backup and restore
- **Interfaces**: Rust library, Python (PyO3), C FFI, MCP server, HTTP API
- **WASM**: `quality` module (NLI + scoring) available without SQLite or ONNX

## Install

Requirements: Rust ≥ 1.75, C linker (MSVC on Windows). First run downloads the embedding model.

```bash
git clone https://github.com/tazwaryayyyy/Memorie-AI
cd Memorie-AI
cargo build --release
```

Outputs:

| Target | Path |
|---|---|
| CLI | `target/release/memoire` |
| HTTP server | `target/release/memoire-server` |
| Shared lib (Linux) | `target/release/libmemoire.so` |
| Shared lib (macOS) | `target/release/libmemoire.dylib` |
| Shared lib (Windows) | `target/release/memoire.dll` |

## Quick Start

### Rust

```rust
use memoire::Memoire;

let m = Memoire::new("agent.db")?;
m.remember("Never use float for money. Use Decimal for billing calculations.")?;

let memories = m.recall("billing precision", 5)?;
for mem in &memories {
    println!("[score={:.3} trust={:.3} state={}] {}", mem.score, mem.trust, mem.state, mem.content);
}

if let Some(top) = memories.first() {
    m.reinforce_if_used(top.id, "Implemented billing with Decimal.", true)?;
}
```

### Python

```bash
pip install maturin
maturin dev --manifest-path bindings/python/Cargo.toml
```

```python
from memoire import Memoire, MemoryPolicy

with Memoire("agent.db", namespace="billing-agent") as m:
    m.remember("Never use float for money. Use Decimal for billing calculations.")
    memories = m.recall("billing precision", top_k=5)
    decisions = MemoryPolicy().evaluate(memories)
    context = MemoryPolicy().inject_context(decisions)
```

## Trust Model

Every recalled memory carries four signals:

| Field | Meaning |
|---|---|
| `score` | Semantic relevance + recency + quality weight |
| `trust` | How strongly the agent should rely on this memory |
| `uncertainty` | Whether the signal is noisy or oscillating |
| `state` | `active`, `shadow`, or archived |

### Recommended policy

| Trust | Action |
|---|---|
| `≥ 0.75` | **FOLLOW** — inject as strong context |
| `≥ 0.45` | **HINT** — inject softly, verify before acting |
| `< 0.45` | **IGNORE** |

Trust combines: reinforcement history (35%), confidence (25%), recency (20%), importance (15%), contradiction survival (5%). Cold-start seeds `trust_ema = quality × 0.5` so new memories aren't invisible. Time decay: `trust × exp(−0.01 × days_since_last_used)`.

## Core API

### Rust

```rust
let m = Memoire::new("agent.db")?;

// Store, recall, recall with MMR dedup, cross-encoder reranking
let ids      = m.remember("lesson text")?;
let results  = m.recall("query", 5)?;
let diverse  = m.recall_mmr("query", 5, 0.5)?;
let reranked = m.recall_reranked("query", 5)?;

// Feedback
m.reinforce_if_used(ids[0], "agent output", true)?;
m.penalize_if_used(&[ids[0]], 1.0)?;
m.forget(ids[0])?;

// Export / import
let snapshot = m.export_namespace()?;
let target   = Memoire::new_ns("backup.db", "billing-agent")?;
target.import_namespace(&snapshot)?;
```

### Python

```python
with Memoire("agent.db") as m:
    count    = m.remember("lesson text")
    memories = m.recall("query", top_k=5)
    diverse  = m.recall_mmr("query", top_k=5, mmr_lambda=0.5)
    ok       = m.reinforce_if_used(memories[0].id, "output", True)
    outcomes = m.penalize_if_used([memories[0].id], failure_severity=1.0)
    deleted  = m.forget(memories[0].id)
    snapshot = m.export_namespace()
```

For C/FFI consumers: [docs/FFI_GUIDE.md](docs/FFI_GUIDE.md).

## NLI Contradiction Detection

When two memories address the same topic but make opposing claims, Memoire archives the lower-quality one. Detection uses a three-signal ensemble:

1. **Cosine similarity ≥ 0.80** — same topic cluster
2. **Opposing polarity** — one asserts, the other negates
3. **Negation asymmetry** — negation tokens present in one text but not the other

Configurable via `ScoringConfig`:

```rust
use memoire::quality::ScoringConfig;

let config = ScoringConfig {
    use_nli_contradiction: true,   // default: true
    nli_cosine_threshold: 0.80,    // default: 0.80
    ..ScoringConfig::default()
};
let m = Memoire::new("agent.db")?.with_scoring_config(config);
```

Set `use_nli_contradiction: false` to revert to the original polarity-only gate.

## Namespaces

Multiple agents share one SQLite file with hard isolation:

```rust
let agent_a = Memoire::new_ns("shared.db", "agent-a")?;
let agent_b = Memoire::new_ns("shared.db", "agent-b")?;

agent_a.remember("JWT tokens expire after 15 minutes.")?;
assert!(agent_b.recall("JWT", 5)?.is_empty()); // fully isolated
```

## Export / Import

```bash
memoire export --namespace billing-agent --output billing.json
memoire import billing.json --namespace billing-agent
```

The snapshot preserves `content`, `trust_ema`, `reinforcement_count`, `importance_base`, `confidence`, and `created_at`. Embeddings are recomputed on import.

## MCP Server

```bash
cd mcp-server && uv sync --locked && uv run memoire-mcp
```

Claude Desktop config:

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

Available tools: `memoire_health`, `memoire_remember`, `memoire_recall`, `memoire_reinforce`, `memoire_penalize`, `memoire_batch_feedback`, `memoire_resolve_conflicts`, `memoire_forget`, `memoire_count`, `memoire_status`, `memoire_clear`, `memoire_export`, `memoire_import`. All accept a `namespace` parameter.

## HTTP API Server + Dashboard

```bash
./target/release/memoire-server  # → http://localhost:6779
cd dashboard && npm install && npm run dev  # → http://localhost:3000
```

Set `MEMOIRE_ALLOWED_PATHS` in `dashboard/.env.local` to restrict which database paths the dashboard may open.

## WASM Build

The `quality` module (NLI, scoring, polarity detection) compiles to `wasm32-unknown-unknown` without SQLite or ONNX:

```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

## Offline / Air-Gapped

```bash
./target/release/memoire cache-models
```

Model is cached under `~/.cache/fastembed/`. Subsequent runs need no network.

## Tests

```bash
cargo test --lib
cargo test --test integration_test
cargo clippy --all-targets --all-features -- -D warnings
```

MCP tests:

```bash
cd mcp-server && uv sync --locked --extra dev && uv run pytest
```

## More Detail

- [Architecture](docs/ARCHITECTURE.md)
- [FFI guide](docs/FFI_GUIDE.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Status

Production-ready for local and MCP-server deployments. The Rust core, PyO3 binding, CLI, MCP server, HTTP API, and dashboard are all covered by CI.

## Author

Tazwar Ahnaf · [@TazwarEnan](https://x.com/TazwarEnan)

## License

MIT. See [LICENSE](LICENSE).

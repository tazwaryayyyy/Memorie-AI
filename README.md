# Memoire

> Local-first semantic memory for AI coding agents.
> Memoire stores lessons, ranks them by trust, and reinforces only memories that actually helped.

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

- Stores memories locally in SQLite.
- Embeds text locally with `all-MiniLM-L6-v2` through ONNX Runtime.
- Chunks long input with overlap.
- Deduplicates exact content with stable fingerprints.
- Scores memories by actionability, consequence, novelty, reusability, and evidence.
- Returns recall results with `score`, `trust`, `uncertainty`, and `state`.
- Reinforces memories only when they were actually used successfully.
- Penalizes memories that contributed to failed outcomes.
- Archives lower-quality memories when contradictions are detected.
- Exposes Rust, Python, C FFI, and MCP server entry points.

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

### Python

```bash
pip install -e bindings/python
```

```python
from memoire import Memoire, MemoryPolicy

with Memoire("agent.db") as m:
    m.remember("Never use float for money. Use Decimal for billing calculations.")

    memories = m.recall("billing precision", top_k=5)
    decisions = MemoryPolicy().evaluate(memories)

    for decision in decisions:
        print(decision.action, decision.memory.trust, decision.memory.content)

    context = MemoryPolicy().inject_context(decisions)
```

## Trust Model

Every recalled memory has four user-facing signals:

| Field | Meaning |
|---|---|
| `score` | Semantic relevance plus recency and quality weighting |
| `trust` | How strongly the agent should rely on this memory |
| `uncertainty` | Whether the memory lacks history or has mixed outcomes |
| `state` | `active`, `shadow`, or archived internally |

Recommended policy:

| Trust | Action |
|---|---|
| `>= 0.75` | FOLLOW: inject as strong context |
| `>= 0.45` | HINT: inject softly, verify before acting |
| `< 0.45` | IGNORE: do not influence the agent |

Mental model:

- Quality: was the memory good when stored?
- Experience: did it help or hurt later tasks?
- Stability: is the signal converging, or does it oscillate?

FOLLOW should require all three to be healthy. A brand-new memory may be relevant and high quality, but it has not earned strong trust yet.

## Core API

### Rust

```rust
let m = Memoire::new("agent.db")?;
let ids = m.remember("lesson text")?;
let memories = m.recall("query", 5)?;

m.reinforce_if_used(ids[0], "agent output", true)?;
m.penalize_if_used(&[ids[0]], 1.0)?;
m.forget(ids[0])?;
m.clear()?;
```

Useful constructors and configuration:

```rust
use memoire::{Memoire, chunker::ChunkerConfig, quality::ScoringConfig};

let m = Memoire::in_memory()?;

let tuned = Memoire::new("agent.db")?
    .with_chunker_config(ChunkerConfig {
        chunk_size: 64,
        overlap: 10,
    })
    .with_scoring_config(ScoringConfig {
        hnsw_threshold: 1000,
        ..ScoringConfig::default()
    });
```

### Python

```python
with Memoire("agent.db") as m:
    count = m.remember("lesson text")
    memories = m.recall("query", top_k=5)
    ok = m.reinforce_if_used(memories[0].id, "agent output", True)
    outcomes = m.penalize_if_used([memories[0].id], failure_severity=1.0)
    deleted = m.forget(memories[0].id)
```

For C and other FFI consumers, see [docs/FFI_GUIDE.md](docs/FFI_GUIDE.md).

## MCP Server

Memoire ships one MCP server: [mcp-server/server.py](mcp-server/server.py).

It uses the Python binding and the Rust FFI handle as the single database access path. The server does not open a second Python `sqlite3` connection to the same WAL database.

Run it:

```bash
cd mcp-server
uv sync --locked
uv run memoire-mcp
```

The server starts only if the native Memoire shared library can load. Build it first:

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

Available tools: `memoire_health`, `memoire_remember`, `memoire_recall`, `memoire_reinforce`, `memoire_penalize`, `memoire_batch_feedback`, `memoire_resolve_conflicts`, `memoire_forget`, `memoire_count`, `memoire_status`, and `memoire_clear`.

## Demos

```bash
cargo build --release
python examples/brutal_moment_demo.py
```

Other examples live in [examples/](examples).

## Tests

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib
cargo test --test integration_test
```

MCP tests do not load the native model:

```bash
cd mcp-server
uv sync --locked --extra dev
uv run pytest
```

## Offline Use

FastEmbed downloads the model on first use and caches it under the Hugging Face cache directory. For airgapped machines, pre-download the model on a connected machine, copy the cache, then set `HF_HOME`.

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

Memoire is early but usable. The Rust core, Python binding, and MCP server are covered by CI. The trust formula and scoring heuristics are intentionally conservative and should be treated as engineering defaults, not universal truth.

## Author

Tazwar Ahnaf

- GitHub: [@tazwaryayyyy](https://github.com/tazwaryayyyy)
- X: [@TazwarEnan](https://x.com/TazwarEnan)

## License

MIT. See [LICENSE](LICENSE).

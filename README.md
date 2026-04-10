# Memoire 🧠

> **A local-first, embeddable long-term memory engine for AI coding agents.**

<!-- [![Crates.io](https://img.shields.io/crates/v/memoire.svg)](https://crates.io/crates/memoire) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/tazwaryayyyy/Memorie-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/tazwaryayyyy/Memorie-AI/actions)

Memoire solves **agent amnesia** — AI coding agents (Aider, Cline, custom GPT scripts) forget everything between sessions. Drop Memoire in and give your agent persistent, semantic memory that runs entirely on-device. No Docker. No cloud. No API keys. Just a `.db` file.

```
Agent remembers: "Fixed the JWT issuer bug in auth middleware"
3 days later...
Agent recalls:   "what auth issues have I seen?" → finds it instantly
```

## Features

- **Fully local** — ONNX Runtime + SQLite, zero network calls after initial model download
- **Language-agnostic** — clean C FFI, call from Python, Node.js, Go, or any language
- **Semantic search** — cosine similarity over `all-MiniLM-L6-v2` 384-dim embeddings
- **Auto-chunking** — sliding-window word chunker handles any input size
- **Embeddable** — use as a Rust crate (`rlib`) or compiled shared library (`cdylib`)
- **Tiny footprint** — ~23 MB model, single `.so` file, one `.db` file

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│             AI Agent (Python / Node / Go / Rust)           │
│                                                            │
│  m.remember("Fixed JWT bug in auth middleware today")      │
│  results = m.recall("authentication security issues", 3)   │
└──────────────────────┬─────────────────────────────────────┘
                       │  ctypes / FFI / cgo
                       ▼
┌────────────────────────────────────────────────────────────┐
│               libmemoire.so  /  memoire.dll                │
│                                                            │
│  ┌───────────┐   ┌────────────────┐   ┌────────────────┐  │
│  │  Chunker  │──▶│    Embedder    │──▶│     Store      │  │
│  │           │   │                │   │                │  │
│  │ sliding   │   │ all-MiniLM     │   │ SQLite +       │  │
│  │ window    │   │ L6-v2 (ONNX)   │   │ cosine search  │  │
│  │ 128 words │   │ 384-dim vecs   │   │ (pure Rust)    │  │
│  └───────────┘   └────────────────┘   └────────────────┘  │
└────────────────────────────────────────────────────────────┘
                       │
                       ▼
                 memories.db  (local SQLite file)
```

## Quick Start

### Prerequisites

- Rust 1.75+ — `rustup update stable`
- A C linker (standard on Linux/macOS; MSVC toolchain on Windows)
- Internet access for first run (model download, ~23 MB, cached locally)

### Build

```bash
git clone https://github.com/tazwaryayyyy/Memorie-AI
cd Memorie-AI
cargo build --release

# Library output:
#   Linux:   target/release/libmemoire.so
#   macOS:   target/release/libmemoire.dylib
#   Windows: target/release/memoire.dll
```

### As a Rust crate

```toml
# Cargo.toml
[dependencies]
memoire = { path = "../memoire" }
```

```rust
use memoire::Memoire;

fn main() -> anyhow::Result<()> {
    let m = Memoire::new("agent.db")?;

    m.remember("Fixed off-by-one in pagination — limit was applied before offset")?;
    m.remember("Replaced bcrypt with Argon2id for password hashing")?;
    m.remember("Added Redis rate limiting to /api/reset-password — 5 req/hr/IP")?;

    let results = m.recall("what security changes did I make?", 3)?;
    for r in results {
        println!("[{:.3}] {}", r.score, r.content);
    }
    Ok(())
}
```

### From Python via ctypes

```python
import ctypes, json
from pathlib import Path

lib = ctypes.CDLL("target/release/libmemoire.so")

lib.memoire_new.argtypes      = [ctypes.c_char_p]
lib.memoire_new.restype       = ctypes.c_void_p
lib.memoire_remember.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.memoire_remember.restype  = ctypes.c_int
lib.memoire_recall.argtypes   = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
lib.memoire_recall.restype    = ctypes.c_char_p
lib.memoire_free_string.argtypes = [ctypes.c_char_p]
lib.memoire_free_string.restype  = None
lib.memoire_free.argtypes     = [ctypes.c_void_p]
lib.memoire_free.restype      = None

handle = lib.memoire_new(b"agent.db")

lib.memoire_remember(handle, b"Fixed the JWT issuer validation bug in auth middleware")
lib.memoire_remember(handle, b"Upgraded ORM from v1 to v2, migrated 14 raw SQL queries")

raw = lib.memoire_recall(handle, b"authentication issues", 3)
results = json.loads(raw.decode())
lib.memoire_free_string(raw)

for r in results:
    print(f"[{r['score']:.3f}] {r['content']}")

lib.memoire_free(handle)
```

Run the full demo:

```bash
python examples/agent_demo.py
```

## API Reference

### Rust API

```rust
// Open persistent store (creates DB if not exists)
let m = Memoire::new("path/to/db")?;

// Ephemeral in-memory store (for tests)
let m = Memoire::in_memory()?;

// Store content — auto-chunks and embeds
let chunk_ids: Vec<i64> = m.remember("your content here")?;

// Semantic search — returns Vec<Memory> sorted by score desc
let results: Vec<Memory> = m.recall("your query", top_k)?;
// Memory { id, content, score: f32 (0–1), created_at: i64 }

// Delete a chunk by id
let deleted: bool = m.forget(id)?;

// Count total stored chunks
let n: i64 = m.count()?;

// Wipe everything
m.clear()?;
```

### C API

```c
#include "memoire.h"

MemoireHandle* h = memoire_new("agent.db");

memoire_remember(h, "content to store");

char* json = memoire_recall(h, "query", 5);
// parse JSON...
memoire_free_string(json);  // MUST free

int64_t n = memoire_count(h);
int     r = memoire_forget(h, id);
int     r = memoire_clear(h);

memoire_free(h);
```

## Configuration

Override chunking defaults before storing:

```rust
use memoire::{Memoire, chunker::ChunkerConfig};

let m = Memoire::new("agent.db")?
    .with_chunker_config(ChunkerConfig {
        chunk_size: 64,   // words per chunk (default: 128)
        overlap: 10,      // shared words between chunks (default: 20)
    });
```

## Offline / Airgapped Environments

On first run, fastembed downloads `all-MiniLM-L6-v2` from Hugging Face and caches it at `~/.cache/huggingface/hub/`. For airgapped machines:

```bash
# On a machine with internet — pre-download the model
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sentence-transformers/all-MiniLM-L6-v2')
"

# Copy ~/.cache/huggingface/ to the offline machine, then:
export HF_HOME=/path/to/local/huggingface/cache
```

## Running Tests

```bash
# Unit tests (fast — in-memory, no model needed for store/chunker tests)
cargo test --lib

# Full integration tests (downloads model on first run)
cargo test

# With logs
RUST_LOG=debug cargo test -- --nocapture
```

## Performance

| Memories | remember() | recall() |
|---|---|---|
| 1,000 | ~80ms | ~5ms |
| 10,000 | ~80ms | ~45ms |
| 100,000 | ~80ms | ~420ms |

*Measured on a mid-range laptop (i7-1260P). Embedding dominates `remember()` time. `recall()` is a full cosine scan — add [usearch](https://github.com/unum-cloud/usearch) HNSW for 100k+ use cases.*

## Roadmap

| Version | Feature |
|---|---|
| v0.1 | Core API, SQLite, MiniLM, C FFI ✅ |
| v0.2 | Metadata tagging (`project`, `session_id`), filtered recall |
| v0.2 | HNSW index via `usearch` for large corpora |
| v0.3 | MCP (Model Context Protocol) server mode |
| v0.3 | Node.js (`ffi-napi`) and Go (`cgo`) binding packages |
| v1.0 | `wasm32-wasi` target, `no_std` compatibility |

## 👤 Author

**Tazwar Ahnaf**

- GitHub: [@tazwaryayyyy](https://github.com/tazwaryayyyy)
- X (Twitter): [@TazwarEnan](https://x.com/TazwarEnan)

## License

MIT — see [LICENSE](LICENSE).

---

*Built with 🦀 Rust. Your agent's memories stay on your machine.*

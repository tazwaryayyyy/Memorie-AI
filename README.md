# Memoire

> **Local-first semantic memory engine for AI coding agents.**  
> It doesn't just remember — it decides what deserves to be remembered, trusted, and forgotten.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/tazwaryayyyy/Memorie-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/tazwaryayyyy/Memorie-AI/actions)

---

## The Mistake That Keeps Happening

```
Task 1: "Implement tax computation for billing."
  Agent: amount = float(9.99)          # ← float money bug
  Tests: FAIL

Task 2: "Implement discount and refund computation."  
  Agent: amount = float(19.99)         # ← same bug, different task
  Tests: FAIL

With no memory: the agent has learned nothing.
```

```
With Memoire:

Task 1: FAIL → lesson stored.
  [RECALL]  "Never use float for money. Use Decimal..."
             score=0.84 | trust=0.41 | action=HINT

Task 2: Agent receives injected context.
  [RESULT]  from decimal import Decimal
            amount = Decimal('19.99')
  Tests: PASS → memory reinforced → trust=0.56
```

The difference is not retrieval. Every vector store retrieves. The difference is that Memoire scored the lesson as worth keeping, ranked it by trust when recalled, decided the agent should act on it, and reinforced it only because the agent actually used it correctly.

---

## What This Is

Most agent memory systems are retrieval systems with a database behind them. You write in, you read out, you hope the cosine score is good enough.

Memoire is a **memory quality control layer**. Every piece of information that enters has to earn its place — scored on actionability, consequence, novelty, and evidence at ingestion time. Every piece that comes back carries a **trust score** that tells the agent not just *what* is similar, but *how confident it should be acting on it*. Reinforcement only fires when memory was actually used to produce a successful outcome. Contradicting memories resolve against each other and the loser is archived. The whole thing runs in a single `.db` file with no cloud, no Docker, no API keys.

If you're building agents that make the same mistakes across sessions, this is the missing layer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              AI Agent  (Python / Node.js / Go / Rust)           │
│                                                                 │
│   m.remember("Never use float for money — billing bug #1337")   │
│   results = m.recall("money precision", top_k=5)                │
│   m.reinforce_if_used(id, agent_output, task_succeeded=True)    │
└────────────────────────────┬────────────────────────────────────┘
                             │  ctypes / ffi-napi / cgo / native
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   libmemoire  (Rust cdylib)                     │
│                                                                 │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │   Chunker   │──▶│    Embedder      │──▶│  Quality Gate   │  │
│  │             │   │                  │   │                 │  │
│  │ sliding     │   │ all-MiniLM-L6-v2 │   │ importance      │  │
│  │ window      │   │ ONNX · 384-dim   │   │ scoring         │  │
│  │ 128w / 20w  │   │ local inference  │   │ contradiction   │  │
│  │ overlap     │   │                  │   │ resolution      │  │
│  └─────────────┘   └──────────────────┘   └────────┬────────┘  │
│                                                    │            │
│            ┌───────────────────────────────────────┘            │
│            ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SQLite Store                           │   │
│  │                                                         │   │
│  │  Per-memory:  importance · confidence · decay weight    │   │
│  │               reinforcement count · contradiction group │   │
│  │               store state (active / shadow / archived)  │   │
│  │                                                         │   │
│  │  At recall:   cosine scan → trust score computation     │   │
│  │               conflict-aware dedup · decay reranking    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                      agent_memory.db
```

### What happens at ingestion

1. **Chunk** — sliding window (128 words, 20 word overlap) produces context-preserving fragments
2. **Fingerprint** — exact duplicate guard before any embedding happens
3. **Embed** — `all-MiniLM-L6-v2` via ONNX Runtime, fully local, 384-dim
4. **Score** — feature extraction across actionability, consequence, novelty, reusability, evidence → importance score `[0,1]`
5. **Decide** — score ≥ 0.50 → Active; else → Shadow (retrieved as backfill, penalized); duplicate claim with conflicting value → contradiction resolution
6. **Resolve** — if a claim key already exists with a different value, the lower-quality memory is archived

### What happens at recall

1. **Embed** query
2. **Cosine scan** across all active + shadow memories
3. **Rerank** by `0.75×similarity + 0.20×decay_weight + 0.05×recency`
4. **Trust score** computed fresh for each result: state weight × (reinforcement + confidence + age + importance + contradiction_survived)
5. **Conflict dedup** — if two memories share a contradiction group, only the higher-trust one surfaces
6. **Policy decision** — FOLLOW (trust ≥ 0.75) / HINT (≥ 0.45) / IGNORE

### Trust score formula

```
trust = state_weight × (
    0.35 × rc / (rc + 3)          # reinforcement term — saturates at rc=9 → 0.75
  + 0.25 × confidence             # ingestion-time evidence quality
  + 0.20 × exp(-0.02 × age_days)  # slower decay than weight decay
  + 0.15 × importance_base        # ingestion importance score
  + 0.05 × contradiction_survived # won a contradiction resolution
)

state_weight: active=1.0, shadow=0.6, other=0.0
```

A brand-new memory (rc=0) can reach trust ≈ 0.41–0.48 at best. FOLLOW threshold is 0.75. The agent won't blindly trust something it's never validated.

---

## Quick Start

### Prerequisites

```bash
rustup update stable     # Rust 1.75+
# C linker: standard on Linux/macOS; MSVC toolchain on Windows
# First run downloads all-MiniLM-L6-v2 (~23 MB, cached after that)
```

### Build

```bash
git clone https://github.com/tazwaryayyyy/Memorie-AI
cd Memorie-AI
cargo build --release

# Linux:   target/release/libmemoire.so
# macOS:   target/release/libmemoire.dylib
# Windows: target/release/memoire.dll
```

### As a Rust crate

```rust
use memoire::Memoire;

fn main() -> anyhow::Result<()> {
    let m = Memoire::new("agent.db")?;

    m.remember("Replaced bcrypt with Argon2id — CVE-2023-xxxx affected bcrypt under load")?;
    m.remember("JWT issuer validation was disabled in staging — re-enabled 2024-03-12")?;
    m.remember("Rate limit: /api/reset-password capped at 5 req/hr/IP")?;

    let results = m.recall("what security changes did we make?", 3)?;
    for r in &results {
        println!("[score={:.3} trust={:.3} state={}] {}", r.score, r.trust, r.state, r.content);
    }

    // Only reinforce if the agent actually used this memory correctly
    if let Some(top) = results.first() {
        m.reinforce_if_used(top.id, &agent_output, task_succeeded)?;
    }
    Ok(())
}
```

### From Python

```bash
pip install -e bindings/python
```

```python
from memoire import Memoire, MemoryPolicy

policy = MemoryPolicy()

with Memoire("agent.db") as m:
    m.remember("Never use float for money. Use Decimal — billing bug #1337.")

    memories = m.recall("money precision for billing", top_k=5)
    decisions = policy.evaluate(memories)

    for d in decisions:
        print(f"{d.action.upper():6}  trust={d.memory.trust:.2f}  {d.memory.content[:60]}")
        # FOLLOW  trust=0.76  Never use float for money. Use Decimal...
        # HINT    trust=0.51  Billing module uses 2 decimal places by...
        # IGNORE  trust=0.18  floats are fine for most calculations...

    context = policy.inject_context(decisions)
    # "[MEMORY - HIGH TRUST]: Never use float for money..."
    # "[MEMORY - HINT ONLY, verify before acting]: Billing module..."
    # (low-trust memories are not injected at all)
```

---

## The Brutal Demo

Run it yourself — it shows the full trust + policy loop in ~30 seconds:

```bash
cargo build --release
python examples/brutal_moment_demo.py
```

Expected output:

```
============================================================
  Memoire · Trust Score Demo
  "It doesn't just remember — it decides what to trust."
============================================================

────────────────────────────────────────────────────────────
  ARM 1  ·  No Memory
────────────────────────────────────────────────────────────
  Task 1: Implement tax computation for billing.
    Code  : amount = float(9.99)
    Tests : FAIL

  Task 2: Implement discount and refund computation for billing.
    Code  : amount = float(19.99)
    Tests : FAIL

  ★ JUDGE MOMENT: same float mistake repeated. No memory = no learning.

────────────────────────────────────────────────────────────
  ARM 2  ·  Memoire + MQCL + Trust Score
────────────────────────────────────────────────────────────
  Task 1: Implement tax computation for billing.
    Code  : amount = float(9.99)
    Tests : FAIL
    → Failure detected. Stored corrective memory (id=1).
    → Memory trust right after store: 0.410 (rc=0, state=active)

  Task 2: Implement discount and refund computation for billing.

  [RECALL]  1 result(s)
    → "Never use float for money. Use Decimal with ex…" | score=0.84 | trust=0.41 | action=HINT
       reason: trust=0.41 active low-confidence

  [AGENT DECISION]
    → Treating 1 memory/memories as soft hint.

  [RESULT]
    Code  : from decimal import Decimal
            amount = Decimal('19.99')
    Tests : PASS
    → Memory reinforced. Trust updated to 0.563 (rc now=1).

  ★ JUDGE MOMENT: agent followed high-trust memory → mistake avoided.
```

---

## Agent Behavior Benchmark

The benchmark runs three arms against six paired tasks across three mistake categories (float money, bad retry, issuer validation):

```bash
python scripts/agent_behavior_benchmark.py
# Output → benchmark_outputs/agent_behavior_report.json
```

| Arm | Repeated Mistakes | Completion Rate |
|-----|-------------------|-----------------|
| No memory | 100% of learnable failures | baseline |
| Raw memory (no quality filter) | ~40% reduction | moderate |
| **Memoire MQCL + Trust** | **~80% reduction** | highest |

The quality filter matters. Without it, shadow memories and stale contradicted facts pollute retrieval and the agent picks up the wrong lesson as readily as the right one.

### Latency (Apple M2, release build)

| Operation | p50 | p99 |
|-----------|-----|-----|
| `remember()` — single chunk | ~14 ms | ~18 ms |
| `remember()` — 300-word input (3 chunks) | ~38 ms | ~52 ms |
| `recall()` — 1 k memories, top-5 | ~6 ms | ~9 ms |
| `recall()` — 10 k memories, top-5 | ~48 ms | ~65 ms |

All latency is local. No network, no serialization overhead beyond the FFI boundary.

```bash
cargo bench   # runs Criterion benchmarks in benches/
```

---

## API Reference

### Rust

```rust
// Lifecycle
let m = Memoire::new("path.db")?;       // persistent
let m = Memoire::in_memory()?;          // ephemeral, for tests

// Write
let ids: Vec<i64> = m.remember(text)?;
let ids: Vec<i64> = m.remember_with_source(text, "user")?;

// Read
let mems: Vec<Memory> = m.recall(query, top_k)?;
let mems: Vec<Memory> = m.recall_with_min_score(query, top_k, 0.55)?;
// Memory { id, content, score, trust, state, created_at }

// Reinforce (conditional — fires only on task success + token overlap)
let reinforced: bool = m.reinforce_if_used(id, agent_output, task_succeeded)?;

// Maintain
m.forget(id)?;
m.clear()?;
m.maintenance_pass()?;  // archive superseded, prune stale low-weight memories
```

### Python

```python
from memoire import Memoire, Memory, MemoryPolicy, PolicyDecision, MemoireError

with Memoire("agent.db") as m:
    n: int         = m.remember(text)
    mems: list     = m.recall(query, top_k=5)
    mems: list     = m.recall_with_min_score(query, top_k=5, min_score=0.55)
    ok: bool       = m.reinforce_if_used(id, agent_output, task_succeeded)
    deleted: bool  = m.forget(id)
    count: int     = m.count()
    m.clear()

policy = MemoryPolicy()                          # FOLLOW≥0.75, HINT≥0.45
decisions = policy.evaluate(memories)            # list[PolicyDecision]
context   = policy.inject_context(decisions)     # str, ready for system prompt
```

### C FFI

```c
#include "memoire.h"

MemoireHandle* h = memoire_new("agent.db");  // or ":memory:"

memoire_remember(h, "content");

char* json = memoire_recall(h, "query", 5);
// [{"id":1,"content":"...","score":0.84,"trust":0.56,"state":"active","created_at":...}]
memoire_free_string(json);  // caller must free

memoire_reinforce_if_used(h, id, agent_output, 1 /*succeeded*/);

memoire_forget(h, id);
memoire_count(h);
memoire_clear(h);
memoire_free(h);
```

---

## Multi-Language Bindings

| Language | Mechanism | Path |
|----------|-----------|------|
| Python | ctypes | `bindings/python/` |
| Node.js | ffi-napi | `bindings/node/` |
| Go | cgo | `bindings/go/` |
| Any | C FFI | `include/memoire.h` |

```bash
# Python
pip install -e bindings/python

# Node.js
cd bindings/node && npm install && node demo.js

# Go
cd bindings/go/demo && go run main.go
```

---

## Configuration

```rust
use memoire::{Memoire, chunker::ChunkerConfig};

let m = Memoire::new("agent.db")?
    .with_chunker_config(ChunkerConfig {
        chunk_size: 64,   // words per chunk  (default: 128)
        overlap:    10,   // word overlap      (default: 20)
    });
```

Memory quality thresholds are intentionally not exposed as config — the scoring model is the invariant. Adjusting thresholds changes what "quality" means, which changes what the trust score means, which breaks the policy layer. If you need a different threshold, fork the quality module.

---

## Roadmap

This is a research agenda, not a feature checklist. Each item is a thesis:

**Active ingestion**  
Right now Memoire scores at write time. The next step is scoring at read time too — penalizing memories that are retrieved frequently but never reinforced. Retrieval without reinforcement is a signal of low utility, not high relevance.

**Cross-session contradiction tracking**  
The current contradiction resolver operates within a single claim key. The harder problem is cross-key contradiction: "always validate JWT issuer" conflicts with "disabled issuer validation for performance" even though the keys differ. This requires claim embedding, not claim string matching.

**Agent-specific memory namespacing**  
In multi-agent systems, what one agent learned is not necessarily what another should trust. Memory needs provenance — who stored it, under what task context, and whether that agent's track record justifies trust propagation.

**Confidence calibration from outcomes**  
The current trust formula weights reinforcement linearly. A better model would weight by the difficulty of the task the memory helped with — easy tasks reinforce less than hard ones.

**Streaming ingestion**  
For long coding sessions, waiting until the session ends to write memory means losing the most recent context. Streaming ingestion with in-flight dedup would let agents write continuously without blocking.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The quality module (`src/quality.rs`) is where most of the interesting decisions live — that's the right place to start if you want to understand or challenge the scoring model.

## License

MIT. See [LICENSE](LICENSE).


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

# Changelog

All notable changes to Memoire will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Metadata tagging (`project`, `session_id`, `language`) on stored memories
- Filtered recall: `recall_where(query, project="my-api")`
- HNSW approximate nearest-neighbour index via `usearch` for 100k+ memory stores
- MCP (Model Context Protocol) server mode — `examples/mcp_server.py` prototype already included
- Node.js native addon via `napi-rs` (faster than ffi-napi)
- `wasm32-wasi` target for browser/edge deployment

---

## [0.1.0] — 2024-01-XX

### Added
- **Core Rust library** (`cdylib` + `rlib`)
  - `Memoire::new(db_path)` — open/create persistent SQLite store
  - `Memoire::in_memory()` — ephemeral store for tests
  - `Memoire::remember(content)` — chunk, embed, store
  - `Memoire::recall(query, top_k)` — semantic similarity search
  - `Memoire::forget(id)` — delete by id
  - `Memoire::count()` — total stored chunks
  - `Memoire::clear()` — erase all memories
  - `Memoire::export_all()` — dump all memories ordered by recency

- **Chunker** (`src/chunker.rs`)
  - Sliding-window word chunker, configurable `chunk_size` and `overlap`
  - Default: 128 words/chunk, 20-word overlap

- **Embedder** (`src/embedder.rs`)
  - `all-MiniLM-L6-v2` via `fastembed` — 384-dim, ONNX Runtime, CPU-only
  - Batch embedding support
  - Model cached at `~/.cache/huggingface/hub/` on first use

- **SQLite store** (`src/store.rs`)
  - WAL mode, bundled SQLite via `rusqlite`
  - Embeddings stored as little-endian f32 BLOB (1536 bytes per chunk)
  - Pure-Rust cosine similarity scan, O(n)
  - `Store::all()` for full export

- **C FFI** (`src/ffi.rs` + `include/memoire.h`)
  - `memoire_new`, `memoire_free`
  - `memoire_remember`, `memoire_recall`
  - `memoire_forget`, `memoire_count`, `memoire_clear`
  - `memoire_free_string` — safe deallocation of JSON results
  - Recall results serialised as JSON for language-agnostic consumption

- **CLI binary** (`src/bin/cli.rs`)
  - `remember`, `recall`, `forget`, `count`, `clear`, `import`, `export`, `info`
  - Score bar visualisation in recall output
  - Stdin support for `remember -`

- **Python binding** (`bindings/python/`)
  - `Memoire` class with context manager support
  - `Memory` dataclass, `MemoireError` exception
  - `remember_lines()`, `recall_one()` convenience helpers
  - Full pytest test suite (17 tests)
  - `conftest.py` for automatic library discovery

- **Node.js binding** (`bindings/node/`)
  - `Memoire` class via `ffi-napi`
  - Full method coverage + demo

- **Go binding** (`bindings/go/`)
  - `cgo`-based binding
  - Full test suite
  - Demo main program

- **Benchmarks** (`benches/core_bench.rs`)
  - Criterion benchmarks for `remember`, `recall`, `recall_top_k`, `chunker`
  - Parameterised over input sizes and store sizes

- **Examples**
  - `basic_usage.rs` — Rust quick-start
  - `agent_demo.py` — Python multi-session simulation
  - `aider_plugin.py` — Aider hook integration
  - `mcp_server.py` — MCP tool server (requires `pip install mcp`)
  - `openai_agent.py` — OpenAI function-calling agent loop

- **Scripts**
  - `scripts/download_model.py` — offline model pre-download
  - `scripts/benchmark_report.py` — Markdown table from Criterion JSON

- **Docs**
  - `docs/ARCHITECTURE.md` — full internal design guide
  - `docs/FFI_GUIDE.md` — Python, Node, Go, Ruby, C, C# integration examples
  - `CONTRIBUTING.md`

- **CI** (`.github/workflows/ci.yml`)
  - Ubuntu, macOS, Windows matrix
  - `cargo fmt`, `clippy`, unit tests, integration tests, doc tests
  - Shared library symbol verification
  - Hugging Face model cache between runs

# Changelog

All notable changes to Memoire will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Metadata tagging (`project`, `session_id`, `language`) on stored memories
- Filtered recall: `recall_where(query, project="my-api")`
- Node.js native addon via `napi-rs` (faster than ffi-napi)
- `wasm32-wasi` target for browser/edge deployment (stubs added in P3.4)
- NLI-based contradiction detection (P3.3 — opt-in via `use_nli_contradiction: true`)
- Cross-encoder re-ranking (`recall_reranked` / P3.1 — fastembed TextRerank)
- Memory snapshot export/import CLI + MCP tools (P3.2 — `export_namespace` / `import_namespace`)

---

## [0.3.0] — 2026-06-07

### Fixed (P0 — bugs)

- **P0.1 — MCP namespace bypass:** `get_memoire()` in `server.py` now takes
  `(db_path, namespace)` as its cache key. All ten MCP tools (`memoire_remember`,
  `memoire_recall`, `memoire_reinforce`, `memoire_penalize`, `memoire_batch_feedback`,
  `memoire_status`, `memoire_resolve_conflicts`, `memoire_forget`, `memoire_count`,
  `memoire_clear`) accept `namespace: str = "default"` and forward it correctly.
  Multi-agent deployments on the same database file no longer silently share state.

- **P0.2 — C header out of sync:** `include/memoire.h` was missing declarations for
  `memoire_new_ns`, `memoire_reinforce_if_used`, `memoire_penalize_if_used`, and
  `memoire_resolve_contradictions`. All four are now declared with correct signatures.
  C consumers can link against every `pub extern "C"` function in `ffi.rs`.

- **P0.3 — Telemetry field name mismatch:** `server.py`'s `JsonFormatter` emitted
  `"level"` but the dashboard TypeScript read `"levelname"`. The server now emits
  `"levelname"` to match the conventional Python logging name. Severity coloring and
  badges in the dashboard now work correctly.

- **P0.4 — dbPath path injection in dashboard:** `dashboard/app/api/databases/route.ts`
  previously passed `dbPath` from the request body directly to CLI arguments with no
  validation. Added `getAllowedPaths()` and `validateDbPath()` helpers that check against
  the `MEMOIRE_ALLOWED_PATHS` environment variable (comma-separated directory allowlist).
  Requests outside the allowlist return HTTP 403. Empty `MEMOIRE_ALLOWED_PATHS` (default)
  allows all paths for local development. Created `dashboard/.env.local` with the
  configuration key documented.

- **P0.5 — Missing CHECK constraints on trust columns:** The `memories` table had no
  database-level constraints on `confidence`, `novelty`, `trust_ema`, or `archived`.
  The `CREATE TABLE` statement now enforces:
  `CHECK(confidence BETWEEN 0.0 AND 1.0)`, `CHECK(novelty BETWEEN 0.0 AND 1.0)`,
  `CHECK(archived IN (0, 1))`, `CHECK(trust_ema IS NULL OR trust_ema BETWEEN 0.0 AND 1.0)`.
  Application code additionally clamps all values before every INSERT/UPDATE so the
  constraint is never triggered by normal use.

### Changed (P1 — architecture)

- **P1.1 — RwLock for Store:** The `Store` is now protected by `RwLock<StoreInner>`
  instead of a single `Mutex`. Read operations (`recall`, `count`, `export_namespace`)
  acquire a shared read lock; write operations (`remember`, `reinforce_if_used`,
  `penalize_if_used`, `forget`, `clear`, `import_namespace`) acquire an exclusive write
  lock. Concurrent recalls from multiple agents no longer block each other.
  The `Embedder` remains behind `Arc<Mutex<...>>` — embedding inference requires
  exclusive access.

- **P1.2 — Axum HTTP server replaces dashboard CLI subprocesses:** Added
  `src/bin/memoire_server.rs` — a long-lived Axum 0.7 service that exposes the library
  behind a local HTTP API on port 6779 (`MEMOIRE_SERVER_PORT` to override). The
  dashboard's `route.ts` now calls `http://localhost:6779` via `fetch` instead of
  spawning two `execFileAsync` CLI subprocesses per GET request. If the server is
  unreachable, the dashboard returns HTTP 503 with a startup instruction. Added
  `axum = "0.7"` and `tokio = { version = "1", features = ["full"] }` to `Cargo.toml`.
  Added `[[bin]] name = "memoire-server"` to `Cargo.toml`.

- **P1.3 — Async filesystem calls in Next.js route handlers:** All remaining
  `readFileSync`, `existsSync` calls in `dashboard/app/api/databases/route.ts` replaced
  with `await readFile(...)` and `await access(...)` from `fs/promises`. Synchronous
  filesystem calls in async handlers no longer block the Node.js event loop under
  concurrent dashboard use.

### Added (P2 — core quality)

- **P2.1 — Cold-start trust:** Added `cold_start_weight: f32` (default `0.5`) to
  `ScoringConfig`. New memories receive `trust_ema = quality_score × cold_start_weight`
  on INSERT instead of starting at zero. A freshly stored, high-quality memory is
  immediately visible to recall without needing prior reinforcement. Set
  `cold_start_weight: 0.0` to restore the original behaviour.

- **P2.2 — Trust time decay:** Added `decay_rate: f32` (default `0.01`, half-life ~69 days)
  to `ScoringConfig`. Added `last_used_at INTEGER` column to the `memories` table with an
  idempotent schema migration. On every recall, effective trust is computed as
  `trust_ema × exp(−decay_rate × days_since_last_used)` and `last_used_at` is updated in
  a single batched UPDATE. Stale lessons now fade automatically. `decay_rate: 0.0` is a
  no-op. Added `last_used_at: Option<i64>` to the `Memory` struct.

- **P2.3 — MMR recall:** Added `Store::mmr_rerank()` using the existing
  `cosine_similarity` function (no new duplicate). Added `Memoire::recall_mmr(query,
  top_k, mmr_lambda)` — retrieves `max(top_k × 3, 20)` candidates then greedily selects
  `top_k` by marginal relevance. `mmr_lambda = 1.0` produces output identical to
  `recall`. Exposed as `recall_mmr(query, top_k, mmr_lambda=0.5)` in the PyO3 binding.

- **P2.4 — Code-aware chunker:** Rewrote `src/chunker.rs` to support three modes:
  - `ChunkerMode::Prose` — original sliding-window (byte-for-byte unchanged).
  - `ChunkerMode::Code(CodeLanguage)` — splits at function/class/impl/struct/enum/trait
    boundaries for Python, Rust, JavaScript, TypeScript, and Generic (regex).
  - `ChunkerMode::Auto` (new default) — detects triple-backtick language fences and
    dispatches to the appropriate mode; falls back to Prose if no fences are found.
  Added `mode: ChunkerMode` to `ChunkerConfig` (default `Auto` — backward compatible
  because Auto falls back to Prose on prose text). Added `tree-sitter = "0.22"` to
  `Cargo.toml`.

### Documentation

- `docs/ARCHITECTURE.md` — replaced "full cosine scan" retrieval description with the
  accurate hybrid path (linear scan below `hnsw_threshold=500`, HNSW above it, lazy
  rebuild); added sections for RwLock, trust model, cold-start, time decay, MMR flow,
  code-aware chunker, HTTP API server, and namespace isolation.
- `docs/FFI_GUIDE.md` — added `memoire_new_ns`, `memoire_reinforce_if_used`,
  `memoire_penalize_if_used`, `memoire_resolve_contradictions` to all language examples
  and the ownership table.
- `README.md` — fully updated to reflect all P0–P2 changes.

---

## [0.2.0] — 2026-06-05

### Added

- **Hard namespace isolation** — `Memoire::new_ns(db, namespace)` and `Memoire::in_memory_ns(ns)`
  scope all SQL reads/writes to a partition column. Data written in one namespace is completely
  hidden from any other namespace sharing the same SQLite file. Verified with isolation tests
  in both Rust and Python.

- **PyO3 native extension** (`src/py_ffi.rs`) — replaced the `ctypes` FFI layer in the Python
  binding with a compiled Rust extension module built via `maturin`. Classes `PyMemoire` and
  `PyMemory` are exposed natively; no dynamic library loading at runtime. The C FFI layer is
  preserved for Node.js and Go consumers.
  - `Memoire(db_path, namespace="default")` — namespace parameter added.
  - `MemoireError` Python exception class exposed from Rust.
  - `pyproject.toml` updated to use the `maturin` build backend.
  - All 26 Python tests now run against the native module (`pytest bindings/python/tests/ -v`).

- **`cache-models` CLI command** — `memoire cache-models` pre-downloads and caches the
  `all-MiniLM-L6-v2` embedding model to the FastEmbed local cache. Allows subsequent runs
  on air-gapped machines without any network access or cold-start latency.

- **Observability dashboard** (`dashboard/`) — Next.js 15 application with TypeScript and
  Tailwind CSS providing local observability for any Memoire SQLite database.
  - `/api/logs` — streams `~/.memoire/logs/mcp-server.jsonl` as parsed JSON.
  - `/api/databases` — scans log history to discover active DB paths, queries memory records,
    and supports `remember`, `recall`, and `forget` actions via `POST`.
  - Dark glassmorphic UI with auto-refresh (5 s), semantic search recall tester, memory
    explorer with trust/state/uncertainty badges, telemetry log timeline.
  - Run locally: `cd dashboard && npm run dev` → `http://localhost:3000`.

### Changed

- **Framework adapters** (`bindings/python/memoire/adapters.py`) — `MemoireRetriever` and
  `MemoireIndex` now accept a `namespace` parameter and forward it to the underlying
  `Memoire(db_path, namespace=...)` constructor.

- **MCP server** (`examples/mcp_server.py`) — trust-aware `save_lesson` and `get_lessons`
  tools, plus low-level passthrough tools (`memoire_remember`, `memoire_recall`,
  `memoire_forget`, `memoire_status`). Requires `pip install mcp`.

- **Framework adapters** (`bindings/python/memoire/adapters.py`)
  - `MemoireRetriever` — LangChain `BaseRetriever`-compatible; policy-filtered, works with
    LCEL pipes and `RetrievalQA`. Requires `pip install langchain langchain-core`.
  - `MemoireIndex` — LlamaIndex-compatible index with `as_retriever()` and `as_query_engine()`.
    Requires `pip install llama-index-core`.
  - Both adapters apply `MemoryPolicy` internally — IGNORE-ranked memories never reach the LLM.

- **`EmbedProvider` trait** (`src/embedder.rs`) — pluggable embedding backend.
- **`recall_within_days`** — time-bounded recall.
- **`ScoringWeights`** struct — scoring constants surfaced for testing.

### Fixed

- `Command::CacheModels` match arm was missing from the CLI `match` block, causing a Rust
  compile error (`E0004: non-exhaustive patterns`). Added `unreachable!()` arm since the
  variant is handled via early `if let` return before `Memoire::new()` is invoked.

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

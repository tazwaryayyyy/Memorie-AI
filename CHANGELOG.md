# Changelog

All notable changes to Memoire will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

> **Release status:** no version has been tagged or published to any registry. `Cargo.toml`,
> `bindings/python/pyproject.toml` and `mcp-server/pyproject.toml` are all at `0.1.0`. The
> `0.2.0` / `0.3.0` entries below record development milestones, not published releases.

---

## [Unreleased]
### Changed (Step 8b-2: weights, contradiction verdicts, shadow corroboration; breaking, pre-publish)
- **`ScoringWeights::default()`**: `novelty` and `evidence` weigh `0.0`; the three semantic
  centroid weights are renormalised to `3/7`, `5/14`, `3/14` (original 0.30 : 0.25 : 0.15
  ratio, sum 1). Both features are still computed and stored. `confidence` is now
  `0.50 + 0.20 * actionability` (the `0.30 * evidence` term is gone), so attacker-controlled
  keywords can no longer raise importance or confidence. Scores are on a different scale:
  with `AdmissionMode::QualityGate`, the unchanged `>= 0.50` bar now hides most realistic
  lessons until it is calibrated.
- **Prototype leakage**: the `"Always use Decimal for financial calculations never use
  float"` reusability exemplar in `Memoire::compute_prototypes` is replaced by a
  domain-neutral rule about closing connections and file handles.
- **Lexical contradiction gate** (`NliChecker::lexical_label`): `Contradiction` now needs
  opposing polarity AND `negation_asymmetry >= 1`; opposing polarity alone is `Neutral`.
  `detect_polarity` and `negation_asymmetry` share one lexicon (`NEGATIVE_TERMS`; asymmetry
  also counts `not` / `no`), fixing `"disabled"`, `"removed"`, `"stopped"`, `"reverted"`,
  `"vulnerable"` and `"do not"` counting for polarity but not for asymmetry.
- **Contradiction losers**: a verdict from the lexical fallback now `quarantines` the loser
  (hidden, never deleted, recoverable with `unquarantine`); a verdict from an installed ONNX
  model still archives it. The `Legacy` polarity gate still archives. `count()` therefore no
  longer drops after a lexical contradiction.
- **Shadow corroboration** (`AdmissionMode::QualityGate`): a chunk that restates a hidden
  `shadow` row (cosine >= 0.92, no contradiction, different `remember` call) promotes it to
  `trial` instead of inserting a duplicate; `remember` returns the promoted id. No
  reinforcement is recorded.

### Added
- `NliChecker::check_with_source` and `NliSource`; `Store::corroborate_shadow`,
  `store::NEAR_DUPLICATE_COSINE`.
- Tests: weights/confidence, lexicon alignment, lexical gate (TLS case), verdict source,
  authoritative-archive vs lexical-quarantine, corroboration (similarity, independence,
  state, contradiction), `tests/lifecycle.rs` corroboration scenarios, and oracle tests that
  prove keyword evidence cannot buy a pass.

### Known gaps (not addressed here)
- Tokens are whitespace-delimited, so a lexicon term followed by punctuation (`disabled.`)
  is not matched by `detect_polarity` / `negation_asymmetry`.
- `should_supersede` still compares the incoming memory's ingest importance with
  `0.50 + 0.40 * gravity`; with the new weights the existing memory almost always wins.
- Remaining exemplars still touch payments, hashing, parameterized queries and input
  validation.

### Changed (Step 8b-1: trust, lifecycle, feedback integrity; breaking, pre-publish)
- **Trust is a Beta posterior** `(2 + s) / (4 + s + f)` over `reinforcement_count` /
  `failure_count`, independent of ingest scores; `uncertainty = 4 / (4 + s + f)`. Removed
  `compute_trust`, the `trust_ema` blend and `ScoringConfig::{ema_new_weight, rc_saturation,
  cold_start_weight}`. Decay now regresses toward 0.5 using days since last *feedback*.
- **Ranking** is `0.75 * cosine + 0.20 * trust + 0.05 * recency`.
- **States**: `trial` and `active` are live; `shadow`, `probationary`, `quarantined` are
  hidden. `trial -> active` at `3s >= 8 + 7f`; live `-> quarantined` at `2f > 3s + 2`.
  `Store::search(.., include_hidden)` (renamed from `include_shadow`; default now excludes
  `shadow`) skips hidden rows before the top-k cosine cut, so they cannot crowd out live ones.
- **Bounded trial exposure**: a default recall returns at most `ScoringConfig::trial_cap`
  (default 1) unverified rows; `trial_min_cos` (default 0.0) is an optional relevance floor.
  With no feedback history this limits a default recall to one result.
- **`ScoringConfig`**: `admission: AdmissionMode::{AdmitAll (default), QualityGate}`,
  `contradiction_mode: ContradictionMode::{Off, Legacy, Nli (default)}` replacing
  `use_nli_contradiction`, `containment_threshold` replacing `jaccard_threshold`.
- **Feedback integrity**: `reinforce_if_used` / `penalize_if_used` run in immediate
  transactions; one vote per (memory, session id); `penalize_if_used` no longer decrements
  `reinforcement_count` or rewrites `importance_base` / `confidence`, skips quarantined rows
  and returns database errors instead of skipping the row. `failure_severity` now only gates
  (`> 0` counts one failure).
- **Attribution**: asymmetric containment of the memory's distinctive tokens replaces
  whole-output symmetric Jaccard, so long outputs are not penalised.
- `sample_namespace` samples `trial`/`active` rows; export excludes `quarantined` rows;
  `PreserveBounded` import re-derives state from the imported counts.
- Database `user_version` 2 -> 3 (`migrations/v3.sql`): live states re-derived from counts
  (approximate for databases written by earlier engines).

### Added
- `Memoire::{reinforce_declared, unquarantine, promote_shadow, recall_plain}` and the
  `Store` equivalents; `Memory::exposure`; Python `Memoire(admission=, contradiction=,
  trial_cap=)`, `reinforce_declared`, `unquarantine`, `promote_shadow`, `recall_plain`;
  `memoire-cli unquarantine <ID>`.
- `tests/lifecycle.rs` and in-crate tests for the state machine, trial cap, quarantine,
  migration, concurrency and plain recall (deterministic; no LLM).

### Fixed
- `Cargo.toml` `include` now ships `/migrations/**`; `include_str!("../migrations/v2.sql")`
  failed from a packaged crate.

### Not changed yet (Step 8b-3)
- MCP/SDK thresholds, `exposure` in MCP output, session ids in MCP, `MEMOIRE_TRIAL_CAP`.

### Documentation (earlier in Unreleased)

### Changed (breaking, pre-release)
- Python import package renamed `memoire` -> `memoire_engine` (distribution name `memoire-engine`,
  provisional and unpublished); native module is `memoire_engine.memoire_engine`. The unrelated
  PyPI project `memoire` ships an overlapping top-level `memoire` package.
- MCP server is now the `memoire_mcp` package (`memoire-mcp` / `python -m memoire_mcp`) instead of a
  generic top-level `server` module. `python mcp-server/server.py` no longer exists.
- Cargo package name is `memoire-engine`; the library target stays `memoire` (`use memoire::...`,
  `libmemoire.so` and the C ABI are unchanged).
- `memoire_export` (MCP) returns `{ok, error, snapshot}`; `memoire_recall_explained` gates shadow
  memories like `memoire_recall` (`include_shadow`).

### Fixed
- Namespace isolation in feedback paths: `Store::reinforce_if_used`, `penalize_if_used`,
  `update_imported_metadata`, the `last_used_at` batch update in `search`, and the follow-up
  reads/writes in `resolve_contradictions_for_id` now carry `AND namespace = ?`. Previously a
  handle could reinforce or penalize another namespace's row (sharing one SQLite file) by passing
  that row's id. Regression tests: `store.rs` (`*_is_namespace_scoped`) and `tests/namespace_feedback.rs`.
- `memoire_engine.adapters.MemoireRetriever` is now a real `langchain_core.retrievers.BaseRetriever`
  (`invoke`/`ainvoke`, LCEL pipes). `langchain-core` stays optional; `memoire_engine` and
  `memoire_engine.adapters` import with neither it nor `llama-index-core` installed. Call
  `retriever.invoke(query)`; the old ad-hoc `__call__` was removed and fields are pydantic fields.
  New extras: `memoire-engine[langchain]`, `memoire-engine[llama-index]`.
- `MemoireIndex.as_query_engine()` / `as_retriever()` passed a LlamaIndex `QueryBundle` to
  `Memoire.recall()`, which raised `TypeError`; the retriever wrapper now unwraps `query_str`.
- C FFI: `memoire_remember` is documented as returning `>= 0` (0 for blank input, duplicates and
  rejected chunks), `cbindgen.toml` lists all 12 exported functions, `memoire_new_ns`'s parameter is
  `namespace_` (`namespace` is a C++ keyword), and every `extern "C"` entry point that runs engine
  code converts a panic into its error return (`-1` / `NULL`) instead of aborting the host process.
  `tests/ffi_header_parity.rs` fails if `src/ffi.rs`, `include/memoire.h` and `cbindgen.toml` drift.
- ONNX Runtime is now linked instead of loaded at runtime: removed `ort`'s `load-dynamic` feature.
  It disabled linking entirely, so the first `Session`/embedder use panicked with "Failed to load
  ONNX Runtime dylib" unless a compatible runtime happened to be findable (on Windows it could
  pick up an unrelated `System32\onnxruntime.dll`). `libloading` is no longer in the dependency graph.
  Added `tests/ort_smoke.rs`, which runs real ONNX inference, so CI now catches this.
  Builds now download pyke's prebuilt ONNX Runtime (or use `ORT_LIB_LOCATION`); see `docs/RELEASE.md`.

### Security
- DB path: `MEMOIRE_DB_PATH` / `MEMOIRE_DB` now override `.memoire_path`, and a `.memoire_path`
  target must resolve inside the workspace root (symlinks followed); otherwise an error is returned.
- Imports are untrusted by default: `Memoire::import_namespace`, the Python `import_namespace`, the
  CLI `import` and the HTTP `/import` re-ingest `content` only and ignore trust metadata. Restoring
  your own backup is opt-in (`ImportTrust::PreserveBounded`, `preserve_trust=True`,
  `--preserve-trust`) and clamps every field (`trust_ema`/`confidence`/`importance_base` to [0, 1],
  `reinforcement_count` to at most `IMPORT_MAX_REINFORCEMENT`, `created_at` to [0, now]).
- MCP `memoire_clear` and `memoire_import` are not exposed unless `MEMOIRE_ENABLE_ADMIN_TOOLS=1`;
  `memoire_import` never preserves trust and accepts at most 1000 memories per call.

### Fixed (stabilization, `fix/p0-stabilization`, steps 1-7a)
- `Store::clone()` now shares one `Arc<RwLock<StoreInner>>`, so clones see the same
  `embedding_cache`, `fingerprints` and HNSW index instead of diverging copies.
- `Memoire::start_background_maintenance` reuses an ambient Tokio runtime or starts a private
  one-worker runtime, and `Drop` aborts all watcher tasks and shuts the owned runtime down.
- Namespace scoping also applied to `forget`, `clear` and `maintenance_pass`; fingerprints are
  released when a row is deleted or pruned, so identical content can be stored again.
- `embedder.dim()` fallback, `r2d2` `min_idle(1)` (no 20 eager connections at startup) and
  `as_chunks::<4>()` for embedding blob decoding.
- `OnnxNliChecker` now parses `id2label` from the model's `config.json`, its label is
  authoritative when a model is set (`Store::set_nli_checker`), and the lexical fallback runs
  only with no ONNX model or on inference error. There is still no automated test with a real
  MNLI model.
- Build: removed `+avx2,+fma` and `jobs=4` from `.cargo/config.toml` (binaries must run on any
  x86_64 CPU); `wasm32` dependencies are gated in `Cargo.toml`; `*.pdb` is gitignored.
- Text handling: char-boundary-safe `truncate_chars` / `preview` (`src/text.rs`).
- PyO3 calls convert panics into `MemoireError` (`src/panic_guard.rs`, `with_inner` +
  `catch_panic`) without poisoning the handle mutex; C FFI uses `guarded`.
- Python: `Memoire.remember_ids()` returns chunk ids (`remember()` returns the chunk count);
  Python 3.10 compatibility in the examples; `examples/agent_demo.py` frees `c_char_p` results
  correctly.
- MCP server: import order, UTC timestamps in logs, shadow gate on recall, `.memoire_path`
  workspace containment.
- CLI (`memoire-cli`) rejects unknown or misplaced flags (for example `--namespace` after the
  subcommand) and bounds `--top` to `1..=1000`.
- `examples/brutal_moment_demo.py` prints honest conditional banners instead of a fixed
  pass message, and `scripts/agent_behavior_benchmark.py` is labelled
  `[SYNTHETIC SMOKE TEST, not an LLM benchmark]`. Neither is evidence of agent improvement.

### Security (steps 6-7a, continued)
- `memoire-server` (experimental, local-only): binds `127.0.0.1` only; `MEMOIRE_API_TOKEN`
  unset means unauthenticated loopback, set-but-blank exits with code `2`, set requires a
  constant-time BLAKE3-checked Bearer token (except `GET /health`); loopback `Host` required
  and loopback `Origin` required when present; `db` must resolve inside `MEMOIRE_ALLOWED_PATHS`
  (default: working directory); `/clear` and `preserve_trust` need `MEMOIRE_ENABLE_ADMIN_TOOLS`;
  2 MiB body limit and at most 16 open (database, namespace) instances.
- Dashboard (experimental, local-only): bound to `127.0.0.1`; API routes use
  `dashboard/src/lib/localGuard.ts` (Host, Origin, `Sec-Fetch-Site`, JSON content type),
  default-deny path roots, forward `MEMOIRE_API_TOKEN`; the `clear` action was removed.

### Added
- `SECURITY.md`: threat model, `memoire-server` / MCP / dashboard controls, import trust, SQLite
  deletion and WAL caveats, supply-chain notes, and a disclosure process.
- CI now also runs `tests/ffi_header_parity.rs` and `tests/namespace_feedback.rs`.

### Documentation
- `README.md` rewritten to match the code: correct package and binary names (nothing is
  published yet), working source-build commands, `MemoryPolicy` thresholds (`0.70`/`0.40`,
  strict `0.80`/`0.50`), `remember()` vs `remember_ids()`, model-cache precedence
  (`$HF_HOME`, `$FASTEMBED_CACHE_DIR`, `./.fastembed_cache`) and the cold-start download, a
  support matrix, an accurate description of what CI runs, and a Known Limitations section
  (cold-start `shadow` deadlock, no real-LLM evaluation).
- `docs/ARCHITECTURE.md`: corrected the contradiction-resolution rule, the NLI path, the
  recall ranking formula, import trust, background tasks, SIMD, WASM, schema and HTTP
  server descriptions.

### Known issues (open at this point)
- Cold-start `shadow` deadlock: addressed in Step 8b-1 above (unmeasured with real agents).
- No real-LLM evaluation of the trust hypothesis exists yet. Step 8.

### Planned
- Metadata tagging (`project`, `session_id`, `language`) on stored memories
- Filtered recall: `recall_where(query, project="my-api")`
- Node.js native addon via `napi-rs` (the `bindings/node` C-FFI wrapper is unmaintained)
- A working `wasm32` runtime (`quality::now_ts()` currently uses `SystemTime::now()`)
- CI coverage for the Python SDK tests, `wasm32` and dashboard checks

*Previously listed here but already present in the code at `fe10f11`:* NLI contradiction
detection (`NliChecker`, enabled by default via `use_nli_contradiction`), cross-encoder
re-ranking (`Memoire::with_reranker` / `recall_reranked`), and namespace snapshot export/import
(CLI, MCP, HTTP, Python, Rust).

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

- **P1.1 — RwLock + SQLite pool for Store:** The `Store` now keeps in-memory search
  state behind `RwLock<StoreInner>` and uses an `r2d2_sqlite` connection pool for DB
  access. Hot recall paths use shared read locks and independent SQLite connections;
  writes that mutate the embedding/fingerprint/HNSW cache still acquire an exclusive
  cache lock. The `Embedder` remains behind `Arc<Mutex<...>>` — embedding inference
  requires exclusive access.

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

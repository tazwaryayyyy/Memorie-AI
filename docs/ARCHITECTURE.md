# Memoire — Architecture Deep Dive

## Overview

Memoire is structured as four clean layers with no circular dependencies:

```
   Callers (CLI / MCP / HTTP API / C FFI / PyO3)
     │
     ▼
 ┌──────────────────────────────────────────────────────────┐
 │  Public API  (src/lib.rs)                                │
 │  struct Memoire { store, embedder, chunker_config,        │
 │                   scoring_config, reranker? }             │
 │  fn new / new_ns / remember / recall / recall_mmr /       │
 │     recall_reranked / forget / count / clear /            │
 │     export_namespace / import_namespace                   │
 └────────────┬──────────────────────────┬──────────────────┘
              │                          │
              ▼                          ▼
 ┌────────────────────┐      ┌─────────────────────────────┐
 │  Chunker           │      │  Embedder                   │
 │  (src/chunker.rs)  │      │  (src/embedder.rs)          │
 │                    │      │                             │
 │  Auto-detecting    │      │  EmbedProvider trait        │
 │  prose + code-     │      │  `Embedder` impl (fastembed)│
 │  aware splitter    │      │  all-MiniLM-L6-V2           │
 │  → Vec<String>     │      │  ONNX Runtime (local)       │
 │                    │      │  → Vec<Vec<f32>> (384-dim)  │
 └────────────────────┘      └─────────────────────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────────┐
                             │  Store                      │
                             │  (src/store.rs)             │
                             │                             │
                             │  rusqlite (bundled SQLite)  │
                             │  r2d2 SQLite pool           │
                             │  RwLock<StoreInner> cache   │
                             │  Cosine / HNSW search       │
                             │  MMR reranking              │
                             │  Trust decay on recall      │
                             │  → Vec<Memory>              │
                             └─────────────────────────────┘

 FFI layer (src/ffi.rs) sits alongside the public API and wraps it
 with C-compatible extern "C" functions. Excluded from WASM builds.

 HTTP API server (src/bin/memoire_server.rs) is a separate binary
 that wraps the library behind Axum endpoints on port 6779.
```

---

## Data Flow — `remember(content)`

```
content (String)
    │
    ▼
Chunker::chunk(content, config)
    │   Mode::Auto: detect code fences → pick strategy
    │   Mode::Prose: sliding window (128 words, 20-word overlap)
    │   Mode::Code(lang): split at fn/class/impl/struct/enum boundaries
    │
    ▼
Vec<String>  (one per chunk)
    │
    ▼
Embedder::embed(chunks)
    │   ONNX Runtime inference (CPU)
    │   all-MiniLM-L6-v2 → 384-dimensional float vectors
    │   Mean pooling + L2 normalisation (done by fastembed)
    │
    ▼
Vec<Vec<f32>>  (one 384-dim vector per chunk)
    │
    ▼
Store::insert_with_quality(chunk_text, embedding, scoring_config)
    │   Serialises f32 slice to LE bytes (BLOB column)
    │   Computes quality score (actionability, novelty, evidence, …)
    |   Trust starts at the neutral prior 0.5 (not derived from the quality score)
    │   Sets last_used_at = now()
    │   Writes to memories table in WAL-mode SQLite (CHECK constraints enforced)
    │
    ▼
Vec<i64>  (inserted row ids)
```

## Data Flow — `recall(query, top_k)`

```
query (String)
    │
    ▼
Embedder::embed_one(query)
    │   Same ONNX model, same normalisation
    │
    ▼
Vec<f32>  (384-dim query vector)
    │
    ▼
Store::search(query_vec, top_k)
    │   Path selected by store size vs. hnsw_threshold (default 500):
    │     < threshold → linear cosine scan (exact, O(n))
    │     ≥ threshold → HNSW ANN index (approximate, O(log n))
    |   Rank by score = 0.75*cosine + 0.20*trust + 0.05*recency_bonus
    |   Rows: store_state IN ('active','trial') and archived = 0;
    |   include_hidden=true additionally admits 'shadow', 'probationary', 'quarantined'
    |   and skips the trial cap. Hidden ids are excluded from the top-k*2 cosine
    |   candidate set so they cannot crowd out live rows.
    |   Trust per hit: (2+s)/(4+s+f), regressed toward 0.5 by days since last feedback
    |   Unverified rows (s+f == 0) are capped at trial_cap per recall
    │   Clamp trust to [0.0, 1.0]
    │   UPDATE last_used_at = now() for all returned IDs
    │   Truncate to top_k
    │
    ▼
Vec<Memory>  (sorted by score desc)
```

## Data Flow — `recall_mmr(query, top_k, lambda)`

```
query (String), lambda ∈ [0.0, 1.0]
    │
    ▼
recall inner pipeline with candidate_k = max(top_k × 3, 20)
    │   Retrieves a larger candidate set than needed
    │
    ▼
Store::mmr_rerank(candidates, query_embedding, top_k, lambda)
    │   Greedy MMR loop:
    │     score_i = lambda × sim(query, cand_i)
    │              − (1−lambda) × max_j∈selected sim(cand_i, selected_j)
    │   lambda=1.0 → pure relevance (identical to recall)
    │   lambda=0.0 → maximum diversity
    │
    ▼
Vec<Memory>  (deduplicated, top_k results)
```

---

## SQLite Schema

The block below is copied from `const SCHEMA` in `src/store.rs` (older databases are
upgraded in place by `ensure_quality_columns`, which adds any missing column).

```sql
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace   TEXT    NOT NULL DEFAULT 'default',
    content     TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    importance_base REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0.0 AND 1.0),
    novelty REAL NOT NULL DEFAULT 0.5 CHECK(novelty BETWEEN 0.0 AND 1.0),
    actionability REAL NOT NULL DEFAULT 0.5,
    evidence REAL NOT NULL DEFAULT 0.5,
    consequence REAL NOT NULL DEFAULT 0.5,
    reusability REAL NOT NULL DEFAULT 0.5,
    source_kind TEXT NOT NULL DEFAULT 'agent',
    store_state TEXT NOT NULL DEFAULT 'active',
    reinforcement_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at INTEGER,
    superseded_by INTEGER,
    contradiction_group TEXT,
    archived INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0, 1)),
    effective_weight REAL NOT NULL DEFAULT 1.0,
    fingerprint TEXT NOT NULL DEFAULT '',
    claim_key TEXT,
    claim_value TEXT,
    failure_count INTEGER NOT NULL DEFAULT 0,
    trust_ema     REAL CHECK(trust_ema IS NULL OR trust_ema BETWEEN 0.0 AND 1.0),
    trust_variance REAL NOT NULL DEFAULT 0.0,
    trust_samples INTEGER NOT NULL DEFAULT 0,
    trust_last_session_id TEXT,
    ingestion_driver TEXT,
    last_used_at  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_memories_created_at
    ON memories (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_claim_key
    ON memories (claim_key);
CREATE INDEX IF NOT EXISTS idx_memories_store_state
    ON memories (store_state, archived, superseded_by);
CREATE INDEX IF NOT EXISTS idx_memories_namespace
    ON memories (namespace, archived, superseded_by);
```

**CHECK constraints** are enforced at the SQLite level on `confidence`, `novelty`, `archived` and `trust_ema`. Application code additionally clamps values before every INSERT/UPDATE to prevent the constraint from ever being triggered by normal use.

**Storage estimate:** Each memory chunk costs:
- ~100–600 bytes for `content` (128 words average)
- 1,536 bytes for the `embedding` BLOB (384 × 4 bytes)
- ~80 bytes overhead (id, namespace, fingerprint, timestamps, trust columns, index entries)

Total ≈ **2 KB per chunk** on average. A database with 10,000 chunks ≈ **20 MB**.

---

## Trust Model

### Trust is a Beta posterior over attributed feedback

```
trust       = (2 + s) / (4 + s + f)       s = reinforcement_count, f = failure_count
uncertainty = 4 / (4 + s + f)
```

A prior of four pseudo-observations at 0.5. Trust does not depend on any ingest-time
score, so every admitted memory starts at exactly `0.5`. These constants are heuristic
defaults, not validated. The legacy `trust_ema`, `confidence`-weighted and
`cold_start_weight` machinery is gone; the `trust_ema` column is kept only for
snapshot compatibility and is no longer updated.

### Time decay

Displayed trust regresses toward the prior as feedback evidence ages:

```
trust_t = 0.5 + (trust - 0.5) * exp(-decay_rate * days_since_last_feedback)
```

Default `decay_rate: 0.01` (half-life about 69 days); `0.0` disables it. The clock is
`last_accessed_at`, which only feedback (and insert) writes, so merely being recalled
does not keep a memory fresh. Recall still records `last_used_at`.

### States and exposure

`store_state`: `trial` and `active` are live; `shadow`, `probationary` and `quarantined`
are hidden from default recall. Transitions use integer rules on `(s, f)` and are written
in the same immediate transaction as the feedback update:

| Transition | Rule |
|---|---|
| `trial -> active` | `3s >= 8 + 7f` (trust >= 0.70) |
| live `-> quarantined` | `2f > 3s + 2` (trust < 0.40) |
| `shadow` / `probationary -> live` | `Store::promote_shadow` (not yet called by ingest) |
| `quarantined -> live` | operator `unquarantine`, which clears `failure_count` |

Default recall returns at most `ScoringConfig::trial_cap` (default 1) unverified
(`s + f == 0`) rows; with no feedback history that bounds a default recall to one result.
Each result carries `exposure`: `follow`, `hint`, `trial`, `hidden` or `plain`.
Migration `user_version` 2 -> 3 (`migrations/v3.sql`) re-derives live states from the
stored counts; legacy penalties decremented `reinforcement_count`, so that backfill is
approximate.

---

## Embedding Layer

### Model: `all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Dimensions | 384 |
| Max tokens | 256 (word-piece) |
| Model size | ~23 MB (ONNX format) |
| Inference | CPU-only, ~50–100ms per batch on a laptop |
| License | Apache 2.0 |

The chunker's default chunk size of 128 words (~180 word-piece tokens on average) keeps all chunks comfortably within the 256-token limit.

### Why fastembed?

`fastembed` was chosen over raw `candle` or `ort` because:

1. **Zero configuration** — model download, tokenisation, mean pooling, and L2 normalisation are all handled internally.
2. **Prebuilt ONNX Runtime** — no system ONNX library is required; the `ort` crate downloads and statically links a prebuilt runtime at build time (needs network, or `ORT_LIB_LOCATION`). Do not enable `ort`'s `load-dynamic` or `copy-dylibs` features (see `CHANGELOG.md`). The embedding **model** is downloaded on first use (about 23 MB); cache directory precedence is `$HF_HOME`, else `$FASTEMBED_CACHE_DIR`, else `./.fastembed_cache` relative to the current working directory.
3. **Batch inference** — `embed(Vec<String>)` runs the full batch in a single ONNX call.
4. **Active maintenance** — kept up-to-date with the latest fastembed/ONNX Runtime releases.

---

## Similarity Search

Retrieval uses a hybrid path. Below `hnsw_threshold` (configurable via `ScoringConfig`,
default 500 in `quality.rs`), recall performs a linear scan over all stored embeddings.
Above the threshold, an HNSW approximate nearest neighbor index is used. The HNSW
index is rebuilt lazily when dirty. The linear scan path is exact; the HNSW path is
approximate with controllable recall-precision tradeoff.

### Path selection

| Condition | Path | Characteristics |
|---|---|---|
| `store_size < hnsw_threshold` (< 500) | Linear scan over in-memory embedding cache | Exact, O(n), zero index overhead |
| `store_size >= hnsw_threshold` (≥ 500) | HNSW ANN index (instant-distance) | Approximate, O(log n), rebuilt lazily on dirty flag |

For the primary use case — a developer's personal coding history — the memory store
commonly stays below the 500-chunk threshold, giving exact results with no index cost.
Above 500 chunks, the HNSW index activates automatically. The index is backed by the
in-memory embedding cache and rebuilt lazily after any insert or archive operation that
sets the `hnsw_dirty` flag.

---

## Cosine Similarity Implementation

```rust
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot:    f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
    dot / (norm_a * norm_b)
}
```

Because fastembed L2-normalises embeddings before returning them, `norm_a ≈ norm_b ≈ 1.0`. The division is essentially free and the dot product **is** the cosine similarity. The normalisation guard is kept for correctness.

The LLVM auto-vectoriser (enabled by `opt-level = 3`) may vectorise the inner loop, but only with the instruction sets the build target guarantees: on a default x86_64 build that is the SSE2 baseline. `.cargo/config.toml` deliberately does not enable `+avx2,+fma`, because release binaries and wheels must run on any x86_64 CPU. AVX2/FMA code is only produced if you build yourself with something like `RUSTFLAGS="-C target-cpu=native"`.

---

## Chunking Strategy

### Prose mode (sliding-window)

The original chunker is unchanged:

```
Input: [w0 w1 w2 ... w127 | w108 w109 ... w235 | w216 ...]
                      ↑ overlap (20 words)
```

Key properties:
- **Word boundaries only** — no mid-word splits, preserving tokeniser quality
- **At least one chunk always returned** for non-empty input
- **Overlap preserves context** — a sentence straddling a boundary appears in both adjacent chunks

### Code mode

`ChunkerMode::Code(lang)` splits at structural boundaries instead of word windows:

| Language | Boundary node types |
|---|---|
| Python | `function_definition`, `class_definition`, `decorated_definition` |
| Rust | `function_item`, `impl_item`, `struct_item`, `enum_item`, `trait_item` |
| JavaScript / TypeScript | `function_declaration`, `arrow_function`, `class_declaration`, `method_definition` |
| Generic | Lines matching `^def `, `^class `, `^fn `, `^pub fn `, `^impl `, `^struct `, `^enum `, `^trait ` |

Chunks exceeding `chunk_size × 3` characters fall back to the sliding window on that node's text only.

### Auto mode (default)

`ChunkerMode::Auto` inspects the text for triple-backtick fences:
- ` ```python` / ` ```rust` / ` ```js` / ` ```ts` → corresponding `CodeLanguage`
- ` ``` ` without a language tag → `CodeLanguage::Generic`
- No fences → `ChunkerMode::Prose` (byte-for-byte identical to explicit Prose mode)

---

## Thread Safety

The `Store` uses an `r2d2_sqlite` connection pool for SQLite access and a
`RwLock<StoreInner>` for in-memory search state:

- **Read operations** (`recall`, `recall_mmr`, `count`, `export_namespace`) draw their own SQLite connections from the pool. Hot recall paths use shared read locks for the embedding/HNSW cache, so concurrent recalls do not serialize on one `rusqlite::Connection`.
- **Write operations** (`remember`, `reinforce_if_used`, `penalize_if_used`, `forget`, `clear`, `import_namespace`) also draw pooled connections. Operations that mutate the in-memory cache acquire an exclusive write lock for that cache update.

The `Embedder` is always behind `Arc<Mutex<...>>` — embedding inference requires exclusive access regardless of read/write context.

```rust
// Concurrent reads are safe with Arc<Memoire> after P1.1
let m = Arc::new(Memoire::new("agent.db")?);

let m1 = Arc::clone(&m);
let m2 = Arc::clone(&m);

std::thread::spawn(move || { let _ = m1.recall("query", 5); });
std::thread::spawn(move || { let _ = m2.recall("other", 5); });
// Both recalls proceed concurrently
```

SQLite's WAL mode allows one writer and multiple readers to proceed concurrently at the database level. In-memory stores use a single pooled connection because SQLite `:memory:` databases are connection-local.

---

## FFI Design

All data crosses the FFI boundary as:

- **Inputs:** null-terminated UTF-8 `*const c_char` (standard C strings)
- **Outputs:** heap-allocated null-terminated JSON `*mut c_char` (for recall results)
- **Scalars:** `c_int`, `c_longlong` for counts and ids

### Why JSON for recall results?

Returning a `*mut RecallResult` struct requires the caller to match exact struct layout, know array length separately, and call a separate free function per element. JSON + a single `memoire_free_string` is simpler to consume from Python, Node, Go, Ruby, etc. The serialisation cost is negligible compared to embedding inference.

### Memory ownership contract

```
memoire_new()                   → caller owns handle, must call memoire_free()
memoire_new_ns()                → caller owns handle, must call memoire_free()
memoire_recall()                → caller owns returned string, must call memoire_free_string()
memoire_remember()              → no heap allocation returned
memoire_forget()                → no heap allocation returned
memoire_reinforce_if_used()     → no heap allocation returned
memoire_penalize_if_used()      → no heap allocation returned
memoire_resolve_contradictions()→ no heap allocation returned
memoire_free_string()           → transfers string back to Rust for deallocation
```

---

## HTTP API Server

`src/bin/memoire_server.rs` is an **experimental, local-only** Axum binary (`memoire-server`) that exposes the library over HTTP for the dashboard. It binds `127.0.0.1` only. Authentication, `Host`/`Origin` checks, the `MEMOIRE_ALLOWED_PATHS` sandbox and the admin gate are described in [`SECURITY.md`](../SECURITY.md).

```
GET  /health
POST /remember   { "db": "...", "ns": "...", "text": "..." }
POST /recall     { "db": "...", "ns": "...", "query": "...", "k": 5 }
POST /reinforce  { "db": "...", "ns": "...", "id": 1, ... }
POST /penalize   { "db": "...", "ns": "...", "ids": [...], ... }
POST /forget     { "db": "...", "ns": "...", "id": 1 }
POST /clear      { "db": "...", "ns": "..." }                       (admin-gated)
POST /import     { "db": "...", "ns": "...", "snapshot": {...}, "preserve_trust": false }
GET  /info       ?db=...&ns=...
GET  /export     ?db=...&ns=...
```

The server keeps an `Arc<Mutex<HashMap<(PathBuf, String), Arc<Memoire>>>>` cache keyed by (canonical database path, namespace), bounded to 16 open instances, so each pair has one long-lived instance and no per-request model load.

Default port: `6779`. Override with `MEMOIRE_SERVER_PORT`.

---

## Namespace Isolation

All SQL reads and writes are scoped to a `namespace` partition column. Data written in one namespace is completely invisible to any query in another namespace on the same file.

Namespace isolation is enforced at every entry point:
- Rust API (`new_ns`)
- C FFI (`memoire_new_ns`)
- PyO3 Python binding (`Memoire(db, namespace=...)`)
- MCP server (all tools accept `namespace: str = "default"`)
- HTTP API server (all endpoints accept `"ns"` field)
- Dashboard (passes namespace through to server)

The MCP server instance cache key is `(db_path, namespace)` — different namespaces on the same database get separate `Memoire` instances with no state leakage.

---

## NLI Contradiction Detection (P3.3)

`resolve_contradictions_for_id` (`src/store.rs`) is called after every `remember()` for each new chunk. When `ScoringConfig::contradiction_mode` is `Nli` (default), candidates are compared with `NliChecker` (`src/quality.rs`) against existing `active`/`shadow`, non-archived rows in the same namespace.

### Step 1 — Cosine gate
Pairs with cosine similarity below `nli_cosine_threshold` (default 0.80) are treated as different topics and skipped.

### Step 2 — Classification
- **ONNX NLI model supplied** (`Memoire::with_nli_model(path)` → `OnnxNliChecker`, labels read from the model's `config.json` `id2label`): its label is **authoritative**.
- **No ONNX model, or ONNX inference returned an error**: lexical fallback (`NliChecker::lexical_label`). `detect_polarity()` classifies each text as `Affirmative`, `Negative` or `Neutral` with a small keyword lexicon; only `Affirmative↔Negative` pairs can be labelled `Contradiction`. `negation_asymmetry(a, b)` is computed as a secondary signal, but both of its branches (score ≥ 1 and the conservative fallback) currently return `Contradiction`, so it does not change the outcome.

Known gap: `resolve_contradictions_for_id` returns early when the **new** memory's polarity is `Neutral`, so an ONNX model never sees such pairs. There is no automated test that runs a real MNLI model.

### Resolution
`quality::should_supersede(incoming_importance, existing_gravity)`: the incoming memory wins only if `importance_base > 0.50 + 0.40 × gravity`, where gravity comes from `compute_gravity_radius` (posterior trust of the existing memory, `reinforcement_count`, age). The loser is archived (`archived = 1`, `superseded_by` set to the winner's id), removed from the in-memory embedding cache and logged; its row is kept, not deleted. If the incoming memory loses, it is the one archived.

`ScoringConfig::contradiction_mode` selects `Nli` (default), `Legacy` (cosine + stored polarity only) or `Off` (this step is skipped entirely).

---

## Export / Import (P3.2)

`export_namespace()` serialises all non-archived memories in the current namespace to a JSON snapshot:

```json
{
  "memoire_export_version": 1,
  "namespace": "billing-agent",
  "exported_at": 1718000000,
  "memories": [
    { "content": "...", "trust_ema": 0.62, "reinforcement_count": 3, ... }
  ]
}
```

Embeddings and raw IDs are **never exported** — they are recomputed on import. `import_namespace()` calls `remember()` on each content string (which re-embeds, deduplicates, and runs quality scoring). By default (`ImportTrust::Reset`; Python `preserve_trust=False`; CLI without `--preserve-trust`) that is all: snapshot metadata is ignored, so a hostile snapshot cannot inject trust. Only with explicit opt-in (`ImportTrust::PreserveBounded`) are `trust_ema`, `reinforcement_count`, `importance_base`, `confidence` and `created_at` overwritten from the snapshot, each clamped to bounds. The MCP `memoire_import` tool never preserves trust.

---

## WASM Target (P3.4)

On `wasm32-unknown-unknown`, the following modules are excluded via `#[cfg(not(target_arch = "wasm32"))]`:

- `embedder` (requires ONNX Runtime)
- `store` (requires rusqlite)
- `ffi` (native C ABI)
- `py_ffi` (PyO3)
- The `Memoire` struct and all its methods

The `quality` and `chunker` modules compile for `wasm32`. On wasm32 the `tree-sitter` AST chunking is not built (`cfg(not(target_arch = "wasm32"))`), so `chunk_code_ast` falls back to the regex-based code chunker there. This target is **postponed**: `quality::now_ts()` calls `SystemTime::now()`, which compiles on `wasm32-unknown-unknown` but is expected to panic at runtime, and CI does not check this target. WASM re-exports available:

```rust
pub use quality::{NliChecker, NliLabel, ScoringConfig, detect_polarity, cosine_similarity};
```

Build:
```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```

---

## Background Maintenance Scheduler (Component 6)

Memoire includes an asynchronous background maintenance scheduler that cleans up, updates indices (HNSW), and handles other routine tasks. 
- **Scheduler Actions**: Runs `Store::maintenance_pass()` automatically.
- **Triggers**: Supports both time-based (e.g. run every `interval_secs` seconds) and count-based triggers (e.g. run when insertion counter reaches `threshold` inserts).
- **Wiring**: `Memoire::start_background_maintenance` starts the watchers in `src/background.rs`: the interval scheduler (`spawn_maintenance_scheduler`), the insert-count watcher (`spawn_count_watcher`) and a reinforcement watcher that triggers `recompute_prototypes` every 100 reinforcements. It uses the ambient Tokio runtime if one exists, otherwise a private one-worker runtime. `Drop for Memoire` aborts every watcher handle and shuts down the owned runtime if there is one.

---

## Prototype Recomputation (Component 7)

- **Purpose**: Over time, as embedding models evolve or are fine-tuned, the original scoring prototype centroids can drift.
- **Action**: `recompute_prototypes()` re-embeds the 15 fixed prototype sentences (covering consequence, actionability, and reusability dimensions) using the current embedder. This updates the centroids stored in `self.prototypes` in-place, keeping quality scoring aligned with the latest embedding space.

---

## Recall Attribution and Explanation (Component 8)

- **Method**: `recall_explained(query, top_k)` returns a list of `RecallAttribution` structs.
- **Fields**:
  - `memory`: The retrieved memory chunk.
  - `semantic_score`: Cosine similarity of the query and memory.
  - `trust`: Decayed Beta-posterior trust.
  - `uncertainty`: The memory's uncertainty rating.
  - `effective_weight`: Final weighted scoring used for search ranking.
  - `rank_score`: Ensemble score combining `0.65 * score + 0.25 * trust + 0.10 * effective_weight`.
  - `reasons`: Human-readable tags describing why the memory was retrieved/ranked this way (e.g., "high semantic relevance", "hint-level trust", "high uncertainty").


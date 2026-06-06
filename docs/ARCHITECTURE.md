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
 │  prose + code-     │      │  FastEmbedProvider impl     │
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
                             │  RwLock<StoreInner>         │
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
    │   Sets initial trust_ema = quality_score × cold_start_weight
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
    │   Sort descending by weighted score (relevance + quality + trust)
    │   Apply trust decay: trust × exp(−decay_rate × days_since_last_used)
    │   Clamp decayed trust to [0.0, 1.0]
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

```sql
CREATE TABLE IF NOT EXISTS memories (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace            TEXT    NOT NULL DEFAULT 'default',
    content              TEXT    NOT NULL,
    embedding            BLOB    NOT NULL,   -- LE f32 × 384 = 1536 bytes
    fingerprint          TEXT    NOT NULL,
    created_at           INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    last_used_at         INTEGER,            -- updated on every recall
    reinforcement_count  INTEGER NOT NULL DEFAULT 0,
    trust_ema            REAL    CHECK(trust_ema IS NULL OR trust_ema BETWEEN 0.0 AND 1.0),
    confidence           REAL    NOT NULL DEFAULT 0.5 CHECK(confidence BETWEEN 0.0 AND 1.0),
    novelty              REAL    NOT NULL DEFAULT 0.5 CHECK(novelty BETWEEN 0.0 AND 1.0),
    archived             INTEGER NOT NULL DEFAULT 0 CHECK(archived IN (0, 1)),
    importance_base      REAL,
    contradiction_group  TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_namespace
    ON memories (namespace);
CREATE INDEX IF NOT EXISTS idx_memories_created_at
    ON memories (created_at DESC);
```

**CHECK constraints** are enforced at the SQLite level for all trust columns. Application code additionally clamps values before every INSERT/UPDATE to prevent the constraint from ever being triggered by normal use.

**Storage estimate:** Each memory chunk costs:
- ~100–600 bytes for `content` (128 words average)
- 1,536 bytes for the `embedding` BLOB (384 × 4 bytes)
- ~80 bytes overhead (id, namespace, fingerprint, timestamps, trust columns, index entries)

Total ≈ **2 KB per chunk** on average. A database with 10,000 chunks ≈ **20 MB**.

---

## Trust Model

### Cold-start

New memories receive a nonzero initial trust seeded from their quality score:

```
initial trust_ema = quality_score × cold_start_weight
```

Default `cold_start_weight: 0.5`. Set to `0.0` to restore zero-trust-at-birth behavior. This ensures a freshly stored, high-quality memory is immediately visible to recall without needing prior reinforcement.

### Time decay

On every recall, each returned memory has its effective trust decayed by time since last use:

```
effective_trust = stored_trust_ema × exp(−decay_rate × days_since_last_used)
```

Default `decay_rate: 0.01` → half-life ≈ 69 days. Set to `0.0` to disable decay entirely.
After decaying, `last_used_at` is reset to now, restarting the decay clock.

### EMA update (reinforce / penalize)

On reinforcement/penalty events, `trust_ema` is updated with an exponential moving average, weighted by `reinforcement_count`. This is unchanged from the original design.

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
2. **Bundled ONNX Runtime** — no system ONNX library required.
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

The LLVM auto-vectoriser (enabled by `opt-level = 3`) generates SIMD instructions (AVX2/SSE4.2) for the inner loop on x86_64, making the scan faster than it looks at the source level.

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

The `Store` is protected by a `RwLock<StoreInner>`:

- **Read operations** (`recall`, `recall_mmr`, `count`, `export_namespace`) acquire a shared read lock — multiple concurrent recalls do not block each other.
- **Write operations** (`remember`, `reinforce_if_used`, `penalize_if_used`, `forget`, `clear`, `import_namespace`) acquire an exclusive write lock.

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

SQLite's WAL mode allows one writer and multiple readers to proceed concurrently at the database level.

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

`src/bin/memoire_server.rs` is a standalone Axum binary (`memoire-server`) that exposes the library as a local HTTP API. The dashboard uses this instead of spawning CLI subprocesses per request.

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

The server keeps a `HashMap<(db_path, namespace), Arc<Mutex<Memoire>>>` cache so each `(db, namespace)` pair has exactly one long-lived instance — no cold-start model loading per request.

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

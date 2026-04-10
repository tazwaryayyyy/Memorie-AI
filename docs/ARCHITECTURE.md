# Memoire — Architecture Deep Dive

## Overview

Memoire is structured as three clean layers with no circular dependencies:

```
   Callers
     │
     ▼
 ┌──────────────────────────────────────────────────────────┐
 │  Public API  (src/lib.rs)                                │
 │  struct Memoire { store, embedder, chunker_config }       │
 │  fn new / remember / recall / forget / count / clear      │
 └────────────┬──────────────────────────┬──────────────────┘
              │                          │
              ▼                          ▼
 ┌────────────────────┐      ┌─────────────────────────────┐
 │  Chunker           │      │  Embedder                   │
 │  (src/chunker.rs)  │      │  (src/embedder.rs)          │
 │                    │      │                             │
 │  Sliding-window    │      │  fastembed wrapper          │
 │  word tokeniser    │      │  all-MiniLM-L6-V2           │
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
                             │  Cosine similarity scan     │
                             │  → Vec<Memory>              │
                             └─────────────────────────────┘

 FFI layer (src/ffi.rs) sits alongside the public API and wraps it
 with C-compatible extern "C" functions.
```

---

## Data Flow — `remember(content)`

```
content (String)
    │
    ▼
Chunker::chunk_text()
    │   Sliding window with overlap
    │   Default: 128 words/chunk, 20-word overlap
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
Store::insert(chunk_text, embedding)
    │   Serialises f32 slice to LE bytes (BLOB column)
    │   Writes to memories table in WAL-mode SQLite
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
    │   Full table scan: SELECT id, content, embedding FROM memories
    │   For each row:
    │     blob_to_vec(embedding_blob)
    │     cosine_similarity(query_vec, row_vec)
    │   Sort descending by score
    │   Truncate to top_k
    │
    ▼
Vec<Memory>  (sorted by score desc)
```

---

## SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,   -- LE f32 × 384 = 1536 bytes per row
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_created_at
    ON memories (created_at DESC);
```

**Storage estimate:** Each memory chunk costs:
- ~100–600 bytes for `content` (128 words average)
- 1,536 bytes for the `embedding` BLOB (384 × 4 bytes)
- ~40 bytes overhead (id, created_at, rowid, index entry)

Total ≈ **2 KB per chunk** on average.

A database with 10,000 chunks ≈ **20 MB**.

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

The chunker's default chunk size of 128 words (~180 word-piece tokens on average) keeps all chunks comfortably within the 256-token limit. Chunks that exceed the limit are silently truncated by the ONNX tokeniser — the 128-word default is conservative by design.

### Why fastembed?

`fastembed` was chosen over raw `candle` or `ort` because:

1. **Zero configuration** — model download, tokenisation, mean pooling, and L2 normalisation are all handled internally.
2. **Bundled ONNX Runtime** — no system ONNX library required.
3. **Batch inference** — `embed(Vec<String>)` runs the full batch in a single ONNX call, which is faster than calling the model once per chunk.
4. **Active maintenance** — kept up-to-date with the latest fastembed/ONNX Runtime releases.

---

## Similarity Search

Memoire uses **pure-Rust cosine similarity** over a full table scan rather than an approximate nearest-neighbour (ANN) index.

### Why full scan?

| Approach | Pros | Cons |
|---|---|---|
| Full cosine scan | Simple, exact, zero deps | O(n) per query |
| HNSW (e.g. usearch) | O(log n) query | Index build cost, extra dependency |
| sqlite-vec | SQL integration | Fragile extension loading, external dep |

For the primary use case — a developer's personal coding history — the memory store rarely exceeds 50,000 chunks. At 384 dimensions:

- 10,000 chunks → ~45ms recall (well under 100ms UX threshold)
- 50,000 chunks → ~200ms recall (still acceptable)
- 100,000+ chunks → consider upgrading to HNSW (see Roadmap)

The full-scan approach also makes the recall results **exactly deterministic** — no approximate index errors, no graph construction failures.

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

## FFI Design

All data crosses the FFI boundary as:

- **Inputs:** null-terminated UTF-8 `*const c_char` (standard C strings)
- **Outputs:** heap-allocated null-terminated JSON `*mut c_char` (for recall results)
- **Scalars:** `c_int`, `c_longlong` for counts and ids

### Why JSON for recall results?

Returning a `*mut RecallResult` struct requires the caller to:
- Match the exact struct layout (including padding)
- Know the array length separately
- Call a separate free function per element

JSON + a single `memoire_free_string` is simpler to consume from Python, Node, Go, Ruby, etc. The serialisation cost is negligible compared to embedding inference.

### Memory ownership contract

```
memoire_new()         → caller owns handle, must call memoire_free()
memoire_recall()      → caller owns returned string, must call memoire_free_string()
memoire_remember()    → no heap allocation returned
memoire_forget()      → no heap allocation returned
memoire_free_string() → transfers string back to Rust for deallocation
```

---

## Chunking Strategy

The sliding-window word chunker preserves context at chunk boundaries:

```
Input: [w0 w1 w2 ... w127 | w108 w109 ... w235 | w216 ...]
                      ↑ overlap (20 words)
```

Key properties:
- **Word boundaries only** — no mid-word splits, preserving tokeniser quality
- **At least one chunk always returned** for non-empty input
- **Overlap preserves context** — a sentence straddling a boundary appears in both adjacent chunks, ensuring neither chunk loses its context

---

## Thread Safety

`Memoire` is `Send` but not `Sync`. Each instance wraps a `rusqlite::Connection` which is `Send` but not `Sync`. For concurrent use:

```rust
// Option A: one instance per thread
std::thread::spawn(|| {
    let m = Memoire::new("agent.db").unwrap();
    m.remember("...");
});

// Option B: Arc<Mutex<Memoire>>
let m = Arc::new(Mutex::new(Memoire::new("agent.db").unwrap()));
let m2 = Arc::clone(&m);
std::thread::spawn(move || {
    m2.lock().unwrap().remember("...");
});
```

SQLite's WAL mode allows one writer and multiple readers to proceed concurrently at the database level, but `rusqlite::Connection` itself must still be used from one thread at a time.

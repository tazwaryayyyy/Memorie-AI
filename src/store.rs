use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::Serialize;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::error::{MemoireError, Result};
use crate::quality::{
    compute_gravity_radius, cosine_similarity, decayed_trust, derive_live_state, detect_polarity,
    effective_weight, is_live_state, meets_follow, now_ts, posterior_trust, recency_bonus,
    should_supersede, token_containment, trust_uncertainty, ContradictionMode, NliChecker,
    NliLabel, NliSource, OnnxNliChecker, Polarity, QualityMeta, ScoringConfig,
};

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub score: f32,
    pub trust: f32,
    pub trust_variance: f64,
    pub trust_samples: i64,
    pub trust_last_session_id: Option<String>,
    pub effective_weight: f32,
    pub reinforcement_count: i64,
    pub ingestion_driver: String,
    /// Uncertainty âˆˆ [0, 1]: high when little reinforcement or when the memory
    /// has been both reinforced and penalized (oscillation). Agents may use this
    /// to modulate how strongly they act on a recall result.
    pub uncertainty: f32,
    pub state: String,
    /// How this memory is being offered: `"follow"` (posterior trust >= 0.70),
    /// `"hint"` (has feedback evidence, below follow), `"trial"` (no feedback yet:
    /// unverified), `"hidden"` (shadow / probationary / quarantined, only returned when
    /// hidden rows are requested) or `"plain"` (returned by `search_plain`).
    pub exposure: String,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
}

/// Trust and uncertainty delta produced by a single call to `penalize_if_used`.
#[derive(Debug, Clone, Serialize)]
pub struct PenaltyOutcome {
    pub id: i64,
    pub trust_before: f32,
    pub trust_after: f32,
    /// Uncertainty of this memory after the penalty, âˆˆ [0, 1].
    /// Increases as failure_count rises relative to reinforcement_count.
    pub uncertainty_after: f32,
}

pub struct Store {
    pool: Pool<SqliteConnectionManager>,
    /// Shared between clones so a cloned `Store` (e.g. the background
    /// maintenance handle) mutates the same cache/HNSW/fingerprint state.
    inner: Arc<RwLock<StoreInner>>,
    pub config: ScoringConfig,
    /// Namespace this store instance is scoped to. All reads/writes are filtered
    /// by this value. Defaults to `"default"`. Set via `Store::open_ns()`.
    pub namespace: String,
    nli_checker: NliChecker,
}

impl Clone for Store {
    /// Cheap handle clone: shares the connection pool **and** the in-memory
    /// cache (`StoreInner`) with the original. Mutations through any clone
    /// (insert, forget, `maintenance_pass`) are visible to all of them.
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
            inner: Arc::clone(&self.inner),
            config: self.config.clone(),
            namespace: self.namespace.clone(),
            nli_checker: self.nli_checker.clone(),
        }
    }
}
/// âœ… FIX #5 â€” In-memory HNSW embedding point wrapper.
///
/// Wraps a 384-dim embedding vector. Implements `instant_distance::Point`
/// using L2 distance. For L2-normalised fastembed vectors:
///   ||a âˆ’ b||Â² = 2Â·(1 âˆ’ cosine) â†’ closer L2 = higher cosine.
#[derive(Clone)]
pub struct EmbeddingPoint(pub Vec<f32>);

impl instant_distance::Point for EmbeddingPoint {
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }
}

/// In-memory search state protected by `Store::inner`.
///
/// SQLite connections live in `Store::pool`, so readers and writers do not
/// serialize through one `rusqlite::Connection`.
struct StoreInner {
    /// In-memory cache of (sqlite_id, embedding) for all live memories.
    /// Populated lazily on the first `max_similarity` or `search` call so that
    /// startup cost is O(fingerprint count) not O(embedding count).
    embedding_cache: Vec<(i64, Vec<f32>)>,
    /// True once the full embedding cache has been loaded from SQLite.
    /// Inserts update the cache only when this is true; otherwise they write
    /// to SQLite and let the lazy load pick them up.
    embedding_cache_loaded: bool,
    /// Ids in `embedding_cache` whose `store_state` is not live (shadow, probationary,
    /// quarantined). Candidate selection skips them unless hidden rows are requested, so
    /// hidden memories can never crowd live ones out of the top-k cosine candidates.
    /// Only meaningful while `embedding_cache_loaded` is true.
    hidden: HashSet<i64>,
    /// Ids in `embedding_cache` that carry feedback evidence (`s + f >= 1`). Candidate
    /// selection pools them separately from unverified rows, so unverified rows that
    /// happen to be closer in cosine can never crowd evidence-bearing memories out of the
    /// candidate set before the trial cap runs. Only meaningful while
    /// `embedding_cache_loaded` is true.
    verified: HashSet<i64>,
    /// O(1) fingerprint lookup â€” avoids a SQLite round-trip per insert.
    fingerprints: HashSet<String>,
    /// Approximate nearest-neighbour index. Rebuilt lazily when dirty.
    hnsw: Option<instant_distance::HnswMap<EmbeddingPoint, i64>>,
    /// True after any insert or archive â€” triggers a lazy HNSW rebuild.
    hnsw_dirty: bool,
}

impl StoreInner {
    /// Load all embeddings from SQLite into the in-memory cache if not yet loaded.
    /// Fingerprints are loaded eagerly at construction; embeddings are deferred here
    /// to keep startup cost proportional to fingerprint count (small strings) rather
    /// than embedding count (1536 bytes Ã— N).
    fn ensure_embedding_cache(&mut self, conn: &Connection, namespace: &str) -> Result<()> {
        if self.embedding_cache_loaded {
            return Ok(());
        }
        let mut stmt = conn.prepare(
            "SELECT id, embedding, store_state, reinforcement_count + failure_count
             FROM memories
             WHERE namespace = ?1 AND archived = 0 AND superseded_by IS NULL",
        )?;
        for (id, blob, state, evidence) in stmt
            .query_map(rusqlite::params![namespace], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, i64>(3)?,
                ))
            })?
            .flatten()
        {
            if let Some(emb) = blob_to_vec(&blob) {
                self.embedding_cache.push((id, emb));
                if !is_live_state(&state) {
                    self.hidden.insert(id);
                }
                if evidence > 0 {
                    self.verified.insert(id);
                }
            }
        }
        self.embedding_cache_loaded = true;
        self.hnsw_dirty = !self.embedding_cache.is_empty();
        log::debug!(
            "embedding cache loaded: {} vectors (ns={})",
            self.embedding_cache.len(),
            namespace
        );
        Ok(())
    }

    fn remove_from_cache(&mut self, id: i64) {
        if self.embedding_cache_loaded {
            self.embedding_cache.retain(|(cid, _)| *cid != id);
            self.hidden.remove(&id);
            self.verified.remove(&id);
            self.hnsw_dirty = true;
        }
    }

    /// Record whether a cached memory is hidden (non-live state).
    fn set_hidden(&mut self, id: i64, hidden: bool) {
        if hidden {
            self.hidden.insert(id);
        } else {
            self.hidden.remove(&id);
        }
    }
}

/// Largest `reinforcement_count` accepted from an imported snapshot. A sanity
/// bound against absurd values, not a claim about how much trust is deserved.
pub const IMPORT_MAX_REINFORCEMENT: i64 = 1_000;

/// Cosine at or above which two chunks count as restatements of one another when
/// corroborating a hidden `shadow` row (see `Store::corroborate_shadow`).
pub const NEAR_DUPLICATE_COSINE: f32 = 0.92;

/// Cosine similarity between agent output and memory above which text attribution is
/// considered satisfied without lexical overlap.
const COSINE_ATTRIBUTION_THRESHOLD: f32 = 0.75;

const SCHEMA: &str = "
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
";

pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn blob_to_vec(blob: &[u8]) -> Option<Vec<f32>> {
    let (chunks, rem) = blob.as_chunks::<4>();
    if !rem.is_empty() {
        return None;
    }
    Some(chunks.iter().copied().map(f32::from_le_bytes).collect())
}

/// Load only fingerprints at startup â€” O(n Ã— 64 bytes) instead of O(n Ã— 1536 bytes).
/// Embedding vectors are loaded lazily on the first search via `StoreInner::ensure_embedding_cache`.
fn load_fingerprints_from_db(conn: &Connection, namespace: &str) -> Result<HashSet<String>> {
    let mut stmt = conn.prepare(
        "SELECT fingerprint FROM memories
         WHERE namespace = ?1 AND archived = 0 AND superseded_by IS NULL AND fingerprint != ''",
    )?;
    let fingerprints: HashSet<String> = stmt
        .query_map(rusqlite::params![namespace], |r| r.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();
    Ok(fingerprints)
}

/// Drop `fp` from the in-memory dedup set after the row that owned it was
/// deleted, unless another live row in this namespace still carries the same
/// fingerprint. "Live" matches `load_fingerprints_from_db`
/// (`archived = 0 AND superseded_by IS NULL`). Call *after* the row is gone.
fn release_fingerprint(
    inner: &mut StoreInner,
    conn: &Connection,
    namespace: &str,
    fp: &str,
) -> Result<()> {
    if fp.is_empty() {
        return Ok(());
    }
    let still_live: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM memories
                       WHERE namespace = ?1 AND fingerprint = ?2
                         AND archived = 0 AND superseded_by IS NULL)",
        params![namespace, fp],
        |r| r.get(0),
    )?;
    if !still_live {
        inner.fingerprints.remove(fp);
    }
    Ok(())
}

/// Rebuild the HNSW index from the current in-memory embedding cache.
fn rebuild_hnsw(inner: &mut StoreInner) {
    if inner.embedding_cache.is_empty() {
        inner.hnsw = None;
        inner.hnsw_dirty = false;
        return;
    }
    let (points, ids): (Vec<EmbeddingPoint>, Vec<i64>) = inner
        .embedding_cache
        .iter()
        .map(|(id, emb)| (EmbeddingPoint(emb.clone()), *id))
        .unzip();
    let n = points.len();
    let hnsw = instant_distance::Builder::default().build(points, ids);
    inner.hnsw = Some(hnsw);
    inner.hnsw_dirty = false;
    log::debug!("hnsw rebuilt: {} points", n);
}

fn search_candidates_read(
    inner: &StoreInner,
    query_vec: &[f32],
    top_k: usize,
    config: &ScoringConfig,
    include_hidden: bool,
) -> Vec<(i64, f32)> {
    let visible = |id: &i64| include_hidden || !inner.hidden.contains(id);
    let pool_size = top_k * 2;

    // Evidence-bearing pool: exact scan of the (small) verified set, so closer
    // unverified rows cannot crowd these out of the candidates.
    let mut pool: Vec<(i64, f32)> = inner
        .embedding_cache
        .iter()
        .filter(|(id, _)| inner.verified.contains(id) && visible(id))
        .map(|(id, emb)| (*id, cosine_similarity(query_vec, emb)))
        .collect();
    pool.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pool.truncate(pool_size);

    // Unverified pool: ANN above the HNSW threshold, otherwise a linear scan.
    let unverified = |id: &i64| visible(id) && !inner.verified.contains(id);
    if inner.embedding_cache.len() >= config.hnsw_threshold {
        if let Some(hnsw) = &inner.hnsw {
            let query_point = EmbeddingPoint(query_vec.to_vec());
            let mut search = instant_distance::Search::default();
            let results: Vec<(i64, f32)> = hnsw
                .search(&query_point, &mut search)
                .map(|item| {
                    let dist = item.distance;
                    let id = *item.value;
                    let sim = (1.0_f32 - dist * dist / 2.0).clamp(0.0, 1.0);
                    (id, sim)
                })
                .filter(|(id, _)| unverified(id))
                .take(pool_size)
                .collect();
            if !results.is_empty() {
                pool.extend(results);
                return pool;
            }
        }
    }

    let mut rest: Vec<(i64, f32)> = inner
        .embedding_cache
        .iter()
        .filter(|(id, _)| unverified(id))
        .map(|(id, emb)| (*id, cosine_similarity(query_vec, emb)))
        .collect();
    rest.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    rest.truncate(pool_size);
    pool.extend(rest);
    pool
}

/// One ranked search candidate before trial-cap selection.
struct Scored {
    final_score: f32,
    sim: f32,
    trust: f32,
    /// No attributed feedback yet (`reinforcement_count + failure_count == 0`).
    unverified: bool,
    contradiction_group: Option<String>,
    memory: Memory,
}

/// Apply the bounded-trial exposure rule to score-sorted candidates.
///
/// Evidence-bearing memories compete on score once they clear `verified_min_cos` (default
/// 0.0: no floor). Unverified ones (eligible when `sim >= trial_min_cos`) get at most
/// `trial_cap` slots, so feedback can reach them
/// without untested text crowding out verified memories. With no verified candidates
/// the cap alone bounds exposure; with `k == 1` and a verified candidate the verified
/// one always wins. Output is sorted by score, descending.
fn select_with_trial_cap(items: Vec<Scored>, k: usize, cfg: &ScoringConfig) -> Vec<Scored> {
    let (mut verified, mut trial): (Vec<Scored>, Vec<Scored>) =
        items.into_iter().partition(|item| !item.unverified);
    trial.retain(|item| item.sim >= cfg.trial_min_cos);
    verified.retain(|item| item.sim >= cfg.verified_min_cos);

    let max_trial = cfg.trial_cap.min(k);
    let n_trial = if verified.is_empty() {
        trial.len().min(max_trial)
    } else if k < 2 {
        0
    } else {
        trial.len().min(max_trial).min(k - 1)
    };
    let n_verified = verified.len().min(k.saturating_sub(n_trial));
    verified.truncate(n_verified);
    trial.truncate(n_trial);

    let mut out = verified;
    out.extend(trial);
    out.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

/// Decayed posterior trust, regressing toward 0.5 with days since the last feedback.
fn memory_trust(successes: i64, failures: i64, feedback_ts: i64, now: i64, decay_rate: f32) -> f32 {
    let days = ((now - feedback_ts).max(0) as f32) / 86_400.0;
    decayed_trust(posterior_trust(successes, failures), days, decay_rate)
}

/// Offer label for a memory; see `Memory::exposure`.
fn exposure_label(state: &str, successes: i64, failures: i64) -> &'static str {
    if !is_live_state(state) {
        "hidden"
    } else if successes.saturating_add(failures) <= 0 {
        "trial"
    } else if meets_follow(successes, failures) {
        "follow"
    } else {
        "hint"
    }
}

/// State after feedback: live rows follow their counts, hidden rows stay put.
fn next_state(current: &str, successes: i64, failures: i64) -> String {
    if is_live_state(current) {
        derive_live_state(successes, failures).to_string()
    } else {
        current.to_string()
    }
}

/// Columns shared by the listing queries (`all`, `sample_namespace`, `search_plain`).
struct ListingRow {
    id: i64,
    content: String,
    created_at: i64,
    successes: i64,
    state: String,
    failures: i64,
    last_accessed_at: Option<i64>,
}

fn listing_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ListingRow> {
    Ok(ListingRow {
        id: row.get(0)?,
        content: row.get(1)?,
        created_at: row.get(2)?,
        successes: row.get(3)?,
        state: row.get(4)?,
        failures: row.get(5)?,
        last_accessed_at: row.get(6)?,
    })
}

fn listing_memory(row: ListingRow, now: i64, decay_rate: f32) -> Memory {
    let trust = memory_trust(
        row.successes,
        row.failures,
        row.last_accessed_at.unwrap_or(row.created_at),
        now,
        decay_rate,
    );
    let exposure = exposure_label(&row.state, row.successes, row.failures);
    Memory {
        id: row.id,
        content: row.content,
        score: 1.0,
        trust,
        trust_variance: 0.0,
        trust_samples: 0,
        trust_last_session_id: None,
        effective_weight: 0.5,
        reinforcement_count: row.successes,
        ingestion_driver: "Unknown".to_string(),
        uncertainty: trust_uncertainty(row.successes, row.failures),
        state: row.state,
        exposure: exposure.to_string(),
        created_at: row.created_at,
        last_used_at: None,
    }
}

impl Store {
    fn write_lock(&self) -> Result<std::sync::RwLockWriteGuard<'_, StoreInner>> {
        self.inner.write().map_err(|_| MemoireError::LockPoisoned)
    }

    fn read_lock(&self) -> Result<std::sync::RwLockReadGuard<'_, StoreInner>> {
        self.inner.read().map_err(|_| MemoireError::LockPoisoned)
    }

    fn prepare_search_cache(&self, conn: &Connection) -> Result<bool> {
        {
            let inner = self.read_lock()?;
            if inner.embedding_cache_loaded {
                if inner.embedding_cache.is_empty() {
                    return Ok(false);
                }
                let hnsw_ready = inner.embedding_cache.len() < self.config.hnsw_threshold
                    || (!inner.hnsw_dirty && inner.hnsw.is_some());
                if hnsw_ready {
                    return Ok(true);
                }
            }
        }

        let mut inner = self.write_lock()?;
        let ns = self.namespace.clone();
        inner.ensure_embedding_cache(conn, &ns)?;
        if inner.embedding_cache.is_empty() {
            return Ok(false);
        }
        if inner.embedding_cache.len() >= self.config.hnsw_threshold
            && (inner.hnsw_dirty || inner.hnsw.is_none())
        {
            rebuild_hnsw(&mut inner);
        }
        Ok(true)
    }

    fn pooled_file_manager(path: &str) -> SqliteConnectionManager {
        SqliteConnectionManager::file(path).with_init(|conn| {
            conn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA busy_timeout=5000;
                 PRAGMA foreign_keys=ON;",
            )
        })
    }

    fn pooled_memory_manager() -> SqliteConnectionManager {
        SqliteConnectionManager::memory().with_init(|conn| {
            conn.execute_batch(
                "PRAGMA busy_timeout=5000;
                 PRAGMA foreign_keys=ON;",
            )
        })
    }

    pub fn open(path: &str) -> Result<Self> {
        Self::open_ns(path, "default")
    }

    /// Open a store scoped to `namespace`. All reads/writes are isolated to that namespace.
    /// Multiple agents sharing one SQLite file use different namespaces for hard isolation.
    pub fn open_ns(path: &str, namespace: &str) -> Result<Self> {
        Self::open_ns_with_config(path, namespace, ScoringConfig::default())
    }

    pub fn open_with_config(path: &str, config: ScoringConfig) -> Result<Self> {
        Self::open_ns_with_config(path, "default", config)
    }

    pub fn open_ns_with_config(path: &str, namespace: &str, config: ScoringConfig) -> Result<Self> {
        let manager = Self::pooled_file_manager(path);
        // min_idle(1): do not open 20 connections concurrently at startup; the first
        // connection converts a fresh file to WAL on its own, the rest are created lazily.
        let pool = Pool::builder()
            .max_size(20)
            .min_idle(Some(1))
            .build(manager)?;
        let conn = pool.get()?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if path != ":memory:" {
                if let Ok(metadata) = std::fs::metadata(path) {
                    let mut perms = metadata.permissions();
                    if perms.mode() & 0o777 != 0o600 {
                        perms.set_mode(0o600);
                        let _ = std::fs::set_permissions(path, perms);
                    }
                }
            }
        }

        conn.execute_batch(SCHEMA)?;
        Self::ensure_quality_columns(&conn)?;
        let current_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if current_version < 2 {
            let needs_v2_columns = !Self::memories_has_columns(
                &conn,
                &[
                    "trust_variance",
                    "trust_samples",
                    "trust_last_session_id",
                    "ingestion_driver",
                ],
            )?;
            if needs_v2_columns {
                conn.execute_batch(include_str!("../migrations/v2.sql"))?;
            } else {
                conn.execute_batch("PRAGMA user_version = 2;")?;
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS prototype_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        namespace TEXT NOT NULL DEFAULT 'default',
                        created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
                        reinforcements_at_update INTEGER NOT NULL,
                        actionability_centroid BLOB NOT NULL,
                        consequence_centroid BLOB NOT NULL,
                        reusability_centroid BLOB NOT NULL
                    );",
                )?;
            }
        }
        Self::migrate_to_v3(&conn)?;
        let fingerprints = load_fingerprints_from_db(&conn, namespace)?;
        log::debug!(
            "SQLite store opened at {path} ns={namespace}; {} fingerprints loaded (embeddings deferred)",
            fingerprints.len()
        );
        Ok(Self {
            pool,
            inner: Arc::new(RwLock::new(StoreInner {
                embedding_cache: Vec::new(),
                embedding_cache_loaded: false,
                hidden: HashSet::new(),
                verified: HashSet::new(),
                fingerprints,
                hnsw: None,
                hnsw_dirty: false,
            })),
            config,
            namespace: namespace.to_string(),
            nli_checker: NliChecker::new(),
        })
    }

    pub fn in_memory() -> Result<Self> {
        Self::in_memory_with_config(ScoringConfig::default())
    }

    pub fn in_memory_ns(namespace: &str) -> Result<Self> {
        Self::in_memory_ns_with_config(namespace, ScoringConfig::default())
    }

    pub fn in_memory_with_config(config: ScoringConfig) -> Result<Self> {
        Self::in_memory_ns_with_config("default", config)
    }

    pub fn in_memory_ns_with_config(namespace: &str, config: ScoringConfig) -> Result<Self> {
        let manager = Self::pooled_memory_manager();
        let pool = Pool::builder().max_size(1).build(manager)?;
        let conn = pool.get()?;
        conn.execute_batch(SCHEMA)?;
        Self::ensure_quality_columns(&conn)?;
        let current_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if current_version < 2 {
            let needs_v2_columns = !Self::memories_has_columns(
                &conn,
                &[
                    "trust_variance",
                    "trust_samples",
                    "trust_last_session_id",
                    "ingestion_driver",
                ],
            )?;
            if needs_v2_columns {
                conn.execute_batch(include_str!("../migrations/v2.sql"))?;
            } else {
                conn.execute_batch("PRAGMA user_version = 2;")?;
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS prototype_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        namespace TEXT NOT NULL DEFAULT 'default',
                        created_at INTEGER NOT NULL DEFAULT (strftime('%s','now')),
                        reinforcements_at_update INTEGER NOT NULL,
                        actionability_centroid BLOB NOT NULL,
                        consequence_centroid BLOB NOT NULL,
                        reusability_centroid BLOB NOT NULL
                    );",
                )?;
            }
        }
        Self::migrate_to_v3(&conn)?;
        drop(conn);
        Ok(Self {
            pool,
            inner: Arc::new(RwLock::new(StoreInner {
                embedding_cache: Vec::new(),
                // In-memory store starts empty â€” no deferred load needed.
                embedding_cache_loaded: true,
                hidden: HashSet::new(),
                verified: HashSet::new(),
                fingerprints: HashSet::new(),
                hnsw: None,
                hnsw_dirty: false,
            })),
            config,
            namespace: namespace.to_string(),
            nli_checker: NliChecker::new(),
        })
    }

    pub fn insert(&self, content: &str, embedding: &[f32]) -> Result<i64> {
        let meta = QualityMeta::default_active(content);
        self.insert_with_quality(content, embedding, &meta)
    }

    /// Text-attributed reinforcement.
    ///
    /// A success counts only when the agent output demonstrably used the memory:
    /// the memory's distinctive tokens are contained in the output at
    /// `config.containment_threshold` (asymmetric containment, so long outputs are not
    /// penalised), or `output_embedding` has cosine >= 0.75 with the memory. For
    /// callers that know which memories the agent used, see [`Self::reinforce_declared`].
    ///
    /// Returns `Ok(false)` when the task failed, the id is unknown to this namespace,
    /// the memory is quarantined, attribution fails, or `session_id` equals the
    /// memory's last reinforcing session (one vote per memory per session).
    pub fn reinforce_if_used(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
        output_embedding: Option<&[f32]>,
        session_id: Option<&str>,
    ) -> Result<bool> {
        if !task_succeeded {
            return Ok(false);
        }

        let row = {
            let conn = self.pool.get()?;
            conn.query_row(
                "SELECT content, embedding FROM memories WHERE id = ?1 AND namespace = ?2",
                params![memory_id, self.namespace],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?
        };
        let Some((content, embedding_blob)) = row else {
            return Ok(false);
        };

        let text_attributed =
            token_containment(&content, agent_output) >= self.config.containment_threshold;
        let cosine_attributed = match (output_embedding, blob_to_vec(&embedding_blob)) {
            (Some(out), Some(mem)) => cosine_similarity(out, &mem) >= COSINE_ATTRIBUTION_THRESHOLD,
            _ => false,
        };
        if !(text_attributed || cosine_attributed) {
            return Ok(false);
        }
        self.apply_success(memory_id, session_id)
    }

    /// Reinforcement with no text gate: the caller asserts that the agent declared it
    /// used this memory and that the task succeeded. Same namespace, quarantine and
    /// one-vote-per-session rules as [`Self::reinforce_if_used`].
    pub fn reinforce_declared(&self, memory_id: i64, session_id: Option<&str>) -> Result<bool> {
        self.apply_success(memory_id, session_id)
    }

    /// Record one attributed success in a single immediate transaction: the read of
    /// the counts, the state transition and the write cannot interleave with another
    /// writer.
    fn apply_success(&self, memory_id: i64, session_id: Option<&str>) -> Result<bool> {
        let mut conn = self.pool.get()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx
            .query_row(
                "SELECT reinforcement_count, failure_count, store_state, trust_last_session_id
                 FROM memories
                 WHERE id = ?1 AND namespace = ?2 AND archived = 0",
                params![memory_id, self.namespace],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                },
            )
            .optional()?;
        let Some((successes, failures, state, last_session)) = row else {
            return Ok(false);
        };
        if state == "quarantined" {
            return Ok(false);
        }
        if let (Some(sid), Some(last)) = (session_id, last_session.as_deref()) {
            if sid == last {
                return Ok(false);
            }
        }

        let successes = successes.saturating_add(1);
        let new_state = next_state(&state, successes, failures);
        tx.execute(
            "UPDATE memories
             SET reinforcement_count = ?1,
                 store_state = ?2,
                 trust_samples = trust_samples + 1,
                 trust_last_session_id = ?3,
                 last_accessed_at = ?4
             WHERE id = ?5 AND namespace = ?6",
            params![
                successes,
                new_state,
                session_id,
                now_ts(),
                memory_id,
                self.namespace
            ],
        )?;
        tx.commit()?;

        let mut inner = self.write_lock()?;
        if inner.embedding_cache_loaded {
            inner.verified.insert(memory_id);
            if is_live_state(&state) != is_live_state(&new_state) {
                inner.set_hidden(memory_id, !is_live_state(&new_state));
            }
        }
        Ok(true)
    }

    /// Record one attributed failure for each id that exists in this namespace and is
    /// not already quarantined. Unknown or foreign ids are skipped, not errors; every
    /// database error is returned. All ids are processed in one immediate transaction.
    ///
    /// Only the failure evidence changes: ingest-time `importance_base`, `confidence`
    /// and the success count are left untouched. `failure_severity` currently gates
    /// only (`> 0` counts one failure, `0` is a no-op); fractional evidence weights
    /// would need REAL counters and are deferred.
    pub fn penalize_if_used(
        &self,
        memory_ids: &[i64],
        failure_severity: f32,
    ) -> Result<Vec<PenaltyOutcome>> {
        if memory_ids.is_empty() {
            return Ok(vec![]);
        }
        let severity = failure_severity.clamp(0.0, 1.0);
        let mut conn = self.pool.get()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let now = now_ts();
        let mut outcomes = Vec::with_capacity(memory_ids.len());
        let mut visibility_changes: Vec<(i64, bool)> = Vec::new();
        let mut penalized_ids: Vec<i64> = Vec::new();
        for &id in memory_ids {
            let row = tx
                .query_row(
                    "SELECT reinforcement_count, failure_count, store_state
                     FROM memories
                     WHERE id = ?1 AND namespace = ?2 AND archived = 0",
                    params![id, self.namespace],
                    |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, String>(2)?,
                        ))
                    },
                )
                .optional()?;
            let Some((successes, failures, state)) = row else {
                continue;
            };
            if state == "quarantined" {
                continue;
            }

            let trust_before = posterior_trust(successes, failures);
            if severity <= 0.0 {
                outcomes.push(PenaltyOutcome {
                    id,
                    trust_before,
                    trust_after: trust_before,
                    uncertainty_after: trust_uncertainty(successes, failures),
                });
                continue;
            }

            let failures = failures.saturating_add(1);
            let new_state = next_state(&state, successes, failures);
            tx.execute(
                "UPDATE memories
                 SET failure_count = ?1, store_state = ?2, last_accessed_at = ?3
                 WHERE id = ?4 AND namespace = ?5",
                params![failures, new_state, now, id, self.namespace],
            )?;
            penalized_ids.push(id);
            if is_live_state(&state) != is_live_state(&new_state) {
                visibility_changes.push((id, !is_live_state(&new_state)));
            }
            outcomes.push(PenaltyOutcome {
                id,
                trust_before,
                trust_after: posterior_trust(successes, failures),
                uncertainty_after: trust_uncertainty(successes, failures),
            });
        }
        tx.commit()?;

        if !penalized_ids.is_empty() {
            let mut inner = self.write_lock()?;
            if inner.embedding_cache_loaded {
                inner.verified.extend(penalized_ids);
                for (id, hidden) in visibility_changes {
                    inner.set_hidden(id, hidden);
                }
            }
        }
        Ok(outcomes)
    }

    /// Operator action: return a quarantined memory to service.
    ///
    /// Clears its failure evidence (the operator has judged it spurious, and keeping it
    /// would re-quarantine the memory on the next failure) and keeps its successes; the
    /// state is re-derived from the remaining counts. Returns `Ok(false)` if the id is
    /// not a quarantined memory of this namespace.
    pub fn unquarantine(&self, id: i64) -> Result<bool> {
        let mut conn = self.pool.get()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx
            .query_row(
                "SELECT reinforcement_count FROM memories
                 WHERE id = ?1 AND namespace = ?2 AND archived = 0
                   AND store_state = 'quarantined'",
                params![id, self.namespace],
                |r| r.get::<_, i64>(0),
            )
            .optional()?;
        let Some(successes) = row else {
            return Ok(false);
        };
        let new_state = derive_live_state(successes, 0);
        tx.execute(
            "UPDATE memories
             SET failure_count = 0, store_state = ?1, last_accessed_at = ?2
             WHERE id = ?3 AND namespace = ?4",
            params![new_state, now_ts(), id, self.namespace],
        )?;
        tx.commit()?;

        let mut inner = self.write_lock()?;
        if inner.embedding_cache_loaded {
            inner.set_hidden(id, false);
            if successes == 0 {
                inner.verified.remove(&id);
            }
        }
        Ok(true)
    }

    /// Promote a hidden `shadow` / `probationary` memory to the live set (state
    /// re-derived from its counts). This is the primitive behind corroboration- or
    /// operator-driven promotion; ingest does not call it yet. Returns `Ok(false)` if
    /// the id is not such a memory of this namespace.
    pub fn promote_shadow(&self, id: i64) -> Result<bool> {
        let mut conn = self.pool.get()?;
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let row = tx
            .query_row(
                "SELECT reinforcement_count, failure_count FROM memories
                 WHERE id = ?1 AND namespace = ?2 AND archived = 0
                   AND store_state IN ('shadow', 'probationary')",
                params![id, self.namespace],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)),
            )
            .optional()?;
        let Some((successes, failures)) = row else {
            return Ok(false);
        };
        let new_state = derive_live_state(successes, failures);
        tx.execute(
            "UPDATE memories SET store_state = ?1 WHERE id = ?2 AND namespace = ?3",
            params![new_state, id, self.namespace],
        )?;
        tx.commit()?;

        let mut inner = self.write_lock()?;
        if inner.embedding_cache_loaded {
            inner.set_hidden(id, !is_live_state(new_state));
        }
        Ok(true)
    }

    pub fn insert_with_quality(
        &self,
        content: &str,
        embedding: &[f32],
        quality: &QualityMeta,
    ) -> Result<i64> {
        assert!(
            quality.store_state == "active"
                || quality.store_state == "trial"
                || quality.store_state == "shadow"
                || quality.store_state == "quarantined"
                || quality.store_state == "rejected"
                || quality.store_state == "probationary",
            "invalid store_state: {}",
            quality.store_state
        );
        let blob = vec_to_blob(embedding);
        // Clamp trust-related fields to valid ranges before writing so CHECK
        // constraints are never triggered by normal application logic.
        let confidence = quality.confidence.clamp(0.0_f32, 1.0_f32);
        let novelty = quality.novelty.clamp(0.0_f32, 1.0_f32);
        // Live rows take the state their feedback counts imply: an admission decision is
        // not evidence, so e.g. a caller-supplied "active" with no reinforcement is `trial`.
        let store_state = next_state(&quality.store_state, quality.reinforcement_count, 0);
        let now_ts_val = now_ts();
        let conn = self.pool.get()?;
        conn.execute(
            "INSERT INTO memories (
                namespace, content, embedding,
                importance_base, confidence, novelty,
                actionability, evidence, consequence, reusability,
                source_kind, store_state, reinforcement_count,
                last_accessed_at, superseded_by, contradiction_group,
                archived, effective_weight, fingerprint, claim_key, claim_value,
                trust_ema, trust_variance, trust_samples, trust_last_session_id,
                ingestion_driver, last_used_at
            ) VALUES (
                ?1, ?2, ?3,
                ?4, ?5, ?6,
                ?7, ?8, ?9, ?10,
                ?11, ?12, ?13,
                ?14, ?15, ?16,
                ?17, ?18, ?19, ?20, ?21,
                ?22, ?23, ?24, ?25, ?26, ?27
            )",
            params![
                self.namespace,
                content,
                blob,
                quality.importance_base,
                confidence,
                novelty,
                quality.actionability,
                quality.evidence,
                quality.consequence,
                quality.reusability,
                quality.source_kind,
                store_state,
                quality.reinforcement_count,
                quality.last_accessed_at,
                quality.superseded_by,
                quality.contradiction_group,
                quality.archived,
                quality.effective_weight,
                quality.fingerprint,
                quality.claim_key,
                quality.claim_value,
                None::<f32>,
                0.0,
                0,
                None::<String>,
                None::<String>,
                now_ts_val,
            ],
        )?;
        let id = conn.last_insert_rowid();
        if store_state != "rejected" && quality.archived == 0 {
            let mut inner = self.write_lock()?;
            // Only update the in-memory cache if it has already been loaded.
            // If not yet loaded, the lazy load will pick this row up from SQLite.
            if inner.embedding_cache_loaded {
                inner.embedding_cache.push((id, embedding.to_vec()));
                inner.set_hidden(id, !is_live_state(&store_state));
                if quality.reinforcement_count > 0 {
                    inner.verified.insert(id);
                }
                inner.hnsw_dirty = true;
            }
            inner.fingerprints.insert(quality.fingerprint.clone());
        }
        Ok(id)
    }

    /// Install an ONNX NLI model. Previously a silent no-op (the checker was dropped).
    pub fn set_nli_checker(&mut self, checker: OnnxNliChecker) -> Result<()> {
        self.nli_checker = NliChecker::with_onnx(checker);
        Ok(())
    }

    pub fn max_similarity(&self, query_vec: &[f32]) -> Result<f32> {
        let conn = self.pool.get()?;
        if !self.prepare_search_cache(&conn)? {
            return Ok(0.0);
        }
        let inner = self.read_lock()?;
        let max_sim = inner
            .embedding_cache
            .iter()
            .map(|(_, emb)| cosine_similarity(query_vec, emb))
            .fold(0.0_f32, f32::max);
        Ok(max_sim)
    }

    pub fn fingerprint_exists(&self, fp: &str) -> Result<bool> {
        let inner = self.read_lock()?;
        Ok(inner.fingerprints.contains(fp))
    }

    /// Search live memories by similarity.
    ///
    /// Default view: `trial` and `active` rows only, with at most
    /// `config.trial_cap` unverified (`s + f == 0`) rows. With `include_hidden = true`
    /// the hidden states (`shadow`, `probationary`, `quarantined`) are included and the
    /// trial cap is not applied (operator / debugging view).
    ///
    /// Ranking is `0.75 * cos + 0.20 * trust + 0.05 * recency`, where `trust` is the
    /// decayed Beta posterior of the memory's attributed feedback.
    pub fn search(
        &self,
        query_vec: &[f32],
        top_k: usize,
        include_hidden: bool,
    ) -> Result<Vec<Memory>> {
        self.search_filtered(query_vec, top_k, include_hidden, None)
    }

    /// [`Store::search`] with an optional `created_at >= min_created_at` filter applied
    /// *before* the trial cap and before `last_used_at` is written, so rows the filter
    /// removes neither occupy a trial slot nor have their usage timestamp bumped.
    fn search_filtered(
        &self,
        query_vec: &[f32],
        top_k: usize,
        include_hidden: bool,
        min_created_at: Option<i64>,
    ) -> Result<Vec<Memory>> {
        if top_k == 0 {
            return Ok(vec![]);
        }

        let conn = self.pool.get()?;
        if !self.prepare_search_cache(&conn)? {
            return Ok(vec![]);
        }

        let candidates = {
            let inner = self.read_lock()?;
            search_candidates_read(&inner, query_vec, top_k, &self.config, include_hidden)
        };
        let ns = self.namespace.clone();

        let now: i64 =
            conn.query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                r.get(0)
            })?;

        let state_filter = if include_hidden {
            "store_state IN ('active', 'trial', 'shadow', 'probationary', 'quarantined')"
        } else {
            "store_state IN ('active', 'trial')"
        };
        let query_str = format!(
            "SELECT content, created_at,
                    importance_base, reinforcement_count,
                    store_state, contradiction_group,
                    failure_count, last_accessed_at, last_used_at
             FROM memories
             WHERE id = ?1
               AND namespace = ?2
               AND archived = 0
               AND superseded_by IS NULL
               AND {}",
            state_filter
        );

        let mut scored: Vec<Scored> = Vec::new();
        for (id, sim) in &candidates {
            let row = conn.query_row(&query_str, params![id, ns], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, f32>(2)?,
                    r.get::<_, i64>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, Option<String>>(5)?,
                    r.get::<_, i64>(6)?,
                    r.get::<_, Option<i64>>(7)?,
                    r.get::<_, Option<i64>>(8)?,
                ))
            });
            let (
                content,
                created_at,
                importance_base,
                reinforcement_count,
                store_state,
                contradiction_group,
                failure_count,
                last_accessed_at,
                last_used_at,
            ) = match row {
                Ok(v) => v,
                Err(_) => continue,
            };

            if min_created_at.is_some_and(|min| created_at < min) {
                continue;
            }

            let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
            let weight = effective_weight(importance_base, age_days, reinforcement_count);
            let trust = memory_trust(
                reinforcement_count,
                failure_count,
                last_accessed_at.unwrap_or(created_at),
                now,
                self.config.decay_rate,
            );
            let final_score = 0.75 * sim + 0.20 * trust + 0.05 * recency_bonus(created_at, now);
            let exposure = exposure_label(&store_state, reinforcement_count, failure_count);

            scored.push(Scored {
                final_score,
                sim: sim.clamp(0.0, 1.0),
                trust,
                unverified: reinforcement_count.saturating_add(failure_count) <= 0,
                contradiction_group,
                memory: Memory {
                    id: *id,
                    content,
                    score: final_score.clamp(0.0, 1.0),
                    trust,
                    trust_variance: 0.0,
                    trust_samples: 0,
                    trust_last_session_id: None,
                    effective_weight: weight,
                    reinforcement_count,
                    ingestion_driver: "Unknown".to_string(),
                    uncertainty: trust_uncertainty(reinforcement_count, failure_count),
                    state: store_state,
                    exposure: exposure.to_string(),
                    created_at,
                    last_used_at,
                },
            });
        }

        scored.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut group_winner: HashMap<String, usize> = HashMap::new();
        for (idx, item) in scored.iter().enumerate() {
            if let Some(cg) = &item.contradiction_group {
                let entry = group_winner.entry(cg.clone()).or_insert(idx);
                if item.trust > scored[*entry].trust {
                    *entry = idx;
                }
            }
        }
        let weak_groups: HashSet<String> = group_winner
            .iter()
            .filter(|(_, &idx)| scored[idx].trust < 0.4)
            .map(|(cg, _)| cg.clone())
            .collect();

        let eligible: Vec<Scored> = scored
            .into_iter()
            .enumerate()
            .filter(|(idx, item)| match &item.contradiction_group {
                Some(cg) => !weak_groups.contains(cg) && group_winner.get(cg) == Some(idx),
                None => true,
            })
            .map(|(_, item)| item)
            .collect();

        let selected: Vec<Scored> = if include_hidden {
            eligible.into_iter().take(top_k).collect()
        } else {
            select_with_trial_cap(eligible, top_k, &self.config)
        };
        let result: Vec<Memory> = selected.into_iter().map(|item| item.memory).collect();

        // Update last_used_at for all returned memories in a single batch
        if !result.is_empty() {
            let ids: Vec<String> = result.iter().map(|m| m.id.to_string()).collect();
            let placeholders = ids.join(",");
            conn.execute(
                &format!(
                    "UPDATE memories SET last_used_at = ?1
                     WHERE namespace = ?2 AND id IN ({placeholders})"
                ),
                params![now, ns],
            )?;
        }

        Ok(result)
    }

    pub fn search_within_days(
        &self,
        query_vec: &[f32],
        top_k: usize,
        max_age_days: f32,
    ) -> Result<Vec<Memory>> {
        let cutoff_ts = now_ts() - (max_age_days * 86_400.0) as i64;
        // Candidates are the nearest rows overall, so over-fetch to leave room after the
        // age filter; the filter itself runs inside the search, ahead of the trial cap.
        let pool_k = (top_k * 4).max(20);
        let mut pool = self.search_filtered(query_vec, pool_k, false, Some(cutoff_ts))?;
        pool.truncate(top_k);
        Ok(pool)
    }

    /// Corroborate a hidden `shadow` row: if `content` / `embedding` independently
    /// restate one (cosine >= [`NEAR_DUPLICATE_COSINE`], not a contradiction under the
    /// configured `contradiction_mode`), promote that row to the live set and return its
    /// id. The caller then skips inserting a duplicate. Nothing is counted as success
    /// evidence: a restatement is not proof the lesson helps, so `reinforcement_count`
    /// is untouched and the row starts as an unverified `trial`.
    ///
    /// Ids in `exclude` (rows inserted earlier in the same ingest call) are never
    /// promoted, so one call cannot corroborate itself. Exact duplicates are filtered
    /// out earlier by the fingerprint check and never reach this method. Only rows in
    /// state `shadow` qualify, never `quarantined` or `probationary` ones.
    pub fn corroborate_shadow(
        &self,
        content: &str,
        embedding: &[f32],
        exclude: &[i64],
    ) -> Result<Option<i64>> {
        let chosen = {
            let conn = self.pool.get()?;
            if !self.prepare_search_cache(&conn)? {
                return Ok(None);
            }
            let mut near: Vec<(i64, f32, Vec<f32>)> = {
                let inner = self.read_lock()?;
                inner
                    .embedding_cache
                    .iter()
                    .filter(|(id, _)| inner.hidden.contains(id) && !exclude.contains(id))
                    .filter_map(|(id, emb)| {
                        let sim = cosine_similarity(embedding, emb);
                        if sim >= NEAR_DUPLICATE_COSINE {
                            Some((*id, sim, emb.clone()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            near.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut chosen = None;
            for (id, _sim, existing_emb) in near {
                let row = conn
                    .query_row(
                        "SELECT content, store_state, claim_value FROM memories
                         WHERE id = ?1 AND namespace = ?2 AND archived = 0
                           AND superseded_by IS NULL",
                        params![id, self.namespace],
                        |r| {
                            Ok((
                                r.get::<_, String>(0)?,
                                r.get::<_, String>(1)?,
                                r.get::<_, Option<String>>(2)?,
                            ))
                        },
                    )
                    .optional()?;
                let Some((existing_content, state, existing_polarity)) = row else {
                    continue;
                };
                if state != "shadow" {
                    continue;
                }
                if self.contradicts(
                    &existing_content,
                    existing_polarity.as_deref(),
                    &existing_emb,
                    content,
                    embedding,
                ) {
                    continue;
                }
                chosen = Some(id);
                break;
            }
            chosen
        };

        match chosen {
            Some(id) if self.promote_shadow(id)? => Ok(Some(id)),
            _ => Ok(None),
        }
    }

    /// Whether `incoming` contradicts `existing` under the configured mode (used to keep
    /// corroboration from promoting a row that the incoming text actually refutes).
    fn contradicts(
        &self,
        existing: &str,
        existing_polarity: Option<&str>,
        existing_emb: &[f32],
        incoming: &str,
        incoming_emb: &[f32],
    ) -> bool {
        match self.config.contradiction_mode {
            ContradictionMode::Off => false,
            ContradictionMode::Nli => {
                self.nli_checker.check(
                    existing,
                    incoming,
                    existing_emb,
                    incoming_emb,
                    self.config.nli_cosine_threshold,
                ) == NliLabel::Contradiction
            }
            ContradictionMode::Legacy => {
                let existing_pol = existing_polarity
                    .map(Polarity::from_stored)
                    .unwrap_or_else(|| detect_polarity(existing));
                existing_pol.opposes(&detect_polarity(incoming))
            }
        }
    }

    /// Plain vector similarity: exact cosine over every stored (non-archived,
    /// non-superseded) memory of this namespace, ranked by cosine alone. No trust, no
    /// state filter, no trial cap and no `last_used_at` write, so repeated calls leave
    /// the database untouched. This is the "plain vector memory" baseline for
    /// evaluation; returned rows carry `exposure = "plain"`.
    pub fn search_plain(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<Memory>> {
        if top_k == 0 {
            return Ok(vec![]);
        }
        let conn = self.pool.get()?;
        if !self.prepare_search_cache(&conn)? {
            return Ok(vec![]);
        }
        let mut ranked: Vec<(i64, f32)> = {
            let inner = self.read_lock()?;
            inner
                .embedding_cache
                .iter()
                .map(|(id, emb)| (*id, cosine_similarity(query_vec, emb)))
                .collect()
        };
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);

        let now: i64 =
            conn.query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                r.get(0)
            })?;
        let mut out = Vec::with_capacity(ranked.len());
        for (id, sim) in ranked {
            let row = conn
                .query_row(
                    "SELECT id, content, created_at, reinforcement_count, store_state,
                            failure_count, last_accessed_at
                     FROM memories
                     WHERE id = ?1 AND namespace = ?2 AND archived = 0
                       AND superseded_by IS NULL",
                    params![id, self.namespace],
                    listing_row,
                )
                .optional()?;
            if let Some(row) = row {
                let mut memory = listing_memory(row, now, self.config.decay_rate);
                memory.score = sim.clamp(0.0, 1.0);
                memory.exposure = "plain".to_string();
                out.push(memory);
            }
        }
        Ok(out)
    }

    /// NLI-enhanced semantic contradiction detection.
    ///
    /// Behaviour follows `config.contradiction_mode`: `Off` returns immediately,
    /// `Nli` (default) uses `NliChecker`, `Legacy` uses the pure polarity gate.
    /// If an ONNX NLI model is installed (`Memoire::with_nli_model`) its label is
    /// authoritative for pairs above the cosine gate; otherwise `NliChecker` applies
    /// a three-signal lexical ensemble:
    ///   1. Cosine similarity gate (same topic cluster, threshold from config)
    ///   2. Polarity opposition (lexical negation)
    ///   3. Negation asymmetry score (how many negators appear in one but not the other)
    ///
    /// `Legacy` is the original cosine + polarity gate, kept for backward compatibility.
    ///
    /// The loser of a contradiction is hidden in one of two ways. Only a verdict backed by
    /// an installed NLI model archives it (`archived = 1`, deleted by maintenance after a
    /// week). Every heuristic verdict (the lexical fallback and the `Legacy` polarity
    /// gate) `quarantines` the loser instead: hidden from default recall but recoverable
    /// with `unquarantine`, so no heuristic path destroys a memory.
    ///
    /// The winner is decided by feedback evidence (`should_supersede`): the side with
    /// more demonstrated value wins and a tie goes to the newer memory.
    pub fn resolve_contradictions_for_id(&self, id: i64) -> Result<()> {
        if self.config.contradiction_mode == ContradictionMode::Off {
            return Ok(());
        }
        let conn = self.pool.get()?;
        let mut inner = self.write_lock()?;
        let ns = self.namespace.clone();

        let target_row = conn.query_row(
            "SELECT content, embedding, created_at, claim_value,
                    reinforcement_count, failure_count
             FROM memories WHERE id = ?1 AND namespace = ?2 AND archived = 0",
            params![id, ns],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, i64>(4)?,
                    r.get::<_, i64>(5)?,
                ))
            },
        );

        let (new_content, new_blob, new_created_at, stored_polarity, new_rc, new_fc) =
            match target_row {
                Ok(v) => v,
                Err(_) => return Ok(()),
            };

        let new_emb = match blob_to_vec(&new_blob) {
            Some(v) => v,
            None => return Ok(()),
        };

        let new_polarity_str =
            stored_polarity.unwrap_or_else(|| detect_polarity(&new_content).as_str().to_string());
        if new_polarity_str == "neutral" {
            return Ok(());
        }

        let now: i64 =
            conn.query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                r.get(0)
            })?;

        let mut stmt = conn.prepare(
            "SELECT id, content, embedding, claim_value
             FROM memories
             WHERE id != ?1
               AND namespace = ?2
               AND archived = 0
               AND superseded_by IS NULL
               AND store_state IN ('active', 'trial', 'shadow')",
        )?;

        // Build NLI checker once (zero-cost: stateless struct).
        let use_nli = self.config.contradiction_mode == ContradictionMode::Nli;
        let nli_threshold = self.config.nli_cosine_threshold;
        let nli = self.nli_checker.clone();

        // (other id, whether the verdict is only a lexical heuristic)
        let mut contradictions: Vec<(i64, bool)> = Vec::new();
        for row in stmt.query_map(params![id, ns], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
                r.get::<_, Option<String>>(3)?,
            ))
        })? {
            let (other_id, other_content, other_blob, other_polarity_opt) = match row {
                Ok(v) => v,
                Err(_) => continue,
            };

            let other_emb = match blob_to_vec(&other_blob) {
                Some(v) => v,
                None => continue,
            };

            let (is_contradiction, heuristic_only) = if use_nli {
                // NLI path: an installed model is authoritative; otherwise the lexical
                // ensemble decides and its verdict is marked heuristic.
                let (label, source) = nli.check_with_source(
                    &new_content,
                    &other_content,
                    &new_emb,
                    &other_emb,
                    nli_threshold,
                );
                (
                    label == NliLabel::Contradiction,
                    source == NliSource::Lexical,
                )
            } else {
                // Legacy path: cosine + polarity only. A polarity flip is a guess, so
                // the loser is quarantined (recoverable), never archived.
                let sim = cosine_similarity(&new_emb, &other_emb);
                if sim < 0.80 {
                    continue;
                }
                let other_polarity_str = other_polarity_opt
                    .as_deref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| detect_polarity(&other_content).as_str().to_string());
                let new_pol = Polarity::from_stored(&new_polarity_str);
                let other_pol = Polarity::from_stored(&other_polarity_str);
                (new_pol.opposes(&other_pol), true)
            };

            if is_contradiction {
                contradictions.push((other_id, heuristic_only));
            }
        }

        if contradictions.is_empty() {
            return Ok(());
        }

        drop(stmt);

        for (other_id, heuristic_only) in contradictions {
            let existing = conn
                .query_row(
                    "SELECT reinforcement_count, failure_count, created_at FROM memories
                     WHERE id = ?1 AND namespace = ?2",
                    params![other_id, ns],
                    |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, i64>(1)?,
                            r.get::<_, i64>(2)?,
                        ))
                    },
                )
                .ok();
            let existing_gravity = existing
                .map(|(rc, fc, created)| {
                    compute_gravity_radius(
                        posterior_trust(rc, fc) as f64,
                        rc,
                        ((now - created).max(0) as f64) / 86_400.0,
                    )
                })
                .unwrap_or(0.0);
            let existing_created = existing.map(|(_, _, created)| created).unwrap_or(0);
            let incoming_gravity = compute_gravity_radius(
                posterior_trust(new_rc, new_fc) as f64,
                new_rc,
                ((now - new_created_at).max(0) as f64) / 86_400.0,
            );
            let should_replace = should_supersede(
                incoming_gravity,
                existing_gravity,
                new_created_at >= existing_created,
            );
            let (winner_id, loser_id) = if should_replace {
                (id, other_id)
            } else {
                (other_id, id)
            };

            if heuristic_only {
                // Lexical verdicts are guesses: hide the loser but keep it recoverable.
                conn.execute(
                    "UPDATE memories SET store_state = 'quarantined'
                     WHERE id = ?1 AND namespace = ?2",
                    params![loser_id, ns],
                )?;
                if inner.embedding_cache_loaded {
                    inner.set_hidden(loser_id, true);
                }
                log::debug!(
                    "contradiction (lexical): winner={winner_id} loser={loser_id} quarantined"
                );
                continue;
            }

            let group = winner_id.to_string();
            conn.execute(
                "UPDATE memories SET superseded_by = ?1, archived = 1, contradiction_group = ?2
                 WHERE id = ?3 AND namespace = ?4",
                params![winner_id, group, loser_id, ns],
            )?;
            // Do NOT set contradiction_group on the winner. The loser already points to the
            // winner via superseded_by. Setting contradiction_group on the winner would put
            // it into the weak-group deduplication filter in search(), which suppresses
            // memories whose best-survivor trust < 0.40, incorrectly hiding the winner.
            inner.remove_from_cache(loser_id);
            log::debug!(
                "contradiction resolved ({}): winner={winner_id} loser={loser_id} group={group}",
                if use_nli { "nli" } else { "polarity" }
            );
        }

        Ok(())
    }

    /// Archive superseded rows and prune stale ones. Strictly scoped to
    /// `self.namespace`: another namespace sharing the same SQLite file is never touched.
    pub fn maintenance_pass(&self) -> Result<()> {
        let conn = self.pool.get()?;
        let mut inner = self.write_lock()?;
        let ns = &self.namespace;

        conn.execute(
            "UPDATE memories
             SET archived = 1
             WHERE namespace = ?1
               AND superseded_by IS NOT NULL
               AND archived = 0",
            params![ns],
        )?;

        let mut stmt = conn.prepare(
            "SELECT id, fingerprint FROM memories
             WHERE namespace = ?1
               AND archived = 1
               AND (strftime('%s', 'now') - created_at) > 7 * 86400
             UNION ALL
             SELECT id, fingerprint FROM memories
             WHERE namespace = ?1
               AND archived = 0
               AND superseded_by IS NULL
               AND reinforcement_count = 0
               AND importance_base < 0.12
               AND (strftime('%s', 'now') - created_at) > 30 * 86400",
        )?;
        let to_delete: Vec<(i64, String)> = stmt
            .query_map(params![ns], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);

        for (del_id, fp) in to_delete {
            conn.execute(
                "DELETE FROM memories WHERE id = ?1 AND namespace = ?2",
                params![del_id, ns],
            )?;
            inner.remove_from_cache(del_id);
            release_fingerprint(&mut inner, &conn, ns, &fp)?;
        }

        Ok(())
    }

    pub fn all(&self) -> Result<Vec<Memory>> {
        let conn = self.pool.get()?;
        let ns = &self.namespace;
        let now: i64 =
            conn.query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                r.get(0)
            })?;

        let mut stmt = conn.prepare(
            "SELECT id, content, created_at, reinforcement_count, store_state,
                    failure_count, last_accessed_at
             FROM memories WHERE namespace = ?1 ORDER BY created_at DESC",
        )?;

        let decay_rate = self.config.decay_rate;
        let rows = stmt
            .query_map(rusqlite::params![ns], listing_row)?
            .filter_map(|r| r.ok())
            .map(|row| listing_memory(row, now, decay_rate))
            .collect();
        Ok(rows)
    }

    /// Sample up to `limit` live (`trial` / `active`) memories at random from this
    /// namespace. Used by the oracle to evaluate semantic consistency before ingestion.
    pub fn sample_namespace(&self, limit: usize) -> Result<Vec<Memory>> {
        let conn = self.pool.get()?;
        let ns = &self.namespace;
        let now = now_ts();
        let decay_rate = self.config.decay_rate;
        let mut stmt = conn.prepare(
            "SELECT id, content, created_at, reinforcement_count, store_state,
                    failure_count, last_accessed_at
             FROM memories
             WHERE namespace = ?1 AND archived = 0 AND superseded_by IS NULL
               AND store_state IN ('active', 'trial')
             ORDER BY RANDOM() LIMIT ?2",
        )?;
        let rows: Vec<Memory> = stmt
            .query_map(params![ns, limit as i64], listing_row)?
            .filter_map(|r| r.ok())
            .map(|row| listing_memory(row, now, decay_rate))
            .collect();
        Ok(rows)
    }

    pub fn forget(&self, id: i64) -> Result<bool> {
        let conn = self.pool.get()?;
        let mut inner = self.write_lock()?;
        let ns = &self.namespace;
        // Scoped to this namespace: an id owned by another namespace is "not found".
        let fp: Option<String> = conn
            .query_row(
                "SELECT fingerprint FROM memories WHERE id = ?1 AND namespace = ?2",
                params![id, ns],
                |r| r.get(0),
            )
            .optional()?;
        let Some(fp) = fp else {
            return Ok(false);
        };
        let rows = conn.execute(
            "DELETE FROM memories WHERE id = ?1 AND namespace = ?2",
            params![id, ns],
        )?;
        if rows > 0 {
            inner.remove_from_cache(id);
            release_fingerprint(&mut inner, &conn, ns, &fp)?;
        }
        Ok(rows > 0)
    }

    pub fn count(&self) -> Result<i64> {
        let conn = self.pool.get()?;
        Ok(conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE namespace = ?1 AND archived = 0",
            rusqlite::params![self.namespace],
            |r| r.get(0),
        )?)
    }

    pub fn clear(&self) -> Result<()> {
        let conn = self.pool.get()?;
        let mut inner = self.write_lock()?;
        conn.execute(
            "DELETE FROM memories WHERE namespace = ?1",
            params![self.namespace],
        )?;
        inner.embedding_cache.clear();
        inner.hidden.clear();
        inner.verified.clear();
        inner.embedding_cache_loaded = true; // empty store is coherent â€” no load needed
        inner.fingerprints.clear();
        inner.hnsw = None;
        inner.hnsw_dirty = false;
        Ok(())
    }

    pub fn export_namespace(&self) -> Result<serde_json::Value> {
        let conn = self.pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT content, trust_ema, reinforcement_count, importance_base, confidence, created_at
             FROM memories
             WHERE namespace = ?1 AND archived = 0 AND store_state != 'quarantined'"
        )?;

        let ns = &self.namespace;
        let memories: Vec<serde_json::Value> = stmt
            .query_map(params![ns], |r| {
                let content: String = r.get(0)?;
                let trust_ema: Option<f32> = r.get(1)?;
                let reinforcement_count: i64 = r.get(2)?;
                let importance_base: f32 = r.get(3)?;
                let confidence: f32 = r.get(4)?;
                let created_at: i64 = r.get(5)?;

                Ok(serde_json::json!({
                    "content": content,
                    "trust_ema": trust_ema,
                    "reinforcement_count": reinforcement_count,
                    "importance_base": importance_base,
                    "confidence": confidence,
                    "created_at": created_at,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Ok(serde_json::json!({
            "memoire_export_version": 1,
            "namespace": ns,
            "exported_at": now,
            "memories": memories,
        }))
    }

    /// Overwrite metadata on freshly imported rows. Every value is clamped here,
    /// at the only write site, so no caller can store out-of-range metadata:
    /// `trust_ema` and `confidence`/`importance_base` to `[0, 1]` (non-finite
    /// `trust_ema` becomes NULL, i.e. "recompute"), `reinforcement_count` to
    /// `[0, IMPORT_MAX_REINFORCEMENT]` and `created_at` to `[0, now]`.
    /// These are sanity bounds, not a trust model.
    pub fn update_imported_metadata(
        &self,
        ids: &[i64],
        trust_ema: Option<f32>,
        reinforcement_count: i64,
        importance_base: f32,
        confidence: f32,
        created_at: Option<i64>,
    ) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let unit = |v: f32| {
            if v.is_finite() {
                v.clamp(0.0, 1.0)
            } else {
                0.5
            }
        };
        let trust_ema = trust_ema
            .filter(|t| t.is_finite())
            .map(|t| t.clamp(0.0, 1.0));
        let reinforcement_count = reinforcement_count.clamp(0, IMPORT_MAX_REINFORCEMENT);
        let importance_base = unit(importance_base);
        let confidence = unit(confidence);
        let created_at = created_at.map(|c| c.clamp(0, now));
        // Live rows take the state their (imported) counts imply; hidden rows stay hidden
        // and imported rows start with no failure evidence.
        let live_state = derive_live_state(reinforcement_count, 0);
        let conn = self.pool.get()?;
        let placeholders = ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        // `COALESCE(?5, created_at)` keeps the existing value when no timestamp was
        // supplied. Scoped to this namespace so ids owned by another tenant are
        // never touched, whatever the caller passes in `ids`.
        let query = format!(
            "UPDATE memories
             SET trust_ema = ?1, reinforcement_count = ?2, importance_base = ?3,
                 confidence = ?4, created_at = COALESCE(?5, created_at),
                 failure_count = 0,
                 store_state = CASE WHEN store_state IN ('trial', 'active')
                                    THEN ?7 ELSE store_state END
             WHERE namespace = ?6 AND id IN ({placeholders})"
        );
        conn.execute(
            &query,
            params![
                trust_ema,
                reinforcement_count,
                importance_base,
                confidence,
                created_at,
                self.namespace,
                live_state
            ],
        )?;
        drop(conn);
        let mut inner = self.write_lock()?;
        if inner.embedding_cache_loaded {
            for &id in ids {
                if reinforcement_count > 0 {
                    inner.verified.insert(id);
                } else {
                    inner.verified.remove(&id);
                }
            }
        }
        Ok(())
    }

    /// Maximal Marginal Relevance reranking.
    ///
    /// Selects up to `top_k` results from `candidates` that maximise relevance
    /// to `query_embedding` while minimising redundancy with already-selected items.
    ///
    /// `lambda = 1.0` â†’ pure relevance (identical to `recall`).
    /// `lambda = 0.0` â†’ pure diversity.
    pub fn mmr_rerank(
        &self,
        mut candidates: Vec<Memory>,
        _query_embedding: &[f32],
        top_k: usize,
        lambda: f32,
    ) -> Vec<Memory> {
        if lambda >= 1.0 || candidates.len() <= top_k {
            candidates.truncate(top_k);
            return candidates;
        }
        if candidates.is_empty() || top_k == 0 {
            return vec![];
        }

        // We need per-candidate embeddings to compute similarity between candidates.
        // Use the score field (which encodes cosine sim to query) as the relevance term,
        // and do a second pass to compute inter-candidate similarity from the in-memory cache.
        let inner = match self.read_lock() {
            Ok(i) => i,
            Err(_) => {
                candidates.truncate(top_k);
                return candidates;
            }
        };

        let get_emb = |id: i64| -> Option<Vec<f32>> {
            inner
                .embedding_cache
                .iter()
                .find(|(cid, _)| *cid == id)
                .map(|(_, e)| e.clone())
        };

        let mut selected: Vec<Memory> = Vec::with_capacity(top_k);

        while selected.len() < top_k && !candidates.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (i, cand) in candidates.iter().enumerate() {
                let relevance = cand.score; // cosine to query, pre-computed
                let max_sim_to_selected = if selected.is_empty() {
                    0.0_f32
                } else {
                    selected
                        .iter()
                        .map(|s| match (get_emb(cand.id), get_emb(s.id)) {
                            (Some(e_c), Some(e_s)) => cosine_similarity(&e_c, &e_s),
                            _ => 0.0,
                        })
                        .fold(f32::NEG_INFINITY, f32::max)
                };

                let mmr_score = lambda * relevance - (1.0 - lambda) * max_sim_to_selected;
                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = i;
                }
            }

            selected.push(candidates.remove(best_idx));
        }

        selected
    }

    /// `user_version` 2 -> 3: re-derive live row states from feedback counts (see
    /// `migrations/v3.sql`). No-op on databases already at version 3.
    fn migrate_to_v3(conn: &Connection) -> Result<()> {
        let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        if version < 3 {
            conn.execute_batch(include_str!("../migrations/v3.sql"))?;
        }
        Ok(())
    }

    fn memories_has_columns(conn: &Connection, names: &[&str]) -> Result<bool> {
        let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
        let existing: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(names.iter().all(|name| existing.iter().any(|c| c == *name)))
    }

    fn ensure_quality_columns(conn: &Connection) -> Result<()> {
        let mut stmt = conn.prepare("PRAGMA table_info(memories)")?;
        let existing: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .filter_map(|r| r.ok())
            .collect();

        let has = |name: &str| existing.iter().any(|c| c == name);
        let mut alter = Vec::new();
        if !has("importance_base") {
            alter
                .push("ALTER TABLE memories ADD COLUMN importance_base REAL NOT NULL DEFAULT 0.5;");
        }
        if !has("confidence") {
            alter.push("ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 0.5;");
        }
        if !has("novelty") {
            alter.push("ALTER TABLE memories ADD COLUMN novelty REAL NOT NULL DEFAULT 0.5;");
        }
        if !has("actionability") {
            alter.push("ALTER TABLE memories ADD COLUMN actionability REAL NOT NULL DEFAULT 0.5;");
        }
        if !has("evidence") {
            alter.push("ALTER TABLE memories ADD COLUMN evidence REAL NOT NULL DEFAULT 0.5;");
        }
        if !has("consequence") {
            alter.push("ALTER TABLE memories ADD COLUMN consequence REAL NOT NULL DEFAULT 0.5;");
        }
        if !has("reusability") {
            alter.push("ALTER TABLE memories ADD COLUMN reusability REAL NOT NULL DEFAULT 0.5;");
        }
        if !has("source_kind") {
            alter
                .push("ALTER TABLE memories ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'agent';");
        }
        if !has("store_state") {
            alter.push(
                "ALTER TABLE memories ADD COLUMN store_state TEXT NOT NULL DEFAULT 'active';",
            );
        }
        if !has("reinforcement_count") {
            alter.push(
                "ALTER TABLE memories ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0;",
            );
        }
        if !has("last_accessed_at") {
            alter.push("ALTER TABLE memories ADD COLUMN last_accessed_at INTEGER;");
        }
        if !has("superseded_by") {
            alter.push("ALTER TABLE memories ADD COLUMN superseded_by INTEGER;");
        }
        if !has("contradiction_group") {
            alter.push("ALTER TABLE memories ADD COLUMN contradiction_group TEXT;");
        }
        if !has("archived") {
            alter.push("ALTER TABLE memories ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;");
        }
        if !has("effective_weight") {
            alter.push(
                "ALTER TABLE memories ADD COLUMN effective_weight REAL NOT NULL DEFAULT 1.0;",
            );
        }
        if !has("fingerprint") {
            alter.push("ALTER TABLE memories ADD COLUMN fingerprint TEXT NOT NULL DEFAULT '';");
        }
        if !has("claim_key") {
            alter.push("ALTER TABLE memories ADD COLUMN claim_key TEXT;");
        }
        if !has("claim_value") {
            alter.push("ALTER TABLE memories ADD COLUMN claim_value TEXT;");
        }
        if !has("failure_count") {
            alter.push("ALTER TABLE memories ADD COLUMN failure_count INTEGER NOT NULL DEFAULT 0;");
        }
        if !has("trust_ema") {
            alter.push("ALTER TABLE memories ADD COLUMN trust_ema REAL;");
        }
        if !has("trust_variance") {
            alter.push("ALTER TABLE memories ADD COLUMN trust_variance REAL NOT NULL DEFAULT 0.0;");
        }
        if !has("trust_samples") {
            alter.push("ALTER TABLE memories ADD COLUMN trust_samples INTEGER NOT NULL DEFAULT 0;");
        }
        if !has("trust_last_session_id") {
            alter.push("ALTER TABLE memories ADD COLUMN trust_last_session_id TEXT;");
        }
        if !has("ingestion_driver") {
            alter.push("ALTER TABLE memories ADD COLUMN ingestion_driver TEXT;");
        }
        // Namespace column â€” added in v0.3.0. Existing rows get 'default'.
        if !has("namespace") {
            alter
                .push("ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default';");
        }
        if !has("last_used_at") {
            alter.push("ALTER TABLE memories ADD COLUMN last_used_at INTEGER;");
        }

        for sql in alter {
            conn.execute_batch(sql)?;
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_claim_key ON memories (claim_key);
             CREATE INDEX IF NOT EXISTS idx_memories_store_state ON memories (store_state, archived, superseded_by);
             CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories (namespace, archived, superseded_by);",
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Store whose default trial cap does not limit unverified results.
    fn uncapped() -> Store {
        Store::in_memory_with_config(ScoringConfig {
            trial_cap: 8,
            ..ScoringConfig::default()
        })
        .unwrap()
    }

    fn state_of(store: &Store, id: i64) -> String {
        let conn = store.pool.get().unwrap();
        conn.query_row(
            "SELECT store_state FROM memories WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap()
    }

    fn count_in_state(store: &Store, state: &str) -> i64 {
        let conn = store.pool.get().unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE store_state = ?1",
            params![state],
            |r| r.get(0),
        )
        .unwrap()
    }

    fn archived_count(store: &Store) -> i64 {
        let conn = store.pool.get().unwrap();
        conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE archived = 1",
            [],
            |r| r.get(0),
        )
        .unwrap()
    }

    fn live_meta(content: &str, reinforcement_count: i64) -> QualityMeta {
        let mut meta = QualityMeta::default_active(content);
        meta.reinforcement_count = reinforcement_count;
        meta
    }

    #[test]
    fn test_blob_roundtrip() {
        let v: Vec<f32> = vec![1.0, -0.5, 0.0, 0.75];
        let blob = vec_to_blob(&v);
        let rec = blob_to_vec(&blob).unwrap();
        for (a, b) in v.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0_f32, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn test_insert_and_search() {
        // Uncapped: both rows are unverified, and the default trial cap would return one.
        let store = uncapped();
        store.insert("memory one", &[1.0_f32, 0.0, 0.0]).unwrap();
        store.insert("memory two", &[0.0_f32, 1.0, 0.0]).unwrap();
        let results = store.search(&[0.9_f32, 0.1, 0.0], 2, false).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "memory one");
    }

    #[test]
    fn test_forget_releases_fingerprint() {
        let store = Store::in_memory().unwrap();
        let meta = QualityMeta::default_active("fingerprint release check");
        let id = store
            .insert_with_quality("fingerprint release check", &[1.0_f32, 0.0, 0.0], &meta)
            .unwrap();
        assert!(store.fingerprint_exists(&meta.fingerprint).unwrap());
        assert!(store.forget(id).unwrap());
        assert!(!store.fingerprint_exists(&meta.fingerprint).unwrap());
    }

    /// Two namespaces sharing one SQLite file. Rows are made prune-eligible with
    /// raw SQL (low importance, 40 days old, never reinforced).
    fn two_tenant_stores(dir: &std::path::Path) -> (Store, Store) {
        let p = dir.join("isolation.db");
        let p = p.to_str().unwrap();
        (
            Store::open_ns(p, "tenant_a").unwrap(),
            Store::open_ns(p, "tenant_b").unwrap(),
        )
    }

    #[test]
    fn test_maintenance_pass_is_namespace_scoped_and_releases_fingerprints() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_tenant_stores(dir.path());
        let meta_a = QualityMeta::default_active("tenant a stale note");
        let meta_b = QualityMeta::default_active("tenant b stale note");
        a.insert_with_quality("tenant a stale note", &[1.0_f32, 0.0], &meta_a)
            .unwrap();
        b.insert_with_quality("tenant b stale note", &[0.0_f32, 1.0], &meta_b)
            .unwrap();

        a.pool
            .get()
            .unwrap()
            .execute(
                "UPDATE memories SET importance_base = 0.05, created_at = created_at - 40 * 86400",
                [],
            )
            .unwrap();

        a.maintenance_pass().unwrap();

        assert_eq!(a.count().unwrap(), 0, "tenant_a stale row must be pruned");
        assert_eq!(b.count().unwrap(), 1, "tenant_b must be untouched");
        assert_eq!(b.all().unwrap()[0].content, "tenant b stale note");
        assert!(!a.fingerprint_exists(&meta_a.fingerprint).unwrap());
        assert!(b.fingerprint_exists(&meta_b.fingerprint).unwrap());
    }

    #[test]
    fn test_forget_and_clear_are_namespace_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_tenant_stores(dir.path());
        let id_a = a.insert("tenant a item", &[1.0_f32, 0.0]).unwrap();
        b.insert("tenant b item", &[0.0_f32, 1.0]).unwrap();

        assert!(!b.forget(id_a).unwrap(), "b must not forget a's row");
        assert_eq!(a.count().unwrap(), 1);

        a.clear().unwrap();
        assert_eq!(a.count().unwrap(), 0);
        assert_eq!(b.count().unwrap(), 1, "a.clear() must not wipe tenant_b");
    }

    /// Raw feedback-related columns for one row, read with no namespace filter so a
    /// test can see whether another tenant's call changed it.
    #[derive(Debug, PartialEq)]
    struct FeedbackRow {
        reinforcement_count: i64,
        failure_count: i64,
        confidence: f32,
        importance_base: f32,
        trust_ema: Option<f32>,
        trust_samples: i64,
        trust_last_session_id: Option<String>,
        last_accessed_at: Option<i64>,
        created_at: i64,
    }

    fn raw_feedback_row(store: &Store, id: i64) -> FeedbackRow {
        store
            .pool
            .get()
            .unwrap()
            .query_row(
                "SELECT reinforcement_count, failure_count, confidence, importance_base,
                        trust_ema, trust_samples, trust_last_session_id,
                        last_accessed_at, created_at
                 FROM memories WHERE id = ?1",
                params![id],
                |r| {
                    Ok(FeedbackRow {
                        reinforcement_count: r.get(0)?,
                        failure_count: r.get(1)?,
                        confidence: r.get(2)?,
                        importance_base: r.get(3)?,
                        trust_ema: r.get(4)?,
                        trust_samples: r.get(5)?,
                        trust_last_session_id: r.get(6)?,
                        last_accessed_at: r.get(7)?,
                        created_at: r.get(8)?,
                    })
                },
            )
            .unwrap()
    }

    #[test]
    fn test_reinforce_if_used_is_namespace_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_tenant_stores(dir.path());
        let text = "decimal money correct precision";
        let id_a = a.insert(text, &[1.0_f32, 0.0]).unwrap();
        let before = raw_feedback_row(&a, id_a);

        // The output equals the memory, so the attribution gate would pass if the
        // namespace filter were missing: only the namespace stops this call.
        let reinforced = b
            .reinforce_if_used(id_a, text, true, None, Some("tenant_b_session"))
            .unwrap();
        assert!(!reinforced, "tenant_b must not reinforce tenant_a's row");
        assert_eq!(
            raw_feedback_row(&a, id_a),
            before,
            "tenant_a's row must be unchanged"
        );

        // Control: the owning namespace can reinforce the same row.
        assert!(a
            .reinforce_if_used(id_a, text, true, None, Some("tenant_a_session"))
            .unwrap());
        let after = raw_feedback_row(&a, id_a);
        assert_eq!(after.reinforcement_count, 1);
        assert_eq!(
            after.trust_last_session_id.as_deref(),
            Some("tenant_a_session")
        );
    }

    #[test]
    fn test_penalize_if_used_is_namespace_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_tenant_stores(dir.path());
        let id_a = a.insert("tenant a lesson", &[1.0_f32, 0.0]).unwrap();
        let id_b = b.insert("tenant b lesson", &[0.0_f32, 1.0]).unwrap();
        let before_a = raw_feedback_row(&a, id_a);

        // Foreign id alone: no outcome, nothing written.
        let outcomes = b.penalize_if_used(&[id_a], 1.0).unwrap();
        assert!(outcomes.is_empty(), "foreign id must yield no outcome");
        assert_eq!(raw_feedback_row(&a, id_a), before_a);

        // Mixed batch: only the caller's own id is penalized.
        let outcomes = b.penalize_if_used(&[id_a, id_b], 0.5).unwrap();
        let penalized: Vec<i64> = outcomes.iter().map(|o| o.id).collect();
        assert_eq!(penalized, [id_b]);
        assert_eq!(raw_feedback_row(&a, id_a), before_a);
        assert_eq!(raw_feedback_row(&b, id_b).failure_count, 1);

        // Control: the owning namespace can penalize its own row.
        let outcomes = a.penalize_if_used(&[id_a], 1.0).unwrap();
        assert_eq!(outcomes.len(), 1);
        let after_a = raw_feedback_row(&a, id_a);
        assert_eq!(after_a.failure_count, 1);
        // Feedback records evidence only; it never rewrites ingest-time features or the
        // success count.
        assert_eq!(after_a.confidence, before_a.confidence);
        assert_eq!(after_a.importance_base, before_a.importance_base);
        assert_eq!(after_a.reinforcement_count, before_a.reinforcement_count);
    }

    #[test]
    fn test_resolve_contradictions_ignores_foreign_id() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_tenant_stores(dir.path());
        let emb = [0.577_f32, 0.577, 0.577];

        let mut meta1 = QualityMeta::default_active("never use floats");
        meta1.claim_value = Some("negative".to_string());
        a.insert_with_quality("never use floats", &emb, &meta1)
            .unwrap();
        let mut meta2 = QualityMeta::default_active("always use floats");
        meta2.claim_value = Some("affirmative".to_string());
        let id2 = a
            .insert_with_quality("always use floats", &emb, &meta2)
            .unwrap();

        b.resolve_contradictions_for_id(id2).unwrap();
        assert_eq!(
            count_in_state(&a, "quarantined"),
            0,
            "tenant_b must not quarantine a's rows"
        );

        // Control: the owning namespace does resolve the pair (lexical verdict ->
        // the loser is quarantined, not archived).
        a.resolve_contradictions_for_id(id2).unwrap();
        assert_eq!(count_in_state(&a, "quarantined"), 1);
    }

    #[test]
    fn test_update_imported_metadata_is_namespace_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_tenant_stores(dir.path());
        let id_a = a.insert("tenant a import", &[1.0_f32, 0.0]).unwrap();
        let id_b = b.insert("tenant b import", &[0.0_f32, 1.0]).unwrap();
        let before_a = raw_feedback_row(&a, id_a);

        b.update_imported_metadata(&[id_a, id_b], Some(0.9), 500, 0.9, 0.9, Some(1_000))
            .unwrap();

        assert_eq!(
            raw_feedback_row(&a, id_a),
            before_a,
            "tenant_b must not rewrite tenant_a's metadata"
        );
        let row_b = raw_feedback_row(&b, id_b);
        assert_eq!(row_b.reinforcement_count, 500);
        assert_eq!(row_b.trust_ema, Some(0.9));
        assert_eq!(row_b.created_at, 1_000);

        // `created_at = None` keeps the stored timestamp.
        b.update_imported_metadata(&[id_b], Some(0.9), 7, 0.9, 0.9, None)
            .unwrap();
        let row_b = raw_feedback_row(&b, id_b);
        assert_eq!(row_b.reinforcement_count, 7);
        assert_eq!(row_b.created_at, 1_000);
    }

    #[test]
    fn test_authoritative_nli_neutral_prevents_supersede() {
        use crate::quality::{NliChecker, NliClassifier};
        use std::sync::Arc;

        struct AlwaysNeutral;
        impl NliClassifier for AlwaysNeutral {
            fn classify(&self, _p: &str, _h: &str) -> anyhow::Result<NliLabel> {
                Ok(NliLabel::Neutral)
            }
        }

        let emb = [0.577_f32, 0.577, 0.577];
        let seed = |store: &Store| -> i64 {
            for text in ["never use float for money", "always use float for money"] {
                let meta = QualityMeta::default_active(text);
                let id = store.insert_with_quality(text, &emb, &meta).unwrap();
                if text.starts_with("always") {
                    return id;
                }
            }
            unreachable!()
        };

        // Control: lexical heuristic only -> the pair is a contradiction, and since the
        // verdict is a heuristic one side is quarantined (not archived).
        let lexical = Store::in_memory().unwrap();
        let id = seed(&lexical);
        lexical.resolve_contradictions_for_id(id).unwrap();
        assert_eq!(count_in_state(&lexical, "quarantined"), 1);
        assert_eq!(archived_count(&lexical), 0);

        // Authoritative model says Neutral -> nothing is superseded or hidden.
        let mut modelled = Store::in_memory().unwrap();
        modelled.nli_checker = NliChecker::with_classifier(Arc::new(AlwaysNeutral));
        let id = seed(&modelled);
        modelled.resolve_contradictions_for_id(id).unwrap();
        assert_eq!(modelled.count().unwrap(), 2);
        assert_eq!(count_in_state(&modelled, "quarantined"), 0);
    }

    #[test]
    fn authoritative_contradiction_still_archives_the_loser() {
        use crate::quality::{NliChecker, NliClassifier};
        use std::sync::Arc;

        struct AlwaysContradiction;
        impl NliClassifier for AlwaysContradiction {
            fn classify(&self, _p: &str, _h: &str) -> anyhow::Result<NliLabel> {
                Ok(NliLabel::Contradiction)
            }
        }

        let emb = [0.577_f32, 0.577, 0.577];
        let mut store = Store::in_memory().unwrap();
        store.nli_checker = NliChecker::with_classifier(Arc::new(AlwaysContradiction));
        let mut ids = Vec::new();
        for text in ["never use float for money", "always use float for money"] {
            let mut meta = QualityMeta::default_active(text);
            meta.claim_value = Some(detect_polarity(text).as_str().to_string());
            ids.push(store.insert_with_quality(text, &emb, &meta).unwrap());
        }
        store.resolve_contradictions_for_id(ids[1]).unwrap();
        // A model-backed verdict is trusted: archived, not merely quarantined.
        assert_eq!(store.count().unwrap(), 1);
        assert_eq!(archived_count(&store), 1);
        assert_eq!(count_in_state(&store, "quarantined"), 0);
    }

    #[test]
    fn lexical_contradiction_quarantines_the_tls_pair_and_unquarantine_restores_it() {
        let store = Store::in_memory().unwrap();
        let emb = [0.577_f32, 0.577, 0.577];
        let mut ids = Vec::new();
        for text in [
            "TLS verification disabled in staging",
            "always enable TLS verification",
        ] {
            let mut meta = QualityMeta::default_active(text);
            meta.claim_value = Some(detect_polarity(text).as_str().to_string());
            ids.push(store.insert_with_quality(text, &emb, &meta).unwrap());
        }
        store.resolve_contradictions_for_id(ids[1]).unwrap();

        assert_eq!(count_in_state(&store, "quarantined"), 1);
        assert_eq!(archived_count(&store), 0);
        assert_eq!(store.count().unwrap(), 2, "nothing was deleted");
        let loser = ids
            .iter()
            .copied()
            .find(|id| state_of(&store, *id) == "quarantined")
            .expect("one side is quarantined");
        let survivors = uncapped_search(&store);
        assert_eq!(survivors.len(), 1);
        assert_ne!(survivors[0].id, loser);

        assert!(store.unquarantine(loser).unwrap());
        assert_eq!(state_of(&store, loser), "trial");
    }

    fn uncapped_search(store: &Store) -> Vec<Memory> {
        store.search(&[0.577, 0.577, 0.577], 8, false).unwrap()
    }

    #[test]
    fn test_clone_shares_cache_state() {
        // Regression: Store::clone() used to build a detached, empty StoreInner,
        // so deletes performed through a clone (background maintenance) never
        // reached the original's embedding cache.
        let store = Store::in_memory().unwrap();
        let id = store.insert("memory one", &[1.0_f32, 0.0, 0.0]).unwrap();
        assert!(store.max_similarity(&[1.0_f32, 0.0, 0.0]).unwrap() > 0.99);

        let clone = store.clone();
        assert!(clone.forget(id).unwrap());

        assert_eq!(store.max_similarity(&[1.0_f32, 0.0, 0.0]).unwrap(), 0.0);
        assert!(store
            .search(&[1.0_f32, 0.0, 0.0], 3, false)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_all() {
        let store = Store::in_memory().unwrap();
        store.insert("alpha", &[1.0_f32, 0.0]).unwrap();
        store.insert("beta", &[0.0_f32, 1.0]).unwrap();
        let all = store.all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_contradiction_resolution_with_polarity() {
        // Use identical embeddings (cosine = 1.0 > 0.80 threshold) to test the
        // contradiction pipeline deterministically, independent of ONNX model output.
        // This specifically validates the Polarity::from_stored fix: if detect_polarity
        // were called on the stored polarity string instead, it would return Neutral
        // for both and opposes() would always be false, archiving nothing.
        let store = Store::in_memory().unwrap();
        let emb = [0.577_f32, 0.577, 0.577]; // normalised, cosine(emb, emb) = 1.0

        let mut meta1 = QualityMeta::default_active("never use floats");
        meta1.claim_value = Some("negative".to_string());
        let _id1 = store
            .insert_with_quality("never use floats", &emb, &meta1)
            .unwrap();

        let mut meta2 = QualityMeta::default_active("always use floats");
        meta2.claim_value = Some("affirmative".to_string());
        let id2 = store
            .insert_with_quality("always use floats", &emb, &meta2)
            .unwrap();

        store.resolve_contradictions_for_id(id2).unwrap();

        // No NLI model: the lexical verdict quarantines the loser instead of archiving it.
        assert_eq!(
            count_in_state(&store, "quarantined"),
            1,
            "contradiction resolution must quarantine the loser"
        );
        assert_eq!(archived_count(&store), 0);
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn test_scores_descending() {
        let store = uncapped();
        store.insert("closest", &[1.0_f32, 0.0, 0.0]).unwrap();
        store.insert("middle", &[0.7_f32, 0.7, 0.0]).unwrap();
        store.insert("farthest", &[0.0_f32, 1.0, 0.0]).unwrap();
        let results = store.search(&[1.0, 0.0, 0.0], 3, false).unwrap();
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_forget_nonexistent() {
        let store = Store::in_memory().unwrap();
        assert!(!store.forget(99999).unwrap());
    }

    #[test]
    fn test_count_and_clear() {
        let store = Store::in_memory().unwrap();
        store.insert("a", &[1.0, 0.0]).unwrap();
        store.insert("b", &[0.0, 1.0]).unwrap();
        assert_eq!(store.count().unwrap(), 2);
        store.clear().unwrap();
        assert_eq!(store.count().unwrap(), 0);
    }

    // â”€â”€â”€ Failure-feedback / penalize_if_used tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    #[test]
    fn test_penalty_reduces_trust() {
        let store = Store::in_memory().unwrap();
        let mut meta = QualityMeta::default_active("fixed billing replaced float production");
        meta.importance_base = 0.8;
        meta.confidence = 0.8;
        meta.reinforcement_count = 3;
        let id = store
            .insert_with_quality(
                "fixed billing replaced float production",
                &[1.0, 0.0, 0.0],
                &meta,
            )
            .unwrap();
        let outcomes = store.penalize_if_used(&[id], 1.0).unwrap();
        assert_eq!(outcomes.len(), 1);
        assert!(
            outcomes[0].trust_after < outcomes[0].trust_before,
            "trust must decrease after penalty: before={:.4} after={:.4}",
            outcomes[0].trust_before,
            outcomes[0].trust_after,
        );
    }

    #[test]
    fn test_repeated_penalty_drops_below_hint_threshold() {
        // HINT threshold in default MemoryPolicy is 0.40
        let store = Store::in_memory().unwrap();
        let mut meta = QualityMeta::default_active("always use float for money calculations");
        meta.importance_base = 0.7;
        meta.confidence = 0.7;
        meta.reinforcement_count = 2;
        let id = store
            .insert_with_quality(
                "always use float for money calculations",
                &[1.0, 0.0, 0.0],
                &meta,
            )
            .unwrap();
        for _ in 0..5 {
            store.penalize_if_used(&[id], 1.0).unwrap();
        }
        let all = store.all().unwrap();
        let mem = all.iter().find(|m| m.id == id).expect("memory must exist");
        assert!(
            mem.trust < 0.40,
            "trust {:.4} should drop below HINT threshold (0.40) after 5 failures",
            mem.trust,
        );
    }

    #[test]
    fn test_recovery_after_penalty() {
        let store = Store::in_memory().unwrap();
        let mut meta = QualityMeta::default_active("decimal money correct precision");
        meta.importance_base = 0.8;
        meta.confidence = 0.8;
        meta.reinforcement_count = 2;
        let id = store
            .insert_with_quality("decimal money correct precision", &[1.0, 0.0, 0.0], &meta)
            .unwrap();
        store.penalize_if_used(&[id], 1.0).unwrap();
        store.penalize_if_used(&[id], 1.0).unwrap();
        let trust_penalized = store
            .all()
            .unwrap()
            .into_iter()
            .find(|m| m.id == id)
            .unwrap()
            .trust;
        // Reinforce via Jaccard token overlap (>= 0.15)
        store
            .reinforce_if_used(
                id,
                "decimal money correct precision usage",
                true,
                None,
                None,
            )
            .unwrap();
        let trust_recovered = store
            .all()
            .unwrap()
            .into_iter()
            .find(|m| m.id == id)
            .unwrap()
            .trust;
        assert!(
            trust_recovered > trust_penalized,
            "trust must recover after reinforcement: penalized={:.4} recovered={:.4}",
            trust_penalized,
            trust_recovered,
        );
    }

    #[test]
    fn test_ranking_improves_after_penalizing_bad_memory() {
        let store = Store::in_memory().unwrap();
        // Same embedding so cosine sim is equal: ranking separates via `trust` (s=3,f=3 -> 0.50 vs s=3,f=0 -> 0.71)
        let mut bad_meta = QualityMeta::default_active("float is fine for money");
        bad_meta.importance_base = 0.7;
        bad_meta.confidence = 0.7;
        bad_meta.reinforcement_count = 3;
        let bad_id = store
            .insert_with_quality("float is fine for money", &[1.0, 0.0, 0.0], &bad_meta)
            .unwrap();

        let mut good_meta = QualityMeta::default_active("decimal is correct for money");
        good_meta.importance_base = 0.7;
        good_meta.confidence = 0.7;
        good_meta.reinforcement_count = 3;
        store
            .insert_with_quality("decimal is correct for money", &[1.0, 0.0, 0.0], &good_meta)
            .unwrap();

        // Apply 3 failures to the bad memory: only failure_count changes (rc/importance are left alone)
        for _ in 0..3 {
            store.penalize_if_used(&[bad_id], 1.0).unwrap();
        }

        let results = store.search(&[1.0, 0.0, 0.0], 2, false).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].content, "decimal is correct for money",
            "good memory must outrank penalized bad memory after 3 failures"
        );
    }

    /// Unit vector in the (x, y) plane whose cosine with the query `[1, 0]` is `cos`.
    fn at_cosine(cos: f32) -> Vec<f32> {
        vec![cos, (1.0 - cos * cos).sqrt()]
    }

    /// Pilot shape (eval `http_status` s4): the penalized row is a hair closer by raw cosine
    /// (0.614 vs 0.612) but the other row has a recorded success. Ranking is
    /// `0.75*cos + 0.20*trust + 0.05*recency`, so trust 0.60 vs 0.40 is worth +0.04 and must win.
    /// This pins that `trust` (not the unused-for-ranking `effective_weight`) separates the two.
    #[test]
    fn test_verified_trust_outranks_slightly_closer_penalized_row() {
        let store = Store::in_memory().unwrap();
        let query = [1.0_f32, 0.0];
        let gold = store.insert("gold lesson", &at_cosine(0.612)).unwrap();
        let poison = store.insert("poison lesson", &at_cosine(0.614)).unwrap();
        assert!(store.reinforce_declared(gold, Some("s1")).unwrap());
        store.penalize_if_used(&[poison], 1.0).unwrap();

        let results = store.search(&query, 2, false).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].id, gold,
            "verified trust=0.60 must outrank penalized trust=0.40"
        );
        assert_eq!(results[1].id, poison);
        // Beta(2,2) posterior: (2+1)/(4+1+0) = 0.6 and (2+0)/(4+0+1) = 0.4.
        assert!(
            (results[0].trust - 0.6).abs() < 1e-3,
            "gold trust {}",
            results[0].trust
        );
        assert!(
            (results[1].trust - 0.4).abs() < 1e-3,
            "poison trust {}",
            results[1].trust
        );
        assert!(
            results[0].score > results[1].score,
            "output order must be descending score"
        );
    }

    /// With identical cosine the order is decided by trust alone: a recorded success ranks
    /// above a recorded failure.
    #[test]
    fn test_equal_cosine_orders_by_trust_success_then_failure() {
        let store = uncapped();
        let query = [1.0_f32, 0.0];
        let emb = at_cosine(0.7);
        let failed = store.insert("failed lesson", &emb).unwrap();
        let worked = store.insert("worked lesson", &emb).unwrap();
        assert!(store.reinforce_declared(worked, Some("s1")).unwrap());
        store.penalize_if_used(&[failed], 1.0).unwrap();

        let results = store.search(&query, 2, false).unwrap();
        assert_eq!(
            results.iter().map(|m| m.id).collect::<Vec<_>>(),
            vec![worked, failed]
        );
    }

    // â”€â”€â”€ Namespace isolation tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Two in-memory stores with different namespaces must not see each other's rows.
    #[test]
    fn test_namespace_hard_isolation() {
        let store_a = Store::in_memory_ns("agent_a").unwrap();
        let store_b = Store::in_memory_ns("agent_b").unwrap();

        assert_eq!(store_a.namespace, "agent_a");
        assert_eq!(store_b.namespace, "agent_b");

        store_a.insert("only in a", &[1.0_f32, 0.0, 0.0]).unwrap();
        store_b.insert("only in b", &[0.0_f32, 1.0, 0.0]).unwrap();

        assert_eq!(store_a.count().unwrap(), 1);
        assert_eq!(store_b.count().unwrap(), 1);
        assert_eq!(store_a.all().unwrap()[0].content, "only in a");
        assert_eq!(store_b.all().unwrap()[0].content, "only in b");
    }

    /// Shared SQLite file â€” two Store handles with different namespaces must
    /// not bleed across (the real multi-tenant scenario).
    #[test]
    fn test_namespace_shared_file_isolation() {
        let dir = std::env::temp_dir();
        static TEST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = TEST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!(
            "memoire_ns_{}_{}_{}.db",
            std::process::id(),
            unique_id,
            // cheap unique suffix using stack address
            &dir as *const _ as usize,
        ));
        let path_str = path.to_str().unwrap();

        {
            let store_a = Store::open_ns(path_str, "tenant_a").unwrap();
            let store_b = Store::open_ns(path_str, "tenant_b").unwrap();

            store_a.insert("tenant_a secret", &[1.0_f32, 0.0]).unwrap();
            store_b.insert("tenant_b secret", &[0.0_f32, 1.0]).unwrap();

            assert_eq!(store_a.count().unwrap(), 1);
            assert_eq!(store_b.count().unwrap(), 1);

            assert_eq!(store_a.all().unwrap()[0].content, "tenant_a secret");
            assert_eq!(store_b.all().unwrap()[0].content, "tenant_b secret");

            // Search from tenant_a's perspective must not surface tenant_b's memory.
            let found = store_a.search(&[0.0_f32, 1.0], 5, false).unwrap();
            assert!(
                found.iter().all(|m| m.content != "tenant_b secret"),
                "tenant_a search must not return tenant_b's memory"
            );
        }

        // Cleanup â€” ignore errors (WAL files on Windows)
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path_str}-wal"));
        let _ = std::fs::remove_file(format!("{path_str}-shm"));
    }

    // ---- Step 8b-1: posterior trust, exposure, state machine, feedback integrity ----

    #[test]
    fn fresh_rows_start_at_neutral_trust_regardless_of_ingest_features() {
        let store = uncapped();
        let mut low = live_meta("low quality text", 0);
        low.importance_base = 0.05;
        low.confidence = 0.1;
        let mut high = live_meta("high quality text", 0);
        high.importance_base = 0.95;
        high.confidence = 0.95;
        store
            .insert_with_quality("low quality text", &[1.0, 0.0], &low)
            .unwrap();
        store
            .insert_with_quality("high quality text", &[0.0, 1.0], &high)
            .unwrap();
        for m in store.all().unwrap() {
            assert!(
                (m.trust - 0.5).abs() < 1e-6,
                "trust {} for {}",
                m.trust,
                m.content
            );
            assert!((m.uncertainty - 1.0).abs() < 1e-6);
            assert_eq!(m.exposure, "trial");
            assert_eq!(m.state, "trial");
        }
    }

    #[test]
    fn unreinforced_relevant_memory_surfaces_on_first_recall() {
        let store = Store::in_memory().unwrap();
        store
            .insert("a brand new lesson", &[1.0, 0.0, 0.0])
            .unwrap();
        let results = store.search(&[0.6, 0.8, 0.0], 3, false).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].exposure, "trial");
    }

    #[test]
    fn trial_cap_bounds_unverified_results_and_keeps_verified_ones() {
        let store = Store::in_memory().unwrap(); // trial_cap = 1
        store.insert("u1", &[1.0, 0.0, 0.0]).unwrap();
        store.insert("u2", &[0.9, 0.1, 0.0]).unwrap();
        store.insert("u3", &[0.8, 0.2, 0.0]).unwrap();
        store
            .insert_with_quality("verified", &[0.0, 1.0, 0.0], &live_meta("verified", 3))
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 3, false).unwrap();
        let trial: Vec<&Memory> = results.iter().filter(|m| m.exposure == "trial").collect();
        assert_eq!(trial.len(), 1, "exactly one unverified slot");
        assert_eq!(
            trial[0].content, "u1",
            "the slot goes to the closest unverified row"
        );
        assert!(results.iter().any(|m| m.content == "verified"));
        assert_eq!(results.len(), 2);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }

        // k == 1 with a verified candidate: the verified one wins the only slot.
        let one = store.search(&[1.0, 0.0, 0.0], 1, false).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].content, "verified");
        assert_eq!(one[0].exposure, "follow");
    }

    #[test]
    fn trial_cap_alone_bounds_exposure_when_nothing_is_verified() {
        let capped = Store::in_memory().unwrap();
        let wide = Store::in_memory_with_config(ScoringConfig {
            trial_cap: 2,
            ..ScoringConfig::default()
        })
        .unwrap();
        let off = Store::in_memory_with_config(ScoringConfig {
            trial_cap: 0,
            ..ScoringConfig::default()
        })
        .unwrap();
        for store in [&capped, &wide, &off] {
            store.insert("u1", &[1.0, 0.0, 0.0]).unwrap();
            store.insert("u2", &[0.9, 0.1, 0.0]).unwrap();
            store.insert("u3", &[0.8, 0.2, 0.0]).unwrap();
        }
        assert_eq!(capped.search(&[1.0, 0.0, 0.0], 3, false).unwrap().len(), 1);
        assert_eq!(wide.search(&[1.0, 0.0, 0.0], 3, false).unwrap().len(), 2);
        assert!(off.search(&[1.0, 0.0, 0.0], 3, false).unwrap().is_empty());
        // The hidden-rows view is an operator view: no trial cap.
        assert_eq!(capped.search(&[1.0, 0.0, 0.0], 3, true).unwrap().len(), 3);
    }

    #[test]
    fn trial_min_cos_filters_irrelevant_unverified_rows() {
        let store = Store::in_memory_with_config(ScoringConfig {
            trial_cap: 4,
            trial_min_cos: 0.5,
            ..ScoringConfig::default()
        })
        .unwrap();
        store.insert("unrelated", &[0.0, 1.0]).unwrap();
        assert!(store.search(&[1.0, 0.0], 3, false).unwrap().is_empty());
        store.insert("related", &[0.8, 0.6]).unwrap();
        let results = store.search(&[1.0, 0.0], 3, false).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "related");
    }

    #[test]
    fn verified_min_cos_drops_unrelated_verified_rows_but_keeps_related_ones() {
        let store = Store::in_memory_with_config(ScoringConfig {
            verified_min_cos: 0.3,
            ..ScoringConfig::default()
        })
        .unwrap();
        store.insert("trial", &[1.0, 0.0, 0.0]).unwrap();
        store
            .insert_with_quality(
                "unrelated verified",
                &[0.0, 1.0, 0.0],
                &live_meta("unrelated verified", 3),
            )
            .unwrap();
        store
            .insert_with_quality(
                "related verified",
                &[0.8, 0.6, 0.0],
                &live_meta("related verified", 3),
            )
            .unwrap();

        let results = store.search(&[1.0, 0.0, 0.0], 3, false).unwrap();
        assert!(results.iter().any(|m| m.content == "related verified"));
        assert!(results.iter().any(|m| m.content == "trial"));
        assert!(
            !results.iter().any(|m| m.content == "unrelated verified"),
            "a verified row below the floor is not offered"
        );
        // The hidden-rows view is an operator view: no floor, no cap.
        assert_eq!(store.search(&[1.0, 0.0, 0.0], 3, true).unwrap().len(), 3);
    }

    #[test]
    fn verified_below_the_floor_does_not_take_the_only_slot() {
        // Default floor 0.0: with k == 1 an unrelated verified row beats the relevant trial row.
        let open = Store::in_memory().unwrap();
        // Floor above the unrelated row's cosine: the relevant trial row gets the slot.
        let floored = Store::in_memory_with_config(ScoringConfig {
            verified_min_cos: 0.3,
            ..ScoringConfig::default()
        })
        .unwrap();
        for store in [&open, &floored] {
            store.insert("trial", &[1.0, 0.0, 0.0]).unwrap();
            store
                .insert_with_quality(
                    "unrelated verified",
                    &[0.0, 1.0, 0.0],
                    &live_meta("unrelated verified", 3),
                )
                .unwrap();
        }
        let one = open.search(&[1.0, 0.0, 0.0], 1, false).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].content, "unrelated verified");
        let one = floored.search(&[1.0, 0.0, 0.0], 1, false).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].content, "trial");
    }

    #[test]
    fn penalties_quarantine_a_high_cosine_memory_whatever_its_cosine_advantage() {
        let store = uncapped();
        let poison = store
            .insert("poison: use float for money", &[1.0, 0.0, 0.0])
            .unwrap();
        store
            .insert("truth: use Decimal", &[0.5, 0.866, 0.0])
            .unwrap();

        assert_eq!(store.penalize_if_used(&[poison], 1.0).unwrap().len(), 1);
        assert_eq!(
            state_of(&store, poison),
            "trial",
            "one failure is not enough"
        );
        assert_eq!(store.penalize_if_used(&[poison], 1.0).unwrap().len(), 1);
        assert_eq!(state_of(&store, poison), "quarantined");

        let default_view = store.search(&[1.0, 0.0, 0.0], 5, false).unwrap();
        assert!(default_view.iter().all(|m| m.id != poison));
        assert_eq!(default_view.len(), 1);

        let hidden_view = store.search(&[1.0, 0.0, 0.0], 5, true).unwrap();
        let shown = hidden_view
            .iter()
            .find(|m| m.id == poison)
            .expect("operator view");
        assert_eq!(shown.exposure, "hidden");
        assert_eq!(shown.state, "quarantined");

        // Quarantined rows take no further feedback.
        assert!(store.penalize_if_used(&[poison], 1.0).unwrap().is_empty());
        assert!(!store.reinforce_declared(poison, Some("s")).unwrap());
    }

    #[test]
    fn quarantined_rows_do_not_starve_live_candidates() {
        // top_k = 1 -> two cosine candidates. Three quarantined rows are closer to the
        // query than the live one; without hidden-id filtering they would fill both.
        let store = uncapped();
        let mut bad = Vec::new();
        for (i, emb) in [[1.0, 0.0, 0.0], [0.99, 0.14, 0.0], [0.98, 0.2, 0.0]]
            .iter()
            .enumerate()
        {
            bad.push(store.insert(&format!("bad {i}"), emb).unwrap());
        }
        let good = store.insert("good", &[0.6, 0.8, 0.0]).unwrap();
        for id in &bad {
            store.penalize_if_used(&[*id], 1.0).unwrap();
            store.penalize_if_used(&[*id], 1.0).unwrap();
        }
        let results = store.search(&[1.0, 0.0, 0.0], 1, false).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, good);
    }

    #[test]
    fn hidden_state_survives_reopen_of_a_file_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hidden.db");
        let path = path.to_str().unwrap();
        let (bad, good) = {
            let store = Store::open_ns(path, "default").unwrap();
            let bad = store.insert("bad", &[1.0, 0.0, 0.0]).unwrap();
            let good = store.insert("good", &[0.6, 0.8, 0.0]).unwrap();
            store.penalize_if_used(&[bad], 1.0).unwrap();
            store.penalize_if_used(&[bad], 1.0).unwrap();
            (bad, good)
        };
        let reopened = Store::open_ns(path, "default").unwrap();
        let results = reopened.search(&[1.0, 0.0, 0.0], 1, false).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, good);
        assert_ne!(results[0].id, bad);
    }

    #[test]
    fn unquarantine_restores_service_and_clears_failures() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = two_tenant_stores(dir.path());
        let id = a.insert("noisy but good lesson", &[1.0, 0.0]).unwrap();
        a.penalize_if_used(&[id], 1.0).unwrap();
        a.penalize_if_used(&[id], 1.0).unwrap();
        assert_eq!(state_of(&a, id), "quarantined");
        assert!(a.search(&[1.0, 0.0], 3, false).unwrap().is_empty());

        assert!(
            !b.unquarantine(id).unwrap(),
            "foreign namespace must not restore it"
        );
        assert_eq!(state_of(&a, id), "quarantined");

        assert!(a.unquarantine(id).unwrap());
        assert_eq!(raw_feedback_row(&a, id).failure_count, 0);
        assert_eq!(state_of(&a, id), "trial");
        assert_eq!(a.search(&[1.0, 0.0], 3, false).unwrap().len(), 1);
        assert!(!a.unquarantine(id).unwrap(), "no longer quarantined");
        // One more failure no longer quarantines it immediately.
        a.penalize_if_used(&[id], 1.0).unwrap();
        assert_eq!(state_of(&a, id), "trial");
    }

    #[test]
    fn reinforcement_is_one_vote_per_session_and_promotes_to_active() {
        let store = Store::in_memory().unwrap();
        let id = store.insert("some lesson", &[1.0, 0.0]).unwrap();
        assert!(store.reinforce_declared(id, Some("s1")).unwrap());
        assert!(
            !store.reinforce_declared(id, Some("s1")).unwrap(),
            "same session"
        );
        assert_eq!(state_of(&store, id), "trial");
        assert!(store.reinforce_declared(id, Some("s2")).unwrap());
        assert_eq!(state_of(&store, id), "trial");
        assert!(store.reinforce_declared(id, Some("s3")).unwrap());
        assert_eq!(state_of(&store, id), "active");

        let m = &store.search(&[1.0, 0.0], 3, false).unwrap()[0];
        assert_eq!(m.exposure, "follow");
        assert_eq!(m.reinforcement_count, 3);
        assert!((m.trust - 5.0 / 7.0).abs() < 1e-4, "trust {}", m.trust);

        // Without a session id every call counts (callers should supply one).
        let other = store.insert("another lesson", &[0.0, 1.0]).unwrap();
        assert!(store.reinforce_declared(other, None).unwrap());
        assert!(store.reinforce_declared(other, None).unwrap());
    }

    #[test]
    fn text_attribution_uses_containment_not_symmetric_overlap() {
        let store = Store::in_memory().unwrap();
        let lesson = "Quantize prices with Decimal using rounding half even for billing totals";
        let id = store.insert(lesson, &[1.0, 0.0]).unwrap();

        // A 300-word answer that uses the lesson: symmetric Jaccard would be ~0.04.
        let long_answer = format!("{} {}", "filler ".repeat(300), lesson);
        assert!(store
            .reinforce_if_used(id, &long_answer, true, None, Some("s1"))
            .unwrap());

        let unrelated = "filler ".repeat(300);
        assert!(!store
            .reinforce_if_used(id, &unrelated, true, None, Some("s2"))
            .unwrap());
        assert!(!store
            .reinforce_if_used(id, &long_answer, false, None, Some("s3"))
            .unwrap());
        assert_eq!(raw_feedback_row(&store, id).reinforcement_count, 1);
    }

    #[test]
    fn concurrent_feedback_is_not_lost() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent.db");
        let store = Store::open_ns(path.to_str().unwrap(), "default").unwrap();
        let id = store.insert("shared lesson", &[1.0, 0.0]).unwrap();
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let s = store.clone();
                std::thread::spawn(move || {
                    assert!(s
                        .reinforce_declared(id, Some(&format!("session-{i}")))
                        .unwrap());
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(raw_feedback_row(&store, id).reinforcement_count, 8);
    }

    #[test]
    fn penalize_leaves_ingest_features_alone_and_zero_severity_is_a_noop() {
        let store = Store::in_memory().unwrap();
        let mut meta = live_meta("fixed billing bug", 2);
        meta.importance_base = 0.7;
        meta.confidence = 0.6;
        let id = store
            .insert_with_quality("fixed billing bug", &[1.0, 0.0], &meta)
            .unwrap();
        let before = raw_feedback_row(&store, id);

        let noop = store.penalize_if_used(&[id], 0.0).unwrap();
        assert_eq!(noop.len(), 1);
        assert_eq!(noop[0].trust_before, noop[0].trust_after);
        assert_eq!(raw_feedback_row(&store, id), before);

        let out = store.penalize_if_used(&[id], 1.0).unwrap();
        assert!((out[0].trust_before - 4.0 / 6.0).abs() < 1e-6);
        assert!((out[0].trust_after - 4.0 / 7.0).abs() < 1e-6);
        let after = raw_feedback_row(&store, id);
        assert_eq!(after.failure_count, 1);
        assert_eq!(after.reinforcement_count, 2);
        assert_eq!(after.importance_base, before.importance_base);
        assert_eq!(after.confidence, before.confidence);
    }

    #[test]
    fn shadow_rows_are_hidden_until_promoted() {
        let store = uncapped();
        let mut meta = live_meta("hidden lesson", 0);
        meta.store_state = "shadow".to_string();
        let id = store
            .insert_with_quality("hidden lesson", &[1.0, 0.0], &meta)
            .unwrap();
        assert!(store.search(&[1.0, 0.0], 3, false).unwrap().is_empty());
        let hidden = store.search(&[1.0, 0.0], 3, true).unwrap();
        assert_eq!(hidden.len(), 1);
        assert_eq!(hidden[0].exposure, "hidden");

        assert!(store.promote_shadow(id).unwrap());
        assert_eq!(state_of(&store, id), "trial");
        assert_eq!(store.search(&[1.0, 0.0], 3, false).unwrap().len(), 1);
        assert!(!store.promote_shadow(id).unwrap(), "already live");
    }

    #[test]
    fn search_plain_is_cosine_only_and_writes_nothing() {
        let store = Store::in_memory().unwrap();
        let near = store.insert("near", &[1.0, 0.0]).unwrap();
        let far = store.insert("far", &[0.0, 1.0]).unwrap();
        store.penalize_if_used(&[far], 1.0).unwrap();
        store.penalize_if_used(&[far], 1.0).unwrap();
        assert_eq!(state_of(&store, far), "quarantined");
        {
            let conn = store.pool.get().unwrap();
            conn.execute("UPDATE memories SET last_used_at = 1", [])
                .unwrap();
        }

        let plain = store.search_plain(&[0.9, 0.1], 5).unwrap();
        assert_eq!(
            plain.len(),
            2,
            "plain recall ignores state and the trial cap"
        );
        assert_eq!(plain[0].id, near);
        assert!(plain.iter().all(|m| m.exposure == "plain"));
        assert!(plain[0].score >= plain[1].score);
        let last_used: i64 = {
            let conn = store.pool.get().unwrap();
            conn.query_row("SELECT MIN(last_used_at) FROM memories", [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(last_used, 1, "search_plain must not write last_used_at");

        store.search(&[0.9, 0.1], 5, false).unwrap();
        let last_used: i64 = {
            let conn = store.pool.get().unwrap();
            conn.query_row("SELECT MAX(last_used_at) FROM memories", [], |r| r.get(0))
                .unwrap()
        };
        assert!(last_used > 1, "search records last_used_at");
    }

    #[test]
    fn contradiction_mode_decides_how_a_contradiction_is_resolved() {
        let emb = [0.577_f32, 0.577, 0.577];
        // (mode, quarantined, archived): no heuristic mode archives.
        for (mode, quarantined, archived) in [
            (ContradictionMode::Off, 0, 0),
            (ContradictionMode::Nli, 1, 0),
            (ContradictionMode::Legacy, 1, 0),
        ] {
            let store = Store::in_memory_with_config(ScoringConfig {
                contradiction_mode: mode,
                ..ScoringConfig::default()
            })
            .unwrap();
            let mut m1 = QualityMeta::default_active("never use floats");
            m1.claim_value = Some("negative".to_string());
            store
                .insert_with_quality("never use floats", &emb, &m1)
                .unwrap();
            let mut m2 = QualityMeta::default_active("always use floats");
            m2.claim_value = Some("affirmative".to_string());
            let id2 = store
                .insert_with_quality("always use floats", &emb, &m2)
                .unwrap();
            store.resolve_contradictions_for_id(id2).unwrap();
            assert_eq!(
                count_in_state(&store, "quarantined"),
                quarantined,
                "{mode:?}"
            );
            assert_eq!(archived_count(&store), archived, "{mode:?}");
        }
    }

    fn insert_tls_pair(store: &Store) -> Vec<i64> {
        let emb = [0.577_f32, 0.577, 0.577];
        let mut ids = Vec::new();
        for text in [
            "TLS verification disabled in staging",
            "always enable TLS verification",
        ] {
            let mut meta = QualityMeta::default_active(text);
            meta.claim_value = Some(detect_polarity(text).as_str().to_string());
            ids.push(store.insert_with_quality(text, &emb, &meta).unwrap());
        }
        ids
    }

    #[test]
    fn newer_unverified_statement_displaces_an_older_unverified_one() {
        let store = Store::in_memory().unwrap();
        let ids = insert_tls_pair(&store);
        store.resolve_contradictions_for_id(ids[1]).unwrap();
        assert_eq!(state_of(&store, ids[0]), "quarantined", "older loses");
        assert_ne!(state_of(&store, ids[1]), "quarantined", "newer wins");
    }

    #[test]
    fn incumbent_with_feedback_evidence_beats_an_unverified_newcomer() {
        let store = Store::in_memory().unwrap();
        let ids = insert_tls_pair(&store);
        store
            .pool
            .get()
            .unwrap()
            .execute(
                "UPDATE memories SET reinforcement_count = 4, created_at = created_at - 3000000
                 WHERE id = ?1",
                params![ids[0]],
            )
            .unwrap();
        store.resolve_contradictions_for_id(ids[1]).unwrap();
        assert_eq!(
            state_of(&store, ids[1]),
            "quarantined",
            "unverified newcomer loses"
        );
        assert_ne!(state_of(&store, ids[0]), "quarantined");
        assert!(store.unquarantine(ids[1]).unwrap(), "and is recoverable");
    }

    #[test]
    fn legacy_contradiction_quarantines_and_is_recoverable() {
        let store = Store::in_memory_with_config(ScoringConfig {
            contradiction_mode: ContradictionMode::Legacy,
            ..ScoringConfig::default()
        })
        .unwrap();
        let ids = insert_tls_pair(&store);
        store.resolve_contradictions_for_id(ids[1]).unwrap();
        assert_eq!(archived_count(&store), 0);
        assert_eq!(store.count().unwrap(), 2);
        assert!(store.unquarantine(ids[0]).unwrap());
    }

    #[test]
    fn search_within_days_filters_age_before_the_trial_cap() {
        let store = Store::in_memory().unwrap(); // trial_cap = 1
        let q = [1.0_f32, 0.0, 0.0];
        let old = store
            .insert_with_quality("old lesson", &q, &QualityMeta::default_active("old lesson"))
            .unwrap();
        let fresh = store
            .insert_with_quality(
                "fresh lesson",
                &[0.8, 0.6, 0.0],
                &QualityMeta::default_active("fresh lesson"),
            )
            .unwrap();
        store
            .pool
            .get()
            .unwrap()
            .execute(
                "UPDATE memories SET created_at = created_at - 8640000, last_used_at = 1
                 WHERE id = ?1",
                params![old],
            )
            .unwrap();

        // The old row is the nearest and would take the single trial slot; it must not.
        let got = store.search_within_days(&q, 5, 30.0).unwrap();
        assert_eq!(got.iter().map(|m| m.id).collect::<Vec<_>>(), vec![fresh]);
        let last_used: i64 = store
            .pool
            .get()
            .unwrap()
            .query_row(
                "SELECT last_used_at FROM memories WHERE id = ?1",
                params![old],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(last_used, 1, "a filtered-out row is not marked as used");
    }

    #[test]
    fn imported_metadata_rederives_live_state_from_counts() {
        let store = Store::in_memory().unwrap();
        let id = store.insert("imported lesson", &[1.0, 0.0]).unwrap();
        store
            .update_imported_metadata(&[id], Some(0.9), 500, 0.9, 0.9, None)
            .unwrap();
        assert_eq!(state_of(&store, id), "active");
        assert_eq!(raw_feedback_row(&store, id).failure_count, 0);
    }

    #[test]
    fn migration_v3_rederives_live_states_and_bumps_user_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("legacy.db");
        let path = path.to_str().unwrap();
        let ids = {
            let store = Store::open_ns(path, "default").unwrap();
            let mut ids = Vec::new();
            for (i, emb) in [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ]
            .iter()
            .enumerate()
            {
                ids.push(store.insert(&format!("legacy {i}"), emb).unwrap());
            }
            ids
        };
        {
            // Pretend this database was written by the pre-8b engine.
            let conn = Connection::open(path).unwrap();
            let rows = [
                (ids[0], "active", 0, 0),
                (ids[1], "active", 3, 0),
                (ids[2], "active", 0, 2),
                (ids[3], "shadow", 0, 0),
            ];
            for (id, state, rc, fc) in rows {
                conn.execute(
                    "UPDATE memories SET store_state = ?1, reinforcement_count = ?2,
                            failure_count = ?3 WHERE id = ?4",
                    params![state, rc, fc, id],
                )
                .unwrap();
            }
            conn.execute_batch("PRAGMA user_version = 2;").unwrap();
        }
        let store = Store::open_ns(path, "default").unwrap();
        assert_eq!(state_of(&store, ids[0]), "trial");
        assert_eq!(state_of(&store, ids[1]), "active");
        assert_eq!(state_of(&store, ids[2]), "quarantined");
        assert_eq!(state_of(&store, ids[3]), "shadow");
        let version: i64 = store
            .pool
            .get()
            .unwrap()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(version, 3);
    }

    // ---- Step 8b-2: shadow corroboration ----

    fn shadow_meta(text: &str) -> QualityMeta {
        let mut meta = QualityMeta::default_active(text);
        meta.store_state = "shadow".to_string();
        meta.claim_value = Some(detect_polarity(text).as_str().to_string());
        meta
    }

    #[test]
    fn corroboration_promotes_a_shadow_row_without_counting_evidence() {
        let store = Store::in_memory().unwrap();
        let id = store
            .insert_with_quality(
                "set a timeout on every outbound request",
                &[1.0, 0.0, 0.0],
                &shadow_meta("set a timeout on every outbound request"),
            )
            .unwrap();
        assert!(store.search(&[1.0, 0.0, 0.0], 3, false).unwrap().is_empty());

        let promoted = store
            .corroborate_shadow(
                "outbound requests should always get a timeout",
                &[0.99, 0.141, 0.0],
                &[],
            )
            .unwrap();
        assert_eq!(promoted, Some(id));
        assert_eq!(state_of(&store, id), "trial");
        assert_eq!(raw_feedback_row(&store, id).reinforcement_count, 0);
        assert_eq!(raw_feedback_row(&store, id).failure_count, 0);
        assert_eq!(store.count().unwrap(), 1, "no duplicate row");
        assert_eq!(store.search(&[1.0, 0.0, 0.0], 3, false).unwrap().len(), 1);
    }

    #[test]
    fn corroboration_requires_similarity_independence_and_a_shadow_row() {
        let store = Store::in_memory().unwrap();
        let shadow = store
            .insert_with_quality(
                "shadow lesson",
                &[1.0, 0.0, 0.0],
                &shadow_meta("shadow lesson"),
            )
            .unwrap();
        let mut quarantined_meta = shadow_meta("quarantined lesson");
        quarantined_meta.store_state = "quarantined".to_string();
        let quarantined = store
            .insert_with_quality("quarantined lesson", &[0.0, 1.0, 0.0], &quarantined_meta)
            .unwrap();
        let live = store.insert("live lesson", &[0.0, 0.0, 1.0]).unwrap();

        // Below the 0.92 cosine bar.
        assert_eq!(
            store
                .corroborate_shadow("x", &[0.9, 0.436, 0.0], &[])
                .unwrap(),
            None
        );
        // Rows inserted earlier in the same ingest call never corroborate themselves.
        assert_eq!(
            store
                .corroborate_shadow("x", &[1.0, 0.0, 0.0], &[shadow])
                .unwrap(),
            None
        );
        assert_eq!(state_of(&store, shadow), "shadow");
        // Quarantined and already-live rows are not promoted.
        assert_eq!(
            store
                .corroborate_shadow("x", &[0.0, 1.0, 0.0], &[])
                .unwrap(),
            None
        );
        assert_eq!(state_of(&store, quarantined), "quarantined");
        assert_eq!(
            store
                .corroborate_shadow("x", &[0.0, 0.0, 1.0], &[])
                .unwrap(),
            None
        );
        assert_eq!(state_of(&store, live), "trial");
        // Control: with a qualifying near-duplicate and no exclusion, it promotes.
        assert_eq!(
            store
                .corroborate_shadow("x", &[1.0, 0.0, 0.0], &[])
                .unwrap(),
            Some(shadow)
        );
    }

    #[test]
    fn a_closer_quarantined_duplicate_does_not_block_corroborating_a_shadow_row() {
        let store = Store::in_memory().unwrap();
        let mut quarantined_meta = shadow_meta("quarantined twin");
        quarantined_meta.store_state = "quarantined".to_string();
        let quarantined = store
            .insert_with_quality("quarantined twin", &[1.0, 0.0, 0.0], &quarantined_meta)
            .unwrap();
        // Cosine 0.95 with the query: below the quarantined twin (1.0) but above the bar.
        let shadow = store
            .insert_with_quality(
                "shadow twin",
                &[0.95, 0.312, 0.0],
                &shadow_meta("shadow twin"),
            )
            .unwrap();

        let promoted = store
            .corroborate_shadow("a restatement", &[1.0, 0.0, 0.0], &[])
            .unwrap();
        assert_eq!(promoted, Some(shadow));
        assert_eq!(state_of(&store, shadow), "trial");
        assert_eq!(state_of(&store, quarantined), "quarantined");
    }

    #[test]
    fn corroboration_never_promotes_a_row_the_incoming_text_contradicts() {
        let emb = [0.577_f32, 0.577, 0.577];
        for (mode, expected_promoted) in [
            (ContradictionMode::Nli, false),
            (ContradictionMode::Legacy, false),
            (ContradictionMode::Off, true),
        ] {
            let store = Store::in_memory_with_config(ScoringConfig {
                contradiction_mode: mode,
                ..ScoringConfig::default()
            })
            .unwrap();
            let id = store
                .insert_with_quality("never use floats", &emb, &shadow_meta("never use floats"))
                .unwrap();
            let promoted = store
                .corroborate_shadow("always use floats", &emb, &[])
                .unwrap();
            assert_eq!(promoted.is_some(), expected_promoted, "{mode:?}");
            assert_eq!(
                state_of(&store, id),
                if expected_promoted { "trial" } else { "shadow" },
                "{mode:?}"
            );
        }
    }
}

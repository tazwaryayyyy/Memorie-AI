use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, Connection};
use serde::Serialize;

use std::collections::{HashMap, HashSet};
use std::sync::RwLock;

use crate::error::{MemoireError, Result};
use crate::quality::{
    compute_trust, cosine_similarity, detect_polarity, effective_weight, now_ts, recency_bonus,
    NliChecker, NliLabel, Polarity, QualityMeta, ScoringConfig,
};

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub score: f32,
    pub trust: f32,
    /// Uncertainty ∈ [0, 1]: high when little reinforcement or when the memory
    /// has been both reinforced and penalized (oscillation). Agents may use this
    /// to modulate how strongly they act on a recall result.
    pub uncertainty: f32,
    pub state: String,
    pub created_at: i64,
    pub last_used_at: Option<i64>,
}

/// Trust and uncertainty delta produced by a single call to `penalize_if_used`.
#[derive(Debug, Clone, Serialize)]
pub struct PenaltyOutcome {
    pub id: i64,
    pub trust_before: f32,
    pub trust_after: f32,
    /// Uncertainty of this memory after the penalty, ∈ [0, 1].
    /// Increases as failure_count rises relative to reinforcement_count.
    pub uncertainty_after: f32,
}

pub struct Store {
    pool: Pool<SqliteConnectionManager>,
    inner: RwLock<StoreInner>,
    pub config: ScoringConfig,
    /// Namespace this store instance is scoped to. All reads/writes are filtered
    /// by this value. Defaults to `"default"`. Set via `Store::open_ns()`.
    pub namespace: String,
}

/// ✅ FIX #5 — In-memory HNSW embedding point wrapper.
///
/// Wraps a 384-dim embedding vector. Implements `instant_distance::Point`
/// using L2 distance. For L2-normalised fastembed vectors:
///   ||a − b||² = 2·(1 − cosine) → closer L2 = higher cosine.
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
    /// O(1) fingerprint lookup — avoids a SQLite round-trip per insert.
    fingerprints: HashSet<String>,
    /// Approximate nearest-neighbour index. Rebuilt lazily when dirty.
    hnsw: Option<instant_distance::HnswMap<EmbeddingPoint, i64>>,
    /// True after any insert or archive — triggers a lazy HNSW rebuild.
    hnsw_dirty: bool,
}

impl StoreInner {
    /// Load all embeddings from SQLite into the in-memory cache if not yet loaded.
    /// Fingerprints are loaded eagerly at construction; embeddings are deferred here
    /// to keep startup cost proportional to fingerprint count (small strings) rather
    /// than embedding count (1536 bytes × N).
    fn ensure_embedding_cache(&mut self, conn: &Connection, namespace: &str) -> Result<()> {
        if self.embedding_cache_loaded {
            return Ok(());
        }
        let mut stmt = conn.prepare(
            "SELECT id, embedding FROM memories
             WHERE namespace = ?1 AND archived = 0 AND superseded_by IS NULL",
        )?;
        for (id, blob) in stmt
            .query_map(rusqlite::params![namespace], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?
            .flatten()
        {
            if let Some(emb) = blob_to_vec(&blob) {
                self.embedding_cache.push((id, emb));
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
            self.hnsw_dirty = true;
        }
    }
}

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
    if !blob.len().is_multiple_of(4) {
        return None;
    }
    Some(
        blob.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

/// Load only fingerprints at startup — O(n × 64 bytes) instead of O(n × 1536 bytes).
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
) -> Vec<(i64, f32)> {
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
                .take(top_k * 2)
                .collect();
            if !results.is_empty() {
                return results;
            }
        }
    }

    let mut candidates: Vec<(i64, f32)> = inner
        .embedding_cache
        .iter()
        .map(|(id, emb)| (*id, cosine_similarity(query_vec, emb)))
        .collect();
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k * 2);
    candidates
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
        let pool = Pool::builder().max_size(20).build(manager)?;
        let conn = pool.get()?;
        conn.execute_batch(SCHEMA)?;
        Self::ensure_quality_columns(&conn)?;
        let fingerprints = load_fingerprints_from_db(&conn, namespace)?;
        log::debug!(
            "SQLite store opened at {path} ns={namespace}; {} fingerprints loaded (embeddings deferred)",
            fingerprints.len()
        );
        Ok(Self {
            pool,
            inner: RwLock::new(StoreInner {
                embedding_cache: Vec::new(),
                embedding_cache_loaded: false,
                fingerprints,
                hnsw: None,
                hnsw_dirty: false,
            }),
            config,
            namespace: namespace.to_string(),
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
        drop(conn);
        Ok(Self {
            pool,
            inner: RwLock::new(StoreInner {
                embedding_cache: Vec::new(),
                // In-memory store starts empty — no deferred load needed.
                embedding_cache_loaded: true,
                fingerprints: HashSet::new(),
                hnsw: None,
                hnsw_dirty: false,
            }),
            config,
            namespace: namespace.to_string(),
        })
    }

    pub fn insert(&self, content: &str, embedding: &[f32]) -> Result<i64> {
        let meta = QualityMeta::default_active(content);
        self.insert_with_quality(content, embedding, &meta)
    }

    pub fn reinforce_if_used(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
        output_embedding: Option<&[f32]>,
    ) -> Result<bool> {
        if !task_succeeded {
            return Ok(false);
        }

        let conn = self.pool.get()?;
        let row = conn.query_row(
            "SELECT content, embedding,
                    reinforcement_count, confidence, importance_base,
                    created_at, store_state, contradiction_group,
                    failure_count, trust_ema
             FROM memories WHERE id = ?1",
            params![memory_id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, f32>(3)?,
                    r.get::<_, f32>(4)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, String>(6)?,
                    r.get::<_, Option<String>>(7)?,
                    r.get::<_, i64>(8)?,
                    r.get::<_, Option<f32>>(9)?,
                ))
            },
        );

        let (
            content,
            mem_blob,
            rc,
            conf,
            imp,
            created_at,
            store_state,
            cg,
            failure_count,
            trust_ema_stored,
        ) = match row {
            Ok(v) => v,
            Err(_) => return Ok(false),
        };

        // Path 1: Jaccard token overlap (threshold raised to config.jaccard_threshold)
        let memory_tokens: HashSet<&str> = content.split_whitespace().collect();
        let output_tokens: HashSet<&str> = agent_output.split_whitespace().collect();
        let intersection = memory_tokens.intersection(&output_tokens).count();
        let union = memory_tokens.union(&output_tokens).count();
        let jaccard = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };
        let jaccard_ok = jaccard >= self.config.jaccard_threshold;

        // Path 2: cosine similarity
        let cosine_ok = if let Some(out_emb) = output_embedding {
            blob_to_vec(&mem_blob)
                .map(|mem_emb| cosine_similarity(&mem_emb, out_emb) >= 0.75)
                .unwrap_or(false)
        } else {
            false
        };

        if !jaccard_ok && !cosine_ok {
            log::warn!(
                "reinforce_if_used(id={memory_id}): task_succeeded=true but attribution \
                 gate not met (jaccard={jaccard:.3}, threshold={:.3}). Skipping.",
                self.config.jaccard_threshold
            );
            return Ok(false);
        }

        let new_rc = rc + 1;
        let new_conf = if failure_count > 0 {
            (conf * 1.1_f32).clamp(0.0, 1.0)
        } else {
            conf
        };
        let now: i64 =
            conn.query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                r.get(0)
            })?;
        let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
        let contradiction_survived = cg.is_some() && store_state == "active";
        let trust_after = compute_trust(
            new_rc,
            contradiction_survived,
            &store_state,
            imp,
            new_conf,
            age_days,
            &self.config,
        );
        let w = self.config.ema_new_weight;
        let new_ema = match trust_ema_stored {
            Some(ema) => ((1.0 - w) * ema + w * trust_after).clamp(0.0, 1.0),
            None => trust_after,
        };
        conn.execute(
            "UPDATE memories
             SET reinforcement_count = ?1,
                 confidence = ?2,
                 trust_ema = ?3,
                 last_accessed_at = CAST(strftime('%s', 'now') AS INTEGER)
             WHERE id = ?4",
            params![new_rc, new_conf, new_ema, memory_id],
        )?;
        Ok(true)
    }

    pub fn penalize_if_used(
        &self,
        memory_ids: &[i64],
        failure_severity: f32,
    ) -> Result<Vec<PenaltyOutcome>> {
        if memory_ids.is_empty() {
            return Ok(vec![]);
        }
        let severity = failure_severity.clamp(0.0, 1.0);
        let conn = self.pool.get()?;
        let now: i64 =
            conn.query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                r.get(0)
            })?;
        let mut outcomes = Vec::with_capacity(memory_ids.len());
        for &id in memory_ids {
            let row = conn.query_row(
                "SELECT reinforcement_count, confidence, importance_base,
                        created_at, store_state, contradiction_group,
                        failure_count, trust_ema
                 FROM memories WHERE id = ?1 AND archived = 0",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, f32>(1)?,
                        r.get::<_, f32>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, i64>(6)?,
                        r.get::<_, Option<f32>>(7)?,
                    ))
                },
            );
            let (rc, conf, imp, created_at, state, cg, failure_count, trust_ema_stored) = match row
            {
                Ok(v) => v,
                Err(_) => continue,
            };
            let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
            let contradiction_survived = cg.is_some() && state == "active";
            let trust_before = compute_trust(
                rc,
                contradiction_survived,
                &state,
                imp,
                conf,
                age_days,
                &self.config,
            );

            let new_rc = (rc - if severity > 0.0 { 1 } else { 0 }).max(0);
            let new_conf = (conf * (1.0 - 0.2 * severity)).clamp(0.0, 1.0);
            let new_imp = (imp * (1.0 - 0.1 * severity)).clamp(0.0, 1.0);
            let new_fc = failure_count + 1;

            let trust_after = compute_trust(
                new_rc,
                contradiction_survived,
                &state,
                new_imp,
                new_conf,
                age_days,
                &self.config,
            );
            let w = self.config.ema_new_weight;
            let new_ema = match trust_ema_stored {
                Some(ema) => ((1.0 - w) * ema + w * trust_after).clamp(0.0, 1.0),
                None => trust_after,
            };

            let rc_f = new_rc as f32;
            let fc_f = new_fc as f32;
            let uncertainty_after =
                (0.5 / (1.0 + rc_f) + 0.5 * fc_f / (fc_f + rc_f + 1.0)).clamp(0.0, 1.0);

            conn.execute(
                "UPDATE memories
                 SET reinforcement_count = ?1,
                     confidence = ?2,
                     importance_base = ?3,
                     failure_count = ?4,
                     trust_ema = ?5
                 WHERE id = ?6",
                params![new_rc, new_conf, new_imp, new_fc, new_ema, id],
            )?;

            outcomes.push(PenaltyOutcome {
                id,
                trust_before,
                trust_after,
                uncertainty_after,
            });
        }
        Ok(outcomes)
    }

    pub fn insert_with_quality(
        &self,
        content: &str,
        embedding: &[f32],
        quality: &QualityMeta,
    ) -> Result<i64> {
        let blob = vec_to_blob(embedding);
        // Clamp trust-related fields to valid ranges before writing so CHECK
        // constraints are never triggered by normal application logic.
        let confidence = quality.confidence.clamp(0.0_f32, 1.0_f32);
        let novelty = quality.novelty.clamp(0.0_f32, 1.0_f32);
        // Cold-start trust: seed trust_ema from quality score for new memories
        // so high-quality memories surface above HINT before any reinforcement.
        let initial_trust_ema = if quality.reinforcement_count == 0 {
            Some((quality.importance_base * self.config.cold_start_weight).clamp(0.0_f32, 1.0_f32))
        } else {
            None
        };
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
                trust_ema, last_used_at
            ) VALUES (
                ?1, ?2, ?3,
                ?4, ?5, ?6,
                ?7, ?8, ?9, ?10,
                ?11, ?12, ?13,
                ?14, ?15, ?16,
                ?17, ?18, ?19, ?20, ?21,
                ?22, ?23
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
                quality.store_state,
                quality.reinforcement_count,
                quality.last_accessed_at,
                quality.superseded_by,
                quality.contradiction_group,
                quality.archived,
                quality.effective_weight,
                quality.fingerprint,
                quality.claim_key,
                quality.claim_value,
                initial_trust_ema,
                now_ts_val,
            ],
        )?;
        let id = conn.last_insert_rowid();
        if quality.store_state != "rejected" && quality.archived == 0 {
            let mut inner = self.write_lock()?;
            // Only update the in-memory cache if it has already been loaded.
            // If not yet loaded, the lazy load will pick this row up from SQLite.
            if inner.embedding_cache_loaded {
                inner.embedding_cache.push((id, embedding.to_vec()));
                inner.hnsw_dirty = true;
            }
            inner.fingerprints.insert(quality.fingerprint.clone());
        }
        Ok(id)
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

    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<Memory>> {
        if top_k == 0 {
            return Ok(vec![]);
        }

        let conn = self.pool.get()?;
        if !self.prepare_search_cache(&conn)? {
            return Ok(vec![]);
        }

        let candidates = {
            let inner = self.read_lock()?;
            search_candidates_read(&inner, query_vec, top_k, &self.config)
        };
        let ns = self.namespace.clone();

        let now: i64 =
            conn.query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                r.get(0)
            })?;

        let mut scored: Vec<(f32, f32, Option<String>, Memory)> = Vec::new();
        for (id, sim) in &candidates {
            let row = conn.query_row(
                "SELECT content, created_at,
                        importance_base, reinforcement_count, confidence,
                        store_state, contradiction_group,
                        failure_count, trust_ema, last_used_at
                 FROM memories
                 WHERE id = ?1
                   AND namespace = ?2
                   AND archived = 0
                   AND superseded_by IS NULL
                   AND store_state IN ('active', 'shadow')",
                params![id, ns],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, f32>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, f32>(4)?,
                        r.get::<_, String>(5)?,
                        r.get::<_, Option<String>>(6)?,
                        r.get::<_, i64>(7)?,
                        r.get::<_, Option<f32>>(8)?,
                        r.get::<_, Option<i64>>(9)?,
                    ))
                },
            );
            let (
                content,
                created_at,
                importance_base,
                reinforcement_count,
                confidence,
                store_state,
                contradiction_group,
                failure_count,
                trust_ema_stored,
                last_used_at,
            ) = match row {
                Ok(v) => v,
                Err(_) => continue,
            };

            let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
            let weight = effective_weight(importance_base, age_days, reinforcement_count);
            let final_score = 0.75 * sim + 0.20 * weight + 0.05 * recency_bonus(created_at, now);

            let contradiction_survived = contradiction_group.is_some() && store_state == "active";
            let trust_base = compute_trust(
                reinforcement_count,
                contradiction_survived,
                &store_state,
                importance_base,
                confidence,
                age_days,
                &self.config,
            );
            let sim_clamped = sim.clamp(0.0, 1.0);
            let trust_computed = (trust_base * (0.6 + 0.4 * sim_clamped)).clamp(0.0, 1.0);
            let w = self.config.ema_new_weight;
            let trust_ema_blended = match trust_ema_stored {
                Some(ema) => ((1.0 - w) * ema + w * trust_computed).clamp(0.0, 1.0),
                None => trust_computed,
            };
            // Apply time decay based on days since last use
            let trust = if self.config.decay_rate > 0.0 {
                if let Some(last_used) = last_used_at {
                    let days_elapsed = ((now - last_used).max(0) as f32) / 86_400.0;
                    let decay_factor = (-self.config.decay_rate * days_elapsed).exp();
                    (trust_ema_blended * decay_factor).clamp(0.0, 1.0)
                } else {
                    trust_ema_blended
                }
            } else {
                trust_ema_blended
            };
            let rc_f = reinforcement_count as f32;
            let fc_f = failure_count as f32;
            let uncertainty =
                (0.5 / (1.0 + rc_f) + 0.5 * fc_f / (fc_f + rc_f + 1.0)).clamp(0.0, 1.0);

            scored.push((
                final_score,
                trust,
                contradiction_group,
                Memory {
                    id: *id,
                    content,
                    score: final_score.clamp(0.0, 1.0),
                    trust,
                    uncertainty,
                    state: store_state,
                    created_at,
                    last_used_at,
                },
            ));
        }

        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut group_winner: HashMap<String, usize> = HashMap::new();
        for (idx, (_, trust, cg_opt, _)) in scored.iter().enumerate() {
            if let Some(cg) = cg_opt {
                let entry = group_winner.entry(cg.clone()).or_insert(idx);
                if *trust > scored[*entry].1 {
                    *entry = idx;
                }
            }
        }
        let weak_groups: HashSet<String> = group_winner
            .iter()
            .filter(|(_, &idx)| scored[idx].1 < 0.4)
            .map(|(cg, _)| cg.clone())
            .collect();

        let result: Vec<Memory> = scored
            .into_iter()
            .enumerate()
            .filter(|(idx, (_, _, cg_opt, _))| {
                if let Some(cg) = cg_opt {
                    if weak_groups.contains(cg) {
                        return false;
                    }
                    group_winner
                        .get(cg)
                        .map(|best| best == idx)
                        .unwrap_or(false)
                } else {
                    true
                }
            })
            .map(|(_, (_, _, _, m))| m)
            .take(top_k)
            .collect();

        // Update last_used_at for all returned memories in a single batch
        if !result.is_empty() {
            let ids: Vec<String> = result.iter().map(|m| m.id.to_string()).collect();
            let placeholders = ids.join(",");
            conn.execute_batch(&format!(
                "UPDATE memories SET last_used_at = {now} WHERE id IN ({placeholders});",
                now = now
            ))?;
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
        let pool = self.search(query_vec, (top_k * 4).max(20))?;
        Ok(pool
            .into_iter()
            .filter(|m| m.created_at >= cutoff_ts)
            .take(top_k)
            .collect())
    }

    /// NLI-enhanced semantic contradiction detection.
    ///
    /// When `config.use_nli_contradiction` is true (default), uses `NliChecker`
    /// which applies a three-signal ensemble:
    ///   1. Cosine similarity gate (same topic cluster, threshold from config)
    ///   2. Polarity opposition (lexical negation)
    ///   3. Negation asymmetry score (how many negators appear in one but not the other)
    ///
    /// When `use_nli_contradiction` is false, falls back to the original
    /// pure polarity-based gate for backward compatibility.
    pub fn resolve_contradictions_for_id(&self, id: i64) -> Result<()> {
        let conn = self.pool.get()?;
        let mut inner = self.write_lock()?;
        let ns = self.namespace.clone();

        let target_row = conn.query_row(
            "SELECT content, embedding, importance_base, confidence, evidence, created_at, claim_value
             FROM memories WHERE id = ?1 AND namespace = ?2 AND archived = 0",
            params![id, ns],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, f32>(2)?,
                    r.get::<_, f32>(3)?,
                    r.get::<_, f32>(4)?,
                    r.get::<_, i64>(5)?,
                    r.get::<_, Option<String>>(6)?,
                ))
            },
        );

        let (new_content, new_blob, imp, conf, ev, created_at, stored_polarity) = match target_row {
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
            "SELECT id, content, embedding, importance_base, confidence, evidence, created_at, claim_value
             FROM memories
             WHERE id != ?1
               AND namespace = ?2
               AND archived = 0
               AND superseded_by IS NULL
               AND store_state IN ('active', 'shadow')",
        )?;

        // Build NLI checker once (zero-cost: stateless struct).
        let use_nli = self.config.use_nli_contradiction;
        let nli_threshold = self.config.nli_cosine_threshold;
        let nli = NliChecker::new();

        let mut contradictions: Vec<(i64, f32, f32, f32, i64)> = Vec::new();
        for row in stmt.query_map(params![id, ns], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Vec<u8>>(2)?,
                r.get::<_, f32>(3)?,
                r.get::<_, f32>(4)?,
                r.get::<_, f32>(5)?,
                r.get::<_, i64>(6)?,
                r.get::<_, Option<String>>(7)?,
            ))
        })? {
            let (
                other_id,
                other_content,
                other_blob,
                o_imp,
                o_conf,
                o_ev,
                o_created,
                other_polarity_opt,
            ) = match row {
                Ok(v) => v,
                Err(_) => continue,
            };

            let other_emb = match blob_to_vec(&other_blob) {
                Some(v) => v,
                None => continue,
            };

            let is_contradiction = if use_nli {
                // NLI path: three-signal ensemble.
                let label = nli.check(
                    &new_content,
                    &other_content,
                    &new_emb,
                    &other_emb,
                    nli_threshold,
                );
                label == NliLabel::Contradiction
            } else {
                // Legacy path: cosine + polarity only.
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
                new_pol.opposes(&other_pol)
            };

            if !is_contradiction {
                // For the NLI path we need to check the polarity explicitly for legacy compat.
                if !use_nli {
                    continue;
                }
                // NLI returned Neutral or Entailment — skip.
                continue;
            }

            contradictions.push((other_id, o_imp, o_conf, o_ev, o_created));
        }

        if contradictions.is_empty() {
            return Ok(());
        }

        drop(stmt);

        let rec_new = recency_bonus(created_at, now);
        let q_new = 0.45 * imp + 0.25 * conf + 0.20 * rec_new + 0.10 * ev;

        for (other_id, o_imp, o_conf, o_ev, o_created) in contradictions {
            let rec_old = recency_bonus(o_created, now);
            let q_old = 0.45 * o_imp + 0.25 * o_conf + 0.20 * rec_old + 0.10 * o_ev;

            let (winner_id, loser_id) = if q_new >= q_old {
                (id, other_id)
            } else {
                (other_id, id)
            };

            let group = winner_id.to_string();
            conn.execute(
                "UPDATE memories SET superseded_by = ?1, archived = 1, contradiction_group = ?2
                 WHERE id = ?3",
                params![winner_id, group, loser_id],
            )?;
            // Do NOT set contradiction_group on the winner. The loser already points to the
            // winner via superseded_by. Setting contradiction_group on the winner would put
            // it into the weak-group deduplication filter in search(), which suppresses
            // memories whose best-survivor trust < 0.40 — incorrectly hiding the winner.
            inner.remove_from_cache(loser_id);
            log::debug!(
                "contradiction resolved ({}): winner={winner_id} loser={loser_id} group={group}",
                if use_nli { "nli" } else { "polarity" }
            );
        }

        Ok(())
    }

    pub fn maintenance_pass(&self) -> Result<()> {
        let conn = self.pool.get()?;
        let mut inner = self.write_lock()?;

        conn.execute_batch(
            "UPDATE memories
             SET archived = 1
             WHERE superseded_by IS NOT NULL
               AND archived = 0;",
        )?;

        let mut stmt = conn.prepare(
            "SELECT id FROM memories
             WHERE archived = 1
               AND (strftime('%s', 'now') - created_at) > 7 * 86400
             UNION ALL
             SELECT id FROM memories
             WHERE archived = 0
               AND superseded_by IS NULL
               AND reinforcement_count = 0
               AND importance_base < 0.12
               AND (strftime('%s', 'now') - created_at) > 30 * 86400",
        )?;
        let to_delete: Vec<i64> = stmt
            .query_map([], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);

        for del_id in to_delete {
            conn.execute("DELETE FROM memories WHERE id = ?1", params![del_id])?;
            inner.remove_from_cache(del_id);
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
            "SELECT id, content, created_at,
                    importance_base, reinforcement_count, confidence,
                    store_state, contradiction_group,
                    failure_count, trust_ema
             FROM memories WHERE namespace = ?1 ORDER BY created_at DESC",
        )?;

        let config = &self.config;
        let rows = stmt
            .query_map(rusqlite::params![ns], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f32>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, f32>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, Option<f32>>(9)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(
                |(
                    id,
                    content,
                    created_at,
                    imp,
                    rc,
                    conf,
                    state,
                    cg,
                    failure_count,
                    trust_ema_stored,
                )| {
                    let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
                    let contradiction_survived = cg.is_some() && state == "active";
                    let trust_computed = compute_trust(
                        rc,
                        contradiction_survived,
                        &state,
                        imp,
                        conf,
                        age_days,
                        config,
                    );
                    let w = config.ema_new_weight;
                    let trust = match trust_ema_stored {
                        Some(ema) => ((1.0 - w) * ema + w * trust_computed).clamp(0.0, 1.0),
                        None => trust_computed,
                    };
                    let rc_f = rc as f32;
                    let fc_f = failure_count as f32;
                    let uncertainty =
                        (0.5 / (1.0 + rc_f) + 0.5 * fc_f / (fc_f + rc_f + 1.0)).clamp(0.0, 1.0);
                    Memory {
                        id,
                        content,
                        score: 1.0,
                        trust,
                        uncertainty,
                        state,
                        created_at,
                        last_used_at: None,
                    }
                },
            )
            .collect();
        Ok(rows)
    }

    pub fn forget(&self, id: i64) -> Result<bool> {
        let conn = self.pool.get()?;
        let mut inner = self.write_lock()?;
        let rows = conn.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        if rows > 0 {
            inner.remove_from_cache(id);
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
        conn.execute_batch("DELETE FROM memories;")?;
        inner.embedding_cache.clear();
        inner.embedding_cache_loaded = true; // empty store is coherent — no load needed
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
             WHERE namespace = ?1 AND archived = 0"
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
        let conn = self.pool.get()?;
        let placeholders = ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let query = format!(
            "UPDATE memories SET trust_ema = ?1, reinforcement_count = ?2, importance_base = ?3, confidence = ?4 {} WHERE id IN ({})",
            if created_at.is_some() { ", created_at = ?5" } else { "" },
            placeholders
        );
        if let Some(created) = created_at {
            conn.execute(
                &query,
                params![
                    trust_ema,
                    reinforcement_count,
                    importance_base,
                    confidence,
                    created
                ],
            )?;
        } else {
            conn.execute(
                &query,
                params![trust_ema, reinforcement_count, importance_base, confidence],
            )?;
        }
        Ok(())
    }

    /// Maximal Marginal Relevance reranking.
    ///
    /// Selects up to `top_k` results from `candidates` that maximise relevance
    /// to `query_embedding` while minimising redundancy with already-selected items.
    ///
    /// `lambda = 1.0` → pure relevance (identical to `recall`).
    /// `lambda = 0.0` → pure diversity.
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
        // Namespace column — added in v0.3.0. Existing rows get 'default'.
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
        let store = Store::in_memory().unwrap();
        store.insert("memory one", &[1.0_f32, 0.0, 0.0]).unwrap();
        store.insert("memory two", &[0.0_f32, 1.0, 0.0]).unwrap();
        let results = store.search(&[0.9_f32, 0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "memory one");
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

        assert_eq!(
            store.count().unwrap(),
            1,
            "contradiction resolution must archive the loser"
        );
    }

    #[test]
    fn test_scores_descending() {
        let store = Store::in_memory().unwrap();
        store.insert("closest", &[1.0_f32, 0.0, 0.0]).unwrap();
        store.insert("middle", &[0.7_f32, 0.7, 0.0]).unwrap();
        store.insert("farthest", &[0.0_f32, 1.0, 0.0]).unwrap();
        let results = store.search(&[1.0, 0.0, 0.0], 3).unwrap();
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

    // ─── Failure-feedback / penalize_if_used tests ────────────────────────────

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
            .reinforce_if_used(id, "decimal money correct precision usage", true, None)
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
        // Same embedding so cosine sim is equal — ranking separates via effective_weight
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

        // Apply 3 failures to the bad memory — rc: 3→2→1→0, conf/imp decay each round
        for _ in 0..3 {
            store.penalize_if_used(&[bad_id], 1.0).unwrap();
        }

        let results = store.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].content, "decimal is correct for money",
            "good memory must outrank penalized bad memory after 3 failures"
        );
    }

    // ─── Namespace isolation tests ────────────────────────────────────────────

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

    /// Shared SQLite file — two Store handles with different namespaces must
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
            let found = store_a.search(&[0.0_f32, 1.0], 5).unwrap();
            assert!(
                found.iter().all(|m| m.content != "tenant_b secret"),
                "tenant_a search must not return tenant_b's memory"
            );
        }

        // Cleanup — ignore errors (WAL files on Windows)
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{path_str}-wal"));
        let _ = std::fs::remove_file(format!("{path_str}-shm"));
    }
}

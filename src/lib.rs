pub mod chunker;
pub mod embedder;
pub mod error;
pub mod ffi;
pub mod quality;
pub mod store;

use chunker::{chunk_text, ChunkerConfig};
use embedder::Embedder;
use error::{MemoireError, Result};
use quality::{build_quality_meta, fingerprint, IngestDecision};
use store::{Memory, Store};

/// The central Memoire instance.
///
/// # Example
///
/// ```rust,no_run
/// use memoire::Memoire;
///
/// let m = Memoire::new("agent_memory.db").unwrap();
/// m.remember("Fixed the off-by-one error in pagination on 2024-01-15").unwrap();
///
/// let results = m.recall("what bugs did I fix?", 3).unwrap();
/// for r in results {
///     println!("[{:.3}] {}", r.score, r.content);
/// }
/// ```
pub struct Memoire {
    store: Store,
    embedder: Embedder,
    chunker_config: ChunkerConfig,
}

impl Memoire {
    /// Open or create a persistent memory store at `db_path`.
    pub fn new(db_path: &str) -> Result<Self> {
        let _ = env_logger::try_init();
        Ok(Self {
            store: Store::open(db_path)?,
            embedder: Embedder::new().map_err(MemoireError::Embedding)?,
            chunker_config: ChunkerConfig::default(),
        })
    }

    /// In-memory store — not persisted. Useful for tests.
    pub fn in_memory() -> Result<Self> {
        let _ = env_logger::try_init();
        Ok(Self {
            store: Store::in_memory()?,
            embedder: Embedder::new().map_err(MemoireError::Embedding)?,
            chunker_config: ChunkerConfig::default(),
        })
    }

    pub fn with_chunker_config(mut self, config: ChunkerConfig) -> Self {
        self.chunker_config = config;
        self
    }

    /// Chunk `content`, embed each chunk, and persist. Returns inserted row ids.
    pub fn remember(&self, content: &str) -> Result<Vec<i64>> {
        self.remember_with_source(content, "agent")
    }

    /// Chunk `content`, embed each chunk, and persist with `source_kind` metadata.
    pub fn remember_with_source(&self, content: &str, source_kind: &str) -> Result<Vec<i64>> {
        let chunks = chunk_text(content, &self.chunker_config);
        if chunks.is_empty() {
            log::warn!("remember() called with empty input — nothing stored.");
            return Ok(vec![]);
        }
        log::debug!("remember(): {} chunk(s)", chunks.len());

        let embeddings = self
            .embedder
            .embed(chunks.clone())
            .map_err(MemoireError::Embedding)?;

        let mut ids = Vec::with_capacity(chunks.len());
        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let fp = fingerprint(chunk);
            if self.store.fingerprint_exists(&fp)? {
                continue;
            }
            let max_sim = self.store.max_similarity(embedding)?;
            let novelty = (1.0 - max_sim).clamp(0.0, 1.0);
            let (meta, decision) = build_quality_meta(chunk, novelty, source_kind);

            match decision {
                IngestDecision::Reject => continue,
                IngestDecision::Shadow | IngestDecision::Active => {
                    let id = self.store.insert_with_quality(chunk, embedding, &meta)?;
                    self.store.resolve_contradictions_for_id(id)?;
                    ids.push(id);
                }
            }
        }
        self.store.maintenance_pass()?;
        log::debug!("remember(): stored ids: {:?}", ids);
        Ok(ids)
    }

    /// Embed `query` and return the `top_k` most similar memories.
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<Memory>> {
        if self.store.count()? == 0 {
            return Ok(vec![]);
        }
        log::debug!("recall(): {:?} top_k={}", query, top_k);
        let query_vec = self
            .embedder
            .embed_one(query)
            .map_err(MemoireError::Embedding)?;
        self.store.search(&query_vec, top_k)
    }

    /// Recall with a minimum score threshold after ranking.
    pub fn recall_with_min_score(
        &self,
        query: &str,
        top_k: usize,
        min_score: f32,
    ) -> Result<Vec<Memory>> {
        let mut results = self.recall(query, top_k)?;
        results.retain(|m| m.score >= min_score);
        Ok(results)
    }

    /// Reinforce a memory only when it was actually used by the agent.
    ///
    /// Fires when: `task_succeeded` AND (
    ///   Jaccard token-overlap between memory content and `agent_output` >= 15%
    ///   OR cosine(memory_embedding, embed(agent_output)) >= 0.75
    /// )
    ///
    /// The cosine path catches semantic paraphrase where wording differs.
    /// Returns `true` if reinforcement was applied.
    pub fn reinforce_if_used(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
    ) -> Result<bool> {
        // Embed the agent output so the cosine path can fire on paraphrase
        let output_embedding = self.embedder.embed_one(agent_output).ok();
        self.store.reinforce_if_used(
            memory_id,
            agent_output,
            task_succeeded,
            output_embedding.as_deref(),
        )
    }

    /// Penalize memories that contributed to a failed task outcome.
    ///
    /// `failure_severity` ∈ [0.0, 1.0] — scales how harshly the memories are
    /// penalized. Use 1.0 for a direct failure, 0.5 for a partial miss.
    /// Symmetric with `reinforce_if_used`. Returns the trust delta per memory.
    pub fn penalize_if_used(
        &self,
        memory_ids: &[i64],
        failure_severity: f32,
    ) -> Result<Vec<store::PenaltyOutcome>> {
        self.store.penalize_if_used(memory_ids, failure_severity)
    }

    /// Delete a memory by id. Returns true if it existed.
    pub fn forget(&self, id: i64) -> Result<bool> {
        self.store.forget(id)
    }

    /// Return every stored memory ordered by recency. Score is 1.0 (no query).
    pub fn export_all(&self) -> Result<Vec<store::Memory>> {
        self.store.all()
    }

    /// Total stored memory chunks.
    pub fn count(&self) -> Result<i64> {
        self.store.count()
    }

    /// Erase ALL memories. Irreversible.
    pub fn clear(&self) -> Result<()> {
        self.store.clear()
    }

    /// Run a quality-control maintenance pass (pruning + archive enforcement).
    pub fn maintenance_pass(&self) -> Result<()> {
        self.store.maintenance_pass()
    }
}

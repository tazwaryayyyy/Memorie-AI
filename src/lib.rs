pub mod chunker;
pub mod embedder;
pub mod error;
pub mod ffi;
#[cfg(feature = "pyo3")]
pub mod py_ffi;
pub mod quality;
pub mod store;

use chunker::{chunk_text, ChunkerConfig};
use embedder::{EmbedProvider, Embedder, FastEmbedReranker, Reranker};
use error::{MemoireError, Result};
use quality::{
    build_quality_meta, fingerprint, IngestDecision, ScoringConfig, ScoringPrototypes,
    ScoringWeights,
};
use std::sync::{Arc, Mutex};
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
    pub(crate) store: Store,
    embedder: Box<dyn EmbedProvider>,
    reranker: Option<Arc<Mutex<Box<dyn Reranker>>>>,
    chunker_config: ChunkerConfig,
    scoring_weights: ScoringWeights,
    prototypes: ScoringPrototypes,
}

impl Memoire {
    // ─── Constructors ────────────────────────────────────────────────────────

    /// Open or create a persistent memory store at `db_path`.
    pub fn new(db_path: &str) -> Result<Self> {
        let _ = env_logger::try_init();
        let embedder: Box<dyn EmbedProvider> =
            Box::new(Embedder::new().map_err(MemoireError::Embedding)?);
        Self::new_with_embedder(db_path, embedder)
    }

    /// Open a persistent store scoped to `namespace`.
    ///
    /// Multiple agents can share one SQLite file with hard isolation:
    /// ```rust,no_run
    /// # use memoire::Memoire;
    /// let agent_a = Memoire::new_ns("shared.db", "agent_a").unwrap();
    /// let agent_b = Memoire::new_ns("shared.db", "agent_b").unwrap();
    /// // agent_a.recall(…) never returns agent_b's memories and vice-versa.
    /// ```
    pub fn new_ns(db_path: &str, namespace: &str) -> Result<Self> {
        let _ = env_logger::try_init();
        let embedder: Box<dyn EmbedProvider> =
            Box::new(Embedder::new().map_err(MemoireError::Embedding)?);
        Self::new_ns_with_embedder(db_path, namespace, embedder)
    }

    /// Open a persistent store with a custom embedding backend.
    ///
    /// Allows swapping `all-MiniLM-L6-v2` for any model — BERT-large, an
    /// OpenAI API wrapper, or a proprietary encoder — without recompiling.
    /// Implement [`embedder::EmbedProvider`] and pass the backend here.
    pub fn new_with_embedder(db_path: &str, embedder: Box<dyn EmbedProvider>) -> Result<Self> {
        Self::new_ns_with_embedder(db_path, "default", embedder)
    }

    pub fn new_ns_with_embedder(
        db_path: &str,
        namespace: &str,
        embedder: Box<dyn EmbedProvider>,
    ) -> Result<Self> {
        let prototypes = Self::compute_prototypes(&*embedder);
        Ok(Self {
            store: Store::open_ns_with_config(db_path, namespace, ScoringConfig::default())?,
            embedder,
            reranker: None,
            chunker_config: ChunkerConfig::default(),
            scoring_weights: ScoringWeights::default(),
            prototypes,
        })
    }

    /// In-memory store — not persisted. Useful for tests.
    pub fn in_memory() -> Result<Self> {
        let _ = env_logger::try_init();
        let embedder: Box<dyn EmbedProvider> =
            Box::new(Embedder::new().map_err(MemoireError::Embedding)?);
        Self::in_memory_with_embedder(embedder)
    }

    /// In-memory store scoped to `namespace`.
    pub fn in_memory_ns(namespace: &str) -> Result<Self> {
        let _ = env_logger::try_init();
        let embedder: Box<dyn EmbedProvider> =
            Box::new(Embedder::new().map_err(MemoireError::Embedding)?);
        Self::in_memory_ns_with_embedder(namespace, embedder)
    }

    /// In-memory store with a custom embedding backend.
    ///
    /// Useful for deterministic unit tests without initialising the ONNX model.
    pub fn in_memory_with_embedder(embedder: Box<dyn EmbedProvider>) -> Result<Self> {
        Self::in_memory_ns_with_embedder("default", embedder)
    }

    pub fn in_memory_ns_with_embedder(
        namespace: &str,
        embedder: Box<dyn EmbedProvider>,
    ) -> Result<Self> {
        let _ = env_logger::try_init();
        let prototypes = Self::compute_prototypes(&*embedder);
        Ok(Self {
            store: Store::in_memory_ns_with_config(namespace, ScoringConfig::default())?,
            embedder,
            reranker: None,
            chunker_config: ChunkerConfig::default(),
            scoring_weights: ScoringWeights::default(),
            prototypes,
        })
    }

    // ─── Introspection ───────────────────────────────────────────────────────

    /// Return the namespace this instance is scoped to.
    pub fn namespace(&self) -> &str {
        &self.store.namespace
    }

    // ─── Builders ────────────────────────────────────────────────────────────

    /// Override the scoring config used by the store.
    pub fn with_scoring_config(mut self, config: ScoringConfig) -> Self {
        self.store.config = config;
        self
    }

    pub fn with_chunker_config(mut self, config: ChunkerConfig) -> Self {
        self.chunker_config = config;
        self
    }

    /// Enable cross-encoder reranking for `recall_reranked`.
    ///
    /// This is opt-in because the reranker downloads and initialises a separate
    /// model on first construction.
    pub fn with_reranker(mut self) -> Result<Self> {
        self.reranker = Some(Arc::new(Mutex::new(Box::new(FastEmbedReranker::new()?))));
        Ok(self)
    }

    /// Override the default scoring weights used at ingestion time.
    ///
    /// The `ScoringWeights::default()` profile is frozen for reproducibility
    /// and is the baseline for all benchmarks. Use this to tune for your domain
    /// without recompiling — e.g. raise `novelty` to 0.40 for research agents
    /// where unique facts matter more than actionability.
    ///
    /// Does NOT affect the trust formula (computed at recall time from
    /// reinforcement history). Only affects which memories reach `Active`
    /// vs `Shadow` at write time.
    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }

    // ─── Core API ────────────────────────────────────────────────────────────

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
            let (meta, decision) = build_quality_meta(
                chunk,
                embedding,
                novelty,
                source_kind,
                &self.scoring_weights,
                &self.prototypes,
            );

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

    /// Recall but only consider memories created within the last `max_age_days` days.
    ///
    /// Addresses **semantic drift**: a memory accurate for an older library
    /// version should not surface when the API has since changed. This is a
    /// read-only recency gate — stale memories are skipped but their trust
    /// scores are NOT modified. Use `penalize_if_used` only when the memory
    /// was genuinely wrong, not merely outdated.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use memoire::Memoire;
    /// # let m = Memoire::in_memory().unwrap();
    /// // Surface only the last 30 days of lessons about the pandas API.
    /// let mems = m.recall_within_days("pandas pivot API", 5, 30.0)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn recall_within_days(
        &self,
        query: &str,
        top_k: usize,
        max_age_days: f32,
    ) -> Result<Vec<Memory>> {
        if self.store.count()? == 0 {
            return Ok(vec![]);
        }
        let query_vec = self
            .embedder
            .embed_one(query)
            .map_err(MemoireError::Embedding)?;
        self.store
            .search_within_days(&query_vec, top_k, max_age_days)
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

    /// Total stored memory chunks (namespace-scoped).
    pub fn count(&self) -> Result<i64> {
        self.store.count()
    }

    /// Erase ALL memories in this namespace. Irreversible.
    pub fn clear(&self) -> Result<()> {
        self.store.clear()
    }

    /// Run a quality-control maintenance pass (pruning + archive enforcement).
    pub fn maintenance_pass(&self) -> Result<()> {
        self.store.maintenance_pass()
    }

    /// Recall with Maximal Marginal Relevance reranking to reduce redundancy.
    ///
    /// Retrieves `top_k * 3` candidates (min 20) then reranks using MMR so
    /// near-duplicate chunks do not dominate the top-k slots.
    ///
    /// `mmr_lambda = 1.0` → identical to `recall` (pure relevance).
    /// `mmr_lambda = 0.5` (default) → balanced relevance/diversity.
    pub fn recall_mmr(&self, query: &str, top_k: usize, mmr_lambda: f32) -> Result<Vec<Memory>> {
        if self.store.count()? == 0 {
            return Ok(vec![]);
        }
        let candidate_k = (top_k * 3).max(20);
        let query_vec = self
            .embedder
            .embed_one(query)
            .map_err(MemoireError::Embedding)?;
        let candidates = self.store.search(&query_vec, candidate_k)?;
        Ok(self
            .store
            .mmr_rerank(candidates, &query_vec, top_k, mmr_lambda))
    }

    /// Recall with an optional cross-encoder reranking pass.
    ///
    /// Without `with_reranker()`, this falls back to regular recall candidates.
    pub fn recall_reranked(&self, query: &str, top_k: usize) -> Result<Vec<Memory>> {
        if self.store.count()? == 0 || top_k == 0 {
            return Ok(vec![]);
        }

        let candidate_k = (top_k * 4).max(20);
        let candidates = self.recall(query, candidate_k)?;
        if let Some(reranker) = &self.reranker {
            let texts: Vec<&str> = candidates.iter().map(|m| m.content.as_str()).collect();
            let scores = reranker
                .lock()
                .map_err(|_| MemoireError::LockPoisoned)?
                .rerank(query, &texts)?;
            let mut scored: Vec<(f32, Memory)> = scores.into_iter().zip(candidates).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            Ok(scored.into_iter().take(top_k).map(|(_, m)| m).collect())
        } else {
            Ok(candidates.into_iter().take(top_k).collect())
        }
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    fn compute_prototypes(embedder: &dyn EmbedProvider) -> ScoringPrototypes {
        let consequence_sents = [
            "Critical security breach in production database exposing user data".to_string(),
            "System outage caused data loss for all users".to_string(),
            "Race condition in payment processing leads to double charges".to_string(),
            "Authentication bypass vulnerability allows unauthorized access".to_string(),
            "Production database corrupted during migration".to_string(),
        ];
        let actionability_sents = [
            "Fixed the bug by replacing the function with the corrected version".to_string(),
            "Replaced deprecated API call with the new stable endpoint".to_string(),
            "Patched the vulnerability by updating the dependency to latest version".to_string(),
            "Refactored the module to remove dead code and reduce coupling".to_string(),
            "Disabled the feature flag after rollback to restore stable behavior".to_string(),
        ];
        let reusability_sents = [
            "Always use Decimal for financial calculations never use float".to_string(),
            "Never store passwords in plain text always hash with bcrypt".to_string(),
            "API rate limit policy must enforce maximum requests per hour".to_string(),
            "Rule: all database queries must use parameterized statements".to_string(),
            "Mandatory: all external inputs must be validated and sanitized".to_string(),
        ];

        let all: Vec<String> = consequence_sents
            .iter()
            .chain(actionability_sents.iter())
            .chain(reusability_sents.iter())
            .cloned()
            .collect();

        match embedder.embed(all) {
            Ok(vecs) if vecs.len() == 15 => {
                let avg = |group: &[Vec<f32>]| -> Vec<f32> {
                    let dim = group[0].len();
                    let mut acc = vec![0.0_f32; dim];
                    for v in group {
                        for (a, b) in acc.iter_mut().zip(v.iter()) {
                            *a += b;
                        }
                    }
                    acc.iter_mut().for_each(|x| *x /= group.len() as f32);
                    acc
                };
                let consequence = avg(&vecs[0..5]);
                let actionability = avg(&vecs[5..10]);
                let reusability = avg(&vecs[10..15]);
                ScoringPrototypes {
                    consequence,
                    actionability,
                    reusability,
                    is_semantic: true,
                }
            }
            Err(e) => {
                log::warn!("compute_prototypes failed — falling back to neutral: {e}");
                ScoringPrototypes::neutral(384)
            }
            Ok(_) => {
                log::warn!(
                    "compute_prototypes returned unexpected count — falling back to neutral"
                );
                ScoringPrototypes::neutral(384)
            }
        }
    }
}

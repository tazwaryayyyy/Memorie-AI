#[cfg(not(target_arch = "wasm32"))]
pub mod background;
pub mod chunker;
#[cfg(not(target_arch = "wasm32"))]
pub mod embedder;
pub mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;
#[cfg(not(target_arch = "wasm32"))]
pub mod panic_guard;
#[cfg(all(feature = "pyo3", not(target_arch = "wasm32")))]
pub mod py_ffi;
pub mod quality;
#[cfg(not(target_arch = "wasm32"))]
pub mod store;
pub mod text;

#[cfg(not(target_arch = "wasm32"))]
use chunker::{chunk_text, ChunkerConfig};
#[cfg(not(target_arch = "wasm32"))]
use embedder::{EmbedProvider, Embedder, FastEmbedReranker, Reranker};
#[cfg(not(target_arch = "wasm32"))]
use error::{MemoireError, Result};
#[cfg(not(target_arch = "wasm32"))]
use quality::{
    build_quality_meta, fingerprint, ingest_state, AdmissionMode, IngestDecision, IngestionOracle,
    ScoringConfig, ScoringPrototypes, ScoringWeights,
};
#[cfg(not(target_arch = "wasm32"))]
use serde::Serialize;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use store::{Memory, Store};

// â”€â”€â”€ WASM re-exports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// On wasm32, only the pure-Rust quality and NLI functions are available.
// Agents running in a browser/WASM environment can import these to score
// and detect contradictions in memory text without SQLite or ONNX.
#[cfg(target_arch = "wasm32")]
pub use quality::{cosine_similarity, detect_polarity, NliChecker, NliLabel, ScoringConfig};

/// How much of a snapshot's per-memory metadata an import may carry over.
///
/// A snapshot is data from outside the trust boundary: it can come from an
/// agent, a downloaded file or a prompt-injected tool call. Trust in Memoire is
/// meant to be *earned locally* through outcome feedback, so by default an
/// import re-ingests content only.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportTrust {
    /// Re-ingest `content` through the normal `remember()` path and ignore every
    /// metadata field in the snapshot (`trust_ema`, `reinforcement_count`,
    /// `importance_base`, `confidence`, `created_at`). An imported memory is then
    /// no more trusted than one stored with `remember()`. This is the default.
    #[default]
    Reset,
    /// Restore the snapshot's metadata, clamped to explicit bounds (see
    /// [`store::Store::update_imported_metadata`]). Meant for restoring your own
    /// backup. "Bounded" is not "verified": a hostile snapshot can still claim the
    /// maximum in-range trust, so never use this for snapshots you did not export.
    PreserveBounded,
}

/// The central Memoire instance.
///
/// Not available on `wasm32` targets â€” use the quality module directly for
/// scoring and NLI contradiction detection in browser environments.
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
#[cfg(not(target_arch = "wasm32"))]
pub struct Memoire {
    pub(crate) store: Store,
    embedder: Arc<dyn EmbedProvider>,
    reranker: Option<Arc<Mutex<Box<dyn Reranker>>>>,
    chunker_config: ChunkerConfig,
    scoring_weights: ScoringWeights,
    prototypes: Arc<Mutex<ScoringPrototypes>>,
    oracle: Option<Arc<dyn IngestionOracle>>,
    /// Background task handles (time scheduler, insert watcher, reinforcement
    /// watcher). Aborted individually: dropping a `JoinHandle` only detaches.
    maintenance_handles: Vec<tokio::task::JoinHandle<()>>,
    /// Private runtime, created only when `start_background_maintenance` is
    /// called with no ambient Tokio runtime (e.g. from Python / the MCP server).
    maintenance_runtime: Option<tokio::runtime::Runtime>,
    insert_count: Arc<AtomicU32>,
    reinforcement_count: Arc<AtomicU32>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Serialize)]
pub struct RecallAttribution {
    pub memory: Memory,
    pub semantic_score: f32,
    pub trust: f32,
    pub uncertainty: f32,
    pub effective_weight: f32,
    pub rank_score: f32,
    pub reasons: Vec<String>,
}
#[cfg(not(target_arch = "wasm32"))]
impl Drop for Memoire {
    fn drop(&mut self) {
        self.stop_background_maintenance();
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Memoire {
    // â”€â”€â”€ Constructors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    /// // agent_a.recall(â€¦) never returns agent_b's memories and vice-versa.
    /// ```
    pub fn new_ns(db_path: &str, namespace: &str) -> Result<Self> {
        let _ = env_logger::try_init();
        let embedder: Box<dyn EmbedProvider> =
            Box::new(Embedder::new().map_err(MemoireError::Embedding)?);
        Self::new_ns_with_embedder(db_path, namespace, embedder)
    }

    /// Open a persistent store with a custom embedding backend.
    ///
    /// Allows swapping `all-MiniLM-L6-v2` for any model â€” BERT-large, an
    /// OpenAI API wrapper, or a proprietary encoder â€” without recompiling.
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
        let embedder: Arc<dyn EmbedProvider> = Arc::from(embedder);
        Ok(Self {
            store: Store::open_ns_with_config(db_path, namespace, ScoringConfig::default())?,
            embedder,
            reranker: None,
            chunker_config: ChunkerConfig::default(),
            scoring_weights: ScoringWeights::default(),
            prototypes: Arc::new(Mutex::new(prototypes)),
            oracle: None,
            maintenance_handles: Vec::new(),
            maintenance_runtime: None,
            insert_count: Arc::new(AtomicU32::new(0)),
            reinforcement_count: Arc::new(AtomicU32::new(0)),
        })
    }

    /// In-memory store â€” not persisted. Useful for tests.
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
        let embedder: Arc<dyn EmbedProvider> = Arc::from(embedder);
        Ok(Self {
            store: Store::in_memory_ns_with_config(namespace, ScoringConfig::default())?,
            embedder,
            reranker: None,
            chunker_config: ChunkerConfig::default(),
            scoring_weights: ScoringWeights::default(),
            prototypes: Arc::new(Mutex::new(prototypes)),
            oracle: None,
            maintenance_handles: Vec::new(),
            maintenance_runtime: None,
            insert_count: Arc::new(AtomicU32::new(0)),
            reinforcement_count: Arc::new(AtomicU32::new(0)),
        })
    }

    // â”€â”€â”€ Introspection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Return the namespace this instance is scoped to.
    pub fn namespace(&self) -> &str {
        &self.store.namespace
    }

    // â”€â”€â”€ Background Tasks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Start scheduled background maintenance.
    ///
    /// - Runs `maintenance_pass()` every `interval_secs` seconds (time-based).
    /// - Also watches for `threshold` inserts and triggers maintenance early.
    /// - Recomputes scoring prototypes every 100 reinforcements.
    ///
    /// If called inside a Tokio runtime, tasks run on that runtime. Otherwise
    /// (Python, MCP server, plain `fn main`) a private single-worker runtime is
    /// created and owned by this `Memoire`.
    ///
    /// Stops any previous background tasks before starting new ones. Everything
    /// is stopped automatically when `Memoire` is dropped.
    pub fn start_background_maintenance(&mut self, interval_secs: u64, threshold: u32) {
        self.stop_background_maintenance();

        let handle = match tokio::runtime::Handle::try_current() {
            Ok(h) => h,
            Err(_) => {
                match tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(1)
                    .thread_name("memoire-maintenance")
                    .enable_time()
                    .build()
                {
                    Ok(rt) => {
                        let h = rt.handle().clone();
                        self.maintenance_runtime = Some(rt);
                        h
                    }
                    Err(e) => {
                        log::error!(
                            "start_background_maintenance: could not create Tokio runtime ({e}); scheduler not started"
                        );
                        return;
                    }
                }
            }
        };
        // `background::spawn_*` call `tokio::spawn`, which needs a runtime context.
        let _enter = handle.enter();

        // `Store::clone()` shares cache state with `self.store`, so maintenance
        // deletes are visible to `search`/`max_similarity` on this instance.
        let store_arc = Arc::new(self.store.clone());
        let insert_count = Arc::clone(&self.insert_count);
        let reinforcement_count = Arc::clone(&self.reinforcement_count);
        let embedder = Arc::clone(&self.embedder);
        let prototypes = Arc::clone(&self.prototypes);

        let store1 = Arc::clone(&store_arc);
        let time_fn: Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync> =
            Arc::new(move || store1.maintenance_pass().map_err(anyhow::Error::from));
        let h1 = background::spawn_maintenance_scheduler(time_fn, interval_secs.max(1));

        let store2 = Arc::clone(&store_arc);
        let count_fn: Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync> =
            Arc::new(move || store2.maintenance_pass().map_err(anyhow::Error::from));
        let h2 = background::spawn_count_watcher(count_fn, insert_count, threshold.max(1));

        let recompute_fn: Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync> = Arc::new(move || {
            let new_protos = Self::compute_prototypes(&*embedder);
            *prototypes
                .lock()
                .map_err(|_| anyhow::anyhow!("prototype lock poisoned"))? = new_protos;
            Ok(())
        });
        let h3 = background::spawn_reinforcement_watcher(recompute_fn, reinforcement_count, 100);

        self.maintenance_handles = vec![h1, h2, h3];
    }

    /// Abort all background tasks and tear down the private runtime, if any.
    fn stop_background_maintenance(&mut self) {
        for h in self.maintenance_handles.drain(..) {
            h.abort();
        }
        if let Some(rt) = self.maintenance_runtime.take() {
            // Non-blocking: dropping a Runtime from async context would panic.
            rt.shutdown_background();
        }
    }

    // Builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    /// without recompiling â€” e.g. raise `novelty` to 0.40 for research agents
    /// where unique facts matter more than actionability.
    ///
    /// Does NOT affect the trust formula (computed at recall time from
    /// reinforcement history). Only affects which memories reach `Active`
    /// vs `Shadow` at write time.
    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }

    pub fn with_oracle(mut self, oracle: impl IngestionOracle + 'static) -> Self {
        self.oracle = Some(Arc::new(oracle));
        self
    }

    pub fn with_nli_model(mut self, path: &std::path::Path) -> anyhow::Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let checker = quality::OnnxNliChecker::from_path(path)?;
            self.store.set_nli_checker(checker)?;
        }
        Ok(self)
    }

    // â”€â”€â”€ Core API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Chunk `content`, embed each chunk, and persist. Returns inserted row ids.
    pub fn remember(&self, content: &str) -> Result<Vec<i64>> {
        self.remember_with_source(content, "agent")
    }

    /// Chunk `content`, embed each chunk, and persist with `source_kind` metadata.
    ///
    /// Returns the ids of the rows that now represent the chunks. Under
    /// `AdmissionMode::QualityGate`, a chunk that independently restates a hidden `shadow`
    /// row (cosine >= 0.92, no contradiction, from a different ingest call than the row)
    /// promotes that row to `trial` and returns its id instead of inserting a duplicate.
    pub fn remember_with_source(&self, content: &str, source_kind: &str) -> Result<Vec<i64>> {
        let chunks = chunk_text(content, &self.chunker_config);
        if chunks.is_empty() {
            log::warn!("remember() called with empty input â€” nothing stored.");
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

            // Quality-gated admission hides low-scoring chunks as `shadow`. A later,
            // independent restatement of one corroborates it: promote the hidden row to
            // `trial` instead of inserting a duplicate. Not evidence of usefulness, so no
            // reinforcement is recorded; the promoted id is returned in place of a new one.
            if self.store.config.admission == AdmissionMode::QualityGate {
                if let Some(existing) = self.store.corroborate_shadow(chunk, embedding, &ids)? {
                    ids.push(existing);
                    continue;
                }
            }

            let max_sim = self.store.max_similarity(embedding)?;
            let novelty = (1.0 - max_sim).clamp(0.0, 1.0);
            let (mut meta, mut decision) = build_quality_meta(
                chunk,
                embedding,
                novelty,
                source_kind,
                &self.scoring_weights,
                &*self
                    .prototypes
                    .lock()
                    .map_err(|_| MemoireError::LockPoisoned)?,
            );

            // Default admission: ingest text features never gate exposure, so a chunk the
            // quality gate would have hidden enters as `trial` instead of `shadow`.
            if self.store.config.admission == AdmissionMode::AdmitAll
                && decision == IngestDecision::Shadow
            {
                decision = IngestDecision::Active;
                meta.store_state = ingest_state(&decision).to_string();
            }

            if let Some(oracle) = &self.oracle {
                let sample = self.store.sample_namespace(50)?;
                let oracle_decision = oracle.decide(chunk, embedding, &sample);
                if oracle_decision == IngestDecision::Reject {
                    continue;
                }
                decision = oracle_decision;
                meta.store_state = ingest_state(&decision).to_string();
            }

            match decision {
                IngestDecision::Reject => continue,
                IngestDecision::Shadow | IngestDecision::Active | IngestDecision::Probationary => {
                    let id = self.store.insert_with_quality(chunk, embedding, &meta)?;
                    self.store.resolve_contradictions_for_id(id)?;
                    ids.push(id);
                    self.insert_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        log::debug!("remember(): stored ids: {:?}", ids);
        Ok(ids)
    }

    /// Embed `query` and return the `top_k` most similar memories.
    pub fn recall(&self, query: &str, top_k: usize) -> Result<Vec<Memory>> {
        self.recall_with_options(query, top_k, false)
    }

    /// `include_hidden = false` returns live rows only (`trial` / `active`, with at most
    /// `ScoringConfig::trial_cap` unverified rows); `true` additionally returns hidden
    /// rows (`shadow`, `probationary`, `quarantined`) and skips the trial cap. See
    /// `Store::search`.
    pub fn recall_with_options(
        &self,
        query: &str,
        top_k: usize,
        include_hidden: bool,
    ) -> Result<Vec<Memory>> {
        if self.store.count()? == 0 {
            return Ok(vec![]);
        }
        log::debug!(
            "recall_with_options(): {:?} top_k={} include_hidden={}",
            query,
            top_k,
            include_hidden
        );
        let query_vec = self
            .embedder
            .embed_one(query)
            .map_err(MemoireError::Embedding)?;
        self.store.search(&query_vec, top_k, include_hidden)
    }

    /// Plain vector-similarity recall: exact cosine over every stored memory, no trust,
    /// no state filter, no trial cap, no database writes. The baseline for evaluation.
    pub fn recall_plain(&self, query: &str, top_k: usize) -> Result<Vec<Memory>> {
        if self.store.count()? == 0 {
            return Ok(vec![]);
        }
        let query_vec = self
            .embedder
            .embed_one(query)
            .map_err(MemoireError::Embedding)?;
        self.store.search_plain(&query_vec, top_k)
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
    /// read-only recency gate â€” stale memories are skipped but their trust
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
    /// Fires when `task_succeeded` AND at least one attribution gate passes:
    ///
    /// - the memory's distinctive tokens are contained in `agent_output` at least
    ///   `ScoringConfig::containment_threshold` (default 0.35; asymmetric, so long
    ///   outputs are not penalised), or
    /// - cosine(memory_embedding, embed(agent_output)) is at least 0.75.
    ///
    /// The cosine path catches semantic paraphrase where wording differs.
    /// Returns `true` if reinforcement was applied.
    pub fn reinforce_if_used(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
    ) -> Result<bool> {
        self.reinforce_if_used_with_session(memory_id, agent_output, task_succeeded, None)
    }

    pub fn reinforce_if_used_with_session(
        &self,
        memory_id: i64,
        agent_output: &str,
        task_succeeded: bool,
        session_id: Option<&str>,
    ) -> Result<bool> {
        // Embed the agent output so the cosine path can fire on paraphrase
        let output_embedding = self.embedder.embed_one(agent_output).ok();
        match self.store.reinforce_if_used(
            memory_id,
            agent_output,
            task_succeeded,
            output_embedding.as_deref(),
            session_id,
        ) {
            Ok(true) => {
                self.reinforcement_count.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            other => other,
        }
    }

    /// Penalize memories that contributed to a failed task outcome.
    ///
    /// `failure_severity` âˆˆ [0.0, 1.0] â€” scales how harshly the memories are
    /// penalized. Use 1.0 for a direct failure, 0.5 for a partial miss.
    /// Symmetric with `reinforce_if_used`. Returns the trust delta per memory.
    pub fn penalize_if_used(
        &self,
        memory_ids: &[i64],
        failure_severity: f32,
    ) -> Result<Vec<store::PenaltyOutcome>> {
        self.store.penalize_if_used(memory_ids, failure_severity)
    }

    /// Reinforce a memory the agent explicitly declared it used, with no text gate. The
    /// caller asserts both the declaration and the task success. Same namespace,
    /// quarantine and one-vote-per-session rules as `reinforce_if_used`.
    pub fn reinforce_declared(&self, memory_id: i64, session_id: Option<&str>) -> Result<bool> {
        let applied = self.store.reinforce_declared(memory_id, session_id)?;
        if applied {
            self.reinforcement_count.fetch_add(1, Ordering::Relaxed);
        }
        Ok(applied)
    }

    /// Operator action: return a quarantined memory to service, clearing its failure
    /// evidence. Returns false if `id` is not a quarantined memory of this namespace.
    pub fn unquarantine(&self, id: i64) -> Result<bool> {
        self.store.unquarantine(id)
    }

    /// Promote a hidden `shadow` / `probationary` memory to the live set. Returns false
    /// if `id` is not such a memory of this namespace.
    pub fn promote_shadow(&self, id: i64) -> Result<bool> {
        self.store.promote_shadow(id)
    }

    /// Delete a memory by id. Returns true if it existed.
    pub fn forget(&self, id: i64) -> Result<bool> {
        self.store.forget(id)
    }

    /// Return every stored memory ordered by recency. Score is 1.0 (no query).
    pub fn export_all(&self) -> Result<Vec<store::Memory>> {
        self.store.all()
    }

    /// Export the current namespace's non-archived memories as a JSON-serializable snapshot.
    pub fn export_namespace(&self) -> Result<serde_json::Value> {
        self.store.export_namespace()
    }

    /// Import a memory snapshot into the current namespace, treating it as
    /// untrusted: only `content` is used ([`ImportTrust::Reset`]).
    pub fn import_namespace(&self, snapshot: &serde_json::Value) -> Result<usize> {
        self.import_namespace_with(snapshot, ImportTrust::Reset)
    }

    /// Import a memory snapshot, choosing how much of its metadata to keep.
    /// See [`ImportTrust`].
    pub fn import_namespace_with(
        &self,
        snapshot: &serde_json::Value,
        trust: ImportTrust,
    ) -> Result<usize> {
        let memories = snapshot
            .get("memories")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                MemoireError::Embedding(anyhow::anyhow!(
                    "Invalid snapshot format: missing memories array"
                ))
            })?;

        let mut imported_count = 0;
        for mem in memories {
            let content = match mem.get("content").and_then(|v| v.as_str()) {
                Some(c) => c,
                None => continue,
            };

            let ids = self.remember(content)?;
            if ids.is_empty() {
                continue;
            }
            if trust == ImportTrust::PreserveBounded {
                let trust_ema = mem
                    .get("trust_ema")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32);
                let reinforcement_count = mem
                    .get("reinforcement_count")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let importance_base = mem
                    .get("importance_base")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32)
                    .unwrap_or(0.5);
                let confidence = mem
                    .get("confidence")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32)
                    .unwrap_or(0.5);
                let created_at = mem.get("created_at").and_then(|v| v.as_i64());
                self.store.update_imported_metadata(
                    &ids,
                    trust_ema,
                    reinforcement_count,
                    importance_base,
                    confidence,
                    created_at,
                )?;
            }
            imported_count += 1;
        }
        Ok(imported_count)
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
    /// `mmr_lambda = 1.0` â†’ identical to `recall` (pure relevance).
    /// `mmr_lambda = 0.5` (default) â†’ balanced relevance/diversity.
    pub fn recall_mmr(&self, query: &str, top_k: usize, mmr_lambda: f32) -> Result<Vec<Memory>> {
        if self.store.count()? == 0 {
            return Ok(vec![]);
        }
        let candidate_k = (top_k * 3).max(20);
        let query_vec = self
            .embedder
            .embed_one(query)
            .map_err(MemoireError::Embedding)?;
        let candidates = self.store.search(&query_vec, candidate_k, false)?;
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

    /// Recall memories together with an attribution breakdown for ranking and policy use.
    pub fn recall_explained(&self, query: &str, top_k: usize) -> Result<Vec<RecallAttribution>> {
        let memories = self.recall(query, top_k)?;
        Ok(memories
            .into_iter()
            .map(|memory| {
                let mut reasons = Vec::new();
                if memory.score >= 0.75 {
                    reasons.push("high semantic relevance".to_string());
                } else if memory.score >= 0.45 {
                    reasons.push("moderate semantic relevance".to_string());
                } else {
                    reasons.push("weak semantic relevance".to_string());
                }

                if memory.trust >= 0.75 {
                    reasons.push("high trust".to_string());
                } else if memory.trust >= 0.45 {
                    reasons.push("hint-level trust".to_string());
                } else {
                    reasons.push("low trust".to_string());
                }

                if memory.uncertainty >= 0.60 {
                    reasons.push("high uncertainty".to_string());
                }
                if memory.reinforcement_count > 0 {
                    reasons.push(format!(
                        "{} reinforcement event(s)",
                        memory.reinforcement_count
                    ));
                }
                if memory.ingestion_driver != "Unknown" {
                    reasons.push(format!("ingestion driver: {}", memory.ingestion_driver));
                }

                RecallAttribution {
                    semantic_score: memory.score,
                    trust: memory.trust,
                    uncertainty: memory.uncertainty,
                    effective_weight: memory.effective_weight,
                    rank_score: (0.65 * memory.score
                        + 0.25 * memory.trust
                        + 0.10 * memory.effective_weight)
                        .clamp(0.0, 1.0),
                    memory,
                    reasons,
                }
            })
            .collect())
    }
    // â”€â”€â”€ Private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Recompute the semantic scoring prototypes from the current embedder.
    ///
    /// Should be called periodically (e.g., every 100 reinforcement events) so
    /// that the prototype centroids reflect any fine-tuned embedding model.
    /// Internally re-embeds the 15 fixed prototype sentences and updates
    /// `self.prototypes` in-place.
    pub fn recompute_prototypes(&self) -> Result<()> {
        let new_protos = Self::compute_prototypes(&*self.embedder);
        *self
            .prototypes
            .lock()
            .map_err(|_| MemoireError::LockPoisoned)? = new_protos;
        Ok(())
    }

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
            "Always close database connections and file handles in a finally block to avoid leaks"
                .to_string(),
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
                log::warn!("compute_prototypes failed â€” falling back to neutral: {e}");
                ScoringPrototypes::neutral(embedder.dim())
            }
            Ok(_) => {
                log::warn!(
                    "compute_prototypes returned unexpected count â€” falling back to neutral"
                );
                ScoringPrototypes::neutral(embedder.dim())
            }
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lifecycle_tests {
    use super::*;

    struct StubEmbedder {
        dim: usize,
        fail: bool,
    }

    impl EmbedProvider for StubEmbedder {
        fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
            if self.fail {
                anyhow::bail!("stub embed failure");
            }
            Ok(texts.iter().map(|_| vec![0.1_f32; self.dim]).collect())
        }
        fn dim(&self) -> usize {
            self.dim
        }
    }

    fn stub(dim: usize, fail: bool) -> Memoire {
        Memoire::in_memory_with_embedder(Box::new(StubEmbedder { dim, fail })).unwrap()
    }

    #[test]
    fn compute_prototypes_fallback_uses_embedder_dim() {
        let m = stub(8, true);
        let p = m.prototypes.lock().unwrap();
        assert!(!p.is_semantic);
        assert_eq!(p.consequence.len(), 8);
        assert_eq!(p.actionability.len(), 8);
        assert_eq!(p.reusability.len(), 8);
    }

    #[test]
    fn background_maintenance_starts_without_ambient_runtime() {
        let mut m = stub(8, false);
        m.start_background_maintenance(60, 5);
        assert!(m.maintenance_runtime.is_some());
        assert_eq!(m.maintenance_handles.len(), 3);
        assert!(m.maintenance_handles.iter().all(|h| !h.is_finished()));

        // Restart replaces tasks instead of stacking them.
        m.start_background_maintenance(60, 5);
        assert_eq!(m.maintenance_handles.len(), 3);
        drop(m); // must neither hang nor panic
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn background_maintenance_uses_ambient_runtime_and_stops_on_drop() {
        let mut m = stub(8, false);
        m.start_background_maintenance(60, 5);
        assert!(m.maintenance_runtime.is_none());
        let aborts: Vec<_> = m
            .maintenance_handles
            .iter()
            .map(|h| h.abort_handle())
            .collect();
        assert_eq!(aborts.len(), 3);
        drop(m);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(aborts.iter().all(|a| a.is_finished()));
    }
}

pub mod chunker;
pub mod embedder;
pub mod error;
pub mod ffi;
pub mod store;

use chunker::{ChunkerConfig, chunk_text};
use embedder::Embedder;
use error::{MemoireError, Result};
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
            ids.push(self.store.insert(chunk, embedding)?);
        }
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
}

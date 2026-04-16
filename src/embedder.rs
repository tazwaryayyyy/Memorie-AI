use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::sync::Mutex;

pub struct Embedder {
    model: Mutex<TextEmbedding>,
    pub dim: usize,
}

impl Embedder {
    pub fn new() -> Result<Self> {
        log::info!("Initialising local embedding model (all-MiniLM-L6-v2)...");
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        )?;
        log::info!("Embedding model ready.");
        Ok(Self {
            model: Mutex::new(model),
            dim: 384,
        })
    }

    pub fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder mutex poisoned"))?;
        model.embed(texts, None)
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut r = self.embed(vec![text.to_string()])?;
        r.pop()
            .ok_or_else(|| anyhow::anyhow!("embedder returned empty result"))
    }
}

/// Embedding backend abstraction.
///
/// Implement this trait to swap `all-MiniLM-L6-v2` for any model — BERT-large,
/// an OpenAI API wrapper, or a proprietary encoder — without recompiling the
/// Rust core. Pass the custom backend via [`crate::Memoire::new_with_embedder`].
///
/// # Example
///
/// ```rust,no_run
/// use memoire::embedder::EmbedProvider;
///
/// struct MyEmbedder;
/// impl EmbedProvider for MyEmbedder {
///     fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
///         // call your model here
///         Ok(texts.iter().map(|_| vec![0.0_f32; 768]).collect())
///     }
///     fn dim(&self) -> usize { 768 }
/// }
/// ```
pub trait EmbedProvider: Send + Sync {
    /// Embed a batch of texts. Returns one output vector per input.
    fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Embed a single text. Default implementation calls `embed`.
    fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut r = self.embed(vec![text.to_string()])?;
        r.pop()
            .ok_or_else(|| anyhow::anyhow!("embedder returned empty result"))
    }

    /// Dimensionality of the output vectors produced by this backend.
    fn dim(&self) -> usize;
}

impl EmbedProvider for Embedder {
    fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("embedder mutex poisoned"))?;
        model.embed(texts, None)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

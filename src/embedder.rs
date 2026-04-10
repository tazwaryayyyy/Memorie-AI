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
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(true),
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
        Ok(model.embed(texts, None)?)
    }

    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut r = self.embed(vec![text.to_string()])?;
        r.pop().ok_or_else(|| anyhow::anyhow!("embedder returned empty result"))
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoireError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("embedding error: {0}")]
    Embedding(#[from] anyhow::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("invalid utf-8 in FFI input")]
    InvalidUtf8,

    #[error("null pointer in FFI call")]
    NullPointer,

    #[error("empty store — nothing to recall from")]
    EmptyStore,
}

pub type Result<T> = std::result::Result<T, MemoireError>;

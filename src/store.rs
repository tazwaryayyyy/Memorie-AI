use rusqlite::{params, Connection};
use serde::Serialize;

use crate::error::Result;

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub score: f32,
    pub created_at: i64,
}

pub struct Store {
    conn: Connection,
}

const SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS memories (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        content     TEXT    NOT NULL,
        embedding   BLOB    NOT NULL,
        created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
    );
    CREATE INDEX IF NOT EXISTS idx_memories_created_at
        ON memories (created_at DESC);
";

pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn blob_to_vec(blob: &[u8]) -> Option<Vec<f32>> {
    if blob.len() % 4 != 0 {
        return None;
    }
    Some(
        blob.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

impl Store {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        conn.execute_batch(SCHEMA)?;
        log::debug!("SQLite store opened at {path}");
        Ok(Self { conn })
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    pub fn insert(&self, content: &str, embedding: &[f32]) -> Result<i64> {
        let blob = vec_to_blob(embedding);
        self.conn.execute(
            "INSERT INTO memories (content, embedding) VALUES (?1, ?2)",
            params![content, blob],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<Memory>> {
        if top_k == 0 {
            return Ok(vec![]);
        }

        let mut stmt = self.conn.prepare(
            "SELECT id, content, embedding, created_at FROM memories ORDER BY created_at DESC",
        )?;

        let mut scored: Vec<(f32, Memory)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, content, blob, created_at)| {
                let embedding = blob_to_vec(&blob)?;
                let score = cosine_similarity(query_vec, &embedding);
                Some((score, Memory { id, content, score, created_at }))
            })
            .collect();

        scored.sort_unstable_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);

        Ok(scored.into_iter().map(|(_, m)| m).collect())
    }

    /// Return every stored memory ordered by insertion time descending.
    /// Score is set to 1.0 as a sentinel (no query was made).
    pub fn all(&self) -> Result<Vec<Memory>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, created_at FROM memories ORDER BY created_at DESC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Memory {
                    id:         row.get(0)?,
                    content:    row.get(1)?,
                    score:      1.0,
                    created_at: row.get(2)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    pub fn forget(&self, id: i64) -> Result<bool> {
        let rows = self.conn.execute(
            "DELETE FROM memories WHERE id = ?1",
            params![id],
        )?;
        Ok(rows > 0)
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self.conn.query_row(
            "SELECT COUNT(*) FROM memories",
            [],
            |r| r.get(0),
        )?)
    }

    pub fn clear(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM memories;")?;
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
        store.insert("beta",  &[0.0_f32, 1.0]).unwrap();
        let all = store.all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_scores_descending() {
        let store = Store::in_memory().unwrap();
        store.insert("closest",  &[1.0_f32, 0.0, 0.0]).unwrap();
        store.insert("middle",   &[0.7_f32, 0.7, 0.0]).unwrap();
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
}

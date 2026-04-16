use rusqlite::{params, Connection};
use serde::Serialize;

use std::collections::HashMap;

use crate::error::Result;
use crate::quality::{compute_trust, effective_weight, recency_bonus, QualityMeta};

#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub score: f32,
    pub trust: f32,
    pub state: String,
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
        created_at  INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
        importance_base REAL NOT NULL DEFAULT 0.5,
        confidence REAL NOT NULL DEFAULT 0.5,
        novelty REAL NOT NULL DEFAULT 0.5,
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
        archived INTEGER NOT NULL DEFAULT 0,
        effective_weight REAL NOT NULL DEFAULT 1.0,
        fingerprint TEXT NOT NULL DEFAULT '',
        claim_key TEXT,
        claim_value TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_memories_created_at
        ON memories (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_memories_claim_key
        ON memories (claim_key);
    CREATE INDEX IF NOT EXISTS idx_memories_store_state
        ON memories (store_state, archived, superseded_by);
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
        Self::ensure_quality_columns(&conn)?;
        log::debug!("SQLite store opened at {path}");
        Ok(Self { conn })
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(SCHEMA)?;
        Self::ensure_quality_columns(&conn)?;
        Ok(Self { conn })
    }

    pub fn insert(&self, content: &str, embedding: &[f32]) -> Result<i64> {
        let meta = QualityMeta::default_active(content);
        self.insert_with_quality(content, embedding, &meta)
    }

    /// Reinforce a memory only when the agent actually used it successfully.
    ///
    /// Fires when: `task_succeeded` AND (
    ///   Jaccard token overlap >= 0.15   OR
    ///   cosine(memory_embedding, output_embedding) >= 0.75
    /// )
    ///
    /// The cosine path handles semantic paraphrase: "use fixed-point arithmetic"
    /// vs "use Decimal for money" — different words, same lesson.
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

        let (content, mem_blob): (String, Vec<u8>) = match self.conn.query_row(
            "SELECT content, embedding FROM memories WHERE id = ?1",
            params![memory_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ) {
            Ok(v) => v,
            Err(_) => return Ok(false),
        };

        // Path 1: Jaccard token overlap
        let memory_tokens: std::collections::HashSet<&str> = content.split_whitespace().collect();
        let output_tokens: std::collections::HashSet<&str> =
            agent_output.split_whitespace().collect();
        let intersection = memory_tokens.intersection(&output_tokens).count();
        let union = memory_tokens.union(&output_tokens).count();
        let jaccard_ok = union > 0 && (intersection as f32 / union as f32) >= 0.15;

        // Path 2: cosine similarity between memory embedding and output embedding
        let cosine_ok = if let Some(out_emb) = output_embedding {
            blob_to_vec(&mem_blob)
                .map(|mem_emb| cosine_similarity(&mem_emb, out_emb) >= 0.75)
                .unwrap_or(false)
        } else {
            false
        };

        if jaccard_ok || cosine_ok {
            self.conn.execute(
                "UPDATE memories
                 SET reinforcement_count = reinforcement_count + 1,
                     last_accessed_at = CAST(strftime('%s', 'now') AS INTEGER)
                 WHERE id = ?1",
                params![memory_id],
            )?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn insert_with_quality(
        &self,
        content: &str,
        embedding: &[f32],
        quality: &QualityMeta,
    ) -> Result<i64> {
        let blob = vec_to_blob(embedding);
        self.conn.execute(
            "INSERT INTO memories (
                content, embedding,
                importance_base, confidence, novelty,
                actionability, evidence, consequence, reusability,
                source_kind, store_state, reinforcement_count,
                last_accessed_at, superseded_by, contradiction_group,
                archived, effective_weight, fingerprint, claim_key, claim_value
            ) VALUES (
                ?1, ?2,
                ?3, ?4, ?5,
                ?6, ?7, ?8, ?9,
                ?10, ?11, ?12,
                ?13, ?14, ?15,
                ?16, ?17, ?18, ?19, ?20
            )",
            params![
                content,
                blob,
                quality.importance_base,
                quality.confidence,
                quality.novelty,
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
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn max_similarity(&self, query_vec: &[f32]) -> Result<f32> {
        let mut stmt = self.conn.prepare(
            "SELECT embedding FROM memories
             WHERE archived = 0 AND superseded_by IS NULL
             ORDER BY created_at DESC",
        )?;

        let mut max_sim = 0.0_f32;
        for row in stmt.query_map([], |row| row.get::<_, Vec<u8>>(0))? {
            let blob = match row {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Some(v) = blob_to_vec(&blob) else {
                continue;
            };
            let sim = cosine_similarity(query_vec, &v);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        Ok(max_sim)
    }

    pub fn fingerprint_exists(&self, fingerprint: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories
             WHERE fingerprint = ?1
               AND archived = 0
               AND superseded_by IS NULL",
            params![fingerprint],
            |r| r.get(0),
        )?;
        Ok(count > 0)
    }

    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<Memory>> {
        if top_k == 0 {
            return Ok(vec![]);
        }

        let mut stmt = self.conn.prepare(
            "SELECT
                id, content, embedding, created_at,
                importance_base, reinforcement_count, confidence,
                store_state, contradiction_group
             FROM memories
             WHERE archived = 0
               AND superseded_by IS NULL
               AND store_state IN ('active', 'shadow')
             ORDER BY created_at DESC",
        )?;

        let now: i64 =
            self.conn
                .query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                    r.get(0)
                })?;

        // (score, trust, contradiction_group, Memory)
        let mut scored: Vec<(f32, f32, Option<String>, Memory)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, f32>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, f32>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .filter_map(
                |(
                    id,
                    content,
                    blob,
                    created_at,
                    importance_base,
                    reinforcement_count,
                    confidence,
                    store_state,
                    contradiction_group,
                )| {
                    let embedding = blob_to_vec(&blob)?;
                    let sim = cosine_similarity(query_vec, &embedding);
                    let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
                    let weight = effective_weight(importance_base, age_days, reinforcement_count);
                    let final_score =
                        0.75 * sim + 0.20 * weight + 0.05 * recency_bonus(created_at, now);
                    // contradiction_survived: active + has a claim group = won at least one check
                    let contradiction_survived =
                        contradiction_group.is_some() && store_state == "active";
                    let trust_base = compute_trust(
                        reinforcement_count,
                        contradiction_survived,
                        &store_state,
                        importance_base,
                        confidence,
                        age_days,
                    );
                    // Context-aware trust: a memory highly relevant to this specific
                    // query gets a boost; tangentially retrieved memories are dampened.
                    // trust_final = trust_base * (0.6 + 0.4 * similarity)
                    // Range: trust_base*0.6 (orthogonal query) → trust_base (perfect match)
                    let sim_clamped = sim.clamp(0.0, 1.0);
                    let trust = (trust_base * (0.6 + 0.4 * sim_clamped)).clamp(0.0, 1.0);
                    Some((
                        final_score,
                        trust,
                        contradiction_group,
                        Memory {
                            id,
                            content,
                            score: final_score,
                            trust,
                            state: store_state,
                            created_at,
                        },
                    ))
                },
            )
            .collect();

        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Conflict-aware deduplication: for each contradiction_group keep max-trust only.
        // If the winning memory still has trust < 0.4, drop the entire group —
        // surfacing a weak memory from a contested group is worse than surfacing nothing.
        let mut group_winner: HashMap<String, usize> = HashMap::new();
        for (idx, (_, trust, cg_opt, _)) in scored.iter().enumerate() {
            if let Some(cg) = cg_opt {
                let entry = group_winner.entry(cg.clone()).or_insert(idx);
                if *trust > scored[*entry].1 {
                    *entry = idx;
                }
            }
        }

        // Collect groups where even the winner is too weak to trust
        let weak_groups: std::collections::HashSet<String> = group_winner
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
                        return false; // drop entire weak group
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

        Ok(result)
    }

    pub fn resolve_contradictions_for_id(&self, id: i64) -> Result<()> {
        let target = self.conn.query_row(
            "SELECT id, claim_key, claim_value, importance_base, confidence, evidence, created_at
             FROM memories WHERE id = ?1",
            params![id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, f32>(3)?,
                    row.get::<_, f32>(4)?,
                    row.get::<_, f32>(5)?,
                    row.get::<_, i64>(6)?,
                ))
            },
        );

        let (new_id, claim_key, claim_value, imp, conf, ev, created_at) = match target {
            Ok(v) => v,
            Err(_) => return Ok(()),
        };
        let (Some(key), Some(value)) = (claim_key, claim_value) else {
            return Ok(());
        };

        let mut stmt = self.conn.prepare(
            "SELECT id, importance_base, confidence, evidence, created_at
             FROM memories
             WHERE claim_key = ?1
               AND claim_value IS NOT NULL
               AND claim_value != ?2
               AND archived = 0
               AND superseded_by IS NULL
               AND id != ?3",
        )?;

        let now: i64 =
            self.conn
                .query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                    r.get(0)
                })?;

        for row in stmt.query_map(params![key, value, new_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, f32>(1)?,
                row.get::<_, f32>(2)?,
                row.get::<_, f32>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })? {
            let (other_id, o_imp, o_conf, o_ev, o_created) = match row {
                Ok(v) => v,
                Err(_) => continue,
            };
            let rec_new = recency_bonus(created_at, now);
            let rec_old = recency_bonus(o_created, now);
            let q_new = 0.45 * imp + 0.25 * conf + 0.20 * rec_new + 0.10 * ev;
            let q_old = 0.45 * o_imp + 0.25 * o_conf + 0.20 * rec_old + 0.10 * o_ev;

            if q_new >= q_old {
                let _ = self.conn.execute(
                    "UPDATE memories SET superseded_by = ?1, archived = 1 WHERE id = ?2",
                    params![new_id, other_id],
                );
            } else {
                let _ = self.conn.execute(
                    "UPDATE memories SET superseded_by = ?1, archived = 1 WHERE id = ?2",
                    params![other_id, new_id],
                );
            }
        }

        Ok(())
    }

    pub fn maintenance_pass(&self) -> Result<()> {
        self.conn.execute_batch(
            "UPDATE memories
             SET archived = 1
             WHERE superseded_by IS NOT NULL
               AND archived = 0;

             DELETE FROM memories
             WHERE archived = 1
               AND (strftime('%s', 'now') - created_at) > 7 * 86400;

             DELETE FROM memories
             WHERE archived = 0
               AND superseded_by IS NULL
               AND reinforcement_count = 0
               AND importance_base < 0.12
               AND (strftime('%s', 'now') - created_at) > 30 * 86400;",
        )?;
        Ok(())
    }

    /// Return every stored memory ordered by insertion time descending.
    /// Score is set to 1.0 as a sentinel (no query was made).
    pub fn all(&self) -> Result<Vec<Memory>> {
        let now: i64 =
            self.conn
                .query_row("SELECT CAST(strftime('%s', 'now') AS INTEGER)", [], |r| {
                    r.get(0)
                })?;

        let mut stmt = self.conn.prepare(
            "SELECT id, content, created_at,
                    importance_base, reinforcement_count, confidence,
                    store_state, contradiction_group
             FROM memories ORDER BY created_at DESC",
        )?;

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, f32>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, f32>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, content, created_at, imp, rc, conf, state, cg)| {
                let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
                let contradiction_survived = cg.is_some() && state == "active";
                let trust = compute_trust(rc, contradiction_survived, &state, imp, conf, age_days);
                Memory {
                    id,
                    content,
                    score: 1.0,
                    trust,
                    state,
                    created_at,
                }
            })
            .collect();
        Ok(rows)
    }

    pub fn forget(&self, id: i64) -> Result<bool> {
        let rows = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(rows > 0)
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?)
    }

    pub fn clear(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM memories;")?;
        Ok(())
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

        for sql in alter {
            conn.execute_batch(sql)?;
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_memories_claim_key ON memories (claim_key);
             CREATE INDEX IF NOT EXISTS idx_memories_store_state ON memories (store_state, archived, superseded_by);",
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
}

//! `reinforce_if_used` must tell "no such memory" (`Ok(false)`) apart from a
//! database failure (`Err`). A blanket `Err(_) => Ok(false)` made a corrupt or
//! locked database look like "attribution gate not met", so feedback was lost
//! silently.

use memoire::{embedder::EmbedProvider, Memoire};

const CONTENT: &str = "always validate external inputs before processing user data";

/// Deterministic bag-of-words embedder so this test runs offline.
struct TestEmbedder;

impl EmbedProvider for TestEmbedder {
    fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0_f32; 384];
                for token in text.split(|c: char| !c.is_ascii_alphanumeric()) {
                    if token.is_empty() {
                        continue;
                    }
                    let h = token
                        .to_ascii_lowercase()
                        .bytes()
                        .fold(0xcbf29ce484222325_u64, |h, b| {
                            (h ^ b as u64).wrapping_mul(0x100000001b3)
                        });
                    v[h as usize % 384] += 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                v
            })
            .collect())
    }

    fn dim(&self) -> usize {
        384
    }
}

fn open(path: &std::path::Path) -> Memoire {
    Memoire::new_ns_with_embedder(path.to_str().unwrap(), "default", Box::new(TestEmbedder))
        .expect("open Memoire")
}

#[test]
fn unknown_memory_id_is_ok_false() {
    let dir = tempfile::tempdir().unwrap();
    let m = open(&dir.path().join("m.db"));
    assert!(!m.reinforce_if_used(9_999, CONTENT, true).unwrap());
}

#[test]
fn database_failure_is_an_error_not_ok_false() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("m.db");
    let m = open(&db);
    let id = m.remember(CONTENT).unwrap()[0];
    assert!(m.reinforce_if_used(id, CONTENT, true).unwrap());

    // Break the schema underneath the open handle: the SELECT in
    // `Store::reinforce_if_used` now fails with "no such column".
    let other = rusqlite::Connection::open(&db).unwrap();
    other
        .execute_batch("ALTER TABLE memories RENAME COLUMN confidence TO confidence_gone;")
        .unwrap();

    let res = m.reinforce_if_used(id, CONTENT, true);
    assert!(
        res.is_err(),
        "a SQLite failure must surface as Err, got {res:?}"
    );
}

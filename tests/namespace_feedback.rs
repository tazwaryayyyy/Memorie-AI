//! Feedback calls (`reinforce_if_used*`, `penalize_if_used`, `forget`) take raw row
//! ids. Row ids come from one global AUTOINCREMENT sequence, so another tenant's id
//! is trivially guessable. These tests drive the public API against one shared
//! SQLite file and assert a namespace can only act on its own rows.

use memoire::{embedder::EmbedProvider, Memoire};

const CONTENT: &str = "always validate external inputs before processing user data";

/// Deterministic bag-of-words embedder so these tests run offline.
struct TestEmbedder;

impl TestEmbedder {
    const DIM: usize = 384;
}

impl EmbedProvider for TestEmbedder {
    fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0_f32; Self::DIM];
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
                    v[h as usize % Self::DIM] += 1.0;
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
        Self::DIM
    }
}

fn open(path: &str, namespace: &str) -> Memoire {
    Memoire::new_ns_with_embedder(path, namespace, Box::new(TestEmbedder)).expect("open store")
}

#[test]
fn feedback_cannot_cross_namespaces() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared.db");
    let path = path.to_str().unwrap();
    let a = open(path, "tenant_a");
    let b = open(path, "tenant_b");

    let id_a = a
        .remember(CONTENT)
        .unwrap()
        .first()
        .copied()
        .expect("tenant_a memory must be stored");
    let before = a.export_all().unwrap().remove(0);
    assert_eq!(before.id, id_a);
    assert_eq!(before.reinforcement_count, 0);

    // tenant_b replays tenant_a's id with output identical to the memory, so the
    // attribution gates would pass if only the id were checked.
    assert!(
        !b.reinforce_if_used(id_a, CONTENT, true).unwrap(),
        "tenant_b must not reinforce tenant_a's memory"
    );
    assert!(
        !b.reinforce_if_used_with_session(id_a, CONTENT, true, Some("tenant_b_session"))
            .unwrap(),
        "a session id must not bypass namespace scoping"
    );
    assert!(
        b.penalize_if_used(&[id_a], 1.0).unwrap().is_empty(),
        "tenant_b must not penalize tenant_a's memory"
    );
    assert!(
        !b.forget(id_a).unwrap(),
        "tenant_b must not forget tenant_a's memory"
    );

    let after = a.export_all().unwrap().remove(0);
    assert_eq!(after.reinforcement_count, before.reinforcement_count);
    // `uncertainty` is a pure function of reinforcement/failure counts, so it moves
    // if either was touched.
    assert_eq!(after.uncertainty, before.uncertainty);

    // Control: the owning namespace can still give feedback on its own row.
    assert!(a.reinforce_if_used(id_a, CONTENT, true).unwrap());
    assert_eq!(a.export_all().unwrap()[0].reinforcement_count, 1);
    assert_eq!(a.penalize_if_used(&[id_a], 1.0).unwrap().len(), 1);
}

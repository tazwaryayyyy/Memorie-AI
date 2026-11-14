//! Imported snapshots are untrusted input: by default only `content` is used,
//! and even the opt-in "restore" mode clamps every metadata field.

use memoire::{embedder::EmbedProvider, store::IMPORT_MAX_REINFORCEMENT, ImportTrust, Memoire};
use serde_json::json;

const CONTENT: &str = "always validate external inputs before processing user data";
const QUERY: &str = "validate external inputs";

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

fn fresh() -> Memoire {
    Memoire::in_memory_with_embedder(Box::new(TestEmbedder)).expect("in-memory Memoire")
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

fn snapshot(entry: serde_json::Value) -> serde_json::Value {
    let mut entry = entry;
    entry["content"] = json!(CONTENT);
    json!({ "memoire_export_version": 1, "namespace": "default", "memories": [entry] })
}

fn hostile() -> serde_json::Value {
    snapshot(json!({
        "trust_ema": 1.0,
        "reinforcement_count": 9_999_999,
        "importance_base": 1.0,
        "confidence": 1.0,
        "created_at": 9_999_999_999_i64,
    }))
}

#[test]
fn default_import_ignores_hostile_metadata() {
    let imported = fresh();
    assert_eq!(imported.import_namespace(&hostile()).unwrap(), 1);

    let baseline = fresh();
    baseline.remember(CONTENT).unwrap();

    let got = imported.recall(QUERY, 1).unwrap().remove(0);
    let want = baseline.recall(QUERY, 1).unwrap().remove(0);

    assert_eq!(got.reinforcement_count, 0);
    assert!(
        got.created_at <= now() + 5,
        "future created_at leaked through"
    );
    assert!(
        (got.trust - want.trust).abs() < 0.05,
        "imported trust {} should equal remember() trust {}",
        got.trust,
        want.trust
    );
}

#[test]
fn explicit_reset_matches_default() {
    let m = fresh();
    m.import_namespace_with(&hostile(), ImportTrust::Reset)
        .unwrap();
    assert_eq!(m.recall(QUERY, 1).unwrap()[0].reinforcement_count, 0);
}

#[test]
fn preserve_bounded_clamps_out_of_range_metadata() {
    let m = fresh();
    let wild = snapshot(json!({
        "trust_ema": 7.5,
        "reinforcement_count": 10_000_000_i64,
        "importance_base": 5.0,
        "confidence": -3.0,
        "created_at": 9_999_999_999_i64,
    }));
    m.import_namespace_with(&wild, ImportTrust::PreserveBounded)
        .unwrap();

    let got = m.recall(QUERY, 1).unwrap().remove(0);
    assert!(got.trust <= 1.0 && got.trust >= 0.0, "trust {}", got.trust);
    assert_eq!(got.reinforcement_count, IMPORT_MAX_REINFORCEMENT);
    assert!(got.created_at <= now() + 5, "created_at {}", got.created_at);
}

#[test]
fn preserve_bounded_drops_non_finite_trust() {
    let m = fresh();
    // 1e300 is a valid JSON number but overflows f32 to infinity.
    let snap = snapshot(json!({ "trust_ema": 1e300, "reinforcement_count": 1 }));
    m.import_namespace_with(&snap, ImportTrust::PreserveBounded)
        .unwrap();

    let got = m.recall(QUERY, 1).unwrap().remove(0);
    assert!(got.trust.is_finite());
}

#[test]
fn preserve_bounded_keeps_in_range_values() {
    let m = fresh();
    let snap = snapshot(json!({
        "trust_ema": 0.3,
        "reinforcement_count": 4,
        "importance_base": 0.6,
        "confidence": 0.7,
        "created_at": 1_700_000_000_i64,
    }));
    m.import_namespace_with(&snap, ImportTrust::PreserveBounded)
        .unwrap();

    let got = m.recall(QUERY, 1).unwrap().remove(0);
    assert_eq!(got.reinforcement_count, 4);
    assert_eq!(got.created_at, 1_700_000_000);
}

#[test]
fn snapshot_without_memories_list_is_rejected() {
    assert!(fresh()
        .import_namespace(&json!({ "memories": "nope" }))
        .is_err());
}

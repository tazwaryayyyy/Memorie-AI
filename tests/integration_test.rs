use memoire::Memoire;
use std::sync::Arc;

// Verify polarity detection is real logic, not random floats.
// Imported directly from the public quality module.
use memoire::quality::{detect_polarity, Polarity};
fn m() -> Memoire {
    Memoire::in_memory().expect("failed to create in-memory Memoire")
}

#[test]
fn test_remember_and_count() {
    let mem = m();
    assert_eq!(mem.count().unwrap(), 0);
    mem.remember("Fixed the pagination bug in the user list endpoint")
        .unwrap();
    assert!(mem.count().unwrap() >= 1);
}

#[test]
fn test_recall_relevance() {
    let mem = m();
    mem.remember("Fixed a null pointer dereference in the auth middleware")
        .unwrap();
    mem.remember("Refactored the database connection pool for throughput")
        .unwrap();
    mem.remember("Added unit tests for the payment module")
        .unwrap();

    let results = mem.recall("authentication security bug", 3).unwrap();
    assert!(!results.is_empty());
    assert!(
        results[0].content.contains("auth"),
        "Expected auth memory first, got: {}",
        results[0].content
    );
    assert!(results[0].score > 0.0);
}

#[test]
fn test_recall_scores_descending() {
    let mem = m();
    mem.remember("auth bug fix in middleware").unwrap();
    mem.remember("database schema migration").unwrap();
    mem.remember("auth token validation patch").unwrap();

    let results = mem.recall("authentication", 3).unwrap();
    for w in results.windows(2) {
        assert!(w[0].score >= w[1].score, "scores not descending");
    }
}

#[test]
fn test_forget() {
    let mem = m();
    let ids = mem.remember("temporary memory").unwrap();
    let before = mem.count().unwrap();
    assert!(mem.forget(ids[0]).unwrap());
    assert_eq!(mem.count().unwrap(), before - 1);
    assert!(!mem.forget(99999).unwrap());
}

#[test]
fn test_clear() {
    let mem = m();
    mem.remember("one").unwrap();
    mem.remember("two").unwrap();
    assert!(mem.count().unwrap() >= 2);
    mem.clear().unwrap();
    assert_eq!(mem.count().unwrap(), 0);
}

#[test]
fn test_empty_recall_returns_empty() {
    let mem = m();
    assert!(mem.recall("anything", 5).unwrap().is_empty());
}

#[test]
fn test_top_k_limit() {
    let mem = m();
    for i in 0..10 {
        mem.remember(&format!("memory {i} about coding in Rust"))
            .unwrap();
    }
    let results = mem.recall("Rust coding", 3).unwrap();
    assert!(results.len() <= 3);
}

#[test]
fn test_whitespace_only_input_ignored() {
    let mem = m();
    let ids = mem.remember("   ").unwrap();
    assert!(ids.is_empty());
    assert_eq!(mem.count().unwrap(), 0);
}

#[test]
fn test_long_input_is_chunked() {
    let mem = m();
    // ~300 words — should exceed the 128-word default chunk size
    let long = (0..300)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let ids = mem.remember(&long).unwrap();
    assert!(ids.len() > 1, "long input should produce multiple chunks");
}

// ─── FIX #7: Behavioral tests ─────────────────────────────────────────────────

#[test]
fn test_concurrent_remember_recall() {
    let mem = Arc::new(Memoire::in_memory().expect("in_memory"));
    let mut handles = Vec::new();
    for t in 0..10 {
        let m = Arc::clone(&mem);
        handles.push(std::thread::spawn(move || {
            for i in 0..20 {
                m.remember(&format!(
                    "Thread {t} iteration {i}: always validate external inputs before processing"
                ))
                .expect("remember failed");
                m.recall("input validation security", 3)
                    .expect("recall failed");
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }
    assert!(
        mem.count().unwrap() > 0,
        "should have stored at least one memory"
    );
}

#[test]
fn test_fingerprint_stability() {
    let content = "Always use parameterized queries to prevent SQL injection";
    let mem = Memoire::in_memory().unwrap();
    let ids1 = mem.remember(content).unwrap();
    assert!(!ids1.is_empty(), "first insert should succeed");
    // Second insert of identical content must be deduplicated by BLAKE3 fingerprint
    let ids2 = mem.remember(content).unwrap();
    assert!(
        ids2.is_empty(),
        "duplicate content should be rejected by fingerprint check"
    );
    // Count must not grow
    assert_eq!(
        mem.count().unwrap(),
        ids1.len() as i64,
        "store count must equal first-insert chunk count"
    );
}

#[test]
fn test_reinforcement_gate_rejects_low_overlap() {
    let mem = Memoire::in_memory().unwrap();
    let ids = mem
        .remember("Always use Decimal for financial calculations, never float")
        .unwrap();
    assert!(!ids.is_empty());
    let id = ids[0];

    // Read trust before reinforcement attempt
    let before = mem
        .recall("financial calculations", 1)
        .unwrap()
        .into_iter()
        .find(|m| m.id == id)
        .map(|m| m.score)
        .unwrap_or(0.0);

    // Agent output is completely unrelated — attribution gate must reject
    let reinforced = mem
        .reinforce_if_used(id, "deployed kubernetes pod in staging environment", true)
        .unwrap();
    assert!(
        !reinforced,
        "reinforcement must be rejected when output has no overlap with memory"
    );

    let after = mem
        .recall("financial calculations", 1)
        .unwrap()
        .into_iter()
        .find(|m| m.id == id)
        .map(|m| m.score)
        .unwrap_or(0.0);

    assert!(
        (after - before).abs() < 0.05,
        "trust must not increase when reinforcement gate rejects (before={before:.3}, after={after:.3})"
    );
}

#[test]
fn test_semantic_contradiction_detection() {
    // ── Part 1: verify polarity detection is real lexical logic ──────────────
    assert_eq!(
        detect_polarity("Never use floats for financial calculations"),
        Polarity::Negative,
        "detect_polarity must classify 'never' as Negative"
    );
    assert_eq!(
        detect_polarity("Always use floats for financial calculations"),
        Polarity::Affirmative,
        "detect_polarity must classify 'always' as Affirmative"
    );
    assert!(
        Polarity::Negative.opposes(&Polarity::Affirmative),
        "Negative must oppose Affirmative"
    );
    assert!(
        !Polarity::Neutral.opposes(&Polarity::Neutral),
        "Neutral must not oppose Neutral"
    );

    // ── Part 2: verify Polarity::from_stored correctly round-trips stored tags ─
    // This is the fix for the critical bug where detect_polarity was called on
    // "affirmative"/"negative" strings — finding no tokens, returning Neutral,
    // and silently disabling all contradiction detection.
    assert_eq!(Polarity::from_stored("affirmative"), Polarity::Affirmative);
    assert_eq!(Polarity::from_stored("negative"), Polarity::Negative);
    assert_eq!(Polarity::from_stored("neutral"), Polarity::Neutral);
    assert_eq!(Polarity::from_stored(""), Polarity::Neutral);

    // ── Part 3: end-to-end contradiction resolution pipeline ─────────────────
    let mem = Memoire::in_memory().unwrap();

    // Near-paraphrase pair: differ only in polarity word, share all other tokens.
    // MiniLM cosine similarity for such pairs is typically ≥ 0.90, well above the 0.80 gate.
    let ids1 = mem
        .remember("Never cache authentication tokens in local storage under any circumstances")
        .unwrap();
    let ids2 = mem
        .remember("Always cache authentication tokens in local storage for performance")
        .unwrap();

    // Both must be stored (distinct content → distinct BLAKE3 fingerprints)
    assert!(!ids1.is_empty(), "first memory must be stored");
    assert!(!ids2.is_empty(), "second memory must be stored");

    // After contradiction resolution, the active count must be strictly less than
    // total inserted — the loser must be archived. Near-paraphrase pairs reliably
    // exceed the 0.80 cosine gate under MiniLM.
    let active = mem.count().unwrap();
    let total = (ids1.len() + ids2.len()) as i64;
    assert!(
        active < total,
        "contradiction resolution must archive the loser: stored {total} but active={active}"
    );

    // Recall must still work and return valid scores
    let results = mem.recall("authentication token cache storage", 5).unwrap();
    assert!(
        !results.is_empty(),
        "at least one memory must be recallable"
    );
    for m in &results {
        assert!(
            (0.0..=1.0).contains(&m.score),
            "score out of range: {}",
            m.score
        );
        assert!(
            (0.0..=1.0).contains(&m.trust),
            "trust out of range: {}",
            m.trust
        );
    }
}

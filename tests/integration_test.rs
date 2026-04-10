use memoire::Memoire;

fn m() -> Memoire {
    Memoire::in_memory().expect("failed to create in-memory Memoire")
}

#[test]
fn test_remember_and_count() {
    let mem = m();
    assert_eq!(mem.count().unwrap(), 0);
    mem.remember("Fixed the pagination bug in the user list endpoint").unwrap();
    assert!(mem.count().unwrap() >= 1);
}

#[test]
fn test_recall_relevance() {
    let mem = m();
    mem.remember("Fixed a null pointer dereference in the auth middleware").unwrap();
    mem.remember("Refactored the database connection pool for throughput").unwrap();
    mem.remember("Added unit tests for the payment module").unwrap();

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
        mem.remember(&format!("memory {i} about coding in Rust")).unwrap();
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
    let long = (0..300).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
    let ids = mem.remember(&long).unwrap();
    assert!(ids.len() > 1, "long input should produce multiple chunks");
}

//! Deterministic lifecycle tests (Step 8b-1). No LLM and no model download: a
//! hashed bag-of-words embedder stands in for the real one, so every assertion is
//! about the engine's state machine, not about embedding quality.

use memoire::embedder::EmbedProvider;
use memoire::quality::{AdmissionMode, ContradictionMode, ScoringConfig};
use memoire::Memoire;

struct BagEmbedder;

impl BagEmbedder {
    const DIM: usize = 256;

    fn embed_text(text: &str) -> Vec<f32> {
        let mut v = vec![0.0_f32; Self::DIM];
        for raw in text.split(|c: char| !c.is_ascii_alphanumeric()) {
            let token = raw.to_ascii_lowercase();
            if token.is_empty()
                || matches!(
                    token.as_str(),
                    "never" | "always" | "in" | "for" | "under" | "any"
                )
            {
                continue;
            }
            let h = token.bytes().fold(0xcbf29ce484222325_u64, |h, b| {
                (h ^ b as u64).wrapping_mul(0x100000001b3)
            });
            v[h as usize % Self::DIM] += 1.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        } else {
            v[0] = 1.0;
        }
        v
    }
}

impl EmbedProvider for BagEmbedder {
    fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| Self::embed_text(t)).collect())
    }

    fn dim(&self) -> usize {
        Self::DIM
    }
}

fn engine(config: ScoringConfig) -> Memoire {
    Memoire::in_memory_with_embedder(Box::new(BagEmbedder))
        .expect("in-memory Memoire")
        .with_scoring_config(config)
}

fn only_state(m: &Memoire) -> (String, String) {
    let all = m.export_all().unwrap();
    assert_eq!(all.len(), 1, "expected exactly one stored memory");
    (all[0].state.clone(), all[0].exposure.clone())
}

#[test]
fn default_admission_admits_everything_as_unverified_trial() {
    let m = engine(ScoringConfig::default());
    let ids = m.remember("blah").unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(only_state(&m), ("trial".to_string(), "trial".to_string()));
    let recalled = m.recall("blah", 3).unwrap();
    assert_eq!(recalled.len(), 1, "an unverified memory must be recallable");
    assert_eq!(recalled[0].exposure, "trial");
    assert!((recalled[0].trust - 0.5).abs() < 1e-6, "neutral prior");
}

#[test]
fn quality_gate_mode_hides_shadow_rows_until_promoted() {
    let m = engine(ScoringConfig {
        admission: AdmissionMode::QualityGate,
        ..ScoringConfig::default()
    });
    let id = m.remember("blah").unwrap()[0];
    assert_eq!(only_state(&m).0, "shadow");
    assert!(m.recall("blah", 3).unwrap().is_empty(), "shadow is hidden");
    let hidden = m.recall_with_options("blah", 3, true).unwrap();
    assert_eq!(hidden.len(), 1);
    assert_eq!(hidden[0].exposure, "hidden");

    // Plain recall is the baseline: it ignores states entirely.
    let plain = m.recall_plain("blah", 3).unwrap();
    assert_eq!(plain.len(), 1);
    assert_eq!(plain[0].exposure, "plain");

    assert!(m.promote_shadow(id).unwrap());
    assert_eq!(only_state(&m).0, "trial");
    assert_eq!(m.recall("blah", 3).unwrap().len(), 1);
}

#[test]
fn outcome_feedback_promotes_good_memories_and_quarantines_bad_ones() {
    let m = engine(ScoringConfig {
        trial_cap: 2,
        ..ScoringConfig::default()
    });
    let good = m.remember("quantize money with Decimal").unwrap()[0];
    let bad = m.remember("money rounding use float arithmetic").unwrap()[0];
    let query = "money rounding quantize decimal float";

    // Both are unverified and both surface (cap 2).
    assert_eq!(m.recall(query, 3).unwrap().len(), 2);

    for session in ["s1", "s2", "s3"] {
        assert!(m.reinforce_declared(good, Some(session)).unwrap());
    }
    m.penalize_if_used(&[bad], 1.0).unwrap();
    m.penalize_if_used(&[bad], 1.0).unwrap();

    let all = m.export_all().unwrap();
    let state_of = |id: i64| all.iter().find(|x| x.id == id).unwrap().state.clone();
    assert_eq!(state_of(good), "active");
    assert_eq!(state_of(bad), "quarantined");

    let results = m.recall(query, 3).unwrap();
    assert_eq!(
        results.len(),
        1,
        "the quarantined memory is no longer offered"
    );
    assert_eq!(results[0].id, good);
    assert_eq!(results[0].exposure, "follow");

    // The operator view still shows it, and an operator can undo a false positive.
    assert!(m
        .recall_with_options(query, 3, true)
        .unwrap()
        .iter()
        .any(|x| x.id == bad));
    assert!(m.unquarantine(bad).unwrap());
    assert_eq!(m.recall(query, 3).unwrap().len(), 2);
}

#[test]
fn export_does_not_launder_quarantined_memories() {
    let m = engine(ScoringConfig::default());
    let bad = m.remember("a harmful lesson").unwrap()[0];
    m.remember("a helpful lesson").unwrap();
    m.penalize_if_used(&[bad], 1.0).unwrap();
    m.penalize_if_used(&[bad], 1.0).unwrap();

    let snapshot = m.export_namespace().unwrap();
    let exported = snapshot["memories"].as_array().unwrap();
    assert_eq!(exported.len(), 1);
    assert_eq!(exported[0]["content"], "a helpful lesson");
}

#[test]
fn contradiction_mode_controls_whether_anything_is_archived() {
    let pair = [
        "Never cache authentication tokens in local storage under any circumstances",
        "Always cache authentication tokens in local storage for performance",
    ];
    for (mode, expected) in [
        (ContradictionMode::Off, 2),
        (ContradictionMode::Nli, 1),
        (ContradictionMode::Legacy, 1),
    ] {
        let m = engine(ScoringConfig {
            contradiction_mode: mode,
            ..ScoringConfig::default()
        });
        for text in pair {
            assert_eq!(m.remember(text).unwrap().len(), 1);
        }
        assert_eq!(m.count().unwrap(), expected, "mode {mode:?}");
    }
}

#[test]
fn reinforce_if_used_attributes_long_outputs_by_containment() {
    let m = engine(ScoringConfig::default());
    let lesson = "Quantize invoice totals with Decimal using rounding half even";
    let id = m.remember(lesson).unwrap()[0];

    let long_answer = format!("{} {}", "filler ".repeat(300), lesson);
    assert!(m
        .reinforce_if_used_with_session(id, &long_answer, true, Some("s1"))
        .unwrap());
    // Same session again: one vote per memory per session.
    assert!(!m
        .reinforce_if_used_with_session(id, &long_answer, true, Some("s1"))
        .unwrap());
    // Unrelated output: no attribution.
    assert!(!m
        .reinforce_if_used_with_session(id, &"filler ".repeat(300), true, Some("s2"))
        .unwrap());
    assert_eq!(m.export_all().unwrap()[0].reinforcement_count, 1);
}

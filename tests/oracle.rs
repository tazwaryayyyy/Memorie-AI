use memoire::{
    embedder::EmbedProvider,
    quality::{
        DefaultOracle, IngestDecision, IngestionOracle, ScoringPrototypes, ScoringWeights,
        StrongVerifyResult, WeakVerifyResult,
    },
    store::Memory,
    Memoire,
};

struct TestEmbedder;

impl TestEmbedder {
    const DIM: usize = 384;

    fn tokenize(text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter_map(|raw| {
                let token = raw.to_ascii_lowercase();
                let normalized = match token.as_str() {
                    "always" | "never" | "do" | "not" | "must" | "should" | "for" | "in"
                    | "the" | "a" | "an" | "to" | "of" | "and" | "or" | "under" | "any" | "all"
                    | "by" | "with" => return None,
                    _ => token,
                };
                Some(normalized)
            })
            .collect();

        if tokens.is_empty() {
            tokens.push(text.to_ascii_lowercase());
        }
        tokens
    }

    fn hash(token: &str) -> usize {
        token.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ byte as u64).wrapping_mul(0x100000001b3)
        }) as usize
    }

    fn embed_text(text: &str) -> Vec<f32> {
        let mut vector = vec![0.0_f32; Self::DIM];
        for token in Self::tokenize(text) {
            vector[Self::hash(&token) % Self::DIM] += 1.0;
        }

        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut vector {
                *value /= norm;
            }
        }
        vector
    }
}

impl EmbedProvider for TestEmbedder {
    fn embed(&self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| Self::embed_text(text)).collect())
    }

    fn dim(&self) -> usize {
        Self::DIM
    }
}

fn test_memoire() -> Memoire {
    Memoire::in_memory_with_embedder(Box::new(TestEmbedder))
        .expect("failed to create in-memory Memoire")
}

#[test]
fn test_default_oracle_verify_weak() {
    let _embedder = TestEmbedder;
    let prototypes = ScoringPrototypes::neutral(TestEmbedder::DIM);
    let oracle = DefaultOracle {
        weights: ScoringWeights::default(),
        prototypes,
        embedder: None,
        nli_checker: None,
    };

    // Rejects content with importance < 0.50 (e.g. short generic text)
    let text_low = "blah";
    let emb_low = TestEmbedder::embed_text(text_low);
    let res_low = oracle.verify_weak(text_low, &emb_low);
    assert!(!res_low.passed);

    // Passes content with importance >= 0.50 (with "always", numbers, etc.)
    let text_high = "always verify input, tested and confirmed on line 123";
    let emb_high = TestEmbedder::embed_text(text_high);
    let res_high = oracle.verify_weak(text_high, &emb_high);
    assert!(res_high.passed);
}

struct CustomOracle {
    mock_confidence: f64,
}

impl IngestionOracle for CustomOracle {
    fn verify_weak(&self, _content: &str, _embedding: &[f32]) -> WeakVerifyResult {
        WeakVerifyResult {
            passed: true,
            reason: None,
        }
    }

    fn verify_strong(
        &self,
        _content: &str,
        _embedding: &[f32],
        _namespace_sample: &[Memory],
    ) -> StrongVerifyResult {
        StrongVerifyResult {
            confidence: self.mock_confidence,
            coverage: 1.0,
            consistency: 1.0,
        }
    }
}

#[test]
fn test_decide_confidence_mappings() {
    let sample = vec![];
    let emb = vec![0.0_f32; TestEmbedder::DIM];

    // decide() maps confidence >= 0.95 to Active
    let oracle_active = CustomOracle {
        mock_confidence: 0.96,
    };
    assert_eq!(
        oracle_active.decide("test", &emb, &sample),
        IngestDecision::Active
    );

    // decide() maps confidence in [0.80, 0.95) to Probationary
    let oracle_prob = CustomOracle {
        mock_confidence: 0.85,
    };
    assert_eq!(
        oracle_prob.decide("test", &emb, &sample),
        IngestDecision::Probationary
    );

    // decide() maps confidence < 0.80 to Reject
    let oracle_reject = CustomOracle {
        mock_confidence: 0.75,
    };
    assert_eq!(
        oracle_reject.decide("test", &emb, &sample),
        IngestDecision::Reject
    );
}

struct RejectAllOracle;

impl IngestionOracle for RejectAllOracle {
    fn verify_weak(&self, _content: &str, _embedding: &[f32]) -> WeakVerifyResult {
        WeakVerifyResult {
            passed: false,
            reason: Some("reject all".to_string()),
        }
    }

    fn verify_strong(
        &self,
        _content: &str,
        _embedding: &[f32],
        _namespace_sample: &[Memory],
    ) -> StrongVerifyResult {
        StrongVerifyResult {
            confidence: 0.0,
            coverage: 0.0,
            consistency: 0.0,
        }
    }
}

#[test]
fn test_remember_with_reject_oracle_skips_chunk() {
    let mem = test_memoire().with_oracle(RejectAllOracle);
    let ids = mem.remember("always remember to validate input").unwrap();
    assert!(
        ids.is_empty(),
        "Reject oracle should result in no stored memories"
    );
    assert_eq!(mem.count().unwrap(), 0);
}

#[test]
fn test_probationary_memories_visibility() {
    // We want to force a memory into Probationary state
    let oracle = CustomOracle {
        mock_confidence: 0.85,
    };
    let mem = test_memoire().with_oracle(oracle);

    let ids = mem
        .remember("always validate inputs before processing")
        .unwrap();
    assert!(!ids.is_empty(), "should store probationary memory");

    // Probationary memories do not appear in recall() results
    let results_default = mem.recall("validate inputs", 5).unwrap();
    assert!(
        results_default.is_empty(),
        "Probationary memories must be hidden by default"
    );

    // Probationary memories appear when include_hidden=true in recall()
    let results_shadow = mem.recall_with_options("validate inputs", 5, true).unwrap();
    assert_eq!(
        results_shadow.len(),
        1,
        "Probationary memories must be returned when include_hidden=true"
    );
    assert_eq!(results_shadow[0].state, "probationary");
}

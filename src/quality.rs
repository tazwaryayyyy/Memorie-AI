use std::time::{SystemTime, UNIX_EPOCH};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for trust scoring and reinforcement attribution.
///
/// All fields are tuneable after construction via `Memoire::with_scoring_config()`.
/// These are engineering heuristics with default values chosen as reasonable starting
/// points. Tune based on observed agent behaviour; there is no universal calibration.
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Weight applied to the newly computed trust value in EMA updates.
    /// Range: (0.0, 1.0). Default 0.3 (30% new value, 70% historical EMA).
    /// Lower values → slower trust changes, more stable but less responsive.
    pub ema_new_weight: f32,
    /// Saturation constant `k` in the reinforcement term `rc / (rc + k)`.
    /// Default 3.0: rc=3 reaches ~0.5, rc=9 reaches ~0.75 of the reinforcement ceiling.
    /// Increase to require more task confirmations before a memory reaches FOLLOW.
    pub rc_saturation: f32,
    /// Minimum Jaccard token overlap required to attribute reinforcement to a memory.
    /// Range: [0.0, 1.0]. Default 0.35.
    /// Higher → stricter attribution, fewer false-positive trust inflations.
    pub jaccard_threshold: f32,
    /// Memory count above which HNSW approximate search is used instead of linear scan.
    /// Default 500. Below this, a linear scan over the in-memory embedding cache is
    /// faster due to lower constant factors than HNSW index construction.
    pub hnsw_threshold: usize,
    /// Multiplied by quality_score to compute the initial trust_ema for a new memory
    /// that has no reinforcement history (reinforcement_count == 0).
    /// Range: 0.0–1.0. Default: 0.5
    /// Set to 0.0 to disable cold-start trust (original behavior).
    pub cold_start_weight: f32,
    /// Exponential decay rate applied to trust_ema per day since last use.
    /// Effective trust = trust_ema * exp(-decay_rate * days_since_last_used)
    /// Default: 0.01 (half-life ~69 days). Set to 0.0 to disable.
    pub decay_rate: f32,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            ema_new_weight: 0.3,
            rc_saturation: 3.0,
            jaccard_threshold: 0.35,
            hnsw_threshold: 500,
            cold_start_weight: 0.5,
            decay_rate: 0.01,
        }
    }
}

// ─── Semantic Prototypes ──────────────────────────────────────────────────────

/// Pre-computed centroid vectors for semantic quality feature scoring.
///
/// Each centroid is the mean embedding of multiple exemplar sentences for that
/// category, computed via the same ONNX model used for all other embeddings.
/// Computed once at `Memoire` initialisation time and reused for every ingestion.
///
/// Using semantic centroids rather than keyword lists means semantically
/// equivalent phrases ("safeguard" ≈ "security", "mandate" ≈ "always") score
/// similarly without maintaining a vocabulary list.
pub struct ScoringPrototypes {
    /// Centroid for high-consequence memories (security incidents, outages, data loss).
    pub consequence: Vec<f32>,
    /// Centroid for high-actionability memories (concrete changes, fixes, patches applied).
    pub actionability: Vec<f32>,
    /// Centroid for high-reusability memories (rules, policies, invariants, "always/never").
    pub reusability: Vec<f32>,
    /// True if prototypes were computed from the real model; false if using neutral fallback.
    pub is_semantic: bool,
}

impl ScoringPrototypes {
    /// Neutral fallback: all feature scores will be 0.5 (no signal).
    /// Used when the model fails to initialise or the embedder dimension is unknown.
    pub fn neutral(dim: usize) -> Self {
        Self {
            consequence: vec![0.0; dim],
            actionability: vec![0.0; dim],
            reusability: vec![0.0; dim],
            is_semantic: false,
        }
    }
}

// ─── Polarity Detection ───────────────────────────────────────────────────────

/// Polarity of a memory's claim: whether it asserts or negates a practice.
///
/// Used in generalised contradiction detection to identify when two semantically
/// similar memories make opposing claims, regardless of domain.
#[derive(Debug, Clone, PartialEq)]
pub enum Polarity {
    Affirmative,
    Negative,
    Neutral,
}

impl Polarity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Polarity::Affirmative => "affirmative",
            Polarity::Negative => "negative",
            Polarity::Neutral => "neutral",
        }
    }

    /// Parse the stored polarity tag back to its enum variant.
    ///
    /// `detect_polarity` must NOT be called on the stored string — it scans for
    /// tokens like "never"/"always" which don't appear in "affirmative"/"negative".
    /// Always use this method to deserialise a polarity stored in `claim_value`.
    pub fn from_stored(s: &str) -> Self {
        match s {
            "affirmative" => Polarity::Affirmative,
            "negative" => Polarity::Negative,
            _ => Polarity::Neutral,
        }
    }

    /// Returns true iff self and other are opposing (Affirmative vs Negative).
    pub fn opposes(&self, other: &Polarity) -> bool {
        matches!(
            (self, other),
            (Polarity::Affirmative, Polarity::Negative)
                | (Polarity::Negative, Polarity::Affirmative)
        )
    }
}

/// Detect whether a text asserts or negates a practice.
///
/// This is a lightweight lexical heuristic over a small, high-precision token set.
/// It is intentionally documented as a heuristic — not NLI-level inference.
/// Its role is to flag *potential* contradictions for quality-weighted resolution.
/// False negatives (missed contradictions) are harmless; false positives are handled
/// by the quality-weighted archiving step in `resolve_contradictions_for_id`.
pub fn detect_polarity(text: &str) -> Polarity {
    let norm = normalize_text(text);
    let words: Vec<&str> = norm.split_whitespace().collect();

    const NEGATIVE: &[&str] = &[
        "never",
        "avoid",
        "don't",
        "do not",
        "cannot",
        "must not",
        "should not",
        "prohibited",
        "forbidden",
        "disable",
        "disabled",
        "remove",
        "removed",
        "deprecated",
        "stop",
        "stopped",
        "revert",
        "reverted",
        "rollback",
        "dangerous",
        "insecure",
        "vulnerable",
        "broken",
    ];

    const AFFIRMATIVE: &[&str] = &[
        "always",
        "ensure",
        "required",
        "enable",
        "enabled",
        "implement",
        "implemented",
        "enforce",
        "enforced",
        "mandate",
        "prefer",
        "recommend",
        "correct",
        "fixed",
        "patched",
        "replaced",
        "upgraded",
        "migrated",
    ];

    // Check multi-word tokens against the full normalised text (handles "do not", "must not").
    let neg_hits = NEGATIVE
        .iter()
        .filter(|&&t| {
            if t.contains(' ') {
                norm.contains(t)
            } else {
                words.contains(&t)
            }
        })
        .count();
    let aff_hits = AFFIRMATIVE
        .iter()
        .filter(|&&t| {
            if t.contains(' ') {
                norm.contains(t)
            } else {
                words.contains(&t)
            }
        })
        .count();

    if neg_hits > aff_hits {
        Polarity::Negative
    } else if aff_hits > neg_hits {
        Polarity::Affirmative
    } else {
        Polarity::Neutral
    }
}

// ─── Core Types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum IngestDecision {
    Active,
    Shadow,
    Reject,
}

#[derive(Debug, Clone)]
pub struct QualityFeatures {
    pub actionability: f32,
    pub consequence: f32,
    pub novelty: f32,
    pub reusability: f32,
    pub evidence: f32,
}

/// Weights applied to quality features when computing the importance score.
///
/// Supply a custom profile via `Memoire::with_scoring_weights()` for domain-specific
/// weighting without recompiling the Rust core (e.g. raise `novelty` for research
/// agents where unique facts matter more than actionability).
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    pub actionability: f32,
    pub consequence: f32,
    pub novelty: f32,
    pub reusability: f32,
    pub evidence: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            actionability: 0.30,
            consequence: 0.25,
            novelty: 0.20,
            reusability: 0.15,
            evidence: 0.10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QualityMeta {
    pub importance_base: f32,
    pub confidence: f32,
    pub novelty: f32,
    pub actionability: f32,
    pub evidence: f32,
    pub consequence: f32,
    pub reusability: f32,
    pub source_kind: String,
    pub store_state: String,
    pub reinforcement_count: i64,
    pub last_accessed_at: Option<i64>,
    pub superseded_by: Option<i64>,
    pub contradiction_group: Option<String>,
    pub archived: i64,
    pub effective_weight: f32,
    pub fingerprint: String,
    pub claim_key: Option<String>,
    pub claim_value: Option<String>,
}

impl QualityMeta {
    pub fn default_active(content: &str) -> Self {
        let now = now_ts();
        Self {
            importance_base: 0.5,
            confidence: 0.5,
            novelty: 0.5,
            actionability: 0.5,
            evidence: 0.5,
            consequence: 0.5,
            reusability: 0.5,
            source_kind: "agent".to_string(),
            store_state: "active".to_string(),
            reinforcement_count: 0,
            last_accessed_at: Some(now),
            superseded_by: None,
            contradiction_group: None,
            archived: 0,
            effective_weight: 0.5,
            fingerprint: fingerprint(content),
            claim_key: None,
            claim_value: None,
        }
    }
}

pub fn now_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// ✅ FIX #2 — Stable content fingerprint using BLAKE3.
///
/// BLAKE3 produces a deterministic 256-bit hash regardless of Rust version,
/// platform, or compiler flags. The previous `DefaultHasher` was explicitly
/// non-stable across Rust versions, silently breaking deduplication after
/// `rustup update`. Input is case-folded and whitespace-normalised before
/// hashing so semantically identical text always deduplicates.
pub fn fingerprint(text: &str) -> String {
    let norm = normalize_text(text);
    blake3::hash(norm.as_bytes()).to_hex().to_string()
}

/// Cosine similarity between two vectors. Returns 0.0 if lengths differ or either is zero.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub fn score_importance(features: &QualityFeatures, weights: &ScoringWeights) -> f32 {
    (weights.actionability * features.actionability
        + weights.consequence * features.consequence
        + weights.novelty * features.novelty
        + weights.reusability * features.reusability
        + weights.evidence * features.evidence)
        .clamp(0.0, 1.0)
}

pub fn decide_store(score: f32, dup_similarity: f32) -> IngestDecision {
    if dup_similarity > 1.001 {
        return IngestDecision::Reject;
    }
    if score >= 0.50 {
        IngestDecision::Active
    } else {
        IngestDecision::Shadow
    }
}

pub fn effective_weight(base: f32, age_days: f32, reinforcement_count: i64) -> f32 {
    let decay = (-0.035_f32 * age_days).exp();
    let boost = 1.0 + 0.12 * (1.0 + reinforcement_count as f32).ln();
    (base * decay * boost).clamp(0.05, 2.0)
}

pub fn recency_bonus(created_at: i64, now: i64) -> f32 {
    let age_days = ((now - created_at).max(0) as f32) / 86_400.0;
    (1.0 / (1.0 + age_days / 7.0)).clamp(0.0, 1.0)
}

/// ✅ FIX #3 — Semantic feature extraction via prototype cosine similarity.
///
/// Each quality dimension is scored as the cosine similarity between the memory's
/// embedding and a pre-computed centroid for that category. This replaces the
/// previous bag-of-words keyword matching which could not handle semantic
/// equivalents (e.g. "safeguard" vs "security", "mandate" vs "always").
///
/// Evidence is retained as a structural heuristic: presence of numbers and
/// measurement terms reliably indicates an empirical basis and does not require
/// semantic understanding to detect.
///
/// When `prototypes.is_semantic` is false (neutral fallback), semantic dimensions
/// return 0.5 (no signal). A warning is logged at Store initialisation if this occurs.
pub fn extract_features(
    embedding: &[f32],
    text: &str,
    prototypes: &ScoringPrototypes,
    novelty: f32,
) -> QualityFeatures {
    let (consequence, actionability, reusability) = if prototypes.is_semantic
        && !prototypes.consequence.is_empty()
        && embedding.len() == prototypes.consequence.len()
    {
        (
            cosine_similarity(embedding, &prototypes.consequence).clamp(0.0, 1.0),
            cosine_similarity(embedding, &prototypes.actionability).clamp(0.0, 1.0),
            cosine_similarity(embedding, &prototypes.reusability).clamp(0.0, 1.0),
        )
    } else {
        (0.5, 0.5, 0.5)
    };

    // Evidence: high-precision structural signals that complement semantic scoring.
    // Presence of measurement data is a reliable indicator of empirical grounding.
    let norm = normalize_text(text);
    let has_number = norm.chars().any(|c| c.is_ascii_digit());
    let evidence_terms = [
        "confirmed",
        "metrics",
        "benchmark",
        "latency",
        "error rate",
        "measured",
        "tested",
        "verified",
    ];
    let evidence_hits = evidence_terms.iter().filter(|&&t| norm.contains(t)).count() as f32;
    let evidence = (evidence_hits / 3.0 + if has_number { 0.15 } else { 0.0 }).clamp(0.0, 1.0);

    QualityFeatures {
        actionability,
        consequence,
        novelty: novelty.clamp(0.0, 1.0),
        reusability,
        evidence,
    }
}

/// Compute a trust score in [0.0, 1.0] at recall time.
///
/// Weight rationale (heuristic defaults, tunable via ScoringConfig):
///   0.35 reinforcement — most direct signal: memory was acted on and helped
///   0.25 confidence    — initial quality signal from ingestion evidence
///   0.20 age_term      — fresh memories are more likely to still be accurate
///   0.15 importance    — ingestion-time importance score
///   0.05 contradiction — small bonus for surviving a contradiction resolution
///
/// Trust interpretation:
///   ≥ 0.75 → FOLLOW (act on it)
///   ≥ 0.45 → HINT (verify before acting)
///   < 0.45 → IGNORE
pub fn compute_trust(
    reinforcement_count: i64,
    contradiction_survived: bool,
    store_state: &str,
    importance_base: f32,
    confidence: f32,
    age_days: f32,
    config: &ScoringConfig,
) -> f32 {
    let state_weight: f32 = match store_state {
        "active" => 1.0,
        "shadow" => 0.6,
        _ => return 0.0,
    };

    let rc = reinforcement_count as f32;
    let reinforcement_term = rc / (rc + config.rc_saturation);
    let age_term = (-0.02_f32 * age_days).exp().clamp(0.0, 1.0);
    let contradiction_bonus: f32 = if contradiction_survived { 1.0 } else { 0.0 };

    let inner = 0.35 * reinforcement_term
        + 0.25 * confidence.clamp(0.0, 1.0)
        + 0.20 * age_term
        + 0.15 * importance_base.clamp(0.0, 1.0)
        + 0.05 * contradiction_bonus;

    let base_trust = state_weight * inner;

    // Head-start for brand-new, high-quality active memories so critical lessons
    // can reach the HINT threshold before any reinforcement has occurred.
    // Only fires for active, never-reinforced, contradiction-free memories
    // that scored highly on both importance and confidence at ingestion.
    let fast_track = if reinforcement_count == 0
        && store_state == "active"
        && importance_base > 0.8
        && confidence > 0.7
        && !contradiction_survived
    {
        0.15_f32
    } else {
        0.0
    };

    (base_trust + fast_track).clamp(0.0, 1.0)
}

/// Build quality metadata for a memory chunk.
///
/// ✅ FIX #3 / #4: Takes the pre-computed embedding for semantic feature scoring
/// and stores the polarity tag for generalised contradiction detection.
/// The `claim_key`/`claim_value` fields are populated later by
/// `resolve_contradictions_for_id` when a contradiction is confirmed.
pub fn build_quality_meta(
    content: &str,
    embedding: &[f32],
    novelty: f32,
    source_kind: &str,
    weights: &ScoringWeights,
    prototypes: &ScoringPrototypes,
) -> (QualityMeta, IngestDecision) {
    let features = extract_features(embedding, content, prototypes, novelty);
    let score = score_importance(&features, weights);
    let decision = decide_store(score, 1.0 - novelty);
    let polarity = detect_polarity(content);

    let state = match decision {
        IngestDecision::Active => "active",
        IngestDecision::Shadow => "shadow",
        IngestDecision::Reject => "rejected",
    }
    .to_string();

    let meta = QualityMeta {
        importance_base: score,
        confidence: (0.50 + 0.30 * features.evidence + 0.20 * features.actionability)
            .clamp(0.0, 1.0),
        novelty: features.novelty,
        actionability: features.actionability,
        evidence: features.evidence,
        consequence: features.consequence,
        reusability: features.reusability,
        source_kind: source_kind.to_string(),
        store_state: state,
        reinforcement_count: 0,
        last_accessed_at: Some(now_ts()),
        superseded_by: None,
        contradiction_group: None,
        archived: 0,
        effective_weight: score,
        fingerprint: fingerprint(content),
        claim_key: None,
        claim_value: Some(polarity.as_str().to_string()),
    };

    (meta, decision)
}

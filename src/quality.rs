use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{SystemTime, UNIX_EPOCH};

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

pub fn fingerprint(text: &str) -> String {
    let norm = normalize_text(text);
    let mut hasher = DefaultHasher::new();
    norm.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

// SCORING FROZEN — weights and thresholds are fixed for reproducibility.
// Do not adjust without forking the quality module.
pub fn score_importance(features: &QualityFeatures) -> f32 {
    (0.30 * features.actionability
        + 0.25 * features.consequence
        + 0.20 * features.novelty
        + 0.15 * features.reusability
        + 0.10 * features.evidence)
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

pub fn extract_claim(text: &str) -> Option<(String, String)> {
    let t = normalize_text(text);

    if t.contains("issuer") && (t.contains("validate") || t.contains("validation")) {
        let v = if t.contains("not") || t.contains("disabled") {
            "disabled"
        } else {
            "enabled"
        };
        return Some(("auth.jwt.issuer_validation".to_string(), v.to_string()));
    }

    if t.contains("bcrypt") && t.contains("argon2") {
        return Some(("auth.password.hashing".to_string(), "argon2id".to_string()));
    }

    if t.contains("rate limiting") || t.contains("rate limit") {
        let mut value = "enabled".to_string();
        for token in t.split_whitespace() {
            if token.contains("/hr") || token.contains("/hour") || token.contains("req") {
                value = token.to_string();
                break;
            }
        }
        return Some(("api.rate_limit".to_string(), value));
    }

    if t.contains("float") && t.contains("money") {
        return Some((
            "billing.money.numeric_type".to_string(),
            "decimal".to_string(),
        ));
    }

    None
}

pub fn extract_features(content: &str, novelty: f32) -> QualityFeatures {
    let norm = normalize_text(content);
    let word_count = norm.split_whitespace().count() as f32;

    let action_verbs = [
        "fixed",
        "replaced",
        "migrated",
        "added",
        "removed",
        "patched",
        "rolled",
        "disabled",
        "enabled",
        "refactored",
    ];
    let consequence_terms = [
        "security",
        "incident",
        "outage",
        "critical",
        "forged",
        "breach",
        "data loss",
        "prod",
        "production",
        "race condition",
    ];
    let evidence_terms = [
        "test",
        "latency",
        "error rate",
        "dropped",
        "improved",
        "confirmed",
        "metrics",
        "benchmark",
    ];
    let reusable_terms = ["always", "never", "must", "limit", "policy", "rule"];

    let action_hits = action_verbs.iter().filter(|t| norm.contains(**t)).count() as f32;
    let consequence_hits = consequence_terms
        .iter()
        .filter(|t| norm.contains(**t))
        .count() as f32;
    let evidence_hits = evidence_terms.iter().filter(|t| norm.contains(**t)).count() as f32;
    let reuse_hits = reusable_terms.iter().filter(|t| norm.contains(**t)).count() as f32;

    let has_number = norm.chars().any(|c| c.is_ascii_digit());

    let actionability = (action_hits / 4.0 + if has_number { 0.1 } else { 0.0 }).clamp(0.0, 1.0);
    let consequence = (consequence_hits / 3.0).clamp(0.0, 1.0);
    let reusability = (reuse_hits / 3.0 + if word_count > 8.0 { 0.1 } else { 0.0 }).clamp(0.0, 1.0);
    let evidence = (evidence_hits / 3.0 + if has_number { 0.1 } else { 0.0 }).clamp(0.0, 1.0);

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
/// Weights rationale:
///   0.35 reinforcement — most direct signal: this memory was acted on and it helped
///   0.25 confidence    — initial quality signal from ingestion evidence + actionability
///   0.20 age_term      — fresh memories are more likely still accurate
///   0.15 importance    — ingestion-time importance score (actionability + consequence)
///   0.05 contradiction — small bonus for surviving a contradiction resolution
///
/// Edge cases:
///   - rc=0 (brand new): reinforcement_term=0, trust comes from confidence/age/importance only
///   - shadow state: all terms multiplied by 0.6 — low trust backfill
///   - rejected/superseded: state_weight=0.0, returns 0.0 immediately
// SCORING FROZEN — trust formula is immutable after calibration.
pub fn compute_trust(
    reinforcement_count: i64,
    contradiction_survived: bool,
    store_state: &str,
    importance_base: f32,
    confidence: f32,
    age_days: f32,
) -> f32 {
    let state_weight: f32 = match store_state {
        "active" => 1.0,
        "shadow" => 0.6,
        _ => return 0.0,
    };

    // Saturates asymptotically: rc=0 → 0.0, rc=1 → 0.25, rc=3 → 0.5, rc=9 → 0.75
    let rc = reinforcement_count as f32;
    let reinforcement_term = rc / (rc + 3.0);

    // Slower age decay for trust than weight decay (0.02 vs 0.035)
    let age_term = (-0.02_f32 * age_days).exp().clamp(0.0, 1.0);

    let contradiction_bonus: f32 = if contradiction_survived { 1.0 } else { 0.0 };

    let inner = 0.35 * reinforcement_term
        + 0.25 * confidence.clamp(0.0, 1.0)
        + 0.20 * age_term
        + 0.15 * importance_base.clamp(0.0, 1.0)
        + 0.05 * contradiction_bonus;

    let base_trust = state_weight * inner;

    // Fast-track boost: a brand-new active memory with high importance AND
    // high confidence (strong evidence at ingestion) gets a head-start so
    // critical lessons can reach HINT/FOLLOW without waiting for reinforcement.
    // Only fires for active memories that have never been reinforced and are
    // not entangled in a contradiction group (clean, fresh, authoritative).
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

pub fn build_quality_meta(
    content: &str,
    novelty: f32,
    source_kind: &str,
) -> (QualityMeta, IngestDecision) {
    let features = extract_features(content, novelty);
    let score = score_importance(&features);
    let decision = decide_store(score, 1.0 - novelty);
    let claim = extract_claim(content);

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
        contradiction_group: claim.as_ref().map(|(k, _)| k.clone()),
        archived: 0,
        effective_weight: score,
        fingerprint: fingerprint(content),
        claim_key: claim.as_ref().map(|(k, _)| k.clone()),
        claim_value: claim.as_ref().map(|(_, v)| v.clone()),
    };

    (meta, decision)
}

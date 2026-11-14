use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::time::{SystemTime, UNIX_EPOCH};

// ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ NLI-style Contradiction Detection ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

/// Result of an NLI-style entailment check between two memories.
#[derive(Debug, Clone, PartialEq)]
pub enum NliLabel {
    /// Memory A contradicts memory B (one negates what the other asserts).
    Contradiction,
    /// Memory A is consistent with memory B (no opposing polarity detected).
    Neutral,
    /// Memory A entails / confirms memory B (same claim, similar polarity).
    Entailment,
}

/// Pairwise NLI classifier. Implemented by [`OnnxNliChecker`]; unit tests use stubs
/// so no model download is needed.
pub(crate) trait NliClassifier: Send + Sync {
    fn classify(&self, premise: &str, hypothesis: &str) -> anyhow::Result<NliLabel>;
}

/// Lightweight NLI-style contradiction detector.
///
/// Uses a three-signal ensemble ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no external model required:
///
/// 1. **Cosine geometry**: requires high cosine similarity (same topic cluster).
/// 2. **Polarity opposition**: lexical negation heuristic (fast, high-precision).
/// 3. **Negation asymmetry score**: counts negation tokens in one text but not
///    the other. High asymmetry = high contradiction probability.
///
/// This is intentionally cheaper than loading an ONNX NLI model at runtime
/// while providing more signal than pure keyword polarity used before.
///
/// If an ONNX NLI model is installed, its label is **authoritative** for pairs that
/// pass the cosine topic gate; the lexical heuristics only run when no model is
/// installed or the model errors at inference time.
#[derive(Clone)]
pub struct NliChecker {
    pub(crate) onnx: Option<Arc<dyn NliClassifier>>,
}

/// Maps ONNX NLI logit indices to labels, read from the model's `config.json`
/// `id2label`. Label order differs between models (e.g. `bart-large-mnli`:
/// 0=contradiction; `MoritzLaurer/*`: 2=contradiction), so it is never assumed.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub(crate) struct NliLabelMap {
    by_index: [NliLabel; 3],
}

#[cfg(not(target_arch = "wasm32"))]
impl NliLabelMap {
    /// Parse `id2label` (case-insensitive `contradiction` / `entailment` / `neutral`).
    /// Errors on anything missing, ambiguous or unrecognised instead of guessing.
    pub(crate) fn from_id2label_json(config_json: &str) -> anyhow::Result<Self> {
        let root: serde_json::Value = serde_json::from_str(config_json)
            .map_err(|e| anyhow::anyhow!("config.json is not valid JSON: {e}"))?;
        let id2label = root
            .get("id2label")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "config.json has no `id2label` object; cannot tell which logit is contradiction"
                )
            })?;
        if id2label.len() != 3 {
            anyhow::bail!(
                "id2label must have exactly 3 entries (contradiction, entailment, neutral), found {}",
                id2label.len()
            );
        }

        let mut slots: [Option<NliLabel>; 3] = [None, None, None];
        for (key, value) in id2label {
            let idx: usize = key
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("id2label key {key:?} is not an integer index"))?;
            if idx >= 3 {
                anyhow::bail!("id2label index {idx} is out of range (expected 0..=2)");
            }
            let name = value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("id2label[{idx}] is not a string"))?;
            let label = match name.trim().to_ascii_lowercase().as_str() {
                "contradiction" => NliLabel::Contradiction,
                "entailment" => NliLabel::Entailment,
                "neutral" => NliLabel::Neutral,
                _ => anyhow::bail!(
                    "id2label[{idx}] = {name:?} is not contradiction/entailment/neutral \
                     (case-insensitive); refusing to guess the label order"
                ),
            };
            if slots[idx].is_some() {
                anyhow::bail!("id2label index {idx} appears more than once");
            }
            slots[idx] = Some(label);
        }

        let [Some(l0), Some(l1), Some(l2)] = slots else {
            anyhow::bail!("id2label must define indices 0, 1 and 2");
        };
        if l0 == l1 || l0 == l2 || l1 == l2 {
            anyhow::bail!("id2label maps two indices to the same label");
        }
        Ok(Self {
            by_index: [l0, l1, l2],
        })
    }

    pub(crate) fn from_config_file(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!(
                "cannot read {}: {e}. An NLI model directory must contain config.json \
                 (with id2label) next to the .onnx file",
                path.display()
            )
        })?;
        Self::from_id2label_json(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    }

    /// Argmax over exactly three finite logits, mapped through `id2label`.
    pub(crate) fn label_for_logits(&self, logits: &[f32]) -> anyhow::Result<NliLabel> {
        if logits.len() != 3 {
            anyhow::bail!("NLI model returned {} logits, expected 3", logits.len());
        }
        if logits.iter().any(|v| !v.is_finite()) {
            anyhow::bail!("NLI model returned non-finite logits");
        }
        let best = logits
            .iter()
            .enumerate()
            .fold(0, |b, (i, v)| if *v > logits[b] { i } else { b });
        Ok(self.by_index[best].clone())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct OnnxNliChecker {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    max_tokens: usize,
    labels: NliLabelMap,
}

#[cfg(not(target_arch = "wasm32"))]
impl OnnxNliChecker {
    /// Load `model_path` plus `tokenizer.json` and `config.json` from the same
    /// directory. Fails if `config.json` / `id2label` is missing or ambiguous.
    pub fn from_path(model_path: &Path) -> anyhow::Result<Self> {
        if !model_path.exists() {
            anyhow::bail!("onnx model path does not exist: {}", model_path.display());
        }
        let parent = model_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;
        // Cheap checks first: fail before building the ONNX session.
        let labels = NliLabelMap::from_config_file(&parent.join("config.json"))?;

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        let tokenizer_path = parent.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            )
        })?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_tokens: 512,
            labels,
        })
    }

    /// Classify `(text_a, text_b)` as contradiction / entailment / neutral.
    pub fn classify(&self, text_a: &str, text_b: &str) -> anyhow::Result<NliLabel> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: self.max_tokens,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("Failed to configure tokenizer truncation: {}", e))?;

        let encoding = tokenizer
            .encode((text_a, text_b), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let len = ids.len();

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX NLI session lock poisoned"))?;
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let ids_arr = Tensor::from_array(([1usize, len], ids.into_boxed_slice()))?;
        let mask_arr = Tensor::from_array(([1usize, len], mask.into_boxed_slice()))?;

        let mut session_inputs = ort::inputs![
            "input_ids" => ids_arr,
            "attention_mask" => mask_arr,
        ];

        if has_token_type_ids {
            let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();
            let type_ids_arr = Tensor::from_array(([1usize, len], type_ids.into_boxed_slice()))?;
            session_inputs.push(("token_type_ids".into(), type_ids_arr.upcast().into()));
        }

        let outputs = session.run(session_inputs)?;
        if outputs.len() == 0 {
            anyhow::bail!("No output returned from NLI model");
        }
        let output_value = outputs.get("logits").unwrap_or(&outputs[0]);
        let (_, logits) = output_value.try_extract_tensor::<f32>()?;

        self.labels.label_for_logits(logits)
    }

    /// `true` iff the model's label for the pair is `Contradiction`.
    pub fn check_contradiction(&self, text_a: &str, text_b: &str) -> anyhow::Result<bool> {
        Ok(self.classify(text_a, text_b)? == NliLabel::Contradiction)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl NliClassifier for OnnxNliChecker {
    fn classify(&self, premise: &str, hypothesis: &str) -> anyhow::Result<NliLabel> {
        OnnxNliChecker::classify(self, premise, hypothesis)
    }
}
impl NliChecker {
    pub fn new() -> Self {
        Self { onnx: None }
    }

    #[cfg(test)]
    pub(crate) fn with_classifier(classifier: Arc<dyn NliClassifier>) -> Self {
        Self {
            onnx: Some(classifier),
        }
    }

    pub fn with_onnx(onnx: OnnxNliChecker) -> Self {
        Self {
            onnx: Some(Arc::new(onnx)),
        }
    }

    /// Compute an NLI label for the pair (premise, hypothesis).
    ///
    /// 1. **Topic gate** (cosine < `cosine_threshold` -> `Neutral`). Runs first on
    ///    both paths: it is cheap, and it bounds ONNX cost because the store calls
    ///    this against every live memory in the namespace.
    /// 2. **ONNX model**, if installed: its label is returned as-is (including
    ///    `Neutral` / `Entailment`). The lexical heuristics never override it. On an
    ///    inference error a warning is logged and step 3 is used instead.
    /// 3. **Lexical heuristics** (polarity opposition + negation asymmetry).
    pub fn check(
        &self,
        premise_text: &str,
        hypothesis_text: &str,
        premise_emb: &[f32],
        hypothesis_emb: &[f32],
        cosine_threshold: f32,
    ) -> NliLabel {
        let sim = cosine_similarity(premise_emb, hypothesis_emb);
        if sim < cosine_threshold {
            return NliLabel::Neutral;
        }

        if let Some(classifier) = &self.onnx {
            match classifier.classify(premise_text, hypothesis_text) {
                Ok(label) => return label,
                Err(e) => log::warn!(
                    "ONNX NLI inference failed ({e}); falling back to lexical heuristic"
                ),
            }
        }

        Self::lexical_label(premise_text, hypothesis_text)
    }

    /// Lexical fallback: polarity opposition + negation asymmetry. Assumes the
    /// topic gate has already passed.
    fn lexical_label(premise_text: &str, hypothesis_text: &str) -> NliLabel {
        // Gate 2: polarity opposition.
        let pol_a = detect_polarity(premise_text);
        let pol_b = detect_polarity(hypothesis_text);
        if pol_a == Polarity::Neutral || pol_b == Polarity::Neutral {
            // One text has no strong polarity - cannot assert contradiction.
            return NliLabel::Neutral;
        }
        if !pol_a.opposes(&pol_b) {
            // Same polarity direction - possibly entailment.
            return NliLabel::Entailment;
        }

        // Gate 3: negation asymmetry score.
        // We compute how many negation tokens appear exclusively in one text.
        // High asymmetry confirms the contradiction rather than a polarity artifact.
        let neg_score = negation_asymmetry(premise_text, hypothesis_text);
        if neg_score >= 1 {
            return NliLabel::Contradiction;
        }

        // Polarity opposes but no clear negation asymmetry - conservative fallback.
        NliLabel::Contradiction
    }
}

impl Default for NliChecker {
    fn default() -> Self {
        NliChecker::new()
    }
}

/// Count negation tokens present in one text but absent in the other.
/// High values indicate one text strongly negates what the other asserts.
fn negation_asymmetry(a: &str, b: &str) -> usize {
    const NEGATORS: &[&str] = &[
        "never",
        "not",
        "no",
        "avoid",
        "don't",
        "cannot",
        "must not",
        "should not",
        "prohibited",
        "forbidden",
        "disable",
        "remove",
        "deprecated",
        "stop",
        "revert",
        "rollback",
        "dangerous",
        "insecure",
        "broken",
    ];
    let na = normalize_text(a);
    let nb = normalize_text(b);
    let in_a = |t: &&str| -> bool {
        if t.contains(' ') {
            na.contains(*t)
        } else {
            na.split_whitespace().any(|w| w == *t)
        }
    };
    let in_b = |t: &&str| -> bool {
        if t.contains(' ') {
            nb.contains(*t)
        } else {
            nb.split_whitespace().any(|w| w == *t)
        }
    };
    NEGATORS.iter().filter(|t| in_a(t) != in_b(t)).count()
}

// ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Configuration ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

/// Configuration for trust scoring and reinforcement attribution.
///
/// All fields are tuneable after construction via `Memoire::with_scoring_config()`.
/// These are engineering heuristics with default values chosen as reasonable starting
/// points. Tune based on observed agent behaviour; there is no universal calibration.
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Weight applied to the newly computed trust value in EMA updates.
    /// Range: (0.0, 1.0). Default 0.3 (30% new value, 70% historical EMA).
    /// Lower values ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ slower trust changes, more stable but less responsive.
    pub ema_new_weight: f32,
    /// Saturation constant `k` in the reinforcement term `rc / (rc + k)`.
    /// Default 3.0: rc=3 reaches ~0.5, rc=9 reaches ~0.75 of the reinforcement ceiling.
    /// Increase to require more task confirmations before a memory reaches FOLLOW.
    pub rc_saturation: f32,
    /// Minimum Jaccard token overlap required to attribute reinforcement to a memory.
    /// Range: [0.0, 1.0]. Default 0.35.
    /// Higher ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ stricter attribution, fewer false-positive trust inflations.
    pub jaccard_threshold: f32,
    /// Memory count above which HNSW approximate search is used instead of linear scan.
    /// Default 500. Below this, a linear scan over the in-memory embedding cache is
    /// faster due to lower constant factors than HNSW index construction.
    pub hnsw_threshold: usize,
    /// Multiplied by quality_score to compute the initial trust_ema for a new memory
    /// that has no reinforcement history (reinforcement_count == 0).
    /// Range: 0.0ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“1.0. Default: 0.5
    /// Set to 0.0 to disable cold-start trust (original behavior).
    pub cold_start_weight: f32,
    /// Exponential decay rate applied to trust_ema per day since last use.
    /// Effective trust = trust_ema * exp(-decay_rate * days_since_last_used)
    /// Default: 0.01 (half-life ~69 days). Set to 0.0 to disable.
    pub decay_rate: f32,
    /// When true, use `NliChecker` for contradiction detection in addition to
    /// lexical polarity heuristics. Provides higher-precision contradiction
    /// detection by adding negation-asymmetry analysis to the polarity gate.
    /// Default: true.
    pub use_nli_contradiction: bool,
    /// Minimum cosine similarity between two memories for them to be considered
    /// in the same topic cluster during NLI contradiction checking.
    /// Range: [0.0, 1.0]. Default: 0.80.
    pub nli_cosine_threshold: f32,
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
            use_nli_contradiction: true,
            nli_cosine_threshold: 0.80,
        }
    }
}

// ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Semantic Prototypes ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

/// Pre-computed centroid vectors for semantic quality feature scoring.
///
/// Each centroid is the mean embedding of multiple exemplar sentences for that
/// category, computed via the same ONNX model used for all other embeddings.
/// Computed once at `Memoire` initialisation time and reused for every ingestion.
///
/// Using semantic centroids rather than keyword lists means semantically
/// equivalent phrases ("safeguard" ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€  "security", "mandate" ÃƒÂ¢Ã¢â‚¬Â°Ã‹â€  "always") score
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

// ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Polarity Detection ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

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
    /// `detect_polarity` must NOT be called on the stored string ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â it scans for
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
/// It is intentionally documented as a heuristic ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â not NLI-level inference.
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

// ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ Core Types ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬

#[derive(Debug, Clone, PartialEq)]
pub enum IngestDecision {
    Active,
    Probationary,
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

/// ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ FIX #2 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Stable content fingerprint using BLAKE3.
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

/// ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ FIX #3 ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Semantic feature extraction via prototype cosine similarity.
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
///   0.35 reinforcement ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â most direct signal: memory was acted on and helped
///   0.25 confidence    ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â initial quality signal from ingestion evidence
///   0.20 age_term      ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â fresh memories are more likely to still be accurate
///   0.15 importance    ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â ingestion-time importance score
///   0.05 contradiction ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â small bonus for surviving a contradiction resolution
///
/// Trust interpretation:
///   ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥ 0.75 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ FOLLOW (act on it)
///   ÃƒÂ¢Ã¢â‚¬Â°Ã‚Â¥ 0.45 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ HINT (verify before acting)
///   < 0.45 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ IGNORE
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

pub fn compute_gravity_radius(trust_ema: f64, reinforcement_count: i64, age_days: f64) -> f64 {
    let trust = trust_ema.clamp(0.0, 1.0);
    let reinforcement_factor = (1.0_f64 + reinforcement_count as f64).ln();
    let age_stability = match age_days {
        d if d < 7.0 => 0.3,
        d if d < 30.0 => 0.6,
        _ => 1.0,
    };
    trust * reinforcement_factor * age_stability
}

pub fn should_supersede(incoming_importance: f64, existing_gravity: f64) -> bool {
    let threshold = 0.50 + 0.40 * existing_gravity.clamp(0.0, 1.0);
    incoming_importance > threshold
}

/// Build quality metadata for a memory chunk.
///
/// ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ FIX #3 / #4: Takes the pre-computed embedding for semantic feature scoring
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
        IngestDecision::Probationary => "probationary",
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

#[cfg(not(target_arch = "wasm32"))]
pub struct WeakVerifyResult {
    pub passed: bool,
    pub reason: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
pub struct StrongVerifyResult {
    pub confidence: f64,
    pub coverage: f64,
    pub consistency: f64,
}

#[cfg(not(target_arch = "wasm32"))]
pub trait IngestionOracle: Send + Sync {
    fn verify_weak(&self, content: &str, embedding: &[f32]) -> WeakVerifyResult;

    fn verify_strong(
        &self,
        content: &str,
        embedding: &[f32],
        namespace_sample: &[crate::store::Memory],
    ) -> StrongVerifyResult;

    fn decide(
        &self,
        content: &str,
        embedding: &[f32],
        namespace_sample: &[crate::store::Memory],
    ) -> IngestDecision {
        let weak = self.verify_weak(content, embedding);
        if !weak.passed {
            return IngestDecision::Reject;
        }
        let strong = self.verify_strong(content, embedding, namespace_sample);
        match strong.confidence {
            c if c >= 0.95 => IngestDecision::Active,
            c if c >= 0.80 => IngestDecision::Probationary,
            _ => IngestDecision::Reject,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub struct DefaultOracle {
    pub weights: ScoringWeights,
    pub prototypes: ScoringPrototypes,
    pub embedder: Option<Arc<dyn crate::embedder::EmbedProvider>>,
    pub nli_checker: Option<NliChecker>,
}

#[cfg(not(target_arch = "wasm32"))]
impl IngestionOracle for DefaultOracle {
    fn verify_weak(&self, content: &str, embedding: &[f32]) -> WeakVerifyResult {
        if let Some(lang) = crate::chunker::detect_language(content) {
            let ts_lang = match lang {
                crate::chunker::CodeLanguage::Python => Some(tree_sitter_python::LANGUAGE.into()),
                crate::chunker::CodeLanguage::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
                crate::chunker::CodeLanguage::JavaScript => {
                    Some(tree_sitter_javascript::LANGUAGE.into())
                }
                crate::chunker::CodeLanguage::TypeScript => {
                    Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
                }
                crate::chunker::CodeLanguage::Generic => None,
            };
            if let Some(ts_lang) = ts_lang {
                let mut parser = tree_sitter::Parser::new();
                if parser.set_language(&ts_lang).is_ok() {
                    let mut code = content.to_string();
                    if code.contains("```") {
                        let mut inside = false;
                        let mut extracted = Vec::new();
                        for line in content.lines() {
                            if line.trim().starts_with("```") {
                                inside = !inside;
                                continue;
                            }
                            if inside {
                                extracted.push(line);
                            }
                        }
                        if !extracted.is_empty() {
                            code = extracted.join("\n");
                        }
                    }
                    let parsed = parser.parse(&code, None);
                    if parsed.is_none() || parsed.map(|t| t.root_node().has_error()).unwrap_or(true)
                    {
                        return WeakVerifyResult {
                            passed: false,
                            reason: Some("syntax error detected by tree-sitter".to_string()),
                        };
                    }
                }
            }
        }

        let features = extract_features(embedding, content, &self.prototypes, 0.5);
        let score = score_importance(&features, &self.weights);
        let passed = score >= 0.50;
        WeakVerifyResult {
            passed,
            reason: if passed {
                None
            } else {
                Some("importance score below 0.50".to_string())
            },
        }
    }

    fn verify_strong(
        &self,
        content: &str,
        _embedding: &[f32],
        namespace_sample: &[crate::store::Memory],
    ) -> StrongVerifyResult {
        if namespace_sample.is_empty() {
            return StrongVerifyResult {
                confidence: 1.0,
                coverage: 1.0,
                consistency: 1.0,
            };
        }

        // This check only runs if we have an NLI checker; otherwise always pass.
        let nli = match &self.nli_checker {
            Some(n) => n,
            None => {
                return StrongVerifyResult {
                    confidence: 1.0,
                    coverage: 1.0,
                    consistency: 1.0,
                }
            }
        };

        // Find top-5 most similar memories with cosine sim > 0.80
        let mut scored: Vec<(f32, &crate::store::Memory)> = namespace_sample
            .iter()
            .map(|m| {
                // We don't have embeddings on Memory, so use content word-overlap as proxy
                let tokens_a: std::collections::HashSet<&str> =
                    content.split_whitespace().collect();
                let tokens_b: std::collections::HashSet<&str> =
                    m.content.split_whitespace().collect();
                let intersection = tokens_a.intersection(&tokens_b).count();
                let union = tokens_a.union(&tokens_b).count();
                let jaccard = if union > 0 {
                    intersection as f32 / union as f32
                } else {
                    0.0
                };
                (jaccard, m)
            })
            .filter(|(sim, _)| *sim > 0.30) // use jaccard 0.30 as proxy for cosine 0.80
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(5);

        if scored.is_empty() {
            return StrongVerifyResult {
                confidence: 1.0,
                coverage: 1.0,
                consistency: 1.0,
            };
        }

        let total = scored.len() as f64;
        let mut non_contradictory = 0usize;
        for (_, mem) in &scored {
            match nli
                .onnx
                .as_ref()
                .and_then(|onnx| onnx.classify(content, &mem.content).ok())
            {
                Some(NliLabel::Contradiction) => {
                    // contradiction found ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â immediate reject
                    return StrongVerifyResult {
                        confidence: 0.0,
                        coverage: 0.0,
                        consistency: 0.0,
                    };
                }
                _ => non_contradictory += 1,
            }
        }
        let confidence = non_contradictory as f64 / total;
        StrongVerifyResult {
            confidence,
            coverage: confidence,
            consistency: confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; dim];
        v[hot] = 1.0;
        v
    }

    #[test]
    fn test_nli_contradiction_detected() {
        let checker = NliChecker::new();
        // Same embedding = cosine 1.0, opposing polarity ("never" vs "always")
        let emb = vec![0.577_f32, 0.577, 0.577];
        let label = checker.check(
            "never use float for money",
            "always use float for money",
            &emb,
            &emb,
            0.80,
        );
        assert_eq!(
            label,
            NliLabel::Contradiction,
            "opposing polarity with high cosine must be Contradiction"
        );
    }

    #[test]
    fn test_nli_neutral_low_cosine() {
        let checker = NliChecker::new();
        // Orthogonal embeddings = cosine 0 < threshold ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Neutral regardless of polarity
        let a = unit_vec(3, 0);
        let b = unit_vec(3, 1);
        let label = checker.check("never use float for money", "always use int", &a, &b, 0.80);
        assert_eq!(
            label,
            NliLabel::Neutral,
            "low cosine must return Neutral even with opposing polarity"
        );
    }

    #[test]
    fn test_nli_entailment_same_polarity() {
        let checker = NliChecker::new();
        let emb = vec![0.577_f32, 0.577, 0.577];
        let label = checker.check(
            "always use Decimal for money",
            "always prefer Decimal over float",
            &emb,
            &emb,
            0.80,
        );
        // Both affirmative ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â should be Entailment, not Contradiction
        assert_ne!(
            label,
            NliLabel::Contradiction,
            "same polarity must not be flagged as Contradiction"
        );
    }

    #[test]
    fn test_nli_neutral_when_no_polarity() {
        let checker = NliChecker::new();
        let emb = vec![0.577_f32, 0.577, 0.577];
        // Neutral polarity text ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â no strong signal in either direction
        let label = checker.check(
            "the model is a transformer architecture",
            "the model is a recurrent network",
            &emb,
            &emb,
            0.80,
        );
        assert_eq!(
            label,
            NliLabel::Neutral,
            "texts with neutral polarity must not be flagged as Contradiction"
        );
    }

    #[test]
    fn test_polarity_detection() {
        assert_eq!(detect_polarity("never use floats"), Polarity::Negative);
        assert_eq!(detect_polarity("always use Decimal"), Polarity::Affirmative);
        assert_eq!(detect_polarity("the sky is blue"), Polarity::Neutral);
    }

    #[test]
    fn test_polarity_opposes() {
        assert!(Polarity::Affirmative.opposes(&Polarity::Negative));
        assert!(Polarity::Negative.opposes(&Polarity::Affirmative));
        assert!(!Polarity::Affirmative.opposes(&Polarity::Affirmative));
        assert!(!Polarity::Neutral.opposes(&Polarity::Negative));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![0.0_f32, 1.0, 0.0];
        assert!(
            (cosine_similarity(&a, &b)).abs() < 1e-6,
            "orthogonal ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ 0"
        );
        assert!(
            (cosine_similarity(&a, &a) - 1.0).abs() < 1e-6,
            "identical ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ 1"
        );
    }

    // --- ONNX label mapping + authoritative-model behaviour (no model download) ---

    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    fn map_from(pairs: &[(&str, &str)]) -> anyhow::Result<NliLabelMap> {
        let body: Vec<String> = pairs
            .iter()
            .map(|(k, v)| format!("\"{k}\": \"{v}\""))
            .collect();
        NliLabelMap::from_id2label_json(&format!("{{\"id2label\": {{{}}}}}", body.join(", ")))
    }

    #[test]
    fn id2label_contradiction_at_index_0() {
        // facebook/bart-large-mnli ordering
        let m = map_from(&[("0", "contradiction"), ("1", "neutral"), ("2", "entailment")]).unwrap();
        assert_eq!(
            m.label_for_logits(&[2.0, 0.1, -1.0]).unwrap(),
            NliLabel::Contradiction
        );
        assert_eq!(m.label_for_logits(&[0.0, 0.2, 3.0]).unwrap(), NliLabel::Entailment);
    }

    #[test]
    fn id2label_contradiction_at_index_2() {
        // MoritzLaurer/* ordering: the old `max_idx == 0` would invert these.
        let m = map_from(&[("0", "entailment"), ("1", "neutral"), ("2", "contradiction")]).unwrap();
        assert_eq!(
            m.label_for_logits(&[-1.0, 0.0, 3.0]).unwrap(),
            NliLabel::Contradiction
        );
        assert_eq!(m.label_for_logits(&[3.0, 0.0, -1.0]).unwrap(), NliLabel::Entailment);
    }

    #[test]
    fn id2label_other_order_and_case_insensitive() {
        // cross-encoder/nli-* ordering, upper-case names
        let m = map_from(&[("0", "CONTRADICTION"), ("1", "Entailment"), ("2", " neutral ")]).unwrap();
        assert_eq!(m.label_for_logits(&[0.1, 3.0, 0.2]).unwrap(), NliLabel::Entailment);
        assert_eq!(m.label_for_logits(&[0.1, 0.2, 3.0]).unwrap(), NliLabel::Neutral);
    }

    #[test]
    fn id2label_rejects_missing_or_ambiguous_mapping() {
        let bad: [&str; 8] = [
            "{}",
            "not json",
            r#"{"id2label": []}"#,
            r#"{"id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}}"#,
            r#"{"id2label": {"0": "entailment", "1": "not_entailment"}}"#,
            r#"{"id2label": {"0": "neutral", "1": "neutral", "2": "contradiction"}}"#,
            r#"{"id2label": {"0": "entailment", "1": "neutral", "3": "contradiction"}}"#,
            r#"{"id2label": {"a": "entailment", "1": "neutral", "2": "contradiction"}}"#,
        ];
        for json in bad {
            assert!(
                NliLabelMap::from_id2label_json(json).is_err(),
                "must reject: {json}"
            );
        }
        let err = NliLabelMap::from_id2label_json("{}").unwrap_err().to_string();
        assert!(err.contains("id2label"), "error should name id2label: {err}");
    }

    #[test]
    fn logits_must_be_three_finite_values() {
        let m = map_from(&[("0", "contradiction"), ("1", "neutral"), ("2", "entailment")]).unwrap();
        assert!(m.label_for_logits(&[1.0, 2.0]).is_err());
        assert!(m.label_for_logits(&[1.0, 2.0, 3.0, 4.0]).is_err());
        assert!(m.label_for_logits(&[1.0, f32::NAN, 3.0]).is_err());
    }

    struct StubClassifier {
        result: Result<NliLabel, String>,
        calls: AtomicUsize,
    }

    impl StubClassifier {
        fn new(result: Result<NliLabel, String>) -> Arc<Self> {
            Arc::new(Self {
                result,
                calls: AtomicUsize::new(0),
            })
        }
    }

    impl NliClassifier for StubClassifier {
        fn classify(&self, _p: &str, _h: &str) -> anyhow::Result<NliLabel> {
            self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            self.result.clone().map_err(|e| anyhow::anyhow!(e))
        }
    }

    const LEX_CONTRA_A: &str = "never use float for money";
    const LEX_CONTRA_B: &str = "always use float for money";
    const LEX_NEUTRAL_A: &str = "the model is a transformer architecture";
    const LEX_NEUTRAL_B: &str = "the model is a recurrent network";

    fn same_topic() -> Vec<f32> {
        vec![0.577_f32, 0.577, 0.577]
    }

    #[test]
    fn authoritative_onnx_neutral_is_not_overridden_by_lexical_polarity() {
        let emb = same_topic();
        // Sanity: the lexical path alone WOULD say Contradiction for this pair.
        assert_eq!(
            NliChecker::new().check(LEX_CONTRA_A, LEX_CONTRA_B, &emb, &emb, 0.80),
            NliLabel::Contradiction
        );
        let stub = StubClassifier::new(Ok(NliLabel::Neutral));
        let checker = NliChecker::with_classifier(stub.clone());
        assert_eq!(
            checker.check(LEX_CONTRA_A, LEX_CONTRA_B, &emb, &emb, 0.80),
            NliLabel::Neutral
        );
        assert_eq!(stub.calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn authoritative_onnx_entailment_is_not_overridden_by_lexical_polarity() {
        let emb = same_topic();
        let checker = NliChecker::with_classifier(StubClassifier::new(Ok(NliLabel::Entailment)));
        assert_eq!(
            checker.check(LEX_CONTRA_A, LEX_CONTRA_B, &emb, &emb, 0.80),
            NliLabel::Entailment
        );
    }

    #[test]
    fn onnx_contradiction_is_returned_even_when_lexical_sees_none() {
        let emb = same_topic();
        let checker = NliChecker::with_classifier(StubClassifier::new(Ok(NliLabel::Contradiction)));
        assert_eq!(
            checker.check(LEX_NEUTRAL_A, LEX_NEUTRAL_B, &emb, &emb, 0.80),
            NliLabel::Contradiction
        );
    }

    #[test]
    fn onnx_error_falls_back_to_lexical_heuristic() {
        let emb = same_topic();
        let checker = NliChecker::with_classifier(StubClassifier::new(Err("boom".into())));
        assert_eq!(
            checker.check(LEX_CONTRA_A, LEX_CONTRA_B, &emb, &emb, 0.80),
            NliLabel::Contradiction
        );
        assert_eq!(
            checker.check(LEX_NEUTRAL_A, LEX_NEUTRAL_B, &emb, &emb, 0.80),
            NliLabel::Neutral
        );
    }

    #[test]
    fn low_cosine_skips_onnx_inference_entirely() {
        let stub = StubClassifier::new(Ok(NliLabel::Contradiction));
        let checker = NliChecker::with_classifier(stub.clone());
        let label = checker.check(
            LEX_CONTRA_A,
            LEX_CONTRA_B,
            &unit_vec(3, 0),
            &unit_vec(3, 1),
            0.80,
        );
        assert_eq!(label, NliLabel::Neutral);
        assert_eq!(stub.calls.load(AtomicOrdering::SeqCst), 0);
    }
}

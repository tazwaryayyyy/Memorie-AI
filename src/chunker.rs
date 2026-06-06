#[derive(Debug, Clone, PartialEq)]
pub enum CodeLanguage {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    /// Regex-based fallback for unsupported languages.
    Generic,
}

#[derive(Debug, Clone, Default)]
pub enum ChunkerMode {
    /// Sliding-window word tokeniser (original behavior).
    #[default]
    Prose,
    /// Code-aware chunking for a specific language.
    Code(CodeLanguage),
    /// Auto-detect: use code chunking when triple-backtick fences are present,
    /// otherwise fall back to Prose. Backward compatible default.
    Auto,
}

#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub mode: ChunkerMode,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            overlap: 20,
            mode: ChunkerMode::Auto,
        }
    }
}

/// Detect code language from triple-backtick fences.
/// Returns None if no fences are found (prose fallback).
fn detect_language(text: &str) -> Option<CodeLanguage> {
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```python") || trimmed.starts_with("```py") {
            return Some(CodeLanguage::Python);
        }
        if trimmed.starts_with("```rust") || trimmed.starts_with("```rs") {
            return Some(CodeLanguage::Rust);
        }
        if trimmed.starts_with("```javascript") || trimmed.starts_with("```js") {
            return Some(CodeLanguage::JavaScript);
        }
        if trimmed.starts_with("```typescript")
            || trimmed.starts_with("```ts")
            || trimmed.starts_with("```tsx")
        {
            return Some(CodeLanguage::TypeScript);
        }
        if trimmed.starts_with("```") && trimmed.len() > 3 {
            return Some(CodeLanguage::Generic);
        }
    }
    None
}

/// Chunk code text using regex boundary detection.
///
/// Boundaries are lines matching common top-level declaration patterns.
/// A chunk is all lines from one boundary (inclusive) to the next (exclusive).
/// Chunks that exceed `chunk_size * 2` characters are further split by line count.
fn chunk_code(text: &str, lang: &CodeLanguage, config: &ChunkerConfig) -> Vec<String> {
    // Boundary patterns — prefixes that start a new logical code unit
    let is_boundary = |line: &str| -> bool {
        let s = line.trim_start();
        match lang {
            CodeLanguage::Python => {
                s.starts_with("def ")
                    || s.starts_with("async def ")
                    || s.starts_with("class ")
                    || s.starts_with("@")
            }
            CodeLanguage::Rust => {
                s.starts_with("fn ")
                    || s.starts_with("pub fn ")
                    || s.starts_with("pub(crate) fn ")
                    || s.starts_with("async fn ")
                    || s.starts_with("pub async fn ")
                    || s.starts_with("impl ")
                    || s.starts_with("pub impl ")
                    || s.starts_with("struct ")
                    || s.starts_with("pub struct ")
                    || s.starts_with("enum ")
                    || s.starts_with("pub enum ")
                    || s.starts_with("trait ")
                    || s.starts_with("pub trait ")
            }
            CodeLanguage::JavaScript | CodeLanguage::TypeScript => {
                s.starts_with("function ")
                    || s.starts_with("async function ")
                    || s.starts_with("export function ")
                    || s.starts_with("export async function ")
                    || s.starts_with("export default function")
                    || s.starts_with("class ")
                    || s.starts_with("export class ")
                    || s.starts_with("const ")
                    || s.starts_with("export const ")
            }
            CodeLanguage::Generic => {
                s.starts_with("def ")
                    || s.starts_with("class ")
                    || s.starts_with("fn ")
                    || s.starts_with("pub fn ")
                    || s.starts_with("pub(crate) fn ")
                    || s.starts_with("async fn ")
                    || s.starts_with("impl ")
                    || s.starts_with("struct ")
                    || s.starts_with("enum ")
                    || s.starts_with("trait ")
            }
        }
    };

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return vec![];
    }

    // Split into logical blocks at boundary lines
    let mut blocks: Vec<Vec<&str>> = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for line in &lines {
        if is_boundary(line) && !current.is_empty() {
            blocks.push(current);
            current = Vec::new();
        }
        current.push(line);
    }
    if !current.is_empty() {
        blocks.push(current);
    }

    let max_chars = config.chunk_size * 2 * 6; // ~6 chars per word average
    let mut result = Vec::new();

    for block in blocks {
        let text = block.join("\n");
        if text.len() <= max_chars {
            let trimmed = text.trim().to_string();
            if !trimmed.is_empty() {
                result.push(trimmed);
            }
        } else {
            // Oversized block: split by line count
            let step = config.chunk_size.max(1);
            let mut start = 0;
            while start < block.len() {
                let end = (start + step).min(block.len());
                let sub = block[start..end].join("\n");
                let trimmed = sub.trim().to_string();
                if !trimmed.is_empty() {
                    result.push(trimmed);
                }
                start += step;
            }
        }
    }

    if result.is_empty() && !text.trim().is_empty() {
        // No boundaries found — fall back to prose chunking
        return sliding_window(text, config);
    }

    result
}

/// Original sliding-window word tokeniser — behavior is unchanged.
fn sliding_window(text: &str, config: &ChunkerConfig) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return vec![];
    }

    if words.len() <= config.chunk_size {
        return vec![words.join(" ")];
    }

    let mut chunks = Vec::new();
    let step = config.chunk_size.saturating_sub(config.overlap).max(1);
    let mut start = 0;

    while start < words.len() {
        let end = (start + config.chunk_size).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start += step;
    }

    chunks
}

/// Chunk `text` according to `config.mode`.
///
/// - `Prose`: original sliding-window behavior, byte-for-byte identical to pre-change.
/// - `Code(lang)`: code-aware chunking for `lang`.
/// - `Auto`: detect language from triple-backtick fences; fall back to Prose if none.
pub fn chunk_text(text: &str, config: &ChunkerConfig) -> Vec<String> {
    match &config.mode {
        ChunkerMode::Prose => sliding_window(text, config),
        ChunkerMode::Code(lang) => chunk_code(text, lang, config),
        ChunkerMode::Auto => match detect_language(text) {
            Some(lang) => chunk_code(text, &lang, config),
            None => sliding_window(text, config),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_chunk_short_text() {
        let cfg = ChunkerConfig {
            chunk_size: 10,
            overlap: 2,
            mode: ChunkerMode::Prose,
        };
        let chunks = chunk_text("hello world foo bar", &cfg);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_overlap() {
        let words: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        let cfg = ChunkerConfig {
            chunk_size: 10,
            overlap: 3,
            mode: ChunkerMode::Prose,
        };
        let chunks = chunk_text(&words.join(" "), &cfg);
        assert!(chunks.len() > 1);
        let first: Vec<&str> = chunks[0].split_whitespace().collect();
        let second: Vec<&str> = chunks[1].split_whitespace().collect();
        let tail = &first[first.len() - cfg.overlap..];
        let head = &second[..cfg.overlap];
        assert_eq!(tail, head);
    }

    #[test]
    fn test_empty() {
        let cfg = ChunkerConfig::default();
        assert!(chunk_text("", &cfg).is_empty());
        assert!(chunk_text("   ", &cfg).is_empty());
    }

    #[test]
    fn test_auto_falls_back_to_prose_for_plain_text() {
        let prose = "This is a plain text sentence without any code fences or keywords.";
        let prose_cfg = ChunkerConfig {
            mode: ChunkerMode::Prose,
            ..ChunkerConfig::default()
        };
        let auto_cfg = ChunkerConfig {
            mode: ChunkerMode::Auto,
            ..ChunkerConfig::default()
        };
        // Auto on prose must produce identical output to Prose
        assert_eq!(chunk_text(prose, &prose_cfg), chunk_text(prose, &auto_cfg));
    }

    #[test]
    fn test_code_chunker_splits_on_fn_boundaries() {
        let code = "fn alpha() {\n    let x = 1;\n}\nfn beta() {\n    let y = 2;\n}\n";
        let cfg = ChunkerConfig {
            chunk_size: 128,
            overlap: 20,
            mode: ChunkerMode::Code(CodeLanguage::Rust),
        };
        let chunks = chunk_text(code, &cfg);
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );
        assert!(chunks[0].contains("fn alpha"));
        assert!(chunks[1].contains("fn beta"));
    }
}

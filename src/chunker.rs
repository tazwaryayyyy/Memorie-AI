#[derive(Debug, Clone, PartialEq)]
pub enum CodeLanguage {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    /// Regex-based fallback for unsupported languages.
    Generic,
}

impl CodeLanguage {
    fn as_lang_name(&self) -> &'static str {
        match self {
            CodeLanguage::Python => "python",
            CodeLanguage::Rust => "rust",
            CodeLanguage::JavaScript => "javascript",
            CodeLanguage::TypeScript => "typescript",
            CodeLanguage::Generic => "generic",
        }
    }
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
pub fn detect_language(text: &str) -> Option<CodeLanguage> {
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
fn chunk_code_regex(text: &str, lang: &CodeLanguage, config: &ChunkerConfig) -> Vec<String> {
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

#[cfg(not(target_arch = "wasm32"))]
fn node_text(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let span = node.byte_range();
    source.get(span.start..span.end).map(str::to_string)
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_ast_chunks(
    node: tree_sitter::Node<'_>,
    source: &str,
    chunk_size: usize,
    current: &mut Vec<String>,
) {
    let text = match node_text(node, source) {
        Some(text) => text,
        None => return,
    };

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }

    let node_kind = node.kind();
    let is_boundary_node = matches!(
        node_kind,
        "function_definition"
            | "class_definition"
            | "decorated_definition"
            | "function_item"
            | "impl_item"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "mod_item"
            | "function_declaration"
            | "class_declaration"
            | "export_statement"
            | "arrow_function"
            | "lexical_declaration"
            | "variable_declaration"
            | "variable_declarator"
            | "expression_statement"
    );

    if is_boundary_node {
        let normalized = trimmed.to_string();
        if normalized.len() <= chunk_size * 3 {
            current.push(normalized);
            return;
        }
    }

    let child_count = node.child_count();
    if child_count == 0 {
        if trimmed.len() > chunk_size * 3 {
            for line in trimmed.lines() {
                let line_trim = line.trim();
                if !line_trim.is_empty() {
                    current.push(line_trim.to_string());
                }
            }
        } else if is_boundary_node {
            current.push(trimmed.to_string());
        }
        return;
    }

    for child in node.children(&mut node.walk()) {
        collect_ast_chunks(child, source, chunk_size, current);
    }
}

/// Returns chunks split at AST-meaningful boundaries.
/// Falls back to the regex-based code chunker if tree-sitter parsing fails.
#[cfg(not(target_arch = "wasm32"))]
pub fn chunk_code_ast(source: &str, lang: &str, chunk_size: usize) -> Vec<String> {
    let mut parser = tree_sitter::Parser::new();
    let language = match lang {
        "python" | "py" => tree_sitter_python::LANGUAGE.into(),
        "rust" | "rs" => tree_sitter_rust::LANGUAGE.into(),
        "javascript" | "js" => tree_sitter_javascript::LANGUAGE.into(),
        "typescript" | "ts" | "tsx" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        _ => {
            return chunk_code_regex(
                source,
                &CodeLanguage::Generic,
                &ChunkerConfig {
                    chunk_size,
                    overlap: 0,
                    mode: ChunkerMode::Prose,
                },
            )
        }
    };

    if parser.set_language(&language).is_err() {
        return chunk_code_regex(
            source,
            &CodeLanguage::Generic,
            &ChunkerConfig {
                chunk_size,
                overlap: 0,
                mode: ChunkerMode::Prose,
            },
        );
    }

    let tree = match parser.parse(source, None) {
        Some(tree) => tree,
        None => {
            return chunk_code_regex(
                source,
                &CodeLanguage::Generic,
                &ChunkerConfig {
                    chunk_size,
                    overlap: 0,
                    mode: ChunkerMode::Prose,
                },
            );
        }
    };

    let mut chunks = Vec::new();
    let root = tree.root_node();
    for child in root.children(&mut root.walk()) {
        collect_ast_chunks(child, source, chunk_size, &mut chunks);
    }

    if chunks.is_empty() {
        if source.trim().is_empty() {
            vec![]
        } else {
            chunk_code_regex(
                source,
                &CodeLanguage::Generic,
                &ChunkerConfig {
                    chunk_size,
                    overlap: 0,
                    mode: ChunkerMode::Prose,
                },
            )
        }
    } else {
        chunks
    }
}

/// wasm32 build: tree-sitter (C parsers) is unavailable, so always use the
/// regex-based code chunker.
#[cfg(target_arch = "wasm32")]
pub fn chunk_code_ast(source: &str, _lang: &str, chunk_size: usize) -> Vec<String> {
    chunk_code_regex(
        source,
        &CodeLanguage::Generic,
        &ChunkerConfig {
            chunk_size,
            overlap: 0,
            mode: ChunkerMode::Prose,
        },
    )
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
        ChunkerMode::Code(lang) => chunk_code_ast(text, lang.as_lang_name(), config.chunk_size),
        ChunkerMode::Auto => match detect_language(text) {
            Some(lang) => chunk_code_ast(text, lang.as_lang_name(), config.chunk_size),
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

    #[test]
    fn test_js_code_chunker_splits_arrow_functions() {
        let code = "const sum = (a, b) => {\n  return a + b;\n};\n\nconst diff = (a, b) => {\n  return a - b;\n};\n";
        let cfg = ChunkerConfig {
            chunk_size: 128,
            overlap: 20,
            mode: ChunkerMode::Code(CodeLanguage::JavaScript),
        };
        let chunks = chunk_text(code, &cfg);
        assert!(
            chunks.len() >= 2,
            "expected at least 2 chunks, got {}",
            chunks.len()
        );
        assert!(chunks[0].contains("const sum"));
        assert!(chunks[1].contains("const diff"));
    }

    #[test]
    fn test_chunk_code_ast_handles_empty_input() {
        assert!(chunk_code_ast("", "python", 64).is_empty());
    }

    #[test]
    fn test_chunk_code_ast_falls_back_for_unsupported_language() {
        let chunks = chunk_code_ast("fn alpha() {\n  let x = 1;\n}\n", "unknown", 64);
        assert!(!chunks.is_empty());
    }
}

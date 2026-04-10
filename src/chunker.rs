#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 128,
            overlap: 20,
        }
    }
}

pub fn chunk_text(text: &str, config: &ChunkerConfig) -> Vec<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_chunk_short_text() {
        let cfg = ChunkerConfig {
            chunk_size: 10,
            overlap: 2,
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
}

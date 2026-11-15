//! Small UTF-8-safe text helpers.

/// Longest prefix of `s` with at most `max_chars` Unicode scalar values.
///
/// Unlike `&s[..n]`, which panics when byte `n` falls inside a multi-byte
/// character (`\u{2192}`, em dash, emoji, ...), this always cuts on a char boundary.
pub fn truncate_chars(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

/// `truncate_chars` plus a trailing `...` when something was cut off.
pub fn preview(s: &str, max_chars: usize) -> String {
    let head = truncate_chars(s, max_chars);
    if head.len() < s.len() {
        format!("{head}...")
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_never_splits_multibyte_chars_at_byte_60() {
        // byte 60 lands inside each of these characters; `&s[..60]` would panic.
        for ch in ['\u{2192}', '\u{2014}', '\u{1F600}'] {
            let width = ch.len_utf8();
            let lead = 60 - 1; // ASCII bytes before the multi-byte char
            let s = format!("{}{}tail", "a".repeat(lead), ch);
            assert!(!s.is_char_boundary(60), "{ch:?}: byte 60 must be mid-char");
            assert!(width > 1);

            let cut = truncate_chars(&s, 60);
            assert!(s.starts_with(cut));
            assert_eq!(cut.chars().count(), 60);
            assert!(cut.ends_with(ch));
        }
    }

    #[test]
    fn truncate_handles_short_exact_and_zero() {
        assert_eq!(truncate_chars("abc", 60), "abc");
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abc", 2), "ab");
        assert_eq!(truncate_chars("abc", 0), "");
        assert_eq!(truncate_chars("", 5), "");
        assert_eq!(
            truncate_chars("\u{2192}\u{2192}\u{2192}", 2),
            "\u{2192}\u{2192}"
        );
    }

    #[test]
    fn preview_adds_ellipsis_only_when_truncated() {
        let long = format!("{}\u{2014}{}", "a".repeat(59), "b".repeat(40));
        let p = preview(&long, 60);
        assert!(p.ends_with("..."));
        assert_eq!(p.chars().count(), 63);

        assert_eq!(preview("short", 60), "short");
        // 60 chars exactly -> unchanged, no ellipsis
        let exact = "x".repeat(60);
        assert_eq!(preview(&exact, 60), exact);
    }

    #[test]
    fn multibyte_content_longer_than_60_bytes_but_not_60_chars_is_not_truncated() {
        // 30 arrows = 90 bytes, 30 chars. Old `len() > 60` check would have cut it.
        let s = "\u{2192}".repeat(30);
        assert_eq!(preview(&s, 60), s);
    }
}

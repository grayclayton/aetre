//! Tier 0 balance scan for `governed_gate_pr`.
//!
//! The gate's job is to reject obviously broken code before anything expensive
//! runs. That makes a false positive the costly error: valid work is halted and
//! never verified. The scan this replaces walked the source counting brackets and
//! toggling a quote flag, with no notion of a comment, so every one of these was
//! rejected as broken:
//!
//! ```text
//! # the batch's shadow price     -> "Unterminated string literal"
//! # don't do this                -> "Unterminated string literal"
//! # see foo( for details         -> "Unclosed parenthesis"
//! ```
//!
//! Apostrophes and unmatched brackets in English prose comments are ordinary, so
//! this rejected a large share of real Python. It also had no triple-quote rule;
//! a docstring only balanced by luck, because `"""` toggles an odd number of
//! times at each end, and any docstring containing a lone `"` broke it.
//!
//! This scanner is still deliberately shallow - it is a balance check, not a
//! parser, and it does not try to agree with CPython on everything. It does
//! agree on comments, single, double and triple-quoted strings, and escapes.

/// Scans `code` for unbalanced brackets or unterminated strings.
///
/// Returns `None` when nothing is wrong, or the diagnostic to report.
pub fn scan_python_balance(code: &str) -> Option<String> {
    let chars: Vec<char> = code.chars().collect();
    let mut paren: i32 = 0;
    let mut brace: i32 = 0;
    let mut bracket: i32 = 0;
    let mut i = 0usize;

    while i < chars.len() {
        let ch = chars[i];

        // A comment runs to the end of the line and contains no code. This is the
        // case the old scanner was missing entirely.
        if ch == '#' {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }

        if ch == '\'' || ch == '"' {
            let triple = i + 2 < chars.len() && chars[i + 1] == ch && chars[i + 2] == ch;
            match skip_string(&chars, i, ch, triple) {
                Some(next) => {
                    i = next;
                    continue;
                }
                None => return Some("Unterminated string literal".to_string()),
            }
        }

        match ch {
            '(' => paren += 1,
            ')' => {
                paren -= 1;
                if paren < 0 {
                    return Some(format!("Unexpected closing parenthesis at character {i}"));
                }
            }
            '{' => brace += 1,
            '}' => {
                brace -= 1;
                if brace < 0 {
                    return Some(format!("Unexpected closing brace at character {i}"));
                }
            }
            '[' => bracket += 1,
            ']' => {
                bracket -= 1;
                if bracket < 0 {
                    return Some(format!("Unexpected closing bracket at character {i}"));
                }
            }
            _ => {}
        }
        i += 1;
    }

    if paren != 0 {
        Some(format!("Unclosed parenthesis (unbalanced by {paren})"))
    } else if brace != 0 {
        Some(format!("Unclosed brace (unbalanced by {brace})"))
    } else if bracket != 0 {
        Some(format!("Unclosed bracket (unbalanced by {bracket})"))
    } else {
        None
    }
}

/// Consumes the string literal opening at `start`, returning the index just past
/// its closing delimiter, or `None` if it is never closed.
fn skip_string(chars: &[char], start: usize, delim: char, triple: bool) -> Option<usize> {
    let open = if triple { 3 } else { 1 };
    let mut i = start + open;
    while i < chars.len() {
        let ch = chars[i];
        if ch == '\\' {
            // A backslash escapes the next character, including the delimiter.
            i += 2;
            continue;
        }
        if ch == delim {
            if !triple {
                return Some(i + 1);
            }
            if i + 2 < chars.len() && chars[i + 1] == delim && chars[i + 2] == delim {
                return Some(i + 3);
            }
        }
        // An unescaped newline ends a single-quoted string in Python; a triple
        // quoted one spans lines.
        if ch == '\n' && !triple {
            return None;
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(code: &str) {
        assert_eq!(
            scan_python_balance(code),
            None,
            "should have been accepted: {code:?}"
        );
    }

    fn rejected(code: &str) -> String {
        scan_python_balance(code).unwrap_or_else(|| panic!("should have been rejected: {code:?}"))
    }

    /// The reported defect: an apostrophe in a comment read as an open string.
    #[test]
    fn apostrophes_in_comments_are_not_string_literals() {
        ok("x = 1\n# under this batch's shadow price\ny = 2\n");
        ok("x = 1\n# don't do this\ny = 2\n");
        ok("# it's the reviewer's call\n");
    }

    /// The same defect on brackets: prose in a comment is not code.
    #[test]
    fn brackets_in_comments_do_not_count() {
        ok("x = 1\n# see foo( for details\ny = 2\n");
        ok("x = 1\n# closes here ) and there ]\ny = 2\n");
        ok("d = {}\n# a } in prose\n");
    }

    /// A `#` inside a string starts no comment, so the rest of the line is code.
    #[test]
    fn hash_inside_a_string_is_not_a_comment() {
        ok("x = \"# not a comment\"\ny = (1)\n");
        assert!(rejected("x = \"# not a comment\" + (\n").contains("Unclosed parenthesis"));
    }

    #[test]
    fn triple_quoted_strings_are_spanned() {
        ok("def f():\n    \"\"\"a ( unbalanced paren in prose\"\"\"\n    return 1\n");
        ok("s = '''line one\nline two with ' and \" inside\n'''\n");
        // A lone double quote inside a docstring broke the old toggling scan.
        ok("def f():\n    \"\"\"he said \"hi\" to me\"\"\"\n    return 1\n");
        assert!(rejected("s = \"\"\"never closed\n").contains("Unterminated"));
    }

    #[test]
    fn escapes_do_not_end_a_string() {
        ok("x = 'it\\'s fine'\ny = 2\n");
        ok("path = \"C:\\\\dir\"\n");
        ok("x = '\\\\'\ny = 2\n");
    }

    #[test]
    fn genuinely_broken_code_is_still_rejected() {
        assert!(rejected("def broken(:\n    pass\n").contains("Unclosed parenthesis"));
        assert!(rejected("x = (1 + 2))\n").contains("Unexpected closing parenthesis"));
        assert!(rejected("x = 'unterminated\n").contains("Unterminated"));
        assert!(rejected("d = {'a': 1\n").contains("Unclosed brace"));
        assert!(rejected("xs = [1, 2\n").contains("Unclosed bracket"));
        assert!(rejected("xs = [1, 2]]\n").contains("Unexpected closing bracket"));
    }

    #[test]
    fn a_single_quoted_string_does_not_span_lines() {
        // Without this, one stray quote swallows the rest of the file and the
        // diagnostic points nowhere near the actual mistake.
        assert!(rejected("x = 'oops\ny = 'also oops'\n").contains("Unterminated"));
    }

    #[test]
    fn empty_and_trivial_inputs_are_accepted() {
        ok("");
        ok("\n\n");
        ok("# just a comment\n");
    }
}

use pyo3::ffi::PyBuffer_GetPointer;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};
use regex::{Regex, RegexBuilder};
use std::collections::HashSet;
use std::sync::OnceLock;

static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

#[derive(Debug)]
#[pyclass]
struct Term {
    #[pyo3(get, set)]
    term: String,
    #[pyo3(get, set)]
    original_term: String,
    #[pyo3(get, set)]
    position: usize,
    #[pyo3(get, set)]
    start_char_offset: usize,
    #[pyo3(get, set)]
    end_char_offset: usize,
}

#[pymethods]
impl Term {
    #[new]
    fn new(
        term: String,
        original_term: String,
        position: usize,
        start_char_offset: usize,
        end_char_offset: usize,
    ) -> Self {
        Term {
            term,
            original_term,
            position,
            start_char_offset,
            end_char_offset,
        }
    }
}

struct Tokenizer {
    symbols: HashSet<char>,
    keywords: HashSet<String>,
    literal_regex: Regex,
}

impl Tokenizer {
    fn new(
        symbols: HashSet<char>,
        keywords: HashSet<String>,
        literal_pattern: Option<&str>,
    ) -> Self {
        let default_literal_pattern = r#"^(?:
            [0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?|  # numbers
            0x[0-9a-fA-F]+|                            # hex
            0b[01]+|                                   # binary
            true|false|                                # booleans
            \"(?:\\.|[^\"])*\"|                        # string literals
            '(?:\\.|[^'])*'                           # char literals
        )$"#;

        let literal_regex = RegexBuilder::new(literal_pattern.unwrap_or(default_literal_pattern))
            .size_limit(1000 * 1000)
            .dfa_size_limit(1000 * 1000)
            .build()
            .unwrap();

        Self {
            symbols,
            keywords,
            literal_regex,
        }
    }

    fn get_or_init(
        symbols: Option<HashSet<char>>,
        keywords: Option<HashSet<String>>,
        literal_pattern: Option<String>,
    ) -> &'static Self {
        TOKENIZER.get_or_init(|| {
            let default_symbols: HashSet<char> = [
                '(', ')', '[', ']', '{', '}', ',', ';', '.', '+', '-', '*', '/', '=', '<', '>',
                '!', '&', '|', '^',
            ]
            .iter()
            .copied()
            .collect();

            let default_keywords: HashSet<String> = [
                "fn", "let", "const", "mut", "if", "else", "while", "for", "in", "return", "true",
                "false", "struct", "enum", "impl",
            ]
            .iter()
            .map(|&s| s.to_string())
            .collect();

            Tokenizer::new(
                symbols.unwrap_or(default_symbols),
                keywords.unwrap_or(default_keywords),
                literal_pattern.as_deref(),
            )
        })
    }

    fn is_literal(&self, s: &str) -> bool {
        self.literal_regex.is_match(s)
    }

    fn tokenize(&self, text: &str) -> Vec<Term> {
        let mut tokens = Vec::with_capacity(text.len() / 4);
        let mut buffer = String::with_capacity(32);
        let mut char_indices = text.char_indices().peekable();

        while let Some((idx, c)) = char_indices.next() {
            let is_symbol = self.symbols.contains(&c);
            if !c.is_whitespace() & !is_symbol {
                buffer.push(c);
                continue;
            }
            if !buffer.is_empty() {
                if self.is_literal(&buffer) {
                    self.add_token(&buffer, &mut tokens, idx - buffer.len(), idx);
                } else {
                    let buffer_tokens = self.encode(&buffer);
                    let mut curr_start = 0;
                    for token in buffer_tokens {
                        self.add_token(&token, &mut tokens, curr_start, curr_start + token.len());
                        curr_start += token.len();
                    }
                    buffer.clear();
                }
            }

            if self.symbols.contains(&c) {
                tokens.push(Term::new(
                    c.to_string(),
                    c.to_string(),
                    tokens.len(),
                    idx,
                    idx + 1,
                ));
            }
        }

        if !buffer.is_empty() {
            let idx = text.len();
            if self.is_literal(&buffer) {
                self.add_token(&buffer, &mut tokens, idx - buffer.len(), idx);
            } else {
                let buffer_tokens = self.encode(&buffer);
                let mut curr_start = 0;
                for token in buffer_tokens {
                    self.add_token(&token, &mut tokens, curr_start, curr_start + token.len());
                    curr_start += token.len();
                }
                buffer.clear();
            }
        }

        tokens
    }

    fn encode(&self, buffer: &str) -> Vec<String> {
        buffer
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    }

    fn add_token(&self, buffer: &str, tokens: &mut Vec<Term>, start: usize, end: usize) {
        let term = buffer.to_owned();
        tokens.push(Term::new(term.clone(), term, tokens.len(), start, end));
    }
}

#[pyfunction]
fn tokenize_code(py: Python<'_>, text: &str) -> PyResult<Py<PyList>> {
    let tokenizer = Tokenizer::get_or_init(None, None, None);
    let tokens = tokenizer.tokenize(text);

    let py_list = PyList::empty(py);

    tokens
        .into_iter()
        .try_for_each(|token| py_list.append(Py::new(py, token)?))?;

    Ok(py_list.into())
}

#[pymodule]
fn rust_lib(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Term>()?;
    m.add_function(wrap_pyfunction!(tokenize_code, m)?)?;
    Ok(())
}

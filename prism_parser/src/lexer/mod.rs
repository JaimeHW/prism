//! Complete Python 3.12 lexer.
//!
//! A production-grade lexer supporting the full Python 3.12 specification:
//! - All operators and delimiters
//! - INDENT/DEDENT tracking
//! - All numeric formats (int, float, complex, hex, octal, binary)
//! - All string formats (raw, bytes, f-strings, triple-quoted)
//! - Unicode identifiers per PEP 3131
//! - Proper line continuation handling
//! - Comment stripping

pub mod cursor;
pub mod identifier;
pub mod indent;
pub mod number;
pub mod string;

use cursor::Cursor;
use identifier::{is_id_start, parse_identifier};
use indent::IndentStack;
use number::parse_number;
use prism_core::Span;
use string::{StringPrefix, is_string_prefix, parse_string};

use crate::token::{Token, TokenKind};

/// The Python 3.12 lexer.
#[derive(Debug)]
pub struct Lexer<'src> {
    /// Character cursor.
    cursor: Cursor<'src>,
    /// Indentation tracker.
    indent: IndentStack,
    /// Whether we need to emit a NEWLINE before the next token.
    pending_newline: bool,
    /// Whether we've reached EOF.
    at_eof: bool,
    /// Token start position for the current token.
    token_start: usize,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source code.
    #[must_use]
    pub fn new(source: &'src str) -> Self {
        Self {
            cursor: Cursor::new(source),
            indent: IndentStack::new(),
            pending_newline: false,
            at_eof: false,
            token_start: 0,
        }
    }

    /// Get the next token.
    pub fn next_token(&mut self) -> Token {
        // Handle pending DEDENT tokens
        if self.indent.has_pending_dedents() {
            self.indent.consume_dedent();
            return self.make_token(TokenKind::Dedent);
        }

        // Handle pending NEWLINE
        if self.pending_newline {
            self.pending_newline = false;
            return self.make_token(TokenKind::Newline);
        }

        // Handle EOF - drain any remaining DEDENT tokens first
        if self.at_eof {
            if self.indent.has_pending_dedents() {
                self.indent.consume_dedent();
                return self.make_token(TokenKind::Dedent);
            }
            return self.make_token(TokenKind::Eof);
        }

        // Process line start (indentation)
        if self.indent.at_line_start() && self.indent.tracking_indent() {
            if let Some(token) = self.handle_line_start() {
                // After emitting INDENT/DEDENT, mark that we're no longer at line start
                self.indent.consumed_content();
                return token;
            }
            // No INDENT/DEDENT needed, but still need to clear at_line_start
            self.indent.consumed_content();
        }

        // Skip whitespace (but not newlines at line start)
        self.skip_whitespace();

        // Record token start position
        self.token_start = self.cursor.pos();

        // Get next character
        let c = match self.cursor.bump() {
            Some(c) => c,
            None => {
                self.at_eof = true;
                // Generate final DEDENTs
                let _ = self.indent.close_all();
                if self.indent.has_pending_dedents() {
                    self.indent.consume_dedent();
                    return self.make_token(TokenKind::Dedent);
                }
                return self.make_token(TokenKind::Eof);
            }
        };

        // Mark that we've consumed content on this line
        self.indent.consumed_content();

        // Dispatch based on first character
        let kind = match c {
            // Newline
            '\n' => self.handle_newline(),
            '\r' => {
                self.cursor.eat('\n'); // Handle \r\n
                self.handle_newline()
            }

            // Comments
            '#' => {
                self.skip_comment();
                return self.next_token();
            }

            // Operators and delimiters
            '(' => {
                self.indent.open_bracket();
                TokenKind::LeftParen
            }
            ')' => {
                self.indent.close_bracket();
                TokenKind::RightParen
            }
            '[' => {
                self.indent.open_bracket();
                TokenKind::LeftBracket
            }
            ']' => {
                self.indent.close_bracket();
                TokenKind::RightBracket
            }
            '{' => {
                self.indent.open_bracket();
                TokenKind::LeftBrace
            }
            '}' => {
                self.indent.close_bracket();
                TokenKind::RightBrace
            }

            ',' => TokenKind::Comma,
            ':' => {
                if self.cursor.eat('=') {
                    TokenKind::ColonEqual
                } else {
                    TokenKind::Colon
                }
            }
            ';' => TokenKind::Semicolon,
            '~' => TokenKind::Tilde,
            '@' => {
                if self.cursor.eat('=') {
                    TokenKind::AtEqual
                } else {
                    TokenKind::At
                }
            }

            // Dot and ellipsis
            '.' => {
                if self.cursor.first() == '.' && self.cursor.second() == '.' {
                    self.cursor.bump();
                    self.cursor.bump();
                    TokenKind::Ellipsis
                } else if self.cursor.first().is_ascii_digit() {
                    // Float like .5
                    parse_number(&mut self.cursor, c)
                } else {
                    TokenKind::Dot
                }
            }

            // Arithmetic operators
            '+' => {
                if self.cursor.eat('=') {
                    TokenKind::PlusEqual
                } else {
                    TokenKind::Plus
                }
            }
            '-' => {
                if self.cursor.eat('=') {
                    TokenKind::MinusEqual
                } else if self.cursor.eat('>') {
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '*' => {
                if self.cursor.eat('*') {
                    if self.cursor.eat('=') {
                        TokenKind::DoubleStarEqual
                    } else {
                        TokenKind::DoubleStar
                    }
                } else if self.cursor.eat('=') {
                    TokenKind::StarEqual
                } else {
                    TokenKind::Star
                }
            }
            '/' => {
                if self.cursor.eat('/') {
                    if self.cursor.eat('=') {
                        TokenKind::DoubleSlashEqual
                    } else {
                        TokenKind::DoubleSlash
                    }
                } else if self.cursor.eat('=') {
                    TokenKind::SlashEqual
                } else {
                    TokenKind::Slash
                }
            }
            '%' => {
                if self.cursor.eat('=') {
                    TokenKind::PercentEqual
                } else {
                    TokenKind::Percent
                }
            }

            // Comparison and bitwise operators
            '<' => {
                if self.cursor.eat('<') {
                    if self.cursor.eat('=') {
                        TokenKind::LeftShiftEqual
                    } else {
                        TokenKind::LeftShift
                    }
                } else if self.cursor.eat('=') {
                    TokenKind::LessEqual
                } else {
                    TokenKind::Less
                }
            }
            '>' => {
                if self.cursor.eat('>') {
                    if self.cursor.eat('=') {
                        TokenKind::RightShiftEqual
                    } else {
                        TokenKind::RightShift
                    }
                } else if self.cursor.eat('=') {
                    TokenKind::GreaterEqual
                } else {
                    TokenKind::Greater
                }
            }
            '=' => {
                if self.cursor.eat('=') {
                    TokenKind::EqualEqual
                } else {
                    TokenKind::Equal
                }
            }
            '!' => {
                if self.cursor.eat('=') {
                    TokenKind::NotEqual
                } else {
                    TokenKind::Error("unexpected '!'".to_string())
                }
            }
            '&' => {
                if self.cursor.eat('=') {
                    TokenKind::AmpersandEqual
                } else {
                    TokenKind::Ampersand
                }
            }
            '|' => {
                if self.cursor.eat('=') {
                    TokenKind::PipeEqual
                } else {
                    TokenKind::Pipe
                }
            }
            '^' => {
                if self.cursor.eat('=') {
                    TokenKind::CaretEqual
                } else {
                    TokenKind::Caret
                }
            }

            // Strings
            '"' | '\'' => {
                // Put the quote back and parse the string
                let prefix = StringPrefix::default();
                self.parse_string_from_quote(c, prefix)
            }

            // Line continuation
            '\\' if self.cursor.first() == '\n' || self.cursor.first() == '\r' => {
                self.cursor.bump();
                if self.cursor.first() == '\n' {
                    self.cursor.bump();
                }
                return self.next_token();
            }

            // Identifiers, keywords, and prefixed strings
            _ if is_id_start(c) => self.handle_identifier_or_string(c),

            // Numbers
            _ if c.is_ascii_digit() => parse_number(&mut self.cursor, c),

            // Unknown
            _ => TokenKind::Error(format!("unexpected character: {:?}", c)),
        };

        self.make_token(kind)
    }
    /// Handle the start of a logical line (indentation processing).
    fn handle_line_start(&mut self) -> Option<Token> {
        // Skip blank lines and comments
        loop {
            // Measure indentation
            let mut indent = 0;
            while !self.cursor.is_eof() {
                match self.cursor.first() {
                    ' ' => {
                        indent += 1;
                        self.cursor.bump();
                    }
                    '\t' => {
                        // Tabs align to 8-space boundaries
                        indent = (indent / 8 + 1) * 8;
                        self.cursor.bump();
                    }
                    '#' => {
                        // Comment - skip to end of line
                        self.skip_comment();
                        self.cursor.eat('\r');
                        self.cursor.eat('\n');
                        indent = 0;
                        continue;
                    }
                    '\n' | '\r' => {
                        // Blank line - skip
                        self.cursor.bump();
                        if self.cursor.first() == '\n' {
                            self.cursor.bump();
                        }
                        indent = 0;
                        continue;
                    }
                    _ => break,
                }
            }

            // Check for EOF
            if self.cursor.is_eof() {
                return None;
            }

            // Process the indentation level
            match self.indent.process_indent(indent) {
                Ok(Some(true)) => {
                    // INDENT
                    self.token_start = self.cursor.pos();
                    return Some(self.make_token(TokenKind::Indent));
                }
                Ok(Some(false)) => {
                    // DEDENT(s) - will be handled by pending_dedents
                    self.token_start = self.cursor.pos();
                    if self.indent.has_pending_dedents() {
                        self.indent.consume_dedent();
                        return Some(self.make_token(TokenKind::Dedent));
                    }
                }
                Ok(None) => {
                    // Same level, continue
                }
                Err(msg) => {
                    self.token_start = self.cursor.pos();
                    return Some(self.make_token(TokenKind::Error(msg.to_string())));
                }
            }

            break;
        }

        None
    }

    /// Handle a newline character.
    fn handle_newline(&mut self) -> TokenKind {
        if self.indent.tracking_indent() {
            self.indent.new_line();
            TokenKind::Newline
        } else {
            // Inside brackets - newlines are ignored
            return self.next_token().kind;
        }
    }

    /// Skip a comment (everything until end of line).
    fn skip_comment(&mut self) {
        self.cursor.eat_while(|c| c != '\n' && c != '\r');
    }

    /// Skip whitespace (but not newlines).
    fn skip_whitespace(&mut self) {
        self.cursor.eat_while(|c| c == ' ' || c == '\t');
    }

    /// Handle an identifier, keyword, or prefixed string.
    fn handle_identifier_or_string(&mut self, first: char) -> TokenKind {
        // Check for string prefix
        if is_string_prefix(first) {
            // Collect potential prefix characters
            let mut prefix_chars = String::new();
            prefix_chars.push(first);

            // Look ahead for more prefix chars or a quote
            let mut pos = 0;
            loop {
                let c = match pos {
                    0 => self.cursor.first(),
                    1 => self.cursor.second(),
                    2 => self.cursor.third(),
                    _ => break,
                };

                if c == '"' || c == '\'' {
                    // Found a quote - this is a string
                    for _ in 0..pos {
                        self.cursor.bump();
                    }
                    let prefix = StringPrefix::from_chars(&prefix_chars);
                    if !prefix.is_valid() {
                        return TokenKind::Error("invalid string prefix".to_string());
                    }
                    return parse_string(&mut self.cursor, prefix);
                } else if is_string_prefix(c) && pos < 2 {
                    prefix_chars.push(c);
                    pos += 1;
                } else {
                    break;
                }
            }
        }

        // Regular identifier or keyword
        parse_identifier(&mut self.cursor, first)
    }

    /// Parse a string starting from a quote character.
    fn parse_string_from_quote(&mut self, _quote: char, prefix: StringPrefix) -> TokenKind {
        // We need to "unread" the quote since parse_string expects to see it
        // Create a new cursor pointing to the token_start (which includes the quote)
        // and extending to end of source
        let text = &self.cursor.source()[self.token_start..];
        let mut temp_cursor = Cursor::new(text);
        let result = parse_string(&mut temp_cursor, prefix);

        // Advance main cursor to match where temp_cursor ended
        // We need to advance by bytes, not by character count
        // temp_cursor.pos() gives us the byte position in the temp slice
        // We've already consumed 1 character (the first quote), so we need
        // to advance our cursor until it reaches token_start + temp_cursor.pos()
        let target_pos = self.token_start + temp_cursor.pos();
        while self.cursor.pos() < target_pos {
            if self.cursor.bump().is_none() {
                break;
            }
        }
        result
    }

    /// Create a token with the current span.
    fn make_token(&self, kind: TokenKind) -> Token {
        Token::new(
            kind,
            Span::new(self.token_start as u32, self.cursor.pos() as u32),
        )
    }

    /// Tokenize the entire source into a vector.
    #[must_use]
    pub fn tokenize(source: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token();
            let is_eof = token.is_eof();
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        tokens
    }
}

/// Convenience function to tokenize source code.
#[must_use]
pub fn tokenize(source: &str) -> Vec<Token> {
    Lexer::tokenize(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(s: &str) -> Vec<TokenKind> {
        Lexer::tokenize(s).into_iter().map(|t| t.kind).collect()
    }

    fn lex_no_eof(s: &str) -> Vec<TokenKind> {
        lex(s)
            .into_iter()
            .filter(|k| !matches!(k, TokenKind::Eof))
            .collect()
    }

    #[test]
    fn test_empty() {
        let tokens = lex("");
        assert_eq!(tokens, vec![TokenKind::Eof]);
    }

    #[test]
    fn test_single_identifier() {
        let tokens = lex_no_eof("hello");
        assert_eq!(tokens, vec![TokenKind::Ident("hello".to_string())]);
    }

    #[test]
    fn test_multiple_identifiers() {
        let tokens = lex_no_eof("foo bar baz");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Ident("foo".to_string()),
                TokenKind::Ident("bar".to_string()),
                TokenKind::Ident("baz".to_string()),
            ]
        );
    }

    #[test]
    fn test_operators() {
        let tokens = lex_no_eof("+ - * / // ** % @");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::DoubleSlash,
                TokenKind::DoubleStar,
                TokenKind::Percent,
                TokenKind::At,
            ]
        );
    }

    #[test]
    fn test_comparison() {
        let tokens = lex_no_eof("< > <= >= == !=");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Less,
                TokenKind::Greater,
                TokenKind::LessEqual,
                TokenKind::GreaterEqual,
                TokenKind::EqualEqual,
                TokenKind::NotEqual,
            ]
        );
    }

    #[test]
    fn test_augmented_assign() {
        let tokens = lex_no_eof("+= -= *= /=");
        assert_eq!(
            tokens,
            vec![
                TokenKind::PlusEqual,
                TokenKind::MinusEqual,
                TokenKind::StarEqual,
                TokenKind::SlashEqual,
            ]
        );
    }

    #[test]
    fn test_brackets() {
        let tokens = lex_no_eof("( ) [ ] { }");
        assert_eq!(
            tokens,
            vec![
                TokenKind::LeftParen,
                TokenKind::RightParen,
                TokenKind::LeftBracket,
                TokenKind::RightBracket,
                TokenKind::LeftBrace,
                TokenKind::RightBrace,
            ]
        );
    }

    #[test]
    fn test_punctuation() {
        let tokens = lex_no_eof(", : ; . ... -> :=");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Comma,
                TokenKind::Colon,
                TokenKind::Semicolon,
                TokenKind::Dot,
                TokenKind::Ellipsis,
                TokenKind::Arrow,
                TokenKind::ColonEqual,
            ]
        );
    }

    #[test]
    fn test_bitwise() {
        let tokens = lex_no_eof("& | ^ ~ << >>");
        assert_eq!(
            tokens,
            vec![
                TokenKind::Ampersand,
                TokenKind::Pipe,
                TokenKind::Caret,
                TokenKind::Tilde,
                TokenKind::LeftShift,
                TokenKind::RightShift,
            ]
        );
    }

    #[test]
    fn test_integer() {
        let tokens = lex_no_eof("42");
        assert!(matches!(&tokens[0], TokenKind::Int(_)));
    }

    #[test]
    fn test_hex() {
        let tokens = lex_no_eof("0xFF");
        assert_eq!(tokens, vec![TokenKind::Int(255)]);
    }

    #[test]
    fn test_comment_skipped() {
        let tokens = lex_no_eof("x # this is a comment\ny");
        assert!(tokens.contains(&TokenKind::Ident("x".to_string())));
        assert!(tokens.contains(&TokenKind::Ident("y".to_string())));
    }

    #[test]
    fn test_keywords() {
        use crate::token::Keyword;
        let tokens = lex_no_eof("if else while for def class return");
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::If)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::Else)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::While)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::For)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::Def)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::Class)));
        assert!(tokens.contains(&TokenKind::Keyword(Keyword::Return)));
    }

    #[test]
    fn test_newline() {
        let tokens = lex("x\ny");
        assert!(tokens.iter().any(|t| matches!(t, TokenKind::Newline)));
    }

    #[test]
    fn test_newline_ignored_in_brackets() {
        let tokens = lex_no_eof("(\nx\n)");
        // Should not contain NEWLINE inside brackets
        let newline_count = tokens
            .iter()
            .filter(|t| matches!(t, TokenKind::Newline))
            .count();
        assert_eq!(newline_count, 0);
    }

    #[test]
    fn test_indent_dedent() {
        let source = "if x:\n    y\nz";
        let tokens = lex(source);
        assert!(tokens.iter().any(|t| matches!(t, TokenKind::Indent)));
        assert!(tokens.iter().any(|t| matches!(t, TokenKind::Dedent)));
    }

    #[test]
    fn test_multiple_indent_levels() {
        let source = "if x:\n    if y:\n        z\n    w\nv";
        let tokens = lex(source);
        let indent_count = tokens
            .iter()
            .filter(|t| matches!(t, TokenKind::Indent))
            .count();
        let dedent_count = tokens
            .iter()
            .filter(|t| matches!(t, TokenKind::Dedent))
            .count();
        assert_eq!(indent_count, 2);
        assert_eq!(dedent_count, 2);
    }

    #[test]
    fn test_line_continuation() {
        let tokens = lex_no_eof("x + \\\ny");
        // Should be: x + y (line continuation joins lines)
        assert!(tokens.contains(&TokenKind::Ident("x".to_string())));
        assert!(tokens.contains(&TokenKind::Plus));
        assert!(tokens.contains(&TokenKind::Ident("y".to_string())));
    }

    #[test]
    fn test_arrow() {
        let tokens = lex_no_eof("def f() -> int:");
        assert!(tokens.contains(&TokenKind::Arrow));
    }

    #[test]
    fn test_walrus() {
        let tokens = lex_no_eof("if (x := 5):");
        assert!(tokens.contains(&TokenKind::ColonEqual));
    }
}

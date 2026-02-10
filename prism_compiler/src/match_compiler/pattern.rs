//! Pattern normalization and internal representation.
//!
//! Converts AST patterns to a flattened representation optimized for
//! decision tree construction and matrix operations.

use crate::compiler::CompileResult;
use prism_parser::ast::{Expr, ExprKind, Pattern, PatternKind, Singleton};
use std::sync::Arc;

// =============================================================================
// Flat Pattern Representation
// =============================================================================

/// Flattened pattern for matrix operations.
///
/// This representation is optimized for Maranget's algorithm:
/// - Uniform structure for easy comparison
/// - Constructors extracted for specialization
/// - Bindings tracked separately
#[derive(Clone, Debug)]
pub enum FlatPattern {
    /// Wildcard pattern - matches anything, no binding.
    /// Corresponds to `_` or patterns with only bindings.
    Wildcard,

    /// Variable binding - matches anything, binds value.
    /// The actual name storage is in the Binding.
    Binding(Arc<str>),

    /// Literal value equality test.
    Literal(LiteralValue),

    /// Singleton pattern (True, False, None).
    Singleton(Singleton),

    /// Sequence pattern with fixed or variable length.
    Sequence {
        /// Minimum required elements
        min_len: usize,
        /// Index of star pattern (if any)
        star_index: Option<usize>,
        /// Sub-patterns for each position
        patterns: Vec<FlatPattern>,
    },

    /// Mapping pattern.
    Mapping {
        /// Keys to match (as literal values)
        keys: Vec<LiteralValue>,
        /// Patterns for each key's value
        patterns: Vec<FlatPattern>,
        /// Rest binding name (for **rest)
        rest: Option<Arc<str>>,
    },

    /// Class pattern.
    Class {
        /// Class expression to match against
        cls: Box<Expr>,
        /// Positional patterns (matched via __match_args__)
        positional: Vec<FlatPattern>,
        /// Keyword attribute names
        kwd_attrs: Vec<Arc<str>>,
        /// Keyword patterns
        kwd_patterns: Vec<FlatPattern>,
    },

    /// OR pattern - alternatives.
    Or(Vec<FlatPattern>),

    /// AS pattern - match inner pattern and bind.
    As {
        pattern: Box<FlatPattern>,
        name: Arc<str>,
    },
}

/// Literal value for pattern matching.
#[derive(Clone, Debug, PartialEq)]
pub enum LiteralValue {
    Int(i64),
    Float(f64),
    String(Arc<str>),
    Bytes(Arc<[u8]>),
}

// =============================================================================
// Bindings
// =============================================================================

/// A binding captured during pattern matching.
#[derive(Clone, Debug)]
pub struct Binding {
    /// Variable name to bind.
    pub name: Arc<str>,
    /// How to access the value from the subject.
    pub access: AccessPath,
}

/// Path to access a nested value from the subject.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AccessPath {
    /// The root subject value.
    Root,
    /// Index into a sequence: parent[index]
    Index(Box<AccessPath>, usize),
    /// Key lookup in mapping: parent[key]
    Key(Box<AccessPath>, LiteralKey),
    /// Attribute access: parent.attr
    Attr(Box<AccessPath>, Arc<str>),
    /// Star slice: parent[start:end] where end is computed from total length
    StarSlice {
        parent: Box<AccessPath>,
        start: usize,
        from_end: usize,
    },
}

/// Key for mapping access (must be hashable).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiteralKey {
    Int(i64),
    String(Arc<str>),
    Bytes(Arc<[u8]>),
}

impl AccessPath {
    /// Create an access path for indexing.
    pub fn index(self, idx: usize) -> Self {
        AccessPath::Index(Box::new(self), idx)
    }

    /// Create an access path for key lookup.
    pub fn key(self, key: LiteralKey) -> Self {
        AccessPath::Key(Box::new(self), key)
    }

    /// Create an access path for attribute access.
    pub fn attr(self, name: Arc<str>) -> Self {
        AccessPath::Attr(Box::new(self), name)
    }
}

// =============================================================================
// Pattern Flattening
// =============================================================================

/// Flatten an AST pattern into internal representation.
///
/// Returns the flattened pattern and a list of bindings.
pub fn flatten(pattern: &Pattern) -> CompileResult<(FlatPattern, Vec<Binding>)> {
    let mut bindings = Vec::new();
    let flat = flatten_inner(pattern, AccessPath::Root, &mut bindings)?;
    Ok((flat, bindings))
}

/// Internal recursive flattening implementation.
fn flatten_inner(
    pattern: &Pattern,
    access: AccessPath,
    bindings: &mut Vec<Binding>,
) -> CompileResult<FlatPattern> {
    match &pattern.kind {
        PatternKind::MatchAs {
            pattern: inner,
            name,
        } => {
            match (inner, name) {
                // Wildcard: `_`
                (None, None) => Ok(FlatPattern::Wildcard),

                // Simple binding: `x`
                (None, Some(name)) => {
                    let name_arc: Arc<str> = Arc::from(name.as_str());
                    bindings.push(Binding {
                        name: name_arc.clone(),
                        access,
                    });
                    Ok(FlatPattern::Binding(name_arc))
                }

                // Pattern with binding: `pattern as x`
                (Some(inner_pattern), Some(name)) => {
                    let name_arc: Arc<str> = Arc::from(name.as_str());
                    bindings.push(Binding {
                        name: name_arc.clone(),
                        access: access.clone(),
                    });
                    let inner_flat = flatten_inner(inner_pattern, access, bindings)?;
                    Ok(FlatPattern::As {
                        pattern: Box::new(inner_flat),
                        name: name_arc,
                    })
                }

                // Pattern without binding (shouldn't occur in valid AST)
                (Some(inner_pattern), None) => flatten_inner(inner_pattern, access, bindings),
            }
        }

        PatternKind::MatchValue(expr) => flatten_value_pattern(expr),

        PatternKind::MatchSingleton(s) => Ok(FlatPattern::Singleton(*s)),

        PatternKind::MatchSequence(patterns) => flatten_sequence(patterns, access, bindings),

        PatternKind::MatchMapping {
            keys,
            patterns,
            rest,
        } => flatten_mapping(keys, patterns, rest.as_ref(), access, bindings),

        PatternKind::MatchClass {
            cls,
            patterns,
            kwd_attrs,
            kwd_patterns,
        } => flatten_class(cls, patterns, kwd_attrs, kwd_patterns, access, bindings),

        PatternKind::MatchStar(name) => {
            // Star pattern is handled in sequence flattening
            // This case handles standalone star (shouldn't occur)
            if let Some(name) = name {
                let name_arc: Arc<str> = Arc::from(name.as_str());
                bindings.push(Binding {
                    name: name_arc.clone(),
                    access,
                });
                Ok(FlatPattern::Binding(name_arc))
            } else {
                Ok(FlatPattern::Wildcard)
            }
        }

        PatternKind::MatchOr(alternatives) => {
            let mut flat_alts = Vec::with_capacity(alternatives.len());
            for alt in alternatives {
                // Each alternative starts fresh for bindings
                // (they must all bind the same names)
                let (flat, _alt_bindings) = flatten(alt)?;
                flat_alts.push(flat);
            }
            // Use bindings from first alternative
            let (_, first_bindings) = flatten(&alternatives[0])?;
            for binding in first_bindings {
                // Adjust access path - bindings in OR come from whichever branch matched
                bindings.push(Binding {
                    name: binding.name,
                    access: access.clone(),
                });
            }
            Ok(FlatPattern::Or(flat_alts))
        }
    }
}

/// Flatten a value pattern (literal expression).
fn flatten_value_pattern(expr: &Expr) -> CompileResult<FlatPattern> {
    match &expr.kind {
        ExprKind::Int(n) => Ok(FlatPattern::Literal(LiteralValue::Int(*n))),
        ExprKind::Float(f) => Ok(FlatPattern::Literal(LiteralValue::Float(*f))),
        ExprKind::String(s) => Ok(FlatPattern::Literal(LiteralValue::String(Arc::from(
            s.value.as_str(),
        )))),
        ExprKind::Bytes(b) => Ok(FlatPattern::Literal(LiteralValue::Bytes(Arc::from(
            b.as_slice(),
        )))),
        // For complex expressions (attribute access, etc.),
        // we need runtime evaluation - store the expression
        _ => {
            // This handles patterns like `Color.RED`
            Ok(FlatPattern::Literal(LiteralValue::String(Arc::from(
                format!("{:?}", expr.kind),
            ))))
        }
    }
}

/// Flatten a sequence pattern.
fn flatten_sequence(
    patterns: &[Pattern],
    access: AccessPath,
    bindings: &mut Vec<Binding>,
) -> CompileResult<FlatPattern> {
    let mut flat_patterns = Vec::with_capacity(patterns.len());
    let mut star_index = None;

    for (i, pattern) in patterns.iter().enumerate() {
        if matches!(pattern.kind, PatternKind::MatchStar(_)) {
            star_index = Some(i);
        }

        let element_access = if star_index.is_some() && i > star_index.unwrap() {
            // After star: access from end
            AccessPath::StarSlice {
                parent: Box::new(access.clone()),
                start: i,
                from_end: patterns.len() - i,
            }
        } else if star_index == Some(i) {
            // The star itself
            AccessPath::StarSlice {
                parent: Box::new(access.clone()),
                start: i,
                from_end: patterns.len() - i - 1,
            }
        } else {
            // Before star or no star: normal index
            access.clone().index(i)
        };

        let flat = flatten_inner(pattern, element_access, bindings)?;
        flat_patterns.push(flat);
    }

    let min_len = if star_index.is_some() {
        patterns.len() - 1 // Star can match zero elements
    } else {
        patterns.len()
    };

    Ok(FlatPattern::Sequence {
        min_len,
        star_index,
        patterns: flat_patterns,
    })
}

/// Flatten a mapping pattern.
fn flatten_mapping(
    keys: &[Expr],
    patterns: &[Pattern],
    rest: Option<&String>,
    access: AccessPath,
    bindings: &mut Vec<Binding>,
) -> CompileResult<FlatPattern> {
    let mut flat_keys = Vec::with_capacity(keys.len());
    let mut flat_patterns = Vec::with_capacity(patterns.len());

    for (key, pattern) in keys.iter().zip(patterns) {
        let literal_key = expr_to_literal_value(key)?;
        let key_access = access.clone().key(literal_to_key(&literal_key));

        flat_keys.push(literal_key);
        flat_patterns.push(flatten_inner(pattern, key_access, bindings)?);
    }

    let rest_arc = rest.map(|r| {
        let name: Arc<str> = Arc::from(r.as_str());
        // Rest binding gets the remaining dict
        bindings.push(Binding {
            name: name.clone(),
            access: access.clone(), // Will be filtered at runtime
        });
        name
    });

    Ok(FlatPattern::Mapping {
        keys: flat_keys,
        patterns: flat_patterns,
        rest: rest_arc,
    })
}

/// Flatten a class pattern.
fn flatten_class(
    cls: &Expr,
    positional: &[Pattern],
    kwd_attrs: &[String],
    kwd_patterns: &[Pattern],
    access: AccessPath,
    bindings: &mut Vec<Binding>,
) -> CompileResult<FlatPattern> {
    let mut flat_positional = Vec::with_capacity(positional.len());
    for (i, pattern) in positional.iter().enumerate() {
        // Positional args mapped via __match_args__ at runtime
        let pos_access = access.clone().attr(Arc::from(format!("__match_pos_{}", i)));
        flat_positional.push(flatten_inner(pattern, pos_access, bindings)?);
    }

    let mut flat_kwd_attrs = Vec::with_capacity(kwd_attrs.len());
    let mut flat_kwd_patterns = Vec::with_capacity(kwd_patterns.len());

    for (attr, pattern) in kwd_attrs.iter().zip(kwd_patterns) {
        let attr_arc: Arc<str> = Arc::from(attr.as_str());
        let attr_access = access.clone().attr(attr_arc.clone());

        flat_kwd_attrs.push(attr_arc);
        flat_kwd_patterns.push(flatten_inner(pattern, attr_access, bindings)?);
    }

    Ok(FlatPattern::Class {
        cls: Box::new(cls.clone()),
        positional: flat_positional,
        kwd_attrs: flat_kwd_attrs,
        kwd_patterns: flat_kwd_patterns,
    })
}

/// Convert expression to literal value.
fn expr_to_literal_value(expr: &Expr) -> CompileResult<LiteralValue> {
    match &expr.kind {
        ExprKind::Int(n) => Ok(LiteralValue::Int(*n)),
        ExprKind::Float(f) => Ok(LiteralValue::Float(*f)),
        ExprKind::String(s) => Ok(LiteralValue::String(Arc::from(s.value.as_str()))),
        ExprKind::Bytes(b) => Ok(LiteralValue::Bytes(Arc::from(b.as_slice()))),
        _ => Ok(LiteralValue::String(Arc::from(format!("{:?}", expr.kind)))),
    }
}

/// Convert literal value to hashable key.
pub fn literal_to_key(lit: &LiteralValue) -> LiteralKey {
    match lit {
        LiteralValue::Int(n) => LiteralKey::Int(*n),
        LiteralValue::String(s) => LiteralKey::String(s.clone()),
        LiteralValue::Bytes(b) => LiteralKey::Bytes(b.clone()),
        LiteralValue::Float(_) => {
            // Floats not hashable - convert to string representation
            LiteralKey::String(Arc::from(format!("{:?}", lit)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::Span;

    fn make_pattern(kind: PatternKind) -> Pattern {
        Pattern {
            kind,
            span: Span::new(0, 0),
        }
    }

    #[test]
    fn test_flatten_wildcard() {
        let pat = make_pattern(PatternKind::MatchAs {
            pattern: None,
            name: None,
        });
        let (flat, bindings) = flatten(&pat).unwrap();
        assert!(matches!(flat, FlatPattern::Wildcard));
        assert!(bindings.is_empty());
    }

    #[test]
    fn test_flatten_binding() {
        let pat = make_pattern(PatternKind::MatchAs {
            pattern: None,
            name: Some("x".to_string()),
        });
        let (flat, bindings) = flatten(&pat).unwrap();
        assert!(matches!(flat, FlatPattern::Binding(ref n) if n.as_ref() == "x"));
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].name.as_ref(), "x");
    }

    #[test]
    fn test_flatten_singleton() {
        let pat = make_pattern(PatternKind::MatchSingleton(Singleton::None));
        let (flat, _) = flatten(&pat).unwrap();
        assert!(matches!(flat, FlatPattern::Singleton(Singleton::None)));
    }

    #[test]
    fn test_flatten_literal_int() {
        let expr = Expr::new(ExprKind::Int(42), Span::new(0, 0));
        let pat = make_pattern(PatternKind::MatchValue(Box::new(expr)));
        let (flat, _) = flatten(&pat).unwrap();
        assert!(matches!(flat, FlatPattern::Literal(LiteralValue::Int(42))));
    }

    #[test]
    fn test_flatten_sequence_fixed() {
        let patterns = vec![
            make_pattern(PatternKind::MatchAs {
                pattern: None,
                name: Some("a".to_string()),
            }),
            make_pattern(PatternKind::MatchAs {
                pattern: None,
                name: Some("b".to_string()),
            }),
        ];
        let pat = make_pattern(PatternKind::MatchSequence(patterns));
        let (flat, bindings) = flatten(&pat).unwrap();

        if let FlatPattern::Sequence {
            min_len,
            star_index,
            patterns,
        } = flat
        {
            assert_eq!(min_len, 2);
            assert!(star_index.is_none());
            assert_eq!(patterns.len(), 2);
        } else {
            panic!("Expected Sequence pattern");
        }
        assert_eq!(bindings.len(), 2);
    }

    #[test]
    fn test_access_path_chaining() {
        let path = AccessPath::Root.index(0).attr(Arc::from("x"));
        assert!(matches!(path, AccessPath::Attr(_, _)));
    }
}

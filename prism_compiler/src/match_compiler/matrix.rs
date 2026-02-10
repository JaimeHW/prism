//! Pattern matrix for Maranget's algorithm.
//!
//! The pattern matrix is the core data structure for decision tree compilation.
//! It represents the remaining pattern matching work as a 2D array where:
//! - Rows represent different match cases
//! - Columns represent different positions to test in the subject

use super::pattern::{AccessPath, Binding, FlatPattern, LiteralValue};
use prism_parser::ast::{Expr, Singleton};
use std::collections::HashSet;

// =============================================================================
// Pattern Matrix
// =============================================================================

/// A row in the pattern matrix.
#[derive(Clone, Debug)]
pub struct PatternRow {
    /// Remaining patterns to match (columns).
    pub patterns: Vec<FlatPattern>,
    /// Bindings collected so far.
    pub bindings: Vec<Binding>,
    /// Guard expression (if any).
    pub guard: Option<Expr>,
    /// Action index (which case body to execute).
    pub action: usize,
}

/// Pattern matrix (P × A in Maranget's notation).
///
/// Represents the state of pattern matching compilation as a 2D matrix
/// where each row is a match case and columns are pattern positions.
#[derive(Clone, Debug)]
pub struct PatternMatrix {
    /// Rows of the matrix.
    pub rows: Vec<PatternRow>,
    /// Access paths for each column.
    pub column_access: Vec<AccessPath>,
}

impl PatternMatrix {
    /// Create a new pattern matrix from rows.
    pub fn new(rows: Vec<PatternRow>) -> Self {
        let column_count = rows.first().map(|r| r.patterns.len()).unwrap_or(0);
        let column_access = (0..column_count).map(|_| AccessPath::Root).collect();

        Self {
            rows,
            column_access,
        }
    }

    /// Create a matrix with explicit column access paths.
    pub fn with_access(rows: Vec<PatternRow>, column_access: Vec<AccessPath>) -> Self {
        Self {
            rows,
            column_access,
        }
    }

    /// Check if matrix is empty (no rows).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Get number of rows.
    #[inline]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Get number of columns.
    #[inline]
    pub fn column_count(&self) -> usize {
        self.rows.first().map(|r| r.patterns.len()).unwrap_or(0)
    }

    /// Check if the first row has all wildcard/binding patterns (match-all).
    ///
    /// In Maranget's algorithm, we only consider the first row for immediate
    /// matching - if it's all irrefutable, we can return a Leaf.
    pub fn first_match_all(&self) -> Option<&PatternRow> {
        self.rows
            .first()
            .filter(|row| row.patterns.iter().all(|p| is_irrefutable(p)))
    }

    /// Select the best column to test using necessity heuristic.
    ///
    /// The heuristic prefers columns where:
    /// 1. More rows have refutable patterns (higher necessity)
    /// 2. Fewer distinct constructors (smaller switch)
    /// 3. Earlier position (for deterministic output)
    pub fn select_column(&self) -> usize {
        if self.column_count() == 0 {
            return 0;
        }

        let mut best_col = 0;
        let mut best_score = (0i32, 0i32, 0i32);

        for col in 0..self.column_count() {
            let necessity = self.column_necessity(col) as i32;
            let ctor_count = self.distinct_constructors(col).len() as i32;
            // Higher necessity is better, fewer constructors is better
            let score = (necessity, -ctor_count, -(col as i32));

            if score > best_score {
                best_score = score;
                best_col = col;
            }
        }

        best_col
    }

    /// Calculate necessity score for a column.
    /// Higher score = more patterns that require testing.
    fn column_necessity(&self, col: usize) -> usize {
        self.rows
            .iter()
            .filter(|row| !is_irrefutable(&row.patterns[col]))
            .count()
    }

    /// Get distinct constructors used in a column.
    pub fn distinct_constructors(&self, col: usize) -> Vec<Constructor> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for row in &self.rows {
            if let Some(ctor) = pattern_constructor(&row.patterns[col]) {
                let key = ctor.discriminant_key();
                if seen.insert(key) {
                    result.push(ctor);
                }
            }
        }

        result
    }

    /// Get access path for a column.
    pub fn access_path(&self, col: usize) -> AccessPath {
        self.column_access
            .get(col)
            .cloned()
            .unwrap_or(AccessPath::Root)
    }

    /// Specialize the matrix for a specific constructor at the given column.
    ///
    /// This filters rows to those that match the constructor and expands
    /// sub-patterns into new columns.
    pub fn specialize(&self, col: usize, ctor: &Constructor) -> PatternMatrix {
        let base_access = self.access_path(col);
        let mut new_rows = Vec::new();

        for row in &self.rows {
            if let Some(expanded) = specialize_row(row, col, ctor, &base_access) {
                new_rows.push(expanded);
            }
        }

        // Build new column access paths
        let sub_access = ctor.sub_access_paths(&base_access);
        let mut new_access = Vec::new();
        for (i, access) in self.column_access.iter().enumerate() {
            if i < col {
                new_access.push(access.clone());
            } else if i == col {
                new_access.extend(sub_access.clone());
            } else {
                new_access.push(access.clone());
            }
        }

        PatternMatrix::with_access(new_rows, new_access)
    }

    /// Compute the default matrix for a column.
    ///
    /// The default matrix contains rows that match regardless of the
    /// constructor (wildcards, bindings, and compatible OR patterns).
    pub fn default_matrix(&self, col: usize) -> PatternMatrix {
        let mut new_rows = Vec::new();

        for row in &self.rows {
            if let Some(new_row) = default_row(row, col) {
                new_rows.push(new_row);
            }
        }

        // Remove the tested column from access paths
        let new_access: Vec<_> = self
            .column_access
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != col)
            .map(|(_, a)| a.clone())
            .collect();

        PatternMatrix::with_access(new_rows, new_access)
    }
}

// =============================================================================
// Constructors
// =============================================================================

/// Constructor types for pattern specialization.
#[derive(Clone, Debug)]
pub enum Constructor {
    /// Literal value.
    Literal(LiteralValue),
    /// Singleton (True, False, None).
    Singleton(Singleton),
    /// Sequence with specific length (or minimum with star).
    Sequence { len: usize, has_star: bool },
    /// Mapping with specific keys.
    Mapping { keys: Vec<LiteralValue> },
    /// Class with specific type.
    Class { cls: Box<Expr> },
}

impl Constructor {
    /// Get a key for deduplication.
    pub fn discriminant_key(&self) -> String {
        match self {
            Constructor::Literal(lit) => format!("lit:{:?}", lit),
            Constructor::Singleton(s) => format!("singleton:{:?}", s),
            Constructor::Sequence { len, has_star } => format!("seq:{}:{}", len, has_star),
            Constructor::Mapping { keys } => format!("map:{:?}", keys),
            Constructor::Class { cls } => format!("class:{:?}", cls),
        }
    }

    /// Get sub-access paths for this constructor's children.
    pub fn sub_access_paths(&self, base: &AccessPath) -> Vec<AccessPath> {
        match self {
            Constructor::Literal(_) | Constructor::Singleton(_) => vec![],

            Constructor::Sequence {
                len,
                has_star: false,
            } => (0..*len).map(|i| base.clone().index(i)).collect(),

            Constructor::Sequence {
                len,
                has_star: true,
            } => {
                // Star sequences are more complex - handled specially
                (0..*len).map(|i| base.clone().index(i)).collect()
            }

            Constructor::Mapping { keys } => {
                use super::pattern::literal_to_key;
                keys.iter()
                    .map(|k| base.clone().key(literal_to_key(k)))
                    .collect()
            }

            Constructor::Class { .. } => {
                // Class attributes handled via __match_args__
                vec![]
            }
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a pattern is irrefutable (always matches).
fn is_irrefutable(pattern: &FlatPattern) -> bool {
    match pattern {
        FlatPattern::Wildcard | FlatPattern::Binding(_) => true,
        FlatPattern::As { pattern, .. } => is_irrefutable(pattern),
        _ => false,
    }
}

/// Extract the constructor from a pattern.
fn pattern_constructor(pattern: &FlatPattern) -> Option<Constructor> {
    match pattern {
        FlatPattern::Wildcard | FlatPattern::Binding(_) => None,

        FlatPattern::Literal(lit) => Some(Constructor::Literal(lit.clone())),
        FlatPattern::Singleton(s) => Some(Constructor::Singleton(*s)),

        FlatPattern::Sequence {
            min_len,
            star_index,
            ..
        } => Some(Constructor::Sequence {
            len: *min_len,
            has_star: star_index.is_some(),
        }),

        FlatPattern::Mapping { keys, .. } => Some(Constructor::Mapping { keys: keys.clone() }),

        FlatPattern::Class { cls, .. } => Some(Constructor::Class { cls: cls.clone() }),

        FlatPattern::Or(alts) => {
            // Take constructor from first alternative
            alts.first().and_then(pattern_constructor)
        }

        FlatPattern::As { pattern, .. } => pattern_constructor(pattern),
    }
}

/// Specialize a row for a constructor.
fn specialize_row(
    row: &PatternRow,
    col: usize,
    ctor: &Constructor,
    _base_access: &AccessPath,
) -> Option<PatternRow> {
    let pattern = &row.patterns[col];

    match pattern {
        // Wildcards/bindings match any constructor
        FlatPattern::Wildcard | FlatPattern::Binding(_) => {
            // Expand with wildcards for sub-patterns
            let sub_patterns = ctor
                .sub_access_paths(&AccessPath::Root)
                .iter()
                .map(|_| FlatPattern::Wildcard)
                .collect::<Vec<_>>();

            Some(expand_row(row, col, &sub_patterns))
        }

        // Literal must match exactly
        FlatPattern::Literal(lit) => {
            if let Constructor::Literal(ctor_lit) = ctor {
                if lit == ctor_lit {
                    Some(expand_row(row, col, &[]))
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Singleton must match exactly
        FlatPattern::Singleton(s) => {
            if let Constructor::Singleton(ctor_s) = ctor {
                if s == ctor_s {
                    Some(expand_row(row, col, &[]))
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Sequence: check length compatibility and expand sub-patterns
        FlatPattern::Sequence {
            min_len,
            patterns,
            star_index: _,
            ..
        } => {
            if let Constructor::Sequence { len: ctor_len, .. } = ctor {
                if *min_len <= *ctor_len {
                    Some(expand_row(row, col, patterns))
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Mapping: check key compatibility
        FlatPattern::Mapping { keys, patterns, .. } => {
            if let Constructor::Mapping { keys: ctor_keys } = ctor {
                // Check if our keys are a subset of constructor keys
                if keys.iter().all(|k| ctor_keys.contains(k)) {
                    Some(expand_row(row, col, patterns))
                } else {
                    None
                }
            } else {
                None
            }
        }

        // Class: check type compatibility
        FlatPattern::Class {
            cls: _,
            positional,
            kwd_patterns,
            ..
        } => {
            if let Constructor::Class { cls: _ } = ctor {
                // For now, simple structural match
                let sub_patterns: Vec<_> = positional
                    .iter()
                    .chain(kwd_patterns.iter())
                    .cloned()
                    .collect();
                Some(expand_row(row, col, &sub_patterns))
            } else {
                None
            }
        }

        // OR: try to specialize any alternative
        FlatPattern::Or(alts) => {
            for alt in alts {
                let temp_row = PatternRow {
                    patterns: {
                        let mut p = row.patterns.clone();
                        p[col] = alt.clone();
                        p
                    },
                    bindings: row.bindings.clone(),
                    guard: row.guard.clone(),
                    action: row.action,
                };
                if let Some(result) = specialize_row(&temp_row, col, ctor, _base_access) {
                    return Some(result);
                }
            }
            None
        }

        // AS pattern: unwrap and specialize inner
        FlatPattern::As { pattern, name: _ } => {
            let temp_row = PatternRow {
                patterns: {
                    let mut p = row.patterns.clone();
                    p[col] = (**pattern).clone();
                    p
                },
                bindings: row.bindings.clone(),
                guard: row.guard.clone(),
                action: row.action,
            };
            specialize_row(&temp_row, col, ctor, _base_access)
        }
    }
}

/// Expand a row by replacing column with sub-patterns.
fn expand_row(row: &PatternRow, col: usize, sub_patterns: &[FlatPattern]) -> PatternRow {
    let mut new_patterns = Vec::with_capacity(row.patterns.len() - 1 + sub_patterns.len());
    new_patterns.extend(row.patterns[..col].iter().cloned());
    new_patterns.extend(sub_patterns.iter().cloned());
    new_patterns.extend(row.patterns[col + 1..].iter().cloned());

    PatternRow {
        patterns: new_patterns,
        bindings: row.bindings.clone(),
        guard: row.guard.clone(),
        action: row.action,
    }
}

/// Get the default row (if pattern is irrefutable).
fn default_row(row: &PatternRow, col: usize) -> Option<PatternRow> {
    let pattern = &row.patterns[col];

    if is_irrefutable(pattern) {
        // Remove the column
        let new_patterns: Vec<_> = row
            .patterns
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != col)
            .map(|(_, p)| p.clone())
            .collect();

        Some(PatternRow {
            patterns: new_patterns,
            bindings: row.bindings.clone(),
            guard: row.guard.clone(),
            action: row.action,
        })
    } else if let FlatPattern::Or(alts) = pattern {
        // OR pattern: include if any alternative is irrefutable
        if alts.iter().any(is_irrefutable) {
            let new_patterns: Vec<_> = row
                .patterns
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != col)
                .map(|(_, p)| p.clone())
                .collect();

            Some(PatternRow {
                patterns: new_patterns,
                bindings: row.bindings.clone(),
                guard: row.guard.clone(),
                action: row.action,
            })
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wildcard_row(action: usize) -> PatternRow {
        PatternRow {
            patterns: vec![FlatPattern::Wildcard],
            bindings: vec![],
            guard: None,
            action,
        }
    }

    fn literal_row(val: i64, action: usize) -> PatternRow {
        PatternRow {
            patterns: vec![FlatPattern::Literal(LiteralValue::Int(val))],
            bindings: vec![],
            guard: None,
            action,
        }
    }

    #[test]
    fn test_empty_matrix() {
        let matrix = PatternMatrix::new(vec![]);
        assert!(matrix.is_empty());
        assert_eq!(matrix.row_count(), 0);
    }

    #[test]
    fn test_first_match_all() {
        let matrix = PatternMatrix::new(vec![wildcard_row(0)]);
        assert!(matrix.first_match_all().is_some());
    }

    #[test]
    fn test_first_match_all_none() {
        let matrix = PatternMatrix::new(vec![literal_row(42, 0)]);
        assert!(matrix.first_match_all().is_none());
    }

    #[test]
    fn test_select_column_prefers_refutable() {
        let row = PatternRow {
            patterns: vec![
                FlatPattern::Wildcard,
                FlatPattern::Literal(LiteralValue::Int(1)),
            ],
            bindings: vec![],
            guard: None,
            action: 0,
        };
        let matrix = PatternMatrix::new(vec![row]);
        // Column 1 (literal) should be selected over column 0 (wildcard)
        assert_eq!(matrix.select_column(), 1);
    }

    #[test]
    fn test_distinct_constructors() {
        let matrix = PatternMatrix::new(vec![
            literal_row(1, 0),
            literal_row(2, 1),
            literal_row(1, 2), // Duplicate
        ]);
        let ctors = matrix.distinct_constructors(0);
        assert_eq!(ctors.len(), 2); // Only 2 distinct literals
    }

    #[test]
    fn test_specialize_literal() {
        let matrix =
            PatternMatrix::new(vec![literal_row(1, 0), literal_row(2, 1), wildcard_row(2)]);

        let ctor = Constructor::Literal(LiteralValue::Int(1));
        let specialized = matrix.specialize(0, &ctor);

        // Should contain row 0 (matches) and row 2 (wildcard matches anything)
        assert_eq!(specialized.row_count(), 2);
        assert_eq!(specialized.rows[0].action, 0);
        assert_eq!(specialized.rows[1].action, 2);
    }

    #[test]
    fn test_default_matrix() {
        let matrix = PatternMatrix::new(vec![literal_row(1, 0), wildcard_row(1)]);

        let default = matrix.default_matrix(0);

        // Only wildcard row included
        assert_eq!(default.row_count(), 1);
        assert_eq!(default.rows[0].action, 1);
    }
}

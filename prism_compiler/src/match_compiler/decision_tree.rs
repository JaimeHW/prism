//! Decision tree generation using Maranget's algorithm.
//!
//! This module implements the core of optimal pattern matching compilation.
//! Maranget's algorithm generates a decision tree that:
//! - Tests each value at most once
//! - Selects columns to minimize tree size
//! - Handles guards with efficient backtracking

use super::matrix::{Constructor, PatternMatrix};
use super::pattern::{AccessPath, Binding, LiteralValue};
use prism_parser::ast::Expr;

// =============================================================================
// Decision Tree
// =============================================================================

/// A node in the decision tree.
#[derive(Clone, Debug)]
pub enum DecisionTree {
    /// Leaf node: match succeeded, execute action.
    Leaf {
        /// Bindings to store before executing body.
        bindings: Vec<Binding>,
        /// Optional guard expression.
        guard: Option<Expr>,
        /// Action index (case body to execute).
        action: usize,
        /// Fallback tree if guard fails.
        fallback: Option<Box<DecisionTree>>,
    },

    /// Failure node: no match found.
    Fail,

    /// Switch on a value.
    Switch {
        /// Access path to the value being tested.
        access: AccessPath,
        /// Test cases for each constructor.
        cases: Vec<SwitchCase>,
        /// Default case (for wildcards/bindings).
        default: Option<Box<DecisionTree>>,
    },

    /// Type check node (for class patterns).
    TypeCheck {
        /// Access path to the value being tested.
        access: AccessPath,
        /// Class to check against.
        cls: Box<Expr>,
        /// Tree if check succeeds.
        success: Box<DecisionTree>,
        /// Tree if check fails (default/next pattern).
        failure: Box<DecisionTree>,
    },

    /// Sequence length check.
    SequenceCheck {
        /// Access path to the sequence.
        access: AccessPath,
        /// Minimum length required.
        min_len: usize,
        /// Exact length (if no star pattern).
        exact: bool,
        /// Tree if check succeeds.
        success: Box<DecisionTree>,
        /// Tree if check fails.
        failure: Box<DecisionTree>,
    },

    /// Mapping keys check.
    MappingCheck {
        /// Access path to the mapping.
        access: AccessPath,
        /// Keys that must exist.
        keys: Vec<LiteralValue>,
        /// Tree if all keys exist.
        success: Box<DecisionTree>,
        /// Tree if any key missing.
        failure: Box<DecisionTree>,
    },
}

/// A case in a switch node.
#[derive(Clone, Debug)]
pub struct SwitchCase {
    /// Constructor to test for.
    pub ctor: Constructor,
    /// Subtree if constructor matches.
    pub tree: DecisionTree,
}

// =============================================================================
// Maranget's Algorithm
// =============================================================================

/// Compile a pattern matrix into a decision tree.
///
/// This is the main entry point implementing Maranget's algorithm.
pub fn compile(matrix: &PatternMatrix) -> DecisionTree {
    compile_matrix(matrix, 0)
}

/// Core recursive compilation function.
fn compile_matrix(matrix: &PatternMatrix, depth: usize) -> DecisionTree {
    // Sanity check for runaway recursion
    if depth > 100 {
        return DecisionTree::Fail;
    }

    // Base case 1: empty matrix = no match possible
    if matrix.is_empty() {
        return DecisionTree::Fail;
    }

    // Base case 2: first row matches all columns
    if let Some(row) = matrix.first_match_all() {
        // Check if there's a guard
        if row.guard.is_some() {
            // Need fallback for guard failure
            let remaining = PatternMatrix::new(matrix.rows.iter().skip(1).cloned().collect());
            let fallback = compile_matrix(&remaining, depth + 1);

            return DecisionTree::Leaf {
                bindings: row.bindings.clone(),
                guard: row.guard.clone(),
                action: row.action,
                fallback: Some(Box::new(fallback)),
            };
        } else {
            return DecisionTree::Leaf {
                bindings: row.bindings.clone(),
                guard: None,
                action: row.action,
                fallback: None,
            };
        }
    }

    // Base case 3: no columns left but rows present = wildcard match
    if matrix.column_count() == 0 {
        if let Some(row) = matrix.rows.first() {
            return DecisionTree::Leaf {
                bindings: row.bindings.clone(),
                guard: row.guard.clone(),
                action: row.action,
                fallback: None,
            };
        }
        return DecisionTree::Fail;
    }

    // Recursive case: select column and specialize
    let col = matrix.select_column();
    let access = matrix.access_path(col);
    let constructors = matrix.distinct_constructors(col);

    // Check for special pattern types that need dedicated nodes
    if let Some(first_ctor) = constructors.first() {
        match first_ctor {
            Constructor::Sequence { len, has_star } => {
                return compile_sequence_check(matrix, col, *len, *has_star, depth);
            }
            Constructor::Mapping { keys } => {
                return compile_mapping_check(matrix, col, keys, depth);
            }
            Constructor::Class { cls } => {
                return compile_type_check(matrix, col, cls, depth);
            }
            _ => {}
        }
    }

    // Build switch cases for each constructor
    let cases: Vec<SwitchCase> = constructors
        .iter()
        .map(|ctor| {
            let specialized = matrix.specialize(col, ctor);
            SwitchCase {
                ctor: ctor.clone(),
                tree: compile_matrix(&specialized, depth + 1),
            }
        })
        .collect();

    // Build default case
    let default_matrix = matrix.default_matrix(col);
    let default = if !default_matrix.is_empty() {
        Some(Box::new(compile_matrix(&default_matrix, depth + 1)))
    } else {
        None
    };

    DecisionTree::Switch {
        access,
        cases,
        default,
    }
}

/// Compile a sequence length check.
fn compile_sequence_check(
    matrix: &PatternMatrix,
    col: usize,
    min_len: usize,
    has_star: bool,
    depth: usize,
) -> DecisionTree {
    let access = matrix.access_path(col);

    // For sequences, we check length then specialize
    let ctor = Constructor::Sequence {
        len: min_len,
        has_star,
    };
    let specialized = matrix.specialize(col, &ctor);
    let default = matrix.default_matrix(col);

    let success = compile_matrix(&specialized, depth + 1);
    let failure = if !default.is_empty() {
        compile_matrix(&default, depth + 1)
    } else {
        DecisionTree::Fail
    };

    DecisionTree::SequenceCheck {
        access,
        min_len,
        exact: !has_star,
        success: Box::new(success),
        failure: Box::new(failure),
    }
}

/// Compile a mapping keys check.
fn compile_mapping_check(
    matrix: &PatternMatrix,
    col: usize,
    keys: &[LiteralValue],
    depth: usize,
) -> DecisionTree {
    let access = matrix.access_path(col);

    let ctor = Constructor::Mapping {
        keys: keys.to_vec(),
    };
    let specialized = matrix.specialize(col, &ctor);
    let default = matrix.default_matrix(col);

    let success = compile_matrix(&specialized, depth + 1);
    let failure = if !default.is_empty() {
        compile_matrix(&default, depth + 1)
    } else {
        DecisionTree::Fail
    };

    DecisionTree::MappingCheck {
        access,
        keys: keys.to_vec(),
        success: Box::new(success),
        failure: Box::new(failure),
    }
}

/// Compile a type check for class patterns.
fn compile_type_check(
    matrix: &PatternMatrix,
    col: usize,
    cls: &Expr,
    depth: usize,
) -> DecisionTree {
    let access = matrix.access_path(col);

    let ctor = Constructor::Class {
        cls: Box::new(cls.clone()),
    };
    let specialized = matrix.specialize(col, &ctor);
    let default = matrix.default_matrix(col);

    let success = compile_matrix(&specialized, depth + 1);
    let failure = if !default.is_empty() {
        compile_matrix(&default, depth + 1)
    } else {
        DecisionTree::Fail
    };

    DecisionTree::TypeCheck {
        access,
        cls: Box::new(cls.clone()),
        success: Box::new(success),
        failure: Box::new(failure),
    }
}

// =============================================================================
// Analysis Utilities
// =============================================================================

impl DecisionTree {
    /// Count total nodes in the tree.
    pub fn node_count(&self) -> usize {
        match self {
            DecisionTree::Leaf { fallback, .. } => {
                1 + fallback.as_ref().map(|f| f.node_count()).unwrap_or(0)
            }
            DecisionTree::Fail => 1,
            DecisionTree::Switch { cases, default, .. } => {
                1 + cases.iter().map(|c| c.tree.node_count()).sum::<usize>()
                    + default.as_ref().map(|d| d.node_count()).unwrap_or(0)
            }
            DecisionTree::TypeCheck {
                success, failure, ..
            }
            | DecisionTree::SequenceCheck {
                success, failure, ..
            }
            | DecisionTree::MappingCheck {
                success, failure, ..
            } => 1 + success.node_count() + failure.node_count(),
        }
    }

    /// Calculate maximum depth of the tree.
    pub fn max_depth(&self) -> usize {
        match self {
            DecisionTree::Leaf { fallback, .. } => {
                1 + fallback.as_ref().map(|f| f.max_depth()).unwrap_or(0)
            }
            DecisionTree::Fail => 1,
            DecisionTree::Switch { cases, default, .. } => {
                let case_max = cases.iter().map(|c| c.tree.max_depth()).max().unwrap_or(0);
                let default_max = default.as_ref().map(|d| d.max_depth()).unwrap_or(0);
                1 + case_max.max(default_max)
            }
            DecisionTree::TypeCheck {
                success, failure, ..
            }
            | DecisionTree::SequenceCheck {
                success, failure, ..
            }
            | DecisionTree::MappingCheck {
                success, failure, ..
            } => 1 + success.max_depth().max(failure.max_depth()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::match_compiler::matrix::PatternRow;
    use crate::match_compiler::pattern::{FlatPattern, LiteralValue};

    fn literal_row(val: i64, action: usize) -> PatternRow {
        PatternRow {
            patterns: vec![FlatPattern::Literal(LiteralValue::Int(val))],
            bindings: vec![],
            guard: None,
            action,
        }
    }

    fn wildcard_row(action: usize) -> PatternRow {
        PatternRow {
            patterns: vec![FlatPattern::Wildcard],
            bindings: vec![],
            guard: None,
            action,
        }
    }

    #[test]
    fn test_compile_empty() {
        let matrix = PatternMatrix::new(vec![]);
        let tree = compile(&matrix);
        assert!(matches!(tree, DecisionTree::Fail));
    }

    #[test]
    fn test_compile_single_wildcard() {
        let matrix = PatternMatrix::new(vec![wildcard_row(0)]);
        let tree = compile(&matrix);

        if let DecisionTree::Leaf { action, guard, .. } = tree {
            assert_eq!(action, 0);
            assert!(guard.is_none());
        } else {
            panic!("Expected Leaf node");
        }
    }

    #[test]
    fn test_compile_two_literals() {
        let matrix = PatternMatrix::new(vec![literal_row(1, 0), literal_row(2, 1)]);
        let tree = compile(&matrix);

        if let DecisionTree::Switch { cases, default, .. } = tree {
            assert_eq!(cases.len(), 2);
            assert!(default.is_none());
        } else {
            panic!("Expected Switch node");
        }
    }

    #[test]
    fn test_compile_literal_with_default() {
        let matrix = PatternMatrix::new(vec![literal_row(1, 0), wildcard_row(1)]);
        let tree = compile(&matrix);

        if let DecisionTree::Switch { cases, default, .. } = tree {
            assert_eq!(cases.len(), 1);
            assert!(default.is_some());
        } else {
            panic!("Expected Switch node");
        }
    }

    #[test]
    fn test_tree_node_count() {
        let matrix =
            PatternMatrix::new(vec![literal_row(1, 0), literal_row(2, 1), wildcard_row(2)]);
        let tree = compile(&matrix);

        // Should have: 1 switch + 2 leaf cases + 1 default leaf
        assert!(tree.node_count() >= 4);
    }

    #[test]
    fn test_tree_max_depth() {
        let matrix = PatternMatrix::new(vec![wildcard_row(0)]);
        let tree = compile(&matrix);

        // Single leaf = depth 1
        assert_eq!(tree.max_depth(), 1);
    }
}

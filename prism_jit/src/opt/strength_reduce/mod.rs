//! Strength Reduction Optimization
//!
//! This module implements strength reduction, which replaces expensive operations
//! with equivalent but cheaper sequences. This is one of the most effective
//! peephole optimizations for arithmetic-heavy code.
//!
//! # Transformations
//!
//! ## Multiplication
//!
//! - **Power of 2**: `x * 2^n` → `x << n`
//! - **Small constants**: `x * 3` → `x + (x << 1)`, `x * 5` → `x + (x << 2)`
//! - **Near power of 2**: `x * 7` → `(x << 3) - x`, `x * 15` → `(x << 4) - x`
//!
//! ## Division
//!
//! - **Power of 2 (unsigned)**: `x / 2^n` → `x >> n`
//! - **Power of 2 (signed)**: `x / 2^n` → `(x + (x >> 63 >> (64-n))) >> n`
//! - **Constants**: Uses magic number multiplication (Hacker's Delight)
//!
//! ## Modulo
//!
//! - **Power of 2 (unsigned)**: `x % 2^n` → `x & (2^n - 1)`
//! - **Power of 2 (signed)**: More complex, uses conditional adjustment
//!
//! ## Bitwise
//!
//! - **Identity**: `x & -1` → `x`, `x | 0` → `x`
//! - **Absorbing**: `x & 0` → `0`, `x | -1` → `-1`
//! - **Idempotent**: `x & x` → `x`, `x | x` → `x`
//! - **Complement**: `x ^ x` → `0`
//!
//! # Organization
//!
//! - `magic.rs`: Magic number computation for division by constants
//! - `multiply.rs`: Multiplication decomposition into shifts/adds
//! - `mod.rs`: Main optimization pass and integration

pub mod magic;
pub mod multiply;

use crate::ir::graph::Graph;
use crate::ir::node::{InputList, NodeId};
use crate::ir::operators::{ArithOp, BitwiseOp, Operator};
use crate::ir::types::ValueType;
use crate::opt::OptimizationPass;

pub use magic::{SignedMagic, SignedMagic32, UnsignedMagic, UnsignedMagic32};
pub use multiply::{DecompConfig, DecompOp, MulDecomposition};

// =============================================================================
// Strength Reduction Pass
// =============================================================================

/// Strength reduction optimization pass.
///
/// This pass replaces expensive arithmetic operations with cheaper equivalents.
#[derive(Debug)]
pub struct StrengthReduce {
    /// Configuration for this pass.
    config: StrengthReduceConfig,
    /// Statistics from the last run.
    stats: StrengthReduceStats,
}

/// Configuration for strength reduction.
#[derive(Clone, Debug)]
pub struct StrengthReduceConfig {
    /// Enable multiplication decomposition.
    pub decompose_multiply: bool,
    /// Enable division by constant optimization.
    pub optimize_division: bool,
    /// Enable modulo by constant optimization.
    pub optimize_modulo: bool,
    /// Enable bitwise identity optimizations.
    pub optimize_bitwise: bool,
    /// Configuration for multiply decomposition.
    pub multiply_config: DecompConfig,
    /// Minimum divisor magnitude for magic number division.
    pub min_magic_divisor: i64,
}

impl Default for StrengthReduceConfig {
    fn default() -> Self {
        Self {
            decompose_multiply: true,
            optimize_division: true,
            optimize_modulo: true,
            optimize_bitwise: true,
            multiply_config: DecompConfig::default(),
            min_magic_divisor: 2,
        }
    }
}

impl StrengthReduceConfig {
    /// Conservative configuration - only safe transformations.
    pub fn conservative() -> Self {
        Self {
            decompose_multiply: true,
            optimize_division: true,
            optimize_modulo: true,
            optimize_bitwise: true,
            multiply_config: DecompConfig::low_latency(),
            min_magic_divisor: 3,
        }
    }

    /// Aggressive configuration - more transformations.
    pub fn aggressive() -> Self {
        Self {
            decompose_multiply: true,
            optimize_division: true,
            optimize_modulo: true,
            optimize_bitwise: true,
            multiply_config: DecompConfig::aggressive(),
            min_magic_divisor: 2,
        }
    }
}

/// Statistics from strength reduction.
#[derive(Clone, Debug, Default)]
pub struct StrengthReduceStats {
    /// Number of multiply-by-power-of-2 replaced with shift.
    pub mul_to_shift: usize,
    /// Number of multiply decompositions.
    pub mul_decomposed: usize,
    /// Number of unsigned divisions replaced.
    pub div_unsigned_replaced: usize,
    /// Number of signed divisions replaced.
    pub div_signed_replaced: usize,
    /// Number of modulo operations replaced.
    pub mod_replaced: usize,
    /// Number of bitwise simplifications.
    pub bitwise_simplified: usize,
    /// Total transformations.
    pub total: usize,
}

impl StrengthReduceStats {
    /// Reset all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

impl StrengthReduce {
    /// Create a new strength reduction pass with default configuration.
    pub fn new() -> Self {
        Self {
            config: StrengthReduceConfig::default(),
            stats: StrengthReduceStats::default(),
        }
    }

    /// Create with specific configuration.
    pub fn with_config(config: StrengthReduceConfig) -> Self {
        Self {
            config,
            stats: StrengthReduceStats::default(),
        }
    }

    /// Get statistics from the last run.
    pub fn stats(&self) -> &StrengthReduceStats {
        &self.stats
    }

    /// Run strength reduction on a graph.
    fn run_impl(&mut self, graph: &mut Graph) -> bool {
        self.stats.reset();

        // Collect transformations
        let mut transforms: Vec<Transform> = Vec::new();

        for (id, node) in graph.iter() {
            if let Some(transform) = self.analyze_node(graph, id, &node.op) {
                transforms.push(transform);
            }
        }

        // Apply transformations
        for transform in transforms {
            self.apply_transform(graph, transform);
        }

        self.stats.total > 0
    }

    /// Analyze a node for possible strength reduction.
    fn analyze_node(&self, graph: &Graph, id: NodeId, op: &Operator) -> Option<Transform> {
        match op {
            Operator::IntOp(arith_op) => self.analyze_int_op(graph, id, *arith_op),
            Operator::Bitwise(bitwise_op) => self.analyze_bitwise(graph, id, *bitwise_op),
            _ => None,
        }
    }

    /// Analyze integer arithmetic operations.
    fn analyze_int_op(&self, graph: &Graph, id: NodeId, op: ArithOp) -> Option<Transform> {
        let node = graph.node(id);
        let inputs: Vec<NodeId> = node.inputs.iter().collect();

        if inputs.len() != 2 {
            return None;
        }

        let lhs_id = inputs[0];
        let rhs_id = inputs[1];
        let rhs_node = graph.node(rhs_id);

        // Get the constant on the right (if any)
        let const_val = match rhs_node.op {
            Operator::ConstInt(v) => Some(v),
            _ => None,
        };

        match op {
            ArithOp::Mul if self.config.decompose_multiply => {
                self.analyze_multiply(id, lhs_id, const_val?)
            }
            ArithOp::FloorDiv if self.config.optimize_division => {
                self.analyze_division(id, lhs_id, const_val?, true)
            }
            ArithOp::TrueDiv if self.config.optimize_division => {
                // For integer true division, treat as floor division
                self.analyze_division(id, lhs_id, const_val?, true)
            }
            ArithOp::Mod if self.config.optimize_modulo => {
                self.analyze_modulo(id, lhs_id, const_val?)
            }
            _ => None,
        }
    }

    /// Analyze multiplication by a constant.
    fn analyze_multiply(
        &self,
        node_id: NodeId,
        operand: NodeId,
        multiplier: i64,
    ) -> Option<Transform> {
        // Power of 2: x * 2^n → x << n
        if multiplier > 0 && (multiplier as u64).is_power_of_two() {
            let shift = (multiplier as u64).trailing_zeros() as u8;
            return Some(Transform::MulToShift {
                node: node_id,
                operand,
                shift,
            });
        }

        // Negative power of 2: x * -2^n → -(x << n)
        if multiplier < 0 && ((-multiplier) as u64).is_power_of_two() {
            let shift = ((-multiplier) as u64).trailing_zeros() as u8;
            return Some(Transform::MulToNegShift {
                node: node_id,
                operand,
                shift,
            });
        }

        // Try decomposition
        if let Some(decomp) = multiply::decompose_multiply(multiplier, &self.config.multiply_config)
        {
            if decomp.is_profitable(self.config.multiply_config.multiply_cost) {
                return Some(Transform::MulDecompose {
                    node: node_id,
                    operand,
                    decomp,
                });
            }
        }

        None
    }

    /// Analyze division by a constant.
    fn analyze_division(
        &self,
        node_id: NodeId,
        dividend: NodeId,
        divisor: i64,
        signed: bool,
    ) -> Option<Transform> {
        if divisor.abs() < self.config.min_magic_divisor {
            return None;
        }

        // Power of 2 division
        if divisor > 0 && (divisor as u64).is_power_of_two() {
            let shift = (divisor as u64).trailing_zeros() as u8;
            if signed {
                return Some(Transform::SignedDivPow2 {
                    node: node_id,
                    dividend,
                    shift,
                });
            } else {
                return Some(Transform::UnsignedDivPow2 {
                    node: node_id,
                    dividend,
                    shift,
                });
            }
        }

        // Magic number division
        if signed {
            if let Some(magic) = SignedMagic::compute(divisor) {
                return Some(Transform::SignedDivMagic {
                    node: node_id,
                    dividend,
                    divisor,
                    magic,
                });
            }
        } else {
            if let Some(magic) = UnsignedMagic::compute(divisor as u64) {
                return Some(Transform::UnsignedDivMagic {
                    node: node_id,
                    dividend,
                    divisor: divisor as u64,
                    magic,
                });
            }
        }

        None
    }

    /// Analyze modulo by a constant.
    fn analyze_modulo(&self, node_id: NodeId, dividend: NodeId, divisor: i64) -> Option<Transform> {
        // Only handle positive power-of-2 divisors for now
        if divisor > 0 && (divisor as u64).is_power_of_two() {
            let mask = divisor as u64 - 1;
            return Some(Transform::ModPow2 {
                node: node_id,
                dividend,
                mask,
            });
        }

        None
    }

    /// Analyze bitwise operations.
    fn analyze_bitwise(&self, graph: &Graph, id: NodeId, op: BitwiseOp) -> Option<Transform> {
        if !self.config.optimize_bitwise {
            return None;
        }

        let node = graph.node(id);
        let inputs: Vec<NodeId> = node.inputs.iter().collect();

        if inputs.len() != 2 {
            return None;
        }

        let lhs_id = inputs[0];
        let rhs_id = inputs[1];
        let lhs = graph.node(lhs_id);
        let rhs = graph.node(rhs_id);

        // Get constants if any
        let lhs_const = if let Operator::ConstInt(v) = lhs.op {
            Some(v)
        } else {
            None
        };
        let rhs_const = if let Operator::ConstInt(v) = rhs.op {
            Some(v)
        } else {
            None
        };

        match op {
            BitwiseOp::And => {
                // x & 0 → 0
                if rhs_const == Some(0) || lhs_const == Some(0) {
                    return Some(Transform::ReplaceWithConstant { node: id, value: 0 });
                }
                // x & -1 → x
                if rhs_const == Some(-1) {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: lhs_id,
                    });
                }
                if lhs_const == Some(-1) {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: rhs_id,
                    });
                }
                // x & x → x
                if lhs_id == rhs_id {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: lhs_id,
                    });
                }
            }
            BitwiseOp::Or => {
                // x | -1 → -1
                if rhs_const == Some(-1) || lhs_const == Some(-1) {
                    return Some(Transform::ReplaceWithConstant {
                        node: id,
                        value: -1,
                    });
                }
                // x | 0 → x
                if rhs_const == Some(0) {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: lhs_id,
                    });
                }
                if lhs_const == Some(0) {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: rhs_id,
                    });
                }
                // x | x → x
                if lhs_id == rhs_id {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: lhs_id,
                    });
                }
            }
            BitwiseOp::Xor => {
                // x ^ 0 → x
                if rhs_const == Some(0) {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: lhs_id,
                    });
                }
                if lhs_const == Some(0) {
                    return Some(Transform::ReplaceWithNode {
                        node: id,
                        replacement: rhs_id,
                    });
                }
                // x ^ x → 0
                if lhs_id == rhs_id {
                    return Some(Transform::ReplaceWithConstant { node: id, value: 0 });
                }
            }
            _ => {}
        }

        None
    }

    /// Apply a transformation to the graph.
    fn apply_transform(&mut self, graph: &mut Graph, transform: Transform) {
        match transform {
            Transform::MulToShift {
                node,
                operand,
                shift,
            } => {
                self.apply_mul_to_shift(graph, node, operand, shift);
                self.stats.mul_to_shift += 1;
            }
            Transform::MulToNegShift {
                node,
                operand,
                shift,
            } => {
                self.apply_mul_to_neg_shift(graph, node, operand, shift);
                self.stats.mul_to_shift += 1;
            }
            Transform::MulDecompose {
                node,
                operand,
                decomp,
            } => {
                self.apply_mul_decompose(graph, node, operand, &decomp);
                self.stats.mul_decomposed += 1;
            }
            Transform::UnsignedDivPow2 {
                node,
                dividend,
                shift,
            } => {
                self.apply_unsigned_div_pow2(graph, node, dividend, shift);
                self.stats.div_unsigned_replaced += 1;
            }
            Transform::SignedDivPow2 {
                node,
                dividend,
                shift,
            } => {
                self.apply_signed_div_pow2(graph, node, dividend, shift);
                self.stats.div_signed_replaced += 1;
            }
            Transform::UnsignedDivMagic {
                node,
                dividend,
                magic,
                ..
            } => {
                self.apply_unsigned_div_magic(graph, node, dividend, &magic);
                self.stats.div_unsigned_replaced += 1;
            }
            Transform::SignedDivMagic {
                node,
                dividend,
                divisor,
                magic,
            } => {
                self.apply_signed_div_magic(graph, node, dividend, divisor, &magic);
                self.stats.div_signed_replaced += 1;
            }
            Transform::ModPow2 {
                node,
                dividend,
                mask,
            } => {
                self.apply_mod_pow2(graph, node, dividend, mask);
                self.stats.mod_replaced += 1;
            }
            Transform::ReplaceWithConstant { node, value } => {
                let const_node = graph.const_int(value);
                graph.replace_all_uses(node, const_node);
                self.stats.bitwise_simplified += 1;
            }
            Transform::ReplaceWithNode { node, replacement } => {
                graph.replace_all_uses(node, replacement);
                self.stats.bitwise_simplified += 1;
            }
        }
        self.stats.total += 1;
    }

    // =========================================================================
    // Transform Implementations
    // =========================================================================

    fn apply_mul_to_shift(&self, graph: &mut Graph, node: NodeId, operand: NodeId, shift: u8) {
        // x * 2^n → x << n
        let shift_const = graph.const_int(shift as i64);
        let shift_node = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::Shl),
            InputList::Pair(operand, shift_const),
            ValueType::Int64,
        );
        graph.replace_all_uses(node, shift_node);
    }

    fn apply_mul_to_neg_shift(&self, graph: &mut Graph, node: NodeId, operand: NodeId, shift: u8) {
        // x * -2^n → -(x << n)
        let shift_const = graph.const_int(shift as i64);
        let shift_node = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::Shl),
            InputList::Pair(operand, shift_const),
            ValueType::Int64,
        );
        let neg_node = graph.add_node_with_type(
            Operator::IntOp(ArithOp::Neg),
            InputList::Single(shift_node),
            ValueType::Int64,
        );
        graph.replace_all_uses(node, neg_node);
    }

    fn apply_mul_decompose(
        &self,
        graph: &mut Graph,
        node: NodeId,
        operand: NodeId,
        decomp: &MulDecomposition,
    ) {
        // Build the decomposed sequence
        let mut result = operand;

        for op in &decomp.ops {
            result = match *op {
                DecompOp::Shift(n) => {
                    let shift_const = graph.const_int(n as i64);
                    graph.add_node_with_type(
                        Operator::Bitwise(BitwiseOp::Shl),
                        InputList::Pair(operand, shift_const),
                        ValueType::Int64,
                    )
                }
                DecompOp::AddShift(n) => {
                    let shift_const = graph.const_int(n as i64);
                    let shifted = graph.add_node_with_type(
                        Operator::Bitwise(BitwiseOp::Shl),
                        InputList::Pair(operand, shift_const),
                        ValueType::Int64,
                    );
                    graph.add_node_with_type(
                        Operator::IntOp(ArithOp::Add),
                        InputList::Pair(result, shifted),
                        ValueType::Int64,
                    )
                }
                DecompOp::SubShift(n) => {
                    let shift_const = graph.const_int(n as i64);
                    let shifted = graph.add_node_with_type(
                        Operator::Bitwise(BitwiseOp::Shl),
                        InputList::Pair(operand, shift_const),
                        ValueType::Int64,
                    );
                    graph.add_node_with_type(
                        Operator::IntOp(ArithOp::Sub),
                        InputList::Pair(result, shifted),
                        ValueType::Int64,
                    )
                }
                DecompOp::Add => graph.add_node_with_type(
                    Operator::IntOp(ArithOp::Add),
                    InputList::Pair(result, operand),
                    ValueType::Int64,
                ),
                DecompOp::Sub => graph.add_node_with_type(
                    Operator::IntOp(ArithOp::Sub),
                    InputList::Pair(result, operand),
                    ValueType::Int64,
                ),
                DecompOp::Negate => graph.add_node_with_type(
                    Operator::IntOp(ArithOp::Neg),
                    InputList::Single(result),
                    ValueType::Int64,
                ),
            };
        }

        if decomp.negate_result {
            result = graph.add_node_with_type(
                Operator::IntOp(ArithOp::Neg),
                InputList::Single(result),
                ValueType::Int64,
            );
        }

        graph.replace_all_uses(node, result);
    }

    fn apply_unsigned_div_pow2(
        &self,
        graph: &mut Graph,
        node: NodeId,
        dividend: NodeId,
        shift: u8,
    ) {
        // x / 2^n → x >> n (logical shift for unsigned)
        let shift_const = graph.const_int(shift as i64);
        let shift_node = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::Shr),
            InputList::Pair(dividend, shift_const),
            ValueType::Int64,
        );
        graph.replace_all_uses(node, shift_node);
    }

    fn apply_signed_div_pow2(&self, graph: &mut Graph, node: NodeId, dividend: NodeId, shift: u8) {
        // For signed division by 2^n, we need to add a bias for negative numbers
        // to get truncation toward zero.
        //
        // x / 2^n → (x + ((x >> 63) >>> (64 - n))) >> n
        //
        // Where >> is arithmetic shift and >>> is logical shift.

        if shift == 0 {
            // x / 1 = x
            graph.replace_all_uses(node, dividend);
            return;
        }

        let c63 = graph.const_int(63);
        let c_shift_adj = graph.const_int((64 - shift) as i64);
        let c_shift = graph.const_int(shift as i64);

        // sign = x >> 63 (all 1s if negative, all 0s if positive)
        // Note: Using Shr for arithmetic right shift (signed values)
        let sign = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::Shr),
            InputList::Pair(dividend, c63),
            ValueType::Int64,
        );

        // bias = sign >>> (64 - n) (gives 2^n - 1 if negative, 0 if positive)
        let bias = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::Shr),
            InputList::Pair(sign, c_shift_adj),
            ValueType::Int64,
        );

        // adjusted = x + bias
        let adjusted = graph.add_node_with_type(
            Operator::IntOp(ArithOp::Add),
            InputList::Pair(dividend, bias),
            ValueType::Int64,
        );

        // result = adjusted >> n
        let result = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::Shr),
            InputList::Pair(adjusted, c_shift),
            ValueType::Int64,
        );

        graph.replace_all_uses(node, result);
    }

    fn apply_unsigned_div_magic(
        &self,
        graph: &mut Graph,
        node: NodeId,
        dividend: NodeId,
        magic: &UnsignedMagic,
    ) {
        if magic.is_power_of_two() {
            self.apply_unsigned_div_pow2(graph, node, dividend, magic.shift);
            return;
        }

        // Unsigned division by magic multiplication:
        // q = mulhu(x, m) >> s  (without add)
        // or
        // q = ((x - mulhu(x, m)) >> 1 + mulhu(x, m)) >> (s-1)  (with add)

        let m = graph.const_int(magic.multiplier as i64);

        // mulhu (multiply high unsigned) - placeholder, using regular mul for now
        // In practice, this would be a special mulhu node
        let high = graph.add_node_with_type(
            Operator::MulHigh,
            InputList::Pair(dividend, m),
            ValueType::Int64,
        );

        let result = if magic.add {
            // Need add-shift sequence
            let diff = graph.add_node_with_type(
                Operator::IntOp(ArithOp::Sub),
                InputList::Pair(dividend, high),
                ValueType::Int64,
            );

            let c1 = graph.const_int(1);
            let half = graph.add_node_with_type(
                Operator::Bitwise(BitwiseOp::Shr),
                InputList::Pair(diff, c1),
                ValueType::Int64,
            );

            let added = graph.add_node_with_type(
                Operator::IntOp(ArithOp::Add),
                InputList::Pair(half, high),
                ValueType::Int64,
            );

            let shift_const = graph.const_int(magic.shift as i64);
            graph.add_node_with_type(
                Operator::Bitwise(BitwiseOp::Shr),
                InputList::Pair(added, shift_const),
                ValueType::Int64,
            )
        } else {
            let shift_const = graph.const_int(magic.shift as i64);
            graph.add_node_with_type(
                Operator::Bitwise(BitwiseOp::Shr),
                InputList::Pair(high, shift_const),
                ValueType::Int64,
            )
        };

        graph.replace_all_uses(node, result);
    }

    fn apply_signed_div_magic(
        &self,
        graph: &mut Graph,
        node: NodeId,
        dividend: NodeId,
        divisor: i64,
        magic: &SignedMagic,
    ) {
        if magic.is_power_of_two() {
            self.apply_signed_div_pow2(graph, node, dividend, magic.shift);
            return;
        }

        // Signed division by magic multiplication:
        // q = mulhs(x, m)
        // if m < 0: q = q + x
        // q = q >> s
        // q = q - (x >> 63)  // sign correction

        let m = graph.const_int(magic.multiplier);

        // mulhs (multiply high signed)
        let mut q = graph.add_node_with_type(
            Operator::MulHighSigned,
            InputList::Pair(dividend, m),
            ValueType::Int64,
        );

        // Add dividend if multiplier is negative
        if magic.multiplier < 0 {
            q = graph.add_node_with_type(
                Operator::IntOp(ArithOp::Add),
                InputList::Pair(q, dividend),
                ValueType::Int64,
            );
        }

        // Arithmetic shift right
        if magic.shift > 0 {
            let shift_const = graph.const_int(magic.shift as i64);
            q = graph.add_node_with_type(
                Operator::Bitwise(BitwiseOp::Shr),
                InputList::Pair(q, shift_const),
                ValueType::Int64,
            );
        }

        // Sign correction: subtract sign bit of result
        let c63 = graph.const_int(63);
        let sign = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::Shr),
            InputList::Pair(q, c63),
            ValueType::Int64,
        );

        let result = graph.add_node_with_type(
            Operator::IntOp(ArithOp::Sub),
            InputList::Pair(q, sign),
            ValueType::Int64,
        );

        // Negate if divisor was negative
        let final_result = if divisor < 0 {
            graph.add_node_with_type(
                Operator::IntOp(ArithOp::Neg),
                InputList::Single(result),
                ValueType::Int64,
            )
        } else {
            result
        };

        graph.replace_all_uses(node, final_result);
    }

    fn apply_mod_pow2(&self, graph: &mut Graph, node: NodeId, dividend: NodeId, mask: u64) {
        // x % 2^n → x & (2^n - 1)
        let mask_const = graph.const_int(mask as i64);
        let and_node = graph.add_node_with_type(
            Operator::Bitwise(BitwiseOp::And),
            InputList::Pair(dividend, mask_const),
            ValueType::Int64,
        );
        graph.replace_all_uses(node, and_node);
    }
}

impl Default for StrengthReduce {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for StrengthReduce {
    fn name(&self) -> &'static str {
        "strength_reduce"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_impl(graph)
    }
}

// =============================================================================
// Internal Transform Types
// =============================================================================

/// A transformation to apply to the graph.
#[derive(Debug)]
enum Transform {
    /// Replace multiply with shift: x * 2^n → x << n
    MulToShift {
        node: NodeId,
        operand: NodeId,
        shift: u8,
    },
    /// Replace multiply with negated shift: x * -2^n → -(x << n)
    MulToNegShift {
        node: NodeId,
        operand: NodeId,
        shift: u8,
    },
    /// Decompose multiply into shifts and adds
    MulDecompose {
        node: NodeId,
        operand: NodeId,
        decomp: MulDecomposition,
    },
    /// Unsigned division by power of 2
    UnsignedDivPow2 {
        node: NodeId,
        dividend: NodeId,
        shift: u8,
    },
    /// Signed division by power of 2
    SignedDivPow2 {
        node: NodeId,
        dividend: NodeId,
        shift: u8,
    },
    /// Unsigned division by magic number
    UnsignedDivMagic {
        node: NodeId,
        dividend: NodeId,
        divisor: u64,
        magic: UnsignedMagic,
    },
    /// Signed division by magic number
    SignedDivMagic {
        node: NodeId,
        dividend: NodeId,
        divisor: i64,
        magic: SignedMagic,
    },
    /// Modulo by power of 2
    ModPow2 {
        node: NodeId,
        dividend: NodeId,
        mask: u64,
    },
    /// Replace with constant
    ReplaceWithConstant { node: NodeId, value: i64 },
    /// Replace with another node
    ReplaceWithNode { node: NodeId, replacement: NodeId },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ArithmeticBuilder, BitwiseBuilder, ControlBuilder, GraphBuilder};

    // =========================================================================
    // Basic Tests
    // =========================================================================

    #[test]
    fn test_strength_reduce_new() {
        let sr = StrengthReduce::new();
        assert_eq!(sr.stats().total, 0);
    }

    #[test]
    fn test_strength_reduce_default() {
        let sr = StrengthReduce::default();
        assert!(sr.config.decompose_multiply);
        assert!(sr.config.optimize_division);
    }

    #[test]
    fn test_config_conservative() {
        let config = StrengthReduceConfig::conservative();
        assert_eq!(config.multiply_config.max_ops, 2);
    }

    #[test]
    fn test_config_aggressive() {
        let config = StrengthReduceConfig::aggressive();
        assert!(config.multiply_config.max_ops >= 6);
    }

    // =========================================================================
    // Multiplication Tests
    // =========================================================================

    #[test]
    fn test_mul_by_power_of_two() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c4 = builder.const_int(4);
        let mul = builder.int_mul(p0, c4);
        builder.return_value(mul);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().mul_to_shift, 1);
    }

    #[test]
    fn test_mul_by_negative_power_of_two() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c_neg4 = builder.const_int(-4);
        let mul = builder.int_mul(p0, c_neg4);
        builder.return_value(mul);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().mul_to_shift, 1);
    }

    #[test]
    fn test_mul_by_three() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c3 = builder.const_int(3);
        let mul = builder.int_mul(p0, c3);
        builder.return_value(mul);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().mul_decomposed, 1);
    }

    // =========================================================================
    // Division Tests
    // =========================================================================

    #[test]
    fn test_div_by_power_of_two() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c8 = builder.const_int(8);
        let div = builder.int_div(p0, c8);
        builder.return_value(div);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert!(sr.stats().div_signed_replaced > 0 || sr.stats().div_unsigned_replaced > 0);
    }

    // =========================================================================
    // Modulo Tests
    // =========================================================================

    #[test]
    fn test_mod_by_power_of_two() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c16 = builder.const_int(16);
        let modulo = builder.int_mod(p0, c16);
        builder.return_value(modulo);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().mod_replaced, 1);
    }

    // =========================================================================
    // Bitwise Tests
    // =========================================================================

    #[test]
    fn test_and_with_zero() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c0 = builder.const_int(0);
        let and = builder.bitwise_and(p0, c0);
        builder.return_value(and);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().bitwise_simplified, 1);
    }

    #[test]
    fn test_and_with_neg_one() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c_neg1 = builder.const_int(-1);
        let and = builder.bitwise_and(p0, c_neg1);
        builder.return_value(and);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().bitwise_simplified, 1);
    }

    #[test]
    fn test_or_with_zero() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c0 = builder.const_int(0);
        let or = builder.bitwise_or(p0, c0);
        builder.return_value(or);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().bitwise_simplified, 1);
    }

    #[test]
    fn test_or_with_neg_one() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let c_neg1 = builder.const_int(-1);
        let or = builder.bitwise_or(p0, c_neg1);
        builder.return_value(or);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().bitwise_simplified, 1);
    }

    #[test]
    fn test_xor_with_self() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let xor = builder.bitwise_xor(p0, p0);
        builder.return_value(xor);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().bitwise_simplified, 1);
    }

    #[test]
    fn test_and_with_self() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let and = builder.bitwise_and(p0, p0);
        builder.return_value(and);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(changed);
        assert_eq!(sr.stats().bitwise_simplified, 1);
    }

    // =========================================================================
    // Stats Tests
    // =========================================================================

    #[test]
    fn test_stats_reset() {
        let mut stats = StrengthReduceStats::default();
        stats.total = 10;
        stats.mul_to_shift = 5;

        stats.reset();

        assert_eq!(stats.total, 0);
        assert_eq!(stats.mul_to_shift, 0);
    }

    // =========================================================================
    // No-op Tests
    // =========================================================================

    #[test]
    fn test_no_transform_needed() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let mul = builder.int_mul(p0, p1); // Variable * variable, can't reduce
        builder.return_value(mul);

        let mut graph = builder.finish();
        let mut sr = StrengthReduce::new();

        let changed = sr.run(&mut graph);

        assert!(!changed);
        assert_eq!(sr.stats().total, 0);
    }
}

//! Liveness Analysis
//!
//! Computes live ranges for virtual registers using dataflow analysis.
//!
//! # Algorithm
//!
//! Uses reverse dataflow analysis to compute liveness:
//!
//! 1. For each basic block, compute local gen/kill sets
//! 2. Iterate until fixed point: LiveOut = union(LiveIn of successors)
//! 3. LiveIn = gen | (LiveOut - kill)
//! 4. Build live intervals from the liveness information
//!
//! # Complexity
//!
//! O(n * k) where n = number of instructions, k = average iterations to converge

use super::interval::{LiveInterval, LiveRange, ProgPoint, UsePosition};
use super::{RegClass, VReg};
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::Operator;
use crate::ir::types::ValueType;
use std::collections::HashMap;

// =============================================================================
// Liveness Analysis
// =============================================================================

/// Liveness analysis result.
#[derive(Debug)]
pub struct LivenessAnalysis {
    /// Live intervals for each virtual register.
    intervals: HashMap<VReg, LiveInterval>,
    /// Mapping from NodeId to VReg.
    node_to_vreg: HashMap<NodeId, VReg>,
    /// Next available vreg index.
    next_vreg: u32,
    /// Total number of instructions.
    num_instructions: u32,
}

impl LivenessAnalysis {
    /// Create a new liveness analysis.
    pub fn new() -> Self {
        LivenessAnalysis {
            intervals: HashMap::new(),
            node_to_vreg: HashMap::new(),
            next_vreg: 0,
            num_instructions: 0,
        }
    }

    /// Run liveness analysis on a graph.
    pub fn analyze(graph: &Graph) -> Self {
        let mut analysis = LivenessAnalysis::new();
        analysis.compute_liveness(graph);
        analysis
    }

    /// Get or create a vreg for a node.
    fn get_or_create_vreg(&mut self, node: NodeId, reg_class: RegClass) -> VReg {
        if let Some(&vreg) = self.node_to_vreg.get(&node) {
            vreg
        } else {
            let vreg = VReg::new(self.next_vreg);
            self.next_vreg += 1;
            self.node_to_vreg.insert(node, vreg);
            self.intervals
                .insert(vreg, LiveInterval::new(vreg, reg_class));
            vreg
        }
    }

    /// Compute liveness for all nodes in the graph.
    fn compute_liveness(&mut self, graph: &Graph) {
        // Phase 1: Assign instruction indices and create vregs
        let mut inst_idx = 0u32;
        let mut node_to_inst: HashMap<NodeId, u32> = HashMap::new();

        for (id, node) in graph.iter() {
            if node.is_dead() {
                continue;
            }

            node_to_inst.insert(id, inst_idx);

            // Create vreg for this node if it produces a value
            if Self::produces_value(&node.op) {
                let reg_class = Self::reg_class_for_type(node.ty);
                self.get_or_create_vreg(id, reg_class);
            }

            inst_idx += 1;
        }

        self.num_instructions = inst_idx;

        // Phase 2: Build live ranges from definitions and uses
        for (id, node) in graph.iter() {
            if node.is_dead() {
                continue;
            }

            let Some(&def_idx) = node_to_inst.get(&id) else {
                continue;
            };

            // Get the vreg for this node (if it produces a value)
            if let Some(&vreg) = self.node_to_vreg.get(&id) {
                let interval = self.intervals.get_mut(&vreg).unwrap();

                // Add definition
                let def_pos = ProgPoint::after(def_idx);
                interval.add_use(UsePosition::def(def_pos));

                // Compute spill weight based on loop depth
                // (placeholder - would need CFG info for accurate weight)
                interval.spill_weight = 1.0;
            }

            // Process inputs - extend live ranges to uses
            for input in node.inputs.iter() {
                if !input.is_valid() {
                    continue;
                }

                if let Some(&input_vreg) = self.node_to_vreg.get(&input) {
                    let Some(&input_def_idx) = node_to_inst.get(&input) else {
                        continue;
                    };

                    let interval = self.intervals.get_mut(&input_vreg).unwrap();

                    // The value is live from just after its definition
                    // to just before the use
                    let use_pos = ProgPoint::before(def_idx);
                    let def_pos = ProgPoint::after(input_def_idx);

                    if def_pos <= use_pos {
                        interval.add_range(LiveRange::new(def_pos, use_pos.next()));
                        interval.add_use(UsePosition::use_pos(use_pos));
                    }
                }
            }
        }

        // Phase 3: Handle phi nodes - extend ranges to block boundaries
        // (Simplified - a full implementation would use the CFG)
        for (id, node) in graph.iter() {
            if node.is_dead() {
                continue;
            }

            if matches!(node.op, Operator::Phi | Operator::LoopPhi) {
                let Some(&phi_idx) = node_to_inst.get(&id) else {
                    continue;
                };

                // Phi inputs need to be live at the end of their predecessor blocks
                for input in node.inputs.iter().skip(1) {
                    // Skip control input
                    if !input.is_valid() {
                        continue;
                    }

                    if let Some(&input_vreg) = self.node_to_vreg.get(&input) {
                        let interval = self.intervals.get_mut(&input_vreg).unwrap();

                        // Extend to the phi's position
                        let phi_pos = ProgPoint::before(phi_idx);
                        if let Some(first) = interval.first_range() {
                            if first.start < phi_pos {
                                interval.add_range(LiveRange::new(first.start, phi_pos.next()));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check if an operator produces a value that needs a register.
    fn produces_value(op: &Operator) -> bool {
        match op {
            // Control flow doesn't produce allocatable values
            Operator::Control(_) => false,

            // All other operators produce values
            _ => true,
        }
    }

    /// Get the register class for a value type.
    ///
    /// Maps value types to register classes for allocation:
    /// - 512-bit vectors → Vec512 (ZMM registers)
    /// - 256-bit vectors → Vec256 (YMM registers)
    /// - 128-bit vectors/float64 → Float (XMM registers)
    /// - All other types → Int (GPR registers)
    fn reg_class_for_type(ty: ValueType) -> RegClass {
        // Check bit width for vector types
        match ty.bit_width() {
            512 => RegClass::Vec512,                  // ZMM for 512-bit vectors
            256 => RegClass::Vec256,                  // YMM for 256-bit vectors
            128 if ty.is_vector() => RegClass::Float, // XMM for 128-bit vectors
            _ if matches!(ty, ValueType::Float64) => RegClass::Float, // XMM for scalars
            _ => RegClass::Int,                       // GPR for integers and other types
        }
    }

    /// Get the interval for a vreg.
    pub fn interval(&self, vreg: VReg) -> Option<&LiveInterval> {
        self.intervals.get(&vreg)
    }

    /// Get mutable interval for a vreg.
    pub fn interval_mut(&mut self, vreg: VReg) -> Option<&mut LiveInterval> {
        self.intervals.get_mut(&vreg)
    }

    /// Get the vreg for a node.
    pub fn vreg_for_node(&self, node: NodeId) -> Option<VReg> {
        self.node_to_vreg.get(&node).copied()
    }

    /// Iterate over all intervals.
    pub fn intervals(&self) -> impl Iterator<Item = &LiveInterval> {
        self.intervals.values()
    }

    /// Get all intervals as a mutable reference.
    pub fn intervals_mut(&mut self) -> impl Iterator<Item = &mut LiveInterval> {
        self.intervals.values_mut()
    }

    /// Take ownership of all intervals.
    pub fn into_intervals(self) -> Vec<LiveInterval> {
        self.intervals.into_values().collect()
    }

    /// Get the number of virtual registers.
    pub fn vreg_count(&self) -> usize {
        self.intervals.len()
    }

    /// Get the number of instructions.
    pub fn instruction_count(&self) -> u32 {
        self.num_instructions
    }
}

impl Default for LivenessAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{
        ArithmeticBuilder, ContainerBuilder, ControlBuilder, GraphBuilder, ObjectBuilder,
    };

    #[test]
    fn test_liveness_simple() {
        let mut builder = GraphBuilder::new(4, 2);

        let p0 = builder.parameter(0).unwrap();
        let p1 = builder.parameter(1).unwrap();
        let sum = builder.int_add(p0, p1);
        let _ret = builder.return_value(sum);

        let graph = builder.finish();
        let analysis = LivenessAnalysis::analyze(&graph);

        // Should have vregs for parameters and sum
        assert!(analysis.vreg_count() >= 3);
    }

    #[test]
    fn test_liveness_constants() {
        let mut builder = GraphBuilder::new(4, 0);

        let a = builder.const_int(10);
        let b = builder.const_int(20);
        let sum = builder.int_add(a, b);
        let _ret = builder.return_value(sum);

        let graph = builder.finish();
        let analysis = LivenessAnalysis::analyze(&graph);

        // Both constants and sum should have vregs
        assert!(analysis.vreg_count() >= 3);
    }

    #[test]
    fn test_liveness_interval_ranges() {
        let mut builder = GraphBuilder::new(4, 1);

        let p0 = builder.parameter(0).unwrap();
        let one = builder.const_int(1);
        let add1 = builder.int_add(p0, one);
        let add2 = builder.int_add(add1, one);
        let _ret = builder.return_value(add2);

        let graph = builder.finish();
        let analysis = LivenessAnalysis::analyze(&graph);

        // `one` should have a live range spanning both additions
        if let Some(vreg) = analysis.vreg_for_node(one) {
            let interval = analysis.interval(vreg).unwrap();
            assert!(!interval.is_empty());
            // The constant is used in two places, so should have uses
            assert!(interval.uses().len() >= 2);
        }
    }

    // =========================================================================
    // Vector Register Class Tests
    // =========================================================================

    #[test]
    fn test_reg_class_for_int64() {
        // Int64 should map to Int (GPR)
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::Int64),
            RegClass::Int
        );
    }

    #[test]
    fn test_reg_class_for_float64() {
        // Float64 should map to Float (XMM)
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::Float64),
            RegClass::Float
        );
    }

    #[test]
    fn test_reg_class_for_128bit_vectors() {
        // All 128-bit vectors should map to Float (XMM)
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V2I64),
            RegClass::Float
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V4I32),
            RegClass::Float
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V8I16),
            RegClass::Float
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V16I8),
            RegClass::Float
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V2F64),
            RegClass::Float
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V4F32),
            RegClass::Float
        );
    }

    #[test]
    fn test_reg_class_for_256bit_vectors() {
        // All 256-bit vectors should map to Vec256 (YMM)
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V4I64),
            RegClass::Vec256
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V8I32),
            RegClass::Vec256
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V16I16),
            RegClass::Vec256
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V32I8),
            RegClass::Vec256
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V4F64),
            RegClass::Vec256
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V8F32),
            RegClass::Vec256
        );
    }

    #[test]
    fn test_reg_class_for_512bit_vectors() {
        // All 512-bit vectors should map to Vec512 (ZMM)
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V8I64),
            RegClass::Vec512
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V16I32),
            RegClass::Vec512
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V8F64),
            RegClass::Vec512
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::V16F32),
            RegClass::Vec512
        );
    }

    #[test]
    fn test_reg_class_for_bool() {
        // Bool should map to Int (GPR)
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::Bool),
            RegClass::Int
        );
    }

    #[test]
    fn test_reg_class_for_objects() {
        // Object types should map to Int (GPR) - they're pointers
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::Object),
            RegClass::Int
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::List),
            RegClass::Int
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::Tuple),
            RegClass::Int
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::String),
            RegClass::Int
        );
    }

    #[test]
    fn test_reg_class_for_special_types() {
        // Top/Bottom and side-effect types should map to Int
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::Top),
            RegClass::Int
        );
        assert_eq!(
            LivenessAnalysis::reg_class_for_type(ValueType::Bottom),
            RegClass::Int
        );
    }

    #[test]
    fn test_vector_reg_class_width_consistency() {
        // Verify that reg_class width matches ValueType bit_width for vectors
        let test_cases = [
            (ValueType::V2I64, 128),
            (ValueType::V4I64, 256),
            (ValueType::V8I64, 512),
            (ValueType::V2F64, 128),
            (ValueType::V4F64, 256),
            (ValueType::V8F64, 512),
        ];

        for (ty, expected_bits) in test_cases {
            let reg_class = LivenessAnalysis::reg_class_for_type(ty);
            assert_eq!(
                reg_class.width(),
                expected_bits,
                "Mismatch for {:?}: reg_class {:?} has width {} but type has {} bits",
                ty,
                reg_class,
                reg_class.width(),
                expected_bits
            );
        }
    }

    #[test]
    fn test_vector_reg_class_spill_size_matches() {
        // Verify spill sizes are correct for vector types
        let test_cases = [
            (ValueType::Int64, 8),    // GPR: 8 bytes
            (ValueType::Float64, 16), // XMM: 16 bytes
            (ValueType::V2I64, 16),   // XMM: 16 bytes
            (ValueType::V4I64, 32),   // YMM: 32 bytes
            (ValueType::V8I64, 64),   // ZMM: 64 bytes
        ];

        for (ty, expected_spill_bytes) in test_cases {
            let reg_class = LivenessAnalysis::reg_class_for_type(ty);
            assert_eq!(
                reg_class.spill_size(),
                expected_spill_bytes,
                "{:?} should have spill size {} but got {}",
                ty,
                expected_spill_bytes,
                reg_class.spill_size()
            );
        }
    }

    #[test]
    fn test_vector_type_is_vector_class() {
        // All 128+ bit vector types should have is_vector() true for their reg_class
        let vector_types = [
            ValueType::V2I64,
            ValueType::V4I64,
            ValueType::V8I64,
            ValueType::V2F64,
            ValueType::V4F64,
            ValueType::V8F64,
        ];

        for ty in vector_types {
            let reg_class = LivenessAnalysis::reg_class_for_type(ty);
            assert!(
                reg_class.is_vector(),
                "{:?} should map to a vector register class, but got {:?}",
                ty,
                reg_class
            );
        }
    }

    #[test]
    fn test_wide_vector_class_256_512() {
        // 256+ bit vectors should have is_wide_vector() true
        let wide_types = [
            ValueType::V4I64,
            ValueType::V8I64,
            ValueType::V4F64,
            ValueType::V8F64,
        ];

        for ty in wide_types {
            let reg_class = LivenessAnalysis::reg_class_for_type(ty);
            assert!(
                reg_class.is_wide_vector(),
                "{:?} should map to a wide vector class, but got {:?}",
                ty,
                reg_class
            );
        }

        // 128-bit vectors are NOT wide
        let narrow_types = [ValueType::V2I64, ValueType::V2F64];
        for ty in narrow_types {
            let reg_class = LivenessAnalysis::reg_class_for_type(ty);
            assert!(
                !reg_class.is_wide_vector(),
                "{:?} should NOT map to a wide vector class, but got {:?}",
                ty,
                reg_class
            );
        }
    }
}

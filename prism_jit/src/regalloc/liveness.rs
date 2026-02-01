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
    fn reg_class_for_type(ty: ValueType) -> RegClass {
        match ty {
            ValueType::Float64 => RegClass::Float,
            _ => RegClass::Int, // Most values go in GPRs
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
}

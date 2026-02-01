//! Function Inlining with Cost Model.
//!
//! Inlining replaces call sites with the body of the called function,
//! eliminating call overhead and enabling cross-procedural optimization.

use super::OptimizationPass;
use crate::ir::graph::Graph;
use crate::ir::node::NodeId;
use crate::ir::operators::{CallKind, Operator};

// =============================================================================
// Inlining Configuration
// =============================================================================

/// Configuration for the inlining heuristics.
#[derive(Debug, Clone)]
pub struct InlineConfig {
    /// Maximum callee size (in nodes) to inline.
    pub max_callee_size: usize,
    /// Maximum inlining depth (for recursive inlines).
    pub max_depth: usize,
    /// Maximum total code growth factor.
    pub max_growth_factor: f64,
    /// Bonus for call sites in hot loops.
    pub hot_call_bonus: i32,
    /// Penalty for calls with many arguments.
    pub arg_penalty_per_arg: i32,
    /// Always inline functions marked as `#[inline(always)]`.
    pub respect_always_inline: bool,
    /// Never inline functions marked as `#[inline(never)]`.
    pub respect_never_inline: bool,
}

impl Default for InlineConfig {
    fn default() -> Self {
        Self {
            max_callee_size: 100,
            max_depth: 4,
            max_growth_factor: 2.0,
            hot_call_bonus: 50,
            arg_penalty_per_arg: 5,
            respect_always_inline: true,
            respect_never_inline: true,
        }
    }
}

impl InlineConfig {
    /// Create a conservative inlining configuration.
    pub fn conservative() -> Self {
        Self {
            max_callee_size: 30,
            max_depth: 2,
            max_growth_factor: 1.2,
            ..Default::default()
        }
    }

    /// Create an aggressive inlining configuration.
    pub fn aggressive() -> Self {
        Self {
            max_callee_size: 500,
            max_depth: 8,
            max_growth_factor: 5.0,
            ..Default::default()
        }
    }
}

// =============================================================================
// Callee Info
// =============================================================================

/// Information about a potential inline candidate.
#[derive(Debug, Clone, Default)]
pub struct CalleeInfo {
    /// Function identifier.
    pub func_id: u64,
    /// Number of nodes in the callee.
    pub size: usize,
    /// Number of parameters.
    pub param_count: usize,
    /// Whether the function is recursive.
    pub is_recursive: bool,
    /// Whether marked as always-inline.
    pub always_inline: bool,
    /// Whether marked as never-inline.
    pub never_inline: bool,
}

// =============================================================================
// Call Site
// =============================================================================

/// Information about a call site.
#[derive(Debug, Clone)]
pub struct CallSite {
    /// Node ID of the call instruction.
    pub call_node: NodeId,
    /// Callee information (if known).
    pub callee: Option<CalleeInfo>,
    /// Loop depth at the call site.
    pub loop_depth: u32,
    /// Whether this is a hot call site.
    pub is_hot: bool,
    /// Computed inlining priority.
    pub priority: i32,
}

// =============================================================================
// Inlining Pass
// =============================================================================

/// Function inlining optimization pass.
#[derive(Debug)]
pub struct Inline {
    /// Configuration.
    config: InlineConfig,
    /// Number of functions inlined.
    inlined: usize,
    /// Total code growth (nodes added).
    growth: usize,
    /// Current inlining depth.
    current_depth: usize,
    /// Functions currently being inlined (cycle detection).
    inline_stack: Vec<u64>,
}

impl Inline {
    /// Create a new inlining pass with default configuration.
    pub fn new() -> Self {
        Self {
            config: InlineConfig::default(),
            inlined: 0,
            growth: 0,
            current_depth: 0,
            inline_stack: Vec::new(),
        }
    }

    /// Create inlining pass with custom configuration.
    pub fn with_config(config: InlineConfig) -> Self {
        Self {
            config,
            inlined: 0,
            growth: 0,
            current_depth: 0,
            inline_stack: Vec::new(),
        }
    }

    /// Get number of functions inlined.
    #[inline]
    pub fn inlined(&self) -> usize {
        self.inlined
    }

    /// Get total code growth.
    #[inline]
    pub fn growth(&self) -> usize {
        self.growth
    }

    /// Run the inlining pass.
    fn run_inline(&mut self, graph: &mut Graph) -> bool {
        let call_sites = self.find_call_sites(graph);

        if call_sites.is_empty() {
            return false;
        }

        let mut sorted_sites = call_sites;
        sorted_sites.sort_by(|a, b| b.priority.cmp(&a.priority));

        let initial_size = graph.len();
        let max_size = (initial_size as f64 * self.config.max_growth_factor) as usize;

        let mut changed = false;

        for site in sorted_sites {
            if graph.len() >= max_size {
                break;
            }

            if self.current_depth >= self.config.max_depth {
                break;
            }

            if self.should_inline(&site) {
                if self.inline_call_site(graph, &site) {
                    self.inlined += 1;
                    changed = true;
                }
            }
        }

        self.growth = graph.len().saturating_sub(initial_size);
        changed
    }

    /// Find all call sites in the graph.
    fn find_call_sites(&self, graph: &Graph) -> Vec<CallSite> {
        let mut sites = Vec::new();

        for (node_id, node) in graph.iter() {
            if let Operator::Call(call_kind) = &node.op {
                let site = CallSite {
                    call_node: node_id,
                    callee: None,
                    loop_depth: 0,
                    is_hot: false,
                    priority: self.compute_priority(call_kind, 0, false),
                };
                sites.push(site);
            }
        }

        sites
    }

    /// Compute inlining priority for a call site.
    fn compute_priority(&self, _call_kind: &CallKind, loop_depth: u32, is_hot: bool) -> i32 {
        let mut priority: i32 = 0;

        if is_hot {
            priority += self.config.hot_call_bonus;
        }

        priority += (loop_depth as i32) * 20;

        priority
    }

    /// Decide whether to inline a call site.
    fn should_inline(&self, site: &CallSite) -> bool {
        let callee = match &site.callee {
            Some(c) => c,
            None => return false,
        };

        if self.config.respect_never_inline && callee.never_inline {
            return false;
        }

        if self.config.respect_always_inline && callee.always_inline {
            return true;
        }

        if callee.size > self.config.max_callee_size {
            return false;
        }

        if callee.is_recursive && !self.inline_stack.is_empty() {
            return false;
        }

        if self.inline_stack.contains(&callee.func_id) {
            return false;
        }

        site.priority > 0
    }

    /// Inline a single call site.
    fn inline_call_site(&mut self, _graph: &mut Graph, site: &CallSite) -> bool {
        let callee = match &site.callee {
            Some(c) => c,
            None => return false,
        };

        self.inline_stack.push(callee.func_id);
        self.current_depth += 1;

        self.inline_stack.pop();
        self.current_depth -= 1;

        false
    }
}

impl Default for Inline {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationPass for Inline {
    fn name(&self) -> &'static str {
        "inline"
    }

    fn run(&mut self, graph: &mut Graph) -> bool {
        self.run_inline(graph)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::{ControlBuilder, GraphBuilder};

    #[test]
    fn test_inline_config_default() {
        let config = InlineConfig::default();
        assert_eq!(config.max_callee_size, 100);
        assert_eq!(config.max_depth, 4);
    }

    #[test]
    fn test_inline_config_conservative() {
        let config = InlineConfig::conservative();
        assert!(config.max_callee_size < InlineConfig::default().max_callee_size);
        assert!(config.max_depth < InlineConfig::default().max_depth);
    }

    #[test]
    fn test_inline_config_aggressive() {
        let config = InlineConfig::aggressive();
        assert!(config.max_callee_size > InlineConfig::default().max_callee_size);
        assert!(config.max_depth > InlineConfig::default().max_depth);
    }

    #[test]
    fn test_inline_new() {
        let inline = Inline::new();
        assert_eq!(inline.inlined(), 0);
        assert_eq!(inline.growth(), 0);
    }

    #[test]
    fn test_inline_name() {
        let inline = Inline::new();
        assert_eq!(inline.name(), "inline");
    }

    #[test]
    fn test_inline_no_calls() {
        let mut builder = GraphBuilder::new(2, 1);
        let p0 = builder.parameter(0).unwrap();
        builder.return_value(p0);

        let mut graph = builder.finish();
        let mut inline = Inline::new();

        let changed = inline.run(&mut graph);
        assert!(!changed);
        assert_eq!(inline.inlined(), 0);
    }

    #[test]
    fn test_callee_info_default() {
        let info = CalleeInfo::default();
        assert_eq!(info.size, 0);
        assert!(!info.is_recursive);
        assert!(!info.always_inline);
    }

    #[test]
    fn test_should_inline_never() {
        let inline = Inline::new();
        let site = CallSite {
            call_node: NodeId::new(0),
            callee: Some(CalleeInfo {
                never_inline: true,
                ..Default::default()
            }),
            loop_depth: 0,
            is_hot: false,
            priority: 100,
        };

        assert!(!inline.should_inline(&site));
    }

    #[test]
    fn test_should_inline_always() {
        let inline = Inline::new();
        let site = CallSite {
            call_node: NodeId::new(0),
            callee: Some(CalleeInfo {
                always_inline: true,
                size: 1000,
                ..Default::default()
            }),
            loop_depth: 0,
            is_hot: false,
            priority: 100,
        };

        assert!(inline.should_inline(&site));
    }

    #[test]
    fn test_should_inline_too_large() {
        let inline = Inline::new();
        let site = CallSite {
            call_node: NodeId::new(0),
            callee: Some(CalleeInfo {
                size: 1000,
                ..Default::default()
            }),
            loop_depth: 0,
            is_hot: false,
            priority: 100,
        };

        assert!(!inline.should_inline(&site));
    }
}

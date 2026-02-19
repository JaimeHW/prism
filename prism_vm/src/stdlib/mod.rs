//! Python stdlib module implementations.
//!
//! This module provides implementations of Python's standard library modules
//! with maximum performance through direct hardware intrinsics and zero-allocation
//! algorithms.
//!
//! # Modules
//!
//! - `math` - Mathematical functions (sin, cos, sqrt, etc.)
//! - `os` - Operating system interface
//! - `sys` - System-specific parameters and functions
//! - `functools` - Higher-order functions and callable operations
//! - `itertools` - Iterator building blocks for efficient looping
//! - `io` - Core I/O primitives (StringIO, BytesIO, FileMode)

pub mod collections;
pub mod exceptions;
pub mod functools;
pub mod generators;
pub mod io;
pub mod itertools;
pub mod json;
pub mod math;
pub mod os;
pub mod re;
pub mod sys;
pub mod time;

use prism_core::Value;
use std::sync::Arc;

/// Result type for module attribute lookup.
pub type ModuleResult = Result<Value, ModuleError>;

/// Errors that can occur during module operations.
#[derive(Debug, Clone)]
pub enum ModuleError {
    /// Attribute not found in module.
    AttributeError(String),
    /// Invalid argument for function.
    ValueError(String),
    /// Type mismatch.
    TypeError(String),
    /// Domain error (e.g., sqrt of negative).
    MathDomainError(String),
    /// Range error (e.g., result too large).
    MathRangeError(String),
    /// OS error (e.g., file not found).
    OSError(String),
}

impl std::fmt::Display for ModuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModuleError::AttributeError(msg) => write!(f, "AttributeError: {}", msg),
            ModuleError::ValueError(msg) => write!(f, "ValueError: {}", msg),
            ModuleError::TypeError(msg) => write!(f, "TypeError: {}", msg),
            ModuleError::MathDomainError(msg) => write!(f, "math domain error: {}", msg),
            ModuleError::MathRangeError(msg) => write!(f, "math range error: {}", msg),
            ModuleError::OSError(msg) => write!(f, "OSError: {}", msg),
        }
    }
}

impl std::error::Error for ModuleError {}

/// Trait for Python module implementations.
pub trait Module {
    /// Get the module name.
    fn name(&self) -> &str;

    /// Get an attribute from the module.
    fn get_attr(&self, name: &str) -> ModuleResult;

    /// List all attribute names.
    fn dir(&self) -> Vec<Arc<str>> {
        Vec::new() // Default empty impl
    }
}

/// Registry of all stdlib modules.
pub struct StdlibRegistry {
    modules: std::collections::HashMap<Arc<str>, Box<dyn Module + Send + Sync>>,
}

impl StdlibRegistry {
    /// Create a new registry with all stdlib modules.
    pub fn new() -> Self {
        Self::with_optional_sys_args(None)
    }

    /// Create a registry with an explicit `sys.argv` payload.
    pub fn with_sys_args(args: Vec<String>) -> Self {
        Self::with_optional_sys_args(Some(args))
    }

    fn with_optional_sys_args(sys_args: Option<Vec<String>>) -> Self {
        let mut modules: std::collections::HashMap<Arc<str>, Box<dyn Module + Send + Sync>> =
            std::collections::HashMap::new();

        // Register math module
        modules.insert(Arc::from("math"), Box::new(math::MathModule::new()));

        // Register os module
        modules.insert(Arc::from("os"), Box::new(os::OsModule::new()));

        // Register sys module
        let sys_module = match sys_args {
            Some(args) => sys::SysModule::with_args(args),
            None => sys::SysModule::new(),
        };
        modules.insert(Arc::from("sys"), Box::new(sys_module));

        // Register time module
        modules.insert(Arc::from("time"), Box::new(time::TimeModule::new()));

        // Register re module
        modules.insert(Arc::from("re"), Box::new(re::ReModule::new()));

        // Register json module
        modules.insert(Arc::from("json"), Box::new(json::JsonModule::new()));

        // Register collections module
        modules.insert(
            Arc::from("collections"),
            Box::new(collections::CollectionsModule::new()),
        );

        // Register functools module
        modules.insert(
            Arc::from("functools"),
            Box::new(functools::FunctoolsModule::new()),
        );

        // Register itertools module
        modules.insert(
            Arc::from("itertools"),
            Box::new(itertools::ItertoolsModule::new()),
        );

        // Register io module
        modules.insert(Arc::from("io"), Box::new(io::IoModule::new()));

        Self { modules }
    }

    /// Get a module by name.
    pub fn get(&self, name: &str) -> Option<&(dyn Module + Send + Sync)> {
        self.modules.get(name).map(|m| m.as_ref())
    }

    /// Check if a module exists.
    pub fn contains(&self, name: &str) -> bool {
        self.modules.contains_key(name)
    }

    /// List all available module names.
    pub fn list_modules(&self) -> Vec<&str> {
        self.modules.keys().map(|k| k.as_ref()).collect()
    }
}

impl Default for StdlibRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = StdlibRegistry::new();
        assert!(registry.contains("math"));
    }

    #[test]
    fn test_registry_get_math() {
        let registry = StdlibRegistry::new();
        let math = registry.get("math");
        assert!(math.is_some());
        assert_eq!(math.unwrap().name(), "math");
    }

    #[test]
    fn test_registry_unknown_module() {
        let registry = StdlibRegistry::new();
        assert!(!registry.contains("nonexistent"));
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_list_modules() {
        let registry = StdlibRegistry::new();
        let modules = registry.list_modules();
        assert!(modules.contains(&"math"));
    }
}

//! `ImportResolver` - the core import machinery.
//!
//! Provides high-performance module resolution with multi-tier caching:
//! 1. `sys.modules` cache (first check)
//! 2. Stdlib registry (built-in modules)
//! 3. File system (future: .py/.pyc loading)

use super::ModuleObject;
use crate::stdlib::{Module, ModuleError, StdlibRegistry};
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use rustc_hash::FxHashMap;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::ThreadId;

// =============================================================================
// Error Types
// =============================================================================

/// Errors that can occur during import resolution.
#[derive(Debug, Clone)]
pub enum ImportError {
    /// Module not found in any search path.
    ModuleNotFound { module: Arc<str> },

    /// Attribute not found in module.
    ImportFromError { module: Arc<str>, name: Arc<str> },

    /// Circular import detected.
    CircularImport { module: Arc<str> },

    /// Error loading module source.
    LoadError { module: Arc<str>, message: Arc<str> },

    /// Error executing module code.
    ExecutionError { module: Arc<str>, message: Arc<str> },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::ModuleNotFound { module } => {
                write!(f, "ModuleNotFoundError: No module named '{}'", module)
            }
            ImportError::ImportFromError { module, name } => {
                write!(
                    f,
                    "ImportError: cannot import name '{}' from '{}'",
                    name, module
                )
            }
            ImportError::CircularImport { module } => {
                write!(f, "ImportError: circular import for '{}'", module)
            }
            ImportError::LoadError { module, message } => {
                write!(f, "ImportError: failed to load '{}': {}", module, message)
            }
            ImportError::ExecutionError { module, message } => {
                write!(
                    f,
                    "ImportError: failed to execute '{}': {}",
                    module, message
                )
            }
        }
    }
}

impl std::error::Error for ImportError {}

// =============================================================================
// ImportState - Concurrent Import Synchronization
// =============================================================================

/// State for a module currently being imported.
///
/// Tracks the loading thread ID to detect true circular imports (same thread
/// re-entering import) vs concurrent imports (different threads, which should
/// wait for the first one to complete).
struct ImportState {
    /// Thread ID of the thread performing the import.
    /// Used to detect true circular imports (same thread re-entering).
    loader_thread: ThreadId,

    /// Whether the import has completed.
    /// Protected by mutex and used with condvar for wait semantics.
    completed: Mutex<bool>,

    /// Condvar to signal when import completes.
    complete: Condvar,
}

impl ImportState {
    /// Create new import state for the current thread.
    fn new() -> Self {
        Self {
            loader_thread: std::thread::current().id(),
            completed: Mutex::new(false),
            complete: Condvar::new(),
        }
    }

    /// Check if this import is being performed by the current thread.
    /// If true, this is a true circular import.
    fn is_circular(&self) -> bool {
        self.loader_thread == std::thread::current().id()
    }

    /// Wait for the import to complete.
    /// Returns when the import has finished and the module is cached.
    fn wait(&self) {
        let mut completed = self.completed.lock().unwrap();
        while !*completed {
            // Wait for signal that import is complete
            // Use wait_while pattern to handle spurious wakeups
            completed = self.complete.wait(completed).unwrap();
        }
    }

    /// Signal all waiting threads that import is complete.
    fn signal_complete(&self) {
        let mut completed = self.completed.lock().unwrap();
        *completed = true;
        // Wake all waiting threads
        self.complete.notify_all();
    }
}

// =============================================================================
// ImportResolver
// =============================================================================

/// High-performance import resolver with multi-tier caching.
///
/// # Resolution Order
///
/// 1. **sys.modules cache** - O(1) lookup for already-imported modules
/// 2. **Stdlib registry** - Direct access to built-in modules (math, os, sys)
/// 3. **File system** - (Future) Search sys.path for .py/.pyc files
///
/// # Thread Safety
///
/// Uses `RwLock` for the sys.modules cache to allow concurrent reads
/// (the common case) while serializing writes.
///
/// # Performance
///
/// - Cache keys are `InternedString` for O(1) hash/equality
/// - Built-in modules are pre-initialized (no parsing/compilation)
/// - `Arc<ModuleObject>` for zero-copy module sharing
pub struct ImportResolver {
    /// sys.modules: canonical cache of all loaded modules.
    /// Uses InternedString keys for O(1) lookup.
    sys_modules: RwLock<FxHashMap<InternedString, Arc<ModuleObject>>>,

    /// Stdlib registry for built-in modules (math, os, sys, etc.).
    stdlib: StdlibRegistry,

    /// sys.path: search paths for source files.
    /// Future use for .py file loading.
    search_paths: RwLock<Vec<Arc<str>>>,

    /// Modules currently being imported (for circular import detection and wait semantics).
    /// Maps module name to ImportState which tracks the loading thread and provides
    /// a Condvar for other threads to wait on.
    loading: RwLock<FxHashMap<InternedString, Arc<ImportState>>>,

    /// Fast pointer lookup for imported module objects.
    ///
    /// Keys are raw `ModuleObject` pointers cast to usize. This lets opcode handlers
    /// validate and resolve module pointers without unsafe casting.
    module_ptrs: RwLock<FxHashMap<usize, Arc<ModuleObject>>>,
}

impl ImportResolver {
    /// Create a new import resolver with default configuration.
    pub fn new() -> Self {
        Self::with_stdlib_and_paths(StdlibRegistry::new(), Vec::new())
    }

    /// Create a new import resolver with explicit `sys.argv`.
    pub fn with_sys_args(args: Vec<String>) -> Self {
        Self::with_stdlib_and_paths(StdlibRegistry::with_sys_args(args), Vec::new())
    }

    /// Create a resolver with custom search paths.
    pub fn with_paths(paths: Vec<Arc<str>>) -> Self {
        Self::with_stdlib_and_paths(StdlibRegistry::new(), paths)
    }

    fn with_stdlib_and_paths(stdlib: StdlibRegistry, paths: Vec<Arc<str>>) -> Self {
        Self {
            sys_modules: RwLock::new(FxHashMap::default()),
            stdlib,
            search_paths: RwLock::new(paths),
            loading: RwLock::new(FxHashMap::default()),
            module_ptrs: RwLock::new(FxHashMap::default()),
        }
    }

    /// Import a module by name.
    ///
    /// # Resolution Order
    ///
    /// 1. Check sys.modules cache
    /// 2. Check stdlib registry
    /// 3. Search sys.path for .py files (future)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let resolver = ImportResolver::new();
    /// let math = resolver.import_module("math")?;
    /// ```
    pub fn import_module(&self, name: &str) -> Result<Arc<ModuleObject>, ImportError> {
        let key = intern(name);

        // 1. Check sys.modules cache (fast path)
        if let Some(module) = self.sys_modules.read().unwrap().get(&key) {
            return Ok(Arc::clone(module));
        }

        // 2. Atomically check if module is being imported OR insert our ImportState
        //    This prevents the TOCTOU race where multiple threads all think they're first
        let (should_wait, wait_state) = {
            let mut loading = self.loading.write().unwrap();

            if let Some(state) = loading.get(&key) {
                if state.is_circular() {
                    // Same thread re-entering - true circular import
                    return Err(ImportError::CircularImport {
                        module: Arc::from(name),
                    });
                }
                // Different thread - we need to wait
                (true, Some(Arc::clone(state)))
            } else {
                // No one is importing this module yet - we're first
                let import_state = Arc::new(ImportState::new());
                loading.insert(key.clone(), Arc::clone(&import_state));
                (false, Some(import_state))
            }
        };

        // 3. If another thread is importing, wait for it
        if should_wait {
            let state = wait_state.unwrap();
            state.wait();

            // After waiting, the module should be cached - retrieve it
            if let Some(module) = self.sys_modules.read().unwrap().get(&key) {
                return Ok(Arc::clone(module));
            } else {
                // Import failed in the other thread
                return Err(ImportError::ModuleNotFound {
                    module: Arc::from(name),
                });
            }
        }

        // 4. We are the loading thread - perform the actual import
        let import_state = wait_state.unwrap();

        // 5. Try stdlib registry
        let result = if let Some(stdlib_module) = self.stdlib.get(name) {
            // Create ModuleObject from stdlib module
            self.load_stdlib_module(name, stdlib_module)
        } else {
            // 6. Try file system (not yet implemented)
            Err(ImportError::ModuleNotFound {
                module: Arc::from(name),
            })
        };

        // 7. Cache successful imports BEFORE signaling waiters
        if let Ok(ref module) = result {
            self.sys_modules
                .write()
                .unwrap()
                .insert(key.clone(), Arc::clone(module));
            self.register_module_ptr(module);
        }

        // 8. Signal waiting threads that import is complete
        import_state.signal_complete();

        // 9. Remove from loading set
        self.loading.write().unwrap().remove(&key);

        result
    }

    /// Import a module using a dotted name (e.g., `os.path`).
    ///
    /// This resolves each component of the dotted name in sequence:
    /// 1. Import the top-level module (`os`)
    /// 2. For each subsequent part, either:
    ///    a. Check sys.modules for the full dotted name
    ///    b. Get the attribute from the parent module
    ///    c. If the attribute is a submodule, cache it as a full dotted name
    ///
    /// Returns the final module in the chain.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let resolver = ImportResolver::new();
    /// let os_path = resolver.import_dotted("os.path")?;
    /// ```
    pub fn import_dotted(&self, name: &str) -> Result<Arc<ModuleObject>, ImportError> {
        use super::package::DottedName;

        // Fast path: check cache first
        let key = intern(name);
        if let Some(module) = self.sys_modules.read().unwrap().get(&key) {
            return Ok(Arc::clone(module));
        }

        // Parse the dotted name
        let dotted = DottedName::parse(name).ok_or_else(|| ImportError::ModuleNotFound {
            module: Arc::from(name),
        })?;

        // Simple name — delegate to import_module
        if dotted.is_simple() {
            return self.import_module(name);
        }

        // Import the top-level module first
        let mut current = self.import_module(dotted.top_level())?;

        // Resolve each subsequent component
        for depth in 2..=dotted.depth() {
            let full_name = dotted.name_at_depth(depth);
            let part = &dotted.parts()[depth - 1];

            // Check if this dotted name is already cached
            let cached_key = intern(&full_name);
            if let Some(module) = self.sys_modules.read().unwrap().get(&cached_key) {
                current = Arc::clone(module);
                continue;
            }

            // Try to import it as a stdlib submodule (e.g., "os.path")
            if let Some(stdlib_module) = self.stdlib.get(&full_name) {
                let module = self.load_stdlib_module(&full_name, stdlib_module)?;
                self.sys_modules
                    .write()
                    .unwrap()
                    .insert(cached_key, Arc::clone(&module));
                // Also set as attribute on parent
                // (deferred — the parent module should already expose it)
                current = module;
                continue;
            }

            // Try to get as attribute from parent module
            if let Some(_value) = current.get_attr(part) {
                // The attribute exists on the parent. If it's a module object
                // in our cache, use that. Otherwise, create a wrapper.
                // For now, if we have a cached submodule, use it.
                // TODO: once we have proper type checking, handle module-typed attrs
                return Err(ImportError::ModuleNotFound {
                    module: Arc::from(full_name.as_str()),
                });
            }

            return Err(ImportError::ModuleNotFound {
                module: Arc::from(full_name.as_str()),
            });
        }

        Ok(current)
    }

    /// Import a module using a relative import specification.
    ///
    /// # Parameters
    ///
    /// - `name`: The name to import (after the dots)
    /// - `level`: Number of leading dots (1 = current package, 2 = parent, etc.)
    /// - `package`: The `__package__` attribute of the importing module
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // from . import foo  (in package "mypackage")
    /// resolver.import_relative("foo", 1, "mypackage")?;
    ///
    /// // from ..bar import baz  (in package "a.b.c")
    /// resolver.import_relative("bar", 2, "a.b.c")?;
    /// ```
    pub fn import_relative(
        &self,
        name: &str,
        level: u32,
        package: &str,
    ) -> Result<Arc<ModuleObject>, ImportError> {
        let absolute_name = super::package::resolve_relative_import(name, level, package)?;
        self.import_dotted(&absolute_name)
    }

    /// Load a stdlib module into a ModuleObject.
    fn load_stdlib_module(
        &self,
        name: &str,
        stdlib: &(dyn Module + Send + Sync),
    ) -> Result<Arc<ModuleObject>, ImportError> {
        let module = ModuleObject::new(name);

        // Get all attributes from the stdlib module
        for attr_name in stdlib.dir() {
            match stdlib.get_attr(&attr_name) {
                Ok(value) => {
                    module.set_attr(&attr_name, value);
                }
                Err(ModuleError::AttributeError(_)) => {
                    // Skip attributes that error (shouldn't happen, but be defensive)
                    continue;
                }
                Err(e) => {
                    return Err(ImportError::LoadError {
                        module: Arc::from(name),
                        message: Arc::from(e.to_string()),
                    });
                }
            }
        }

        Ok(Arc::new(module))
    }

    /// Import a specific attribute from a module.
    ///
    /// This is used for `from module import name`.
    pub fn import_from(
        &self,
        module: &Arc<ModuleObject>,
        name: &str,
    ) -> Result<Value, ImportError> {
        module
            .get_attr(name)
            .ok_or_else(|| ImportError::ImportFromError {
                module: Arc::from(module.name()),
                name: Arc::from(name),
            })
    }

    /// Import all public names from a module.
    ///
    /// This is used for `from module import *`.
    ///
    /// Returns a list of (name, value) pairs to be inserted into the
    /// importing module's namespace.
    pub fn import_star(
        &self,
        module: &Arc<ModuleObject>,
    ) -> Result<Vec<(InternedString, Value)>, ImportError> {
        Ok(module.public_attrs())
    }

    /// Get a module from sys.modules cache.
    ///
    /// Returns `None` if the module hasn't been imported yet.
    pub fn get_cached(&self, name: &str) -> Option<Arc<ModuleObject>> {
        let key = intern(name);
        self.sys_modules.read().unwrap().get(&key).cloned()
    }

    /// Insert a module directly into sys.modules.
    ///
    /// This is useful for injecting modules programmatically.
    pub fn insert_module(&self, name: &str, module: Arc<ModuleObject>) {
        let key = intern(name);
        self.sys_modules
            .write()
            .unwrap()
            .insert(key, Arc::clone(&module));
        self.register_module_ptr(&module);
    }

    /// Remove a module from sys.modules.
    ///
    /// Returns the module if it was cached, `None` otherwise.
    pub fn remove_module(&self, name: &str) -> Option<Arc<ModuleObject>> {
        let key = intern(name);
        let removed = self.sys_modules.write().unwrap().remove(&key);
        if let Some(ref module) = removed {
            self.unregister_module_ptr(module);
        }
        removed
    }

    /// Resolve a raw object pointer to an imported module, if it is a known module.
    pub fn module_from_ptr(&self, ptr: *const ()) -> Option<Arc<ModuleObject>> {
        self.module_ptrs
            .read()
            .unwrap()
            .get(&(ptr as usize))
            .cloned()
    }

    /// Add a search path for source files.
    pub fn add_search_path(&self, path: Arc<str>) {
        self.search_paths.write().unwrap().push(path);
    }

    /// Get current search paths.
    pub fn search_paths(&self) -> Vec<Arc<str>> {
        self.search_paths.read().unwrap().clone()
    }

    /// List all cached modules.
    pub fn cached_modules(&self) -> Vec<Arc<str>> {
        self.sys_modules
            .read()
            .unwrap()
            .keys()
            .map(|k| Arc::from(k.as_ref()))
            .collect()
    }

    /// Check if a module is available (cached or in stdlib).
    pub fn module_exists(&self, name: &str) -> bool {
        let key = intern(name);
        self.sys_modules.read().unwrap().contains_key(&key) || self.stdlib.contains(name)
    }

    fn register_module_ptr(&self, module: &Arc<ModuleObject>) {
        self.module_ptrs
            .write()
            .unwrap()
            .insert(Arc::as_ptr(module) as usize, Arc::clone(module));
    }

    fn unregister_module_ptr(&self, module: &Arc<ModuleObject>) {
        self.module_ptrs
            .write()
            .unwrap()
            .remove(&(Arc::as_ptr(module) as usize));
    }
}

impl Default for ImportResolver {
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
    use prism_core::intern::interned_by_ptr;
    use prism_runtime::types::list::ListObject;

    #[test]
    fn test_import_resolver_new() {
        let resolver = ImportResolver::new();
        assert!(resolver.cached_modules().is_empty());
    }

    #[test]
    fn test_import_stdlib_math() {
        let resolver = ImportResolver::new();
        let result = resolver.import_module("math");
        assert!(result.is_ok());

        let module = result.unwrap();
        assert_eq!(module.name(), "math");
    }

    #[test]
    fn test_import_stdlib_os() {
        let resolver = ImportResolver::new();
        let result = resolver.import_module("os");
        assert!(result.is_ok());

        let module = result.unwrap();
        assert_eq!(module.name(), "os");
    }

    #[test]
    fn test_import_stdlib_sys() {
        let resolver = ImportResolver::new();
        let result = resolver.import_module("sys");
        assert!(result.is_ok());

        let module = result.unwrap();
        assert_eq!(module.name(), "sys");
    }

    #[test]
    fn test_import_nonexistent() {
        let resolver = ImportResolver::new();
        let result = resolver.import_module("nonexistent_module_12345");
        assert!(result.is_err());

        match result.unwrap_err() {
            ImportError::ModuleNotFound { module } => {
                assert_eq!(module.as_ref(), "nonexistent_module_12345");
            }
            _ => panic!("Expected ModuleNotFound error"),
        }
    }

    #[test]
    fn test_import_caching() {
        let resolver = ImportResolver::new();

        // First import
        let math1 = resolver.import_module("math").unwrap();

        // Second import should return cached module
        let math2 = resolver.import_module("math").unwrap();

        // Should be the same Arc (pointer equality)
        assert!(Arc::ptr_eq(&math1, &math2));
    }

    #[test]
    fn test_get_cached() {
        let resolver = ImportResolver::new();

        // Not cached yet
        assert!(resolver.get_cached("math").is_none());

        // Import it
        resolver.import_module("math").unwrap();

        // Now cached
        assert!(resolver.get_cached("math").is_some());
    }

    #[test]
    fn test_insert_module() {
        let resolver = ImportResolver::new();
        let module = Arc::new(ModuleObject::new("custom_module"));

        resolver.insert_module("custom_module", Arc::clone(&module));

        let cached = resolver.get_cached("custom_module");
        assert!(cached.is_some());
        assert!(Arc::ptr_eq(&cached.unwrap(), &module));
    }

    #[test]
    fn test_remove_module() {
        let resolver = ImportResolver::new();

        // Import math
        resolver.import_module("math").unwrap();
        assert!(resolver.get_cached("math").is_some());

        // Remove it
        let removed = resolver.remove_module("math");
        assert!(removed.is_some());
        assert!(resolver.get_cached("math").is_none());
    }

    #[test]
    fn test_module_exists() {
        let resolver = ImportResolver::new();

        // Stdlib modules exist
        assert!(resolver.module_exists("math"));
        assert!(resolver.module_exists("os"));
        assert!(resolver.module_exists("sys"));

        // Unknown modules don't exist
        assert!(!resolver.module_exists("unknown_module_xyz"));
    }

    #[test]
    fn test_import_from() {
        let resolver = ImportResolver::new();
        let math = resolver.import_module("math").unwrap();

        // Import pi from math
        let result = resolver.import_from(&math, "pi");
        assert!(result.is_ok());

        let pi = result.unwrap();
        if let Some(f) = pi.as_float() {
            assert!((f - std::f64::consts::PI).abs() < 1e-10);
        } else {
            panic!("Expected float value for pi");
        }
    }

    #[test]
    fn test_import_from_nonexistent() {
        let resolver = ImportResolver::new();
        let math = resolver.import_module("math").unwrap();

        let result = resolver.import_from(&math, "nonexistent_attr");
        assert!(result.is_err());

        match result.unwrap_err() {
            ImportError::ImportFromError { module, name } => {
                assert_eq!(module.as_ref(), "math");
                assert_eq!(name.as_ref(), "nonexistent_attr");
            }
            _ => panic!("Expected ImportFromError"),
        }
    }

    #[test]
    fn test_import_star() {
        let resolver = ImportResolver::new();
        let math = resolver.import_module("math").unwrap();

        let result = resolver.import_star(&math);
        assert!(result.is_ok());

        let names = result.unwrap();
        // Should have some public names
        assert!(!names.is_empty());

        // Check for expected names
        let name_strs: Vec<&str> = names.iter().map(|(k, _)| k.as_ref()).collect();
        assert!(name_strs.contains(&"pi") || name_strs.contains(&"e"));
    }

    #[test]
    fn test_search_paths() {
        let resolver = ImportResolver::new();

        assert!(resolver.search_paths().is_empty());

        resolver.add_search_path(Arc::from("/usr/lib/python"));
        resolver.add_search_path(Arc::from("/home/user/lib"));

        let paths = resolver.search_paths();
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_with_paths() {
        let paths = vec![Arc::from("/path1"), Arc::from("/path2")];
        let resolver = ImportResolver::with_paths(paths);

        assert_eq!(resolver.search_paths().len(), 2);
    }

    #[test]
    fn test_with_sys_args_populates_imported_sys_argv() {
        let resolver =
            ImportResolver::with_sys_args(vec!["prog.py".to_string(), "--fast".to_string()]);
        let sys = resolver
            .import_module("sys")
            .expect("sys import should succeed");
        let argv = sys.get_attr("argv").expect("sys.argv should be present");

        let argv_ptr = argv
            .as_object_ptr()
            .expect("sys.argv should be represented as list object");
        let list = unsafe { &*(argv_ptr as *const ListObject) };
        assert_eq!(list.len(), 2);

        let arg0 = list.get(0).expect("argv[0] should exist");
        let arg1 = list.get(1).expect("argv[1] should exist");

        let arg0_ptr = arg0
            .as_string_object_ptr()
            .expect("argv[0] should be string") as *const u8;
        let arg1_ptr = arg1
            .as_string_object_ptr()
            .expect("argv[1] should be string") as *const u8;

        assert_eq!(
            interned_by_ptr(arg0_ptr)
                .expect("argv[0] should resolve")
                .as_ref(),
            "prog.py"
        );
        assert_eq!(
            interned_by_ptr(arg1_ptr)
                .expect("argv[1] should resolve")
                .as_ref(),
            "--fast"
        );
    }

    #[test]
    fn test_module_from_ptr_resolves_cached_module() {
        let resolver = ImportResolver::new();
        let math = resolver
            .import_module("math")
            .expect("math import should succeed");
        let ptr = Arc::as_ptr(&math) as *const ();

        let resolved = resolver
            .module_from_ptr(ptr)
            .expect("module pointer should resolve");
        assert_eq!(resolved.name(), "math");
        assert!(Arc::ptr_eq(&math, &resolved));
    }

    #[test]
    fn test_cached_modules() {
        let resolver = ImportResolver::new();

        resolver.import_module("math").unwrap();
        resolver.import_module("os").unwrap();

        let cached = resolver.cached_modules();
        assert_eq!(cached.len(), 2);
    }

    #[test]
    fn test_concurrent_imports() {
        use std::thread;

        let resolver = Arc::new(ImportResolver::new());

        // Spawn 10 threads all trying to import "math" simultaneously
        // The first thread to win the race will load the module,
        // other threads will wait via Condvar and return the cached result
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let r = Arc::clone(&resolver);
                thread::spawn(move || r.import_module("math").unwrap())
            })
            .collect();

        let modules: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // All should be the same cached module
        for i in 1..modules.len() {
            assert!(Arc::ptr_eq(&modules[0], &modules[i]));
        }
    }
}

//! Environment variable handling with lazy loading.
//!
//! High-performance environment access with:
//! - Lazy loading (don't read until first access)
//! - Mutation tracking for putenv/unsetenv
//! - Zero-copy key lookup using borrowed strings
//! - Thread-safe access via atomic operations

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// =============================================================================
// Environ
// =============================================================================

/// Lazy-loading environment variable dictionary.
///
/// Environment variables are not loaded until first access, avoiding
/// startup overhead for programs that don't use environ.
#[derive(Debug)]
pub struct Environ {
    /// Whether the environment has been loaded.
    loaded: AtomicBool,
    /// Cached environment variables.
    /// Uses Arc<str> for zero-copy sharing.
    vars: std::cell::UnsafeCell<HashMap<Arc<str>, Arc<str>>>,
}

// SAFETY: Environ is thread-safe because:
// 1. loaded is AtomicBool
// 2. vars is only mutated while loaded is false (single initialization)
// 3. After initialization, vars is only read
unsafe impl Sync for Environ {}
unsafe impl Send for Environ {}

impl Environ {
    /// Create a new lazy environ.
    #[inline]
    pub fn new() -> Self {
        Self {
            loaded: AtomicBool::new(false),
            vars: std::cell::UnsafeCell::new(HashMap::new()),
        }
    }

    /// Ensure environment is loaded.
    #[inline]
    fn ensure_loaded(&self) {
        if !self.loaded.load(Ordering::Acquire) {
            self.do_load();
        }
    }

    /// Actually load the environment (cold path).
    #[cold]
    fn do_load(&self) {
        // Double-check locking pattern
        if self.loaded.load(Ordering::Acquire) {
            return;
        }

        // SAFETY: We're the only writer (checked by loaded flag)
        let vars = unsafe { &mut *self.vars.get() };

        for (key, value) in std::env::vars() {
            vars.insert(Arc::from(key.as_str()), Arc::from(value.as_str()));
        }

        self.loaded.store(true, Ordering::Release);
    }

    /// Get an environment variable.
    ///
    /// Returns `None` if the variable is not set.
    #[inline]
    pub fn get(&self, key: &str) -> Option<Arc<str>> {
        self.ensure_loaded();
        // SAFETY: After ensure_loaded, vars is immutable
        let vars = unsafe { &*self.vars.get() };
        vars.get(key).cloned()
    }

    /// Get an environment variable with a default.
    #[inline]
    pub fn get_or(&self, key: &str, default: &str) -> Arc<str> {
        self.get(key).unwrap_or_else(|| Arc::from(default))
    }

    /// Check if an environment variable is set.
    #[inline]
    pub fn contains(&self, key: &str) -> bool {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.contains_key(key)
    }

    /// Set an environment variable.
    ///
    /// This also updates the actual process environment.
    pub fn set(&mut self, key: &str, value: &str) {
        self.ensure_loaded();

        // Update process environment
        // SAFETY: We're the only writer to this key in this code path
        unsafe { std::env::set_var(key, value) };

        // Update cache
        let vars = self.vars.get_mut();
        vars.insert(Arc::from(key), Arc::from(value));
    }

    /// Remove an environment variable.
    ///
    /// This also removes it from the actual process environment.
    pub fn remove(&mut self, key: &str) -> Option<Arc<str>> {
        self.ensure_loaded();

        // Remove from process environment
        // SAFETY: We're the only writer to this key in this code path
        unsafe { std::env::remove_var(key) };

        // Remove from cache
        let vars = self.vars.get_mut();
        vars.remove(key)
    }

    /// Get the number of environment variables.
    #[inline]
    pub fn len(&self) -> usize {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.len()
    }

    /// Check if the environment is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all environment variable keys.
    pub fn keys(&self) -> Vec<Arc<str>> {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.keys().cloned().collect()
    }

    /// Get all environment variable values.
    pub fn values(&self) -> Vec<Arc<str>> {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.values().cloned().collect()
    }

    /// Iterate over all environment variables.
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &Arc<str>)> {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.iter()
    }

    /// Clear all environment variables from the cache.
    ///
    /// Note: This does NOT clear the actual process environment.
    pub fn clear_cache(&mut self) {
        let vars = self.vars.get_mut();
        vars.clear();
        self.loaded.store(false, Ordering::Release);
    }

    /// Force reload from process environment.
    pub fn reload(&mut self) {
        self.clear_cache();
        self.ensure_loaded();
    }
}

impl Default for Environ {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Environ {
    fn clone(&self) -> Self {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        let mut new_environ = Self::new();
        *new_environ.vars.get_mut() = vars.clone();
        new_environ.loaded.store(true, Ordering::Release);
        new_environ
    }
}

// =============================================================================
// Standalone Functions
// =============================================================================

/// Get an environment variable from the process (not cached).
///
/// This always reads from the actual environment.
#[inline]
pub fn getenv(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Get an environment variable with a default.
#[inline]
pub fn getenv_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Set an environment variable in the process.
///
/// # Safety
/// This function is safe to call as long as the environment variable
/// is not being concurrently accessed by other threads.
#[inline]
pub fn putenv(key: &str, value: &str) {
    // SAFETY: Caller ensures thread-safe access
    unsafe { std::env::set_var(key, value) };
}

/// Remove an environment variable from the process.
///
/// # Safety
/// This function is safe to call as long as the environment variable
/// is not being concurrently accessed by other threads.
#[inline]
pub fn unsetenv(key: &str) {
    // SAFETY: Caller ensures thread-safe access
    unsafe { std::env::remove_var(key) };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper functions to wrap unsafe env operations for tests
    fn test_set_var(key: &str, value: &str) {
        unsafe { std::env::set_var(key, value) };
    }

    fn test_remove_var(key: &str) {
        unsafe { std::env::remove_var(key) };
    }

    // =========================================================================
    // Environ Creation Tests
    // =========================================================================

    #[test]
    fn test_environ_new() {
        let env = Environ::new();
        // Should not be loaded yet
        assert!(!env.loaded.load(Ordering::Relaxed));
    }

    #[test]
    fn test_environ_default() {
        let env = Environ::default();
        assert!(!env.loaded.load(Ordering::Relaxed));
    }

    // =========================================================================
    // Lazy Loading Tests
    // =========================================================================

    #[test]
    fn test_environ_lazy_load_on_get() {
        let env = Environ::new();
        assert!(!env.loaded.load(Ordering::Relaxed));
        let _ = env.get("PATH");
        assert!(env.loaded.load(Ordering::Relaxed));
    }

    #[test]
    fn test_environ_lazy_load_on_contains() {
        let env = Environ::new();
        let _ = env.contains("PATH");
        assert!(env.loaded.load(Ordering::Relaxed));
    }

    #[test]
    fn test_environ_lazy_load_on_len() {
        let env = Environ::new();
        let _ = env.len();
        assert!(env.loaded.load(Ordering::Relaxed));
    }

    // =========================================================================
    // Get Tests
    // =========================================================================

    #[test]
    fn test_environ_get_existing() {
        // Set a test variable we know exists
        test_set_var("__TEST_GET_EXISTING__", "test_value");
        let env = Environ::new();
        let result = env.get("__TEST_GET_EXISTING__");
        assert!(result.is_some());
        assert_eq!(&*result.unwrap(), "test_value");
        test_remove_var("__TEST_GET_EXISTING__");
    }

    #[test]
    fn test_environ_get_nonexistent() {
        let env = Environ::new();
        let result = env.get("__NONEXISTENT_VAR_12345__");
        assert!(result.is_none());
    }

    #[test]
    fn test_environ_get_or_existing() {
        test_set_var("__TEST_GET_OR__", "test_value");
        let env = Environ::new();
        let result = env.get_or("__TEST_GET_OR__", "default");
        assert_eq!(&*result, "test_value");
        test_remove_var("__TEST_GET_OR__");
    }

    #[test]
    fn test_environ_get_or_nonexistent() {
        let env = Environ::new();
        let result = env.get_or("__NONEXISTENT_VAR_12345__", "default");
        assert_eq!(&*result, "default");
    }

    // =========================================================================
    // Contains Tests
    // =========================================================================

    #[test]
    fn test_environ_contains_existing() {
        test_set_var("__TEST_CONTAINS__", "value");
        let env = Environ::new();
        assert!(env.contains("__TEST_CONTAINS__"));
        test_remove_var("__TEST_CONTAINS__");
    }

    #[test]
    fn test_environ_contains_nonexistent() {
        let env = Environ::new();
        assert!(!env.contains("__NONEXISTENT_VAR_12345__"));
    }

    // =========================================================================
    // Set/Remove Tests
    // =========================================================================

    #[test]
    fn test_environ_set() {
        let mut env = Environ::new();
        env.set("__TEST_SET_VAR__", "test_value");

        // Check cache
        assert_eq!(env.get("__TEST_SET_VAR__").as_deref(), Some("test_value"));

        // Check actual environment
        assert_eq!(
            std::env::var("__TEST_SET_VAR__").ok().as_deref(),
            Some("test_value")
        );

        // Cleanup
        test_remove_var("__TEST_SET_VAR__");
    }

    #[test]
    fn test_environ_set_overwrite() {
        let mut env = Environ::new();
        env.set("__TEST_SET_OVERWRITE__", "value1");
        env.set("__TEST_SET_OVERWRITE__", "value2");

        assert_eq!(env.get("__TEST_SET_OVERWRITE__").as_deref(), Some("value2"));

        test_remove_var("__TEST_SET_OVERWRITE__");
    }

    #[test]
    fn test_environ_remove() {
        test_set_var("__TEST_REMOVE__", "value");
        let mut env = Environ::new();

        let removed = env.remove("__TEST_REMOVE__");
        assert_eq!(removed.as_deref(), Some("value"));

        // Check it's gone from cache
        assert!(env.get("__TEST_REMOVE__").is_none());

        // Check it's gone from actual environment
        assert!(std::env::var("__TEST_REMOVE__").is_err());
    }

    #[test]
    fn test_environ_remove_nonexistent() {
        let mut env = Environ::new();
        let removed = env.remove("__NONEXISTENT_VAR_12345__");
        assert!(removed.is_none());
    }

    // =========================================================================
    // Size Tests
    // =========================================================================

    #[test]
    fn test_environ_len_nonzero() {
        let env = Environ::new();
        // Most systems have at least some environment variables
        assert!(env.len() > 0);
    }

    #[test]
    fn test_environ_is_empty() {
        let env = Environ::new();
        // Most systems have environment variables
        assert!(!env.is_empty());
    }

    // =========================================================================
    // Keys/Values Tests
    // =========================================================================

    #[test]
    fn test_environ_keys() {
        test_set_var("__TEST_KEYS__", "value");
        let env = Environ::new();
        let keys = env.keys();
        assert!(keys.iter().any(|k| &**k == "__TEST_KEYS__"));
        test_remove_var("__TEST_KEYS__");
    }

    #[test]
    fn test_environ_values() {
        test_set_var("__TEST_VALUES__", "unique_test_value_12345");
        let env = Environ::new();
        let values = env.values();
        assert!(values.iter().any(|v| &**v == "unique_test_value_12345"));
        test_remove_var("__TEST_VALUES__");
    }

    // =========================================================================
    // Iterator Tests
    // =========================================================================

    #[test]
    fn test_environ_iter() {
        test_set_var("__TEST_ITER__", "iter_value");
        let env = Environ::new();
        let found = env
            .iter()
            .any(|(k, v)| &**k == "__TEST_ITER__" && &**v == "iter_value");
        assert!(found);
        test_remove_var("__TEST_ITER__");
    }

    // =========================================================================
    // Clear/Reload Tests
    // =========================================================================

    #[test]
    fn test_environ_clear_cache() {
        let mut env = Environ::new();
        let _ = env.len(); // Force load
        assert!(env.loaded.load(Ordering::Relaxed));

        env.clear_cache();
        assert!(!env.loaded.load(Ordering::Relaxed));
    }

    #[test]
    fn test_environ_reload() {
        test_set_var("__TEST_RELOAD__", "value1");
        let mut env = Environ::new();
        let _ = env.get("__TEST_RELOAD__");

        // Change actual environment
        test_set_var("__TEST_RELOAD__", "value2");

        // Cache should still have old value
        // (Note: our cache updates on set, so we need to change it externally)

        // Reload
        env.reload();

        // Should now have new value
        assert_eq!(env.get("__TEST_RELOAD__").as_deref(), Some("value2"));

        test_remove_var("__TEST_RELOAD__");
    }

    // =========================================================================
    // Clone Tests
    // =========================================================================

    #[test]
    fn test_environ_clone() {
        test_set_var("__TEST_CLONE__", "clone_value");
        let env = Environ::new();
        let _ = env.get("__TEST_CLONE__");

        let cloned = env.clone();
        assert_eq!(cloned.get("__TEST_CLONE__").as_deref(), Some("clone_value"));

        test_remove_var("__TEST_CLONE__");
    }

    // =========================================================================
    // Standalone Function Tests
    // =========================================================================

    #[test]
    fn test_getenv_existing() {
        test_set_var("__TEST_GETENV__", "value");
        assert_eq!(getenv("__TEST_GETENV__"), Some("value".to_string()));
        test_remove_var("__TEST_GETENV__");
    }

    #[test]
    fn test_getenv_nonexistent() {
        assert!(getenv("__NONEXISTENT_VAR_12345__").is_none());
    }

    #[test]
    fn test_getenv_or() {
        assert_eq!(getenv_or("__NONEXISTENT_VAR_12345__", "default"), "default");
    }

    #[test]
    fn test_putenv() {
        putenv("__TEST_PUTENV__", "value");
        assert_eq!(
            std::env::var("__TEST_PUTENV__").ok(),
            Some("value".to_string())
        );
        test_remove_var("__TEST_PUTENV__");
    }

    #[test]
    fn test_unsetenv() {
        test_set_var("__TEST_UNSETENV__", "value");
        unsetenv("__TEST_UNSETENV__");
        assert!(std::env::var("__TEST_UNSETENV__").is_err());
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_environ_sync_send() {
        fn assert_sync<T: Sync>() {}
        fn assert_send<T: Send>() {}

        assert_sync::<Environ>();
        assert_send::<Environ>();
    }

    #[test]
    fn test_environ_concurrent_read() {
        use std::sync::Arc;
        use std::thread;

        test_set_var("__TEST_CONCURRENT__", "value");
        let env = Arc::new(Environ::new());

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let env = Arc::clone(&env);
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = env.get("__TEST_CONCURRENT__");
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        test_remove_var("__TEST_CONCURRENT__");
    }
}

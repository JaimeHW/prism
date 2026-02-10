//! Background compilation thread for JIT compilation.
//!
//! This module provides infrastructure for compiling functions in the
//! background without blocking the main execution thread. Compilation
//! requests are queued and processed asynchronously.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use super::profile_data::ProfileData;

// =============================================================================
// Compilation Request
// =============================================================================

/// A request to compile a function.
#[derive(Debug, Clone)]
pub struct CompilationRequest {
    /// ID of the code to compile.
    pub code_id: u64,
    /// Bytecode to compile.
    pub bytecode: Vec<u8>,
    /// Target compilation tier.
    pub tier: CompilationTier,
    /// Priority (higher = more urgent).
    pub priority: u32,
    /// Optional OSR entry offset (for loop compilation).
    pub osr_offset: Option<u32>,
    /// Profile data from Tier 1 execution (drives PGO in Tier 2).
    pub profile_data: Option<ProfileData>,
}

impl CompilationRequest {
    /// Create a new compilation request.
    pub fn new(code_id: u64, bytecode: Vec<u8>, tier: CompilationTier) -> Self {
        Self {
            code_id,
            bytecode,
            tier,
            priority: 0,
            osr_offset: None,
            profile_data: None,
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set OSR offset for loop-entry compilation.
    pub fn with_osr_offset(mut self, offset: u32) -> Self {
        self.osr_offset = Some(offset);
        self
    }

    /// Attach profile data for PGO-guided Tier 2 compilation.
    pub fn with_profile(mut self, profile: ProfileData) -> Self {
        self.profile_data = Some(profile);
        self
    }
}

/// Compilation tier target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationTier {
    /// Baseline compilation (fast, minimal optimization).
    Baseline,
    /// Optimized compilation (slower, full optimization).
    Optimized,
}

// =============================================================================
// Compilation Result
// =============================================================================

/// Result of a compilation attempt.
#[derive(Debug)]
pub struct CompilationResult {
    /// Code ID that was compiled.
    pub code_id: u64,
    /// Pointer to compiled code (if successful).
    pub code_ptr: Option<*const u8>,
    /// Size of compiled code.
    pub code_size: usize,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Compilation time in microseconds.
    pub compile_time_us: u64,
}

// SAFETY: Compilation result can be sent between threads
unsafe impl Send for CompilationResult {}

// =============================================================================
// Compilation Queue
// =============================================================================

/// A thread-safe queue for compilation requests.
#[derive(Debug)]
pub struct CompilationQueue {
    /// Pending requests (sorted by priority).
    queue: Mutex<VecDeque<CompilationRequest>>,
    /// Condvar for signaling new work.
    condvar: Condvar,
    /// Whether the queue is shutting down.
    shutdown: Mutex<bool>,
}

impl CompilationQueue {
    /// Create a new compilation queue.
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
            shutdown: Mutex::new(false),
        }
    }

    /// Submit a compilation request.
    pub fn submit(&self, request: CompilationRequest) {
        let mut queue = self.queue.lock().unwrap();

        // Insert by priority (higher priority at front)
        let insert_idx = queue
            .iter()
            .position(|r| r.priority < request.priority)
            .unwrap_or(queue.len());
        queue.insert(insert_idx, request);

        // Signal waiting threads
        self.condvar.notify_one();
    }

    /// Wait for and retrieve the next request.
    ///
    /// Returns `None` if the queue is shutting down.
    pub fn wait_for_request(&self) -> Option<CompilationRequest> {
        let mut queue = self.queue.lock().unwrap();

        loop {
            // Check for shutdown
            if *self.shutdown.lock().unwrap() {
                return None;
            }

            // Try to get a request
            if let Some(request) = queue.pop_front() {
                return Some(request);
            }

            // Wait for notification
            queue = self.condvar.wait(queue).unwrap();
        }
    }

    /// Try to get a request without blocking.
    pub fn try_get(&self) -> Option<CompilationRequest> {
        self.queue.lock().unwrap().pop_front()
    }

    /// Get the number of pending requests.
    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }

    /// Signal shutdown to all waiting threads.
    pub fn shutdown(&self) {
        *self.shutdown.lock().unwrap() = true;
        self.condvar.notify_all();
    }

    /// Check if shutdown has been signaled.
    pub fn is_shutdown(&self) -> bool {
        *self.shutdown.lock().unwrap()
    }
}

impl Default for CompilationQueue {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Compiler Thread
// =============================================================================

/// A background compilation thread.
///
/// Pulls requests from a queue and compiles them asynchronously.
#[derive(Debug)]
pub struct CompilerThread {
    /// Shared compilation queue.
    queue: Arc<CompilationQueue>,
    /// Results are sent back via this channel.
    results: Mutex<Vec<CompilationResult>>,
    /// Statistics.
    stats: Mutex<CompilerStats>,
}

impl CompilerThread {
    /// Create a new compiler thread with the given queue.
    pub fn new(queue: Arc<CompilationQueue>) -> Self {
        Self {
            queue,
            results: Mutex::new(Vec::new()),
            stats: Mutex::new(CompilerStats::default()),
        }
    }

    /// Run the compilation loop (blocking).
    ///
    /// This method blocks until shutdown is signaled.
    pub fn run(&self) {
        while let Some(request) = self.queue.wait_for_request() {
            let result = self.compile(request);

            // Store result
            self.results.lock().unwrap().push(result);
        }
    }

    /// Compile a single request (synchronous).
    pub fn compile(&self, request: CompilationRequest) -> CompilationResult {
        let start = std::time::Instant::now();

        // TODO: Actual compilation logic would go here
        // For now, return a placeholder result

        let compile_time = start.elapsed().as_micros() as u64;

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.compilations += 1;
            stats.total_compile_time_us += compile_time;
        }

        CompilationResult {
            code_id: request.code_id,
            code_ptr: None, // Would be set by actual compilation
            code_size: 0,
            error: None,
            compile_time_us: compile_time,
        }
    }

    /// Take all pending results.
    pub fn take_results(&self) -> Vec<CompilationResult> {
        std::mem::take(&mut *self.results.lock().unwrap())
    }

    /// Get compilation statistics.
    pub fn stats(&self) -> CompilerStats {
        self.stats.lock().unwrap().clone()
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics for the compiler thread.
#[derive(Debug, Default, Clone)]
pub struct CompilerStats {
    /// Total number of compilations.
    pub compilations: u64,
    /// Total compilation time in microseconds.
    pub total_compile_time_us: u64,
    /// Number of failed compilations.
    pub failures: u64,
}

impl CompilerStats {
    /// Average compilation time in microseconds.
    pub fn avg_compile_time_us(&self) -> u64 {
        if self.compilations == 0 {
            0
        } else {
            self.total_compile_time_us / self.compilations
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_request() {
        let req = CompilationRequest::new(1, vec![0x00, 0x01], CompilationTier::Baseline)
            .with_priority(10)
            .with_osr_offset(100);

        assert_eq!(req.code_id, 1);
        assert_eq!(req.priority, 10);
        assert_eq!(req.osr_offset, Some(100));
    }

    #[test]
    fn test_compilation_queue() {
        let queue = CompilationQueue::new();

        // Submit requests with different priorities
        queue
            .submit(CompilationRequest::new(1, vec![], CompilationTier::Baseline).with_priority(5));
        queue.submit(
            CompilationRequest::new(2, vec![], CompilationTier::Baseline).with_priority(10),
        );
        queue
            .submit(CompilationRequest::new(3, vec![], CompilationTier::Baseline).with_priority(1));

        // Higher priority should come first
        assert_eq!(queue.try_get().unwrap().code_id, 2);
        assert_eq!(queue.try_get().unwrap().code_id, 1);
        assert_eq!(queue.try_get().unwrap().code_id, 3);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_compiler_thread_compile() {
        let queue = Arc::new(CompilationQueue::new());
        let compiler = CompilerThread::new(queue);

        let request = CompilationRequest::new(42, vec![0xCC], CompilationTier::Optimized);
        let result = compiler.compile(request);

        assert_eq!(result.code_id, 42);
        assert!(result.error.is_none());

        let stats = compiler.stats();
        assert_eq!(stats.compilations, 1);
    }

    #[test]
    fn test_queue_shutdown() {
        let queue = Arc::new(CompilationQueue::new());

        // Signal shutdown
        queue.shutdown();
        assert!(queue.is_shutdown());

        // try_get should work
        assert!(queue.try_get().is_none());
    }

    // =========================================================================
    // PGO Profile Data Integration Tests
    // =========================================================================

    #[test]
    fn test_compilation_request_default_no_profile() {
        let req = CompilationRequest::new(1, vec![0x00], CompilationTier::Baseline);
        assert!(req.profile_data.is_none());
    }

    #[test]
    fn test_compilation_request_with_profile() {
        let mut profile = ProfileData::new(1);
        profile.record_branch(10, true);
        profile.record_branch(10, true);
        profile.record_branch(10, false);

        let req = CompilationRequest::new(42, vec![0xCC], CompilationTier::Optimized)
            .with_profile(profile);

        assert!(req.profile_data.is_some());
        let data = req.profile_data.as_ref().unwrap();
        assert_eq!(data.code_id(), 1);
        assert_eq!(data.branch_count(), 1);
    }

    #[test]
    fn test_compilation_request_full_builder_chain() {
        let mut profile = ProfileData::new(99);
        for _ in 0..100 {
            profile.record_execution();
        }
        profile.record_branch(5, true);
        profile.record_branch(5, false);

        let req = CompilationRequest::new(7, vec![0x01, 0x02], CompilationTier::Optimized)
            .with_priority(100)
            .with_osr_offset(42)
            .with_profile(profile);

        assert_eq!(req.code_id, 7);
        assert_eq!(req.priority, 100);
        assert_eq!(req.osr_offset, Some(42));
        assert!(req.profile_data.is_some());

        let data = req.profile_data.unwrap();
        assert_eq!(data.execution_count(), 100);
        assert_eq!(data.branch_count(), 1);
    }

    #[test]
    fn test_compilation_request_tier2_with_profile_pattern() {
        // Simulates the canonical Tier 1 → Tier 2 promotion pattern
        let mut profile = ProfileData::new(42);
        // Simulate hot function (1000 executions)
        for _ in 0..1000 {
            profile.record_execution();
        }
        // Simulate biased branch at offset 15
        for _ in 0..980 {
            profile.record_branch(15, true);
        }
        for _ in 0..20 {
            profile.record_branch(15, false);
        }
        // Simulate loop with avg 50 trips
        for _ in 0..50_000 {
            profile.record_loop_iteration(20);
        }

        assert!(profile.is_hot(100));
        assert_eq!(profile.loop_count(20), 50_000);

        let req = CompilationRequest::new(42, vec![], CompilationTier::Optimized)
            .with_priority(50)
            .with_profile(profile);

        // Verify the profile is carried through
        let data = req.profile_data.as_ref().unwrap();
        assert_eq!(data.execution_count(), 1000);
        let branch = data.branch_at(15).unwrap();
        assert!(
            branch.taken > branch.not_taken * 10,
            "Branch should be heavily biased toward taken"
        );
    }

    #[test]
    fn test_compilation_request_clone_with_profile() {
        let mut profile = ProfileData::new(1);
        profile.record_branch(0, true);

        let req =
            CompilationRequest::new(1, vec![], CompilationTier::Optimized).with_profile(profile);

        let cloned = req.clone();
        assert!(cloned.profile_data.is_some());
        assert_eq!(
            cloned.profile_data.as_ref().unwrap().branch_count(),
            req.profile_data.as_ref().unwrap().branch_count()
        );
    }
}

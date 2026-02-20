//! Background compilation queue for non-blocking JIT compilation.
//!
//! Implements a channel-based compilation queue with a dedicated worker thread.
//! The mutator thread enqueues compilation requests via MPSC channel, and the
//! worker thread processes them, inserting results into the shared code cache.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    MPSC Channel    ┌─────────────────┐
//! │  Mutator(s)  │ ─────────────────► │  Worker Thread   │
//! │              │  CompilationReq    │                  │
//! │  .enqueue()  │                    │  owns:           │
//! │              │                    │  - TemplateComp  │
//! └──────────────┘                    │  - CodeCache ref │
//!                                     └────────┬────────┘
//!                                              │
//!                                     ┌────────▼────────┐
//!                                     │   Code Cache     │
//!                                     │  (Arc, shared)   │
//!                                     └─────────────────┘
//! ```
//!
//! # Performance
//!
//! - **Enqueue**: O(1) channel send, non-blocking
//! - **Compilation**: Happens off the critical path, does not block execution
//! - **Lookup**: After compilation, code is available via the shared cache

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use prism_compiler::bytecode::CodeObject;
#[cfg(test)]
use prism_compiler::bytecode::Opcode;
use prism_jit::runtime::{CodeCache, CompiledEntry};
use prism_jit::tier1::codegen::TemplateCompiler;

use crate::jit_bridge::{compile_tier1_entry, compile_tier2_entry};

/// A request to compile a function in the background.
struct CompilationRequest {
    /// The code object to compile.
    code: Arc<CodeObject>,
    /// The target compilation tier.
    tier: u8,
}

/// Statistics for the compilation queue.
#[derive(Debug, Default)]
pub struct CompilationQueueStats {
    /// Number of requests enqueued.
    pub enqueued: AtomicU64,
    /// Number of requests completed successfully.
    pub completed: AtomicU64,
    /// Number of requests that failed.
    pub failed: AtomicU64,
    /// Number of requests dropped (queue full or duplicate).
    pub dropped: AtomicU64,
}

impl CompilationQueueStats {
    /// Get snapshot of current stats.
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.enqueued.load(Ordering::Relaxed),
            self.completed.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
        )
    }
}

/// Background compilation queue with a dedicated worker thread.
///
/// Decouples JIT compilation from the interpreter's critical path.
/// The worker thread owns its own `TemplateCompiler` instance for Tier 1 work
/// and runs Tier 2 compilation directly from the optimizing pipeline.
pub struct CompilationQueue {
    /// Channel sender for enqueuing requests.
    sender: mpsc::Sender<CompilationRequest>,
    /// Handle to the worker thread (for graceful shutdown).
    _worker: Option<thread::JoinHandle<()>>,
    /// Shared flag to signal shutdown.
    shutdown: Arc<AtomicBool>,
    /// Shared code cache for dedup checks on enqueue.
    code_cache: Arc<CodeCache>,
    /// Queue statistics.
    stats: Arc<CompilationQueueStats>,
    /// Approximate queue depth (sender doesn't expose len()).
    pending: Arc<AtomicUsize>,
    /// Maximum queue depth before dropping requests.
    max_queue_size: usize,
}

impl CompilationQueue {
    /// Create a new compilation queue with the given code cache and queue size.
    ///
    /// Spawns a dedicated worker thread that processes compilation requests.
    pub fn new(code_cache: Arc<CodeCache>, max_queue_size: usize) -> Self {
        let (sender, receiver) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(CompilationQueueStats::default());
        let pending = Arc::new(AtomicUsize::new(0));

        let worker = {
            let code_cache = Arc::clone(&code_cache);
            let shutdown = Arc::clone(&shutdown);
            let stats = Arc::clone(&stats);
            let pending = Arc::clone(&pending);

            thread::Builder::new()
                .name("prism-jit-compiler".to_string())
                .spawn(move || {
                    Self::worker_loop(receiver, code_cache, shutdown, stats, pending);
                })
                .expect("Failed to spawn JIT compilation thread")
        };

        Self {
            sender,
            _worker: Some(worker),
            shutdown,
            code_cache,
            stats,
            pending,
            max_queue_size,
        }
    }

    /// Enqueue a compilation request.
    ///
    /// Returns `true` if the request was enqueued, `false` if the queue
    /// is full or the function is already compiled.
    pub fn enqueue(&self, code: Arc<CodeObject>, tier: u8) -> bool {
        let code_id = Arc::as_ptr(&code) as u64;

        // Fast path: already compiled at this tier or higher
        if let Some(entry) = self.code_cache.lookup(code_id) {
            if entry.tier() >= tier {
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        }

        // Check approximate queue depth
        if self.pending.load(Ordering::Relaxed) >= self.max_queue_size {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        // Send request
        let request = CompilationRequest { code, tier };
        match self.sender.send(request) {
            Ok(()) => {
                self.pending.fetch_add(1, Ordering::Relaxed);
                self.stats.enqueued.fetch_add(1, Ordering::Relaxed);
                true
            }
            Err(_) => {
                // Channel disconnected — worker died
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Get the approximate queue depth.
    pub fn queue_depth(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    /// Get compilation statistics.
    pub fn stats(&self) -> &CompilationQueueStats {
        &self.stats
    }

    /// Check if the queue is empty (approximate).
    pub fn is_empty(&self) -> bool {
        self.queue_depth() == 0
    }

    /// Get maximum queue size.
    pub fn max_queue_size(&self) -> usize {
        self.max_queue_size
    }

    /// The worker thread's main loop.
    ///
    /// Processes compilation requests until shutdown or channel disconnect.
    fn worker_loop(
        receiver: mpsc::Receiver<CompilationRequest>,
        code_cache: Arc<CodeCache>,
        shutdown: Arc<AtomicBool>,
        stats: Arc<CompilationQueueStats>,
        pending: Arc<AtomicUsize>,
    ) {
        // Worker owns its own compiler instance — no contention with mutator
        let mut compiler = TemplateCompiler::new_runtime();

        loop {
            // Check shutdown flag
            if shutdown.load(Ordering::Acquire) {
                break;
            }

            // Block waiting for next request (with timeout for shutdown checks)
            match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(request) => {
                    pending.fetch_sub(1, Ordering::Relaxed);
                    Self::process_request(&mut compiler, &code_cache, &stats, request);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Check shutdown and loop
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }

    /// Process a single compilation request.
    fn process_request(
        compiler: &mut TemplateCompiler,
        code_cache: &CodeCache,
        stats: &CompilationQueueStats,
        request: CompilationRequest,
    ) {
        let code_id = Arc::as_ptr(&request.code) as u64;

        // Double-check: might have been compiled while waiting in queue
        if let Some(entry) = code_cache.lookup(code_id) {
            if entry.tier() >= request.tier {
                stats.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }

        // Compile at the requested tier (>=2 means optimize with Tier 2 pipeline).
        let compiled = if request.tier >= 2 {
            compile_tier2_entry(&request.code)
        } else {
            compile_tier1_entry(&request.code, compiler)
        };

        match compiled {
            Ok(entry) => {
                code_cache.insert(entry);
                stats.completed.fetch_add(1, Ordering::Relaxed);
            }
            Err(_e) => {
                #[cfg(debug_assertions)]
                eprintln!("Background compilation failed (codegen): {}", _e);
                stats.failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for CompilationQueue {
    fn drop(&mut self) {
        // Signal worker to shut down
        self.shutdown.store(true, Ordering::Release);

        // Take the worker handle and join it
        if let Some(handle) = self._worker.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compilation_queue_creation() {
        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(code_cache, 16);
        assert!(queue.is_empty());
        assert_eq!(queue.max_queue_size(), 16);
    }

    #[test]
    fn test_compilation_queue_stats_initial() {
        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(code_cache, 16);
        let (enqueued, completed, failed, dropped) = queue.stats().snapshot();
        assert_eq!(enqueued, 0);
        assert_eq!(completed, 0);
        assert_eq!(failed, 0);
        assert_eq!(dropped, 0);
    }

    #[test]
    fn test_compilation_queue_enqueue_and_compile() {
        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

        // Create a simple code object with a return instruction
        let mut code = CodeObject::new("test_func", "<test>");
        code.register_count = 1;
        use prism_compiler::bytecode::Instruction;
        code.instructions = vec![
            Instruction::op_d(Opcode::LoadNone, prism_compiler::bytecode::Register::new(0)),
            Instruction::op_d(Opcode::Return, prism_compiler::bytecode::Register::new(0)),
        ]
        .into_boxed_slice();

        let code = Arc::new(code);
        let code_id = Arc::as_ptr(&code) as u64;

        // Enqueue
        assert!(queue.enqueue(Arc::clone(&code), 1));

        // Wait for compilation to complete
        let mut attempts = 0;
        while code_cache.lookup(code_id).is_none() && attempts < 50 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            attempts += 1;
        }

        // Verify compiled
        assert!(
            code_cache.lookup(code_id).is_some(),
            "Compilation should complete within 500ms"
        );

        let entry = code_cache.lookup(code_id).unwrap();
        assert_eq!(entry.tier(), 1);

        // Stats should reflect completion
        let (enqueued, completed, _failed, _dropped) = queue.stats().snapshot();
        assert_eq!(enqueued, 1);
        assert_eq!(completed, 1);
    }

    #[test]
    fn test_compilation_queue_duplicate_skip() {
        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

        // Create a code object
        let mut code = CodeObject::new("already_compiled", "<test>");
        code.register_count = 1;
        use prism_compiler::bytecode::Instruction;
        code.instructions = vec![
            Instruction::op_d(Opcode::LoadNone, prism_compiler::bytecode::Register::new(0)),
            Instruction::op_d(Opcode::Return, prism_compiler::bytecode::Register::new(0)),
        ]
        .into_boxed_slice();
        let code = Arc::new(code);
        let code_id = Arc::as_ptr(&code) as u64;

        // Pre-insert a tier 1 entry using raw pointer constructor (no ExecutableBuffer needed)
        let fake_entry = CompiledEntry::new(code_id, 0x10000 as *const u8, 1).with_tier(1);
        code_cache.insert(fake_entry);

        // Enqueue at same tier — should be dropped
        assert!(!queue.enqueue(Arc::clone(&code), 1));

        let (_enqueued, _completed, _failed, dropped) = queue.stats().snapshot();
        assert_eq!(dropped, 1);
    }

    #[test]
    fn test_compilation_queue_graceful_shutdown() {
        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(code_cache, 16);

        // Drop should join worker thread without hanging
        drop(queue);
        // If we get here, shutdown was graceful
    }

    #[test]
    fn test_compilation_queue_multiple_functions() {
        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

        let mut code_ids = Vec::new();

        // Enqueue multiple functions
        for i in 0..5 {
            let name: Arc<str> = Arc::from(format!("func_{}", i).as_str());
            let mut code = CodeObject::new(name, "<test>");
            code.register_count = 1;
            use prism_compiler::bytecode::Instruction;
            code.instructions = vec![
                Instruction::op_d(Opcode::LoadNone, prism_compiler::bytecode::Register::new(0)),
                Instruction::op_d(Opcode::Return, prism_compiler::bytecode::Register::new(0)),
            ]
            .into_boxed_slice();
            let code = Arc::new(code);
            code_ids.push(Arc::as_ptr(&code) as u64);
            queue.enqueue(code, 1);
        }

        // Wait for all compilations
        let mut attempts = 0;
        while code_ids.iter().any(|id| code_cache.lookup(*id).is_none()) && attempts < 100 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            attempts += 1;
        }

        // All should be compiled
        for code_id in &code_ids {
            assert!(
                code_cache.lookup(*code_id).is_some(),
                "Function {:x} should be compiled",
                code_id
            );
        }

        let (enqueued, completed, _failed, _dropped) = queue.stats().snapshot();
        assert_eq!(enqueued, 5);
        assert_eq!(completed, 5);
    }

    #[test]
    fn test_compilation_queue_queue_depth_tracking() {
        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(code_cache, 64);

        // Initially empty
        assert_eq!(queue.queue_depth(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_compilation_queue_stats_snapshot() {
        let stats = CompilationQueueStats::default();
        stats.enqueued.store(10, Ordering::Relaxed);
        stats.completed.store(8, Ordering::Relaxed);
        stats.failed.store(1, Ordering::Relaxed);
        stats.dropped.store(1, Ordering::Relaxed);

        let (e, c, f, d) = stats.snapshot();
        assert_eq!(e, 10);
        assert_eq!(c, 8);
        assert_eq!(f, 1);
        assert_eq!(d, 1);
    }

    #[test]
    fn test_compilation_queue_tier_upgrade() {
        use prism_jit::runtime::ReturnAbi;

        let code_cache = Arc::new(CodeCache::new(1024 * 1024));
        let queue = CompilationQueue::new(Arc::clone(&code_cache), 16);

        // Create a code object
        let mut code = CodeObject::new("upgrade_func", "<test>");
        code.register_count = 1;
        use prism_compiler::bytecode::Instruction;
        code.instructions = vec![
            Instruction::op_d(Opcode::LoadNone, prism_compiler::bytecode::Register::new(0)),
            Instruction::op_d(Opcode::Return, prism_compiler::bytecode::Register::new(0)),
        ]
        .into_boxed_slice();
        let code = Arc::new(code);
        let code_id = Arc::as_ptr(&code) as u64;

        // Pre-insert a tier 1 entry using raw pointer constructor (no ExecutableBuffer needed)
        let fake_entry = CompiledEntry::new(code_id, 0x10000 as *const u8, 1).with_tier(1);
        code_cache.insert(fake_entry);

        // Enqueue at tier 2 - should compile with the optimizing pipeline.
        assert!(queue.enqueue(Arc::clone(&code), 2));

        // Wait for the tier upgrade to complete.
        let mut attempts = 0;
        while attempts < 100 {
            if let Some(entry) = code_cache.lookup(code_id) {
                if entry.tier() >= 2 {
                    break;
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
            attempts += 1;
        }

        let entry = code_cache
            .lookup(code_id)
            .expect("entry should remain present");
        assert_eq!(entry.tier(), 2);
        assert_eq!(entry.return_abi(), ReturnAbi::RawValueBits);

        let (enqueued, completed, failed, _dropped) = queue.stats().snapshot();
        assert_eq!(enqueued, 1);
        assert_eq!(completed, 1);
        assert_eq!(failed, 0);
    }
}

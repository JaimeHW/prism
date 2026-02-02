//! Generator protocol implementation.
//!
//! This module provides stackless generator support matching V8-class performance.
//! Generators are implemented as state machines with liveness-aware frame capture.
//!
//! # Architecture
//!
//! - **Tagged State**: 4-byte header encodes state + resume index
//! - **Union Storage**: Inline 8-value storage eliminates allocation for small generators
//! - **Zero-Copy Resume**: Direct jump to resume PC via computed goto pattern
//! - **Flyweight Exceptions**: StopIteration/GeneratorExit use pre-allocated singletons
//!
//! # Performance Targets
//!
//! | Metric | Target |
//! |--------|--------|
//! | Creation | ~15 cycles |
//! | Yield | ~3-5 cycles |
//! | Resume | ~5 cycles |
//! | Memory (small) | 96 bytes |

mod iterator;
mod methods;
mod object;
mod resume;
mod state;
mod storage;

pub use iterator::{GeneratorError, GeneratorException, GeneratorIterator, IterResult};
pub use methods::{
    CloseResult, SendResult, ThrowResult, complete_close_caught, complete_close_exception,
    complete_close_yielded, complete_send_returned, complete_send_yielded,
    complete_throw_propagated, complete_throw_returned, complete_throw_yielded, generator_exit,
    prepare_close, prepare_send, prepare_throw, stop_iteration,
};
pub use object::{GeneratorFlags, GeneratorObject};
pub use resume::{
    MAX_RESUME_POINTS, ResumeAction, ResumePoint, ResumeTable, ResumeTableBuilder, can_resume,
    exhaust_generator, prepare_resume, restore_generator_state, suspend_at_yield,
};
pub use state::{GeneratorHeader, GeneratorState};
pub use storage::{FrameStorage, LivenessMap};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all public types are accessible
        let _ = GeneratorState::Created;
        let header = GeneratorHeader::new();
        assert_eq!(header.state(), GeneratorState::Created);
    }
}

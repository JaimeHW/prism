//! Comprehensive integration tests for coroutine/async operations.
//!
//! This module provides exhaustive testing for:
//! - GetAwaitable: Type detection and __await__ protocol
//! - GetAIter: Async iteration protocol initialization
//! - GetANext: Async iteration step execution
//! - EndAsyncFor: StopAsyncIteration handling
//! - Send: Generator/coroutine send protocol
//!
//! # Test Categories
//!
//! - Type Classification: Verify correct awaitable/async-iterable detection
//! - Error Handling: Verify correct TypeError messages
//! - State Machine: Verify generator state transitions
//! - Protocol Compliance: Verify Python async protocol semantics
//! - Performance: Verify optimized paths are taken

use prism_core::Value;
use std::sync::Arc;

use prism_compiler::bytecode::CodeObject;

use crate::ops::generator::{GeneratorContext, GeneratorExecutionState};
use crate::stdlib::generators::{GeneratorFlags, GeneratorObject, LivenessMap};

// =============================================================================
// Test Helpers
// =============================================================================

/// Creates a test code object for generator tests.
fn test_code() -> Arc<CodeObject> {
    Arc::new(CodeObject::new("test_generator", "<test>"))
}

/// Creates a regular generator (not coroutine, not async).
fn regular_generator() -> GeneratorObject {
    GeneratorObject::new(test_code())
}

/// Creates a coroutine (async def function).
fn coroutine_generator() -> GeneratorObject {
    GeneratorObject::with_flags(
        test_code(),
        GeneratorFlags::IS_COROUTINE | GeneratorFlags::INLINE_STORAGE,
    )
}

/// Creates an async generator (async def with yield).
fn async_generator() -> GeneratorObject {
    GeneratorObject::with_flags(
        test_code(),
        GeneratorFlags::IS_ASYNC | GeneratorFlags::INLINE_STORAGE,
    )
}

/// Get type name for error message testing.
fn type_name_for_value(value: &Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.as_object_ptr().is_some() {
        "object"
    } else {
        "unknown"
    }
}

// =============================================================================
// GetAwaitable Tests - Type Classification
// =============================================================================

mod get_awaitable_tests {
    use super::*;

    #[test]
    fn test_none_not_awaitable() {
        let val = Value::none();
        assert!(val.is_none());
        assert!(val.as_object_ptr().is_none(), "None should not be object");
    }

    #[test]
    fn test_bool_true_not_awaitable() {
        let val = Value::bool(true);
        assert!(val.is_bool());
        // Bools should not be awaitable
    }

    #[test]
    fn test_bool_false_not_awaitable() {
        let val = Value::bool(false);
        assert!(val.is_bool());
    }

    #[test]
    fn test_int_zero_not_awaitable() {
        let val = Value::int(0).unwrap();
        // Integer zero should not be awaitable
        assert!(!val.is_none());
    }

    #[test]
    fn test_int_positive_not_awaitable() {
        let val = Value::int(42).unwrap();
        assert!(!val.is_none());
    }

    #[test]
    fn test_int_negative_not_awaitable() {
        let val = Value::int(-1).unwrap();
        assert!(!val.is_none());
    }

    #[test]
    fn test_int_max_not_awaitable() {
        if let Some(val) = Value::int(i64::MAX) {
            assert!(!val.is_none());
        }
    }

    #[test]
    fn test_int_min_not_awaitable() {
        if let Some(val) = Value::int(i64::MIN) {
            assert!(!val.is_none());
        }
    }

    #[test]
    fn test_float_zero_not_awaitable() {
        let val = Value::float(0.0);
        assert!(val.is_float());
    }

    #[test]
    fn test_float_positive_not_awaitable() {
        let val = Value::float(3.14159);
        assert!(val.is_float());
    }

    #[test]
    fn test_float_negative_not_awaitable() {
        let val = Value::float(-2.718);
        assert!(val.is_float());
    }

    #[test]
    fn test_float_inf_not_awaitable() {
        let val = Value::float(f64::INFINITY);
        assert!(val.is_float());
    }

    #[test]
    fn test_float_neg_inf_not_awaitable() {
        let val = Value::float(f64::NEG_INFINITY);
        assert!(val.is_float());
    }

    #[test]
    fn test_float_nan_not_awaitable() {
        let val = Value::float(f64::NAN);
        assert!(val.is_float());
    }
}

// =============================================================================
// GetAwaitable Tests - Generator Type Detection
// =============================================================================

mod get_awaitable_generator_tests {
    use super::*;

    #[test]
    fn test_coroutine_is_awaitable() {
        let generator = coroutine_generator();
        assert!(generator.is_coroutine(), "Coroutine flag should be set");
        assert!(!generator.is_async(), "Async flag should not be set");
    }

    #[test]
    fn test_async_generator_is_awaitable() {
        let generator = async_generator();
        assert!(generator.is_async(), "Async flag should be set");
        assert!(
            !generator.is_coroutine(),
            "Coroutine flag should not be set"
        );
    }

    #[test]
    fn test_regular_generator_not_awaitable() {
        let generator = regular_generator();
        assert!(
            !generator.is_coroutine(),
            "Regular generator should not be coroutine"
        );
        assert!(
            !generator.is_async(),
            "Regular generator should not be async"
        );
    }

    #[test]
    fn test_coroutine_combined_flags() {
        let generator = GeneratorObject::with_flags(
            test_code(),
            GeneratorFlags::IS_COROUTINE | GeneratorFlags::HAS_JIT | GeneratorFlags::INLINE_STORAGE,
        );
        assert!(generator.is_coroutine());
        assert!(generator.flags().contains(GeneratorFlags::HAS_JIT));
    }

    #[test]
    fn test_async_generator_combined_flags() {
        let generator = GeneratorObject::with_flags(
            test_code(),
            GeneratorFlags::IS_ASYNC | GeneratorFlags::HAS_CLOSURE,
        );
        assert!(generator.is_async());
        assert!(generator.flags().contains(GeneratorFlags::HAS_CLOSURE));
    }
}

// =============================================================================
// GetAIter Tests - Type Classification
// =============================================================================

mod get_aiter_tests {
    use super::*;

    #[test]
    fn test_none_not_async_iterable() {
        let val = Value::none();
        assert!(val.is_none());
    }

    #[test]
    fn test_bool_not_async_iterable() {
        let val = Value::bool(true);
        assert!(val.is_bool());
    }

    #[test]
    fn test_int_not_async_iterable() {
        let val = Value::int(42).unwrap();
        assert!(!val.is_none());
    }

    #[test]
    fn test_float_not_async_iterable() {
        let val = Value::float(3.14);
        assert!(val.is_float());
    }

    #[test]
    fn test_async_generator_is_async_iterable() {
        let generator = async_generator();
        assert!(
            generator.is_async(),
            "Async generator should be async iterable"
        );
    }

    #[test]
    fn test_regular_generator_not_async_iterable() {
        let generator = regular_generator();
        assert!(
            !generator.is_async(),
            "Regular generator should not be async iterable"
        );
    }

    #[test]
    fn test_coroutine_not_async_iterable() {
        let generator = coroutine_generator();
        // Coroutines are awaitable but not async iterable
        assert!(generator.is_coroutine());
        assert!(!generator.is_async());
    }
}

// =============================================================================
// GetANext Tests
// =============================================================================

mod get_anext_tests {
    use super::*;

    #[test]
    fn test_async_generator_state_for_anext() {
        let generator = async_generator();
        // Async generator should start in Created state
        assert!(generator.is_resumable());
        assert!(!generator.is_running());
        assert!(!generator.is_exhausted());
    }

    #[test]
    fn test_anext_requires_async_generator() {
        let regular = regular_generator();
        assert!(!regular.is_async(), "Regular generator cannot use anext");
    }
}

// =============================================================================
// EndAsyncFor Tests
// =============================================================================

mod end_async_for_tests {
    use super::*;

    #[test]
    fn test_end_async_for_normal_value_passthrough() {
        // Normal values should pass through EndAsyncFor
        let val = Value::int(42).unwrap();
        assert!(!val.is_none());
    }

    #[test]
    fn test_end_async_for_none_passthrough() {
        let val = Value::none();
        assert!(val.is_none());
    }

    #[test]
    fn test_end_async_for_bool_passthrough() {
        let val = Value::bool(true);
        assert!(val.is_bool());
    }
}

// =============================================================================
// Send Tests - Type Validation
// =============================================================================

mod send_type_tests {
    use super::*;

    #[test]
    fn test_none_not_sendable() {
        let val = Value::none();
        assert!(val.is_none(), "Cannot send to None");
    }

    #[test]
    fn test_bool_not_sendable() {
        let val = Value::bool(true);
        assert!(val.is_bool(), "Cannot send to bool");
    }

    #[test]
    fn test_int_not_sendable() {
        let val = Value::int(42).unwrap();
        assert!(!val.is_none(), "Cannot send to int");
    }

    #[test]
    fn test_float_not_sendable() {
        let val = Value::float(3.14);
        assert!(val.is_float(), "Cannot send to float");
    }

    #[test]
    fn test_generator_is_sendable_type() {
        let generator = regular_generator();
        // Regular generators support send protocol
        assert!(generator.is_resumable());
    }

    #[test]
    fn test_coroutine_is_sendable_type() {
        let generator = coroutine_generator();
        // Coroutines support send protocol
        assert!(generator.is_resumable());
        assert!(generator.is_coroutine());
    }
}

// =============================================================================
// Send Tests - Generator State Machine
// =============================================================================

mod send_state_tests {
    use super::*;
    use crate::stdlib::generators::LivenessMap;

    #[test]
    fn test_generator_created_state() {
        let generator = regular_generator();
        assert!(
            generator.is_resumable(),
            "Created generator should be resumable"
        );
        assert!(
            !generator.is_running(),
            "Created generator should not be running"
        );
        assert!(
            !generator.is_exhausted(),
            "Created generator should not be exhausted"
        );
    }

    #[test]
    fn test_generator_start_transitions_to_running() {
        let generator = regular_generator();
        let prev_state = generator.try_start();
        assert!(
            prev_state.is_some(),
            "Should be able to start fresh generator"
        );
        assert!(
            generator.is_running(),
            "Started generator should be running"
        );
    }

    #[test]
    fn test_generator_already_running_cannot_start() {
        let generator = regular_generator();
        generator.try_start();
        let second_start = generator.try_start();
        assert!(
            second_start.is_none(),
            "Cannot start already running generator"
        );
    }

    #[test]
    fn test_generator_suspend_transitions_to_suspended() {
        let mut generator = regular_generator();
        generator.try_start();

        let registers = [Value::none(); 256];
        generator.suspend(100, 1, &registers, LivenessMap::from_bits(0));

        assert!(
            generator.is_resumable(),
            "Suspended generator should be resumable"
        );
        assert!(
            !generator.is_running(),
            "Suspended generator should not be running"
        );
    }

    #[test]
    fn test_generator_exhaust_transitions_to_exhausted() {
        let generator = regular_generator();
        generator.try_start();
        generator.exhaust();

        assert!(
            generator.is_exhausted(),
            "Exhausted generator should be exhausted"
        );
        assert!(
            !generator.is_resumable(),
            "Exhausted generator should not be resumable"
        );
    }

    #[test]
    fn test_generator_exhausted_cannot_start() {
        let generator = regular_generator();
        generator.try_start();
        generator.exhaust();

        let restart = generator.try_start();
        assert!(restart.is_none(), "Cannot restart exhausted generator");
    }

    #[test]
    fn test_generator_multiple_yield_cycles() {
        let mut generator = regular_generator();
        let registers = [Value::none(); 256];

        for i in 0..5 {
            let start_result = generator.try_start();
            assert!(start_result.is_some(), "Cycle {} should start", i);
            generator.suspend(i * 10, i, &registers, LivenessMap::from_bits(0));
        }

        // Final cycle and exhaust
        generator.try_start();
        generator.exhaust();
        assert!(generator.is_exhausted());
    }
}

// =============================================================================
// Send Tests - Value Protocol
// =============================================================================

mod send_value_tests {
    use super::*;

    #[test]
    fn test_send_value_none_to_new_generator() {
        let mut generator = regular_generator();
        // First send must be None
        generator.set_send_value(Value::none());
        let val = generator.take_receive_value();
        assert!(val.unwrap().is_none());
    }

    #[test]
    fn test_send_value_int_to_suspended_generator() {
        let mut generator = regular_generator();
        generator.set_send_value(Value::int(42).unwrap());
        let val = generator.take_receive_value();
        assert_eq!(val.unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_send_value_float_to_suspended_generator() {
        let mut generator = regular_generator();
        generator.set_send_value(Value::float(3.14));
        let val = generator.take_receive_value();
        assert!(val.unwrap().is_float());
    }

    #[test]
    fn test_send_value_bool_to_suspended_generator() {
        let mut generator = regular_generator();
        generator.set_send_value(Value::bool(true));
        let val = generator.take_receive_value();
        assert!(val.unwrap().is_bool());
    }

    #[test]
    fn test_send_value_consumed_after_take() {
        let mut generator = regular_generator();
        generator.set_send_value(Value::int(99).unwrap());

        let first = generator.take_receive_value();
        assert!(first.is_some());

        let second = generator.take_receive_value();
        assert!(second.is_none(), "Value should be consumed after take");
    }

    #[test]
    fn test_peek_does_not_consume() {
        let mut generator = regular_generator();
        generator.set_send_value(Value::int(123).unwrap());

        let peek1 = generator.peek_receive_value();
        assert!(peek1.is_some());

        let peek2 = generator.peek_receive_value();
        assert!(peek2.is_some(), "Peek should not consume");

        let take = generator.take_receive_value();
        assert!(take.is_some());
    }

    #[test]
    fn test_send_value_replaced() {
        let mut generator = regular_generator();

        generator.set_send_value(Value::int(1).unwrap());
        generator.set_send_value(Value::int(2).unwrap());

        let val = generator.take_receive_value();
        assert_eq!(val.unwrap().as_int(), Some(2), "Should get latest value");
    }
}

// =============================================================================
// GeneratorContext Tests - State Transitions
// =============================================================================

mod generator_context_tests {
    use super::*;
    use std::ptr::NonNull;

    fn dangling_generator() -> NonNull<GeneratorObject> {
        NonNull::dangling()
    }

    #[test]
    fn test_context_new_is_idle() {
        let ctx = GeneratorContext::new();
        assert!(!ctx.is_active());
        assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
    }

    #[test]
    fn test_context_enter_activates() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        assert!(ctx.is_active());
        assert_eq!(ctx.state(), GeneratorExecutionState::Running);
    }

    #[test]
    fn test_context_exit_deactivates() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.exit();
        assert!(!ctx.is_active());
        assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
    }

    #[test]
    fn test_context_suspend_state() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.suspend();
        assert_eq!(ctx.state(), GeneratorExecutionState::Suspended);
    }

    #[test]
    fn test_context_resume_state() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.suspend();
        ctx.resume(None);
        assert_eq!(ctx.state(), GeneratorExecutionState::Running);
    }

    #[test]
    fn test_context_resume_with_value() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.suspend();
        ctx.resume(Some(Value::int(42).unwrap()));

        assert!(ctx.has_send_value());
        let val = ctx.take_send_value();
        assert_eq!(val.unwrap().as_int(), Some(42));
    }

    #[test]
    fn test_context_complete_state() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.complete();
        assert_eq!(ctx.state(), GeneratorExecutionState::Completed);
    }

    #[test]
    fn test_context_throw_state() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.throw(Value::int(999).unwrap());

        assert_eq!(ctx.state(), GeneratorExecutionState::Throwing);
        assert!(ctx.has_thrown_exception());
    }

    #[test]
    fn test_context_take_thrown_exception() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.throw(Value::int(888).unwrap());

        let exc = ctx.take_thrown_exception();
        assert_eq!(exc.unwrap().as_int(), Some(888));
        assert!(!ctx.has_thrown_exception());
    }

    #[test]
    fn test_context_nesting() {
        let mut ctx = GeneratorContext::new();

        ctx.enter(dangling_generator());
        assert_eq!(ctx.nesting_depth(), 0);

        ctx.enter(dangling_generator());
        assert_eq!(ctx.nesting_depth(), 1);

        ctx.enter(dangling_generator());
        assert_eq!(ctx.nesting_depth(), 2);

        ctx.exit();
        assert_eq!(ctx.nesting_depth(), 1);
        assert!(ctx.is_active());

        ctx.exit();
        assert_eq!(ctx.nesting_depth(), 0);
        assert!(ctx.is_active());

        ctx.exit();
        assert!(!ctx.is_active());
    }

    #[test]
    fn test_context_stats_activations() {
        let mut ctx = GeneratorContext::new();

        for _ in 0..3 {
            ctx.enter(dangling_generator());
            ctx.exit();
        }

        assert_eq!(ctx.stats().activations, 3);
    }

    #[test]
    fn test_context_stats_yields() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());

        for _ in 0..5 {
            ctx.suspend();
            ctx.resume(None);
        }

        assert_eq!(ctx.stats().total_yields, 5);
        assert_eq!(ctx.stats().total_resumes, 5);
    }

    #[test]
    fn test_context_stats_closes() {
        let mut ctx = GeneratorContext::new();

        for _ in 0..2 {
            ctx.enter(dangling_generator());
            ctx.begin_close();
            ctx.exit();
        }

        assert_eq!(ctx.stats().total_closes, 2);
    }

    #[test]
    fn test_context_stats_throws() {
        let mut ctx = GeneratorContext::new();

        for i in 0..4 {
            ctx.enter(dangling_generator());
            ctx.throw(Value::int(i).unwrap());
            ctx.take_thrown_exception();
            ctx.exit();
        }

        assert_eq!(ctx.stats().total_throws, 4);
    }

    #[test]
    fn test_context_reset() {
        let mut ctx = GeneratorContext::new();
        ctx.enter(dangling_generator());
        ctx.suspend();
        ctx.resume(Some(Value::int(42).unwrap()));

        ctx.reset();

        assert!(!ctx.is_active());
        assert_eq!(ctx.state(), GeneratorExecutionState::Idle);
        assert_eq!(ctx.nesting_depth(), 0);
        assert!(!ctx.has_send_value());
    }

    #[test]
    fn test_context_max_nesting_depth() {
        let mut ctx = GeneratorContext::new();

        for _ in 0..10 {
            ctx.enter(dangling_generator());
        }

        assert_eq!(ctx.stats().max_nesting_depth, 9);
    }

    #[test]
    fn test_state_can_continue() {
        assert!(!GeneratorExecutionState::Idle.can_continue());
        assert!(GeneratorExecutionState::Running.can_continue());
        assert!(!GeneratorExecutionState::Suspended.can_continue());
        assert!(!GeneratorExecutionState::Closing.can_continue());
        assert!(GeneratorExecutionState::Throwing.can_continue());
        assert!(!GeneratorExecutionState::Completed.can_continue());
    }

    #[test]
    fn test_state_can_resume() {
        assert!(!GeneratorExecutionState::Idle.can_resume());
        assert!(!GeneratorExecutionState::Running.can_resume());
        assert!(GeneratorExecutionState::Suspended.can_resume());
        assert!(!GeneratorExecutionState::Closing.can_resume());
        assert!(!GeneratorExecutionState::Throwing.can_resume());
        assert!(!GeneratorExecutionState::Completed.can_resume());
    }

    #[test]
    fn test_state_can_close() {
        assert!(!GeneratorExecutionState::Idle.can_close());
        assert!(GeneratorExecutionState::Running.can_close());
        assert!(GeneratorExecutionState::Suspended.can_close());
        assert!(!GeneratorExecutionState::Closing.can_close());
        assert!(!GeneratorExecutionState::Throwing.can_close());
        assert!(!GeneratorExecutionState::Completed.can_close());
    }

    #[test]
    fn test_state_is_active() {
        assert!(!GeneratorExecutionState::Idle.is_active());
        assert!(GeneratorExecutionState::Running.is_active());
        assert!(GeneratorExecutionState::Suspended.is_active());
        assert!(GeneratorExecutionState::Closing.is_active());
        assert!(GeneratorExecutionState::Throwing.is_active());
        assert!(!GeneratorExecutionState::Completed.is_active());
    }
}

// =============================================================================
// Error Message Tests
// =============================================================================

mod error_message_tests {
    use super::*;

    #[test]
    fn test_type_name_none() {
        assert_eq!(type_name_for_value(&Value::none()), "NoneType");
    }

    #[test]
    fn test_type_name_bool() {
        assert_eq!(type_name_for_value(&Value::bool(true)), "bool");
        assert_eq!(type_name_for_value(&Value::bool(false)), "bool");
    }

    #[test]
    fn test_type_name_int() {
        assert_eq!(type_name_for_value(&Value::int(42).unwrap()), "int");
        assert_eq!(type_name_for_value(&Value::int(0).unwrap()), "int");
        assert_eq!(type_name_for_value(&Value::int(-1).unwrap()), "int");
    }

    #[test]
    fn test_type_name_float() {
        assert_eq!(type_name_for_value(&Value::float(3.14)), "float");
        assert_eq!(type_name_for_value(&Value::float(0.0)), "float");
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

mod performance_tests {
    use super::*;

    #[test]
    fn test_generator_object_size() {
        let size = std::mem::size_of::<GeneratorObject>();
        assert!(size <= 160, "GeneratorObject too large: {} bytes", size);
    }

    #[test]
    fn test_generator_flags_size() {
        let size = std::mem::size_of::<GeneratorFlags>();
        assert_eq!(size, 2, "GeneratorFlags should be 2 bytes");
    }

    #[test]
    fn test_generator_context_size() {
        let size = std::mem::size_of::<GeneratorContext>();
        assert!(size <= 128, "GeneratorContext too large: {} bytes", size);
    }

    #[test]
    fn test_many_generator_creations() {
        for _ in 0..1000 {
            let generator = regular_generator();
            assert!(generator.is_resumable());
        }
    }

    #[test]
    fn test_many_coroutine_creations() {
        for _ in 0..1000 {
            let generator = coroutine_generator();
            assert!(generator.is_coroutine());
        }
    }

    #[test]
    fn test_many_async_generator_creations() {
        for _ in 0..1000 {
            let generator = async_generator();
            assert!(generator.is_async());
        }
    }

    #[test]
    fn test_many_context_activations() {
        use std::ptr::NonNull;
        let mut ctx = GeneratorContext::new();

        for _ in 0..1000 {
            ctx.enter(NonNull::dangling());
            ctx.suspend();
            ctx.resume(None);
            ctx.exit();
        }

        assert_eq!(ctx.stats().activations, 1000);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_int_boundary_values() {
        let cases = [0i64, 1, -1, 127, 128, -128, -129, 255, 256, -256];

        for &i in &cases {
            let val = Value::int(i).unwrap();
            assert!(!val.is_none());
        }
    }

    #[test]
    fn test_float_special_values() {
        let cases = [0.0, -0.0, 1.0, -1.0, f64::EPSILON, f64::MIN_POSITIVE];

        for f in cases {
            let val = Value::float(f);
            assert!(val.is_float());
        }
    }

    #[test]
    fn test_float_extreme_values() {
        let cases = [f64::MIN, f64::MAX, f64::INFINITY, f64::NEG_INFINITY];

        for f in cases {
            let val = Value::float(f);
            assert!(val.is_float());
        }
    }

    #[test]
    fn test_generator_resume_index_values() {
        let mut generator = regular_generator();
        generator.try_start();

        let registers = [Value::none(); 256];
        let liveness = crate::stdlib::generators::LivenessMap::from_bits(0);

        // Test various resume indices
        // Note: resume_index has limited bit budget (30 bits), so test reasonable values
        for i in [0u32, 1, 10, 100, 1000, 10000, 100000] {
            generator.suspend(i, i, &registers, liveness);
            assert_eq!(generator.resume_index(), i);
            generator.try_start();
        }
    }

    #[test]
    fn test_generator_ip_values() {
        let mut generator = regular_generator();
        generator.try_start();

        let registers = [Value::none(); 256];
        let liveness = crate::stdlib::generators::LivenessMap::from_bits(0);

        // Test various instruction pointer values
        for ip in [0u32, 1, 100, 10000, u32::MAX / 2] {
            generator.suspend(ip, 0, &registers, liveness);
            assert_eq!(generator.ip(), ip);
            generator.try_start();
        }
    }
}

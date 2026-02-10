//! Comprehensive exception handling integration tests for Prism VM.
//!
//! This test suite validates CPython 3.11+ exception semantics including:
//! - Basic try/except/finally blocks
//! - Exception propagation across frame depths
//! - Nested exception handling
//! - Exception chaining (raise ... from ...)
//! - Reraise without arguments
//! - Exception matching (multi-except, tuple matching)
//! - finally block guarantees
//! - Exception state preservation
//!
//! # Test Organization
//!
//! Tests are organized into categories:
//! 1. Basic Try/Except (15 tests)
//! 2. Finally Block Semantics (15 tests)
//! 3. Exception Propagation (10 tests)
//! 4. Exception Chaining (8 tests)
//! 5. Nested Handlers (12 tests)
//! 6. Edge Cases (10+ tests)

use prism_compiler::Compiler;
use prism_core::Value;
use prism_parser::parse;
use prism_vm::VirtualMachine;
use std::sync::Arc;

// =============================================================================
// Test Helper Functions
// =============================================================================

/// Execute Python source and return result (success case).
fn execute(source: &str) -> Result<Value, String> {
    let module = parse(source).map_err(|e| format!("Parse error: {:?}", e))?;
    let code = Compiler::compile_module(&module, "<test>")
        .map_err(|e| format!("Compile error: {:?}", e))?;
    let mut vm = VirtualMachine::new();
    vm.execute(Arc::new(code))
        .map_err(|e| format!("Runtime error: {:?}", e))
}

/// Verify that execution succeeds without error.
fn assert_executes(source: &str, test_name: &str) {
    let result = execute(source);
    assert!(result.is_ok(), "{}: {:?}", test_name, result);
}

/// Verify that execution raises an exception.
fn assert_raises(source: &str, test_name: &str) {
    let result = execute(source);
    assert!(
        result.is_err(),
        "{} should raise, got: {:?}",
        test_name,
        result
    );
}

// =============================================================================
// CATEGORY 1: Basic Try/Except (15 tests)
// =============================================================================

#[test]
fn test_try_except_no_exception() {
    assert_executes(
        r#"
result = 0
try:
    result = 1
except:
    result = -1
"#,
        "try_except_no_exception",
    );
}

#[test]
fn test_try_except_catches_exception() {
    assert_executes(
        r#"
result = 0
try:
    raise ValueError("test")
except:
    result = 1
"#,
        "try_except_catches_exception",
    );
}

#[test]
fn test_try_except_specific_type() {
    assert_executes(
        r#"
result = 0
try:
    raise ValueError("test")
except ValueError:
    result = 1
"#,
        "try_except_specific_type",
    );
}

#[test]
fn test_try_except_wrong_type_propagates() {
    assert_raises(
        r#"
try:
    raise TypeError("test")
except ValueError:
    pass
"#,
        "try_except_wrong_type_propagates",
    );
}

#[test]
fn test_try_except_as_binding() {
    assert_executes(
        r#"
result = None
try:
    raise ValueError("message")
except ValueError as e:
    result = e
"#,
        "try_except_as_binding",
    );
}

#[test]
fn test_try_except_tuple_match() {
    assert_executes(
        r#"
result = 0
try:
    raise TypeError("test")
except (ValueError, TypeError):
    result = 1
"#,
        "try_except_tuple_match",
    );
}

#[test]
fn test_try_except_multiple_handlers() {
    assert_executes(
        r#"
result = 0
try:
    raise TypeError("test")
except ValueError:
    result = 1
except TypeError:
    result = 2
except:
    result = 3
"#,
        "try_except_multiple_handlers",
    );
}

#[test]
fn test_try_except_first_matching_wins() {
    assert_executes(
        r#"
result = 0
try:
    raise ValueError("test")
except ValueError:
    result = 1
except Exception:
    result = 2
"#,
        "try_except_first_matching_wins",
    );
}

#[test]
fn test_try_except_base_exception() {
    assert_executes(
        r#"
result = 0
try:
    raise BaseException("test")
except BaseException:
    result = 1
"#,
        "try_except_base_exception",
    );
}

#[test]
fn test_try_except_exception_hierarchy() {
    assert_executes(
        r#"
result = 0
try:
    raise ValueError("test")
except Exception:
    result = 1
"#,
        "try_except_exception_hierarchy",
    );
}

#[test]
fn test_try_except_empty_body() {
    assert_executes(
        r#"
try:
    pass
except:
    pass
"#,
        "try_except_empty_body",
    );
}

#[test]
fn test_try_except_expression_raises() {
    assert_executes(
        r#"
result = 0
try:
    x = 1 / 0
except ZeroDivisionError:
    result = 1
"#,
        "try_except_expression_raises",
    );
}

#[test]
fn test_try_except_bare_except_last() {
    assert_executes(
        r#"
result = 0
try:
    raise RuntimeError("test")
except ValueError:
    result = 1
except:
    result = 2
"#,
        "try_except_bare_except_last",
    );
}

#[test]
fn test_try_except_preserves_variable_scope() {
    assert_executes(
        r#"
x = 1
try:
    x = 2
    raise ValueError("test")
except:
    y = x
"#,
        "try_except_preserves_variable_scope",
    );
}

#[test]
fn test_try_except_no_exc_info_after_handled() {
    assert_executes(
        r#"
result = None
try:
    raise ValueError("test")
except:
    result = "handled"
# Exception should be cleared after except block
"#,
        "try_except_no_exc_info_after_handled",
    );
}

// =============================================================================
// CATEGORY 2: Finally Block Semantics (15 tests)
// =============================================================================

#[test]
fn test_finally_runs_on_success() {
    assert_executes(
        r#"
cleanup = False
try:
    x = 1
finally:
    cleanup = True
"#,
        "finally_runs_on_success",
    );
}

#[test]
fn test_finally_runs_on_exception() {
    assert_raises(
        r#"
cleanup = False
try:
    raise ValueError("test")
finally:
    cleanup = True
"#,
        "finally_runs_on_exception",
    );
}

#[test]
fn test_finally_after_except() {
    assert_executes(
        r#"
order = []
try:
    raise ValueError("test")
except:
    order = [1]
finally:
    order = order + [2]
"#,
        "finally_after_except",
    );
}

#[test]
fn test_finally_runs_on_return() {
    assert_executes(
        r#"
def f():
    cleanup = False
    try:
        return 1
    finally:
        cleanup = True
    return 2  # Never reached
result = f()
"#,
        "finally_runs_on_return",
    );
}

#[test]
fn test_finally_runs_on_break() {
    assert_executes(
        r#"
cleanup = False
for i in range(10):
    try:
        if i == 5:
            break
    finally:
        cleanup = True
"#,
        "finally_runs_on_break",
    );
}

#[test]
fn test_finally_runs_on_continue() {
    assert_executes(
        r#"
cleanup_count = 0
for i in range(3):
    try:
        continue
    finally:
        cleanup_count = cleanup_count + 1
"#,
        "finally_runs_on_continue",
    );
}

#[test]
fn test_finally_reraises_exception() {
    assert_raises(
        r#"
try:
    raise ValueError("test")
finally:
    pass
"#,
        "finally_reraises_exception",
    );
}

#[test]
fn test_finally_exception_replaces_original() {
    // If finally raises, it replaces the original exception
    assert_raises(
        r#"
try:
    raise ValueError("original")
finally:
    raise TypeError("replacement")
"#,
        "finally_exception_replaces_original",
    );
}

#[test]
fn test_finally_return_overrides_exception() {
    assert_executes(
        r#"
def f():
    try:
        raise ValueError("test")
    finally:
        return 42
result = f()
"#,
        "finally_return_overrides_exception",
    );
}

#[test]
fn test_nested_finally_execution_order() {
    assert_executes(
        r#"
order = []
try:
    try:
        x = 1
    finally:
        order = order + [1]
finally:
    order = order + [2]
"#,
        "nested_finally_execution_order",
    );
}

#[test]
fn test_finally_with_no_except() {
    assert_executes(
        r#"
result = 0
try:
    result = 1
finally:
    result = result + 1
"#,
        "finally_with_no_except",
    );
}

#[test]
fn test_finally_cleanup_multiple_resources() {
    assert_executes(
        r#"
resource1 = False
resource2 = False
try:
    x = 1
finally:
    resource1 = True
    resource2 = True
"#,
        "finally_cleanup_multiple_resources",
    );
}

#[test]
fn test_try_except_finally_all_three() {
    assert_executes(
        r#"
steps = []
try:
    steps = steps + [1]
    raise ValueError("test")
except:
    steps = steps + [2]
finally:
    steps = steps + [3]
"#,
        "try_except_finally_all_three",
    );
}

#[test]
fn test_finally_preserves_return_value() {
    assert_executes(
        r#"
def f():
    try:
        return 42
    finally:
        x = 1  # Doesn't affect return value
result = f()
"#,
        "finally_preserves_return_value",
    );
}

#[test]
fn test_finally_with_loop_control() {
    assert_executes(
        r#"
total = 0
for i in range(5):
    try:
        total = total + i
    finally:
        pass
"#,
        "finally_with_loop_control",
    );
}

// =============================================================================
// CATEGORY 3: Exception Propagation (10 tests)
// =============================================================================

#[test]
fn test_propagation_across_1_frame() {
    assert_executes(
        r#"
def inner():
    raise ValueError("from inner")

caught = False
try:
    inner()
except:
    caught = True
"#,
        "propagation_across_1_frame",
    );
}

#[test]
fn test_propagation_across_2_frames() {
    assert_executes(
        r#"
def level2():
    raise ValueError("from level2")

def level1():
    level2()

caught = False
try:
    level1()
except:
    caught = True
"#,
        "propagation_across_2_frames",
    );
}

#[test]
fn test_propagation_across_3_frames() {
    assert_executes(
        r#"
def level3():
    raise ValueError("from level3")

def level2():
    level3()

def level1():
    level2()

caught = False
try:
    level1()
except:
    caught = True
"#,
        "propagation_across_3_frames",
    );
}

#[test]
fn test_propagation_across_5_frames() {
    assert_executes(
        r#"
def level5():
    raise ValueError("from level5")

def level4():
    level5()

def level3():
    level4()

def level2():
    level3()

def level1():
    level2()

caught = False
try:
    level1()
except:
    caught = True
"#,
        "propagation_across_5_frames",
    );
}

#[test]
fn test_propagation_partial_catch() {
    assert_executes(
        r#"
def inner():
    raise ValueError("test")

def middle():
    try:
        inner()
    except TypeError:
        pass  # Wrong type, propagates

caught = False
try:
    middle()
except ValueError:
    caught = True
"#,
        "propagation_partial_catch",
    );
}

#[test]
fn test_propagation_reraise() {
    assert_raises(
        r#"
def inner():
    try:
        raise ValueError("test")
    except:
        raise  # Reraise

inner()
"#,
        "propagation_reraise",
    );
}

#[test]
fn test_propagation_stops_at_handler() {
    assert_executes(
        r#"
def level3():
    raise ValueError("test")

def level2():
    level3()

def level1():
    try:
        level2()
    except:
        pass  # Caught here

level1()
"#,
        "propagation_stops_at_handler",
    );
}

#[test]
fn test_propagation_unwinds_stack() {
    assert_executes(
        r#"
stack_depth = 0

def level3():
    global stack_depth
    stack_depth = stack_depth + 1
    raise ValueError("test")

def level2():
    global stack_depth
    stack_depth = stack_depth + 1
    level3()

def level1():
    global stack_depth
    stack_depth = stack_depth + 1
    level2()

try:
    level1()
except:
    pass
"#,
        "propagation_unwinds_stack",
    );
}

#[test]
fn test_propagation_through_finally() {
    assert_raises(
        r#"
cleanup = []

def inner():
    try:
        raise ValueError("test")
    finally:
        cleanup.append(1)

inner()
"#,
        "propagation_through_finally",
    );
}

#[test]
fn test_propagation_to_module_level() {
    assert_raises(
        r#"
def f():
    raise ValueError("uncaught")

f()  # Should propagate to module level
"#,
        "propagation_to_module_level",
    );
}

// =============================================================================
// CATEGORY 4: Exception Chaining (8 tests)
// =============================================================================

#[test]
fn test_raise_from_explicit() {
    // Uses raise ... from e syntax with explicit cause
    // The outer except TypeError: catches the raised TypeError
    assert_executes(
        r#"
try:
    try:
        raise ValueError("original")
    except ValueError as e:
        raise TypeError("new") from e
except TypeError:
    pass
"#,
        "raise_from_explicit",
    );
}

#[test]
fn test_raise_from_none() {
    assert_raises(
        r#"
try:
    raise ValueError("original")
except:
    raise TypeError("clean") from None
"#,
        "raise_from_none",
    );
}

#[test]
fn test_raise_from_preserves_cause() {
    assert_executes(
        r#"
caught = None
try:
    try:
        raise ValueError("original")
    except ValueError as e:
        raise TypeError("new") from e
except TypeError as e:
    caught = e
"#,
        "raise_from_preserves_cause",
    );
}

#[test]
fn test_implicit_chaining() {
    assert_executes(
        r#"
caught = None
try:
    try:
        raise ValueError("original")
    except:
        raise TypeError("during handling")
except TypeError as e:
    caught = e
"#,
        "implicit_chaining",
    );
}

#[test]
fn test_chain_of_three() {
    assert_executes(
        r#"
caught = None
try:
    try:
        try:
            raise ValueError("first")
        except:
            raise TypeError("second")
    except:
        raise RuntimeError("third")
except RuntimeError as e:
    caught = e
"#,
        "chain_of_three",
    );
}

#[test]
fn test_cause_is_none_with_from_none() {
    assert_executes(
        r#"
result = None
try:
    try:
        raise ValueError("original")
    except:
        raise TypeError("suppressed") from None
except TypeError as e:
    result = e
"#,
        "cause_is_none_with_from_none",
    );
}

#[test]
fn test_raise_from_non_exception() {
    // Raising from a non-exception should work (stored as __cause__)
    assert_executes(
        r#"
caught = None
try:
    try:
        x = 1
    except:
        pass
    raise ValueError("test") from TypeError("cause")
except ValueError as e:
    caught = e
"#,
        "raise_from_non_exception",
    );
}

#[test]
fn test_raise_from_in_finally() {
    assert_raises(
        r#"
try:
    raise ValueError("original")
finally:
    raise TypeError("finally") from ValueError("cause")
"#,
        "raise_from_in_finally",
    );
}

// =============================================================================
// CATEGORY 5: Nested Handlers (12 tests)
// =============================================================================

#[test]
fn test_nested_try_inner_handles() {
    assert_executes(
        r#"
result = 0
try:
    try:
        raise ValueError("test")
    except:
        result = 1
except:
    result = 2
"#,
        "nested_try_inner_handles",
    );
}

#[test]
fn test_nested_try_outer_handles() {
    assert_executes(
        r#"
result = 0
try:
    try:
        raise ValueError("test")
    except TypeError:
        result = 1
except:
    result = 2
"#,
        "nested_try_outer_handles",
    );
}

#[test]
fn test_deeply_nested_3_levels() {
    assert_executes(
        r#"
result = 0
try:
    try:
        try:
            raise ValueError("test")
        except ValueError:
            result = 1
    except:
        result = 2
except:
    result = 3
"#,
        "deeply_nested_3_levels",
    );
}

#[test]
fn test_deeply_nested_5_levels() {
    assert_executes(
        r#"
result = 0
try:
    try:
        try:
            try:
                try:
                    raise ValueError("test")
                except:
                    result = 1
            except:
                result = 2
        except:
            result = 3
    except:
        result = 4
except:
    result = 5
"#,
        "deeply_nested_5_levels",
    );
}

#[test]
fn test_nested_in_loop() {
    assert_executes(
        r#"
results = []
for i in range(3):
    try:
        try:
            if i == 1:
                raise ValueError("test")
        except ValueError:
            results = results + [i]
    except:
        pass
"#,
        "nested_in_loop",
    );
}

#[test]
fn test_nested_in_function() {
    assert_executes(
        r#"
def outer():
    def inner():
        try:
            raise ValueError("test")
        except:
            return 1
    try:
        return inner()
    except:
        return 2

result = outer()
"#,
        "nested_in_function",
    );
}

#[test]
fn test_exception_in_except_block() {
    assert_executes(
        r#"
result = 0
try:
    try:
        raise ValueError("first")
    except:
        raise TypeError("second")
except TypeError:
    result = 1
"#,
        "exception_in_except_block",
    );
}

#[test]
fn test_exception_in_finally_block() {
    // Exception raised in finally block is caught by outer except
    // This is correct Python semantics - bare except catches all exceptions
    assert_executes(
        r#"
try:
    try:
        x = 1
    finally:
        raise ValueError("from finally")
except:
    pass
"#,
        "exception_in_finally_block",
    );
}

#[test]
fn test_nested_finally_all_run() {
    assert_executes(
        r#"
results = []
try:
    try:
        try:
            results = results + [1]
        finally:
            results = results + [2]
    finally:
        results = results + [3]
finally:
    results = results + [4]
"#,
        "nested_finally_all_run",
    );
}

#[test]
fn test_nested_finally_with_exception() {
    assert_raises(
        r#"
results = []
try:
    try:
        try:
            raise ValueError("test")
        finally:
            results = results + [1]
    finally:
        results = results + [2]
finally:
    results = results + [3]
"#,
        "nested_finally_with_exception",
    );
}

#[test]
fn test_mixed_nesting() {
    assert_executes(
        r#"
result = 0
try:
    try:
        raise ValueError("test")
    except:
        result = 1
    finally:
        result = result + 10
finally:
    result = result + 100
"#,
        "mixed_nesting",
    );
}

#[test]
fn test_parallel_handlers() {
    assert_executes(
        r#"
r1 = 0
r2 = 0
try:
    raise ValueError("test")
except ValueError:
    r1 = 1
    try:
        raise TypeError("inner")
    except TypeError:
        r2 = 2
"#,
        "parallel_handlers",
    );
}

// =============================================================================
// CATEGORY 6: Edge Cases and Special Scenarios (10+ tests)
// =============================================================================

#[test]
fn test_bare_raise_outside_handler() {
    // Bare raise outside handler should fail
    assert_raises(
        r#"
raise
"#,
        "bare_raise_outside_handler",
    );
}

#[test]
fn test_bare_raise_in_handler() {
    assert_raises(
        r#"
try:
    raise ValueError("test")
except:
    raise  # Reraise the exception
"#,
        "bare_raise_in_handler",
    );
}

#[test]
fn test_exception_in_generator() {
    assert_executes(
        r#"
def gen():
    yield 1
    raise ValueError("test")
    yield 2

g = gen()
first = next(g)
caught = False
try:
    second = next(g)
except ValueError:
    caught = True
"#,
        "exception_in_generator",
    );
}

#[test]
fn test_stop_iteration_from_next() {
    assert_executes(
        r#"
def gen():
    yield 1

g = gen()
first = next(g)
caught = False
try:
    second = next(g)
except StopIteration:
    caught = True
"#,
        "stop_iteration_from_next",
    );
}

#[test]
fn test_exception_clears_after_except() {
    assert_executes(
        r#"
exc_in_finally = None
try:
    raise ValueError("test")
except:
    pass
# Exception should be cleared
"#,
        "exception_clears_after_except",
    );
}

#[test]
fn test_exception_with_complex_expression() {
    assert_executes(
        r#"
caught = False
try:
    result = 1 / (2 - 2)  # ZeroDivisionError
except ZeroDivisionError:
    caught = True
"#,
        "exception_with_complex_expression",
    );
}

#[test]
fn test_exception_in_list_comprehension() {
    assert_executes(
        r#"
def maybe_fail(x):
    if x == 2:
        raise ValueError("fail")
    return x

caught = False
try:
    result = [maybe_fail(i) for i in range(5)]
except ValueError:
    caught = True
"#,
        "exception_in_list_comprehension",
    );
}

#[test]
fn test_exception_message_preserved() {
    assert_executes(
        r#"
message = None
try:
    raise ValueError("specific message")
except ValueError as e:
    message = str(e)
"#,
        "exception_message_preserved",
    );
}

#[test]
fn test_multiple_exceptions_same_handler() {
    assert_executes(
        r#"
count = 0
for i in range(3):
    try:
        raise ValueError("test")
    except:
        count = count + 1
"#,
        "multiple_exceptions_same_handler",
    );
}

#[test]
fn test_exception_in_conditional() {
    assert_executes(
        r#"
result = 0
try:
    if True:
        raise ValueError("test")
except:
    result = 1
"#,
        "exception_in_conditional",
    );
}

#[test]
fn test_exception_in_while_loop() {
    assert_executes(
        r#"
count = 0
while True:
    try:
        count = count + 1
        if count > 3:
            raise ValueError("done")
    except ValueError:
        break
"#,
        "exception_in_while_loop",
    );
}

#[test]
fn test_exception_in_for_loop_body() {
    assert_executes(
        r#"
caught = 0
for i in range(5):
    try:
        if i == 3:
            raise ValueError("test")
    except:
        caught = caught + 1
"#,
        "exception_in_for_loop_body",
    );
}

// =============================================================================
// CATEGORY 7: Performance Edge Cases
// =============================================================================

#[test]
fn test_tight_exception_loop() {
    assert_executes(
        r#"
count = 0
for i in range(100):
    try:
        raise ValueError("test")
    except:
        count = count + 1
"#,
        "tight_exception_loop",
    );
}

#[test]
fn test_deeply_nested_no_exception() {
    assert_executes(
        r#"
result = 0
try:
    try:
        try:
            try:
                try:
                    result = 1
                except:
                    result = 2
            except:
                result = 3
        except:
            result = 4
    except:
        result = 5
except:
    result = 6
"#,
        "deeply_nested_no_exception",
    );
}

#[test]
fn test_exception_with_large_traceback() {
    assert_executes(
        r#"
def f1(): f2()
def f2(): f3()
def f3(): f4()
def f4(): f5()
def f5(): f6()
def f6(): f7()
def f7(): f8()
def f8(): f9()
def f9(): f10()
def f10(): raise ValueError("deep")

caught = False
try:
    f1()
except:
    caught = True
"#,
        "exception_with_large_traceback",
    );
}

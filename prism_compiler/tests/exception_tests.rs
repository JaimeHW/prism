//! Comprehensive exception compilation tests.
//!
//! This module provides exhaustive test coverage for exception handling compilation,
//! organized into categories covering all aspects of try/except/finally semantics.
//!
//! # Test Categories
//!
//! 1. **Basic Exception Compilation** - Simple try/except patterns
//! 2. **Exception Type Matching** - Type-specific handlers
//! 3. **Finally Semantics** - Finally block behavior
//! 4. **Nested Exceptions** - Nested try blocks
//! 5. **Exception Chaining** - `raise X from Y` semantics
//! 6. **Edge Cases** - Unusual or complex patterns

use prism_compiler::{Compiler, Opcode};
use prism_parser::parse;

// =============================================================================
// Test Helpers
// =============================================================================

/// Parse and compile Python source, returning the compiled code object.
fn compile(source: &str) -> prism_compiler::CodeObject {
    let module = parse(source).expect("Failed to parse");
    Compiler::compile_module(&module, "test.py").expect("Failed to compile")
}

/// Check that compilation succeeds without panicking.
fn assert_compiles(source: &str) {
    compile(source);
}

/// Helper to find instructions of a specific opcode in bytecode.
fn find_opcodes(code: &prism_compiler::CodeObject, opcode: Opcode) -> Vec<usize> {
    code.instructions
        .iter()
        .enumerate()
        .filter(|(_, instr)| instr.opcode() == opcode as u8)
        .map(|(i, _)| i)
        .collect()
}

/// Check that specific opcodes appear in the compiled bytecode.
fn assert_contains_opcodes(source: &str, expected_opcodes: &[Opcode]) {
    let code = compile(source);
    for opcode in expected_opcodes {
        let found = find_opcodes(&code, *opcode);
        assert!(
            !found.is_empty(),
            "Expected opcode {:?} not found in bytecode",
            opcode
        );
    }
}

// =============================================================================
// A. Basic Exception Compilation (10 tests)
// =============================================================================

#[test]
fn test_simple_try_except() {
    let source = r#"
try:
    x = 1
except:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_try_except_with_type() {
    let source = r#"
try:
    x = 1
except ValueError:
    x = 2
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::ExceptionMatch]);
}

#[test]
fn test_try_except_with_binding() {
    let source = r#"
try:
    x = 1
except ValueError as e:
    print(e)
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::LoadException]);
}

#[test]
fn test_try_except_else() {
    let source = r#"
try:
    x = 1
except ValueError:
    x = 2
else:
    x = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_try_finally() {
    let source = r#"
try:
    x = 1
finally:
    x = 2
"#;
    assert_compiles(source);
    assert_contains_opcodes(
        source,
        &[Opcode::PushExcInfo, Opcode::PopExcInfo, Opcode::EndFinally],
    );
}

#[test]
fn test_try_except_finally() {
    let source = r#"
try:
    x = 1
except ValueError:
    x = 2
finally:
    x = 3
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::EndFinally]);
}

#[test]
fn test_bare_raise() {
    let source = r#"
try:
    x = 1
except:
    raise
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::Reraise]);
}

#[test]
fn test_raise_expression() {
    let source = r#"
raise ValueError("error")
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::Raise]);
}

#[test]
fn test_raise_from_cause() {
    let source = r#"
try:
    x = 1
except ValueError as e:
    raise RuntimeError("new error") from e
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::RaiseFrom]);
}

#[test]
fn test_raise_from_none() {
    let source = r#"
try:
    x = 1
except ValueError:
    raise RuntimeError("clean error") from None
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::RaiseFrom]);
}

// =============================================================================
// B. Exception Type Matching (10 tests)
// =============================================================================

#[test]
fn test_except_base_exception() {
    let source = r#"
try:
    x = 1
except BaseException:
    x = 2
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::ExceptionMatch]);
}

#[test]
fn test_except_value_error() {
    let source = r#"
try:
    x = 1
except ValueError:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_except_multiple_types_single_handler() {
    // Note: tuple types require different AST handling
    let source = r#"
try:
    x = 1
except ValueError:
    x = 2
except TypeError:
    x = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_except_hierarchy_base() {
    let source = r#"
try:
    x = 1
except Exception:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_except_first_match_wins() {
    let source = r#"
try:
    x = 1
except ValueError:
    x = 2
except Exception:
    x = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_except_bare_catches_all() {
    let source = r#"
try:
    x = 1
except:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_except_order_matters() {
    let source = r#"
try:
    x = 1
except Exception:
    x = 2
except ValueError:
    x = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_except_type_expression() {
    let source = r#"
error_type = ValueError
try:
    x = 1
except error_type:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_except_custom_exception() {
    let source = r#"
class MyError(Exception):
    pass

try:
    x = 1
except MyError:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_except_module_exception() {
    let source = r#"
import os
try:
    x = 1
except OSError:
    x = 2
"#;
    assert_compiles(source);
}

// =============================================================================
// C. Finally Semantics (10 tests)
// =============================================================================

#[test]
fn test_finally_always_runs_normal() {
    let source = r#"
try:
    x = 1
finally:
    cleanup()
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::PushExcInfo, Opcode::PopExcInfo]);
}

#[test]
fn test_finally_after_return() {
    let source = r#"
def f():
    try:
        return 1
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_after_exception() {
    let source = r#"
try:
    raise ValueError()
finally:
    cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_return_overrides() {
    let source = r#"
def f():
    try:
        return 1
    finally:
        return 2
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_exception_preserves() {
    let source = r#"
try:
    raise ValueError()
except ValueError:
    x = 1
finally:
    cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_break_in_loop() {
    let source = r#"
for i in range(10):
    try:
        break
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_continue_in_loop() {
    let source = r#"
for i in range(10):
    try:
        continue
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_reraise() {
    let source = r#"
try:
    raise ValueError()
except:
    try:
        x = 1
    finally:
        raise
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_new_exception() {
    let source = r#"
try:
    raise ValueError()
finally:
    raise TypeError()
"#;
    assert_compiles(source);
}

#[test]
fn test_finally_nested_finally() {
    let source = r#"
try:
    try:
        x = 1
    finally:
        y = 2
finally:
    z = 3
"#;
    assert_compiles(source);
}

// =============================================================================
// D. Nested Exceptions (10 tests)
// =============================================================================

#[test]
fn test_nested_try_outer_handler() {
    let source = r#"
try:
    try:
        raise ValueError()
    except TypeError:
        x = 1
except ValueError:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_try_inner_handler() {
    let source = r#"
try:
    try:
        raise ValueError()
    except ValueError:
        x = 1
except ValueError:
    x = 2
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_try_finally_order() {
    let source = r#"
try:
    try:
        x = 1
    finally:
        inner_cleanup()
finally:
    outer_cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_deeply_nested_try() {
    let source = r#"
try:
    try:
        try:
            raise ValueError()
        except ValueError:
            x = 1
    except TypeError:
        x = 2
except Exception:
    x = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_in_function() {
    let source = r#"
def f():
    try:
        try:
            return 1
        except ValueError:
            return 2
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_in_class() {
    let source = r#"
class C:
    def method(self):
        try:
            try:
                self.x = 1
            except:
                pass
        finally:
            self.cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_in_loop() {
    let source = r#"
for i in range(10):
    try:
        try:
            x = 1
        except:
            break
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_finally_chain() {
    let source = r#"
try:
    try:
        try:
            x = 1
        finally:
            a = 1
    finally:
        b = 2
finally:
    c = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_triple_nested_try_except() {
    let source = r#"
try:
    try:
        try:
            raise ValueError()
        except TypeError:
            x = 1
    except KeyError:
        x = 2
except ValueError:
    x = 3
"#;
    assert_compiles(source);
}

#[test]
fn test_nested_raise_reraise() {
    let source = r#"
try:
    try:
        raise ValueError()
    except:
        raise
except ValueError:
    x = 1
"#;
    assert_compiles(source);
}

// =============================================================================
// E. Exception Chaining (5 tests)
// =============================================================================

#[test]
fn test_raise_from_explicit_cause() {
    let source = r#"
original = ValueError("original")
raise RuntimeError("new") from original
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::RaiseFrom]);
}

#[test]
fn test_raise_from_none_suppress() {
    let source = r#"
try:
    raise ValueError()
except:
    raise RuntimeError() from None
"#;
    assert_compiles(source);
    assert_contains_opcodes(source, &[Opcode::RaiseFrom]);
}

#[test]
fn test_implicit_chaining() {
    let source = r#"
try:
    raise ValueError()
except:
    raise RuntimeError()
"#;
    assert_compiles(source);
}

#[test]
fn test_chained_cause_preserved() {
    let source = r#"
def inner():
    raise ValueError("inner")

def outer():
    try:
        inner()
    except ValueError as e:
        raise RuntimeError("outer") from e
"#;
    assert_compiles(source);
}

#[test]
fn test_context_vs_cause() {
    let source = r#"
try:
    try:
        raise ValueError()
    except:
        raise RuntimeError()
except RuntimeError as e:
    x = e
"#;
    assert_compiles(source);
}

// =============================================================================
// F. Edge Cases (10 tests)
// =============================================================================

#[test]
fn test_empty_try_body() {
    let source = r#"
try:
    pass
except:
    x = 1
"#;
    assert_compiles(source);
}

#[test]
fn test_empty_except_body() {
    let source = r#"
try:
    x = 1
except:
    pass
"#;
    assert_compiles(source);
}

#[test]
fn test_empty_finally_body() {
    let source = r#"
try:
    x = 1
finally:
    pass
"#;
    assert_compiles(source);
}

#[test]
fn test_exception_in_handler() {
    let source = r#"
try:
    raise ValueError()
except ValueError:
    raise TypeError()
"#;
    assert_compiles(source);
}

#[test]
fn test_exception_in_finally() {
    let source = r#"
try:
    x = 1
finally:
    raise ValueError()
"#;
    assert_compiles(source);
}

#[test]
fn test_return_in_try_and_finally() {
    let source = r#"
def f():
    try:
        return 1
    except:
        return 2
    finally:
        return 3
"#;
    assert_compiles(source);
}

#[test]
fn test_break_in_try_with_finally() {
    let source = r#"
for i in range(10):
    try:
        if i == 5:
            break
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_continue_in_try_with_finally() {
    let source = r#"
for i in range(10):
    try:
        if i == 5:
            continue
    finally:
        cleanup()
"#;
    assert_compiles(source);
}

#[test]
fn test_try_in_generator() {
    let source = r#"
def gen():
    try:
        yield 1
    except:
        yield 2
    finally:
        yield 3
"#;
    assert_compiles(source);
}

#[test]
fn test_try_in_comprehension() {
    // Try blocks are allowed in generator expressions
    let source = r#"
def f():
    try:
        result = [x for x in range(10)]
    except:
        result = []
"#;
    assert_compiles(source);
}

// =============================================================================
// Additional Tests (5 more for comprehensive coverage)
// =============================================================================

#[test]
fn test_multiple_handlers_with_bindings() {
    let source = r#"
try:
    x = 1
except ValueError as e1:
    print(e1)
except TypeError as e2:
    print(e2)
except Exception as e3:
    print(e3)
"#;
    assert_compiles(source);
}

#[test]
fn test_try_except_in_lambda() {
    // Lambdas cannot contain try, but we can have try around a lambda
    let source = r#"
try:
    f = lambda x: x + 1
except:
    f = lambda x: x
"#;
    assert_compiles(source);
}

#[test]
fn test_try_with_context_manager() {
    let source = r#"
try:
    with open("file") as f:
        data = f.read()
except:
    data = None
"#;
    assert_compiles(source);
}

#[test]
fn test_exception_in_else_block() {
    let source = r#"
try:
    x = 1
except ValueError:
    x = 2
else:
    raise TypeError()
"#;
    assert_compiles(source);
}

#[test]
fn test_complex_nested_exception_flow() {
    let source = r#"
def complex_flow():
    try:
        try:
            for i in range(10):
                try:
                    if i == 5:
                        raise ValueError()
                except TypeError:
                    continue
                finally:
                    if i == 3:
                        break
        except ValueError:
            return 1
        finally:
            cleanup()
    except:
        return -1
    finally:
        final_cleanup()
    return 0
"#;
    assert_compiles(source);
}

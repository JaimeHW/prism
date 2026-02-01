//! Core Python types module.
//!
//! Contains implementations for built-in Python types.

pub mod dict;
pub mod function;
pub mod iter;
pub mod list;
pub mod range;
pub mod set;
pub mod string;
pub mod tuple;

// Re-export commonly used types
pub use dict::DictObject;
pub use function::FunctionObject;
pub use list::ListObject;
pub use range::RangeObject;
pub use set::SetObject;
pub use string::{SSO_MAX_LEN, StringObject, empty_string};
pub use tuple::TupleObject;

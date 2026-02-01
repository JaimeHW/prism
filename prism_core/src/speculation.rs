//! Speculation hints for JIT compilation.
//!
//! This module defines type speculation hints that can be shared between
//! the VM (which collects type feedback) and the JIT (which uses it for
//! type-specialized code generation).

/// Type speculation hint for binary operations.
///
/// These hints are derived from runtime type feedback and inform the JIT
/// about observed operand types to enable speculative compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TypeHint {
    /// No type information available (cold or polymorphic site).
    #[default]
    None = 0,
    /// Both operands are integers.
    IntInt = 1,
    /// Both operands are floats.
    FloatFloat = 2,
    /// Left operand is int, right is float.
    IntFloat = 3,
    /// Left operand is float, right is int.
    FloatInt = 4,
    /// Both operands are strings.
    StrStr = 5,
    /// String and int (for repetition).
    StrInt = 6,
    /// Int and string (for repetition).
    IntStr = 7,
}

impl TypeHint {
    /// Check if this hint suggests integer operations.
    #[inline]
    pub const fn is_int(self) -> bool {
        matches!(self, Self::IntInt)
    }

    /// Check if this hint suggests float operations.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::FloatFloat | Self::IntFloat | Self::FloatInt)
    }

    /// Check if this hint suggests string operations.
    #[inline]
    pub const fn is_string(self) -> bool {
        matches!(self, Self::StrStr | Self::StrInt | Self::IntStr)
    }

    /// Check if this hint is valid (not None).
    #[inline]
    pub const fn is_valid(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Provider of type speculation hints for JIT compilation.
///
/// This trait allows the JIT to query speculation information without
/// direct dependency on VM internals.
pub trait SpeculationProvider {
    /// Get the type hint for a specific bytecode site.
    ///
    /// # Arguments
    /// * `code_id` - Unique identifier for the compiled code unit
    /// * `bc_offset` - Bytecode offset within the code unit
    ///
    /// # Returns
    /// The observed type hint, or `TypeHint::None` if no information is available.
    fn get_type_hint(&self, code_id: u32, bc_offset: u32) -> TypeHint;
}

/// No-op speculation provider that always returns None.
///
/// Useful for testing or when speculation data is not available.
#[derive(Debug, Default)]
pub struct NoSpeculation;

impl SpeculationProvider for NoSpeculation {
    #[inline]
    fn get_type_hint(&self, _code_id: u32, _bc_offset: u32) -> TypeHint {
        TypeHint::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_hint_is_int() {
        assert!(TypeHint::IntInt.is_int());
        assert!(!TypeHint::FloatFloat.is_int());
        assert!(!TypeHint::StrStr.is_int());
        assert!(!TypeHint::None.is_int());
    }

    #[test]
    fn test_type_hint_is_float() {
        assert!(TypeHint::FloatFloat.is_float());
        assert!(TypeHint::IntFloat.is_float());
        assert!(TypeHint::FloatInt.is_float());
        assert!(!TypeHint::IntInt.is_float());
        assert!(!TypeHint::StrStr.is_float());
    }

    #[test]
    fn test_type_hint_is_string() {
        assert!(TypeHint::StrStr.is_string());
        assert!(TypeHint::StrInt.is_string());
        assert!(TypeHint::IntStr.is_string());
        assert!(!TypeHint::IntInt.is_string());
        assert!(!TypeHint::FloatFloat.is_string());
    }

    #[test]
    fn test_type_hint_is_valid() {
        assert!(!TypeHint::None.is_valid());
        assert!(TypeHint::IntInt.is_valid());
        assert!(TypeHint::FloatFloat.is_valid());
        assert!(TypeHint::StrStr.is_valid());
    }

    #[test]
    fn test_no_speculation_provider() {
        let provider = NoSpeculation;
        assert_eq!(provider.get_type_hint(1, 0), TypeHint::None);
        assert_eq!(provider.get_type_hint(100, 50), TypeHint::None);
    }
}

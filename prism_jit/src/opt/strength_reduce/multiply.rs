//! Multiplication Decomposition
//!
//! This module implements decomposition of multiplication by constants into
//! sequences of shifts and adds. This is often faster than hardware multiply
//! for small constants on some architectures.
//!
//! # Examples
//!
//! ```text
//! x * 3  = x + (x << 1)           // 1 shift, 1 add
//! x * 5  = x + (x << 2)           // 1 shift, 1 add
//! x * 7  = (x << 3) - x           // 1 shift, 1 sub
//! x * 9  = x + (x << 3)           // 1 shift, 1 add
//! x * 10 = (x << 1) + (x << 3)    // 2 shifts, 1 add
//! x * 15 = (x << 4) - x           // 1 shift, 1 sub
//! ```
//!
//! # Cost Model
//!
//! - Shift: 1 cycle
//! - Add/Sub: 1 cycle
//! - Multiply: 3-4 cycles (modern x86-64)
//!
//! We only decompose if the decomposed sequence is cheaper than multiply.

// =============================================================================
// Decomposition Result
// =============================================================================

/// A decomposition of multiplication into shifts and adds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MulDecomposition {
    /// The operations to perform in order.
    pub ops: Vec<DecompOp>,
    /// Whether the final result is negated.
    pub negate_result: bool,
}

/// A single operation in a multiplication decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompOp {
    /// Shift left by n bits: x << n
    Shift(u8),
    /// Add shifted value: accumulator + (x << n)
    AddShift(u8),
    /// Subtract shifted value: accumulator - (x << n)
    SubShift(u8),
    /// Add x to accumulator: accumulator + x
    Add,
    /// Subtract x from accumulator: accumulator - x
    Sub,
    /// Negate: -accumulator
    Negate,
}

impl MulDecomposition {
    /// Create a new decomposition.
    fn new() -> Self {
        Self {
            ops: Vec::with_capacity(4),
            negate_result: false,
        }
    }

    /// Create a decomposition that's just a shift.
    fn shift(n: u8) -> Self {
        Self {
            ops: vec![DecompOp::Shift(n)],
            negate_result: false,
        }
    }

    /// Get the cost of this decomposition.
    ///
    /// Each shift/add/sub costs 1 cycle.
    pub fn cost(&self) -> u32 {
        self.ops.len() as u32 + if self.negate_result { 1 } else { 0 }
    }

    /// Check if this decomposition is cheaper than a multiply.
    pub fn is_profitable(&self, multiply_cost: u32) -> bool {
        self.cost() < multiply_cost
    }
}

// =============================================================================
// Decomposition Algorithm
// =============================================================================

/// Configuration for multiplication decomposition.
#[derive(Clone, Copy, Debug)]
pub struct DecompConfig {
    /// Maximum number of operations allowed.
    pub max_ops: u32,
    /// Cost of a hardware multiply instruction.
    pub multiply_cost: u32,
    /// Whether to allow decomposition that requires negation.
    pub allow_negate: bool,
}

impl Default for DecompConfig {
    fn default() -> Self {
        Self {
            max_ops: 4,
            multiply_cost: 4, // Typical for modern x86-64
            allow_negate: true,
        }
    }
}

impl DecompConfig {
    /// Configuration optimized for minimal latency.
    pub fn low_latency() -> Self {
        Self {
            max_ops: 2,
            multiply_cost: 3,
            allow_negate: false,
        }
    }

    /// Configuration that allows more complex decompositions.
    pub fn aggressive() -> Self {
        Self {
            max_ops: 6,
            multiply_cost: 5,
            allow_negate: true,
        }
    }
}

/// Decompose multiplication by a constant into shifts and adds.
///
/// Returns `None` if no profitable decomposition exists.
pub fn decompose_multiply(n: i64, config: &DecompConfig) -> Option<MulDecomposition> {
    // Handle special cases
    if n == 0 {
        // x * 0 = 0 (should be handled by constant folding, not here)
        return None;
    }

    if n == 1 {
        // x * 1 = x (identity, no decomposition needed)
        return None;
    }

    if n == -1 {
        // x * -1 = -x
        return Some(MulDecomposition {
            ops: vec![DecompOp::Negate],
            negate_result: false,
        });
    }

    // Handle negative constants
    let (abs_n, negate) = if n < 0 {
        if !config.allow_negate {
            return None;
        }
        ((-n) as u64, true)
    } else {
        (n as u64, false)
    };

    // Try to decompose the absolute value
    let mut decomp = decompose_unsigned(abs_n, config)?;

    if negate {
        decomp.negate_result = true;
    }

    // Only return if profitable
    if decomp.is_profitable(config.multiply_cost) {
        Some(decomp)
    } else {
        None
    }
}

/// Decompose multiplication by an unsigned constant.
fn decompose_unsigned(n: u64, config: &DecompConfig) -> Option<MulDecomposition> {
    // Power of 2: just shift
    if n.is_power_of_two() {
        let shift = n.trailing_zeros() as u8;
        return Some(MulDecomposition::shift(shift));
    }

    // Try simple patterns first (most common cases)

    // Pattern: 2^k + 1 (e.g., 3, 5, 9, 17, ...)
    // x * (2^k + 1) = x + (x << k)
    if let Some(k) = is_power_of_two_plus_one(n) {
        return Some(MulDecomposition {
            ops: vec![DecompOp::Shift(k), DecompOp::Add],
            negate_result: false,
        });
    }

    // Pattern: 2^k - 1 (e.g., 3, 7, 15, 31, ...)
    // x * (2^k - 1) = (x << k) - x
    if let Some(k) = is_power_of_two_minus_one(n) {
        return Some(MulDecomposition {
            ops: vec![DecompOp::Shift(k), DecompOp::Sub],
            negate_result: false,
        });
    }

    // Pattern: 2^j + 2^k where j > k
    // x * (2^j + 2^k) = (x << j) + (x << k)
    if let Some((j, k)) = is_sum_of_two_powers(n) {
        return Some(MulDecomposition {
            ops: vec![DecompOp::Shift(j), DecompOp::AddShift(k)],
            negate_result: false,
        });
    }

    // Pattern: 2^j - 2^k where j > k
    // x * (2^j - 2^k) = (x << j) - (x << k)
    if let Some((j, k)) = is_diff_of_two_powers(n) {
        return Some(MulDecomposition {
            ops: vec![DecompOp::Shift(j), DecompOp::SubShift(k)],
            negate_result: false,
        });
    }

    // Try factorization for composite numbers
    if config.max_ops >= 3 {
        if let Some(decomp) = try_factorization(n, config) {
            return Some(decomp);
        }
    }

    // Try Booth-like decomposition for more complex cases
    if config.max_ops >= 4 {
        if let Some(decomp) = booth_decompose(n, config) {
            return Some(decomp);
        }
    }

    None
}

/// Check if n = 2^k + 1, return k.
fn is_power_of_two_plus_one(n: u64) -> Option<u8> {
    if n < 3 {
        return None;
    }
    let m = n - 1;
    if m.is_power_of_two() {
        Some(m.trailing_zeros() as u8)
    } else {
        None
    }
}

/// Check if n = 2^k - 1, return k.
fn is_power_of_two_minus_one(n: u64) -> Option<u8> {
    if n == 0 {
        return None;
    }
    let m = n + 1;
    if m.is_power_of_two() {
        Some(m.trailing_zeros() as u8)
    } else {
        None
    }
}

/// Check if n = 2^j + 2^k where j > k, return (j, k).
fn is_sum_of_two_powers(n: u64) -> Option<(u8, u8)> {
    if n.count_ones() != 2 {
        return None;
    }
    let k = n.trailing_zeros() as u8;
    let j = (63 - n.leading_zeros()) as u8;
    Some((j, k))
}

/// Check if n = 2^j - 2^k where j > k, return (j, k).
fn is_diff_of_two_powers(n: u64) -> Option<(u8, u8)> {
    // n = 2^j - 2^k = 2^k * (2^(j-k) - 1)
    // This means n has the form: 111...110...0 in binary
    // (all 1s followed by all 0s)
    if n == 0 {
        return None;
    }

    let k = n.trailing_zeros() as u8;
    let m = n >> k; // Should be 2^(j-k) - 1, i.e., all 1s

    // Check if m is all 1s (2^t - 1)
    if m == 0 {
        return None;
    }
    let t = m.count_ones();
    if m == (1u64 << t) - 1 {
        let j = k + t as u8;
        Some((j, k))
    } else {
        None
    }
}

/// Try to factor n and build a decomposition from factors.
///
/// This does NOT recurse into decompose_unsigned to avoid stack overflow.
fn try_factorization(n: u64, config: &DecompConfig) -> Option<MulDecomposition> {
    // Only try small factors
    const SMALL_PRIMES: [u64; 5] = [3, 5, 7, 9, 11];

    for &p in &SMALL_PRIMES {
        if n % p == 0 {
            let q = n / p;

            // Check if q is a power of 2
            if q.is_power_of_two() {
                let shift = q.trailing_zeros() as u8;

                // x * n = (x * p) << shift
                // Get a simple decomposition of p (no recursion)
                let p_decomp = simple_decompose(p)?;
                let mut ops = p_decomp.ops;

                // Apply shift to result
                if shift > 0 {
                    // If the last op is a simple value, we can modify it
                    // Otherwise append a separate shift operation
                    // For now, just return None if too complex
                    if ops.len() + 1 <= config.max_ops as usize {
                        // This is a simplification - ideally we'd fold the shift
                        ops.push(DecompOp::Shift(shift));
                    } else {
                        return None;
                    }
                }

                if ops.len() <= config.max_ops as usize {
                    return Some(MulDecomposition {
                        ops,
                        negate_result: false,
                    });
                }
            }
        }
    }

    None
}

/// Simple decomposition without recursion - only handles basic patterns.
fn simple_decompose(n: u64) -> Option<MulDecomposition> {
    if n.is_power_of_two() {
        let shift = n.trailing_zeros() as u8;
        return Some(MulDecomposition::shift(shift));
    }

    // Pattern: 2^k + 1 (e.g., 3, 5, 9, 17, ...)
    if let Some(k) = is_power_of_two_plus_one(n) {
        return Some(MulDecomposition {
            ops: vec![DecompOp::Shift(k), DecompOp::Add],
            negate_result: false,
        });
    }

    // Pattern: 2^k - 1 (e.g., 3, 7, 15, 31, ...)
    if let Some(k) = is_power_of_two_minus_one(n) {
        return Some(MulDecomposition {
            ops: vec![DecompOp::Shift(k), DecompOp::Sub],
            negate_result: false,
        });
    }

    None
}

/// Booth-like decomposition using signed digit representation.
///
/// This uses a non-adjacent form (NAF) or similar to minimize operations.
fn booth_decompose(n: u64, config: &DecompConfig) -> Option<MulDecomposition> {
    // Convert to non-adjacent form (NAF)
    // NAF has no adjacent non-zero digits
    let naf = to_naf(n);

    // Count non-zero digits
    let nonzero_count = naf.iter().filter(|&&d| d != 0).count();

    // Each non-zero digit except the first needs an add/sub
    if nonzero_count > config.max_ops as usize {
        return None;
    }

    // Build the decomposition
    let mut ops = Vec::with_capacity(nonzero_count * 2);
    let mut first = true;

    for (i, &digit) in naf.iter().enumerate() {
        if digit == 0 {
            continue;
        }

        let shift = i as u8;

        if first {
            // First non-zero: start with shift
            ops.push(DecompOp::Shift(shift));
            if digit < 0 {
                ops.push(DecompOp::Negate);
            }
            first = false;
        } else {
            // Subsequent: add or subtract shifted value
            if digit > 0 {
                ops.push(DecompOp::AddShift(shift));
            } else {
                ops.push(DecompOp::SubShift(shift));
            }
        }
    }

    if ops.len() <= config.max_ops as usize {
        Some(MulDecomposition {
            ops,
            negate_result: false,
        })
    } else {
        None
    }
}

/// Convert to non-adjacent form (NAF).
///
/// NAF representation uses digits {-1, 0, 1} with no adjacent non-zero digits.
fn to_naf(mut n: u64) -> Vec<i8> {
    let mut result = Vec::with_capacity(65);

    // Handle 0 early
    if n == 0 {
        return result;
    }

    while n > 0 {
        if n & 1 != 0 {
            // Odd: determine digit
            // If n ends in ...11, we set digit = -1 and add 1 to n
            // If n ends in ...01, we set digit = 1 and subtract 1 from n
            let digit = if n & 2 != 0 { -1i8 } else { 1i8 };
            result.push(digit);

            if digit == 1 {
                n = (n - 1) >> 1;
            } else {
                // digit == -1
                n = (n + 1) >> 1;
            }
        } else {
            result.push(0);
            n >>= 1;
        }
    }

    result
}

// =============================================================================
// Apply Decomposition (for verification)
// =============================================================================

impl MulDecomposition {
    /// Apply this decomposition to compute x * n.
    ///
    /// This is used for testing/verification.
    pub fn apply(&self, x: i64) -> i64 {
        let mut result = 0i64;
        let mut acc = 0i64;

        for op in &self.ops {
            match *op {
                DecompOp::Shift(n) => {
                    acc = x << n;
                    result = acc;
                }
                DecompOp::AddShift(n) => {
                    result = acc.wrapping_add(x << n);
                    acc = result;
                }
                DecompOp::SubShift(n) => {
                    result = acc.wrapping_sub(x << n);
                    acc = result;
                }
                DecompOp::Add => {
                    result = acc.wrapping_add(x);
                    acc = result;
                }
                DecompOp::Sub => {
                    result = acc.wrapping_sub(x);
                    acc = result;
                }
                DecompOp::Negate => {
                    result = -acc;
                    acc = result;
                }
            }
        }

        if self.negate_result {
            -result
        } else {
            result
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Pattern Detection Tests
    // =========================================================================

    #[test]
    fn test_power_of_two_plus_one() {
        assert_eq!(is_power_of_two_plus_one(3), Some(1)); // 2^1 + 1
        assert_eq!(is_power_of_two_plus_one(5), Some(2)); // 2^2 + 1
        assert_eq!(is_power_of_two_plus_one(9), Some(3)); // 2^3 + 1
        assert_eq!(is_power_of_two_plus_one(17), Some(4)); // 2^4 + 1
        assert_eq!(is_power_of_two_plus_one(7), None);
        assert_eq!(is_power_of_two_plus_one(10), None);
    }

    #[test]
    fn test_power_of_two_minus_one() {
        assert_eq!(is_power_of_two_minus_one(1), Some(1)); // 2^1 - 1
        assert_eq!(is_power_of_two_minus_one(3), Some(2)); // 2^2 - 1
        assert_eq!(is_power_of_two_minus_one(7), Some(3)); // 2^3 - 1
        assert_eq!(is_power_of_two_minus_one(15), Some(4)); // 2^4 - 1
        assert_eq!(is_power_of_two_minus_one(31), Some(5)); // 2^5 - 1
        assert_eq!(is_power_of_two_minus_one(10), None);
    }

    #[test]
    fn test_sum_of_two_powers() {
        assert_eq!(is_sum_of_two_powers(3), Some((1, 0))); // 2^1 + 2^0
        assert_eq!(is_sum_of_two_powers(5), Some((2, 0))); // 2^2 + 2^0
        assert_eq!(is_sum_of_two_powers(6), Some((2, 1))); // 2^2 + 2^1
        assert_eq!(is_sum_of_two_powers(10), Some((3, 1))); // 2^3 + 2^1
        assert_eq!(is_sum_of_two_powers(7), None); // 3 bits set
        assert_eq!(is_sum_of_two_powers(8), None); // 1 bit set
    }

    #[test]
    fn test_diff_of_two_powers() {
        assert_eq!(is_diff_of_two_powers(2), Some((2, 1))); // 4 - 2
        assert_eq!(is_diff_of_two_powers(6), Some((3, 1))); // 8 - 2
        assert_eq!(is_diff_of_two_powers(12), Some((4, 2))); // 16 - 4
        assert_eq!(is_diff_of_two_powers(14), Some((4, 1))); // 16 - 2
        assert_eq!(is_diff_of_two_powers(5), None);
    }

    // =========================================================================
    // NAF Conversion Tests
    // =========================================================================

    #[test]
    fn test_naf_simple() {
        // 3 = 4 - 1 = 2^2 - 2^0, NAF = [-1, 0, 1]
        let naf = to_naf(3);
        assert_eq!(naf, vec![-1, 0, 1]);

        // 7 = 8 - 1, NAF = [-1, 0, 0, 1]
        let naf = to_naf(7);
        assert_eq!(naf, vec![-1, 0, 0, 1]);
    }

    #[test]
    fn test_naf_no_adjacent_nonzero() {
        for n in 1u64..1000 {
            let naf = to_naf(n);

            // Verify no adjacent non-zero digits
            for i in 0..naf.len().saturating_sub(1) {
                if naf[i] != 0 && naf[i + 1] != 0 {
                    panic!("NAF has adjacent non-zero digits for n={}: {:?}", n, naf);
                }
            }

            // Verify it computes the right value
            let mut computed = 0i64;
            for (i, &d) in naf.iter().enumerate() {
                computed += (d as i64) << i;
            }
            assert_eq!(computed, n as i64, "NAF mismatch for n={}", n);
        }
    }

    // =========================================================================
    // Decomposition Tests
    // =========================================================================

    #[test]
    fn test_decompose_special_cases() {
        let config = DecompConfig::default();

        // x * 0 - not decomposed here
        assert!(decompose_multiply(0, &config).is_none());

        // x * 1 - identity
        assert!(decompose_multiply(1, &config).is_none());

        // x * -1 - negate
        let decomp = decompose_multiply(-1, &config).unwrap();
        assert_eq!(decomp.ops, vec![DecompOp::Negate]);
    }

    #[test]
    fn test_decompose_powers_of_two() {
        let config = DecompConfig::default();

        for shift in 1u8..62 {
            let n = 1i64 << shift;
            let decomp = decompose_multiply(n, &config).unwrap();
            assert_eq!(decomp.ops, vec![DecompOp::Shift(shift)]);
            assert!(!decomp.negate_result);

            // Verify correctness (use wrapping to avoid overflow)
            for x in [1i64, 7, 100, -50] {
                assert_eq!(decomp.apply(x), x.wrapping_mul(n));
            }
        }
    }

    #[test]
    fn test_decompose_small_constants() {
        let config = DecompConfig::default();

        // Test 2-20
        for n in 2i64..=20 {
            let decomp_opt = decompose_multiply(n, &config);

            // We expect most small constants to decompose
            if let Some(decomp) = decomp_opt {
                // Verify correctness
                for x in [0i64, 1, -1, 7, -7, 100, -100, 12345] {
                    let expected = x.wrapping_mul(n);
                    let actual = decomp.apply(x);
                    assert_eq!(
                        actual, expected,
                        "Decomposition failed for x={}, n={}: got {}, expected {}",
                        x, n, actual, expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_decompose_negative_constants() {
        let config = DecompConfig::default();

        for n in [-2i64, -3, -5, -7, -10, -15] {
            let decomp = decompose_multiply(n, &config).unwrap();
            assert!(decomp.negate_result);

            for x in [1i64, 7, -7, 100] {
                let expected = x.wrapping_mul(n);
                let actual = decomp.apply(x);
                assert_eq!(actual, expected, "Failed for x={}, n={}", x, n);
            }
        }
    }

    #[test]
    fn test_decompose_specific_patterns() {
        let config = DecompConfig::default();

        // x * 3 = x + (x << 1)
        let d3 = decompose_multiply(3, &config).unwrap();
        assert!(d3.cost() <= 2);
        assert_eq!(d3.apply(10), 30);

        // x * 5 = x + (x << 2)
        let d5 = decompose_multiply(5, &config).unwrap();
        assert!(d5.cost() <= 2);
        assert_eq!(d5.apply(10), 50);

        // x * 7 = (x << 3) - x
        let d7 = decompose_multiply(7, &config).unwrap();
        assert!(d7.cost() <= 2);
        assert_eq!(d7.apply(10), 70);

        // x * 9 = x + (x << 3)
        let d9 = decompose_multiply(9, &config).unwrap();
        assert!(d9.cost() <= 2);
        assert_eq!(d9.apply(10), 90);

        // x * 15 = (x << 4) - x
        let d15 = decompose_multiply(15, &config).unwrap();
        assert!(d15.cost() <= 2);
        assert_eq!(d15.apply(10), 150);
    }

    #[test]
    fn test_decompose_cost() {
        let config = DecompConfig::default();

        // Power of 2 should be cheapest (1 op)
        let d2 = decompose_multiply(2, &config).unwrap();
        assert_eq!(d2.cost(), 1);

        // 2^k + 1 should be 2 ops
        let d3 = decompose_multiply(3, &config).unwrap();
        assert!(d3.cost() <= 2);

        // 2^k - 1 should be 2 ops
        let d7 = decompose_multiply(7, &config).unwrap();
        assert!(d7.cost() <= 2);
    }

    #[test]
    fn test_decompose_profitability() {
        // With low multiply cost, fewer decompositions are profitable
        let low_cost = DecompConfig {
            max_ops: 4,
            multiply_cost: 2,
            allow_negate: true,
        };

        // Power of 2 still profitable
        assert!(decompose_multiply(4, &low_cost).is_some());

        // More complex might not be
        let d = decompose_multiply(10, &low_cost);
        if let Some(decomp) = d {
            assert!(decomp.is_profitable(2));
        }
    }

    #[test]
    fn test_decompose_comprehensive() {
        let config = DecompConfig::default();

        // Test many constants
        for n in 2i64..500 {
            if let Some(decomp) = decompose_multiply(n, &config) {
                // Verify correctness on multiple inputs
                let test_values = [0i64, 1, -1, 10, -10, 127, -128, 1000, -1000];

                for x in test_values {
                    let expected = x.wrapping_mul(n);
                    let actual = decomp.apply(x);
                    assert_eq!(
                        actual, expected,
                        "Decomposition failed for x={}, n={}: got {}, expected {}. Decomp: {:?}",
                        x, n, actual, expected, decomp
                    );
                }
            }
        }
    }

    #[test]
    fn test_decompose_config_low_latency() {
        let config = DecompConfig::low_latency();

        // Should only allow very simple decompositions
        assert_eq!(config.max_ops, 2);

        // Power of 2 works
        assert!(decompose_multiply(4, &config).is_some());

        // Simple patterns work
        assert!(decompose_multiply(3, &config).is_some());
    }

    #[test]
    fn test_decompose_config_aggressive() {
        let config = DecompConfig::aggressive();

        // Should allow more complex decompositions
        assert!(config.max_ops >= 6);

        // More constants should decompose
        let count = (2i64..100)
            .filter(|&n| decompose_multiply(n, &config).is_some())
            .count();

        // Most should decompose with aggressive settings
        assert!(count > 50);
    }

    #[test]
    fn test_decomp_op_coverage() {
        let config = DecompConfig::aggressive();

        // Find a decomposition that uses each op type
        let mut found_add_shift = false;
        let mut found_sub_shift = false;

        for n in 2i64..1000 {
            if let Some(decomp) = decompose_multiply(n, &config) {
                for op in &decomp.ops {
                    match op {
                        DecompOp::AddShift(_) => found_add_shift = true,
                        DecompOp::SubShift(_) => found_sub_shift = true,
                        _ => {}
                    }
                }
            }
        }

        assert!(found_add_shift, "No AddShift operations found");
        assert!(found_sub_shift, "No SubShift operations found");
    }
}

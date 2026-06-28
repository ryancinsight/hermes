//! Value-semantic contract tests for the integer `NumericElement` impls.
//!
//! Covers the signed (`i8`/`i16`/`i32`/`i64`) and unsigned (`u8`/`u16`/`u32`/
//! `u64`) implementations, cross-checking every trait operation against `std`
//! semantics rather than asserting mere existence.

use hermes_numeric::{CastFrom, NumericElement};

/// Assert the full integer `NumericElement` contract for one type.
///
/// `$abs_in`/`$abs_out` parameterize the single sign-dependent case (`abs`):
/// signed invocations pass a negative input, unsigned a non-negative one.
macro_rules! integer_element_contract {
    ($name:ident, $t:ty, abs($abs_in:expr) == $abs_out:expr) => {
        #[test]
        fn $name() {
            // Identity / boundary constants match std.
            assert_eq!(<$t as NumericElement>::ZERO, 0 as $t);
            assert_eq!(<$t as NumericElement>::ONE, 1 as $t);
            assert_eq!(<$t as NumericElement>::MIN_VALUE, <$t>::MIN);
            assert_eq!(<$t as NumericElement>::MAX_VALUE, <$t>::MAX);
            assert_eq!(<$t as NumericElement>::ALL_ONES, !(0 as $t));
            assert_eq!(
                <$t as NumericElement>::BYTE_WIDTH,
                core::mem::size_of::<$t>()
            );

            // Identity arithmetic.
            let a: $t = 7;
            assert_eq!(a + <$t as NumericElement>::ZERO, a);
            assert_eq!(a * <$t as NumericElement>::ONE, a);

            // Bitwise ops and popcount match the std operators bit-for-bit.
            let x: $t = 0b1010;
            let y: $t = 0b0110;
            assert_eq!(NumericElement::bitand(x, y), x & y);
            assert_eq!(NumericElement::bitor(x, y), x | y);
            assert_eq!(NumericElement::bitxor(x, y), x ^ y);
            assert_eq!(NumericElement::count_ones(x), x.count_ones());

            // Ordering reductions return the correct operand.
            assert_eq!(NumericElement::min_scalar(a, 3 as $t), 3 as $t);
            assert_eq!(NumericElement::max_scalar(a, 3 as $t), 7 as $t);

            // Fused multiply-add follows the documented wrapping contract.
            let (b, c): ($t, $t) = (3, 4);
            assert_eq!(a.scalar_fmadd(b, c), a.wrapping_mul(b).wrapping_add(c));
            assert_eq!(a.scalar_fmadd(b, c), 25 as $t);

            // Integers are always finite, never NaN.
            assert!(NumericElement::is_finite(a));
            assert!(!NumericElement::is_nan(a));

            // Lossless widening to f64.
            assert_eq!(NumericElement::to_f64(a), 7.0_f64);

            // Integer (floor) square root is exact (no f64 round-trip). 16 fits
            // every width including i8/u8.
            assert_eq!(NumericElement::sqrt(16 as $t), 4 as $t);
            assert_eq!(NumericElement::sqrt(15 as $t), 3 as $t);
            assert_eq!(NumericElement::sqrt(1 as $t), 1 as $t);
            assert_eq!(
                NumericElement::sqrt(<$t as NumericElement>::ZERO),
                <$t as NumericElement>::ZERO
            );

            // Sign-dependent absolute value.
            assert_eq!(NumericElement::abs($abs_in as $t), $abs_out as $t);

            // CastFrom<i32> maps an in-range value exactly.
            assert_eq!(<$t as CastFrom<i32>>::cast_from(5_i32), 5 as $t);
        }
    };
}

integer_element_contract!(i8_element_contract, i8, abs(-5) == 5);
integer_element_contract!(i16_element_contract, i16, abs(-5) == 5);
integer_element_contract!(i32_element_contract, i32, abs(-5) == 5);
integer_element_contract!(i64_element_contract, i64, abs(-5) == 5);
integer_element_contract!(u8_element_contract, u8, abs(5) == 5);
integer_element_contract!(u16_element_contract, u16, abs(5) == 5);
integer_element_contract!(u32_element_contract, u32, abs(5) == 5);
integer_element_contract!(u64_element_contract, u64, abs(5) == 5);

/// Integer `sqrt` is exact for large operands where the former
/// `(self as f64).sqrt() as Self` path lost precision (the operand rounds to f64
/// before the root above 2^53). These are the regression cases.
#[test]
fn integer_sqrt_is_exact_for_large_operands() {
    // u64::MAX = 2^64−1; floor(sqrt) = 2^32−1 = 4_294_967_295. The f64 path
    // returned 2^32 = 4_294_967_296, whose square overflows u64 — provably wrong.
    assert_eq!(NumericElement::sqrt(u64::MAX), 4_294_967_295_u64);
    assert_eq!(
        u64::MAX.isqrt(),
        4_294_967_295_u64,
        "std isqrt is the oracle"
    );

    // i64::MAX = 2^63−1; floor(sqrt) = 3_037_000_499.
    assert_eq!(NumericElement::sqrt(i64::MAX), 3_037_000_499_i64);

    // Defining property r² ≤ n < (r+1)² on an operand above 2^53 (no overflow in
    // the check: r ≈ 2^20, (r+1)² ≪ u64::MAX).
    let n: u64 = (1u64 << 40) + 7;
    let r = NumericElement::sqrt(n);
    assert!(
        r * r <= n && (r + 1) * (r + 1) > n,
        "isqrt invariant for {n}"
    );
}

/// Negative signed inputs have no real root and no NaN; the documented contract
/// returns 0 (total, non-panicking — `isqrt` itself would panic on negatives).
#[test]
fn signed_integer_sqrt_of_negative_is_zero() {
    assert_eq!(NumericElement::sqrt(-4_i32), 0_i32);
    assert_eq!(NumericElement::sqrt(-1_i64), 0_i64);
    assert_eq!(NumericElement::sqrt(i64::MIN), 0_i64);
}

/// Cross-width `CastFrom` round-trips for an in-range value preserve it exactly.
#[test]
fn cross_width_cast_round_trip() {
    let v: u8 = 200;
    let widened = u32::cast_from(v);
    assert_eq!(widened, 200_u32);
    assert_eq!(u8::cast_from(widened), v);

    let s: i64 = -42;
    let narrowed = i32::cast_from(s);
    assert_eq!(narrowed, -42_i32);
    assert_eq!(i64::cast_from(narrowed), s);
}

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

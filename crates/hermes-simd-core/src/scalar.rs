//! Sealed element-type trait for SIMD-eligible scalars.

use crate::private::Sealed;

use eunomia::convert::IntegerTarget;
/// Backward-compatible alias for `NumericElement`.
pub use eunomia::NumericElement as Scalar;
pub use eunomia::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8, I16, I32, I8};
pub use eunomia::{FloatElement, NumericElement};

/// Float scalars with an exact round-to-nearest-ties-to-even primitive.
///
/// Backs [`crate::kernel::SimdArith::round`], whose contract is banker's
/// rounding: x86 `roundps`/`vrndscaleps` `_MM_FROUND_TO_NEAREST_INT` and NEON
/// `FRINTN` both resolve exact halfway values to the even neighbor. eunomia's
/// [`FloatElement::round`] follows libm's `roundf` (half-away-from-zero), so it
/// cannot back the kernel default; this sealed trait routes each float type to
/// the matching half-even primitive — `f32`/`f64` to their inherent
/// `round_ties_even` (`core`, stable 1.77), and the reduced-precision wrappers
/// to the exact `from_f32`/`from_f64` round-narrow path (`F16`/`Bf16` and the
/// 8-/4-bit types represent every integer their exponent range allows, so the
/// round-trip is exact).
pub trait RoundTiesEven: Sealed + FloatElement {
    /// Round to the nearest integer, ties to the even neighbor.
    #[must_use]
    fn round_ties_even(self) -> Self;
}

impl Sealed for f32 {}
impl Sealed for f64 {}
impl Sealed for I32 {}
impl Sealed for I16 {}
impl Sealed for I8 {}
impl Sealed for i32 {}
impl Sealed for i16 {}
impl Sealed for i8 {}
impl Sealed for F16 {}
impl Sealed for F32 {}
impl Sealed for Bf16 {}
impl Sealed for F64 {}
impl Sealed for F8 {}
impl Sealed for F4 {}
impl Sealed for Bf8 {}
impl Sealed for Bf4 {}

impl RoundTiesEven for f32 {
    fn round_ties_even(self) -> Self {
        f32::round_ties_even(self)
    }
}

impl RoundTiesEven for f64 {
    fn round_ties_even(self) -> Self {
        f64::round_ties_even(self)
    }
}

impl RoundTiesEven for F32 {
    fn round_ties_even(self) -> Self {
        Self::from_f32(f32::round_ties_even(self.to_f32()))
    }
}

impl RoundTiesEven for F64 {
    fn round_ties_even(self) -> Self {
        Self::from_f64(f64::round_ties_even(self.to_f64()))
    }
}

impl RoundTiesEven for F16 {
    fn round_ties_even(self) -> Self {
        Self::from_f32(f32::round_ties_even(self.to_f32()))
    }
}

impl RoundTiesEven for Bf16 {
    fn round_ties_even(self) -> Self {
        Self::from_f32(f32::round_ties_even(self.to_f32()))
    }
}

impl RoundTiesEven for F8 {
    fn round_ties_even(self) -> Self {
        Self::from_f32(f32::round_ties_even(self.to_f32()))
    }
}

impl RoundTiesEven for F4 {
    fn round_ties_even(self) -> Self {
        Self::from_f32(f32::round_ties_even(self.to_f32()))
    }
}

impl RoundTiesEven for Bf8 {
    fn round_ties_even(self) -> Self {
        Self::from_f32(f32::round_ties_even(self.to_f32()))
    }
}

impl RoundTiesEven for Bf4 {
    fn round_ties_even(self) -> Self {
        Self::from_f32(f32::round_ties_even(self.to_f32()))
    }
}

/// Scalar lane cast with a stated contract (replaces the retired pair-generic
/// conversion trait from the numeric provider).
///
/// Converts through an exact `f64` intermediate (`NumericElement::to_f64`):
/// float targets convert with `FloatElement::from_f64` (one rounding, identical
/// to `as`); integer targets truncate toward zero and saturate with
/// `IntegerTarget::from_truncated` (identical to `as` for float sources,
/// including NaN-to-0; out-of-range integer sources saturate instead of
/// wrapping, a deliberate hardening — every SIMD lane type is at most 32 bits
/// wide, so the `f64` intermediate itself never rounds a lane value).
pub trait LaneCast: Sealed {
    /// Cast an `f64` lane value into this scalar type under the trait contract.
    #[must_use]
    fn cast_lane(x: f64) -> Self;
}

impl LaneCast for f32 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for f64 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for F16 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for F32 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for Bf16 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for F64 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for F8 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for F4 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for Bf8 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for Bf4 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl LaneCast for I32 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self(i32::from_truncated(x))
    }
}

impl LaneCast for i32 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_truncated(x)
    }
}

impl LaneCast for I16 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self(i16::from_truncated(x))
    }
}

impl LaneCast for i16 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_truncated(x)
    }
}

impl LaneCast for I8 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self(i8::from_truncated(x))
    }
}

impl LaneCast for i8 {
    #[inline(always)]
    fn cast_lane(x: f64) -> Self {
        Self::from_truncated(x)
    }
}

//! Sealed element-type trait for SIMD-eligible scalars.

use crate::private::Sealed;

/// Backward-compatible alias for NumericElement.
pub use eunomia::NumericElement as Scalar;
pub use eunomia::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8, I16, I32, I8};
pub use eunomia::{CastFrom, CastTo, FloatElement, NumericElement};

/// Float scalars with an exact round-to-nearest-ties-to-even primitive.
///
/// Backs [`crate::kernel::SimdKernel::round`], whose contract is banker's
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
    fn round_ties_even(self) -> Self;
}

impl Sealed for f32 {}
impl Sealed for f64 {}
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

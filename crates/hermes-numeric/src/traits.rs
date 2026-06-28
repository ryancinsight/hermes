pub(crate) mod private {
    pub trait Sealed {}
}

/// Core numeric element trait. The main extension point for monomorphized operations across all precisions.
pub trait NumericElement:
    private::Sealed
    + Copy
    + Default
    + Send
    + Sync
    + 'static
    + PartialOrd
    + PartialEq
    + core::fmt::Debug
    + core::ops::Add<Output = Self>
    + core::ops::AddAssign
    + core::ops::Sub<Output = Self>
    + core::ops::SubAssign
    + core::ops::Mul<Output = Self>
    + core::ops::MulAssign
    + core::ops::Div<Output = Self>
    + CastFrom<i32>
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// IEEE 754 not-a-number sentinel.
    const NAN: Self;
    /// IEEE 754 positive infinity.
    const INFINITY: Self;
    /// Number of bytes per element.
    const BYTE_WIDTH: usize;
    /// Bitwise representation with all bits set to 1.
    const ALL_ONES: Self;
    /// IEEE 754 sign-bit mask: only the most-significant bit is set.
    ///
    /// XOR-ing any value with this mask negates it (flips the sign bit).
    /// Used by the default `SimdKernel::neg` implementation (hermes-simd-core) to
    /// avoid subtraction, which is not universally available across SIMD backends.
    const SIGN_MASK: Self;

    /// Absolute value.
    fn abs(self) -> Self;
    /// Scalar fused multiply-add: (self * b) + c.
    fn scalar_fmadd(self, b: Self, c: Self) -> Self;
    /// Square root. Floats follow IEEE 754 (`NaN` for negative inputs); integers
    /// return the exact floor integer square root (`isqrt`), with negative signed
    /// inputs defined to return 0 (integers have no `NaN` to signal the domain
    /// error). No `f64` round-trip, so the integer result is exact for all operands.
    fn sqrt(self) -> Self;
    /// Returns true if finite.
    fn is_finite(self) -> bool;
    /// Returns true if NaN.
    fn is_nan(self) -> bool;
    /// Cast to f64.
    fn to_f64(self) -> f64;
    /// Bitwise AND.
    fn bitand(self, rhs: Self) -> Self;
    /// Bitwise OR.
    fn bitor(self, rhs: Self) -> Self;
    /// Bitwise XOR.
    fn bitxor(self, rhs: Self) -> Self;
    /// Count set bits (population count).
    fn count_ones(self) -> u32;

    /// Elementwise minimum: returns `self` if `self <= other`, else `other`.
    ///
    /// Default: uses `PartialOrd` comparison. Concrete impls (e.g. `f32`, `f64`) may
    /// override with a hardware intrinsic.
    #[inline(always)]
    fn min_scalar(self, other: Self) -> Self
    where
        Self: PartialOrd,
    {
        if self <= other {
            self
        } else {
            other
        }
    }

    /// Elementwise maximum: returns `self` if `self >= other`, else `other`.
    ///
    /// Default: uses `PartialOrd` comparison. Concrete impls (e.g. `f32`, `f64`) may
    /// override with a hardware intrinsic.
    #[inline(always)]
    fn max_scalar(self, other: Self) -> Self
    where
        Self: PartialOrd,
    {
        if self >= other {
            self
        } else {
            other
        }
    }

    /// The minimum representable finite value (negative infinity or `i32::MIN`).
    ///
    /// Used as the identity element for `Max` reductions.
    const MIN_VALUE: Self;

    /// The maximum representable finite value (positive infinity or `i32::MAX`).
    ///
    /// Used as the identity element for `Min` reductions.
    const MAX_VALUE: Self;
}

/// Float-specific capabilities.
pub trait FloatElement: private::Sealed + NumericElement {
    /// Convert from f32.
    fn from_f32(val: f32) -> Self;
    /// Convert from f64.
    fn from_f64(val: f64) -> Self;
    /// Cast to f32.
    fn to_f32(self) -> f32;
}

/// Helper trait for generic casting between SIMD scalar types.
pub trait CastFrom<T>: Copy {
    /// Cast from type `T` to `Self`.
    fn cast_from(val: T) -> Self;
}

/// Helper trait for generic casting to another SIMD scalar type.
pub trait CastTo: Copy {
    /// Cast `self` to type `U`.
    #[inline(always)]
    fn cast_to<U>(self) -> U
    where
        U: CastFrom<Self>,
    {
        U::cast_from(self)
    }
}

impl<T: Copy> CastTo for T {}

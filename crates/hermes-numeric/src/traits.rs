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
    + core::ops::Neg<Output = Self>
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

    /// Absolute value.
    fn abs(self) -> Self;
    /// Scalar fused multiply-add: (self * b) + c.
    fn scalar_fmadd(self, b: Self, c: Self) -> Self;
    /// Square root.
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

/// Trait for 4-bit types that can be packed two per byte.
pub trait Packable4: Copy + 'static {
    /// Pack a low and high element into a single byte.
    fn pack_pair(low: Self, high: Self) -> u8;
    /// Unpack a single byte into a low and high element.
    fn unpack_pair(packed: u8) -> (Self, Self);
}

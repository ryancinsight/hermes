use crate::traits::{private, FloatElement, NumericElement};

impl private::Sealed for f32 {}
impl private::Sealed for f64 {}
impl private::Sealed for half::f16 {}
impl private::Sealed for half::bf16 {}
impl private::Sealed for i8 {}
impl private::Sealed for i16 {}
impl private::Sealed for i32 {}
impl private::Sealed for i64 {}
impl private::Sealed for u8 {}
impl private::Sealed for u16 {}
impl private::Sealed for u32 {}
impl private::Sealed for u64 {}

impl NumericElement for f32 {
    const ZERO: Self = 0.0_f32;
    const ONE: Self = 1.0_f32;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const BYTE_WIDTH: usize = 4;
    const ALL_ONES: Self = f32::from_bits(0xFFFF_FFFF);
    const SIGN_MASK: Self = f32::from_bits(0x8000_0000);
    const MIN_VALUE: Self = f32::NEG_INFINITY;
    const MAX_VALUE: Self = f32::INFINITY;

    #[inline(always)]
    fn abs(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.abs()
        }
        #[cfg(not(feature = "std"))]
        {
            f32::from_bits(self.to_bits() & 0x7FFF_FFFF)
        }
    }
    #[inline(always)]
    fn scalar_fmadd(self, b: Self, c: Self) -> Self {
        #[cfg(feature = "std")]
        {
            self.mul_add(b, c)
        }
        #[cfg(not(feature = "std"))]
        {
            libm::fmaf(self, b, c)
        }
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::sqrtf(self)
        }
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() & rhs.to_bits())
    }
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() | rhs.to_bits())
    }
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() ^ rhs.to_bits())
    }
    #[inline(always)]
    fn count_ones(self) -> u32 {
        self.to_bits().count_ones()
    }
    /// Use native `f32::min` which correctly handles NaN propagation.
    #[inline(always)]
    fn min_scalar(self, other: Self) -> Self {
        self.min(other)
    }
    /// Use native `f32::max` which correctly handles NaN propagation.
    #[inline(always)]
    fn max_scalar(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericElement for f64 {
    const ZERO: Self = 0.0_f64;
    const ONE: Self = 1.0_f64;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const BYTE_WIDTH: usize = 8;
    const ALL_ONES: Self = f64::from_bits(0xFFFF_FFFF_FFFF_FFFF);
    const SIGN_MASK: Self = f64::from_bits(0x8000_0000_0000_0000);
    const MIN_VALUE: Self = f64::NEG_INFINITY;
    const MAX_VALUE: Self = f64::INFINITY;

    #[inline(always)]
    fn abs(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.abs()
        }
        #[cfg(not(feature = "std"))]
        {
            f64::from_bits(self.to_bits() & 0x7FFF_FFFF_FFFF_FFFF)
        }
    }
    #[inline(always)]
    fn scalar_fmadd(self, b: Self, c: Self) -> Self {
        #[cfg(feature = "std")]
        {
            self.mul_add(b, c)
        }
        #[cfg(not(feature = "std"))]
        {
            libm::fma(self, b, c)
        }
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            self.sqrt()
        }
        #[cfg(not(feature = "std"))]
        {
            libm::sqrt(self)
        }
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() & rhs.to_bits())
    }
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() | rhs.to_bits())
    }
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() ^ rhs.to_bits())
    }
    #[inline(always)]
    fn count_ones(self) -> u32 {
        self.to_bits().count_ones()
    }
    /// Use native `f64::min` which correctly handles NaN propagation.
    #[inline(always)]
    fn min_scalar(self, other: Self) -> Self {
        self.min(other)
    }
    /// Use native `f64::max` which correctly handles NaN propagation.
    #[inline(always)]
    fn max_scalar(self, other: Self) -> Self {
        self.max(other)
    }
}

impl NumericElement for half::f16 {
    const ZERO: Self = half::f16::ZERO;
    const ONE: Self = half::f16::ONE;
    const NAN: Self = half::f16::NAN;
    const INFINITY: Self = half::f16::INFINITY;
    const BYTE_WIDTH: usize = 2;
    const ALL_ONES: Self = half::f16::from_bits(0xFFFF);
    const SIGN_MASK: Self = half::f16::from_bits(0x8000);
    const MIN_VALUE: Self = half::f16::NEG_INFINITY;
    const MAX_VALUE: Self = half::f16::INFINITY;

    #[inline(always)]
    fn abs(self) -> Self {
        half::f16::from_f32(self.to_f32().abs())
    }
    #[inline(always)]
    fn scalar_fmadd(self, b: Self, c: Self) -> Self {
        half::f16::from_f32(self.to_f32().scalar_fmadd(b.to_f32(), c.to_f32()))
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        half::f16::from_f32(self.to_f32().sqrt())
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        half::f16::is_finite(self)
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        half::f16::is_nan(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() & rhs.to_bits())
    }
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() | rhs.to_bits())
    }
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() ^ rhs.to_bits())
    }
    #[inline(always)]
    fn count_ones(self) -> u32 {
        self.to_bits().count_ones()
    }
    #[inline(always)]
    fn min_scalar(self, other: Self) -> Self {
        half::f16::from_f32(self.to_f32().min(other.to_f32()))
    }
    #[inline(always)]
    fn max_scalar(self, other: Self) -> Self {
        half::f16::from_f32(self.to_f32().max(other.to_f32()))
    }
}

impl NumericElement for half::bf16 {
    const ZERO: Self = half::bf16::ZERO;
    const ONE: Self = half::bf16::ONE;
    const NAN: Self = half::bf16::NAN;
    const INFINITY: Self = half::bf16::INFINITY;
    const BYTE_WIDTH: usize = 2;
    const ALL_ONES: Self = half::bf16::from_bits(0xFFFF);
    const SIGN_MASK: Self = half::bf16::from_bits(0x8000);
    const MIN_VALUE: Self = half::bf16::NEG_INFINITY;
    const MAX_VALUE: Self = half::bf16::INFINITY;

    #[inline(always)]
    fn abs(self) -> Self {
        half::bf16::from_f32(self.to_f32().abs())
    }
    #[inline(always)]
    fn scalar_fmadd(self, b: Self, c: Self) -> Self {
        half::bf16::from_f32(self.to_f32().scalar_fmadd(b.to_f32(), c.to_f32()))
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        half::bf16::from_f32(self.to_f32().sqrt())
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        half::bf16::is_finite(self)
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        half::bf16::is_nan(self)
    }
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() & rhs.to_bits())
    }
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() | rhs.to_bits())
    }
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self::from_bits(self.to_bits() ^ rhs.to_bits())
    }
    #[inline(always)]
    fn count_ones(self) -> u32 {
        self.to_bits().count_ones()
    }
    #[inline(always)]
    fn min_scalar(self, other: Self) -> Self {
        half::bf16::from_f32(self.to_f32().min(other.to_f32()))
    }
    #[inline(always)]
    fn max_scalar(self, other: Self) -> Self {
        half::bf16::from_f32(self.to_f32().max(other.to_f32()))
    }
}

/// Shared `NumericElement` body for the built-in signed integer types. Differs
/// from [`impl_numeric_element_unsigned`] only in `ALL_ONES` (-1), `SIGN_MASK`
/// (`T::MIN`), and `abs`. `min_scalar`/`max_scalar` use the `PartialOrd`-based
/// trait defaults.
macro_rules! impl_numeric_element_signed {
    ($t:ty, $byte_width:expr) => {
        impl NumericElement for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const NAN: Self = 0;
            const INFINITY: Self = 0;
            const BYTE_WIDTH: usize = $byte_width;
            const ALL_ONES: Self = -1;
            const SIGN_MASK: Self = <$t>::MIN;
            const MIN_VALUE: Self = <$t>::MIN;
            const MAX_VALUE: Self = <$t>::MAX;

            #[inline(always)]
            fn abs(self) -> Self {
                self.abs()
            }
            #[inline(always)]
            fn scalar_fmadd(self, b: Self, c: Self) -> Self {
                self.wrapping_mul(b).wrapping_add(c)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                // Exact integer (floor) square root. The previous
                // `(self as f64).sqrt() as Self` rounded operands above 2^53 to f64
                // *before* taking the root, losing precision (e.g. `i64::MAX`).
                // `isqrt` is exact. Negative inputs have no real root and integers
                // have no NaN to signal it, so they return 0 (documented contract).
                if self < 0 {
                    0
                } else {
                    self.isqrt()
                }
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                true
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                false
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                self & rhs
            }
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                self | rhs
            }
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                self ^ rhs
            }
            #[inline(always)]
            fn count_ones(self) -> u32 {
                self.count_ones()
            }
        }
    };
}

impl_numeric_element_signed!(i8, 1);
impl_numeric_element_signed!(i16, 2);
impl_numeric_element_signed!(i32, 4);
impl_numeric_element_signed!(i64, 8);

impl FloatElement for f32 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        val
    }
    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
}

impl FloatElement for f64 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        val as f64
    }
    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        val
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl FloatElement for half::f16 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        half::f16::from_f32(val)
    }
    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        half::f16::from_f64(val)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl FloatElement for half::bf16 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        half::bf16::from_f32(val)
    }
    #[inline(always)]
    fn from_f64(val: f64) -> Self {
        half::bf16::from_f64(val)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

macro_rules! impl_numeric_element_unsigned {
    ($t:ty, $byte_width:expr, $min_value:expr, $max_value:expr) => {
        impl NumericElement for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const NAN: Self = 0;
            const INFINITY: Self = 0;
            const BYTE_WIDTH: usize = $byte_width;
            const ALL_ONES: Self = !0;
            const SIGN_MASK: Self = 0;
            const MIN_VALUE: Self = $min_value;
            const MAX_VALUE: Self = $max_value;

            #[inline(always)]
            fn abs(self) -> Self {
                self
            }
            #[inline(always)]
            fn scalar_fmadd(self, b: Self, c: Self) -> Self {
                self.wrapping_mul(b).wrapping_add(c)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                // Exact integer (floor) square root; no f64 round-trip (the old
                // `(self as f64).sqrt() as Self` lost precision above 2^53, e.g.
                // `u64::MAX`).
                self.isqrt()
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                true
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                false
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                self & rhs
            }
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                self | rhs
            }
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                self ^ rhs
            }
            #[inline(always)]
            fn count_ones(self) -> u32 {
                self.count_ones()
            }
        }
    };
}

impl_numeric_element_unsigned!(u8, 1, u8::MIN, u8::MAX);
impl_numeric_element_unsigned!(u16, 2, u16::MIN, u16::MAX);
impl_numeric_element_unsigned!(u32, 4, u32::MIN, u32::MAX);
impl_numeric_element_unsigned!(u64, 8, u64::MIN, u64::MAX);

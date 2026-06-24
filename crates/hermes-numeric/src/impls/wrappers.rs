use crate::traits::{private, FloatElement, NumericElement};
use crate::types::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8, I16, I32, I8};

macro_rules! impl_numeric_element {
    (
        $t:ident,
        $zero:expr,
        $one:expr,
        $nan:expr,
        $inf:expr,
        $min:expr,
        $max:expr,
        $width:expr,
        $ones:expr,
        $sign_mask:expr,
        $to_f64:expr,
        $fmadd:expr,
        $abs:expr,
        $sqrt:expr,
        $finite:expr,
        $nan_check:expr,
        $and:expr,
        $or:expr,
        $xor:expr,
        $count_ones:expr
    ) => {
        impl private::Sealed for $t {}

        impl NumericElement for $t {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const NAN: Self = $nan;
            const INFINITY: Self = $inf;
            const MIN_VALUE: Self = $min;
            const MAX_VALUE: Self = $max;
            const BYTE_WIDTH: usize = $width;
            const ALL_ONES: Self = $ones;
            const SIGN_MASK: Self = $sign_mask;

            #[inline(always)]
            fn abs(self) -> Self {
                $abs(self)
            }
            #[inline(always)]
            fn scalar_fmadd(self, b: Self, c: Self) -> Self {
                $fmadd(self, b, c)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                $sqrt(self)
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                $finite(self)
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                $nan_check(self)
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                $to_f64(self)
            }
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                $and(self, rhs)
            }
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                $or(self, rhs)
            }
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                $xor(self, rhs)
            }
            #[inline(always)]
            fn count_ones(self) -> u32 {
                $count_ones(self)
            }
        }

        const _: () = {
            assert!(
                core::mem::size_of::<$t>() == $width,
                "Byte width assertion failed"
            );
        };
    };
}

impl_numeric_element!(
    F16,
    F16(half::f16::ZERO),
    F16(half::f16::ONE),
    F16(half::f16::NAN),
    F16(half::f16::INFINITY),
    F16(half::f16::NEG_INFINITY),
    F16(half::f16::INFINITY),
    2,
    F16(half::f16::from_bits(0xFFFF)),
    F16(half::f16::from_bits(0x8000)), // sign bit
    |x: F16| x.0.to_f32() as f64,
    |x: F16, b: F16, c: F16| F16(half::f16::from_f32(
        x.0.to_f32().scalar_fmadd(b.0.to_f32(), c.0.to_f32())
    )),
    |x: F16| F16(half::f16::from_f32(x.0.to_f32().abs())),
    |x: F16| F16(half::f16::from_f32(x.0.to_f32().sqrt())),
    |x: F16| x.0.is_finite(),
    |x: F16| x.0.is_nan(),
    |x: F16, y: F16| F16(half::f16::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: F16, y: F16| F16(half::f16::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: F16, y: F16| F16(half::f16::from_bits(x.0.to_bits() ^ y.0.to_bits())),
    |x: F16| x.0.to_bits().count_ones()
);

impl_numeric_element!(
    F32,
    F32(0.0),
    F32(1.0),
    F32(f32::NAN),
    F32(f32::INFINITY),
    F32(f32::NEG_INFINITY),
    F32(f32::INFINITY),
    4,
    F32(f32::from_bits(0xFFFF_FFFF)),
    F32(f32::from_bits(0x8000_0000)), // sign bit
    |x: F32| x.0 as f64,
    |x: F32, b: F32, c: F32| F32(x.0.scalar_fmadd(b.0, c.0)),
    |x: F32| F32(x.0.abs()),
    |x: F32| F32(x.0.sqrt()),
    |x: F32| x.0.is_finite(),
    |x: F32| x.0.is_nan(),
    |x: F32, y: F32| F32(f32::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: F32, y: F32| F32(f32::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: F32, y: F32| F32(f32::from_bits(x.0.to_bits() ^ y.0.to_bits())),
    |x: F32| x.0.to_bits().count_ones()
);

impl_numeric_element!(
    F64,
    F64(0.0),
    F64(1.0),
    F64(f64::NAN),
    F64(f64::INFINITY),
    F64(f64::NEG_INFINITY),
    F64(f64::INFINITY),
    8,
    F64(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)),
    F64(f64::from_bits(0x8000_0000_0000_0000)), // sign bit
    |x: F64| x.0,
    |x: F64, b: F64, c: F64| F64(x.0.scalar_fmadd(b.0, c.0)),
    |x: F64| F64(x.0.abs()),
    |x: F64| F64(x.0.sqrt()),
    |x: F64| x.0.is_finite(),
    |x: F64| x.0.is_nan(),
    |x: F64, y: F64| F64(f64::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: F64, y: F64| F64(f64::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: F64, y: F64| F64(f64::from_bits(x.0.to_bits() ^ y.0.to_bits())),
    |x: F64| x.0.to_bits().count_ones()
);

impl_numeric_element!(
    Bf16,
    Bf16(half::bf16::ZERO),
    Bf16(half::bf16::ONE),
    Bf16(half::bf16::NAN),
    Bf16(half::bf16::INFINITY),
    Bf16(half::bf16::NEG_INFINITY),
    Bf16(half::bf16::INFINITY),
    2,
    Bf16(half::bf16::from_bits(0xFFFF)),
    Bf16(half::bf16::from_bits(0x8000)), // sign bit
    |x: Bf16| x.0.to_f32() as f64,
    |x: Bf16, b: Bf16, c: Bf16| Bf16(half::bf16::from_f32(
        x.0.to_f32().scalar_fmadd(b.0.to_f32(), c.0.to_f32())
    )),
    |x: Bf16| Bf16(half::bf16::from_f32(x.0.to_f32().abs())),
    |x: Bf16| Bf16(half::bf16::from_f32(x.0.to_f32().sqrt())),
    |x: Bf16| x.0.is_finite(),
    |x: Bf16| x.0.is_nan(),
    |x: Bf16, y: Bf16| Bf16(half::bf16::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: Bf16, y: Bf16| Bf16(half::bf16::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: Bf16, y: Bf16| Bf16(half::bf16::from_bits(x.0.to_bits() ^ y.0.to_bits())),
    |x: Bf16| x.0.to_bits().count_ones()
);

macro_rules! impl_numeric_for_byte_float {
    ($t:ident, $zero:expr, $one:expr, $nan:expr, $inf:expr, $min:expr, $max:expr, $sign_mask:expr) => {
        impl_numeric_element!(
            $t,
            $zero,
            $one,
            $nan,
            $inf,
            $min,
            $max,
            1,
            $t(0xFF),
            $sign_mask,
            |x: $t| x.to_f32() as f64,
            |x: $t, b: $t, c: $t| $t::from_f32(x.to_f32().scalar_fmadd(b.to_f32(), c.to_f32())),
            |x: $t| $t::from_f32(x.to_f32().abs()),
            |x: $t| $t::from_f32(x.to_f32().sqrt()),
            |x: $t| x.to_f32().is_finite(),
            |x: $t| x.to_f32().is_nan(),
            |x: $t, y: $t| $t(x.0 & y.0),
            |x: $t, y: $t| $t(x.0 | y.0),
            |x: $t, y: $t| $t(x.0 ^ y.0),
            |x: $t| x.0.count_ones()
        );
    };
}

// Bf8: 1.4.3 format — sign bit is bit 7 (0x80)
impl_numeric_for_byte_float!(
    Bf8,
    Bf8(0),
    Bf8(0x3C),
    Bf8(0x7F),
    Bf8(0x7C),
    Bf8(0xFC),
    Bf8(0x7C),
    Bf8(0x80)
);
// Bf4: 4-bit packed in u8 — sign bit is bit 3 (0x08)
impl_numeric_for_byte_float!(
    Bf4,
    Bf4(0),
    Bf4(0x02),
    Bf4(0x07),
    Bf4(0x06),
    Bf4(0x86),
    Bf4(0x06),
    Bf4(0x08)
);
// F8: 1.4.3 format — sign bit is bit 7 (0x80)
impl_numeric_for_byte_float!(
    F8,
    F8(0),
    F8(0x38),
    F8(0x7F),
    F8(0x77),
    F8(0xF7),
    F8(0x77),
    F8(0x80)
);
// F4: 4-bit packed in u8 — sign bit is bit 3 (0x08)
impl_numeric_for_byte_float!(
    F4,
    F4(0),
    F4(0x03),
    F4(0x07),
    F4(0x06),
    F4(0x86),
    F4(0x06),
    F4(0x08)
);

impl_numeric_element!(
    I8,
    I8(0),
    I8(1),
    I8(0),
    I8(0),
    I8(i8::MIN),
    I8(i8::MAX),
    1,
    I8(-1),
    I8(i8::MIN), // sign bit = 0x80 as two's complement = i8::MIN
    |x: I8| x.0 as f64,
    |x: I8, b: I8, c: I8| I8(x.0.wrapping_mul(b.0).wrapping_add(c.0)),
    |x: I8| I8(x.0.abs()),
    |x: I8| I8((x.0 as f32).sqrt() as i8),
    |_| true,
    |_| false,
    |x: I8, y: I8| I8(x.0 & y.0),
    |x: I8, y: I8| I8(x.0 | y.0),
    |x: I8, y: I8| I8(x.0 ^ y.0),
    |x: I8| x.0.count_ones()
);

impl_numeric_element!(
    I16,
    I16(0),
    I16(1),
    I16(0),
    I16(0),
    I16(i16::MIN),
    I16(i16::MAX),
    2,
    I16(-1),
    I16(i16::MIN), // sign bit = bit 15
    |x: I16| x.0 as f64,
    |x: I16, b: I16, c: I16| I16(x.0.wrapping_mul(b.0).wrapping_add(c.0)),
    |x: I16| I16(x.0.abs()),
    |x: I16| I16((x.0 as f32).sqrt() as i16),
    |_| true,
    |_| false,
    |x: I16, y: I16| I16(x.0 & y.0),
    |x: I16, y: I16| I16(x.0 | y.0),
    |x: I16, y: I16| I16(x.0 ^ y.0),
    |x: I16| x.0.count_ones()
);

impl_numeric_element!(
    I32,
    I32(0),
    I32(1),
    I32(0),
    I32(0),
    I32(i32::MIN),
    I32(i32::MAX),
    4,
    I32(-1),
    I32(i32::MIN), // sign bit = bit 31
    |x: I32| x.0 as f64,
    |x: I32, b: I32, c: I32| I32(x.0.wrapping_mul(b.0).wrapping_add(c.0)),
    |x: I32| I32(x.0.abs()),
    |x: I32| I32((x.0 as f64).sqrt() as i32),
    |_| true,
    |_| false,
    |x: I32, y: I32| I32(x.0 & y.0),
    |x: I32, y: I32| I32(x.0 | y.0),
    |x: I32, y: I32| I32(x.0 ^ y.0),
    |x: I32| x.0.count_ones()
);

macro_rules! impl_float_element {
    ($t:ident, $from_f32:expr, $from_f64:expr, $to_f32:expr) => {
        impl FloatElement for $t {
            #[inline(always)]
            fn from_f32(val: f32) -> Self {
                $from_f32(val)
            }
            #[inline(always)]
            fn from_f64(val: f64) -> Self {
                $from_f64(val)
            }
            #[inline(always)]
            fn to_f32(self) -> f32 {
                $to_f32(self)
            }
        }
    };
}

impl_float_element!(
    F16,
    |val| F16(half::f16::from_f32(val)),
    |val| F16(half::f16::from_f64(val)),
    |x: F16| x.0.to_f32()
);
impl_float_element!(F32, F32, |val| F32(val as f32), |x: F32| x.0);
impl_float_element!(F64, |val| F64(val as f64), F64, |x: F64| x.0 as f32);
impl_float_element!(
    Bf16,
    |val| Bf16(half::bf16::from_f32(val)),
    |val| Bf16(half::bf16::from_f64(val)),
    |x: Bf16| x.0.to_f32()
);
impl_float_element!(
    Bf8,
    Bf8::from_f32,
    |val| Bf8::from_f32(val as f32),
    |x: Bf8| x.to_f32()
);
impl_float_element!(
    Bf4,
    Bf4::from_f32,
    |val| Bf4::from_f32(val as f32),
    |x: Bf4| x.to_f32()
);
impl_float_element!(F8, F8::from_f32, |val| F8::from_f32(val as f32), |x: F8| x
    .to_f32());
impl_float_element!(F4, F4::from_f32, |val| F4::from_f32(val as f32), |x: F4| x
    .to_f32());

use crate::traits::{private, NumericElement, FloatElement};
use crate::types::{F16, F32, F64, Bf16, Bf8, Bf4, F8, F4, I8, I16, I32};

macro_rules! impl_numeric_element {
    ($t:ident, $zero:expr, $one:expr, $nan:expr, $inf:expr, $width:expr, $ones:expr, $to_f64:expr, $fmadd:expr, $abs:expr, $sqrt:expr, $finite:expr, $nan_check:expr, $and:expr, $or:expr, $xor:expr) => {
        impl private::Sealed for $t {}

        impl NumericElement for $t {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const NAN: Self = $nan;
            const INFINITY: Self = $inf;
            const BYTE_WIDTH: usize = $width;
            const ALL_ONES: Self = $ones;

            #[inline(always)]
            fn abs(self) -> Self { $abs(self) }
            #[inline(always)]
            fn scalar_fmadd(self, b: Self, c: Self) -> Self { $fmadd(self, b, c) }
            #[inline(always)]
            fn sqrt(self) -> Self { $sqrt(self) }
            #[inline(always)]
            fn is_finite(self) -> bool { $finite(self) }
            #[inline(always)]
            fn is_nan(self) -> bool { $nan_check(self) }
            #[inline(always)]
            fn to_f64(self) -> f64 { $to_f64(self) }
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self { $and(self, rhs) }
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self { $or(self, rhs) }
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self { $xor(self, rhs) }
        }

        const _: () = {
            assert!(core::mem::size_of::<$t>() == $width, "Byte width assertion failed");
        };
    };
}

impl_numeric_element!(
    F16,
    F16(half::f16::ZERO),
    F16(half::f16::ONE),
    F16(half::f16::NAN),
    F16(half::f16::INFINITY),
    2,
    F16(half::f16::from_bits(0xFFFF)),
    |x: F16| x.0.to_f32() as f64,
    |x: F16, b: F16, c: F16| F16(half::f16::from_f32(x.0.to_f32().mul_add(b.0.to_f32(), c.0.to_f32()))),
    |x: F16| F16(half::f16::from_f32(x.0.to_f32().abs())),
    |x: F16| F16(half::f16::from_f32(x.0.to_f32().sqrt())),
    |x: F16| x.0.is_finite(),
    |x: F16| x.0.is_nan(),
    |x: F16, y: F16| F16(half::f16::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: F16, y: F16| F16(half::f16::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: F16, y: F16| F16(half::f16::from_bits(x.0.to_bits() ^ y.0.to_bits()))
);

impl_numeric_element!(
    F32,
    F32(0.0),
    F32(1.0),
    F32(f32::NAN),
    F32(f32::INFINITY),
    4,
    F32(f32::from_bits(0xFFFF_FFFF)),
    |x: F32| x.0 as f64,
    |x: F32, b: F32, c: F32| F32(x.0.mul_add(b.0, c.0)),
    |x: F32| F32(x.0.abs()),
    |x: F32| F32(x.0.sqrt()),
    |x: F32| x.0.is_finite(),
    |x: F32| x.0.is_nan(),
    |x: F32, y: F32| F32(f32::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: F32, y: F32| F32(f32::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: F32, y: F32| F32(f32::from_bits(x.0.to_bits() ^ y.0.to_bits()))
);

impl_numeric_element!(
    F64,
    F64(0.0),
    F64(1.0),
    F64(f64::NAN),
    F64(f64::INFINITY),
    8,
    F64(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)),
    |x: F64| x.0,
    |x: F64, b: F64, c: F64| F64(x.0.mul_add(b.0, c.0)),
    |x: F64| F64(x.0.abs()),
    |x: F64| F64(x.0.sqrt()),
    |x: F64| x.0.is_finite(),
    |x: F64| x.0.is_nan(),
    |x: F64, y: F64| F64(f64::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: F64, y: F64| F64(f64::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: F64, y: F64| F64(f64::from_bits(x.0.to_bits() ^ y.0.to_bits()))
);

impl_numeric_element!(
    Bf16,
    Bf16(half::bf16::ZERO),
    Bf16(half::bf16::ONE),
    Bf16(half::bf16::NAN),
    Bf16(half::bf16::INFINITY),
    2,
    Bf16(half::bf16::from_bits(0xFFFF)),
    |x: Bf16| x.0.to_f32() as f64,
    |x: Bf16, b: Bf16, c: Bf16| Bf16(half::bf16::from_f32(x.0.to_f32().mul_add(b.0.to_f32(), c.0.to_f32()))),
    |x: Bf16| Bf16(half::bf16::from_f32(x.0.to_f32().abs())),
    |x: Bf16| Bf16(half::bf16::from_f32(x.0.to_f32().sqrt())),
    |x: Bf16| x.0.is_finite(),
    |x: Bf16| x.0.is_nan(),
    |x: Bf16, y: Bf16| Bf16(half::bf16::from_bits(x.0.to_bits() & y.0.to_bits())),
    |x: Bf16, y: Bf16| Bf16(half::bf16::from_bits(x.0.to_bits() | y.0.to_bits())),
    |x: Bf16, y: Bf16| Bf16(half::bf16::from_bits(x.0.to_bits() ^ y.0.to_bits()))
);

macro_rules! impl_numeric_for_byte_float {
    ($t:ident) => {
        impl_numeric_element!(
            $t,
            $t(0),
            $t(0),
            $t(0),
            $t(0),
            1,
            $t(0xFF),
            |x: $t| x.to_f32() as f64,
            |x: $t, b: $t, c: $t| $t::from_f32(x.to_f32().mul_add(b.to_f32(), c.to_f32())),
            |x: $t| $t::from_f32(x.to_f32().abs()),
            |x: $t| $t::from_f32(x.to_f32().sqrt()),
            |x: $t| x.to_f32().is_finite(),
            |x: $t| x.to_f32().is_nan(),
            |x: $t, y: $t| $t(x.0 & y.0),
            |x: $t, y: $t| $t(x.0 | y.0),
            |x: $t, y: $t| $t(x.0 ^ y.0)
        );
    };
}

impl_numeric_for_byte_float!(Bf8);
impl_numeric_for_byte_float!(Bf4);
impl_numeric_for_byte_float!(F8);
impl_numeric_for_byte_float!(F4);

impl_numeric_element!(
    I8,
    I8(0),
    I8(1),
    I8(0),
    I8(0),
    1,
    I8(-1),
    |x: I8| x.0 as f64,
    |x: I8, b: I8, c: I8| I8(x.0.wrapping_mul(b.0).wrapping_add(c.0)),
    |x: I8| I8(x.0.abs()),
    |x: I8| I8((x.0 as f32).sqrt() as i8),
    |_| true,
    |_| false,
    |x: I8, y: I8| I8(x.0 & y.0),
    |x: I8, y: I8| I8(x.0 | y.0),
    |x: I8, y: I8| I8(x.0 ^ y.0)
);

impl_numeric_element!(
    I16,
    I16(0),
    I16(1),
    I16(0),
    I16(0),
    2,
    I16(-1),
    |x: I16| x.0 as f64,
    |x: I16, b: I16, c: I16| I16(x.0.wrapping_mul(b.0).wrapping_add(c.0)),
    |x: I16| I16(x.0.abs()),
    |x: I16| I16((x.0 as f32).sqrt() as i16),
    |_| true,
    |_| false,
    |x: I16, y: I16| I16(x.0 & y.0),
    |x: I16, y: I16| I16(x.0 | y.0),
    |x: I16, y: I16| I16(x.0 ^ y.0)
);

impl_numeric_element!(
    I32,
    I32(0),
    I32(1),
    I32(0),
    I32(0),
    4,
    I32(-1),
    |x: I32| x.0 as f64,
    |x: I32, b: I32, c: I32| I32(x.0.wrapping_mul(b.0).wrapping_add(c.0)),
    |x: I32| I32(x.0.abs()),
    |x: I32| I32((x.0 as f64).sqrt() as i32),
    |_| true,
    |_| false,
    |x: I32, y: I32| I32(x.0 & y.0),
    |x: I32, y: I32| I32(x.0 | y.0),
    |x: I32, y: I32| I32(x.0 ^ y.0)
);

macro_rules! impl_float_element {
    ($t:ident, $from_f32:expr, $from_f64:expr, $to_f32:expr) => {
        impl FloatElement for $t {
            #[inline(always)]
            fn from_f32(val: f32) -> Self { $from_f32(val) }
            #[inline(always)]
            fn from_f64(val: f64) -> Self { $from_f64(val) }
            #[inline(always)]
            fn to_f32(self) -> f32 { $to_f32(self) }
        }
    };
}

impl_float_element!(F16, |val| F16(half::f16::from_f32(val)), |val| F16(half::f16::from_f64(val)), |x: F16| x.0.to_f32());
impl_float_element!(F32, F32, |val| F32(val as f32), |x: F32| x.0);
impl_float_element!(F64, |val| F64(val as f64), F64, |x: F64| x.0 as f32);
impl_float_element!(Bf16, |val| Bf16(half::bf16::from_f32(val)), |val| Bf16(half::bf16::from_f64(val)), |x: Bf16| x.0.to_f32());
impl_float_element!(Bf8, Bf8::from_f32, |val| Bf8::from_f32(val as f32), |x: Bf8| x.to_f32());
impl_float_element!(Bf4, Bf4::from_f32, |val| Bf4::from_f32(val as f32), |x: Bf4| x.to_f32());
impl_float_element!(F8, F8::from_f32, |val| F8::from_f32(val as f32), |x: F8| x.to_f32());
impl_float_element!(F4, F4::from_f32, |val| F4::from_f32(val as f32), |x: F4| x.to_f32());

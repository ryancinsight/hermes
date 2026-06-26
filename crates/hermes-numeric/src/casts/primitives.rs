use crate::traits::{CastFrom, FloatElement, NumericElement};
use crate::types::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8, I16, I32, I8};

// Implement CastFrom between all primitives
macro_rules! impl_cast_primitive_between {
    ($src:ty, $dst:ty) => {
        impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                val as $dst
            }
        }
    };
}

impl_cast_primitive_between!(i8, i8);
impl_cast_primitive_between!(i8, i16);
impl_cast_primitive_between!(i8, i32);
impl_cast_primitive_between!(i16, i8);
impl_cast_primitive_between!(i16, i16);
impl_cast_primitive_between!(i16, i32);
impl_cast_primitive_between!(i32, i8);
impl_cast_primitive_between!(i32, i16);
impl_cast_primitive_between!(i32, i32);
impl_cast_primitive_between!(i32, i64);
impl_cast_primitive_between!(i32, u8);
impl_cast_primitive_between!(i32, u16);
impl_cast_primitive_between!(i32, u32);
impl_cast_primitive_between!(i32, u64);

impl_cast_primitive_between!(i64, i64);
impl_cast_primitive_between!(i64, i8);
impl_cast_primitive_between!(i64, i16);
impl_cast_primitive_between!(i64, i32);

impl_cast_primitive_between!(u8, u8);
impl_cast_primitive_between!(u8, u16);
impl_cast_primitive_between!(u8, u32);
impl_cast_primitive_between!(u8, u64);
impl_cast_primitive_between!(u8, i8);
impl_cast_primitive_between!(u8, i16);
impl_cast_primitive_between!(u8, i32);
impl_cast_primitive_between!(u8, i64);

impl_cast_primitive_between!(u16, u8);
impl_cast_primitive_between!(u16, u16);
impl_cast_primitive_between!(u16, u32);
impl_cast_primitive_between!(u16, u64);
impl_cast_primitive_between!(u16, i8);
impl_cast_primitive_between!(u16, i16);
impl_cast_primitive_between!(u16, i32);
impl_cast_primitive_between!(u16, i64);

impl_cast_primitive_between!(u32, u8);
impl_cast_primitive_between!(u32, u16);
impl_cast_primitive_between!(u32, u32);
impl_cast_primitive_between!(u32, u64);
impl_cast_primitive_between!(u32, i8);
impl_cast_primitive_between!(u32, i16);
impl_cast_primitive_between!(u32, i32);
impl_cast_primitive_between!(u32, i64);

impl_cast_primitive_between!(u64, u8);
impl_cast_primitive_between!(u64, u16);
impl_cast_primitive_between!(u64, u32);
impl_cast_primitive_between!(u64, u64);
impl_cast_primitive_between!(u64, i8);
impl_cast_primitive_between!(u64, i16);
impl_cast_primitive_between!(u64, i32);
impl_cast_primitive_between!(u64, i64);

impl_cast_primitive_between!(f32, f32);
impl_cast_primitive_between!(f32, f64);
impl_cast_primitive_between!(f64, f32);
impl_cast_primitive_between!(f64, f64);

macro_rules! impl_cast_primitive_float_int {
    ($src:ty, $dst:ty) => {
        impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                val as $dst
            }
        }
    };
}
impl_cast_primitive_float_int!(f32, i8);
impl_cast_primitive_float_int!(f32, i16);
impl_cast_primitive_float_int!(f32, i32);
impl_cast_primitive_float_int!(f64, i8);
impl_cast_primitive_float_int!(f64, i16);
impl_cast_primitive_float_int!(f64, i32);
impl_cast_primitive_float_int!(i8, f32);
impl_cast_primitive_float_int!(i8, f64);
impl_cast_primitive_float_int!(i16, f32);
impl_cast_primitive_float_int!(i16, f64);
impl_cast_primitive_float_int!(i32, f32);
impl_cast_primitive_float_int!(i32, f64);
impl_cast_primitive_float_int!(i64, f32);
impl_cast_primitive_float_int!(i64, f64);
impl_cast_primitive_float_int!(u8, f32);
impl_cast_primitive_float_int!(u8, f64);
impl_cast_primitive_float_int!(u16, f32);
impl_cast_primitive_float_int!(u16, f64);
impl_cast_primitive_float_int!(u32, f32);
impl_cast_primitive_float_int!(u32, f64);
impl_cast_primitive_float_int!(u64, f32);
impl_cast_primitive_float_int!(u64, f64);

impl_cast_primitive_float_int!(f32, i64);
impl_cast_primitive_float_int!(f32, u8);
impl_cast_primitive_float_int!(f32, u16);
impl_cast_primitive_float_int!(f32, u32);
impl_cast_primitive_float_int!(f32, u64);
impl_cast_primitive_float_int!(f64, i64);
impl_cast_primitive_float_int!(f64, u8);
impl_cast_primitive_float_int!(f64, u16);
impl_cast_primitive_float_int!(f64, u32);
impl_cast_primitive_float_int!(f64, u64);

// Implement CastFrom for half types
impl CastFrom<half::f16> for half::f16 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val
    }
}
impl CastFrom<half::f16> for half::bf16 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        half::bf16::from_f32(val.to_f32())
    }
}
impl CastFrom<half::f16> for f32 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32()
    }
}
impl CastFrom<half::f16> for f64 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f64()
    }
}
impl CastFrom<half::f16> for i8 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as i8
    }
}
impl CastFrom<half::f16> for i16 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as i16
    }
}
impl CastFrom<half::f16> for i32 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as i32
    }
}
impl CastFrom<half::f16> for i64 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as i64
    }
}
impl CastFrom<half::f16> for u8 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as u8
    }
}
impl CastFrom<half::f16> for u16 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as u16
    }
}
impl CastFrom<half::f16> for u32 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as u32
    }
}
impl CastFrom<half::f16> for u64 {
    #[inline(always)]
    fn cast_from(val: half::f16) -> Self {
        val.to_f32() as u64
    }
}

impl CastFrom<half::bf16> for half::f16 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        half::f16::from_f32(val.to_f32())
    }
}
impl CastFrom<half::bf16> for half::bf16 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val
    }
}
impl CastFrom<half::bf16> for f32 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32()
    }
}
impl CastFrom<half::bf16> for f64 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as f64
    }
}
impl CastFrom<half::bf16> for i8 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as i8
    }
}
impl CastFrom<half::bf16> for i16 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as i16
    }
}
impl CastFrom<half::bf16> for i32 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as i32
    }
}
impl CastFrom<half::bf16> for i64 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as i64
    }
}
impl CastFrom<half::bf16> for u8 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as u8
    }
}
impl CastFrom<half::bf16> for u16 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as u16
    }
}
impl CastFrom<half::bf16> for u32 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as u32
    }
}
impl CastFrom<half::bf16> for u64 {
    #[inline(always)]
    fn cast_from(val: half::bf16) -> Self {
        val.to_f32() as u64
    }
}

macro_rules! impl_cast_from_primitive_to_half {
    ($src:ty) => {
        impl CastFrom<$src> for half::f16 {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                half::f16::from_f32(val as f32)
            }
        }
        impl CastFrom<$src> for half::bf16 {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                half::bf16::from_f32(val as f32)
            }
        }
    };
}
impl_cast_from_primitive_to_half!(f32);
impl_cast_from_primitive_to_half!(f64);
impl_cast_from_primitive_to_half!(i8);
impl_cast_from_primitive_to_half!(i16);
impl_cast_from_primitive_to_half!(i32);
impl_cast_from_primitive_to_half!(i64);
impl_cast_from_primitive_to_half!(u8);
impl_cast_from_primitive_to_half!(u16);
impl_cast_from_primitive_to_half!(u32);
impl_cast_from_primitive_to_half!(u64);

// Casts between wrappers and primitives
macro_rules! impl_cast_float_wrapper_primitive {
    ($wrap:ident, $prim:ty) => {
        impl CastFrom<$wrap> for $prim {
            #[inline(always)]
            fn cast_from(val: $wrap) -> Self {
                <Self as CastFrom<f64>>::cast_from(val.to_f64())
            }
        }
        impl CastFrom<$prim> for $wrap {
            #[inline(always)]
            fn cast_from(val: $prim) -> Self {
                let f = <f64 as CastFrom<$prim>>::cast_from(val);
                <$wrap as FloatElement>::from_f64(f)
            }
        }
    };
}

macro_rules! impl_cast_int_wrapper_primitive {
    ($wrap:ident, $inner:ty, $prim:ty) => {
        impl CastFrom<$wrap> for $prim {
            #[inline(always)]
            fn cast_from(val: $wrap) -> Self {
                <Self as CastFrom<$inner>>::cast_from(val.0)
            }
        }
        impl CastFrom<$prim> for $wrap {
            #[inline(always)]
            fn cast_from(val: $prim) -> Self {
                Self(<$inner as CastFrom<$prim>>::cast_from(val))
            }
        }
    };
}

macro_rules! impl_cast_float_wrapper_primitives {
    ($wrap:ident) => {
        impl_cast_float_wrapper_primitive!($wrap, f32);
        impl_cast_float_wrapper_primitive!($wrap, f64);
        impl_cast_float_wrapper_primitive!($wrap, half::f16);
        impl_cast_float_wrapper_primitive!($wrap, half::bf16);
        impl_cast_float_wrapper_primitive!($wrap, i8);
        impl_cast_float_wrapper_primitive!($wrap, i16);
        impl_cast_float_wrapper_primitive!($wrap, i32);
    };
}

impl_cast_float_wrapper_primitives!(F16);
impl_cast_float_wrapper_primitives!(F32);
impl_cast_float_wrapper_primitives!(F64);
impl_cast_float_wrapper_primitives!(Bf16);
impl_cast_float_wrapper_primitives!(Bf8);
impl_cast_float_wrapper_primitives!(Bf4);
impl_cast_float_wrapper_primitives!(F8);
impl_cast_float_wrapper_primitives!(F4);

macro_rules! impl_cast_int_wrapper_primitives {
    ($wrap:ident, $inner:ty) => {
        impl_cast_int_wrapper_primitive!($wrap, $inner, f32);
        impl_cast_int_wrapper_primitive!($wrap, $inner, f64);
        impl_cast_int_wrapper_primitive!($wrap, $inner, half::f16);
        impl_cast_int_wrapper_primitive!($wrap, $inner, half::bf16);
        impl_cast_int_wrapper_primitive!($wrap, $inner, i8);
        impl_cast_int_wrapper_primitive!($wrap, $inner, i16);
        impl_cast_int_wrapper_primitive!($wrap, $inner, i32);
    };
}

impl_cast_int_wrapper_primitives!(I8, i8);
impl_cast_int_wrapper_primitives!(I16, i16);
impl_cast_int_wrapper_primitives!(I32, i32);

// Float-to-Int wrapper casts
macro_rules! impl_float_to_int {
    ($src:ident, $dst:ident) => {
        impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                Self(val.to_f64() as _)
            }
        }
    };
}
impl_float_to_int!(F16, I8);
impl_float_to_int!(F16, I16);
impl_float_to_int!(F16, I32);
impl_float_to_int!(F32, I8);
impl_float_to_int!(F32, I16);
impl_float_to_int!(F32, I32);
impl_float_to_int!(F64, I8);
impl_float_to_int!(F64, I16);
impl_float_to_int!(F64, I32);
impl_float_to_int!(Bf16, I8);
impl_float_to_int!(Bf16, I16);
impl_float_to_int!(Bf16, I32);
impl_float_to_int!(Bf8, I8);
impl_float_to_int!(Bf8, I16);
impl_float_to_int!(Bf8, I32);
impl_float_to_int!(Bf4, I8);
impl_float_to_int!(Bf4, I16);
impl_float_to_int!(Bf4, I32);
impl_float_to_int!(F8, I8);
impl_float_to_int!(F8, I16);
impl_float_to_int!(F8, I32);
impl_float_to_int!(F4, I8);
impl_float_to_int!(F4, I16);
impl_float_to_int!(F4, I32);

// Int-to-Float wrapper casts
macro_rules! impl_int_to_float {
    ($src:ident, $dst:ident) => {
        impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                Self::from_f64(val.0 as f64)
            }
        }
    };
}
impl_int_to_float!(I8, F16);
impl_int_to_float!(I8, F32);
impl_int_to_float!(I8, F64);
impl_int_to_float!(I8, Bf16);
impl_int_to_float!(I8, Bf8);
impl_int_to_float!(I8, Bf4);
impl_int_to_float!(I8, F8);
impl_int_to_float!(I8, F4);
impl_int_to_float!(I16, F16);
impl_int_to_float!(I16, F32);
impl_int_to_float!(I16, F64);
impl_int_to_float!(I16, Bf16);
impl_int_to_float!(I16, Bf8);
impl_int_to_float!(I16, Bf4);
impl_int_to_float!(I16, F8);
impl_int_to_float!(I16, F4);
impl_int_to_float!(I32, F16);
impl_int_to_float!(I32, F32);
impl_int_to_float!(I32, F64);
impl_int_to_float!(I32, Bf16);
impl_int_to_float!(I32, Bf8);
impl_int_to_float!(I32, Bf4);
impl_int_to_float!(I32, F8);
impl_int_to_float!(I32, F4);

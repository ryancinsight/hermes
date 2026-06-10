use crate::traits::{CastFrom, FloatElement, NumericElement};
use crate::types::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8, I16, I32, I8};

macro_rules! impl_cast_between {
    ($src:ident, $dst:ident) => {
        impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                <$dst as FloatElement>::from_f64(val.to_f64())
            }
        }
    };
}

// Float-to-Float casts
impl_cast_between!(F16, F32);
impl_cast_between!(F16, F64);
impl_cast_between!(F16, Bf16);
impl_cast_between!(F16, Bf8);
impl_cast_between!(F16, Bf4);
impl_cast_between!(F16, F8);
impl_cast_between!(F16, F4);
impl_cast_between!(F32, F16);
impl_cast_between!(F32, F64);
impl_cast_between!(F32, Bf16);
impl_cast_between!(F32, Bf8);
impl_cast_between!(F32, Bf4);
impl_cast_between!(F32, F8);
impl_cast_between!(F32, F4);
impl_cast_between!(F64, F16);
impl_cast_between!(F64, F32);
impl_cast_between!(F64, Bf16);
impl_cast_between!(F64, Bf8);
impl_cast_between!(F64, Bf4);
impl_cast_between!(F64, F8);
impl_cast_between!(F64, F4);
impl_cast_between!(Bf16, F16);
impl_cast_between!(Bf16, F32);
impl_cast_between!(Bf16, F64);
impl_cast_between!(Bf16, Bf8);
impl_cast_between!(Bf16, Bf4);
impl_cast_between!(Bf16, F8);
impl_cast_between!(Bf16, F4);
impl_cast_between!(Bf8, F16);
impl_cast_between!(Bf8, F32);
impl_cast_between!(Bf8, F64);
impl_cast_between!(Bf8, Bf16);
impl_cast_between!(Bf8, Bf4);
impl_cast_between!(Bf8, F8);
impl_cast_between!(Bf8, F4);
impl_cast_between!(Bf4, F16);
impl_cast_between!(Bf4, F32);
impl_cast_between!(Bf4, F64);
impl_cast_between!(Bf4, Bf16);
impl_cast_between!(Bf4, Bf8);
impl_cast_between!(Bf4, F8);
impl_cast_between!(Bf4, F4);
impl_cast_between!(F8, F16);
impl_cast_between!(F8, F32);
impl_cast_between!(F8, F64);
impl_cast_between!(F8, Bf16);
impl_cast_between!(F8, Bf8);
impl_cast_between!(F8, Bf4);
impl_cast_between!(F8, F4);
impl_cast_between!(F4, F16);
impl_cast_between!(F4, F32);
impl_cast_between!(F4, F64);
impl_cast_between!(F4, Bf16);
impl_cast_between!(F4, Bf8);
impl_cast_between!(F4, Bf4);
impl_cast_between!(F4, F8);

// Identity casts
impl CastFrom<F16> for F16 {
    #[inline(always)]
    fn cast_from(val: F16) -> Self {
        val
    }
}
impl CastFrom<F32> for F32 {
    #[inline(always)]
    fn cast_from(val: F32) -> Self {
        val
    }
}
impl CastFrom<F64> for F64 {
    #[inline(always)]
    fn cast_from(val: F64) -> Self {
        val
    }
}
impl CastFrom<Bf16> for Bf16 {
    #[inline(always)]
    fn cast_from(val: Bf16) -> Self {
        val
    }
}
impl CastFrom<Bf8> for Bf8 {
    #[inline(always)]
    fn cast_from(val: Bf8) -> Self {
        val
    }
}
impl CastFrom<Bf4> for Bf4 {
    #[inline(always)]
    fn cast_from(val: Bf4) -> Self {
        val
    }
}
impl CastFrom<F8> for F8 {
    #[inline(always)]
    fn cast_from(val: F8) -> Self {
        val
    }
}
impl CastFrom<F4> for F4 {
    #[inline(always)]
    fn cast_from(val: F4) -> Self {
        val
    }
}
impl CastFrom<I8> for I8 {
    #[inline(always)]
    fn cast_from(val: I8) -> Self {
        val
    }
}
impl CastFrom<I16> for I16 {
    #[inline(always)]
    fn cast_from(val: I16) -> Self {
        val
    }
}
impl CastFrom<I32> for I32 {
    #[inline(always)]
    fn cast_from(val: I32) -> Self {
        val
    }
}

// Int-to-Int casts
macro_rules! impl_int_to_int {
    ($src:ident, $dst:ident) => {
        impl CastFrom<$src> for $dst {
            #[inline(always)]
            fn cast_from(val: $src) -> Self {
                Self(val.0 as _)
            }
        }
    };
}
impl_int_to_int!(I8, I16);
impl_int_to_int!(I8, I32);
impl_int_to_int!(I16, I8);
impl_int_to_int!(I16, I32);
impl_int_to_int!(I32, I8);
impl_int_to_int!(I32, I16);

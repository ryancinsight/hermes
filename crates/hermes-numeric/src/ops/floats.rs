use crate::types::{Bf16, Bf4, Bf8, F16, F32, F4, F64, F8};

macro_rules! impl_arithmetic {
    ($t:ident, $inner:ty, $conv_to:expr, $conv_from:expr) => {
        impl core::ops::Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                let a = $conv_to(self);
                let b = $conv_to(rhs);
                $conv_from(a + b)
            }
        }
        impl core::ops::AddAssign for $t {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl core::ops::Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                let a = $conv_to(self);
                let b = $conv_to(rhs);
                $conv_from(a - b)
            }
        }
        impl core::ops::SubAssign for $t {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl core::ops::Mul for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                let a = $conv_to(self);
                let b = $conv_to(rhs);
                $conv_from(a * b)
            }
        }
        impl core::ops::MulAssign for $t {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl core::ops::Div for $t {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                let a = $conv_to(self);
                let b = $conv_to(rhs);
                $conv_from(a / b)
            }
        }
        impl core::ops::Neg for $t {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                let a = $conv_to(self);
                $conv_from(-a)
            }
        }
    };
}

impl_arithmetic!(F16, half::f16, |x: F16| x.0.to_f32(), |val| F16(
    half::f16::from_f32(val)
));
impl_arithmetic!(F32, f32, |x: F32| x.0, F32);
impl_arithmetic!(F64, f64, |x: F64| x.0, F64);
impl_arithmetic!(Bf16, half::bf16, |x: Bf16| x.0.to_f32(), |val| Bf16(
    half::bf16::from_f32(val)
));
impl_arithmetic!(Bf8, u8, |x: Bf8| x.to_f32(), Bf8::from_f32);
impl_arithmetic!(Bf4, u8, |x: Bf4| x.to_f32(), Bf4::from_f32);
impl_arithmetic!(F8, u8, |x: F8| x.to_f32(), F8::from_f32);
impl_arithmetic!(F4, u8, |x: F4| x.to_f32(), F4::from_f32);

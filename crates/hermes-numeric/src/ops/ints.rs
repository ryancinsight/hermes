use crate::types::{I16, I32, I8};

macro_rules! impl_integer_arithmetic {
    ($t:ident, $inner:ty) => {
        impl core::ops::Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(self.0.wrapping_add(rhs.0))
            }
        }
        impl core::ops::AddAssign for $t {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0.wrapping_add(rhs.0);
            }
        }
        impl core::ops::Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0.wrapping_sub(rhs.0))
            }
        }
        impl core::ops::SubAssign for $t {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.0 = self.0.wrapping_sub(rhs.0);
            }
        }
        impl core::ops::Mul for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(self.0.wrapping_mul(rhs.0))
            }
        }
        impl core::ops::MulAssign for $t {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                self.0 = self.0.wrapping_mul(rhs.0);
            }
        }
        impl core::ops::Div for $t {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                if rhs.0 == 0 {
                    Self(0)
                } else {
                    Self(self.0.wrapping_div(rhs.0))
                }
            }
        }
        impl core::ops::Neg for $t {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(self.0.wrapping_neg())
            }
        }
    };
}

impl_integer_arithmetic!(I8, i8);
impl_integer_arithmetic!(I16, i16);
impl_integer_arithmetic!(I32, i32);

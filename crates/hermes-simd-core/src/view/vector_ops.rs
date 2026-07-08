//! Operator overloads for SIMD vectors.

//! Operator overload implementations for the `Vector` register wrapper.

use super::vector_reg::{assert_runtime_supported, Vector};
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

// -----------------------------------------------------------------------------
// Operator Overloads
// -----------------------------------------------------------------------------

impl<T, Arch> core::ops::Add for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::add(self.raw, rhs.raw) })
    }
}

impl<T, Arch> core::ops::AddAssign for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        assert_runtime_supported::<T, Arch>();
        self.raw = unsafe { Arch::add(self.raw, rhs.raw) };
    }
}

impl<T, Arch> core::ops::Sub for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::sub(self.raw, rhs.raw) })
    }
}

impl<T, Arch> core::ops::SubAssign for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        assert_runtime_supported::<T, Arch>();
        self.raw = unsafe { Arch::sub(self.raw, rhs.raw) };
    }
}

impl<T, Arch> core::ops::Mul for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::mul(self.raw, rhs.raw) })
    }
}

impl<T, Arch> core::ops::MulAssign for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        assert_runtime_supported::<T, Arch>();
        self.raw = unsafe { Arch::mul(self.raw, rhs.raw) };
    }
}

impl<T, Arch> core::ops::Div for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::div(self.raw, rhs.raw) })
    }
}

impl<T, Arch> core::ops::DivAssign for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        assert_runtime_supported::<T, Arch>();
        self.raw = unsafe { Arch::div(self.raw, rhs.raw) };
    }
}

impl<T, Arch> core::ops::BitAnd for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::bitand(self.raw, rhs.raw) })
    }
}

impl<T, Arch> core::ops::BitAndAssign for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        assert_runtime_supported::<T, Arch>();
        self.raw = unsafe { Arch::bitand(self.raw, rhs.raw) };
    }
}

impl<T, Arch> core::ops::BitOr for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::bitor(self.raw, rhs.raw) })
    }
}

impl<T, Arch> core::ops::BitOrAssign for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        assert_runtime_supported::<T, Arch>();
        self.raw = unsafe { Arch::bitor(self.raw, rhs.raw) };
    }
}

impl<T, Arch> core::ops::BitXor for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::bitxor(self.raw, rhs.raw) })
    }
}

impl<T, Arch> core::ops::BitXorAssign for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        assert_runtime_supported::<T, Arch>();
        self.raw = unsafe { Arch::bitxor(self.raw, rhs.raw) };
    }
}

impl<T, Arch> core::ops::Neg for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::neg(self.raw) })
    }
}

impl<T, Arch> core::ops::Neg for &Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Vector<T, Arch>;
    #[inline(always)]
    fn neg(self) -> Self::Output {
        assert_runtime_supported::<T, Arch>();
        Vector::new(unsafe { Arch::neg(self.raw) })
    }
}

impl<T, Arch> core::ops::Not for Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        assert_runtime_supported::<T, Arch>();
        Self::new(unsafe { Arch::bitnot(self.raw) })
    }
}

impl<T, Arch> core::ops::Not for &Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    type Output = Vector<T, Arch>;
    #[inline(always)]
    fn not(self) -> Self::Output {
        assert_runtime_supported::<T, Arch>();
        Vector::new(unsafe { Arch::bitnot(self.raw) })
    }
}

macro_rules! impl_ref_binary_op {
    ($op_trait:ident, $op_method:ident, $kernel_method:ident) => {
        impl<'a, T, Arch> core::ops::$op_trait<Vector<T, Arch>> for &'a Vector<T, Arch>
        where
            Arch: SimdArch + SimdKernel<T>,
            T: Scalar,
        {
            type Output = Vector<T, Arch>;
            #[inline(always)]
            fn $op_method(self, rhs: Vector<T, Arch>) -> Self::Output {
                assert_runtime_supported::<T, Arch>();
                Vector::new(unsafe { Arch::$kernel_method(self.raw, rhs.raw) })
            }
        }
        impl<'a, T, Arch> core::ops::$op_trait<&'a Vector<T, Arch>> for Vector<T, Arch>
        where
            Arch: SimdArch + SimdKernel<T>,
            T: Scalar,
        {
            type Output = Vector<T, Arch>;
            #[inline(always)]
            fn $op_method(self, rhs: &'a Vector<T, Arch>) -> Self::Output {
                assert_runtime_supported::<T, Arch>();
                Vector::new(unsafe { Arch::$kernel_method(self.raw, rhs.raw) })
            }
        }
        impl<'a, 'b, T, Arch> core::ops::$op_trait<&'b Vector<T, Arch>> for &'a Vector<T, Arch>
        where
            Arch: SimdArch + SimdKernel<T>,
            T: Scalar,
        {
            type Output = Vector<T, Arch>;
            #[inline(always)]
            fn $op_method(self, rhs: &'b Vector<T, Arch>) -> Self::Output {
                assert_runtime_supported::<T, Arch>();
                Vector::new(unsafe { Arch::$kernel_method(self.raw, rhs.raw) })
            }
        }
    };
}

impl_ref_binary_op!(Add, add, add);
impl_ref_binary_op!(Sub, sub, sub);
impl_ref_binary_op!(Mul, mul, mul);
impl_ref_binary_op!(Div, div, div);
impl_ref_binary_op!(BitAnd, bitand, bitand);
impl_ref_binary_op!(BitOr, bitor, bitor);
impl_ref_binary_op!(BitXor, bitxor, bitxor);

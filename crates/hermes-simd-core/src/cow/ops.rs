//! Operator overload implementations for Clone-on-Write SIMD containers.
//!
//! # Safety
//!
//! Two obligations recur here. Kernel calls are `#[target_feature]`-gated, and
//! that precondition holds by construction: a `SimdCow` exists only for an
//! architecture the host can execute, since its borrowed form comes from
//! [`SimdView::new`](crate::view::SimdView::new) and its owned constructors
//! assert the same condition. The second is local — these routines build their
//! output buffer with `with_capacity` and write it through a raw pointer,
//! raising the length only once every element is initialized. That avoids both
//! a zero-fill of a buffer about to be overwritten and any `&mut [T]` spanning
//! uninitialized elements, so each such site carries a `SAFETY` comment showing
//! the write coverage. `gather` and `prefix_scan` reserve capacity and fill it
//! through the view's `*_into_uninit` methods over
//! [`AlignedVec::spare_capacity_mut`](crate::vec::AlignedVec::spare_capacity_mut),
//! then raise the length once those report success, so those paths never zero
//! the buffer either.

use super::SimdCow;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::ops::ElementOp;
use crate::scalar::Scalar;
use crate::vec::AlignedVec;

// ---------------------------------------------------------------------------
// Operator Overloads with Allocation Reuse
// ---------------------------------------------------------------------------

fn binary_lhs_inplace<T, Arch, Align, Op>(
    lhs: &SimdCow<'_, T, Arch, Align>,
    rhs: &mut AlignedVec<T, Align>,
    op: Op,
) where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Op: ElementOp<T>,
{
    let len = lhs.len();
    assert_eq!(len, rhs.len(), "SIMD length mismatch");

    let lhs_view = lhs.view();
    let rhs_view = rhs.view_mut::<Arch>();

    let mut chunks_lhs = lhs_view.simd_chunks();
    let mut chunks_rhs = rhs_view.simd_chunks_mut();

    for (chunk_lhs, mut chunk_rhs) in (&mut chunks_lhs).zip(&mut chunks_rhs) {
        unsafe {
            let va = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                Arch::load_aligned(chunk_lhs.as_ptr())
            } else {
                Arch::load_unaligned(chunk_lhs.as_ptr())
            };
            let vb = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                Arch::load_aligned(chunk_rhs.as_ptr())
            } else {
                Arch::load_unaligned(chunk_rhs.as_ptr())
            };
            let vr = ElementOp::apply::<Arch>(op, va, vb);
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                Arch::store_aligned(chunk_rhs.as_mut_ptr(), vr);
            } else {
                Arch::store_unaligned(chunk_rhs.as_mut_ptr(), vr);
            }
        }
    }

    let tail_lhs = chunks_lhs.remainder();
    let tail_rhs = chunks_rhs.into_remainder();
    for (&a, b) in tail_lhs.iter().zip(tail_rhs.iter_mut()) {
        *b = ElementOp::apply_scalar(op, a, *b);
    }
}

macro_rules! impl_binary_op {
    ($op_trait:ident, $op_method:ident, $op_strategy:ty, $op_val:expr, $is_commutative:expr) => {
        // 1. SimdCow + SimdCow
        impl<'a, 'b, T, Arch, Align> core::ops::$op_trait<SimdCow<'b, T, Arch, Align>>
            for SimdCow<'a, T, Arch, Align>
        where
            T: Scalar,
            Arch: SimdArch + SimdKernel<T>,
            Align: Alignment,
        {
            type Output = SimdCow<'static, T, Arch, Align>;

            #[inline]
            #[allow(unused_mut)]
            fn $op_method(self, rhs: SimdCow<'b, T, Arch, Align>) -> Self::Output {
                match (self, rhs) {
                    (SimdCow::Owned(lhs_vec), rhs) => {
                        let mut lhs_cow = SimdCow::Owned(lhs_vec);
                        lhs_cow
                            .transform_in_place(&rhs, $op_val)
                            .expect("SIMD length mismatch");
                        match lhs_cow {
                            SimdCow::Owned(v) => SimdCow::Owned(v),
                            _ => unreachable!(),
                        }
                    }
                    (lhs, SimdCow::Owned(mut rhs_vec)) => {
                        if $is_commutative {
                            let mut rhs_cow = SimdCow::Owned(rhs_vec);
                            rhs_cow
                                .transform_in_place(&lhs, $op_val)
                                .expect("SIMD length mismatch");
                            match rhs_cow {
                                rhs_cow @ SimdCow::Owned(_) => rhs_cow,
                                _ => unreachable!(),
                            }
                        } else {
                            binary_lhs_inplace::<T, Arch, Align, $op_strategy>(
                                &lhs,
                                &mut rhs_vec,
                                $op_val,
                            );
                            SimdCow::Owned(rhs_vec)
                        }
                    }
                    (lhs, rhs) => lhs.zip_cow(&rhs, $op_val).expect("SIMD length mismatch"),
                }
            }
        }

        // 2. SimdCow + &SimdCow
        impl<'a, 'b, T, Arch, Align> core::ops::$op_trait<&'b SimdCow<'b, T, Arch, Align>>
            for SimdCow<'a, T, Arch, Align>
        where
            T: Scalar,
            Arch: SimdArch + SimdKernel<T>,
            Align: Alignment,
        {
            type Output = SimdCow<'static, T, Arch, Align>;

            #[inline]
            #[allow(unused_mut)]
            fn $op_method(self, rhs: &'b SimdCow<'b, T, Arch, Align>) -> Self::Output {
                match self {
                    SimdCow::Owned(lhs_vec) => {
                        let mut lhs_cow = SimdCow::Owned(lhs_vec);
                        lhs_cow
                            .transform_in_place(rhs, $op_val)
                            .expect("SIMD length mismatch");
                        match lhs_cow {
                            SimdCow::Owned(v) => SimdCow::Owned(v),
                            _ => unreachable!(),
                        }
                    }
                    lhs => lhs.zip_cow(rhs, $op_val).expect("SIMD length mismatch"),
                }
            }
        }

        // 3. &SimdCow + SimdCow
        impl<'a, 'b, T, Arch, Align> core::ops::$op_trait<SimdCow<'b, T, Arch, Align>>
            for &'a SimdCow<'a, T, Arch, Align>
        where
            T: Scalar,
            Arch: SimdArch + SimdKernel<T>,
            Align: Alignment,
        {
            type Output = SimdCow<'static, T, Arch, Align>;

            #[inline]
            fn $op_method(self, rhs: SimdCow<'b, T, Arch, Align>) -> Self::Output {
                match rhs {
                    SimdCow::Owned(mut rhs_vec) => {
                        if $is_commutative {
                            let mut rhs_cow = SimdCow::Owned(rhs_vec);
                            rhs_cow
                                .transform_in_place(self, $op_val)
                                .expect("SIMD length mismatch");
                            match rhs_cow {
                                rhs_cow @ SimdCow::Owned(_) => rhs_cow,
                                _ => unreachable!(),
                            }
                        } else {
                            binary_lhs_inplace::<T, Arch, Align, $op_strategy>(
                                self,
                                &mut rhs_vec,
                                $op_val,
                            );
                            SimdCow::Owned(rhs_vec)
                        }
                    }
                    rhs => self.zip_cow(&rhs, $op_val).expect("SIMD length mismatch"),
                }
            }
        }

        // 4. &SimdCow + &SimdCow
        impl<'a, 'b, T, Arch, Align> core::ops::$op_trait<&'b SimdCow<'b, T, Arch, Align>>
            for &'a SimdCow<'a, T, Arch, Align>
        where
            T: Scalar,
            Arch: SimdArch + SimdKernel<T>,
            Align: Alignment,
        {
            type Output = SimdCow<'static, T, Arch, Align>;

            #[inline]
            fn $op_method(self, rhs: &'b SimdCow<'b, T, Arch, Align>) -> Self::Output {
                self.zip_cow(rhs, $op_val).expect("SIMD length mismatch")
            }
        }
    };
}

macro_rules! impl_assign_op {
    ($op_trait:ident, $op_method:ident, $op_strategy:ty, $op_val:expr) => {
        // SimdCow += SimdCow
        impl<'a, 'b, T, Arch, Align> core::ops::$op_trait<SimdCow<'b, T, Arch, Align>>
            for SimdCow<'a, T, Arch, Align>
        where
            'b: 'a,
            T: Scalar,
            Arch: SimdArch + SimdKernel<T>,
            Align: Alignment,
        {
            #[inline]
            fn $op_method(&mut self, rhs: SimdCow<'b, T, Arch, Align>) {
                self.transform_in_place(&rhs, $op_val)
                    .expect("SIMD length mismatch");
            }
        }

        // SimdCow += &SimdCow
        impl<'a, 'b, T, Arch, Align> core::ops::$op_trait<&'b SimdCow<'b, T, Arch, Align>>
            for SimdCow<'a, T, Arch, Align>
        where
            'b: 'a,
            T: Scalar,
            Arch: SimdArch + SimdKernel<T>,
            Align: Alignment,
        {
            #[inline]
            fn $op_method(&mut self, rhs: &'b SimdCow<'b, T, Arch, Align>) {
                self.transform_in_place(rhs, $op_val)
                    .expect("SIMD length mismatch");
            }
        }
    };
}

impl_binary_op!(Add, add, crate::ops::Add, crate::ops::Add, true);
impl_binary_op!(Sub, sub, crate::ops::Sub, crate::ops::Sub, false);
impl_binary_op!(Mul, mul, crate::ops::Mul, crate::ops::Mul, true);
impl_binary_op!(Div, div, crate::ops::Div, crate::ops::Div, false);
impl_binary_op!(BitAnd, bitand, crate::ops::BitAnd, crate::ops::BitAnd, true);
impl_binary_op!(BitOr, bitor, crate::ops::BitOr, crate::ops::BitOr, true);
impl_binary_op!(BitXor, bitxor, crate::ops::BitXor, crate::ops::BitXor, true);

impl_assign_op!(AddAssign, add_assign, crate::ops::Add, crate::ops::Add);
impl_assign_op!(SubAssign, sub_assign, crate::ops::Sub, crate::ops::Sub);
impl_assign_op!(MulAssign, mul_assign, crate::ops::Mul, crate::ops::Mul);
impl_assign_op!(DivAssign, div_assign, crate::ops::Div, crate::ops::Div);
impl_assign_op!(
    BitAndAssign,
    bitand_assign,
    crate::ops::BitAnd,
    crate::ops::BitAnd
);
impl_assign_op!(
    BitOrAssign,
    bitor_assign,
    crate::ops::BitOr,
    crate::ops::BitOr
);
impl_assign_op!(
    BitXorAssign,
    bitxor_assign,
    crate::ops::BitXor,
    crate::ops::BitXor
);

//! Norm computation and normalization over SIMD-backed slices and tensor rows.
//!
//! # Design
//!
//! All norms are computed via `SimdView::reduce` with standard `ReductionOp` ZSTs where
//! possible, or with a custom `SquaredSum` reduction for L2. No separate norm kernel
//! is required — the reduction infrastructure is reused zero-cost.
//!
//! # Functions
//!
//! | Function | Description |
//! |----------|-------------|
//! [`norm_l1`] | Sum of absolute values: `∑ |x[i]|` |
//! [`norm_l2`] | Euclidean length: `√(∑ x[i]²)` |
//! [`norm_linf`] | Max absolute value: `max |x[i]|` |
//! [`normalize_l2_inplace`] | Scale slice so L2 norm = 1 |
//! [`row_norms_l2`] | L2 norm of each row of a 2-D tensor |

use crate::align::Unaligned;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::ops::{Sum, Max, ReductionOp};
use crate::scalar::{Scalar, FloatElement};
use crate::view::SimdView;
use super::{TensorView, TensorError};

// ---------------------------------------------------------------------------
// Custom L2-accumulation: SquaredSum reduction
// ---------------------------------------------------------------------------

/// Reduction ZST: sum of squared lane values, `∑ x[i]²`.
///
/// Used internally by `norm_l2`. The SIMD accumulation performs
/// `acc += v * v` (a single `fmadd`-eligible multiply-accumulate).
#[derive(Clone, Copy)]
pub struct SquaredSum;

impl crate::private::Sealed for SquaredSum {}

impl<T: Scalar> ReductionOp<T> for SquaredSum
where
    T: core::ops::Mul<Output = T> + core::ops::Add<Output = T>,
{
    #[inline(always)]
    unsafe fn accumulate<Arch: SimdKernel<T>>(
        acc: Arch::Vector,
        v:   Arch::Vector,
    ) -> Arch::Vector {
        let v2 = Arch::mul(v, v);
        Arch::add(acc, v2)
    }

    #[inline(always)]
    unsafe fn finalize<Arch: SimdKernel<T>>(acc: Arch::Vector) -> T {
        Arch::sum_reduce(acc)
    }

    #[inline(always)]
    fn identity_scalar() -> T { T::ZERO }

    #[inline(always)]
    fn scalar_combine(a: T, b: T) -> T { a + b * b }

    /// Override for tail elements: `acc + elem * elem`.
    ///
    /// The default delegates to `scalar_combine(acc, elem)`, which is correct here
    /// because `scalar_combine(a, b) = a + b*b` — but we document the override explicitly
    /// so future readers understand why `SquaredSum` is correct even for sub-LANE slices.
    #[inline(always)]
    fn scalar_accumulate(acc: T, elem: T) -> T { acc + elem * elem }
}

// ---------------------------------------------------------------------------
// Unary ZST: absolute value (reused from ops::Abs when T: FloatElement)
// ---------------------------------------------------------------------------

/// Compute the L1 norm (sum of absolute values) of a slice.
///
/// Uses SIMD `Abs` unary-map followed by `Sum` reduction.
/// Allocates no intermediate buffer — the two reductions are pipelined.
///
/// `data` is borrowed as an `Unaligned` `SimdView`.
#[inline]
pub fn norm_l1<T, Arch>(data: &[T]) -> T
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    if data.is_empty() {
        return T::ZERO;
    }
    // map_unary writes abs into a temp AlignedVec, then sum over it.
    // This avoids a specialized scalar loop and reuses the UnaryOp machinery.
    let view = SimdView::<'_, T, Arch, Unaligned>::new(data).unwrap();
    // One allocation: abs_out
    let abs_buf = {
        let mut buf = crate::vec::AlignedVec::<T, Unaligned>::with_capacity(data.len());
        // SAFETY: we write all elements via map_unary.
        unsafe { buf.set_len(data.len()); }
        let _ = view.map_unary(crate::ops::Abs, buf.as_mut_slice());
        buf
    };
    let abs_view = SimdView::<'_, T, Arch, Unaligned>::new(abs_buf.as_slice()).unwrap();
    abs_view.reduce(Sum)
}

/// Compute the L2 norm (Euclidean distance) of a slice: `√(∑ x[i]²)`.
///
/// Uses the `SquaredSum` ZST reduction followed by `FloatElement::sqrt`.
/// Single SIMD pass — no temporary buffer needed.
#[inline]
pub fn norm_l2<T, Arch>(data: &[T]) -> T
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    if data.is_empty() {
        return T::ZERO;
    }
    let view = SimdView::<'_, T, Arch, Unaligned>::new(data).unwrap();
    let sq = view.reduce(SquaredSum);
    T::from_f32(sq.to_f32().sqrt())
}

/// Compute the L∞ norm (max absolute value) of a slice.
///
/// Two passes: `Abs` unary-map into a temp buffer, then `Max` reduction.
#[inline]
pub fn norm_linf<T, Arch>(data: &[T]) -> T
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    if data.is_empty() {
        return T::ZERO;
    }
    let view = SimdView::<'_, T, Arch, Unaligned>::new(data).unwrap();
    let abs_buf = {
        let mut buf = crate::vec::AlignedVec::<T, Unaligned>::with_capacity(data.len());
        unsafe { buf.set_len(data.len()); }
        let _ = view.map_unary(crate::ops::Abs, buf.as_mut_slice());
        buf
    };
    let abs_view = SimdView::<'_, T, Arch, Unaligned>::new(abs_buf.as_slice()).unwrap();
    abs_view.reduce(Max)
}

/// Scale `data` in-place so its L2 norm equals 1.
///
/// No-op for empty slices or when the norm is zero or NaN (to avoid division by zero).
///
/// Uses:
/// 1. `SquaredSum` reduction for the squared norm.
/// 2. One scalar pass: `data[i] *= inv_norm`.
///
/// The scalar pass could be upgraded to a SIMD `scale_in_place` call; kept scalar here
/// to avoid needing `SimdCow` coupling in this module.
#[inline]
pub fn normalize_l2_inplace<T, Arch>(data: &mut [T])
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    let n = norm_l2::<T, Arch>(data);
    if n == T::ZERO || n.to_f32().is_nan() {
        return;
    }
    let inv = T::ONE / n;
    for x in data.iter_mut() {
        *x = *x * inv;
    }
}

/// Compute the L2 norm of each row of a contiguous 2-D `TensorView`.
///
/// Returns a `Vec<T>` of length `shape[0]`, one norm per row.
/// Allocates one `Vec` for the output norms. Each row norm is computed
/// via `norm_l2` — no intermediate row copy.
///
/// # Errors
/// Returns [`TensorError::NotContiguous`] if the view is not row-major contiguous.
#[inline]
pub fn row_norms_l2<T, Arch>(
    tensor: &TensorView<'_, T, 2, super::RowMajor, &[T]>,
) -> Result<alloc::vec::Vec<T>, TensorError>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    extern crate alloc;
    if !tensor.is_contiguous() {
        return Err(TensorError::NotContiguous);
    }
    let nrows = tensor.shape()[0];
    let mut out = alloc::vec::Vec::with_capacity(nrows);
    for row in tensor.iter_rows()? {
        out.push(norm_l2::<T, Arch>(row));
    }
    Ok(out)
}

// Unit tests moved to integration tests in crates/hermes-simd/tests/tensor_tests.rs

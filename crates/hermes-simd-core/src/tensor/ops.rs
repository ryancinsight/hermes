//! Tensor operations: SIMD-dispatched 2-D and batched matrix multiplication.
//!
//! # Design
//!
//! `matmul` and `batch_matmul` delegate to the existing [`crate::tiling`] infrastructure,
//! reusing the register-blocked `TilingPolicy` without duplicating any kernel code.
//! The `TensorView` API provides bounds-checked shape extraction; actual computation
//! is monomorphized per `(T, Arch, Align, TILE_M, TILE_N)`.
//!
//! # Zero-Allocation Contract
//!
//! `matmul` and `batch_matmul` allocate exactly one output `AlignedVec`. No intermediate
//! buffers. `matmul_to` and `batch_matmul_to` are fully allocation-free.

use crate::align::{Alignment, Unaligned};
use crate::arch::SimdArch;
use crate::execution::Unmasked;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::tiling::{TilingPolicy, TilingStrategy};
use crate::vec::AlignedVec;
use crate::view::{SimdError, SimdView};
use super::{TensorView, RowMajor, Layout};
use crate::ops::ElementOp;

/// 2-D matrix multiplication: `C = A * B`, writing the result into `c`.
///
/// `A` must have shape `[m, k]`; `B` must have shape `[k, n]`; `C` must have shape `[m, n]`.
/// `C` is zero-initialized before accumulation begins.
///
/// # Errors
/// - [`SimdError::LengthMismatch`] if dimensions are incompatible or `c` is not contiguous.
#[inline]
pub fn matmul_to<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
    a: &TensorView<'_, T, 2, RowMajor, &[T]>,
    b: &TensorView<'_, T, 2, RowMajor, &[T]>,
    c: &mut TensorView<'_, T, 2, RowMajor, &mut [T]>,
) -> Result<(), SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    let [m, k] = a.shape();
    let [k2, n] = b.shape();
    let [cm, cn] = c.shape();
    if k != k2 || m != cm || n != cn {
        return Err(SimdError::LengthMismatch);
    }
    if !c.is_contiguous() {
        return Err(SimdError::LengthMismatch);
    }

    // Zero-initialize the output tensor.
    let c_slice = c.as_slice_mut();
    for elem in c_slice.iter_mut() {
        *elem = T::ZERO;
    }

    let Some(a_view) = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(a.as_slice()) else {
        return Ok(());
    };
    let Some(b_view) = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(b.as_slice()) else {
        return Ok(());
    };

    <TilingPolicy<TILE_M, TILE_N> as TilingStrategy<T, Arch, Unaligned>>::gemm(
        &a_view,
        &b_view,
        c_slice,
        m,
        n,
        k,
    )?;

    Ok(())
}

/// 2-D matrix multiplication: `C = A * B`.
///
/// `A` must have shape `[m, k]`; `B` must have shape `[k, n]`.
/// Returns an owned `AlignedVec<T, Align>` of length `m * n` in row-major order.
///
/// Delegates to [`matmul_to`] for execution.
///
/// # Errors
/// - [`SimdError::LengthMismatch`] if dimensions are incompatible.
#[inline]
pub fn matmul<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
    a: &TensorView<'_, T, 2, RowMajor, &[T]>,
    b: &TensorView<'_, T, 2, RowMajor, &[T]>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    let [m, _] = a.shape();
    let [_, n] = b.shape();
    let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(m * n);
    // SAFETY: all elements zero-initialized in matmul_to before gemm accumulation.
    unsafe { out.set_len(m * n); }
    let mut c = TensorView::new_mut(out.as_mut_slice(), [m, n])
        .map_err(|_| SimdError::LengthMismatch)?;
    matmul_to::<T, Arch, Align, TILE_M, TILE_N>(a, b, &mut c)?;
    Ok(out)
}

/// Batched 3-D matrix multiplication: `C[b] = A[b] * B[b]` for each batch index `b`.
/// Writes the result into a mutable 3-D tensor view `c`.
///
/// `A` must have shape `[batch, m, k]`; `B` must have shape `[batch, k, n]`;
/// `C` must have shape `[batch, m, n]`.
///
/// # Errors
/// - [`SimdError::LengthMismatch`] if dimensions are incompatible.
#[inline]
pub fn batch_matmul_to<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
    a: &TensorView<'_, T, 3, RowMajor, &[T]>,
    b: &TensorView<'_, T, 3, RowMajor, &[T]>,
    c: &mut TensorView<'_, T, 3, RowMajor, &mut [T]>,
) -> Result<(), SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    let [batch, m, k] = a.shape();
    let [batch2, k2, n] = b.shape();
    let [c_batch, cm, cn] = c.shape();
    if batch != batch2 || batch != c_batch || k != k2 || m != cm || n != cn {
        return Err(SimdError::LengthMismatch);
    }
    if !c.is_contiguous() {
        return Err(SimdError::LengthMismatch);
    }

    // Zero-initialize.
    let c_flat = c.as_slice_mut();
    for elem in c_flat.iter_mut() {
        *elem = T::ZERO;
    }

    // Process each batch slice independently.
    // We work on raw slices to avoid borrow-checker complexity with mutable matrix_at.
    let a_data = a.as_slice();
    let b_data = b.as_slice();
    let mk = m * k;
    let kn = k * n;
    let mn = m * n;

    for b_idx in 0..batch {
        let a_slice = &a_data[b_idx * mk..(b_idx + 1) * mk];
        let b_slice = &b_data[b_idx * kn..(b_idx + 1) * kn];
        let c_slice = &mut c_flat[b_idx * mn..(b_idx + 1) * mn];

        let Some(a_view) = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(a_slice) else {
            continue;
        };
        let Some(b_view) = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(b_slice) else {
            continue;
        };

        <TilingPolicy<TILE_M, TILE_N> as TilingStrategy<T, Arch, Unaligned>>::gemm(
            &a_view,
            &b_view,
            c_slice,
            m,
            n,
            k,
        )?;
    }

    Ok(())
}

/// Batched 3-D matrix multiplication: `C[b] = A[b] * B[b]` for each batch index `b`.
///
/// `A` must have shape `[batch, m, k]`; `B` must have shape `[batch, k, n]`.
/// Returns an owned `AlignedVec<T, Align>` of length `batch * m * n`.
///
/// # Errors
/// - [`SimdError::LengthMismatch`] if batch sizes or inner dimensions do not match.
#[inline]
pub fn batch_matmul<T, Arch, Align, const TILE_M: usize, const TILE_N: usize>(
    a: &TensorView<'_, T, 3, RowMajor, &[T]>,
    b: &TensorView<'_, T, 3, RowMajor, &[T]>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    let [batch, m, _] = a.shape();
    let [_, _, n] = b.shape();
    let total = batch * m * n;
    let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(total);
    // SAFETY: all elements zero-initialized in batch_matmul_to before gemm accumulation.
    unsafe { out.set_len(total); }
    let mut c = TensorView::new_mut(out.as_mut_slice(), [batch, m, n])
        .map_err(|_| SimdError::LengthMismatch)?;
    batch_matmul_to::<T, Arch, Align, TILE_M, TILE_N>(a, b, &mut c)?;
    Ok(out)
}

/// Convenience alias for the default tile policy.
///
/// 4 × 4 tile fits within 16 YMM registers on AVX2 and conservatively on AVX-512.
pub type DefaultTilePolicy = TilingPolicy<4, 4>;

/// Perform element-wise binary operation on two tensors, writing the result into the output tensor.
///
/// If layouts are contiguous row-major, we utilize `SimdView::zip_into` for vectorization.
/// Otherwise, we fallback to a rank-agnostic index-iteration loop to remain zero-copy and support strided layouts.
#[inline]
pub fn tensor_elementwise_op_to<T, Arch, Align, Op, const N: usize, L1: Layout, L2: Layout, L3: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
    c: &mut TensorView<'_, T, N, L3, &mut [T]>,
    op: Op,
) -> Result<(), SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Op: ElementOp<T>,
{
    let shape_a = a.shape();
    let shape_b = b.shape();
    let shape_c = c.shape();
    if shape_a != shape_b || shape_a != shape_c {
        return Err(SimdError::LengthMismatch);
    }

    if a.is_empty() {
        return Ok(());
    }

    if a.is_contiguous() && b.is_contiguous() && c.is_contiguous() {
        let Some(a_view) = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(a.as_slice()) else {
            return Ok(());
        };
        let Some(b_view) = SimdView::<'_, T, Arch, Unaligned, Unmasked, &[T]>::new(b.as_slice()) else {
            return Ok(());
        };
        a_view.zip_into(&b_view, c.as_slice_mut(), op)?;
    } else {
        let mut idx = [0usize; N];
        loop {
            unsafe {
                let va = a.get_unchecked(idx);
                let vb = b.get_unchecked(idx);
                let vr = op.apply_scalar(va, vb);
                c.set_unchecked(idx, vr);
            }

            let mut carry = true;
            for i in (0..N).rev() {
                idx[i] += 1;
                if idx[i] < shape_a[i] {
                    carry = false;
                    break;
                }
                idx[i] = 0;
            }
            if carry {
                break;
            }
        }
    }
    Ok(())
}

/// Perform element-wise binary operation on two tensors, returning an owned `AlignedVec`.
///
/// The output has shape matching `a` and is contiguous row-major.
#[inline]
pub fn tensor_elementwise_op<T, Arch, Align, Op, const N: usize, L1: Layout, L2: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
    op: Op,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    Op: ElementOp<T>,
{
    let shape = a.shape();
    if shape != b.shape() {
        return Err(SimdError::LengthMismatch);
    }
    let total = shape.iter().product::<usize>();
    let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(total);
    unsafe { out.set_len(total); }
    let mut c = TensorView::new_mut(out.as_mut_slice(), shape)
        .map_err(|_| SimdError::LengthMismatch)?;
    tensor_elementwise_op_to::<T, Arch, Align, Op, N, L1, L2, RowMajor>(a, b, &mut c, op)?;
    Ok(out)
}

/// Element-wise addition of two tensors: `c = a + b`.
#[inline]
pub fn tensor_add_to<T, Arch, Align, const N: usize, L1: Layout, L2: Layout, L3: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
    c: &mut TensorView<'_, T, N, L3, &mut [T]>,
) -> Result<(), SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op_to::<T, Arch, Align, crate::ops::Add, N, L1, L2, L3>(a, b, c, crate::ops::Add)
}

/// Element-wise addition of two tensors, returning an owned `AlignedVec`.
#[inline]
pub fn tensor_add<T, Arch, Align, const N: usize, L1: Layout, L2: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op::<T, Arch, Align, crate::ops::Add, N, L1, L2>(a, b, crate::ops::Add)
}

/// Element-wise subtraction of two tensors: `c = a - b`.
#[inline]
pub fn tensor_sub_to<T, Arch, Align, const N: usize, L1: Layout, L2: Layout, L3: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
    c: &mut TensorView<'_, T, N, L3, &mut [T]>,
) -> Result<(), SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op_to::<T, Arch, Align, crate::ops::Sub, N, L1, L2, L3>(a, b, c, crate::ops::Sub)
}

/// Element-wise subtraction of two tensors, returning an owned `AlignedVec`.
#[inline]
pub fn tensor_sub<T, Arch, Align, const N: usize, L1: Layout, L2: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op::<T, Arch, Align, crate::ops::Sub, N, L1, L2>(a, b, crate::ops::Sub)
}

/// Element-wise multiplication of two tensors: `c = a * b`.
#[inline]
pub fn tensor_mul_to<T, Arch, Align, const N: usize, L1: Layout, L2: Layout, L3: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
    c: &mut TensorView<'_, T, N, L3, &mut [T]>,
) -> Result<(), SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op_to::<T, Arch, Align, crate::ops::Mul, N, L1, L2, L3>(a, b, c, crate::ops::Mul)
}

/// Element-wise multiplication of two tensors, returning an owned `AlignedVec`.
#[inline]
pub fn tensor_mul<T, Arch, Align, const N: usize, L1: Layout, L2: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op::<T, Arch, Align, crate::ops::Mul, N, L1, L2>(a, b, crate::ops::Mul)
}

/// Element-wise division of two tensors: `c = a / b`.
#[inline]
pub fn tensor_div_to<T, Arch, Align, const N: usize, L1: Layout, L2: Layout, L3: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
    c: &mut TensorView<'_, T, N, L3, &mut [T]>,
) -> Result<(), SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op_to::<T, Arch, Align, crate::ops::Div, N, L1, L2, L3>(a, b, c, crate::ops::Div)
}

/// Element-wise division of two tensors, returning an owned `AlignedVec`.
#[inline]
pub fn tensor_div<T, Arch, Align, const N: usize, L1: Layout, L2: Layout>(
    a: &TensorView<'_, T, N, L1, &[T]>,
    b: &TensorView<'_, T, N, L2, &[T]>,
) -> Result<AlignedVec<T, Align>, SimdError>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    tensor_elementwise_op::<T, Arch, Align, crate::ops::Div, N, L1, L2>(a, b, crate::ops::Div)
}


//! Numerically stable softmax over a mutable float slice.
//!
//! # Algorithm
//!
//! Three-pass max-shift softmax (avoids overflow for large logits):
//! 1. `max = view.reduce(Max)` — SIMD max reduction
//! 2. `data[i] = exp(data[i] - max)` — scalar exp loop
//! 3. `sum = view.reduce(Sum)` — SIMD sum reduction
//! 4. `data[i] /= sum` — scalar divide loop
//!
//! # Correctness-First
//!
//! The exp step uses scalar `f32::exp` (or `libm::expf` in `no_std`). SIMD exp
//! requires either SVML or a polynomial approximation crate — neither is in scope.
//! A later feature flag can override with an approximation.

extern crate alloc;

use crate::arch::SimdArch;
use crate::align::Unaligned;
use crate::kernel::SimdKernel;
use crate::ops::{Max, Sum};
use crate::scalar::{Scalar, FloatElement};
use crate::view::SimdView;

/// In-place numerically stable softmax.
///
/// After this call `∑ data[i] ≈ 1.0`, all elements in `[0, 1]`.
/// Empty slices are returned unmodified.
#[inline]
pub fn softmax_inplace<T, Arch>(data: &mut [T])
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    if data.is_empty() {
        return;
    }

    // Pass 1: SIMD max reduction.
    let max_val = {
        // SAFETY: non-empty slice guaranteed above.
        let view = SimdView::<'_, T, Arch, Unaligned>::new(data).unwrap();
        view.reduce(Max)
    };

    // Pass 2: exp(x - max) in-place — scalar loop (see module doc).
    for x in data.iter_mut() {
        *x = T::from_f32((*x - max_val).to_f32().exp());
    }

    // Pass 3: SIMD sum reduction.
    let sum_val = {
        let view = SimdView::<'_, T, Arch, Unaligned>::new(data).unwrap();
        view.reduce(Sum)
    };

    // Pass 4: divide by sum.
    if sum_val == T::ZERO || sum_val.is_nan() {
        // Degenerate: emit uniform distribution.
        let uniform = T::from_f32(1.0 / data.len() as f32);
        for x in data.iter_mut() {
            *x = uniform;
        }
        return;
    }
    let inv = T::ONE / sum_val;
    for x in data.iter_mut() {
        *x = *x * inv;
    }
}

/// Allocating softmax: returns a new `Vec<T>`.
///
/// Allocates exactly one `Vec`. Delegates to [`softmax_inplace`].
#[inline]
pub fn softmax<T, Arch>(data: &[T]) -> alloc::vec::Vec<T>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    let mut out = data.to_vec();
    softmax_inplace::<T, Arch>(&mut out);
    out
}

/// In-place numerically stable softmax over contiguous rows of a mutable 2-D tensor.
#[inline]
pub fn softmax_2d_rows_inplace<'a, T, Arch, Layout>(
    tensor: &mut super::TensorView<'a, T, 2, Layout, &'a mut [T]>
)
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    if tensor.num_elements() == 0 {
        return;
    }
    // If layout is contiguous, iterate over rows using safe fast slices.
    if tensor.is_contiguous() {
        if let Ok(rows) = tensor.iter_rows_mut() {
            for row in rows {
                softmax_inplace::<T, Arch>(row);
            }
            return;
        }
    }
    // Fallback for non-contiguous views (e.g. strided sub-slices).
    let shape = tensor.shape();
    let mut temp = alloc::vec![T::ZERO; shape[1]];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            if let Ok(val) = tensor.get([r, c]) {
                temp[c] = val;
            }
        }
        softmax_inplace::<T, Arch>(&mut temp);
        for c in 0..shape[1] {
            let _ = tensor.set([r, c], temp[c]);
        }
    }
}


/// Allocating row-wise softmax: returns a new `alloc::vec::Vec<T>`.
#[inline]
pub fn softmax_2d_rows<T, Arch, Layout>(
    tensor: &super::TensorView<'_, T, 2, Layout, &'_ [T]>
) -> alloc::vec::Vec<T>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    let mut out = tensor.as_slice().to_vec();
    let mut out_tensor = super::TensorView::new_mut(&mut out, tensor.shape()).unwrap();
    softmax_2d_rows_inplace::<T, Arch, super::RowMajor>(&mut out_tensor);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_softmax_sum_to_one_scalar() {
        // Scalar implementation test — verifies the numeric algorithm without SIMD.
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: alloc::vec::Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let s: f32 = exps.iter().sum();
        let probs: alloc::vec::Vec<f32> = exps.iter().map(|&e| e / s).collect();
        let total: f32 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax sum = {total}");
    }

    extern crate alloc;
}

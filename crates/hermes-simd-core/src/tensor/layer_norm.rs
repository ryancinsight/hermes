//! SIMD-accelerated Layer Normalization.
//!
//! # Algorithm
//!
//! Standard Layer Norm over a flat slice `x` of length `d`:
//! 1. Compute mean: `μ = (1/d) ∑ x[i]`            — one SIMD sum-reduction pass.
//! 2. Compute variance: `σ² = (1/d) ∑ (x[i]−μ)²`  — second SIMD dot-product pass.
//! 3. Normalize: `y[i] = (x[i]−μ) / √(σ²+ε)`      — scalar loop (no SIMD exp needed).
//! 4. Scale + shift: `y[i] = γ·y[i] + β`           — SIMD broadcast-multiply-add.
//!
//! # Zero-Allocation Contract
//!
//! `layer_norm_inplace` operates fully in-place with no heap allocation.
//! `layer_norm` allocates exactly one `Vec<T>` for the output.
//!
//! # Monomorphization
//!
//! All parameters are generic over `T: Scalar + FloatElement` and `Arch: SimdKernel<T>`.
//! Each combination produces a distinct, fully inlined binary.

extern crate alloc;

use crate::align::Unaligned;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::ops::{Sum, Dot};
use crate::scalar::{Scalar, FloatElement};
use crate::view::SimdView;

/// Numerically stable in-place layer normalization over `data`.
///
/// `gamma` and `beta` are affine parameters (scale and shift). Pass `None`
/// for identity scaling (`gamma = 1`, `beta = 0`).
///
/// # Panics
/// Panics if `gamma.is_some() && gamma.unwrap().len() != data.len()`.
/// Panics if `beta.is_some() && beta.unwrap().len() != data.len()`.
#[inline]
pub fn layer_norm_inplace<T, Arch>(
    data: &mut [T],
    eps: T,
    gamma: Option<&[T]>,
    beta: Option<&[T]>,
)
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    let d = data.len();
    if d == 0 {
        return;
    }

    if let Some(g) = gamma { assert_eq!(g.len(), d, "gamma length mismatch"); }
    if let Some(b) = beta  { assert_eq!(b.len(), d, "beta length mismatch"); }

    let d_f32 = d as f32;

    // --- Pass 1: SIMD sum → mean ---
    let sum = {
        let view = SimdView::<'_, T, Arch, Unaligned>::new(data).unwrap();
        view.reduce(Sum)
    };
    let mean = T::from_f32(sum.to_f32() / d_f32);

    // --- Pass 2: variance via dot product of (x − mean) with itself ---
    // We compute ∑ (x[i] − mean)² using the existing Dot reduction by
    // centering into a temporary scalar loop (avoids an extra allocation by
    // reusing the data slice as the centered buffer temporarily; we restore
    // after the variance computation).
    //
    // Order:
    //   a) subtract mean in-place   O(d)
    //   b) SIMD dot with itself     O(d)
    //   c) normalize in-place       O(d)
    //   d) scale + shift in-place   O(d)
    for x in data.iter_mut() {
        *x = *x - mean;
    }

    let var = {
        let view = SimdView::<'_, T, Arch, Unaligned>::new(data).unwrap();
        view.zip_reduce(&view, Dot).unwrap_or(T::ZERO)
    };
    let var_f32 = var.to_f32() / d_f32;
    let inv_std = T::from_f32(1.0 / (var_f32 + eps.to_f32()).sqrt());

    // --- Pass 3: normalize + scale + shift ---
    match (gamma, beta) {
        (Some(g), Some(b)) => {
            for (i, x) in data.iter_mut().enumerate() {
                *x = (*x * inv_std) * g[i] + b[i];
            }
        }
        (Some(g), None) => {
            for (i, x) in data.iter_mut().enumerate() {
                *x = (*x * inv_std) * g[i];
            }
        }
        (None, Some(b)) => {
            for x in data.iter_mut().zip(b.iter()) {
                *x.0 = *x.0 * inv_std + *x.1;
            }
        }
        (None, None) => {
            for x in data.iter_mut() {
                *x = *x * inv_std;
            }
        }
    }
}

use super::TensorView;

/// Allocating layer normalization: returns a new `Vec<T>`.
///
/// Allocates exactly one `Vec`. Delegates to [`layer_norm_inplace`].
#[inline]
pub fn layer_norm<T, Arch>(
    data: &[T],
    eps: T,
    gamma: Option<&[T]>,
    beta: Option<&[T]>,
) -> alloc::vec::Vec<T>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    let mut out = data.to_vec();
    layer_norm_inplace::<T, Arch>(&mut out, eps, gamma, beta);
    out
}

/// In-place numerically stable layer normalization over rows of a mutable 2-D tensor.
///
/// `gamma` and `beta` are optional 1-D affine parameter slices of length equal to the number of columns.
#[inline]
pub fn layer_norm_2d_rows_inplace<'a, T, Arch, Layout>(
    tensor: &mut TensorView<'a, T, 2, Layout, &'a mut [T]>,
    eps: T,
    gamma: Option<&[T]>,
    beta: Option<&[T]>,
)
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    if tensor.num_elements() == 0 {
        return;
    }
    
    let shape = tensor.shape();
    
    if let Some(g) = gamma { assert_eq!(g.len(), shape[1], "gamma length mismatch"); }
    if let Some(b) = beta  { assert_eq!(b.len(), shape[1], "beta length mismatch"); }

    if tensor.is_contiguous() {
        if let Ok(rows) = tensor.iter_rows_mut() {
            for row in rows {
                layer_norm_inplace::<T, Arch>(row, eps, gamma, beta);
            }
            return;
        }
    }
    
    let mut temp = alloc::vec![T::ZERO; shape[1]];
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            if let Ok(val) = tensor.get([r, c]) {
                temp[c] = val;
            }
        }
        layer_norm_inplace::<T, Arch>(&mut temp, eps, gamma, beta);
        for c in 0..shape[1] {
            let _ = tensor.set([r, c], temp[c]);
        }
    }
}

/// Allocating row-wise layer normalization: returns a new `alloc::vec::Vec<T>`.
///
/// The returned vector contains the flattened row-wise normalized tensor in C-order.
#[inline]
pub fn layer_norm_2d_rows<T, Arch, Layout>(
    tensor: &TensorView<'_, T, 2, Layout, &'_ [T]>,
    eps: T,
    gamma: Option<&[T]>,
    beta: Option<&[T]>,
) -> alloc::vec::Vec<T>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
{
    let shape = tensor.shape();
    let total = shape[0] * shape[1];
    let mut out = alloc::vec![T::ZERO; total];
    if tensor.is_contiguous() {
        out.copy_from_slice(tensor.as_slice());
    } else {
        let mut idx = 0;
        for r in 0..shape[0] {
            for c in 0..shape[1] {
                out[idx] = tensor.get([r, c]).unwrap_or(T::ZERO);
                idx += 1;
            }
        }
    }
    
    let mut out_tensor = TensorView::new_mut(&mut out, shape).unwrap();
    layer_norm_2d_rows_inplace::<T, Arch, super::RowMajor>(&mut out_tensor, eps, gamma, beta);
    out
}


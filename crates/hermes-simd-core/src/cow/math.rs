//! Mathematical extensions for [`SimdCow`]: norm, normalize, scalar-broadcast ops.
//!
//! # Zero-Cost Contract
//!
//! All methods monomorphize per `(T, Arch, Align)`. The `Arch` and `Align` ZST markers
//! are erased at codegen. Scalar-broadcast ops (`add_scalar_cow`, `mul_scalar_cow`, etc.)
//! allocate exactly one `AlignedVec` output; no intermediate buffer is allocated.
//!
//! # Norm and Normalize
//!
//! - `norm_sq` — squared Euclidean norm: `∑ self[i]²`. Delegates to `zip_reduce(Dot)`.
//! - `norm`    — Euclidean norm: `sqrt(norm_sq)`. Uses `T::sqrt_scalar` from `FloatElement`.
//! - `normalize` — returns a unit-length owned `SimdCow<'static, T, Arch, Align>`.
//!   Empty or zero-norm vectors return `zeros(self.len())` rather than NaN / division by zero.
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
use crate::ops::{Dot, Sub};
use crate::scalar::{FloatElement, Scalar};
use crate::vec::AlignedVec;
use crate::view::SimdError;

extern crate alloc;

// ---------------------------------------------------------------------------
// Scalar-broadcast arithmetic
// ---------------------------------------------------------------------------

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    /// Add scalar `rhs` to every element: `out[i] = self[i] + rhs`.
    ///
    /// One allocation. No second `SimdCow` allocation.
    #[inline]
    pub fn add_scalar_cow(&self, rhs: T) -> SimdCow<'static, T, Arch, Align> {
        broadcast_op::<T, Arch, Align>(
            self,
            rhs,
            |a, b| a + b,
            |va, vsplat| unsafe { Arch::add(va, vsplat) },
        )
    }

    /// Subtract scalar `rhs` from every element: `out[i] = self[i] - rhs`.
    ///
    /// One allocation.
    #[inline]
    pub fn sub_scalar_cow(&self, rhs: T) -> SimdCow<'static, T, Arch, Align> {
        broadcast_op::<T, Arch, Align>(
            self,
            rhs,
            |a, b| a - b,
            |va, vsplat| unsafe { Arch::sub(va, vsplat) },
        )
    }

    /// Multiply every element by scalar `rhs`: `out[i] = self[i] * rhs`.
    ///
    /// One allocation. For in-place scaling use [`SimdCow::scale_in_place`].
    #[inline]
    pub fn mul_scalar_cow(&self, rhs: T) -> SimdCow<'static, T, Arch, Align> {
        broadcast_op::<T, Arch, Align>(
            self,
            rhs,
            |a, b| a * b,
            |va, vsplat| unsafe { Arch::mul(va, vsplat) },
        )
    }

    /// Elementwise division by scalar `rhs`: `out[i] = self[i] / rhs`.
    ///
    /// One allocation.
    #[inline]
    pub fn div_scalar_cow(&self, rhs: T) -> SimdCow<'static, T, Arch, Align> {
        broadcast_op::<T, Arch, Align>(
            self,
            rhs,
            |a, b| a / b,
            |va, vsplat| unsafe { Arch::div(va, vsplat) },
        )
    }

    /// Elementwise division: `out[i] = self[i] / other[i]`.
    ///
    /// One allocation.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline]
    pub fn div_cow(
        &self,
        other: &SimdCow<'_, T, Arch, Align>,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, crate::ops::Div)
    }

    /// Elementwise subtraction returning owned `SimdCow`, non-method form for symmetry.
    ///
    /// Equivalent to `self.sub_cow(other)` — delegates to `zip_cow(Sub)`.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline]
    pub fn sub_cow_op(
        &self,
        other: &SimdCow<'_, T, Arch, Align>,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, Sub)
    }
}

// ---------------------------------------------------------------------------
// Norm and normalize (float-element only)
// ---------------------------------------------------------------------------

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    T: Scalar + FloatElement,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    /// Squared Euclidean norm: `∑ self[i]²`.
    ///
    /// Zero-copy: delegates to `zip_reduce(Dot)` which is a single SIMD pass.
    #[inline]
    pub fn norm_sq(&self) -> T {
        let v = self.view();
        v.zip_reduce(&v, Dot).unwrap_or(T::ZERO)
    }

    /// Euclidean norm: `√(∑ self[i]²)`.
    ///
    /// Delegates to `norm_sq` then `T::sqrt_scalar`.
    #[inline]
    #[must_use]
    pub fn norm(&self) -> T {
        self.norm_sq().sqrt()
    }

    /// Returns a unit-length copy: `self / ‖self‖`.
    ///
    /// - Empty vector → empty `SimdCow<'static, T, Arch, Align>`.
    /// - Zero-norm vector → `zeros(self.len())` (safe, no NaN).
    /// - Otherwise → `self * (1 / ‖self‖)`.
    ///
    /// One allocation for the output `AlignedVec`.
    #[inline]
    #[must_use]
    pub fn normalize(&self) -> SimdCow<'static, T, Arch, Align> {
        let n = self.norm();
        if n == T::ZERO {
            return SimdCow::zeros(self.len());
        }
        // Compute reciprocal once, multiply — avoids a per-element division.
        let inv = T::ONE / n;
        self.mul_scalar_cow(inv)
    }

    /// Scalar histogram over this cow's values.
    ///
    /// Partitions `[lo, hi)` into `n_bins` equal-width bins and counts how many
    /// elements fall in each bin. Elements outside `[lo, hi)` are ignored.
    ///
    /// Returns a `Vec<usize>` of length `n_bins`.
    ///
    /// # Panics
    /// Panics if `n_bins == 0` or `lo >= hi`.
    #[inline]
    pub fn histogram_cow(&self, n_bins: usize, lo: T, hi: T) -> alloc::vec::Vec<usize>
    where
        T: PartialOrd,
    {
        assert!(n_bins > 0, "n_bins must be > 0");
        assert!(lo < hi, "lo must be < hi");

        // Bin indices are computed in f64: it is a strict superset of every
        // supported lane precision (f16/bf16/f32/f64), so the index — an
        // integer output, never narrowed back to `T` — is exact for all `T`.
        let lo_w = lo.to_f64();
        let bin_width = (hi.to_f64() - lo_w) / n_bins as f64;
        let mut counts = alloc::vec![0usize; n_bins];

        for &x in self.as_ref() {
            if x < lo || x >= hi {
                continue;
            }
            let bin = (((x.to_f64() - lo_w) / bin_width) as usize).min(n_bins - 1);
            counts[bin] += 1;
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// SIMD broadcast-apply: `out[i] = scalar_op(data[i], rhs)`.
///
/// Uses `Arch::splat(rhs)` once, then loops over SIMD vectors with `vector_op`,
/// followed by a scalar tail loop with `scalar_op`. One allocation.
#[inline(always)]
fn broadcast_op<T, Arch, Align>(
    cow: &SimdCow<'_, T, Arch, Align>,
    rhs: T,
    scalar_op: impl Fn(T, T) -> T + Copy,
    vector_op: impl Fn(Arch::Vector, Arch::Vector) -> Arch::Vector + Copy,
) -> SimdCow<'static, T, Arch, Align>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    let data = cow.as_ref();
    let len = data.len();
    let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);

    let lane_count = Arch::LANE_COUNT;
    let simd_len = (len / lane_count) * lane_count;
    let ptr_in = data.as_ptr();
    let ptr_out = out.as_mut_ptr();

    // SAFETY: `with_capacity(len)` reserved `len` elements and `ptr_in` covers
    // the same `len` elements, so every access below stays inside its
    // allocation. The length is raised only after the vector and scalar loops
    // have together written every element, so no reference spans uninitialized
    // memory.
    unsafe {
        let vsplat = Arch::splat(rhs);
        let load = |p: *const T| -> Arch::Vector {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                Arch::load_aligned(p)
            } else {
                Arch::load_unaligned(p)
            }
        };
        let store = |p: *mut T, v: Arch::Vector| {
            if crate::align::is_aligned_for_arch::<Arch, Align>() {
                Arch::store_aligned(p, v);
            } else {
                Arch::store_unaligned(p, v);
            }
        };
        let mut i = 0usize;
        while i < simd_len {
            let va = load(ptr_in.add(i));
            let vr = vector_op(va, vsplat);
            store(ptr_out.add(i), vr);
            i += lane_count;
        }
        for i in simd_len..len {
            core::ptr::write(ptr_out.add(i), scalar_op(*ptr_in.add(i), rhs));
        }
        out.set_len(len);
    }

    SimdCow::Owned(out)
}

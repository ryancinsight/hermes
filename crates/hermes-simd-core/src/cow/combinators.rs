//! CoW combinators: `zip_cow`, `transform_in_place`, `reduce`, arithmetic shorthands,
//! and ergonomic `From`/`Extend` conversions for `SimdCow`.
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

use super::types::SimdCow;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::Unmasked;
use crate::kernel::SimdKernel;
use crate::ops::{ElementOp, ReductionOp};
use crate::scalar::Scalar;
use crate::vec::AlignedVec;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    /// Sum all elements using `SimdView::sum`.
    #[inline(always)]
    #[must_use]
    pub fn sum(&self) -> T {
        self.view().sum()
    }

    /// Compute the dot product with another `SimdCow`.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> Result<T, SimdError> {
        self.view().dot(&other.view())
    }

    /// Apply an `ElementOp` pairwise to `self` and `other`, returning a fully-owned
    /// `SimdCow<'static, T, Arch, Align>` backed by a single `AlignedVec` allocation.
    ///
    /// The SIMD vectorized loop processes `floor(len / LANE_COUNT) * LANE_COUNT` elements.
    /// The scalar tail uses `Op::apply_scalar` directly on individual elements — no
    /// vector loads or stores are performed in the tail, avoiding all boundary-condition UB.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if `self.len() != other.len()`.
    pub fn zip_cow<Op: ElementOp<T>>(
        &self,
        other: &SimdCow<'_, T, Arch, Align>,
        _op: Op,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        if self.len() != other.len() {
            return Err(SimdError::LengthMismatch);
        }
        let len = self.len();
        if len == 0 {
            return Ok(SimdCow::Owned(AlignedVec::new()));
        }

        let mut out = AlignedVec::with_capacity(len);
        let out_ptr: *mut T = out.as_mut_ptr();

        let view_self = self.view();
        let view_other = other.view();

        let mut chunks_self = view_self.simd_chunks();
        let mut chunks_other = view_other.simd_chunks();

        let mut i = 0usize;
        for (chunk_self, chunk_other) in (&mut chunks_self).zip(&mut chunks_other) {
            unsafe {
                let va = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(chunk_self.as_ptr())
                } else {
                    Arch::load_unaligned(chunk_self.as_ptr())
                };
                let vb = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(chunk_other.as_ptr())
                } else {
                    Arch::load_unaligned(chunk_other.as_ptr())
                };
                let vr = _op.apply::<Arch>(va, vb);
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::store_aligned(out_ptr.add(i), vr);
                } else {
                    Arch::store_unaligned(out_ptr.add(i), vr);
                }
            }
            i += Arch::LANE_COUNT;
        }

        let remainder_self = chunks_self.remainder();
        let remainder_other = chunks_other.remainder();

        for (&a, &b) in remainder_self.iter().zip(remainder_other.iter()) {
            unsafe {
                core::ptr::write(out_ptr.add(i), _op.apply_scalar(a, b));
            }
            i += 1;
        }

        // SAFETY: `with_capacity(len)` reserved `len` elements, and the vector
        // and remainder loops above have together written every one of them
        // through `out_ptr`. The length is raised only now, so no reference to
        // this buffer ever spanned uninitialized memory.
        unsafe {
            out.set_len(len);
        }

        Ok(SimdCow::Owned(out))
    }

    /// Apply an `ElementOp` in-place: `self[i] = op(self[i], other[i])`.
    ///
    /// If `self` is `Borrowed`, promotes to `Owned` first (one allocation).
    /// Subsequent calls on the same already-owned `self` are allocation-free.
    ///
    /// The scalar tail uses `Op::apply_scalar` to avoid vector boundary UB.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if `self.len() != other.len()`.
    ///
    /// # Panics
    ///
    /// Panics if the owned buffer cannot be represented as the declared
    /// alignment. This indicates a violated `SimdCow` alignment invariant.
    pub fn transform_in_place<Op: ElementOp<T>>(
        &mut self,
        other: &SimdCow<'_, T, Arch, Align>,
        _op: Op,
    ) -> Result<(), SimdError> {
        if self.len() != other.len() {
            return Err(SimdError::LengthMismatch);
        }
        // `to_mut` promotes borrowed → owned (one allocation if borrowed, free if owned).
        // Use the returned reference directly — no secondary match required.
        let out_slice = self.to_mut().as_mut_slice();

        let other_view = other.view();

        let self_view: SimdView<'_, T, Arch, Align, Unmasked, &mut [T]> =
            SimdView::new_mut(out_slice).expect("alignment invariant violated");

        let mut chunks_self = self_view.simd_chunks_mut();
        let mut chunks_other = other_view.simd_chunks();

        for (mut chunk_self, chunk_other) in (&mut chunks_self).zip(&mut chunks_other) {
            unsafe {
                let va = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(chunk_self.as_ptr())
                } else {
                    Arch::load_unaligned(chunk_self.as_ptr())
                };
                let vb = if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(chunk_other.as_ptr())
                } else {
                    Arch::load_unaligned(chunk_other.as_ptr())
                };
                let vr = _op.apply::<Arch>(va, vb);
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::store_aligned(chunk_self.as_mut_ptr(), vr);
                } else {
                    Arch::store_unaligned(chunk_self.as_mut_ptr(), vr);
                }
            }
        }

        let tail_self = chunks_self.into_remainder();
        let tail_other = chunks_other.remainder();
        for (a, &b) in tail_self.iter_mut().zip(tail_other.iter()) {
            *a = _op.apply_scalar(*a, b);
        }

        Ok(())
    }

    /// Apply a `ReductionOp` to this `SimdCow`, delegating to `SimdView::reduce`.
    ///
    /// Monomorphization is shared with the view path — no duplicate code.
    #[inline(always)]
    pub fn reduce<Op: ReductionOp<T>>(&self, op: Op) -> T {
        self.view().reduce(op)
    }

    // -----------------------------------------------------------------------
    // Arithmetic combinators — each returns `SimdCow<'static, ...>` (owned)
    // -----------------------------------------------------------------------

    /// Elementwise addition: `self[i] + other[i]`.
    ///
    /// Allocates one `AlignedVec` output. Zero-copy on both operands.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn add_cow(
        &self,
        other: &SimdCow<'_, T, Arch, Align>,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, crate::ops::Add)
    }

    /// Elementwise subtraction: `self[i] - other[i]`.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn sub_cow(
        &self,
        other: &SimdCow<'_, T, Arch, Align>,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, crate::ops::Sub)
    }

    /// Elementwise multiplication: `self[i] * other[i]`.
    ///
    /// # Errors
    /// Returns `SimdError::LengthMismatch` if lengths differ.
    #[inline(always)]
    pub fn mul_cow(
        &self,
        other: &SimdCow<'_, T, Arch, Align>,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        self.zip_cow(other, crate::ops::Mul)
    }
}

// ---------------------------------------------------------------------------
// Ergonomic conversions
// ---------------------------------------------------------------------------

/// Adopt an `AlignedVec` as an owned `SimdCow` — zero-cost, no allocation.
impl<T, Arch, Align> From<AlignedVec<T, Align>> for SimdCow<'_, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    #[inline]
    fn from(vec: AlignedVec<T, Align>) -> Self {
        Self::Owned(vec)
    }
}

/// Copy a standard `Vec<T>` into a new owned `SimdCow`, allocating one aligned buffer.
impl<T: Copy, Arch, Align> From<alloc::vec::Vec<T>> for SimdCow<'_, T, Arch, Align>
where
    Arch: SimdArch,
    Align: Alignment,
{
    #[inline]
    fn from(v: alloc::vec::Vec<T>) -> Self {
        Self::Owned(AlignedVec::from_slice(&v))
    }
}

impl<'a, T: Copy + 'a, Arch, Align> Extend<T> for SimdCow<'a, T, Arch, Align>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    /// Extend the `SimdCow`, promoting to owned if currently borrowed.
    ///
    /// After promotion, subsequent `extend` calls are allocation-free as long as the
    /// `AlignedVec` has sufficient capacity.
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let vec = self.to_mut();
        // Reserve the iterator's lower-bound size up front so a bulk extend does
        // one reallocation rather than the ⌈log₂ n⌉ a push loop would incur
        // (`size_hint().0` is exact for the common sized-iterator sources).
        vec.reserve(iter.size_hint().0);
        for item in iter {
            vec.push(item);
        }
    }
}

//! CoW combinators: `zip_cow`, `transform_in_place`, `reduce`, arithmetic shorthands,
//! and ergonomic `From`/`Extend` conversions for `SimdCow`.

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
impl<'a, T, Arch, Align> From<AlignedVec<T, Align>> for SimdCow<'a, T, Arch, Align>
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
impl<'a, T: Copy, Arch, Align> From<alloc::vec::Vec<T>> for SimdCow<'a, T, Arch, Align>
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
        let vec = self.to_mut();
        for item in iter {
            vec.push(item);
        }
    }
}

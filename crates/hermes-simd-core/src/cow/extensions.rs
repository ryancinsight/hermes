//! Unary map, in-place scale, splat-fill, argmin/argmax, gather, and prefix-scan
//! extensions for `SimdCow`.
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
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::vec::AlignedVec;
use crate::view::SimdError;

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    /// Apply a `UnaryOp<T>` to every element, returning a fully-owned
    /// `SimdCow<'static, T, Arch, Align>` backed by a single `AlignedVec` allocation.
    ///
    /// Zero intermediate copies: one allocation, one vectorized pass.
    #[inline]
    pub fn map_unary<Op: crate::ops::UnaryOp<T>>(
        &self,
        op: Op,
    ) -> SimdCow<'static, T, Arch, Align> {
        // Same operation as `map_cow`, which owns the single implementation:
        // it writes the output buffer through a raw pointer and raises the
        // length only once every element is initialized.
        self.map_cow(op)
    }

    /// Apply a `UnaryOp<T>` in-place: `self[i] = op(self[i])`.
    ///
    /// Promotes `self` to owned if currently borrowed (one allocation).
    /// Subsequent calls on the same already-owned `SimdCow` are allocation-free.
    #[inline]
    pub fn map_unary_in_place<Op: crate::ops::UnaryOp<T>>(&mut self, op: Op) {
        self.view_mut().map_unary_in_place(op);
    }

    /// Multiply every element by `scalar` in-place: `self[i] *= scalar`.
    ///
    /// Uses `Arch::splat(scalar)` + `Arch::mul` to broadcast-multiply without
    /// a second `SimdCow`. Promotes to owned if currently borrowed (one allocation).
    #[inline]
    pub fn scale_in_place(&mut self, scalar: T) {
        let len = self.len();
        if len == 0 {
            return;
        }
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let vec = self.to_mut();
        let ptr = vec.as_mut_ptr();

        unsafe {
            let vsplat = Arch::splat(scalar);

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
                let p = ptr.add(i);
                let v = load(p);
                store(p, Arch::mul(v, vsplat));
                i += lane_count;
            }
        }

        // Scalar tail
        let slice = vec.as_mut_slice();
        for i in simd_len..len {
            slice[i] = slice[i] * scalar;
        }
    }

    /// Return an owned `SimdCow` with every element multiplied by `scalar`.
    ///
    /// One allocation. Delegates to the fused [`SimdCow::mul_scalar_cow`]
    /// broadcast kernel (single read+write pass); the previous copy-then-
    /// `scale_in_place` body cost a second full read+write pass over the
    /// buffer for a bitwise-identical result.
    #[inline]
    pub fn scale(&self, scalar: T) -> SimdCow<'static, T, Arch, Align> {
        self.mul_scalar_cow(scalar)
    }

    /// Construct an owned `SimdCow` of length `len` with every element set to `value`.
    ///
    /// Uses `Arch::splat` + `Arch::store_unaligned` for the SIMD prefix;
    /// scalar assignment for the tail. One allocation.
    #[inline]
    pub fn splat_fill(value: T, len: usize) -> SimdCow<'static, T, Arch, Align> {
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr = out.as_mut_ptr();

        // SAFETY: `with_capacity(len)` reserved `len` elements, so every store
        // below `len` stays inside the allocation. The vector's length is
        // raised only after the vector and scalar loops have together written
        // all `len` elements, so no reference spans uninitialized memory.
        unsafe {
            let vsplat = Arch::splat(value);
            let mut i = 0usize;
            while i < simd_len {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::store_aligned(ptr.add(i), vsplat);
                } else {
                    Arch::store_unaligned(ptr.add(i), vsplat);
                }
                i += lane_count;
            }
            for i in simd_len..len {
                core::ptr::write(ptr.add(i), value);
            }
            out.set_len(len);
        }

        SimdCow::Owned(out)
    }

    /// Construct an owned `SimdCow` of length `len` filled with `T::ZERO`.
    #[inline]
    pub fn zeros(len: usize) -> SimdCow<'static, T, Arch, Align> {
        Self::splat_fill(T::ZERO, len)
    }

    /// Construct an owned `SimdCow` of length `len` filled with `T::ONE`.
    #[inline]
    pub fn ones(len: usize) -> SimdCow<'static, T, Arch, Align> {
        Self::splat_fill(T::ONE, len)
    }

    /// Returns the first minimum, or `None` for empty or NaN-containing data.
    #[inline]
    #[must_use]
    pub fn argmin(&self) -> Option<(usize, T)>
    where
        T: crate::scalar::NumericElement,
    {
        self.view().argmin()
    }

    /// Returns the first maximum, or `None` for empty or NaN-containing data.
    #[inline]
    #[must_use]
    pub fn argmax(&self) -> Option<(usize, T)>
    where
        T: crate::scalar::NumericElement,
    {
        self.view().argmax()
    }

    /// Indirectly load (gather) elements from this view using indices, returning a new owned `SimdCow`.
    ///
    /// # Errors
    /// Returns `SimdError::IndexOutOfBounds` if any index in `indices` is out of bounds.
    #[inline]
    pub fn gather(&self, indices: &[i32]) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        let len = indices.len();
        let mut out = AlignedVec::with_capacity(len);
        // Gather fills the reserved capacity directly and reports how many
        // elements it wrote; nothing is written on the error path, so `out`
        // stays length-zero and drops no uninitialized element.
        self.view()
            .gather_into_uninit(indices, out.spare_capacity_mut())?;
        // SAFETY: `gather_into_uninit` returned `Ok`, so it initialized exactly
        // `len` elements of the reserved capacity.
        unsafe { out.set_len(len) };
        Ok(SimdCow::Owned(out))
    }

    /// Perform a prefix scan (inclusive or exclusive) of the view using the specified operation,
    /// returning a new owned `SimdCow`.
    #[inline]
    pub fn prefix_scan<Op, SMode>(
        &self,
        op: Op,
        mode: SMode,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError>
    where
        Op: crate::ops::ScanOp<T>,
        SMode: crate::ops::ScanMode,
    {
        let len = self.len();
        let mut out = AlignedVec::with_capacity(len);
        // Scan fills the reserved capacity directly; the only error is an
        // insufficient-length one it checks before writing, so on error `out`
        // stays length-zero and drops no uninitialized element.
        self.view()
            .prefix_scan_into_uninit(out.spare_capacity_mut(), op, mode)?;
        // SAFETY: `prefix_scan_into_uninit` returned `Ok`, so it initialized
        // exactly `len` elements of the reserved capacity.
        unsafe { out.set_len(len) };
        Ok(SimdCow::Owned(out))
    }

    /// Perform an in-place prefix scan (inclusive or exclusive) of the view using the specified operation.
    ///
    /// Promotes `self` to owned if currently borrowed (one allocation).
    /// Subsequent calls on the same already-owned `SimdCow` are allocation-free.
    #[inline]
    pub fn prefix_scan_in_place<Op, SMode>(&mut self, op: Op, mode: SMode) -> Result<(), SimdError>
    where
        Op: crate::ops::ScanOp<T>,
        SMode: crate::ops::ScanMode,
    {
        // `view_mut` promotes borrowed → owned (one allocation if borrowed,
        // free if owned). The scan itself is the single authoritative
        // vectorized implementation on `SimdView`.
        self.view_mut().prefix_scan_in_place(op, mode);
        Ok(())
    }
}

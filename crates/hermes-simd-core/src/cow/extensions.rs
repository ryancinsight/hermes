//! Unary map, in-place scale, splat-fill, argmin/argmax, gather, and prefix-scan
//! extensions for `SimdCow`.

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
        let len = self.len();
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        // SAFETY: we write every element below before returning.
        unsafe {
            out.set_len(len);
        }
        self.view()
            .map_unary(op, out.as_mut_slice())
            .expect("invariant: output length equals input length");
        SimdCow::Owned(out)
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
    /// One allocation. Delegates to `clone` + `scale_in_place`.
    #[inline]
    pub fn scale(&self, scalar: T) -> SimdCow<'static, T, Arch, Align> {
        let mut owned: SimdCow<'static, T, Arch, Align> = SimdCow::from_slice(self.as_ref());
        owned.scale_in_place(scalar);
        owned
    }

    /// Construct an owned `SimdCow` of length `len` with every element set to `value`.
    ///
    /// Uses `Arch::splat` + `Arch::store_unaligned` for the SIMD prefix;
    /// scalar assignment for the tail. One allocation.
    #[inline]
    pub fn splat_fill(value: T, len: usize) -> SimdCow<'static, T, Arch, Align> {
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        unsafe {
            out.set_len(len);
        }
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr = out.as_mut_ptr();

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
        }

        let slice = out.as_mut_slice();
        for i in simd_len..len {
            slice[i] = value;
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

    /// Returns `Some((index, value))` of the minimum element, or `None` for empty.
    #[inline]
    pub fn argmin(&self) -> Option<(usize, T)>
    where
        T: crate::scalar::NumericElement,
    {
        self.view().argmin()
    }

    /// Returns `Some((index, value))` of the maximum element, or `None` for empty.
    #[inline]
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
        unsafe {
            out.set_len(len);
        }
        self.view().gather(indices, out.as_mut_slice())?;
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
        unsafe {
            out.set_len(len);
        }
        self.view().prefix_scan(out.as_mut_slice(), op, mode)?;
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

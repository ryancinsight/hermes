//! Unary and ternary CoW transformations: `map_cow`, `fma_cow`.
//!
//! # Design
//!
//! All functions take `&SimdCow` (borrow) and return a new `SimdCow<'static, T, Arch, Align>`
//! backed by exactly one `AlignedVec` allocation. The input is never mutated.
//!
//! `map_cow` generalizes `UnaryOp` dispatch so callers pass a ZST strategy type
//! (e.g. `ops::Abs`, `ops::Neg`, `ops::Sqrt`) and the kernel dispatches through
//! the existing `UnaryOp<T>` trait without any runtime branches.
//!
//! `fma_cow` implements `out[i] = a[i] * b[i] + c[i]` using `Arch::fmadd` in the
//! SIMD region and `T::scalar_fmadd` in the scalar tail.
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
use crate::ops::UnaryOp;
use crate::scalar::Scalar;
use crate::vec::AlignedVec;
use crate::view::SimdError;

// ---------------------------------------------------------------------------
// map_cow — generic unary op
// ---------------------------------------------------------------------------

impl<'a, T: 'a, Arch, Align> SimdCow<'a, T, Arch, Align>
where
    T: Scalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    /// Apply a zero-sized `UnaryOp` to every element, returning a new owned `SimdCow`.
    ///
    /// One allocation. The input is unchanged.
    ///
    /// # Example
    /// ```rust,ignore
    /// let abs_cow = cow.map_cow(ops::Abs);
    /// let neg_cow = cow.map_cow(ops::Neg);
    /// let sqrt_cow = cow.map_cow(ops::Sqrt);
    /// ```
    #[inline]
    pub fn map_cow<Op: UnaryOp<T>>(&self, op: Op) -> SimdCow<'static, T, Arch, Align> {
        let data = self.as_ref();
        let len = data.len();
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);

        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr_in = data.as_ptr();
        let ptr_out = out.as_mut_ptr();

        // SAFETY: `with_capacity(len)` reserved `len` elements, so writes below
        // `len` stay inside the allocation, and `ptr_in` covers the same `len`
        // elements. The vector's length is raised only once both loops have
        // written every element, so no reference ever spans uninitialized
        // memory and nothing observes the buffer before it is complete.
        unsafe {
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
                let v = load(ptr_in.add(i));
                let r = UnaryOp::apply::<Arch>(op, v);
                store(ptr_out.add(i), r);
                i += lane_count;
            }
            for i in simd_len..len {
                core::ptr::write(ptr_out.add(i), UnaryOp::apply_scalar(op, *ptr_in.add(i)));
            }
            out.set_len(len);
        }

        SimdCow::Owned(out)
    }

    /// Fused multiply-add: `out[i] = self[i] * b[i] + c[i]`.
    ///
    /// Uses `Arch::fmadd` in the SIMD region and `T::scalar_fmadd` in the tail.
    /// One allocation. Returns `Err(SimdError::LengthMismatch)` if lengths differ.
    #[inline]
    pub fn fma_cow(
        &self,
        b: &SimdCow<'_, T, Arch, Align>,
        c: &SimdCow<'_, T, Arch, Align>,
    ) -> Result<SimdCow<'static, T, Arch, Align>, SimdError> {
        let data_a = self.as_ref();
        let data_b = b.as_ref();
        let data_c = c.as_ref();

        let len = data_a.len();
        if len != data_b.len() || len != data_c.len() {
            return Err(SimdError::LengthMismatch);
        }

        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);

        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr_a = data_a.as_ptr();
        let ptr_b = data_b.as_ptr();
        let ptr_c = data_c.as_ptr();
        let ptr_o = out.as_mut_ptr();

        // SAFETY: `with_capacity(len)` reserved `len` elements and the three
        // inputs were length-checked above, so every access below stays inside
        // its allocation. The vector's length is raised only after both loops
        // have written every element, so no reference spans uninitialized
        // memory and nothing observes the buffer before it is complete.
        unsafe {
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
                let va = load(ptr_a.add(i));
                let vb = load(ptr_b.add(i));
                let vc = load(ptr_c.add(i));
                let vr = Arch::fmadd(va, vb, vc);
                store(ptr_o.add(i), vr);
                i += lane_count;
            }
            for i in simd_len..len {
                let value = *ptr_a.add(i) * *ptr_b.add(i) + *ptr_c.add(i);
                core::ptr::write(ptr_o.add(i), value);
            }
            out.set_len(len);
        }

        Ok(SimdCow::Owned(out))
    }
}

// Unit tests moved to integration tests in crates/hermes-simd/tests/select_unary_tests.rs

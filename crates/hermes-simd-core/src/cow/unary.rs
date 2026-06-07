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

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::ops::UnaryOp;
use crate::scalar::Scalar;
use crate::vec::AlignedVec;
use crate::view::SimdError;
use super::SimdCow;

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
        let len  = data.len();
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        // SAFETY: every element written in the loop below.
        unsafe { out.set_len(len); }

        let lane_count = Arch::LANE_COUNT;
        let simd_len   = (len / lane_count) * lane_count;
        let ptr_in  = data.as_ptr();
        let ptr_out = out.as_mut_ptr();

        unsafe {
            let load = |p: *const T| -> Arch::Vector {
                if Align::IS_ALIGNED { Arch::load_aligned(p) }
                else                 { Arch::load_unaligned(p) }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if Align::IS_ALIGNED { Arch::store_aligned(p, v); }
                else                 { Arch::store_unaligned(p, v); }
            };
            let mut i = 0usize;
            while i < simd_len {
                let v = load(ptr_in.add(i));
                let r = UnaryOp::apply::<Arch>(op, v);
                store(ptr_out.add(i), r);
                i += lane_count;
            }
        }

        let out_slice = out.as_mut_slice();
        for i in simd_len..len {
            out_slice[i] = UnaryOp::apply_scalar(op, data[i]);
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
        unsafe { out.set_len(len); }

        let lane_count = Arch::LANE_COUNT;
        let simd_len   = (len / lane_count) * lane_count;
        let ptr_a = data_a.as_ptr();
        let ptr_b = data_b.as_ptr();
        let ptr_c = data_c.as_ptr();
        let ptr_o = out.as_mut_ptr();

        unsafe {
            let load = |p: *const T| -> Arch::Vector {
                if Align::IS_ALIGNED { Arch::load_aligned(p) }
                else                 { Arch::load_unaligned(p) }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if Align::IS_ALIGNED { Arch::store_aligned(p, v); }
                else                 { Arch::store_unaligned(p, v); }
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
        }

        let out_slice = out.as_mut_slice();
        for i in simd_len..len {
            out_slice[i] = data_a[i] * data_b[i] + data_c[i];
        }

        Ok(SimdCow::Owned(out))
    }
}

// Unit tests moved to integration tests in crates/hermes-simd/tests/select_unary_tests.rs

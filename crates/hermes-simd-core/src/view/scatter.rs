//! Vectorized indirect store (scatter) on views backed by `&'a mut [T]`.
//!
//! The write-side dual of [`gather`](crate::view::gather): where `gather` reads
//! `view[indices[i]]` into contiguous output, `scatter` writes contiguous input
//! into `view[indices[i]]`. Both route full vectors through the provider seam
//! ([`crate::kernel::SimdGather::scatter`]) and the final partial vector
//! through the masked seam ([`crate::kernel::SimdGather::scatter_masked`]),
//! so no element-at-a-time tail loop and
//! no read or write past the live slice remains.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::{SimdGather, SimdLoadStore, SimdMask, MAX_SIMD_LANES};
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<
        'a,
        T: 'a,
        Arch: SimdArch + SimdGather<T> + SimdLoadStore<T> + SimdMask<T>,
        Align: Alignment,
        Mode: ExecutionMode,
    > SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
where
    T: Scalar,
{
    /// Indirectly store (scatter) `src` into this view: `self[indices[i]] = src[i]`.
    ///
    /// Every index is validated before any element is written, so the `Err` path
    /// leaves the view untouched.
    ///
    /// # Duplicate indices
    ///
    /// When `indices` repeats a value, the element from the highest such `i`
    /// wins. This is the hardware last-writer-wins rule
    /// (`vscatterdps`/`vscatterdpd`), so it holds identically on the native
    /// AVX-512 path and the lane-sequential fallback. Scatter is a store, not an
    /// accumulate: callers wanting `+=` over duplicate indices must deduplicate
    /// or use a sparse accumulation kernel.
    ///
    /// # Errors
    /// Returns [`SimdError::InsufficientInputLength`] if `src.len() < indices.len()`.
    /// Returns [`SimdError::IndexOutOfBounds`] if any index is negative or `>=` the view length.
    #[inline]
    pub fn scatter(&mut self, indices: &[i32], src: &[T]) -> Result<(), SimdError> {
        if src.len() < indices.len() {
            return Err(SimdError::InsufficientInputLength);
        }
        let len = self.len();
        // Validate every index first: no element is written unless all are in
        // range, which is what makes the `Err` path non-destructive. Compare in
        // `usize` after the sign check: an `i32` bound (`len as i32`) would
        // truncate for views of 2^31 elements or more and misclassify valid
        // indices.
        for &idx in indices {
            if idx < 0 || idx as usize >= len {
                return Err(SimdError::IndexOutOfBounds);
            }
        }

        let indices_len = indices.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (indices_len / lane_count) * lane_count;
        let src_ptr = src.as_ptr();
        let base_ptr = self.as_slice_mut().as_mut_ptr();

        // SAFETY: every index was validated in `[0, len)`, so each scattered
        // store targets a live element of the view. `src` holds at least
        // `indices_len` elements (checked above), so each full-width load reads
        // initialized memory. The tail stages its live lanes in fully
        // initialized local buffers and masks the dead ones off, so neither the
        // load nor the store touches memory outside the live range.
        unsafe {
            for i in (0..simd_len).step_by(lane_count) {
                let idx_vec =
                    crate::sparse::spmv::build_index_vector::<T, Arch>(&indices[i..i + lane_count]);
                let v = Arch::load_unaligned(src_ptr.add(i));
                Arch::scatter(base_ptr, idx_vec, v);
            }

            let rem = indices_len - simd_len;
            if rem > 0 {
                // Dead index lanes are never dereferenced by the masked scatter,
                // but they are still materialized into the index vector, so they
                // are seeded with a valid in-range 0 rather than left undefined.
                let mut idx_buf = [0_i32; MAX_SIMD_LANES];
                idx_buf[..rem].copy_from_slice(&indices[simd_len..indices_len]);

                let mut lane_buf = [T::ZERO; MAX_SIMD_LANES];
                lane_buf[..rem].copy_from_slice(&src[simd_len..indices_len]);

                let idx_vec =
                    crate::sparse::spmv::build_index_vector::<T, Arch>(&idx_buf[..lane_count]);
                let v = Arch::load_unaligned(lane_buf.as_ptr());
                let mask = Arch::leading_k_mask(rem);
                Arch::scatter_masked(base_ptr, idx_vec, mask, v);
            }
        }
        Ok(())
    }
}

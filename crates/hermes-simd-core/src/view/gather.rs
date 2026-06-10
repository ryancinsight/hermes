use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Indirectly load (gather) elements from this view using indices, writing them contiguous to `out`.
    ///
    /// # Errors
    /// Returns `SimdError::InsufficientOutputLength` if `out.len() < indices.len()`.
    /// Returns `SimdError::IndexOutOfBounds` if any index in `indices` is out of bounds (negative or >= view len).
    #[inline(always)]
    pub fn gather(&self, indices: &[i32], out: &mut [T]) -> Result<(), SimdError> {
        let len = self.len();
        if out.len() < indices.len() {
            return Err(SimdError::InsufficientOutputLength);
        }
        let max_idx = len as i32;
        // Validate all indices first to ensure transactional correctness
        for &idx in indices {
            if idx < 0 || idx >= max_idx {
                return Err(SimdError::IndexOutOfBounds);
            }
        }

        let indices_len = indices.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (indices_len / lane_count) * lane_count;
        let base_ptr = self.as_slice().as_ptr();
        let out_ptr = out.as_mut_ptr();

        unsafe {
            for i in (0..simd_len).step_by(lane_count) {
                let idx_slice = &indices[i..i + lane_count];
                let idx_vec = crate::sparse::spmv::build_index_vector::<T, Arch>(idx_slice);
                let v = Arch::gather(base_ptr, idx_vec);
                Arch::store_unaligned(out_ptr.add(i), v);
            }
        }

        // Scalar tail loop
        let slice = self.as_slice();
        for i in simd_len..indices_len {
            out[i] = slice[indices[i] as usize];
        }

        Ok(())
    }
}

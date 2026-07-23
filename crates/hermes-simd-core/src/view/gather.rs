use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};
use core::mem::MaybeUninit;

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
        // SAFETY: an initialized `[T]` is a valid `[MaybeUninit<T>]` — the cast
        // only widens the permitted state, never narrows it — and `T: Scalar` is
        // `Copy`, so overwriting slots with `MaybeUninit::write` drops nothing.
        let out_uninit = unsafe {
            core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut MaybeUninit<T>, out.len())
        };
        self.gather_into_uninit(indices, out_uninit)?;
        Ok(())
    }

    /// Gather into a possibly-uninitialized buffer, returning the initialized prefix.
    ///
    /// This is the single gather implementation; [`gather`](Self::gather) is the
    /// initialized-slice wrapper over it. On `Ok` exactly the first
    /// `indices.len()` elements of `out` are initialized (and returned as an
    /// initialized slice); on `Err` nothing is written, since indices are fully
    /// validated before any store. Filling an `AlignedVec`'s
    /// [`spare_capacity_mut`](crate::vec::AlignedVec::spare_capacity_mut) through
    /// this method and then advancing its length avoids the zero-fill that an
    /// initialized-slice API would otherwise force.
    ///
    /// # Errors
    /// Returns `SimdError::InsufficientOutputLength` if `out.len() < indices.len()`.
    /// Returns `SimdError::IndexOutOfBounds` if any index is negative or `>=` the view length.
    #[inline]
    pub fn gather_into_uninit<'o>(
        &self,
        indices: &[i32],
        out: &'o mut [MaybeUninit<T>],
    ) -> Result<&'o mut [T], SimdError> {
        let len = self.len();
        if out.len() < indices.len() {
            return Err(SimdError::InsufficientOutputLength);
        }
        let max_idx = len as i32;
        // Validate all indices first: no element is written unless every index
        // is in range, so the `Err` path leaves `out` untouched.
        for &idx in indices {
            if idx < 0 || idx >= max_idx {
                return Err(SimdError::IndexOutOfBounds);
            }
        }

        let indices_len = indices.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (indices_len / lane_count) * lane_count;
        let base_ptr = self.as_slice().as_ptr();
        let slice = self.as_slice();
        // Derive the output pointer once and write exclusively through it: mixing
        // it with `out[i]` slice reborrows would invalidate its provenance under
        // Stacked Borrows.
        let out_ptr = out.as_mut_ptr().cast::<T>();

        // SAFETY: every gathered index was validated in range, so each load
        // reads a live element of the view. `out` holds at least `indices_len`
        // slots (checked above); `MaybeUninit<T>` shares `T`'s layout, so the
        // vector and scalar stores below fill `[0, indices_len)` through
        // `out_ptr` without reading any slot first. No slot is written twice.
        unsafe {
            for i in (0..simd_len).step_by(lane_count) {
                let idx_slice = &indices[i..i + lane_count];
                let idx_vec = crate::sparse::spmv::build_index_vector::<T, Arch>(idx_slice);
                let v = Arch::gather(base_ptr, idx_vec);
                Arch::store_unaligned(out_ptr.add(i), v);
            }
            for i in simd_len..indices_len {
                core::ptr::write(out_ptr.add(i), slice[indices[i] as usize]);
            }
            // Every element of `[0, indices_len)` is now initialized.
            Ok(core::slice::from_raw_parts_mut(out_ptr, indices_len))
        }
    }
}

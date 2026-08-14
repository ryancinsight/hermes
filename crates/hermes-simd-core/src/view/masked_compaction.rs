use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::{SimdKernel, MAX_SIMD_LANES};
use crate::mask::BitMask;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Compress active masked elements contiguously into `out`.
    ///
    /// # Errors
    /// Returns [`SimdError::InsufficientOutputLength`] when `out` is shorter
    /// than the input view.
    ///
    /// # Panics
    /// Panics if the mask lane count `N` does not equal the architecture lane
    /// count.
    #[inline(always)]
    pub fn compress<const N: usize>(
        &self,
        mask: &BitMask<N>,
        out: &mut [T],
    ) -> Result<usize, SimdError> {
        assert_eq!(
            N,
            Arch::LANE_COUNT,
            "mask lane count must match SIMD lane count"
        );
        super::check_output_length(self.len(), out.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let mut ptr = self.as_slice().as_ptr();
        let mut ptr_out = out.as_mut_ptr();
        let mut total_written = 0;

        unsafe {
            let load = |p| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };

            let native_mask = mask.to_native_mask::<T, Arch>();
            // The same mask applies to every chunk, so its popcount is loop-invariant.
            let pop = mask.popcount() as usize;

            // Scratch for one compacted vector, hoisted out of the loop. The
            // store writes `lane_count` lanes and the copy reads only `pop ≤
            // lane_count`, so no lane is read before the store initializes it —
            // `MaybeUninit` avoids re-zeroing a full `MAX_SIMD_LANES` buffer every
            // chunk. The compile-time `LANE_BOUND_CHECK` guarantees that the
            // store stays in bounds.
            let () = <Arch as SimdKernel<T>>::LANE_BOUND_CHECK;
            let mut temp = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];

            for _ in 0..(simd_len / lane_count) {
                let v = load(ptr);
                let compressed = Arch::compress(v, native_mask);

                Arch::store_unaligned(temp.as_mut_ptr().cast::<T>(), compressed);
                core::ptr::copy_nonoverlapping(temp.as_ptr().cast::<T>(), ptr_out, pop);

                ptr = ptr.add(lane_count);
                ptr_out = ptr_out.add(pop);
                total_written += pop;
            }
        }

        let s_slice = self.as_slice();
        for i in simd_len..len {
            let lane_idx = i - simd_len;
            if mask.is_lane_active(lane_idx) {
                out[total_written] = s_slice[i];
                total_written += 1;
            }
        }

        Ok(total_written)
    }

    /// Expand active masked elements from `self` into `out`, filling inactive positions.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when `self` does not contain all
    /// active source elements, or [`SimdError::InsufficientOutputLength`] when
    /// `fill` is shorter than `out`.
    ///
    /// # Panics
    /// Panics if the mask lane count `N` does not equal the architecture lane
    /// count.
    #[inline(always)]
    pub fn expand<ORef, const N: usize>(
        &self,
        mask: &BitMask<N>,
        fill: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        out: &mut [T],
    ) -> Result<(), SimdError>
    where
        ORef: 'a,
    {
        assert_eq!(
            N,
            Arch::LANE_COUNT,
            "mask lane count must match SIMD lane count"
        );
        let out_len = out.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (out_len / lane_count) * lane_count;
        let pop = mask.popcount() as usize;

        // Calculate the number of source elements required from `self`.
        let num_simd_chunks = simd_len / lane_count;
        let mut required_len = num_simd_chunks * pop;
        for i in simd_len..out_len {
            let lane_idx = i - simd_len;
            if mask.is_lane_active(lane_idx) {
                required_len += 1;
            }
        }

        if self.len() < required_len {
            return Err(SimdError::LengthMismatch);
        }
        if fill.len() < out_len {
            return Err(SimdError::InsufficientOutputLength);
        }

        let src_len = self.len();
        let mut ptr_src = self.as_slice().as_ptr();
        let mut ptr_fill = fill.as_slice().as_ptr();
        let mut ptr_out = out.as_mut_ptr();

        unsafe {
            let load = |p| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };

            let store = |p, val| {
                let is_out_aligned = crate::align::is_aligned_for_arch::<Arch, Align>()
                    && (p as usize) % Align::ALIGN_BYTES == 0;

                if is_out_aligned {
                    Arch::store_aligned(p, val);
                } else {
                    Arch::store_unaligned(p, val);
                }
            };

            let load_safe = |p: *const T, remaining: usize| {
                if remaining >= lane_count {
                    load(p)
                } else {
                    let mut temp = [T::ZERO; 64];
                    core::ptr::copy_nonoverlapping(p, temp.as_mut_ptr(), remaining);
                    load(temp.as_ptr())
                }
            };

            let native_mask = mask.to_native_mask::<T, Arch>();

            for chunk_idx in 0..num_simd_chunks {
                let offset = chunk_idx * pop;
                let remaining = src_len - offset;
                let src_v = load_safe(ptr_src, remaining);
                let fill_v = load(ptr_fill);
                let res = Arch::expand(src_v, native_mask, fill_v);
                store(ptr_out, res);

                ptr_src = ptr_src.add(pop);
                ptr_fill = ptr_fill.add(lane_count);
                ptr_out = ptr_out.add(lane_count);
            }
        }

        let s_slice = self.as_slice();
        let f_slice = fill.as_slice();
        let mut src_idx = num_simd_chunks * pop;
        for i in simd_len..out_len {
            let lane_idx = i - simd_len;
            if mask.is_lane_active(lane_idx) {
                out[i] = s_slice[src_idx];
                src_idx += 1;
            } else {
                out[i] = f_slice[i];
            }
        }

        Ok(())
    }
}

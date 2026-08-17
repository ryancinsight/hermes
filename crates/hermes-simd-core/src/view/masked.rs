use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::{SimdKernel, SimdStorage, MAX_SIMD_LANES};
use crate::mask::BitMask;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Add elementwise where mask is active, writing to `out`. Inactive lanes copy from `self`.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when the operand lengths differ,
    /// or [`SimdError::InsufficientOutputLength`] when `out` is too short.
    ///
    /// # Panics
    ///
    /// Panics if the mask lane count `N` does not equal the architecture lane
    /// count.
    #[inline(always)]
    pub fn masked_add<ORef, const N: usize>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        mask: &BitMask<N>,
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
        super::check_lengths_equal(self.len(), other.len())?;
        super::check_output_length(self.len(), out.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let mut ptr1 = self.as_slice().as_ptr();
        let mut ptr2 = other.as_slice().as_ptr();
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

            let native_mask = mask.to_native_mask::<T, Arch>();

            for _ in 0..(simd_len / lane_count) {
                let v1 = load(ptr1);
                let v2 = load(ptr2);
                let res = Arch::masked_add(v1, v2, native_mask, v1);
                store(ptr_out, res);

                ptr1 = ptr1.add(lane_count);
                ptr2 = ptr2.add(lane_count);
                ptr_out = ptr_out.add(lane_count);
            }
        }

        let tail = len - simd_len;
        if tail != 0 {
            // Blend-based backends perform a full-width vector load even when
            // only part of the final logical vector is live. Copying the live
            // prefix into initialized provider-local buffers makes that load
            // bounded by the local allocation; the combined mask prevents the
            // inactive tail lanes from reaching the output.
            const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
            let mut left = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            let mut right = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            let mut result = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            for i in 0..tail {
                left[i].write(self.as_slice()[simd_len + i]);
                right[i].write(other.as_slice()[simd_len + i]);
            }
            for i in tail..lane_count {
                left[i].write(T::ZERO);
                right[i].write(T::ZERO);
                result[i].write(T::ZERO);
            }
            unsafe {
                let tail_mask = *mask & BitMask::<N>::leading_k(tail);
                let value = Arch::masked_add(
                    Arch::load_unaligned(left.as_ptr().cast::<T>()),
                    Arch::load_unaligned(right.as_ptr().cast::<T>()),
                    tail_mask.to_native_mask::<T, Arch>(),
                    Arch::load_unaligned(left.as_ptr().cast::<T>()),
                );
                Arch::store_unaligned(result.as_mut_ptr().cast::<T>(), value);
            }
            out[simd_len..].copy_from_slice(unsafe {
                core::slice::from_raw_parts(result.as_ptr().cast::<T>(), tail)
            });
        }

        Ok(())
    }

    /// Multiply elementwise where mask is active, writing to `out`. Inactive lanes copy from `self`.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when the operand lengths differ,
    /// or [`SimdError::InsufficientOutputLength`] when `out` is too short.
    ///
    /// # Panics
    ///
    /// Panics if the mask lane count `N` does not equal the architecture lane
    /// count.
    #[inline(always)]
    pub fn masked_mul<ORef, const N: usize>(
        &self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        mask: &BitMask<N>,
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
        super::check_lengths_equal(self.len(), other.len())?;
        super::check_output_length(self.len(), out.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let mut ptr1 = self.as_slice().as_ptr();
        let mut ptr2 = other.as_slice().as_ptr();
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

            let native_mask = mask.to_native_mask::<T, Arch>();

            for _ in 0..(simd_len / lane_count) {
                let v1 = load(ptr1);
                let v2 = load(ptr2);
                let res = Arch::masked_mul(v1, v2, native_mask, v1);
                store(ptr_out, res);

                ptr1 = ptr1.add(lane_count);
                ptr2 = ptr2.add(lane_count);
                ptr_out = ptr_out.add(lane_count);
            }
        }

        let tail = len - simd_len;
        if tail != 0 {
            const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
            let mut left = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            let mut right = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            let mut result = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            for i in 0..tail {
                left[i].write(self.as_slice()[simd_len + i]);
                right[i].write(other.as_slice()[simd_len + i]);
            }
            for i in tail..lane_count {
                left[i].write(T::ZERO);
                right[i].write(T::ZERO);
                result[i].write(T::ZERO);
            }
            unsafe {
                let tail_mask = *mask & BitMask::<N>::leading_k(tail);
                let value = Arch::masked_mul(
                    Arch::load_unaligned(left.as_ptr().cast::<T>()),
                    Arch::load_unaligned(right.as_ptr().cast::<T>()),
                    tail_mask.to_native_mask::<T, Arch>(),
                    Arch::load_unaligned(left.as_ptr().cast::<T>()),
                );
                Arch::store_unaligned(result.as_mut_ptr().cast::<T>(), value);
            }
            out[simd_len..].copy_from_slice(unsafe {
                core::slice::from_raw_parts(result.as_ptr().cast::<T>(), tail)
            });
        }

        Ok(())
    }

    /// Fused multiply-add where mask is active: `(self * b) + c`, writing to `out`. Inactive lanes copy from `c`.
    ///
    /// # Errors
    /// Returns [`SimdError::LengthMismatch`] when operand lengths differ, or
    /// [`SimdError::InsufficientOutputLength`] when `out` is too short.
    ///
    /// # Panics
    ///
    /// Panics if the mask lane count `N` does not equal the architecture lane
    /// count.
    #[inline(always)]
    pub fn masked_fmadd<ORef1, ORef2, const N: usize>(
        &self,
        b: &SimdView<'_, T, Arch, Align, Mode, ORef1>,
        c: &SimdView<'_, T, Arch, Align, Mode, ORef2>,
        mask: &BitMask<N>,
        out: &mut [T],
    ) -> Result<(), SimdError>
    where
        ORef1: 'a,
        ORef2: 'a,
    {
        assert_eq!(
            N,
            Arch::LANE_COUNT,
            "mask lane count must match SIMD lane count"
        );
        super::check_lengths_equal(self.len(), b.len())?;
        super::check_lengths_equal(self.len(), c.len())?;
        super::check_output_length(self.len(), out.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let mut ptr_a = self.as_slice().as_ptr();
        let mut ptr_b = b.as_slice().as_ptr();
        let mut ptr_c = c.as_slice().as_ptr();
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

            let native_mask = mask.to_native_mask::<T, Arch>();

            for _ in 0..(simd_len / lane_count) {
                let va = load(ptr_a);
                let vb = load(ptr_b);
                let vc = load(ptr_c);
                let res = Arch::masked_fmadd(va, vb, vc, native_mask);
                store(ptr_out, res);

                ptr_a = ptr_a.add(lane_count);
                ptr_b = ptr_b.add(lane_count);
                ptr_c = ptr_c.add(lane_count);
                ptr_out = ptr_out.add(lane_count);
            }
        }

        let tail = len - simd_len;
        if tail != 0 {
            const { <Arch as SimdStorage<T>>::LANE_BOUND_CHECK };
            let mut left = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            let mut right = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            let mut addend = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            let mut result = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
            for i in 0..tail {
                left[i].write(self.as_slice()[simd_len + i]);
                right[i].write(b.as_slice()[simd_len + i]);
                addend[i].write(c.as_slice()[simd_len + i]);
            }
            for i in tail..lane_count {
                left[i].write(T::ZERO);
                right[i].write(T::ZERO);
                addend[i].write(T::ZERO);
                result[i].write(T::ZERO);
            }
            unsafe {
                let tail_mask = *mask & BitMask::<N>::leading_k(tail);
                let value = Arch::masked_fmadd(
                    Arch::load_unaligned(left.as_ptr().cast::<T>()),
                    Arch::load_unaligned(right.as_ptr().cast::<T>()),
                    Arch::load_unaligned(addend.as_ptr().cast::<T>()),
                    tail_mask.to_native_mask::<T, Arch>(),
                );
                Arch::store_unaligned(result.as_mut_ptr().cast::<T>(), value);
            }
            out[simd_len..].copy_from_slice(unsafe {
                core::slice::from_raw_parts(result.as_ptr().cast::<T>(), tail)
            });
        }

        Ok(())
    }
}

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
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
            // SAFETY: both inputs and the validated output contain the exact
            // `tail` suffix. `live_mask` selects that whole suffix for memory;
            // `operation_mask` selects the subset that participates in add.
            unsafe {
                let live_mask = BitMask::<N>::leading_k(tail).to_native_mask::<T, Arch>();
                let operation_mask =
                    (*mask & BitMask::<N>::leading_k(tail)).to_native_mask::<T, Arch>();
                let left = Arch::masked_load_partial(
                    self.as_slice().as_ptr().add(simd_len),
                    tail,
                    live_mask,
                    Arch::zero(),
                );
                let right = Arch::masked_load_partial(
                    other.as_slice().as_ptr().add(simd_len),
                    tail,
                    live_mask,
                    Arch::zero(),
                );
                let value = Arch::masked_add(left, right, operation_mask, left);
                Arch::masked_store_partial(out.as_mut_ptr().add(simd_len), tail, live_mask, value);
            }
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
            // SAFETY: both inputs and the validated output contain the exact
            // `tail` suffix. `live_mask` selects that whole suffix for memory;
            // `operation_mask` selects the subset that participates in multiply.
            unsafe {
                let live_mask = BitMask::<N>::leading_k(tail).to_native_mask::<T, Arch>();
                let operation_mask =
                    (*mask & BitMask::<N>::leading_k(tail)).to_native_mask::<T, Arch>();
                let left = Arch::masked_load_partial(
                    self.as_slice().as_ptr().add(simd_len),
                    tail,
                    live_mask,
                    Arch::zero(),
                );
                let right = Arch::masked_load_partial(
                    other.as_slice().as_ptr().add(simd_len),
                    tail,
                    live_mask,
                    Arch::zero(),
                );
                let value = Arch::masked_mul(left, right, operation_mask, left);
                Arch::masked_store_partial(out.as_mut_ptr().add(simd_len), tail, live_mask, value);
            }
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
            // SAFETY: all inputs and the validated output contain the exact
            // `tail` suffix. `live_mask` selects that whole suffix for memory;
            // `operation_mask` selects the subset that participates in FMA.
            unsafe {
                let live_mask = BitMask::<N>::leading_k(tail).to_native_mask::<T, Arch>();
                let operation_mask =
                    (*mask & BitMask::<N>::leading_k(tail)).to_native_mask::<T, Arch>();
                let left = Arch::masked_load_partial(
                    self.as_slice().as_ptr().add(simd_len),
                    tail,
                    live_mask,
                    Arch::zero(),
                );
                let right = Arch::masked_load_partial(
                    b.as_slice().as_ptr().add(simd_len),
                    tail,
                    live_mask,
                    Arch::zero(),
                );
                let addend = Arch::masked_load_partial(
                    c.as_slice().as_ptr().add(simd_len),
                    tail,
                    live_mask,
                    Arch::zero(),
                );
                let value = Arch::masked_fmadd(left, right, addend, operation_mask);
                Arch::masked_store_partial(out.as_mut_ptr().add(simd_len), tail, live_mask, value);
            }
        }

        Ok(())
    }
}

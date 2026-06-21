use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::ops::UnaryOp;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Apply a `UnaryOp<T>` to every element, writing results to `out`.
    ///
    /// SIMD-vectorized: processes `floor(len / LANE_COUNT) * LANE_COUNT` elements via
    /// the hardware path, then applies `Op::apply_scalar` to the tail.
    ///
    /// # Errors
    /// Returns `SimdError::InsufficientOutputLength` if `out.len() < self.len()`.
    #[inline(always)]
    pub fn map_unary<Op: UnaryOp<T>>(&self, op: Op, out: &mut [T]) -> Result<(), SimdError> {
        let data = self.as_slice();
        let len = data.len();
        if out.len() < len {
            return Err(SimdError::InsufficientOutputLength);
        }
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr_in = data.as_ptr();
        let ptr_out = out.as_mut_ptr();

        unsafe {
            let load = |p: *const T| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                // Output alignment matches input when writing into the same AlignedVec;
                // for cross-buffer writes, the output may differ — use Align to govern both.
                if crate::align::is_aligned_for_arch::<Arch, Align>()
                    && (p as usize) % Align::ALIGN_BYTES == 0
                {
                    Arch::store_aligned(p, v)
                } else {
                    Arch::store_unaligned(p, v)
                }
            };
            for i in (0..simd_len).step_by(lane_count) {
                let v = load(ptr_in.add(i));
                let r = op.apply::<Arch>(v);
                store(ptr_out.add(i), r);
            }
        }

        for i in simd_len..len {
            out[i] = op.apply_scalar(data[i]);
        }

        Ok(())
    }
}

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
where
    T: Scalar,
{
    /// Apply a `UnaryOp<T>` in-place: `self[i] = op(self[i])`.
    ///
    /// SIMD-vectorized with load-modify-store per lane group.
    /// Both load and store respect `Align::IS_ALIGNED`.
    #[inline(always)]
    pub fn map_unary_in_place<Op: UnaryOp<T>>(&mut self, op: Op) {
        let slice = self.as_slice_mut();
        let len = slice.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr = slice.as_mut_ptr();

        unsafe {
            let load = |p: *mut T| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if crate::align::is_aligned_for_arch::<Arch, Align>() {
                    Arch::store_aligned(p, v)
                } else {
                    Arch::store_unaligned(p, v)
                }
            };
            for i in (0..simd_len).step_by(lane_count) {
                let v = load(ptr.add(i));
                let r = op.apply::<Arch>(v);
                store(ptr.add(i), r);
            }
        }

        for i in simd_len..len {
            slice[i] = op.apply_scalar(slice[i]);
        }
    }
}

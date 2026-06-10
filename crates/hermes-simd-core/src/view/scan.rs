use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::ops::{ScanMode, ScanOp};
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Perform a prefix scan (inclusive or exclusive) of the view using the specified operation,
    /// writing results to `out`.
    ///
    /// # Errors
    /// Returns `SimdError::InsufficientOutputLength` if `out.len() < self.len()`.
    #[inline(always)]
    pub fn prefix_scan<Op, SMode>(
        &self,
        out: &mut [T],
        _op: Op,
        _mode: SMode,
    ) -> Result<(), SimdError>
    where
        Op: ScanOp<T>,
        SMode: ScanMode,
    {
        let len = self.len();
        if out.len() < len {
            return Err(SimdError::InsufficientOutputLength);
        }
        if len == 0 {
            return Ok(());
        }

        let src = self.as_slice();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let ptr_in = src.as_ptr();
        let ptr_out = out.as_mut_ptr();

        let mut carry = Op::identity();

        unsafe {
            let load = |p: *const T| {
                if Align::IS_ALIGNED {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if Align::IS_ALIGNED {
                    Arch::store_aligned(p, v)
                } else {
                    Arch::store_unaligned(p, v)
                }
            };

            for i in (0..simd_len).step_by(lane_count) {
                let v = load(ptr_in.add(i));
                let (r, next_carry) = Arch::scan_vector::<Op, SMode>(v, carry);
                store(ptr_out.add(i), r);
                carry = next_carry;
            }
        }

        if SMode::IS_INCLUSIVE {
            for i in simd_len..len {
                carry = Op::combine(carry, src[i]);
                out[i] = carry;
            }
        } else {
            for i in simd_len..len {
                let temp = src[i];
                out[i] = carry;
                carry = Op::combine(carry, temp);
            }
        }

        Ok(())
    }
}

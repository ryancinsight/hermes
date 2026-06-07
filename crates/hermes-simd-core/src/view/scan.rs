use crate::arch::SimdArch;
use crate::align::Alignment;
use crate::kernel::SimdKernel;
use crate::execution::ExecutionMode;
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};
use crate::ops::{ScanOp, ScanMode};

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
    pub fn prefix_scan<Op, SMode>(&self, out: &mut [T], _op: Op, _mode: SMode) -> Result<(), SimdError>
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
        let mut acc = Op::identity();

        if SMode::IS_INCLUSIVE {
            for i in 0..len {
                acc = Op::combine(acc, src[i]);
                out[i] = acc;
            }
        } else {
            for i in 0..len {
                let temp = src[i];
                out[i] = acc;
                acc = Op::combine(acc, temp);
            }
        }

        Ok(())
    }
}

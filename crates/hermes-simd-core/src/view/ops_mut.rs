//! Mutable elementwise SIMD operations on views backed by `&'a mut [T]`.
//!
//! All methods on `SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>` that modify the
//! underlying slice in-place live here, keeping `ops.rs` (read-only) and `ops_mut.rs`
//! (write) as two separate bounded-context files.
//!
//! # DRY Note
//!
//! Concrete `add_assign` and `mul_assign` delegate to the generic `transform_in_place`
//! kernel. This yields a single authoritative SIMD loop body that is monomorphized per
//! `(T, Arch, Align, Op)` — not duplicated for each binary operation.

use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::execution::ExecutionMode;
use crate::kernel::SimdKernel;
use crate::ops::{Add, ElementOp, Mul};
use crate::scalar::Scalar;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode>
    SimdView<'a, T, Arch, Align, Mode, &'a mut [T]>
where
    T: Scalar,
{
    /// Apply an elementwise `ElementOp<T>` on `self` and `other` in-place:
    /// `self[i] = op(self[i], other[i])`.
    ///
    /// This is the canonical generic in-place kernel. `add_assign` and `mul_assign`
    /// delegate here. The operation ZST is erased at every monomorphization site.
    ///
    /// # Zero-Cost Contract
    ///
    /// `Op` is a ZST (`size_of::<Op>() == 0` for `Add`, `Mul`, etc.). The compiler
    /// inlines the `op.apply::<Arch>` call and the alignment branch is eliminated by DCE
    /// at each `(T, Arch, Align, Op)` monomorphization.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] if operand lengths differ.
    #[inline(always)]
    pub fn transform_in_place<ORef, Op>(
        &mut self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
        op: Op,
    ) -> Result<(), SimdError>
    where
        ORef: 'a,
        Op: ElementOp<T>,
    {
        super::check_lengths_equal(self.len(), other.len())?;

        let len = self.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        let ptr_self = self.as_slice_mut().as_mut_ptr();
        let ptr_other = other.as_slice().as_ptr();

        unsafe {
            // Alignment-dependent load/store closures. The `Align::IS_ALIGNED` branch is
            // a compile-time constant: DCE removes the unused arm at every monomorphization.
            let load_self = |p: *const T| {
                if Align::IS_ALIGNED {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let load_other = |p: *const T| {
                if Align::IS_ALIGNED {
                    Arch::load_aligned(p)
                } else {
                    Arch::load_unaligned(p)
                }
            };
            let store = |p: *mut T, v: Arch::Vector| {
                if Align::IS_ALIGNED {
                    Arch::store_aligned(p, v);
                } else {
                    Arch::store_unaligned(p, v);
                }
            };

            for i in (0..simd_len).step_by(lane_count) {
                let va = load_self(ptr_self.add(i) as *const T);
                let vb = load_other(ptr_other.add(i));
                let vr = op.apply::<Arch>(va, vb);
                store(ptr_self.add(i), vr);
            }
        }

        // Scalar tail — elements that do not fill a complete SIMD vector.
        let s_mut_slice = self.as_slice_mut();
        let o_slice = other.as_slice();
        for i in simd_len..len {
            s_mut_slice[i] = op.apply_scalar(s_mut_slice[i], o_slice[i]);
        }

        Ok(())
    }

    /// Add another view elementwise to this mutable view in-place.
    ///
    /// Delegates to [`Self::transform_in_place`] with the [`Add`] strategy.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] if operand lengths do not match.
    #[inline(always)]
    pub fn add_assign<ORef>(
        &mut self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
    ) -> Result<(), SimdError>
    where
        ORef: 'a,
    {
        self.transform_in_place(other, Add)
    }

    /// Multiply another view elementwise with this mutable view in-place.
    ///
    /// Delegates to [`Self::transform_in_place`] with the [`Mul`] strategy.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] if operand lengths do not match.
    #[inline(always)]
    pub fn mul_assign<ORef>(
        &mut self,
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
    ) -> Result<(), SimdError>
    where
        ORef: 'a,
    {
        self.transform_in_place(other, Mul)
    }
}

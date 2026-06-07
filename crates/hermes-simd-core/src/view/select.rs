//! Conditional SIMD blend/select operations on [`SimdView`].
//!
//! `select` performs a lane-wise conditional merge: for each index `i`,
//! the output is `self[i]` if `mask[i]` is true, else `other[i]`.
//!
//! # Architecture mapping
//!
//! | Method | Instruction family |
//! |---|---|
//! | `blend` (float mask) | AVX-512 `_mm512_mask_blend_ps`, AVX2 `_mm256_blendv_ps`, NEON `vbslq_f32` |
//! | Scalar fallback | `if mask[i] { self[i] } else { other[i] }` |
//!
//! # Zero-Cost Contract
//!
//! Selection is monomorphized per `(T, Arch, Align)`. The `Align` ZST governs
//! which load instruction is emitted; `Arch` is a ZST erased after codegen.

use crate::arch::SimdArch;
use crate::align::Alignment;
use crate::kernel::SimdKernel;
use crate::execution::ExecutionMode;
use crate::scalar::Scalar;
use crate::vec::AlignedVec;
use crate::view::{SimdError, SimdView};

impl<'a, T: 'a, Arch: SimdArch + SimdKernel<T>, Align: Alignment, Mode: ExecutionMode, Ref: 'a>
    SimdView<'a, T, Arch, Align, Mode, Ref>
where
    T: Scalar,
{
    /// Lane-wise conditional select: `out[i] = if mask[i] { self[i] } else { other[i] }`.
    ///
    /// Allocates one `AlignedVec<T, Align>` of `self.len()` elements.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::LengthMismatch`] if `other.len() != self.len()`, or
    /// [`SimdError::InsufficientOutputLength`] if `mask.len() < self.len()`.
    ///
    /// # Implementation
    ///
    /// The scalar fallback loop is the authoritative path. Hardware backends override
    /// `SimdKernel::blend` with the matching intrinsic; the compiler selects the
    /// appropriate specialization at monomorphization.
    pub fn select<ORef>(
        &self,
        mask: &[bool],
        other: &SimdView<'_, T, Arch, Align, Mode, ORef>,
    ) -> Result<AlignedVec<T, Align>, SimdError>
    where
        ORef: 'a,
    {
        super::check_lengths_equal(self.len(), other.len())?;
        if mask.len() < self.len() {
            return Err(SimdError::InsufficientOutputLength);
        }

        let a = self.as_slice();
        let b = other.as_slice();
        let len = a.len();

        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        // SAFETY: every element is written below.
        unsafe { out.set_len(len); }
        let out_slice = out.as_mut_slice();

        let lane_count = Arch::LANE_COUNT;
        let mut i = 0;
        unsafe {
            while i + lane_count <= len {
                let m = Arch::mask_from_bools(&mask[i..i + lane_count]);
                let vb = Arch::load_unaligned(b.as_ptr().add(i));
                let v_res = Arch::masked_load_unaligned(a.as_ptr().add(i), m, vb);
                Arch::store_unaligned(out_slice.as_mut_ptr().add(i), v_res);
                i += lane_count;
            }
        }
        for j in i..len {
            out_slice[j] = if mask[j] { a[j] } else { b[j] };
        }

        Ok(out)
    }

    /// Lane-wise conditional negate: `out[i] = if mask[i] { -self[i] } else { self[i] }`.
    ///
    /// Allocates one `AlignedVec<T, Align>`. Mask length must be `≥ self.len()`.
    ///
    /// # Errors
    ///
    /// Returns [`SimdError::InsufficientOutputLength`] if `mask.len() < self.len()`.
    pub fn masked_negate(&self, mask: &[bool]) -> Result<AlignedVec<T, Align>, SimdError>
    where
        T: core::ops::Neg<Output = T>,
    {
        if mask.len() < self.len() {
            return Err(SimdError::InsufficientOutputLength);
        }

        let data = self.as_slice();
        let len = data.len();
        let mut out: AlignedVec<T, Align> = AlignedVec::with_capacity(len);
        // SAFETY: every element is written in the loop below.
        unsafe { out.set_len(len); }
        let out_slice = out.as_mut_slice();

        // Scalar loop: correct for all T including two's-complement integers.
        // The compiler auto-vectorizes this on float types where negation is a
        // single VXORPS/VXORPD; integer SIMD negation requires architecture-specific
        // override of SimdKernel::neg which is not universally available.
        for i in 0..len {
            out_slice[i] = if mask[i] { -data[i] } else { data[i] };
        }

        Ok(out)
    }
}

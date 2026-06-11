//! Stable inline-assembly instruction forms for x86 kernels.
//!
//! This module is intentionally narrow: it exposes instruction-level helpers
//! only where stable Rust intrinsics are unavailable or are not the selected
//! code-generation contract. Portable APIs remain the trait surfaces above it.

/// Emits `vpdpbssd dst, src1, src2`.
///
/// Instruction contract: `dst += dot4(src1, src2)` with the hardware semantics
/// of AVX-512 VNNI signed-byte dot product accumulation. This stays as a macro
/// so the asm expands inside the caller's `#[target_feature]` context without a
/// function-call boundary in the tile inner loop.
///
/// # Safety
/// The expansion site must be gated for `avx512f`, `avx512vnni`, and
/// `avx512vl`.
#[cfg(not(miri))]
macro_rules! vpdpbssd {
    ($dst:expr, $src1:expr, $src2:expr) => {{
        let mut acc = $dst;
        core::arch::asm!(
            "vpdpbssd {dst}, {src1}, {src2}",
            dst = inout(zmm_reg) acc,
            src1 = in(zmm_reg) $src1,
            src2 = in(zmm_reg) $src2,
            options(nostack, preserves_flags, nomem),
        );
        acc
    }};
}

/// Miri has no AVX-512 register or instruction model. If a test reaches this
/// macro under Miri, the test crossed from Rust-side safety into hardware
/// execution and must be redirected to the scalar semantic path instead.
#[cfg(miri)]
macro_rules! vpdpbssd {
    ($dst:expr, $src1:expr, $src2:expr) => {{
        $crate::x86_64::asm_intrinsics::miri_unavailable_vpdpbssd($dst, $src1, $src2)
    }};
}

pub(crate) use vpdpbssd;

#[cfg(miri)]
#[cold]
pub(crate) fn miri_unavailable_vpdpbssd<T>(dst: T, src1: T, src2: T) -> T {
    let _ = (dst, src1, src2);
    panic!("vpdpbssd hardware execution is not available under Miri")
}

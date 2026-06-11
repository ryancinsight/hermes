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

pub(crate) use vpdpbssd;

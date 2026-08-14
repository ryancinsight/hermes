//! AVX-VNNI (256-bit VEX-encoded VNNI) `TileMatrixMultiply` for `x86_64`.
//!
//! Serves client CPUs (Alder Lake and newer, Zen 5) that have `vpdpbusd` on
//! YMM registers but no AVX-512. Sits between the AVX-512 VNNI tile kernel and
//! the scalar fallback in the int8 dispatch ladder.
//!
//! # Signed×signed via the unsigned-signed instruction
//!
//! Base AVX-VNNI provides only `vpdpbusd` (first operand **unsigned** bytes,
//! second signed); the signed-signed `vpdpbssd` is the separate `avxvnniint8`
//! feature this kernel deliberately does not require. Signed A is biased to
//! unsigned per byte, and the bias is subtracted exactly:
//!
//! ```text
//! a_u = a + 128                            (per byte, XOR 0x80)
//! Σ_k a·b = Σ_k a_u·b − 128·Σ_k b
//! ```
//!
//! The correction term `128·Σ_k B[k,j]` is row-independent, so it is
//! accumulated once per column group with `vpdpbusd(bias, splat(0x80), b_vec)`
//! and subtracted from every row's accumulator after the K loop. All
//! accumulation is wrapping i32 (`vpdpbusd` does not saturate; `vpdpbusds`
//! would), and modular arithmetic makes the identity exact under wraparound —
//! the result is **bitwise-equal** to the wrapping scalar reference.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use crate::AvxVnni;
use eunomia::{I32, I8};
use hermes_simd_core::view::TileMatrixMultiply;

#[cfg(target_arch = "x86")]
use core::arch::x86 as arch;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64 as arch;

/// 16×16×64 i8·i8→i32 tile kernel on 256-bit VNNI.
///
/// The 16 output columns split into two 8-column YMM groups; each group runs a
/// 2×8-row register block (8 accumulators + 1 bias accumulator + operands stays
/// within the 16 YMM registers, so the hot loop is spill-free).
///
/// # Safety
/// Caller must ensure the CPU supports `avxvnni` (enforced by the
/// `#[target_feature]` gate plus runtime `is_x86_feature_detected!("avxvnni")`
/// at the dispatch site) and that `a`, `b`, `c` address a full 16×16×64 tile at
/// the given strides: reads `a[i*a_stride + k]` for i<16, k<64, reads
/// `b[k*b_stride + j]` for k<64, j<16, and reads/writes `c[i*c_stride + j]` for
/// i<16, j<16.
#[target_feature(enable = "avxvnni")]
unsafe fn tile_matmul_i8(
    c: *mut i32,
    c_stride: usize,
    a: *const i8,
    a_stride: usize,
    b: *const i8,
    b_stride: usize,
) {
    use arch::{
        __m128i, __m256i, _mm256_dpbusd_avx_epi32, _mm256_loadu_si256, _mm256_set1_epi32,
        _mm256_set1_epi8, _mm256_set_m128i, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_sub_epi32, _mm256_xor_si256, _mm_loadl_epi64, _mm_unpackhi_epi16,
        _mm_unpacklo_epi16, _mm_unpacklo_epi8,
    };

    // Per-byte +128 bias for the unsigned operand of `vpdpbusd` (XOR 0x80).
    let sign_flip = _mm256_set1_epi8(-128);
    // Unsigned 0x80 bytes: accumulates the exact correction 128·Σ_k b.
    let bias_mult = _mm256_set1_epi8(-128);

    // Two 8-column groups × two 8-row blocks.
    for jb in 0..2usize {
        let col = jb * 8;
        for ib in 0..2usize {
            let row = ib * 8;

            // Load the 8 output rows for this block (8 columns each).
            let mut acc = [_mm256_setzero_si256(); 8];
            for (i, slot) in acc.iter_mut().enumerate() {
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "_mm256_loadu_si256 accepts the deliberately unaligned tile row"
                )]
                let c_row = c.add((row + i) * c_stride + col).cast::<__m256i>();
                *slot = _mm256_loadu_si256(c_row);
            }
            let mut bias_acc = _mm256_setzero_si256();

            for k in (0..64).step_by(4) {
                // Pack B[k..k+4, col..col+8] so 32-bit lane j holds the four
                // consecutive-k bytes of column `col + j` — the operand shape
                // `vpdpbusd` contracts over.
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "_mm_loadl_epi64 accepts the deliberately unaligned tile row"
                )]
                let r0 = _mm_loadl_epi64(b.add(k * b_stride + col).cast::<__m128i>());
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "_mm_loadl_epi64 accepts the deliberately unaligned tile row"
                )]
                let r1 = _mm_loadl_epi64(b.add((k + 1) * b_stride + col).cast::<__m128i>());
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "_mm_loadl_epi64 accepts the deliberately unaligned tile row"
                )]
                let r2 = _mm_loadl_epi64(b.add((k + 2) * b_stride + col).cast::<__m128i>());
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "_mm_loadl_epi64 accepts the deliberately unaligned tile row"
                )]
                let r3 = _mm_loadl_epi64(b.add((k + 3) * b_stride + col).cast::<__m128i>());

                let lo01 = _mm_unpacklo_epi8(r0, r1); // r0[j],r1[j] interleaved
                let lo23 = _mm_unpacklo_epi8(r2, r3); // r2[j],r3[j] interleaved
                let cols_0_3 = _mm_unpacklo_epi16(lo01, lo23); // lanes j=0..3
                let cols_4_7 = _mm_unpackhi_epi16(lo01, lo23); // lanes j=4..7
                let b_vec = _mm256_set_m128i(cols_4_7, cols_0_3);

                // Row-independent bias: bias_acc[j] += Σ 128·b[k..k+4, col+j].
                bias_acc = _mm256_dpbusd_avx_epi32(bias_acc, bias_mult, b_vec);

                for (i, slot) in acc.iter_mut().enumerate() {
                    // A[row+i, k..k+4] as one 32-bit load, biased to unsigned.
                    let a_val =
                        core::ptr::read_unaligned(a.add((row + i) * a_stride + k).cast::<i32>());
                    let a_vec = _mm256_xor_si256(_mm256_set1_epi32(a_val), sign_flip);
                    *slot = _mm256_dpbusd_avx_epi32(*slot, a_vec, b_vec);
                }
            }

            for (i, slot) in acc.iter().enumerate() {
                let corrected = _mm256_sub_epi32(*slot, bias_acc);
                #[expect(
                    clippy::cast_ptr_alignment,
                    reason = "_mm256_storeu_si256 accepts the deliberately unaligned tile row"
                )]
                let c_row = c.add((row + i) * c_stride + col).cast::<__m256i>();
                _mm256_storeu_si256(c_row, corrected);
            }
        }
    }
}

impl TileMatrixMultiply<i8, i8, i32, AvxVnni, AvxVnni, 16, 16, 64> for AvxVnni {
    // SAFETY: caller must ensure the target CPU supports `avxvnni` (enforced by
    // the `#[target_feature]` gate on `tile_matmul_i8` plus runtime
    // `is_x86_feature_detected!("avxvnni")` selection at the dispatch site) and
    // that `a`, `b`, `c` address a valid 16x16x64 tile at the given strides.
    #[target_feature(enable = "avxvnni")]
    unsafe fn tile_matmul(
        c: *mut i32,
        c_stride: usize,
        a: *const i8,
        a_stride: usize,
        b: *const i8,
        b_stride: usize,
    ) {
        tile_matmul_i8(c, c_stride, a, a_stride, b, b_stride);
    }
}

impl TileMatrixMultiply<I8, I8, I32, AvxVnni, AvxVnni, 16, 16, 64> for AvxVnni {
    // SAFETY: same ISA and tile preconditions as the `i8` impl; `I8`/`I32` are
    // `#[repr(transparent)]` over `i8`/`i32`, so the pointer casts preserve
    // layout exactly.
    #[target_feature(enable = "avxvnni")]
    unsafe fn tile_matmul(
        c: *mut I32,
        c_stride: usize,
        a: *const I8,
        a_stride: usize,
        b: *const I8,
        b_stride: usize,
    ) {
        tile_matmul_i8(
            c.cast::<i32>(),
            c_stride,
            a.cast::<i8>(),
            a_stride,
            b.cast::<i8>(),
            b_stride,
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::Scalar;

    /// Differential: the VNNI kernel must be bitwise-equal to the wrapping
    /// scalar reference on a full-range signed tile. Inputs cycle the whole
    /// i8 domain (including -128 and sign changes) so a bias-correction error
    /// of even ±1·128 in any lane fails the exact assertion.
    #[test]
    fn avx_vnni_tile_matches_scalar_bitwise() {
        if !std::is_x86_feature_detected!("avxvnni") {
            eprintln!("skipping: host lacks avxvnni");
            return;
        }

        let a: Vec<i8> = (0..16 * 64)
            .map(|i| ((i * 37 + 11) % 256) as u8 as i8)
            .collect();
        let b: Vec<i8> = (0..64 * 16)
            .map(|i| ((i * 73 + 190) % 256) as u8 as i8)
            .collect();
        // Nonzero initial C exercises the accumulate (`+=`) contract.
        let c_init: Vec<i32> = (0..16 * 16).map(|i| i * 1001 - 12345).collect();

        let mut c_vnni = c_init.clone();
        let mut c_ref = c_init;

        unsafe {
            <AvxVnni as TileMatrixMultiply<i8, i8, i32, AvxVnni, AvxVnni, 16, 16, 64>>::tile_matmul(
                c_vnni.as_mut_ptr(),
                16,
                a.as_ptr(),
                64,
                b.as_ptr(),
                16,
            );
            <Scalar as TileMatrixMultiply<i8, i8, i32, Scalar, Scalar, 16, 16, 64>>::tile_matmul(
                c_ref.as_mut_ptr(),
                16,
                a.as_ptr(),
                64,
                b.as_ptr(),
                16,
            );
        }

        assert_eq!(
            c_vnni, c_ref,
            "AVX-VNNI tile diverges from scalar reference"
        );
    }

    /// The bias identity must hold under i32 wraparound: extreme inputs
    /// (-128 everywhere) push `Σ a_u·b` and the correction term apart by the
    /// maximum distance, catching any non-modular shortcut.
    #[test]
    fn avx_vnni_tile_extreme_negative_operands() {
        if !std::is_x86_feature_detected!("avxvnni") {
            eprintln!("skipping: host lacks avxvnni");
            return;
        }

        let a = vec![-128i8; 16 * 64];
        let b = vec![-128i8; 64 * 16];
        let mut c_vnni = vec![i32::MAX - 7; 16 * 16];
        let mut c_ref = vec![i32::MAX - 7; 16 * 16];

        unsafe {
            <AvxVnni as TileMatrixMultiply<i8, i8, i32, AvxVnni, AvxVnni, 16, 16, 64>>::tile_matmul(
                c_vnni.as_mut_ptr(),
                16,
                a.as_ptr(),
                64,
                b.as_ptr(),
                16,
            );
            <Scalar as TileMatrixMultiply<i8, i8, i32, Scalar, Scalar, 16, 16, 64>>::tile_matmul(
                c_ref.as_mut_ptr(),
                16,
                a.as_ptr(),
                64,
                b.as_ptr(),
                16,
            );
        }

        assert_eq!(c_vnni, c_ref, "wraparound semantics diverge from scalar");
    }
}

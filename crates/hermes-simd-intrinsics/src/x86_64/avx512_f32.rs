//! AVX-512F f32 hardware kernel.
//!
//! 16-lane f32 (`__m512`). Uses native AVX-512 predicated instructions:
//! - Masked load/store: `_mm512_mask_loadu_ps`, `_mm512_mask_storeu_ps`.
//! - Masked arithmetic: `_mm512_mask_add_ps`, `_mm512_mask_mul_ps`.
//! - Masked FMA: `_mm512_mask3_fmadd_ps` (inactive lanes retain `c`).
//! - Compress: `_mm512_maskz_compress_ps` (native — no emulation needed).
//! - Expand: `_mm512_mask_expand_ps` (native).
//! - Gather: `_mm512_i32gather_ps`, `_mm512_mask_i32gather_ps`.
//! - Mask register: `__mmask16` (16-bit integer).

use crate::Avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m512, __m512i, __mmask16, _mm512_add_ps, _mm512_and_si512, _mm512_andnot_si512,
    _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cmp_ps_mask, _mm512_cmplt_epi32_mask,
    _mm512_div_ps, _mm512_fmadd_ps, _mm512_fmaddsub_ps, _mm512_fmsub_ps, _mm512_fmsubadd_ps,
    _mm512_fnmadd_ps, _mm512_i32gather_ps, _mm512_i32scatter_ps, _mm512_load_ps, _mm512_loadu_ps,
    _mm512_mask3_fmadd_ps, _mm512_mask_add_ps, _mm512_mask_blend_ps, _mm512_mask_expand_ps,
    _mm512_mask_i32gather_ps, _mm512_mask_i32scatter_ps, _mm512_mask_loadu_ps, _mm512_mask_mov_ps,
    _mm512_mask_mul_ps, _mm512_mask_storeu_ps, _mm512_maskz_compress_ps, _mm512_max_ps,
    _mm512_min_ps, _mm512_movehdup_ps, _mm512_moveldup_ps, _mm512_mul_ps, _mm512_or_si512,
    _mm512_permute_ps, _mm512_reduce_add_ps, _mm512_roundscale_ps, _mm512_rsqrt14_ps,
    _mm512_set1_ps, _mm512_setzero_ps, _mm512_setzero_si512, _mm512_sqrt_ps, _mm512_store_ps,
    _mm512_storeu_ps, _mm512_stream_ps, _mm512_sub_ps, _mm512_xor_si512, _CMP_EQ_OQ, _CMP_GE_OQ,
    _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_NEQ_UQ, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT,
    _MM_FROUND_TO_NEG_INF, _MM_FROUND_TO_POS_INF, _MM_FROUND_TO_ZERO,
};
#[cfg(not(hermes_benchmark_generic_default))]
use core::arch::x86_64::{_mm512_permutex2var_ps, _mm512_permutexvar_ps, _mm512_setr_epi32};
use hermes_simd_core::kernel::BackendKernel;

/// Newtype over `__m512` providing `Send + Sync`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx512F32Vec(pub __m512);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx512F32Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx512F32Vec {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl BackendKernel<f32> for Avx512 {
    type Vector = Avx512F32Vec;
    /// Native AVX-512 16-bit mask register. Bit `i` set → lane `i` active.
    type Mask = __mmask16;
    /// 16 × i32 index vector for gather (`__m512i`).
    type IndexVector = __m512i;
    const LANE_COUNT: usize = 16;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vector {
        Avx512F32Vec(_mm512_load_ps(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vector {
        Avx512F32Vec(_mm512_loadu_ps(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f32, val: Self::Vector) {
        _mm512_store_ps(ptr, val.0);
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f32, val: Self::Vector) {
        _mm512_storeu_ps(ptr, val.0);
    }

    const SUPPORTS_NT_STORE: bool = true;

    // SAFETY: caller must ensure the target CPU supports `avx512f` (as above) and that `ptr` is aligned to the 16-lane width; `_mm512_stream_ps` (`vmovntps`/`vmovntpd`) faults on misalignment.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_streaming(ptr: *mut f32, val: Self::Vector) {
        _mm512_stream_ps(ptr, val.0);
    }

    #[inline]
    fn stream_write_barrier() {
        // SAFETY: `_mm_sfence` (SSE) is unconditionally available on x86_64.
        unsafe { core::arch::x86_64::_mm_sfence() };
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_add_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_mul_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_sub_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_fmadd_ps(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f`; operands
    // are registers of this backend and require no pointer validation.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        // SAFETY: this function enables AVX-512F, and all operands are valid
        // registers of the matching backend.
        Avx512F32Vec(_mm512_fmsub_ps(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_permute_ps(v.0, 0b1011_0001))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn swap_pairs(v: Self::Vector) -> Self::Vector {
        // Each 128-bit block holds two f32 pairs; `0b0100_1110` exchanges the
        // two 64-bit pairs within every block.
        Avx512F32Vec(_mm512_permute_ps::<0b0100_1110>(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_moveldup_ps(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_movehdup_ps(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_fmaddsub_ps(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_fmsubadd_ps(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f32 {
        _mm512_reduce_add_ps(v.0)
    }

    // -----------------------------------------------------------------------
    // Native masked load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f32,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_loadu_ps(src.0, mask, ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f32, mask: Self::Mask, val: Self::Vector) {
        _mm512_mask_storeu_ps(ptr, mask, val.0);
    }

    // -----------------------------------------------------------------------
    // Native masked arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_add_ps(src.0, mask, a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_mul_ps(src.0, mask, a.0, b.0))
    }

    /// `mask3_fmadd`: inactive lanes retain `c` (the addend / 3rd operand).
    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask3_fmadd_ps(a.0, b.0, c.0, mask))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f32 {
        let zero = _mm512_setzero_ps();
        _mm512_reduce_add_ps(_mm512_mask_mov_ps(zero, mask, v.0))
    }

    // -----------------------------------------------------------------------
    // Native compress / expand
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        Avx512F32Vec(_mm512_maskz_compress_ps(mask, src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_expand_ps(fill.0, mask, src.0))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather(base: *const f32, indices: Self::IndexVector) -> Self::Vector {
        Avx512F32Vec(_mm512_i32gather_ps(indices, base, 4))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f32,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_i32gather_ps(src.0, mask, indices, base, 4))
    }

    // -----------------------------------------------------------------------
    // Cross-lane permutes (native `vpermps` / `vpermi2ps`)
    // -----------------------------------------------------------------------
    //
    // AVX-512 expresses each of these as one full-width permute, so no
    // cross-half fixup is needed: `vpermps` reads any source lane, and
    // `vpermi2ps` indexes the 32-lane concatenation `a || b` (0..15 selects
    // `a`, 16..31 selects `b`), which is exactly the flat sequence the trait
    // contract is written on.

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn reverse(v: Self::Vector) -> Self::Vector {
        // Result lane i takes source lane idx[i], so descending indices reverse.
        let idx = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        Avx512F32Vec(_mm512_permutexvar_ps(idx, v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // Flat position p holds a[p/2] for even p and b[p/2] for odd p. The low
        // half covers p = 0..15 and the high half p = 16..31; `b` lanes are
        // selected by adding 16 to the index.
        let lo_idx = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
        let hi_idx =
            _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
        (
            Avx512F32Vec(_mm512_permutex2var_ps(a.0, lo_idx, b.0)),
            Avx512F32Vec(_mm512_permutex2var_ps(a.0, hi_idx, b.0)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // Over the concatenation `a || b`, the even output is positions 2i and
        // the odd output positions 2i+1 — the indices are the positions
        // themselves, since 0..15 already selects `a` and 16..31 selects `b`.
        let even_idx = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        let odd_idx = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        (
            Avx512F32Vec(_mm512_permutex2var_ps(a.0, even_idx, b.0)),
            Avx512F32Vec(_mm512_permutex2var_ps(a.0, odd_idx, b.0)),
        )
    }

    // -----------------------------------------------------------------------
    // Scatter (native `vscatterdps`)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn scatter(base: *mut f32, indices: Self::IndexVector, val: Self::Vector) {
        _mm512_i32scatter_ps(base, indices, val.0, 4);
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds. Inactive mask lanes are not dereferenced by the instruction.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn scatter_masked(
        base: *mut f32,
        indices: Self::IndexVector,
        mask: Self::Mask,
        val: Self::Vector,
    ) {
        _mm512_mask_i32scatter_ps(base, mask, indices, val.0, 4);
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 16);
        let mut mask: u64 = 0;
        for i in 0..16 {
            if bits[i] {
                mask |= 1 << i;
            }
        }
        mask as Self::Mask
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(16);
        if k == 0 {
            0 as Self::Mask
        } else {
            let mask = (1u64 << k) - 1;
            mask as Self::Mask
        }
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        Avx512F32Vec(_mm512_setzero_ps())
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn splat(val: f32) -> Self::Vector {
        Avx512F32Vec(_mm512_set1_ps(val))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_div_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_castsi512_ps(_mm512_and_si512(
            _mm512_castps_si512(a.0),
            _mm512_castps_si512(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_castsi512_ps(_mm512_or_si512(
            _mm512_castps_si512(a.0),
            _mm512_castps_si512(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_castsi512_ps(_mm512_xor_si512(
            _mm512_castps_si512(a.0),
            _mm512_castps_si512(b.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm512_castps_si512(_mm512_set1_ps(-0.0));
        Avx512F32Vec(_mm512_castsi512_ps(_mm512_andnot_si512(
            sign_mask,
            _mm512_castps_si512(a.0),
        )))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_min_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_max_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx512F32Vec(_mm512_sqrt_ps(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        let y0 = _mm512_rsqrt14_ps(a.0);
        let y0_sq = _mm512_mul_ps(y0, y0);
        let half_y0 = _mm512_mul_ps(y0, _mm512_set1_ps(0.5));
        let term = _mm512_fnmadd_ps(a.0, y0_sq, _mm512_set1_ps(3.0));
        Avx512F32Vec(_mm512_mul_ps(half_y0, term))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn floor(a: Self::Vector) -> Self::Vector {
        // `_mm512_floor_ps` is not exposed in Rust stdarch; use the equivalent
        // `vrndscaleps` with round-toward-negative-infinity mode.
        Avx512F32Vec(_mm512_roundscale_ps::<
            { _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn ceil(a: Self::Vector) -> Self::Vector {
        // `_mm512_ceil_ps` is not exposed in Rust stdarch; use the equivalent
        // `vrndscaleps` with round-toward-positive-infinity mode.
        Avx512F32Vec(_mm512_roundscale_ps::<
            { _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn round(a: Self::Vector) -> Self::Vector {
        // `vrndscaleps` in `_MM_FROUND_TO_NEAREST_INT` mode (imm low bits 00)
        // rounds ties to the even neighbor, matching the scalar `round_ties_even`
        // contract. `_MM_FROUND_NO_EXC` (imm bit 3) suppresses the inexact exception.
        Avx512F32Vec(_mm512_roundscale_ps::<
            { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn trunc(a: Self::Vector) -> Self::Vector {
        // `vrndscaleps` in `_MM_FROUND_TO_ZERO` mode (imm low bits 11) rounds toward zero.
        Avx512F32Vec(_mm512_roundscale_ps::<
            { _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); every 512-bit intrinsic below is AVX512F-only (no DQ/BW/VL — the dispatcher probes only `avx512f`), and the 256-bit helpers are AVX2, which every AVX512F processor implements; there are no memory operands.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        use core::arch::x86_64::{
            __m256, __m256i, _mm256_add_epi8, _mm256_and_si256, _mm256_castps_pd,
            _mm256_cvtepi32_ps, _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi16,
            _mm256_set1_epi8, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_srli_epi16,
            _mm512_castpd_ps, _mm512_castps256_ps512, _mm512_castps_pd, _mm512_castps_si512,
            _mm512_extracti64x4_epi64, _mm512_insertf64x4,
        };
        let v_si512 = _mm512_castps_si512(a.0);
        let lo = _mm512_extracti64x4_epi64(v_si512, 0);
        let hi = _mm512_extracti64x4_epi64(v_si512, 1);

        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );

        let run_avx2_popcount = |v: __m256i| -> __m256 {
            let v_lo = _mm256_and_si256(v, low_mask);
            let v_hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            let pop_lo = _mm256_shuffle_epi8(lookup, v_lo);
            let pop_hi = _mm256_shuffle_epi8(lookup, v_hi);
            let pop_bytes = _mm256_add_epi8(pop_lo, pop_hi);
            let pop_u16 = _mm256_maddubs_epi16(pop_bytes, _mm256_set1_epi8(1));
            let pop_u32 = _mm256_madd_epi16(pop_u16, _mm256_set1_epi16(1));
            _mm256_cvtepi32_ps(pop_u32)
        };

        let res_lo = run_avx2_popcount(lo);
        let res_hi = run_avx2_popcount(hi);

        // `_mm512_insertf32x8` would express this insertion directly but is an
        // AVX512DQ intrinsic — a #UD on F-only silicon (Knights Landing class).
        // `vinsertf64x4` is AVX512F and inserts the identical 256 high bits; the
        // `pd` casts are bit-preserving reinterpretations that keep the value in
        // the floating-point domain.
        let merged = _mm512_castpd_ps(_mm512_insertf64x4(
            _mm512_castps_pd(_mm512_castps256_ps512(res_lo)),
            _mm256_castps_pd(res_hi),
            1,
        ));
        Avx512F32Vec(merged)
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_EQ_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_NEQ_UQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_LT_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_LE_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_GT_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_ps_mask(a.0, b.0, _CMP_GE_OQ);
        Avx512F32Vec(_mm512_mask_blend_ps(
            m,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        // Selection is on the mask lane's sign bit. A "differs from zero"
        // comparison cannot express that: an active lane's `ALL_ONES` pattern is
        // a NaN, which an ordered predicate reports as inactive, while `-0.0`
        // carries a sign bit yet compares equal to zero under any predicate.
        Avx512F32Vec(_mm512_mask_blend_ps(
            <Self as BackendKernel<f32>>::vector_to_mask(mask),
            false_val.0,
            true_val.0,
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        u64::from(mask)
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 16-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        Avx512F32Vec(_mm512_mask_blend_ps(
            mask,
            _mm512_setzero_ps(),
            _mm512_set1_ps(f32::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); this is a register-to-register comparison with no memory operands.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // A signed integer lane is negative exactly when its sign bit is set, so
        // comparing the reinterpreted lanes against zero collects the sign bits
        // into a k-register. `_mm512_movepi32_mask` would express
        // this directly but requires AVX512DQ, which this backend does not enable.
        _mm512_cmplt_epi32_mask(_mm512_castps_si512(v.0), _mm512_setzero_si512())
    }
}

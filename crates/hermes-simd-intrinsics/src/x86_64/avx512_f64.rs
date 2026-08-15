//! AVX-512F f64 hardware kernel.
//!
//! 8-lane f64 (`__m512d`). Uses native AVX-512 predicated instructions:
//! - Masked load/store: `_mm512_mask_loadu_pd`, `_mm512_mask_storeu_pd`.
//! - Masked arithmetic: `_mm512_mask_add_pd`, `_mm512_mask_mul_pd`.
//! - Masked FMA: `_mm512_mask3_fmadd_pd` (inactive lanes retain `c`).
//! - Compress: `_mm512_maskz_compress_pd` (native — no emulation needed).
//! - Expand: `_mm512_mask_expand_pd` (native).
//! - Gather: `_mm512_i32gather_pd`, `_mm512_mask_i32gather_pd`.
//! - Mask register: `__mmask8` (8-bit integer).

use crate::Avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m256i, __m512d, __mmask8, _mm512_add_pd, _mm512_and_pd, _mm512_andnot_pd,
    _mm512_castpd_si512, _mm512_cmp_pd_mask, _mm512_cmplt_epi64_mask, _mm512_div_pd,
    _mm512_fmadd_pd, _mm512_fmaddsub_pd, _mm512_fmsubadd_pd, _mm512_i32gather_pd,
    _mm512_i32scatter_pd, _mm512_load_pd, _mm512_loadu_pd, _mm512_mask3_fmadd_pd,
    _mm512_mask_add_pd, _mm512_mask_blend_pd, _mm512_mask_expand_pd, _mm512_mask_i32gather_pd,
    _mm512_mask_i32scatter_pd, _mm512_mask_loadu_pd, _mm512_mask_mov_pd, _mm512_mask_mul_pd,
    _mm512_mask_storeu_pd, _mm512_maskz_compress_pd, _mm512_max_pd, _mm512_min_pd,
    _mm512_movedup_pd, _mm512_mul_pd, _mm512_or_pd, _mm512_permute_pd, _mm512_permutex2var_pd,
    _mm512_permutexvar_pd, _mm512_reduce_add_pd, _mm512_roundscale_pd, _mm512_set1_pd,
    _mm512_setr_epi64, _mm512_setzero_pd, _mm512_setzero_si512, _mm512_sqrt_pd, _mm512_store_pd,
    _mm512_storeu_pd, _mm512_stream_pd, _mm512_sub_pd, _mm512_xor_pd, _CMP_EQ_OQ, _CMP_GE_OQ,
    _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_NEQ_UQ, _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT,
    _MM_FROUND_TO_NEG_INF, _MM_FROUND_TO_POS_INF, _MM_FROUND_TO_ZERO,
};
use hermes_simd_core::kernel::BackendKernel;

/// Newtype over `__m512d` providing `Send + Sync`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx512F64Vec(pub __m512d);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx512F64Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx512F64Vec {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl BackendKernel<f64> for Avx512 {
    type Vector = Avx512F64Vec;
    /// Native AVX-512 8-bit mask register. Bit `i` set → lane `i` active.
    type Mask = __mmask8;
    /// 8 × i32 index vector for gather (`__m256i`).
    type IndexVector = __m256i;
    const LANE_COUNT: usize = 8;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f64) -> Self::Vector {
        Avx512F64Vec(_mm512_load_pd(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f64) -> Self::Vector {
        Avx512F64Vec(_mm512_loadu_pd(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f64, val: Self::Vector) {
        _mm512_store_pd(ptr, val.0);
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f64, val: Self::Vector) {
        _mm512_storeu_pd(ptr, val.0);
    }

    const SUPPORTS_NT_STORE: bool = true;

    // SAFETY: caller must ensure the target CPU supports `avx512f` (as above) and that `ptr` is aligned to the 8-lane width; `_mm512_stream_pd` (`vmovntps`/`vmovntpd`) faults on misalignment.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_streaming(ptr: *mut f64, val: Self::Vector) {
        _mm512_stream_pd(ptr, val.0);
    }

    #[inline]
    fn stream_write_barrier() {
        // SAFETY: `_mm_sfence` (SSE) is unconditionally available on x86_64.
        unsafe { core::arch::x86_64::_mm_sfence() };
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_add_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_mul_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_sub_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_fmadd_pd(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_permute_pd(v.0, 0b0101_0101))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_movedup_pd(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_permute_pd(v.0, 0b1111_1111))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_fmaddsub_pd(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_fmsubadd_pd(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f64 {
        _mm512_reduce_add_pd(v.0)
    }

    // -----------------------------------------------------------------------
    // Native masked load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f64,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_loadu_pd(src.0, mask, ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f64, mask: Self::Mask, val: Self::Vector) {
        _mm512_mask_storeu_pd(ptr, mask, val.0);
    }

    // -----------------------------------------------------------------------
    // Native masked arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_add_pd(src.0, mask, a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_mul_pd(src.0, mask, a.0, b.0))
    }

    /// `mask3_fmadd`: inactive lanes retain `c` (the addend / 3rd operand).
    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask3_fmadd_pd(a.0, b.0, c.0, mask))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f64 {
        let zero = _mm512_setzero_pd();
        _mm512_reduce_add_pd(_mm512_mask_mov_pd(zero, mask, v.0))
    }

    // -----------------------------------------------------------------------
    // Native compress / expand
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        Avx512F64Vec(_mm512_maskz_compress_pd(mask, src.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_expand_pd(fill.0, mask, src.0))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather(base: *const f64, indices: Self::IndexVector) -> Self::Vector {
        Avx512F64Vec(_mm512_i32gather_pd(indices, base, 8))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f64,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_i32gather_pd(src.0, mask, indices, base, 8))
    }

    // -----------------------------------------------------------------------
    // Cross-lane permutes (native `vpermpd` / `vpermi2pd`)
    // -----------------------------------------------------------------------
    //
    // Same shape as the f32 kernel at 8 lanes: `vpermi2pd` indexes the 16-lane
    // concatenation `a || b`, where 0..7 selects `a` and 8..15 selects `b`.
    // The index vector is `i64` lanes here, not `i32`.

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn reverse(v: Self::Vector) -> Self::Vector {
        let idx = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
        Avx512F64Vec(_mm512_permutexvar_pd(idx, v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        let lo_idx = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let hi_idx = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        (
            Avx512F64Vec(_mm512_permutex2var_pd(a.0, lo_idx, b.0)),
            Avx512F64Vec(_mm512_permutex2var_pd(a.0, hi_idx, b.0)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        let even_idx = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let odd_idx = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        (
            Avx512F64Vec(_mm512_permutex2var_pd(a.0, even_idx, b.0)),
            Avx512F64Vec(_mm512_permutex2var_pd(a.0, odd_idx, b.0)),
        )
    }

    // -----------------------------------------------------------------------
    // Scatter (native `vscatterdpd`)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn scatter(base: *mut f64, indices: Self::IndexVector, val: Self::Vector) {
        _mm512_i32scatter_pd(base, indices, val.0, 8);
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds. Inactive mask lanes are not dereferenced by the instruction.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn scatter_masked(
        base: *mut f64,
        indices: Self::IndexVector,
        mask: Self::Mask,
        val: Self::Vector,
    ) {
        _mm512_mask_i32scatter_pd(base, mask, indices, val.0, 8);
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 8);
        let mut mask: u64 = 0;
        for i in 0..8 {
            if bits[i] {
                mask |= 1 << i;
            }
        }
        mask as Self::Mask
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(8);
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

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        Avx512F64Vec(_mm512_setzero_pd())
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn splat(val: f64) -> Self::Vector {
        Avx512F64Vec(_mm512_set1_pd(val))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_div_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_and_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_or_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_xor_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm512_set1_pd(-0.0);
        Avx512F64Vec(_mm512_andnot_pd(sign_mask, a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_min_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_max_pd(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx512F64Vec(_mm512_sqrt_pd(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        // Full f64 precision via correctly-rounded sqrt + divide. The `rsqrt14` seed
        // (~14 bits) plus one Newton step reaches only ~28 bits, far below f64's 52;
        // see the avx2 f64 note.
        Avx512F64Vec(_mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(a.0)))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn floor(a: Self::Vector) -> Self::Vector {
        // `_mm512_floor_pd` is not exposed in Rust stdarch; use the equivalent
        // `vrndscalepd` with round-toward-negative-infinity mode.
        Avx512F64Vec(_mm512_roundscale_pd::<
            { _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn ceil(a: Self::Vector) -> Self::Vector {
        // `_mm512_ceil_pd` is not exposed in Rust stdarch; use the equivalent
        // `vrndscalepd` with round-toward-positive-infinity mode.
        Avx512F64Vec(_mm512_roundscale_pd::<
            { _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn round(a: Self::Vector) -> Self::Vector {
        // `vrndscalepd` in `_MM_FROUND_TO_NEAREST_INT` mode (imm low bits 00)
        // rounds ties to the even neighbor, matching the scalar `round_ties_even`
        // contract. `_MM_FROUND_NO_EXC` (imm bit 3) suppresses the inexact exception.
        Avx512F64Vec(_mm512_roundscale_pd::<
            { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn trunc(a: Self::Vector) -> Self::Vector {
        // `vrndscalepd` in `_MM_FROUND_TO_ZERO` mode (imm low bits 11) rounds toward zero.
        Avx512F64Vec(_mm512_roundscale_pd::<
            { _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        use core::arch::x86_64::{
            __m256d, __m256i, _mm256_add_epi8, _mm256_and_si256, _mm256_castsi256_si128,
            _mm256_cvtepi32_pd, _mm256_extractf128_si256, _mm256_sad_epu8, _mm256_set1_epi8,
            _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_shuffle_epi8,
            _mm256_srli_epi16, _mm512_castpd256_pd512, _mm512_castpd_si512,
            _mm512_extracti64x4_epi64, _mm512_insertf64x4, _mm_unpacklo_epi64,
        };
        let v_si512 = _mm512_castpd_si512(a.0);
        let lo = _mm512_extracti64x4_epi64(v_si512, 0);
        let hi = _mm512_extracti64x4_epi64(v_si512, 1);

        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );

        let run_avx2_popcount_f64 = |v: __m256i| -> __m256d {
            let v_lo = _mm256_and_si256(v, low_mask);
            let v_hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            let pop_lo = _mm256_shuffle_epi8(lookup, v_lo);
            let pop_hi = _mm256_shuffle_epi8(lookup, v_hi);
            let pop_bytes = _mm256_add_epi8(pop_lo, pop_hi);
            let pop_u64 = _mm256_sad_epu8(pop_bytes, _mm256_setzero_si256());
            let shuffled = _mm256_shuffle_epi32(pop_u64, 0xD8);
            let low = _mm256_castsi256_si128(shuffled);
            let high = _mm256_extractf128_si256(shuffled, 1);
            let packed = _mm_unpacklo_epi64(low, high);
            _mm256_cvtepi32_pd(packed)
        };

        let res_lo = run_avx2_popcount_f64(lo);
        let res_hi = run_avx2_popcount_f64(hi);

        let merged = _mm512_insertf64x4(_mm512_castpd256_pd512(res_lo), res_hi, 1);
        Avx512F64Vec(merged)
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_EQ_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(
            m,
            _mm512_setzero_pd(),
            _mm512_set1_pd(f64::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_NEQ_UQ);
        Avx512F64Vec(_mm512_mask_blend_pd(
            m,
            _mm512_setzero_pd(),
            _mm512_set1_pd(f64::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_LT_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(
            m,
            _mm512_setzero_pd(),
            _mm512_set1_pd(f64::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_LE_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(
            m,
            _mm512_setzero_pd(),
            _mm512_set1_pd(f64::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_GT_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(
            m,
            _mm512_setzero_pd(),
            _mm512_set1_pd(f64::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = _mm512_cmp_pd_mask(a.0, b.0, _CMP_GE_OQ);
        Avx512F64Vec(_mm512_mask_blend_pd(
            m,
            _mm512_setzero_pd(),
            _mm512_set1_pd(f64::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
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
        Avx512F64Vec(_mm512_mask_blend_pd(
            <Self as BackendKernel<f64>>::vector_to_mask(mask),
            false_val.0,
            true_val.0,
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        u64::from(mask)
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        Avx512F64Vec(_mm512_mask_blend_pd(
            mask,
            _mm512_setzero_pd(),
            _mm512_set1_pd(f64::from_bits(!0)),
        ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx512f` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); this is a register-to-register comparison with no memory operands.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // A signed integer lane is negative exactly when its sign bit is set, so
        // comparing the reinterpreted lanes against zero collects the sign bits
        // into a k-register. `_mm512_movepi64_mask` would express
        // this directly but requires AVX512DQ, which this backend does not enable.
        _mm512_cmplt_epi64_mask(_mm512_castpd_si512(v.0), _mm512_setzero_si512())
    }
}

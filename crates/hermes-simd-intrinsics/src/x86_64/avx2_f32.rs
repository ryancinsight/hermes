//! AVX2 f32 hardware kernel.
//!
//! 8-lane f32 (`__m256`). Masked operations use `_mm256_blendv_ps`
//! (blend-by-sign-bit). Gather uses native `_mm256_i32gather_ps` (scale = 4).
//! Compress/expand are emulated via scalar loops — AVX2 has no native
//! `vcompress` instruction (that requires AVX-512F).

use crate::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m256, __m256i, _mm256_add_epi8, _mm256_add_ps, _mm256_and_ps, _mm256_and_si256,
    _mm256_andnot_ps, _mm256_blendv_ps, _mm256_castps256_ps128, _mm256_castps_si256,
    _mm256_ceil_ps, _mm256_cmp_ps, _mm256_cvtepi32_ps, _mm256_div_ps, _mm256_extractf128_ps,
    _mm256_floor_ps, _mm256_fmadd_ps, _mm256_fmaddsub_ps, _mm256_fmsub_ps, _mm256_fmsubadd_ps,
    _mm256_fnmadd_ps, _mm256_i32gather_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_madd_epi16,
    _mm256_maddubs_epi16, _mm256_mask_i32gather_ps, _mm256_maskstore_ps, _mm256_max_ps,
    _mm256_min_ps, _mm256_movehdup_ps, _mm256_moveldup_ps, _mm256_movemask_ps, _mm256_mul_ps,
    _mm256_or_ps, _mm256_permute_ps, _mm256_permutevar8x32_ps, _mm256_round_ps, _mm256_rsqrt_ps,
    _mm256_set1_epi16, _mm256_set1_epi8, _mm256_set1_ps, _mm256_setr_epi32, _mm256_setr_epi8,
    _mm256_setzero_ps, _mm256_shuffle_epi8, _mm256_sqrt_ps, _mm256_srli_epi16, _mm256_store_ps,
    _mm256_storeu_ps, _mm256_stream_ps, _mm256_sub_ps, _mm256_xor_ps, _mm_add_ps, _mm_cvtss_f32,
    _mm_shuffle_ps, _CMP_EQ_OQ, _CMP_GE_OQ, _CMP_GT_OQ, _CMP_LE_OQ, _CMP_LT_OQ, _CMP_NEQ_UQ,
    _MM_FROUND_NO_EXC, _MM_FROUND_TO_NEAREST_INT, _MM_FROUND_TO_ZERO,
};
// Used only by the native interleave/deinterleave overrides, which the
// generic-default benchmark cfg compiles out.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(hermes_benchmark_generic_default)
))]
use core::arch::x86_64::{
    _mm256_permute2f128_ps, _mm256_shuffle_ps, _mm256_unpackhi_ps, _mm256_unpacklo_ps,
};
use hermes_simd_core::kernel::BackendKernel;

/// Newtype over `__m256` so `Send + Sync` can be implemented on the wrapper.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F32Vec(pub __m256);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F32Vec {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F32Vec {}

/// AVX2 f32 blend mask.
///
/// Stored as a `__m256` register. Lane `i` is active when the sign bit
/// of `mask[i]` is set.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2F32Mask(pub __m256);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2F32Mask {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2F32Mask {}

/// AVX2 gather index vector.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Avx2IdxI32(pub __m256i);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for Avx2IdxI32 {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for Avx2IdxI32 {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl BackendKernel<f32> for Avx2 {
    type Vector = Avx2F32Vec;
    type Mask = Avx2F32Mask;
    type IndexVector = Avx2IdxI32;
    const LANE_COUNT: usize = 8;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_aligned(ptr: *const f32) -> Self::Vector {
        Avx2F32Vec(_mm256_load_ps(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const f32) -> Self::Vector {
        Avx2F32Vec(_mm256_loadu_ps(ptr))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut f32, val: Self::Vector) {
        _mm256_store_ps(ptr, val.0);
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut f32, val: Self::Vector) {
        _mm256_storeu_ps(ptr, val.0);
    }

    const SUPPORTS_NT_STORE: bool = true;

    // SAFETY: caller must ensure the target CPU supports `avx2` (as above) and that `ptr` is aligned to the 8-lane width; `_mm256_stream_ps` (`vmovntps`/`vmovntpd`) faults on misalignment.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_streaming(ptr: *mut f32, val: Self::Vector) {
        _mm256_stream_ps(ptr, val.0);
    }

    #[inline]
    fn stream_write_barrier() {
        // SAFETY: `_mm_sfence` (SSE) is unconditionally available on x86_64.
        unsafe { core::arch::x86_64::_mm_sfence() };
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_add_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_mul_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_sub_ps(a.0, b.0))
    }

    /// FMA requires `avx2` + `fma` target features.
    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_fmadd_ps(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2,fma`; operands
    // are registers of this backend and require no pointer validation.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        // SAFETY: this function enables AVX2 and FMA, and all operands are
        // valid registers of the matching backend.
        Avx2F32Vec(_mm256_fmsub_ps(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_permute_ps(v.0, 0b1011_0001))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swap_pairs(v: Self::Vector) -> Self::Vector {
        // Each 128-bit half holds two f32 pairs; `0b0100_1110` exchanges the
        // two 64-bit pairs within each half.
        Avx2F32Vec(_mm256_permute_ps(v.0, 0b0100_1110))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn reverse(v: Self::Vector) -> Self::Vector {
        // `vpermps` is a full cross-lane permute (unlike `_mm256_permute_ps`,
        // which is per-128-bit-half), so one instruction expresses the flat
        // reversal. Index lane `i` selects source lane `7 - i`.
        let idx = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        Avx2F32Vec(_mm256_permutevar8x32_ps(v.0, idx))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_moveldup_ps(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_movehdup_ps(v.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // `unpacklo/hi_ps` weave within 128-bit halves; the `vperm2f128`
        // pair reassembles the flat interleave of the 16-lane sequence.
        let u_lo = _mm256_unpacklo_ps(a.0, b.0);
        let u_hi = _mm256_unpackhi_ps(a.0, b.0);
        (
            Avx2F32Vec(_mm256_permute2f128_ps::<0x20>(u_lo, u_hi)),
            Avx2F32Vec(_mm256_permute2f128_ps::<0x31>(u_lo, u_hi)),
        )
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    #[cfg(not(hermes_benchmark_generic_default))]
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        // Regroup the 128-bit halves so each operand of `shuffle_ps` holds
        // eight consecutive source lanes; the shuffles then pick the flat
        // even and odd subsequences.
        let t0 = _mm256_permute2f128_ps::<0x20>(a.0, b.0);
        let t1 = _mm256_permute2f128_ps::<0x31>(a.0, b.0);
        (
            Avx2F32Vec(_mm256_shuffle_ps::<0b10_00_10_00>(t0, t1)),
            Avx2F32Vec(_mm256_shuffle_ps::<0b11_01_11_01>(t0, t1)),
        )
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_fmaddsub_ps(a.0, b.0, c.0))
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_fmsubadd_ps(a.0, b.0, c.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> f32 {
        let hi_quad = _mm256_extractf128_ps(v.0, 1);
        let lo_quad = _mm256_castps256_ps128(v.0);
        let sum_quad = _mm_add_ps(lo_quad, hi_quad);
        let hi_dual = _mm_shuffle_ps(sum_quad, sum_quad, 0x4E);
        let sum_dual = _mm_add_ps(sum_quad, hi_dual);
        let hi_single = _mm_shuffle_ps(sum_dual, sum_dual, 0xB1);
        _mm_cvtss_f32(_mm_add_ps(sum_dual, hi_single))
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    /// Merge-masked load via `blendv`: sign bit of mask selects loaded vs src.
    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const f32,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let loaded = _mm256_loadu_ps(ptr);
        Avx2F32Vec(_mm256_blendv_ps(src.0, loaded, mask.0))
    }

    /// Masked store via maskstore; only lanes with sign bit set are written.
    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut f32, mask: Self::Mask, val: Self::Vector) {
        _mm256_maskstore_ps(ptr, _mm256_castps_si256(mask.0), val.0);
    }

    // -----------------------------------------------------------------------
    // Masked arithmetic
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = _mm256_add_ps(a.0, b.0);
        Avx2F32Vec(_mm256_blendv_ps(src.0, result, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = _mm256_mul_ps(a.0, b.0);
        Avx2F32Vec(_mm256_blendv_ps(src.0, result, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2,fma` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        let result = _mm256_fmadd_ps(a.0, b.0, c.0);
        Avx2F32Vec(_mm256_blendv_ps(c.0, result, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> f32 {
        let zero = _mm256_setzero_ps();
        let selected = _mm256_blendv_ps(zero, v.0, mask.0);
        Self::sum_reduce(Avx2F32Vec(selected))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated)
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mask_bits = _mm256_movemask_ps(mask.0) as u32;
        let mut arr = [0.0f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), src.0);
        let mut out = [0.0f32; 8];
        let mut k = 0usize;
        for i in 0..8 {
            if (mask_bits >> i) & 1 != 0 {
                out[k] = arr[i];
                k += 1;
            }
        }
        Avx2F32Vec(_mm256_loadu_ps(out.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mask_bits = _mm256_movemask_ps(mask.0) as u32;
        let mut src_arr = [0.0f32; 8];
        _mm256_storeu_ps(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0f32; 8];
        _mm256_storeu_ps(out_arr.as_mut_ptr(), fill.0);
        let mut k = 0usize;
        for i in 0..8 {
            if (mask_bits >> i) & 1 != 0 {
                out_arr[i] = src_arr[k];
                k += 1;
            }
        }
        Avx2F32Vec(_mm256_loadu_ps(out_arr.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather(base: *const f32, indices: Self::IndexVector) -> Self::Vector {
        Avx2F32Vec(_mm256_i32gather_ps(base, indices.0, 4))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather_masked(
        base: *const f32,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        Avx2F32Vec(_mm256_mask_i32gather_ps(src.0, base, indices.0, mask.0, 4))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 8);
        let vals: [f32; 8] =
            core::array::from_fn(|i| if bits[i] { <f32>::from_bits(!0) } else { 0.0 });
        Avx2F32Mask(_mm256_loadu_ps(vals.as_ptr()))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(8);
        let vals: [f32; 8] =
            core::array::from_fn(|i| if i < k { <f32>::from_bits(!0) } else { 0.0 });
        Avx2F32Mask(_mm256_loadu_ps(vals.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        Avx2F32Vec(_mm256_setzero_ps())
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn splat(val: f32) -> Self::Vector {
        Avx2F32Vec(_mm256_set1_ps(val))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_div_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_and_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_or_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_xor_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = _mm256_set1_ps(-0.0);
        Avx2F32Vec(_mm256_andnot_ps(sign_mask, a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_min_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_max_ps(a.0, b.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_sqrt_ps(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        let y0 = _mm256_rsqrt_ps(a.0);
        let y0_sq = _mm256_mul_ps(y0, y0);
        let half_y0 = _mm256_mul_ps(y0, _mm256_set1_ps(0.5));
        let term = _mm256_fnmadd_ps(a.0, y0_sq, _mm256_set1_ps(3.0));
        Avx2F32Vec(_mm256_mul_ps(half_y0, term))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn floor(a: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_floor_ps(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn ceil(a: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_ceil_ps(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn round(a: Self::Vector) -> Self::Vector {
        // `vroundps` in `_MM_FROUND_TO_NEAREST_INT` mode (imm low bits 00) rounds
        // ties to the even neighbor, matching the scalar `round_ties_even` contract.
        // `_MM_FROUND_NO_EXC` (imm bit 3) suppresses the per-element inexact exception.
        Avx2F32Vec(_mm256_round_ps::<
            { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
        >(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn trunc(a: Self::Vector) -> Self::Vector {
        // `vroundps` in `_MM_FROUND_TO_ZERO` mode (imm low bits 11) rounds toward zero.
        Avx2F32Vec(_mm256_round_ps::<{ _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC }>(a.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        let v = _mm256_castps_si256(a.0);
        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        let pop_lo = _mm256_shuffle_epi8(lookup, lo);
        let pop_hi = _mm256_shuffle_epi8(lookup, hi);
        let pop_bytes = _mm256_add_epi8(pop_lo, pop_hi);
        let pop_u16 = _mm256_maddubs_epi16(pop_bytes, _mm256_set1_epi8(1));
        let pop_u32 = _mm256_madd_epi16(pop_u16, _mm256_set1_epi16(1));
        let pop_f32 = _mm256_cvtepi32_ps(pop_u32);
        Avx2F32Vec(pop_f32)
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_EQ_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_NEQ_UQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_LT_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_LE_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_GT_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        Avx2F32Vec(_mm256_cmp_ps(a.0, b.0, _CMP_GE_OQ))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        Avx2F32Vec(_mm256_blendv_ps(false_val.0, true_val.0, mask.0))
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        _mm256_movemask_ps(mask.0) as u64
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); any pointer operands are valid for the 8-lane vector width within caller-validated bounds.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        Avx2F32Vec(mask.0)
    }

    // SAFETY: caller must ensure the target CPU supports `avx2` (enforced by the `#[target_feature]` gate above plus runtime `is_x86_feature_detected!` selection in the hermes-simd dispatcher (`target.rs`/`lib.rs`)); this is a register-to-register reinterpretation with no memory operands.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // Mask and vector share the `__m256` representation, and the
        // reinterpretation preserves lane sign bits for `_mm256_movemask_ps`.
        Avx2F32Mask(v.0)
    }
}

use std::fs::File;
use std::io::Write;
use std::path::Path;

struct Param {
    scalar_type: &'static str,
    vector_type: &'static str,
    vector_underlying: &'static str,
    mask_type: &'static str,
    mask_underlying: &'static str,
    index_vector: &'static str,
    index_underlying: &'static str,
    lane_count: usize,
    scale: usize,
    prefix: &'static str,
    suffix: &'static str,
    suffix_upper: &'static str,
    permute_mask: &'static str,
    dup_even_code: &'static str,
    dup_odd_code: &'static str,
    sum_reduce_code: &'static str,
    recip_sqrt_code: &'static str,
    popcount_code: &'static str,
    cmp_eq_val: &'static str,
    cmp_ne_val: &'static str,
    cmp_lt_val: &'static str,
    cmp_le_val: &'static str,
    cmp_gt_val: &'static str,
    cmp_ge_val: &'static str,
    one_half_val: &'static str,
    three_val: &'static str,
}

fn main() {
    let avx2_params = vec![
        Param {
            scalar_type: "f32",
            vector_type: "Avx2F32Vec",
            vector_underlying: "__m256",
            mask_type: "Avx2F32Mask",
            mask_underlying: "__m256",
            index_vector: "Avx2IdxI32",
            index_underlying: "__m256i",
            lane_count: 8,
            scale: 4,
            prefix: "_mm256",
            suffix: "ps",
            suffix_upper: "PS",
            permute_mask: "0b1011_0001",
            dup_even_code: "Avx2F32Vec(_mm256_moveldup_ps(v.0))",
            dup_odd_code: "Avx2F32Vec(_mm256_movehdup_ps(v.0))",
            cmp_eq_val: "_CMP_EQ_OQ",
            cmp_ne_val: "_CMP_NEQ_OQ",
            cmp_lt_val: "_CMP_LT_OQ",
            cmp_le_val: "_CMP_LE_OQ",
            cmp_gt_val: "_CMP_GT_OQ",
            cmp_ge_val: "_CMP_GE_OQ",
            one_half_val: "0.5f32",
            three_val: "3.0f32",
            sum_reduce_code: r#"
        let hi_quad = _mm256_extractf128_ps(v.0, 1);
        let lo_quad = _mm256_castps256_ps128(v.0);
        let sum_quad = _mm_add_ps(lo_quad, hi_quad);
        let hi_dual = _mm_shuffle_ps(sum_quad, sum_quad, 0x4E);
        let sum_dual = _mm_add_ps(sum_quad, hi_dual);
        let hi_single = _mm_shuffle_ps(sum_dual, sum_dual, 0xB1);
        _mm_cvtss_f32(_mm_add_ps(sum_dual, hi_single))
"#,
            recip_sqrt_code: r#"
        let y0 = _mm256_rsqrt_ps(a.0);
        let y0_sq = _mm256_mul_ps(y0, y0);
        let half_y0 = _mm256_mul_ps(y0, _mm256_set1_ps(0.5));
        let term = _mm256_fnmadd_ps(a.0, y0_sq, _mm256_set1_ps(3.0));
        Avx2F32Vec(_mm256_mul_ps(half_y0, term))
"#,
            popcount_code: r#"
        let v = _mm256_castps_si256(a.0);
        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
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
"#,
        },
        Param {
            scalar_type: "f64",
            vector_type: "Avx2F64Vec",
            vector_underlying: "__m256d",
            mask_type: "Avx2F64Mask",
            mask_underlying: "__m256d",
            index_vector: "Avx2F64Idx",
            index_underlying: "__m128i",
            lane_count: 4,
            scale: 8,
            prefix: "_mm256",
            suffix: "pd",
            suffix_upper: "PD",
            permute_mask: "0b0101",
            dup_even_code: "Avx2F64Vec(_mm256_movedup_pd(v.0))",
            dup_odd_code: "Avx2F64Vec(_mm256_permute_pd(v.0, 0b1111))",
            cmp_eq_val: "_CMP_EQ_OQ",
            cmp_ne_val: "_CMP_NEQ_OQ",
            cmp_lt_val: "_CMP_LT_OQ",
            cmp_le_val: "_CMP_LE_OQ",
            cmp_gt_val: "_CMP_GT_OQ",
            cmp_ge_val: "_CMP_GE_OQ",
            one_half_val: "0.5f64",
            three_val: "3.0f64",
            sum_reduce_code: r#"
        let hi = _mm256_extractf128_pd(v.0, 1);
        let lo = _mm256_castpd256_pd128(v.0);
        let sum_128 = _mm_add_pd(lo, hi);
        let hi_lane = _mm_unpackhi_pd(sum_128, sum_128);
        _mm_cvtsd_f64(_mm_add_pd(sum_128, hi_lane))
"#,
            recip_sqrt_code: r#"
        use core::arch::x86_64::*;
        let a_f32 = _mm256_cvtpd_ps(a.0);
        let r_f32 = _mm_rsqrt_ps(a_f32);
        let y0 = _mm256_cvtps_pd(r_f32);
        let y0_sq = _mm256_mul_pd(y0, y0);
        let half_y0 = _mm256_mul_pd(y0, _mm256_set1_pd(0.5));
        let term = _mm256_fnmadd_pd(a.0, y0_sq, _mm256_set1_pd(3.0));
        Avx2F64Vec(_mm256_mul_pd(half_y0, term))
"#,
            popcount_code: r#"
        use core::arch::x86_64::*;
        let v = _mm256_castpd_si256(a.0);
        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
        );
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        let pop_lo = _mm256_shuffle_epi8(lookup, lo);
        let pop_hi = _mm256_shuffle_epi8(lookup, hi);
        let pop_bytes = _mm256_add_epi8(pop_lo, pop_hi);
        let pop_u64 = _mm256_sad_epu8(pop_bytes, _mm256_setzero_si256());
        let shuffled = _mm256_shuffle_epi32(pop_u64, 0xD8);
        let low = _mm256_castsi256_si128(shuffled);
        let high = _mm256_extractf128_si256(shuffled, 1);
        let packed = _mm_unpacklo_epi64(low, high);
        let pop_f64 = _mm256_cvtepi32_pd(packed);
        Avx2F64Vec(pop_f64)
"#,
        },
    ];

    let avx512_params = vec![
        Param {
            scalar_type: "f32",
            vector_type: "Avx512F32Vec",
            vector_underlying: "__m512",
            mask_type: "__mmask16",
            mask_underlying: "__mmask16",
            index_vector: "__m512i",
            index_underlying: "__m512i",
            lane_count: 16,
            scale: 4,
            prefix: "_mm512",
            suffix: "ps",
            suffix_upper: "PS",
            permute_mask: "0b1011_0001",
            dup_even_code: "Avx512F32Vec(_mm512_moveldup_ps(v.0))",
            dup_odd_code: "Avx512F32Vec(_mm512_movehdup_ps(v.0))",
            cmp_eq_val: "_CMP_EQ_OQ",
            cmp_ne_val: "_CMP_NEQ_OQ",
            cmp_lt_val: "_CMP_LT_OQ",
            cmp_le_val: "_CMP_LE_OQ",
            cmp_gt_val: "_CMP_GT_OQ",
            cmp_ge_val: "_CMP_GE_OQ",
            one_half_val: "0.5f32",
            three_val: "3.0f32",
            sum_reduce_code: "        _mm512_reduce_add_ps(v.0)",
            recip_sqrt_code: r#"
        let y0 = _mm512_rsqrt14_ps(a.0);
        let y0_sq = _mm512_mul_ps(y0, y0);
        let half_y0 = _mm512_mul_ps(y0, _mm512_set1_ps(0.5));
        let term = _mm512_fnmadd_ps(a.0, y0_sq, _mm512_set1_ps(3.0));
        Avx512F32Vec(_mm512_mul_ps(half_y0, term))
"#,
            popcount_code: r#"
        use core::arch::x86_64::*;
        let v_si512 = _mm512_castps_si512(a.0);
        let lo = _mm512_extracti64x4_epi64(v_si512, 0);
        let hi = _mm512_extracti64x4_epi64(v_si512, 1);

        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
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

        let merged = _mm512_insertf32x8(_mm512_castps256_ps512(res_lo), res_hi, 1);
        Avx512F32Vec(merged)
"#,
        },
        Param {
            scalar_type: "f64",
            vector_type: "Avx512F64Vec",
            vector_underlying: "__m512d",
            mask_type: "__mmask8",
            mask_underlying: "__mmask8",
            index_vector: "__m256i",
            index_underlying: "__m256i",
            lane_count: 8,
            scale: 8,
            prefix: "_mm512",
            suffix: "pd",
            suffix_upper: "PD",
            permute_mask: "0b0101_0101",
            dup_even_code: "Avx512F64Vec(_mm512_movedup_pd(v.0))",
            dup_odd_code: "Avx512F64Vec(_mm512_permute_pd(v.0, 0b1111_1111))",
            cmp_eq_val: "_CMP_EQ_OQ",
            cmp_ne_val: "_CMP_NEQ_OQ",
            cmp_lt_val: "_CMP_LT_OQ",
            cmp_le_val: "_CMP_LE_OQ",
            cmp_gt_val: "_CMP_GT_OQ",
            cmp_ge_val: "_CMP_GE_OQ",
            one_half_val: "0.5f64",
            three_val: "3.0f64",
            sum_reduce_code: "        _mm512_reduce_add_pd(v.0)",
            recip_sqrt_code: r#"
        let y0 = _mm512_rsqrt14_pd(a.0);
        let y0_sq = _mm512_mul_pd(y0, y0);
        let half_y0 = _mm512_mul_pd(y0, _mm512_set1_pd(0.5));
        let term = _mm512_fnmadd_pd(a.0, y0_sq, _mm512_set1_pd(3.0));
        Avx512F64Vec(_mm512_mul_pd(half_y0, term))
"#,
            popcount_code: r#"
        use core::arch::x86_64::*;
        let v_si512 = _mm512_castpd_si512(a.0);
        let lo = _mm512_extracti64x4_epi64(v_si512, 0);
        let hi = _mm512_extracti64x4_epi64(v_si512, 1);

        let low_mask = _mm256_set1_epi8(0x0F);
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
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
"#,
        },
    ];

    let base_dir = Path::new("crates/hermes-simd-intrinsics/src/x86_64");

    for p in avx2_params {
        let content = render_avx2(&p);
        let file_path = base_dir.join(format!("avx2_{}.rs", p.scalar_type));
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        println!("Generated {:?}", file_path);
    }

    for p in avx512_params {
        let content = render_avx512(&p);
        let file_path = base_dir.join(format!("avx512_{}.rs", p.scalar_type));
        let mut file = File::create(&file_path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        println!("Generated {:?}", file_path);
    }
}

fn render_avx2(p: &Param) -> String {
    let t = r#"//! AVX2 __SCALAR_TYPE__ hardware kernel.
//!
//! __LANE_COUNT__-lane __SCALAR_TYPE__ (`__VECTOR_UNDERLYING__`). Masked operations use `__PREFIX___blendv___SUFFIX__`
//! (blend-by-sign-bit). Gather uses native `__PREFIX___i32gather___SUFFIX__` (scale = __SCALE__).
//! Compress/expand are emulated via scalar loops — AVX2 has no native
//! `vcompress` instruction (that requires AVX-512F).

use crate::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use hermes_simd_core::kernel::SimdKernel;

/// Newtype over `__VECTOR_UNDERLYING__` so `Send + Sync` can be implemented on the wrapper.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct __VECTOR_TYPE__(pub __VECTOR_UNDERLYING__);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for __VECTOR_TYPE__ {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for __VECTOR_TYPE__ {}

/// AVX2 __SCALAR_TYPE__ blend mask.
///
/// Stored as a `__MASK_UNDERLYING__` register. Lane `i` is active when the sign bit
/// of `mask[i]` is set.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct __MASK_TYPE__(pub __MASK_UNDERLYING__);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for __MASK_TYPE__ {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for __MASK_TYPE__ {}

/// AVX2 gather index vector.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct __INDEX_VECTOR__(pub __INDEX_UNDERLYING__);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for __INDEX_VECTOR__ {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for __INDEX_VECTOR__ {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<__SCALAR_TYPE__> for Avx2 {
    type Vector = __VECTOR_TYPE__;
    type Mask = __MASK_TYPE__;
    type IndexVector = __INDEX_VECTOR__;
    const LANE_COUNT: usize = __LANE_COUNT__;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_aligned(ptr: *const __SCALAR_TYPE__) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___load___SUFFIX__(ptr))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const __SCALAR_TYPE__) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___loadu___SUFFIX__(ptr))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut __SCALAR_TYPE__, val: Self::Vector) {
        __PREFIX___store___SUFFIX__(ptr, val.0);
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut __SCALAR_TYPE__, val: Self::Vector) {
        __PREFIX___storeu___SUFFIX__(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___add___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mul___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___sub___SUFFIX__(a.0, b.0))
    }

    /// FMA requires `avx2` + `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___fmadd___SUFFIX__(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___permute___SUFFIX__(v.0, __PERMUTE_MASK__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        __DUP_EVEN_CODE__
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        __DUP_ODD_CODE__
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___fmaddsub___SUFFIX__(a.0, b.0, c.0))
    }

    /// Alternating FMA requires `avx2` + `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___fmsubadd___SUFFIX__(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> __SCALAR_TYPE__ {
        __SUM_REDUCE_CODE__
    }

    // -----------------------------------------------------------------------
    // Masked load / store
    // -----------------------------------------------------------------------

    /// Merge-masked load via `blendv`: sign bit of mask selects loaded vs src.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const __SCALAR_TYPE__,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let loaded = __PREFIX___loadu___SUFFIX__(ptr);
        __VECTOR_TYPE__(__PREFIX___blendv___SUFFIX__(src.0, loaded, mask.0))
    }

    /// Masked store via maskstore; only lanes with sign bit set are written.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut __SCALAR_TYPE__, mask: Self::Mask, val: Self::Vector) {
        __PREFIX___maskstore___SUFFIX__(ptr, __PREFIX___cast__SUFFIX___si256(mask.0), val.0);
    }

    // -----------------------------------------------------------------------
    // Masked arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = __PREFIX___add___SUFFIX__(a.0, b.0);
        __VECTOR_TYPE__(__PREFIX___blendv___SUFFIX__(src.0, result, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        let result = __PREFIX___mul___SUFFIX__(a.0, b.0);
        __VECTOR_TYPE__(__PREFIX___blendv___SUFFIX__(src.0, result, mask.0))
    }

    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        let result = __PREFIX___fmadd___SUFFIX__(a.0, b.0, c.0);
        __VECTOR_TYPE__(__PREFIX___blendv___SUFFIX__(c.0, result, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> __SCALAR_TYPE__ {
        let zero = __PREFIX___setzero___SUFFIX__();
        let selected = __PREFIX___blendv___SUFFIX__(zero, v.0, mask.0);
        Self::sum_reduce(__VECTOR_TYPE__(selected))
    }

    // -----------------------------------------------------------------------
    // Compress / Expand (emulated)
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mask_bits = __PREFIX___movemask___SUFFIX__(mask.0) as u32;
        let mut arr = [0.0__SCALAR_TYPE__; __LANE_COUNT__];
        __PREFIX___storeu___SUFFIX__(arr.as_mut_ptr(), src.0);
        let mut out = [0.0__SCALAR_TYPE__; __LANE_COUNT__];
        let mut k = 0usize;
        for i in 0..__LANE_COUNT__ {
            if (mask_bits >> i) & 1 != 0 {
                out[k] = arr[i];
                k += 1;
            }
        }
        __VECTOR_TYPE__(__PREFIX___loadu___SUFFIX__(out.as_ptr()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mask_bits = __PREFIX___movemask___SUFFIX__(mask.0) as u32;
        let mut src_arr = [0.0__SCALAR_TYPE__; __LANE_COUNT__];
        __PREFIX___storeu___SUFFIX__(src_arr.as_mut_ptr(), src.0);
        let mut out_arr = [0.0__SCALAR_TYPE__; __LANE_COUNT__];
        __PREFIX___storeu___SUFFIX__(out_arr.as_mut_ptr(), fill.0);
        let mut k = 0usize;
        for i in 0..__LANE_COUNT__ {
            if (mask_bits >> i) & 1 != 0 {
                out_arr[i] = src_arr[k];
                k += 1;
            }
        }
        __VECTOR_TYPE__(__PREFIX___loadu___SUFFIX__(out_arr.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather(base: *const __SCALAR_TYPE__, indices: Self::IndexVector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___i32gather___SUFFIX__(base, indices.0, __SCALE__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn gather_masked(
        base: *const __SCALAR_TYPE__,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask_i32gather___SUFFIX__(src.0, base, indices.0, mask.0, __SCALE__))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), __LANE_COUNT__);
        let vals: [__SCALAR_TYPE__; __LANE_COUNT__] = core::array::from_fn(|i| {
            if bits[i] {
                <__SCALAR_TYPE__>::from_bits(!0)
            } else {
                0.0
            }
        });
        __MASK_TYPE__(__PREFIX___loadu___SUFFIX__(vals.as_ptr()))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(__LANE_COUNT__);
        let vals: [__SCALAR_TYPE__; __LANE_COUNT__] = core::array::from_fn(|i| {
            if i < k {
                <__SCALAR_TYPE__>::from_bits(!0)
            } else {
                0.0
            }
        });
        __MASK_TYPE__(__PREFIX___loadu___SUFFIX__(vals.as_ptr()))
    }

    // -----------------------------------------------------------------------
    // Broadcast / zero
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___setzero___SUFFIX__())
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn splat(val: __SCALAR_TYPE__) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___set1___SUFFIX__(val))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___div___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___and___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___or___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___xor___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = __PREFIX___set1___SUFFIX__(-0.0);
        __VECTOR_TYPE__(__PREFIX___andnot___SUFFIX__(sign_mask, a.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___min___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___max___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___sqrt___SUFFIX__(a.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        __RECIP_SQRT_CODE__
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        __POPCOUNT_CODE__
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___cmp___SUFFIX__(a.0, b.0, __CMP_EQ_VAL__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___cmp___SUFFIX__(a.0, b.0, __CMP_NE_VAL__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___cmp___SUFFIX__(a.0, b.0, __CMP_LT_VAL__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___cmp___SUFFIX__(a.0, b.0, __CMP_LE_VAL__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___cmp___SUFFIX__(a.0, b.0, __CMP_GT_VAL__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___cmp___SUFFIX__(a.0, b.0, __CMP_GE_VAL__))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___blendv___SUFFIX__(false_val.0, true_val.0, mask.0))
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        __PREFIX___movemask___SUFFIX__(mask.0) as u64
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        __VECTOR_TYPE__(mask.0)
    }
}
"#;
    t.replace("__SCALAR_TYPE__", p.scalar_type)
        .replace("__VECTOR_TYPE__", p.vector_type)
        .replace("__VECTOR_UNDERLYING__", p.vector_underlying)
        .replace("__MASK_TYPE__", p.mask_type)
        .replace("__MASK_UNDERLYING__", p.mask_underlying)
        .replace("__INDEX_VECTOR__", p.index_vector)
        .replace("__INDEX_UNDERLYING__", p.index_underlying)
        .replace("__LANE_COUNT__", &p.lane_count.to_string())
        .replace("__SCALE__", &p.scale.to_string())
        .replace("__PREFIX__", p.prefix)
        .replace("__SUFFIX__", p.suffix)
        .replace("__SUFFIX_UPPER__", p.suffix_upper)
        .replace("__PERMUTE_MASK__", p.permute_mask)
        .replace("__DUP_EVEN_CODE__", p.dup_even_code)
        .replace("__DUP_ODD_CODE__", p.dup_odd_code)
        .replace("__SUM_REDUCE_CODE__", p.sum_reduce_code)
        .replace("__RECIP_SQRT_CODE__", p.recip_sqrt_code)
        .replace("__POPCOUNT_CODE__", p.popcount_code)
        .replace("__CMP_EQ_VAL__", p.cmp_eq_val)
        .replace("__CMP_NE_VAL__", p.cmp_ne_val)
        .replace("__CMP_LT_VAL__", p.cmp_lt_val)
        .replace("__CMP_LE_VAL__", p.cmp_le_val)
        .replace("__CMP_GT_VAL__", p.cmp_gt_val)
        .replace("__CMP_GE_VAL__", p.cmp_ge_val)
        .replace("__ONE_HALF_VAL__", p.one_half_val)
        .replace("__THREE_VAL__", p.three_val)
}

fn render_avx512(p: &Param) -> String {
    let t = r#"//! AVX-512F __SCALAR_TYPE__ hardware kernel.
//!
//! __LANE_COUNT__-lane __SCALAR_TYPE__ (`__VECTOR_UNDERLYING__`). Uses native AVX-512 predicated instructions:
//! - Masked load/store: `__PREFIX___mask_loadu___SUFFIX__`, `__PREFIX___mask_storeu___SUFFIX__`.
//! - Masked arithmetic: `__PREFIX___mask_add___SUFFIX__`, `__PREFIX___mask_mul___SUFFIX__`.
//! - Masked FMA: `__PREFIX___mask3_fmadd___SUFFIX__` (inactive lanes retain `c`).
//! - Compress: `__PREFIX___maskz_compress___SUFFIX__` (native — no emulation needed).
//! - Expand: `__PREFIX___mask_expand___SUFFIX__` (native).
//! - Gather: `__PREFIX___i32gather___SUFFIX__`, `__PREFIX___mask_i32gather___SUFFIX__`.
//! - Mask register: `__MASK_TYPE__` (__LANE_COUNT__-bit integer).

use crate::Avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use hermes_simd_core::kernel::SimdKernel;

/// Newtype over `__VECTOR_UNDERLYING__` providing `Send + Sync`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct __VECTOR_TYPE__(pub __VECTOR_UNDERLYING__);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Send for __VECTOR_TYPE__ {}
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe impl Sync for __VECTOR_TYPE__ {}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<__SCALAR_TYPE__> for Avx512 {
    type Vector = __VECTOR_TYPE__;
    /// Native AVX-512 __LANE_COUNT__-bit mask register. Bit `i` set → lane `i` active.
    type Mask = __MASK_TYPE__;
    /// __LANE_COUNT__ × i32 index vector for gather (`__INDEX_UNDERLYING__`).
    type IndexVector = __INDEX_UNDERLYING__;
    const LANE_COUNT: usize = __LANE_COUNT__;
    const UNROLL_FACTOR: usize = 4;

    // -----------------------------------------------------------------------
    // Load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_aligned(ptr: *const __SCALAR_TYPE__) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___load___SUFFIX__(ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn load_unaligned(ptr: *const __SCALAR_TYPE__) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___loadu___SUFFIX__(ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_aligned(ptr: *mut __SCALAR_TYPE__, val: Self::Vector) {
        __PREFIX___store___SUFFIX__(ptr, val.0);
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn store_unaligned(ptr: *mut __SCALAR_TYPE__, val: Self::Vector) {
        __PREFIX___storeu___SUFFIX__(ptr, val.0);
    }

    // -----------------------------------------------------------------------
    // Arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___add___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mul___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___sub___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___fmadd___SUFFIX__(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___permute___SUFFIX__(v.0, __PERMUTE_MASK__))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        __DUP_EVEN_CODE__
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        __DUP_ODD_CODE__
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___fmaddsub___SUFFIX__(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___fmsubadd___SUFFIX__(a.0, b.0, c.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sum_reduce(v: Self::Vector) -> __SCALAR_TYPE__ {
        __SUM_REDUCE_CODE__
    }

    // -----------------------------------------------------------------------
    // Native masked load / store
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_load_unaligned(
        ptr: *const __SCALAR_TYPE__,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask_loadu___SUFFIX__(src.0, mask, ptr))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_store_unaligned(ptr: *mut __SCALAR_TYPE__, mask: Self::Mask, val: Self::Vector) {
        __PREFIX___mask_storeu___SUFFIX__(ptr, mask, val.0);
    }

    // -----------------------------------------------------------------------
    // Native masked arithmetic
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask_add___SUFFIX__(src.0, mask, a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask_mul___SUFFIX__(src.0, mask, a.0, b.0))
    }

    /// `mask3_fmadd`: inactive lanes retain `c` (the addend / 3rd operand).
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask3_fmadd___SUFFIX__(a.0, b.0, c.0, mask))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> __SCALAR_TYPE__ {
        let zero = __PREFIX___setzero___SUFFIX__();
        __PREFIX___reduce_add___SUFFIX__(__PREFIX___mask_mov___SUFFIX__(zero, mask, v.0))
    }

    // -----------------------------------------------------------------------
    // Native compress / expand
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___maskz_compress___SUFFIX__(mask, src.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask_expand___SUFFIX__(fill.0, mask, src.0))
    }

    // -----------------------------------------------------------------------
    // Gather
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather(base: *const __SCALAR_TYPE__, indices: Self::IndexVector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___i32gather___SUFFIX__(indices, base, __SCALE__))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn gather_masked(
        base: *const __SCALAR_TYPE__,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask_i32gather___SUFFIX__(src.0, mask, indices, base, __SCALE__))
    }

    // -----------------------------------------------------------------------
    // Mask construction
    // -----------------------------------------------------------------------

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), __LANE_COUNT__);
        let mut mask: u64 = 0;
        for i in 0..__LANE_COUNT__ {
            if bits[i] {
                mask |= 1 << i;
            }
        }
        mask as Self::Mask
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        let k = k.min(__LANE_COUNT__);
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

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn zero() -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___setzero___SUFFIX__())
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn splat(val: __SCALAR_TYPE__) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___set1___SUFFIX__(val))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___div___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___and___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___or___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___xor___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let sign_mask = __PREFIX___set1___SUFFIX__(-0.0);
        __VECTOR_TYPE__(__PREFIX___andnot___SUFFIX__(sign_mask, a.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___min___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___max___SUFFIX__(a.0, b.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___sqrt___SUFFIX__(a.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn recip_sqrt(a: Self::Vector) -> Self::Vector {
        __RECIP_SQRT_CODE__
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn popcount(a: Self::Vector) -> Self::Vector {
        __POPCOUNT_CODE__
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = __PREFIX___cmp___SUFFIX___mask(a.0, b.0, __CMP_EQ_VAL__);
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(
            m,
            __PREFIX___setzero___SUFFIX__(),
            __PREFIX___set1___SUFFIX__(__SCALAR_TYPE__::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = __PREFIX___cmp___SUFFIX___mask(a.0, b.0, __CMP_NE_VAL__);
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(
            m,
            __PREFIX___setzero___SUFFIX__(),
            __PREFIX___set1___SUFFIX__(__SCALAR_TYPE__::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = __PREFIX___cmp___SUFFIX___mask(a.0, b.0, __CMP_LT_VAL__);
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(
            m,
            __PREFIX___setzero___SUFFIX__(),
            __PREFIX___set1___SUFFIX__(__SCALAR_TYPE__::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = __PREFIX___cmp___SUFFIX___mask(a.0, b.0, __CMP_LE_VAL__);
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(
            m,
            __PREFIX___setzero___SUFFIX__(),
            __PREFIX___set1___SUFFIX__(__SCALAR_TYPE__::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = __PREFIX___cmp___SUFFIX___mask(a.0, b.0, __CMP_GT_VAL__);
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(
            m,
            __PREFIX___setzero___SUFFIX__(),
            __PREFIX___set1___SUFFIX__(__SCALAR_TYPE__::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let m = __PREFIX___cmp___SUFFIX___mask(a.0, b.0, __CMP_GE_VAL__);
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(
            m,
            __PREFIX___setzero___SUFFIX__(),
            __PREFIX___set1___SUFFIX__(__SCALAR_TYPE__::from_bits(!0)),
        ))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn blend(
        mask: Self::Vector,
        true_val: Self::Vector,
        false_val: Self::Vector,
    ) -> Self::Vector {
        let m = __PREFIX___cmp___SUFFIX___mask(mask.0, __PREFIX___setzero___SUFFIX__(), __CMP_NE_VAL__);
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(m, false_val.0, true_val.0))
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        mask as u64
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        __VECTOR_TYPE__(__PREFIX___mask_blend___SUFFIX__(
            mask,
            __PREFIX___setzero___SUFFIX__(),
            __PREFIX___set1___SUFFIX__(__SCALAR_TYPE__::from_bits(!0)),
        ))
    }
}
"#;
    t.replace("__SCALAR_TYPE__", p.scalar_type)
        .replace("__VECTOR_TYPE__", p.vector_type)
        .replace("__VECTOR_UNDERLYING__", p.vector_underlying)
        .replace("__MASK_TYPE__", p.mask_type)
        .replace("__MASK_UNDERLYING__", p.mask_underlying)
        .replace("__INDEX_UNDERLYING__", p.index_underlying)
        .replace("__LANE_COUNT__", &p.lane_count.to_string())
        .replace("__SCALE__", &p.scale.to_string())
        .replace("__PREFIX__", p.prefix)
        .replace("__SUFFIX__", p.suffix)
        .replace("__SUFFIX_UPPER__", p.suffix_upper)
        .replace("__PERMUTE_MASK__", p.permute_mask)
        .replace("__DUP_EVEN_CODE__", p.dup_even_code)
        .replace("__DUP_ODD_CODE__", p.dup_odd_code)
        .replace("__SUM_REDUCE_CODE__", p.sum_reduce_code)
        .replace("__RECIP_SQRT_CODE__", p.recip_sqrt_code)
        .replace("__POPCOUNT_CODE__", p.popcount_code)
        .replace("__CMP_EQ_VAL__", p.cmp_eq_val)
        .replace("__CMP_NE_VAL__", p.cmp_ne_val)
        .replace("__CMP_LT_VAL__", p.cmp_lt_val)
        .replace("__CMP_LE_VAL__", p.cmp_le_val)
        .replace("__CMP_GT_VAL__", p.cmp_gt_val)
        .replace("__CMP_GE_VAL__", p.cmp_ge_val)
        .replace("__ONE_HALF_VAL__", p.one_half_val)
        .replace("__THREE_VAL__", p.three_val)
}

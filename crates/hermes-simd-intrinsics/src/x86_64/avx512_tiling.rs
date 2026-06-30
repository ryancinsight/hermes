//! AVX-512 specific implementations of TileMatrixMultiply for x86_64.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use crate::Avx512;
use eunomia::{Bf16, Bf4, Bf8, F32, I32, I8};
use hermes_simd_core::view::TileMatrixMultiply;

// ---------------------------------------------------------------------------
// AVX-512 Implementations
// ---------------------------------------------------------------------------

impl TileMatrixMultiply<half::bf16, half::bf16, f32, Avx512, Avx512, 16, 16, 32> for Avx512 {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn tile_matmul(
        c: *mut f32,
        c_stride: usize,
        a: *const half::bf16,
        a_stride: usize,
        b: *const half::bf16,
        b_stride: usize,
    ) {
        use core::arch::x86_64::*;

        let mut c_regs = [_mm512_setzero_ps(); 16];
        for i in 0..16 {
            c_regs[i] = _mm512_loadu_ps(c.add(i * c_stride));
        }

        for k in 0..32 {
            let mut a_vals = [0f32; 16];
            for i in 0..16 {
                a_vals[i] = (*a.add(i * a_stride + k)).to_f32();
            }

            let b_ptr = b.add(k * b_stride);
            let mut b_vals = [0f32; 16];
            for j in 0..16 {
                b_vals[j] = (*b_ptr.add(j)).to_f32();
            }
            let b_vec = _mm512_loadu_ps(b_vals.as_ptr());

            for i in 0..16 {
                let a_vec = _mm512_set1_ps(a_vals[i]);
                c_regs[i] = _mm512_fmadd_ps(a_vec, b_vec, c_regs[i]);
            }
        }

        for i in 0..16 {
            _mm512_storeu_ps(c.add(i * c_stride), c_regs[i]);
        }
    }
}

impl TileMatrixMultiply<Bf16, Bf16, F32, Avx512, Avx512, 16, 16, 32> for Avx512 {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn tile_matmul(
        c: *mut F32,
        c_stride: usize,
        a: *const Bf16,
        a_stride: usize,
        b: *const Bf16,
        b_stride: usize,
    ) {
        use core::arch::x86_64::*;

        let mut c_regs = [_mm512_setzero_ps(); 16];
        for i in 0..16 {
            c_regs[i] = _mm512_loadu_ps(c.add(i * c_stride) as *const f32);
        }

        for k in 0..32 {
            let mut a_vals = [0f32; 16];
            for i in 0..16 {
                a_vals[i] = (*a.add(i * a_stride + k)).0.to_f32();
            }

            let b_ptr = b.add(k * b_stride);
            let mut b_vals = [0f32; 16];
            for j in 0..16 {
                b_vals[j] = (*b_ptr.add(j)).0.to_f32();
            }
            let b_vec = _mm512_loadu_ps(b_vals.as_ptr());

            for i in 0..16 {
                let a_vec = _mm512_set1_ps(a_vals[i]);
                c_regs[i] = _mm512_fmadd_ps(a_vec, b_vec, c_regs[i]);
            }
        }

        for i in 0..16 {
            _mm512_storeu_ps(c.add(i * c_stride) as *mut f32, c_regs[i]);
        }
    }
}

impl TileMatrixMultiply<i8, i8, i32, Avx512, Avx512, 16, 16, 64> for Avx512 {
    #[target_feature(enable = "avx512f,avx512vnni,avx512vl")]
    unsafe fn tile_matmul(
        c: *mut i32,
        c_stride: usize,
        a: *const i8,
        a_stride: usize,
        b: *const i8,
        b_stride: usize,
    ) {
        use core::arch::x86_64::*;

        let mut c_regs = [_mm512_setzero_si512(); 16];
        for i in 0..16 {
            c_regs[i] = _mm512_loadu_si512(c.add(i * c_stride) as *const _);
        }

        for k in (0..64).step_by(4) {
            let row0 = _mm_loadu_si128(b.add(k * b_stride) as *const __m128i);
            let row1 = _mm_loadu_si128(b.add((k + 1) * b_stride) as *const __m128i);
            let row2 = _mm_loadu_si128(b.add((k + 2) * b_stride) as *const __m128i);
            let row3 = _mm_loadu_si128(b.add((k + 3) * b_stride) as *const __m128i);

            let unpack_lo_01 = _mm_unpacklo_epi8(row0, row1);
            let unpack_hi_01 = _mm_unpackhi_epi8(row0, row1);

            let unpack_lo_23 = _mm_unpacklo_epi8(row2, row3);
            let unpack_hi_23 = _mm_unpackhi_epi8(row2, row3);

            let packed_lo = _mm_unpacklo_epi16(unpack_lo_01, unpack_lo_23);
            let packed_mid_lo = _mm_unpackhi_epi16(unpack_lo_01, unpack_lo_23);
            let packed_mid_hi = _mm_unpacklo_epi16(unpack_hi_01, unpack_hi_23);
            let packed_hi = _mm_unpackhi_epi16(unpack_hi_01, unpack_hi_23);

            let b_vec = _mm512_inserti32x4(
                _mm512_inserti32x4(
                    _mm512_inserti32x4(_mm512_castsi128_si512(packed_lo), packed_mid_lo, 1),
                    packed_mid_hi,
                    2,
                ),
                packed_hi,
                3,
            );

            for i in 0..16 {
                let a_val = core::ptr::read_unaligned(a.add(i * a_stride + k) as *const i32);
                let a_vec = _mm512_set1_epi32(a_val);
                c_regs[i] = crate::x86_64::asm_intrinsics::vpdpbssd!(c_regs[i], a_vec, b_vec);
            }
        }

        for i in 0..16 {
            _mm512_storeu_si512(c.add(i * c_stride) as *mut _, c_regs[i]);
        }
    }
}

impl TileMatrixMultiply<I8, I8, I32, Avx512, Avx512, 16, 16, 64> for Avx512 {
    #[target_feature(enable = "avx512f,avx512vnni,avx512vl")]
    unsafe fn tile_matmul(
        c: *mut I32,
        c_stride: usize,
        a: *const I8,
        a_stride: usize,
        b: *const I8,
        b_stride: usize,
    ) {
        use core::arch::x86_64::*;

        let mut c_regs = [_mm512_setzero_si512(); 16];
        for i in 0..16 {
            c_regs[i] = _mm512_loadu_si512(c.add(i * c_stride) as *const _);
        }

        for k in (0..64).step_by(4) {
            let row0 = _mm_loadu_si128(b.add(k * b_stride) as *const __m128i);
            let row1 = _mm_loadu_si128(b.add((k + 1) * b_stride) as *const __m128i);
            let row2 = _mm_loadu_si128(b.add((k + 2) * b_stride) as *const __m128i);
            let row3 = _mm_loadu_si128(b.add((k + 3) * b_stride) as *const __m128i);

            let unpack_lo_01 = _mm_unpacklo_epi8(row0, row1);
            let unpack_hi_01 = _mm_unpackhi_epi8(row0, row1);

            let unpack_lo_23 = _mm_unpacklo_epi8(row2, row3);
            let unpack_hi_23 = _mm_unpackhi_epi8(row2, row3);

            let packed_lo = _mm_unpacklo_epi16(unpack_lo_01, unpack_lo_23);
            let packed_mid_lo = _mm_unpackhi_epi16(unpack_lo_01, unpack_lo_23);
            let packed_mid_hi = _mm_unpacklo_epi16(unpack_hi_01, unpack_hi_23);
            let packed_hi = _mm_unpackhi_epi16(unpack_hi_01, unpack_hi_23);

            let b_vec = _mm512_inserti32x4(
                _mm512_inserti32x4(
                    _mm512_inserti32x4(_mm512_castsi128_si512(packed_lo), packed_mid_lo, 1),
                    packed_mid_hi,
                    2,
                ),
                packed_hi,
                3,
            );

            for i in 0..16 {
                let a_val = core::ptr::read_unaligned(a.add(i * a_stride + k) as *const i32);
                let a_vec = _mm512_set1_epi32(a_val);
                c_regs[i] = crate::x86_64::asm_intrinsics::vpdpbssd!(c_regs[i], a_vec, b_vec);
            }
        }

        for i in 0..16 {
            _mm512_storeu_si512(c.add(i * c_stride) as *mut _, c_regs[i]);
        }
    }
}

/// Unpacks packed 4-bit signed integers (stored 2 per byte) into an 8-bit signed integer slice.
/// Optimized using AVX2 when available.
#[inline]
pub fn unpack_int4(packed: &[u8], unpacked: &mut [i8]) {
    let len = packed.len();
    assert!(unpacked.len() >= len * 2);

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe {
            unpack_int4_avx2(packed, unpacked);
        }
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        unpack_int4_scalar(packed, unpacked);
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
fn unpack_int4_scalar(packed: &[u8], unpacked: &mut [i8]) {
    for (i, byte) in packed.iter().copied().enumerate() {
        let byte = byte as i8;
        unpacked[2 * i] = (byte << 4) >> 4;
        unpacked[2 * i + 1] = byte >> 4;
    }
}

/// AVX2 optimized int4 unpacker.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn unpack_int4_avx2(packed: &[u8], unpacked: &mut [i8]) {
    use core::arch::x86_64::*;

    let len = packed.len();
    let mut i = 0;

    let mask = _mm256_set1_epi8(0x0F);
    let lookup = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6,
        -5, -4, -3, -2, -1,
    );

    while i + 32 <= len {
        let v = _mm256_loadu_si256(packed.as_ptr().add(i) as *const _);

        let low_nibbles = _mm256_and_si256(v, mask);
        let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);

        let low_signed = _mm256_shuffle_epi8(lookup, low_nibbles);
        let high_signed = _mm256_shuffle_epi8(lookup, high_nibbles);

        let res_lo = _mm256_unpacklo_epi8(low_signed, high_signed);
        let res_hi = _mm256_unpackhi_epi8(low_signed, high_signed);

        let res0 = _mm256_permute2x128_si256(res_lo, res_hi, 0x20);
        let res1 = _mm256_permute2x128_si256(res_lo, res_hi, 0x31);

        _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i) as *mut _, res0);
        _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i + 32) as *mut _, res1);

        i += 32;
    }

    for j in i..len {
        let byte = packed[j] as i8;
        unpacked[2 * j] = (byte << 4) >> 4;
        unpacked[2 * j + 1] = byte >> 4;
    }
}

/// Unpacks Bf8 elements to Bf16 for accumulation.
#[inline]
pub fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf8_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512bw") {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf8_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_bf8_to_bf16(packed, unpacked);
}

/// Unpacks Bf4 elements to Bf16 for accumulation.
#[inline]
pub fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512bw") {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_bf4_to_bf16(packed, unpacked);
}

/// Unpacks packed Bf4 elements (stored 2 per byte in `packed`) into a Bf16 slice.
#[inline]
pub fn unpack_packed_bf4_to_bf16(packed: &[u8], unpacked: &mut [Bf16]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512bw")
                && std::is_x86_feature_detected!("avx512vl")
            {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16_packed(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512bw") {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_bf4_to_bf16_packed(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_bf4_to_bf16_packed(packed, unpacked);
}

/// Unpacks packed F4 elements (stored 2 per byte in `packed`) into an F32 slice.
#[inline]
pub fn unpack_packed_f4_to_f32(packed: &[u8], unpacked: &mut [F32]) {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl")
            {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_f4_to_f32_packed(packed, unpacked);
                    return;
                }
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512f") {
                unsafe {
                    eunomia::unsafe_intrinsics::avx512::unpack_f4_to_f32_packed(packed, unpacked);
                    return;
                }
            }
        }
    }
    eunomia::unpack_f4_to_f32_packed(packed, unpacked);
}

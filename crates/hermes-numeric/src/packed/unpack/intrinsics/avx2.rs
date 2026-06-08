#![allow(dead_code)]
use super::super::conv::{bf4_to_bf16_bits, f4_to_f32_bits, f8_to_f32_bits};
use crate::types::{Bf16, Bf4, Bf8, F32, F4, F8};

#[target_feature(enable = "avx2")]
pub unsafe fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    let len = packed.len().min(unpacked.len());
    let mut i = 0;
    while i + 16 <= len {
        let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
        let v_in = core::arch::x86_64::_mm_loadu_si128(ptr);
        let v_u16 = core::arch::x86_64::_mm256_cvtepu8_epi16(v_in);
        let sign = core::arch::x86_64::_mm256_slli_epi16(
            core::arch::x86_64::_mm256_and_si256(
                v_u16,
                core::arch::x86_64::_mm256_set1_epi16(0x80),
            ),
            8,
        );
        let rest = core::arch::x86_64::_mm256_slli_epi16(
            core::arch::x86_64::_mm256_and_si256(
                v_u16,
                core::arch::x86_64::_mm256_set1_epi16(0x7f),
            ),
            5,
        );
        let is_zero = core::arch::x86_64::_mm256_cmpeq_epi16(
            rest,
            core::arch::x86_64::_mm256_setzero_si256(),
        );
        let bias_diff = core::arch::x86_64::_mm256_andnot_si256(
            is_zero,
            core::arch::x86_64::_mm256_set1_epi16(112 << 7),
        );
        let rest_biased = core::arch::x86_64::_mm256_add_epi16(rest, bias_diff);
        let result = core::arch::x86_64::_mm256_or_si256(sign, rest_biased);
        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut core::arch::x86_64::__m256i;
        core::arch::x86_64::_mm256_storeu_si256(out_ptr, result);
        i += 16;
    }
    for j in i..len {
        let b = packed[j].0 as u16;
        let sign = (b & 0x80) << 8;
        let rest = (b & 0x7f) << 5;
        let bias_diff = if rest == 0 { 0 } else { 112 << 7 };
        unpacked[j] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
    use core::arch::x86_64::*;
    let len = packed.len().min(unpacked.len());
    let mut i = 0;

    static TABLE_LO: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = (bf4_to_bf16_bits(idx as u8) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_HI: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = (bf4_to_bf16_bits(idx as u8) >> 8) as u8;
            idx += 1;
        }
        t
    };

    let table_lo = _mm_loadu_si128(TABLE_LO.as_ptr() as *const _);
    let table_hi = _mm_loadu_si128(TABLE_HI.as_ptr() as *const _);
    let mask_0f = _mm_set1_epi8(0x0F);

    while i + 16 <= len {
        let ptr = packed.as_ptr().add(i) as *const __m128i;
        let v_in = _mm_loadu_si128(ptr);
        let indices = _mm_and_si128(v_in, mask_0f);

        let res_lo = _mm_shuffle_epi8(table_lo, indices);
        let res_hi = _mm_shuffle_epi8(table_hi, indices);

        let out_lo = _mm_unpacklo_epi8(res_lo, res_hi);
        let out_hi = _mm_unpackhi_epi8(res_lo, res_hi);

        _mm_storeu_si128(unpacked.as_mut_ptr().add(i) as *mut _, out_lo);
        _mm_storeu_si128(unpacked.as_mut_ptr().add(i + 8) as *mut _, out_hi);

        i += 16;
    }
    for j in i..len {
        let b = packed[j].0;
        unpacked[j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b)));
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn unpack_bf4_to_bf16_packed(packed: &[u8], unpacked: &mut [Bf16]) {
    use core::arch::x86_64::*;
    let len = packed.len();
    let n = len.min(unpacked.len() / 2);
    let mut i = 0;

    static TABLE_LO: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = (bf4_to_bf16_bits(idx as u8) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_HI: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = (bf4_to_bf16_bits(idx as u8) >> 8) as u8;
            idx += 1;
        }
        t
    };

    let table_lo = _mm_loadu_si128(TABLE_LO.as_ptr() as *const _);
    let table_hi = _mm_loadu_si128(TABLE_HI.as_ptr() as *const _);
    let mask_0f = _mm_set1_epi8(0x0F);

    while i + 16 <= n {
        let ptr = packed.as_ptr().add(i) as *const __m128i;
        let v = _mm_loadu_si128(ptr);

        let low_nibbles = _mm_and_si128(v, mask_0f);
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);

        let res_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
        let res_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

        let res_lo_lo = _mm_shuffle_epi8(table_lo, res_lo);
        let res_lo_hi = _mm_shuffle_epi8(table_hi, res_lo);
        let out_lo0 = _mm_unpacklo_epi8(res_lo_lo, res_lo_hi);
        let out_lo1 = _mm_unpackhi_epi8(res_lo_lo, res_lo_hi);

        let res_hi_lo = _mm_shuffle_epi8(table_lo, res_hi);
        let res_hi_hi = _mm_shuffle_epi8(table_hi, res_hi);
        let out_hi0 = _mm_unpacklo_epi8(res_hi_lo, res_hi_hi);
        let out_hi1 = _mm_unpackhi_epi8(res_hi_lo, res_hi_hi);

        _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i) as *mut _, out_lo0);
        _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i + 8) as *mut _, out_lo1);
        _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i + 16) as *mut _, out_hi0);
        _mm_storeu_si128(unpacked.as_mut_ptr().add(2 * i + 24) as *mut _, out_hi1);

        i += 16;
    }

    for j in i..n {
        let byte = packed[j];
        unpacked[2 * j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(byte & 0x0F)));
        unpacked[2 * j + 1] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits((byte >> 4) & 0x0F)));
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn unpack_f4_to_f32(packed: &[F4], unpacked: &mut [F32]) {
    let len = packed.len().min(unpacked.len());
    let mut i = 0;

    static TABLE_BITS: [u32; 16] = {
        let mut t = [0u32; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = f4_to_f32_bits(idx as u8);
            idx += 1;
        }
        t
    };

    while i + 8 <= len {
        let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
        let v_in = core::arch::x86_64::_mm_loadl_epi64(ptr);
        let v_u32 = core::arch::x86_64::_mm256_cvtepu8_epi32(v_in);
        let indices = core::arch::x86_64::_mm256_and_si256(
            v_u32,
            core::arch::x86_64::_mm256_set1_epi32(0x0f),
        );

        let result =
            core::arch::x86_64::_mm256_i32gather_ps(TABLE_BITS.as_ptr() as *const _, indices, 4);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
        core::arch::x86_64::_mm256_storeu_ps(out_ptr, result);
        i += 8;
    }

    for j in i..len {
        unpacked[j] = F32(f32::from_bits(TABLE_BITS[(packed[j].0 & 0x0f) as usize]));
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn unpack_f4_to_f32_packed(packed: &[u8], unpacked: &mut [F32]) {
    let len = packed.len();
    let n = len.min(unpacked.len() / 2);
    let mut i = 0;

    static TABLE_BITS: [u32; 16] = {
        let mut t = [0u32; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = f4_to_f32_bits(idx as u8);
            idx += 1;
        }
        t
    };

    while i + 8 <= n {
        let bytes_ptr = packed.as_ptr().add(i);
        let v_bytes = core::arch::x86_64::_mm_loadl_epi64(bytes_ptr as *const _);
        let v_u32 = core::arch::x86_64::_mm256_cvtepu8_epi32(v_bytes);

        let low_nibbles = core::arch::x86_64::_mm256_and_si256(
            v_u32,
            core::arch::x86_64::_mm256_set1_epi32(0x0f),
        );
        let high_nibbles = core::arch::x86_64::_mm256_and_si256(
            core::arch::x86_64::_mm256_srli_epi32(v_u32, 4),
            core::arch::x86_64::_mm256_set1_epi32(0x0f),
        );

        let val_low = core::arch::x86_64::_mm256_i32gather_ps(
            TABLE_BITS.as_ptr() as *const _,
            low_nibbles,
            4,
        );
        let val_high = core::arch::x86_64::_mm256_i32gather_ps(
            TABLE_BITS.as_ptr() as *const _,
            high_nibbles,
            4,
        );

        let val_lo_u = core::arch::x86_64::_mm256_castps_si256(val_low);
        let val_hi_u = core::arch::x86_64::_mm256_castps_si256(val_high);

        let res0 = core::arch::x86_64::_mm256_unpacklo_epi32(val_lo_u, val_hi_u);
        let res1 = core::arch::x86_64::_mm256_unpackhi_epi32(val_lo_u, val_hi_u);

        let out0 = core::arch::x86_64::_mm256_permute2x128_si256(res0, res1, 0x20);
        let out1 = core::arch::x86_64::_mm256_permute2x128_si256(res0, res1, 0x31);

        let out_ptr0 = unpacked.as_mut_ptr().add(2 * i) as *mut core::arch::x86_64::__m256i;
        core::arch::x86_64::_mm256_storeu_si256(out_ptr0, out0);
        let out_ptr1 = unpacked.as_mut_ptr().add(2 * i + 8) as *mut core::arch::x86_64::__m256i;
        core::arch::x86_64::_mm256_storeu_si256(out_ptr1, out1);

        i += 8;
    }

    for j in i..n {
        let byte = packed[j];
        unpacked[2 * j] = F32(f32::from_bits(TABLE_BITS[(byte & 0x0F) as usize]));
        unpacked[2 * j + 1] = F32(f32::from_bits(TABLE_BITS[((byte >> 4) & 0x0F) as usize]));
    }
}

#[target_feature(enable = "avx2")]
pub unsafe fn unpack_f8_to_f32(packed: &[F8], unpacked: &mut [F32]) {
    let len = packed.len().min(unpacked.len());
    let mut i = 0;

    static TABLE_BITS: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut idx = 0;
        while idx < 256 {
            t[idx] = f8_to_f32_bits(idx as u8);
            idx += 1;
        }
        t
    };

    while i + 8 <= len {
        let ptr = packed.as_ptr().add(i) as *const core::arch::x86_64::__m128i;
        let v_in = core::arch::x86_64::_mm_loadl_epi64(ptr);
        let v_u32 = core::arch::x86_64::_mm256_cvtepu8_epi32(v_in);

        let result =
            core::arch::x86_64::_mm256_i32gather_ps(TABLE_BITS.as_ptr() as *const _, v_u32, 4);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
        core::arch::x86_64::_mm256_storeu_ps(out_ptr, result);
        i += 8;
    }

    for j in i..len {
        unpacked[j] = F32(f32::from_bits(TABLE_BITS[packed[j].0 as usize]));
    }
}

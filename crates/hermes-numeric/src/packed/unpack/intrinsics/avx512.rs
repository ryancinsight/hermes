#![allow(dead_code)]
use super::super::conv::{bf4_to_bf16_bits, f4_to_f32_bits, f8_to_f32_bits};
use crate::types::{Bf16, Bf4, Bf8, F32, F4, F8};
use core::arch::x86_64::*;

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    let len = packed.len().min(unpacked.len());
    let mut i = 0;

    while i + 32 <= len {
        let ptr = packed.as_ptr().add(i) as *const __m256i;
        let v_in = _mm256_loadu_si256(ptr);
        let v_u16 = _mm512_cvtepu8_epi16(v_in);

        let sign = _mm512_slli_epi16(_mm512_and_si512(v_u16, _mm512_set1_epi16(0x80)), 8);
        let rest = _mm512_slli_epi16(_mm512_and_si512(v_u16, _mm512_set1_epi16(0x7f)), 5);

        let is_zero_mask = _mm512_cmpeq_epi16_mask(rest, _mm512_setzero_si512());
        let is_not_zero_mask = !is_zero_mask;

        let rest_biased =
            _mm512_mask_add_epi16(rest, is_not_zero_mask, rest, _mm512_set1_epi16(112 << 7));

        let result = _mm512_or_si512(sign, rest_biased);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut __m512i;
        _mm512_storeu_si512(out_ptr, result);

        i += 32;
    }

    for j in i..len {
        let b = packed[j].0 as u16;
        let sign = (b & 0x80) << 8;
        let rest = (b & 0x7f) << 5;
        let bias_diff = if rest == 0 { 0 } else { 112 << 7 };
        unpacked[j] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
    let len = packed.len().min(unpacked.len());
    let mut i = 0;

    static TABLE_BITS: [u16; 16] = {
        let mut t = [0u16; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = bf4_to_bf16_bits(idx as u8);
            idx += 1;
        }
        t
    };
    let table_zmm = core::mem::transmute::<[i16; 32], __m512i>([
        TABLE_BITS[0] as i16,
        TABLE_BITS[1] as i16,
        TABLE_BITS[2] as i16,
        TABLE_BITS[3] as i16,
        TABLE_BITS[4] as i16,
        TABLE_BITS[5] as i16,
        TABLE_BITS[6] as i16,
        TABLE_BITS[7] as i16,
        TABLE_BITS[8] as i16,
        TABLE_BITS[9] as i16,
        TABLE_BITS[10] as i16,
        TABLE_BITS[11] as i16,
        TABLE_BITS[12] as i16,
        TABLE_BITS[13] as i16,
        TABLE_BITS[14] as i16,
        TABLE_BITS[15] as i16,
        TABLE_BITS[0] as i16,
        TABLE_BITS[1] as i16,
        TABLE_BITS[2] as i16,
        TABLE_BITS[3] as i16,
        TABLE_BITS[4] as i16,
        TABLE_BITS[5] as i16,
        TABLE_BITS[6] as i16,
        TABLE_BITS[7] as i16,
        TABLE_BITS[8] as i16,
        TABLE_BITS[9] as i16,
        TABLE_BITS[10] as i16,
        TABLE_BITS[11] as i16,
        TABLE_BITS[12] as i16,
        TABLE_BITS[13] as i16,
        TABLE_BITS[14] as i16,
        TABLE_BITS[15] as i16,
    ]);

    while i + 32 <= len {
        let ptr = packed.as_ptr().add(i) as *const __m256i;
        let v_in = _mm256_loadu_si256(ptr);
        let v_u16 = _mm512_cvtepu8_epi16(v_in);
        let indices = _mm512_and_si512(v_u16, _mm512_set1_epi16(0x0f));

        let result = _mm512_permutexvar_epi16(indices, table_zmm);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut __m512i;
        _mm512_storeu_si512(out_ptr, result);

        i += 32;
    }

    for j in i..len {
        let b = packed[j].0;
        unpacked[j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b)));
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn unpack_bf4_to_bf16_packed(packed: &[u8], unpacked: &mut [Bf16]) {
    let len = packed.len();
    let n = len.min(unpacked.len() / 2);
    let mut i = 0;

    static TABLE_BITS: [u16; 16] = {
        let mut t = [0u16; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = bf4_to_bf16_bits(idx as u8);
            idx += 1;
        }
        t
    };

    let table_ymm = _mm256_loadu_si256(TABLE_BITS.as_ptr() as *const _);
    let mask_0f = _mm_set1_epi8(0x0F);

    while i + 16 <= n {
        let v = _mm_loadu_si128(packed.as_ptr().add(i) as *const _);

        let low_nibbles = _mm_and_si128(v, mask_0f);
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);

        let res_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
        let res_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

        let idx_lo = _mm256_cvtepu8_epi16(res_lo);
        let idx_hi = _mm256_cvtepu8_epi16(res_hi);

        let val_lo = _mm256_permutexvar_epi16(idx_lo, table_ymm);
        let val_hi = _mm256_permutexvar_epi16(idx_hi, table_ymm);

        _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i) as *mut _, val_lo);
        _mm256_storeu_si256(unpacked.as_mut_ptr().add(2 * i + 16) as *mut _, val_hi);

        i += 16;
    }

    static TABLE: [Bf16; 16] = {
        let mut t = [Bf16(half::bf16::ZERO); 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = Bf16(half::bf16::from_bits(TABLE_BITS[idx]));
            idx += 1;
        }
        t
    };
    for j in i..n {
        let byte = packed[j];
        unpacked[2 * j] = TABLE[(byte & 0x0F) as usize];
        unpacked[2 * j + 1] = TABLE[((byte >> 4) & 0x0F) as usize];
    }
}

#[target_feature(enable = "avx512f,avx512vl")]
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

    let table_zmm = _mm512_loadu_si512(TABLE_BITS.as_ptr() as *const _);

    while i + 16 <= len {
        let ptr = packed.as_ptr().add(i) as *const __m128i;
        let v_in = _mm_loadu_si128(ptr);
        let v_u32 = _mm512_cvtepu8_epi32(v_in);
        let indices = _mm512_and_si512(v_u32, _mm512_set1_epi32(0x0f));

        let result = _mm512_permutexvar_epi32(indices, table_zmm);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
        _mm512_storeu_si512(out_ptr as *mut _, result);
        i += 16;
    }

    for j in i..len {
        unpacked[j] = F32(f32::from_bits(TABLE_BITS[(packed[j].0 & 0x0f) as usize]));
    }
}

#[target_feature(enable = "avx512f,avx512vl")]
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

    let table_zmm = _mm512_loadu_si512(TABLE_BITS.as_ptr() as *const _);
    let mask_0f = _mm_set1_epi8(0x0F);

    while i + 16 <= n {
        let v = _mm_loadu_si128(packed.as_ptr().add(i) as *const _);

        let low_nibbles = _mm_and_si128(v, mask_0f);
        let high_nibbles = _mm_and_si128(_mm_srli_epi16(v, 4), mask_0f);

        let res_lo = _mm_unpacklo_epi8(low_nibbles, high_nibbles);
        let res_hi = _mm_unpackhi_epi8(low_nibbles, high_nibbles);

        let idx_lo = _mm512_cvtepu8_epi32(res_lo);
        let idx_hi = _mm512_cvtepu8_epi32(res_hi);

        let val_lo = _mm512_permutexvar_epi32(idx_lo, table_zmm);
        let val_hi = _mm512_permutexvar_epi32(idx_hi, table_zmm);

        _mm512_storeu_si512(unpacked.as_mut_ptr().add(2 * i) as *mut _, val_lo);
        _mm512_storeu_si512(unpacked.as_mut_ptr().add(2 * i + 16) as *mut _, val_hi);

        i += 16;
    }

    static TABLE: [F32; 16] = {
        let mut t = [F32(0.0); 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = F32(f32::from_bits(TABLE_BITS[idx]));
            idx += 1;
        }
        t
    };
    for j in i..n {
        let byte = packed[j];
        unpacked[2 * j] = TABLE[(byte & 0x0F) as usize];
        unpacked[2 * j + 1] = TABLE[((byte >> 4) & 0x0F) as usize];
    }
}

#[target_feature(enable = "avx512f,avx512vl")]
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

    while i + 16 <= len {
        let ptr = packed.as_ptr().add(i) as *const __m128i;
        let v_in = _mm_loadu_si128(ptr);
        let v_u32 = _mm512_cvtepu8_epi32(v_in);

        let result = _mm512_i32gather_ps(v_u32, TABLE_BITS.as_ptr() as *const _, 4);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut f32;
        _mm512_storeu_ps(out_ptr, result);
        i += 16;
    }

    for j in i..len {
        unpacked[j] = F32(f32::from_bits(TABLE_BITS[packed[j].0 as usize]));
    }
}

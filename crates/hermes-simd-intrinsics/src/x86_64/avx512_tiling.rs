//! AVX-512 specific implementations of TileMatrixMultiply for x86_64.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use hermes_simd_core::view::TileMatrixMultiply;
use crate::Avx512;
use hermes_numeric::{Bf16, F32, Bf8, Bf4, I8, I32};

// ---------------------------------------------------------------------------
// AVX-512 Implementations
// ---------------------------------------------------------------------------

impl TileMatrixMultiply<half::bf16, half::bf16, f32, Avx512, Avx512, 16, 16, 32> for Avx512 {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn tile_matmul(
        c: *mut f32, c_stride: usize,
        a: *const half::bf16, a_stride: usize,
        b: *const half::bf16, b_stride: usize,
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
        c: *mut F32, c_stride: usize,
        a: *const Bf16, a_stride: usize,
        b: *const Bf16, b_stride: usize,
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
        c: *mut i32, c_stride: usize,
        a: *const i8, a_stride: usize,
        b: *const i8, b_stride: usize,
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
                    _mm512_inserti32x4(
                        _mm512_castsi128_si512(packed_lo),
                        packed_mid_lo,
                        1,
                    ),
                    packed_mid_hi,
                    2,
                ),
                packed_hi,
                3,
            );

            for i in 0..16 {
                let a_val = core::ptr::read_unaligned(a.add(i * a_stride + k) as *const i32);
                let a_vec = _mm512_set1_epi32(a_val);
                let mut dst = c_regs[i];
                core::arch::asm!(
                    "vpdpbssd {dst}, {src1}, {src2}",
                    dst = inout(zmm_reg) dst,
                    src1 = in(zmm_reg) a_vec,
                    src2 = in(zmm_reg) b_vec,
                );
                c_regs[i] = dst;
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
        c: *mut I32, c_stride: usize,
        a: *const I8, a_stride: usize,
        b: *const I8, b_stride: usize,
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
                    _mm512_inserti32x4(
                        _mm512_castsi128_si512(packed_lo),
                        packed_mid_lo,
                        1,
                    ),
                    packed_mid_hi,
                    2,
                ),
                packed_hi,
                3,
            );

            for i in 0..16 {
                let a_val = core::ptr::read_unaligned(a.add(i * a_stride + k) as *const i32);
                let a_vec = _mm512_set1_epi32(a_val);
                let mut dst = c_regs[i];
                core::arch::asm!(
                    "vpdpbssd {dst}, {src1}, {src2}",
                    dst = inout(zmm_reg) dst,
                    src1 = in(zmm_reg) a_vec,
                    src2 = in(zmm_reg) b_vec,
                );
                c_regs[i] = dst;
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
    
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            unsafe {
                unpack_int4_avx2(packed, unpacked);
                return;
            }
        }
    }
    
    // Fallback scalar loop
    for i in 0..len {
        let byte = packed[i] as i8;
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
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5, 6, 7,
        -8, -7, -6, -5, -4, -3, -2, -1,
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
    hermes_numeric::unpack_bf8_to_bf16(packed, unpacked);
}

/// Unpacks Bf4 elements to Bf16 for accumulation.
#[inline]
pub fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
    hermes_numeric::unpack_bf4_to_bf16(packed, unpacked);
}

const fn bf4_to_bf16_bits(bits: u8) -> u16 {
    let bits = bits & 0x0F;
    let sign = (bits & 0x08) as u32;
    let exp = (bits & 0x06) >> 1;
    let mant = bits & 0x01;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 28
        } else {
            if sign != 0 { 0xBE00_0000 } else { 0x3E00_0000 }
        }
    } else if exp == 3 {
        if sign != 0 { 0xFFC0_0000 } else { 0x7FC0_0000 }
    } else {
        let f32_exp = (exp as u32 + 127 - 1) << 23;
        let f32_mant = (mant as u32) << 22;
        (sign << 28) | f32_exp | f32_mant
    };
    (f32_bits >> 16) as u16
}

/// Unpacks packed Bf4 elements (stored 2 per byte in `packed`) into a Bf16 slice.
#[inline]
pub fn unpack_packed_bf4_to_bf16(packed: &[u8], unpacked: &mut [Bf16]) {
    let len = packed.len();
    assert!(unpacked.len() >= len * 2);
    
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx512bw")]
        {
            if std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vl") {
                unsafe {
                    unpack_packed_bf4_to_bf16_avx512(packed, unpacked);
                    return;
                }
            }
        }
    }

    // Fallback scalar loop
    static TABLE: [Bf16; 16] = {
        let mut t = [Bf16(half::bf16::ZERO); 16];
        let mut i = 0;
        while i < 16 {
            t[i] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(i as u8)));
            i += 1;
        }
        t
    };
    for i in 0..len {
        let byte = packed[i];
        unpacked[2 * i] = TABLE[(byte & 0x0F) as usize];
        unpacked[2 * i + 1] = TABLE[((byte >> 4) & 0x0F) as usize];
    }
}

/// AVX-512 optimized packed Bf4 unpacker.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub unsafe fn unpack_packed_bf4_to_bf16_avx512(packed: &[u8], unpacked: &mut [Bf16]) {
    use core::arch::x86_64::*;
    
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

const fn f4_to_f32_bits(bits: u8) -> u32 {
    let bits = bits & 0x0F;
    let sign = (bits & 0x08) as u32;
    let exp = bits & 0x07;
    if exp == 0 {
        sign << 28
    } else if exp == 7 {
        0x7FC0_0000
    } else {
        let f32_exp = (exp as u32 + 127 - 3) << 23;
        (sign << 28) | f32_exp
    }
}

/// Unpacks packed F4 elements (stored 2 per byte in `packed`) into an F32 slice.
#[inline]
pub fn unpack_packed_f4_to_f32(packed: &[u8], unpacked: &mut [F32]) {
    let len = packed.len();
    assert!(unpacked.len() >= len * 2);
    
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx512f")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
                unsafe {
                    unpack_packed_f4_to_f32_avx512(packed, unpacked);
                    return;
                }
            }
        }
    }

    // Fallback scalar loop
    static TABLE: [F32; 16] = {
        let mut t = [F32(0.0); 16];
        let mut i = 0;
        while i < 16 {
            t[i] = F32(f32::from_bits(f4_to_f32_bits(i as u8)));
            i += 1;
        }
        t
    };
    for i in 0..len {
        let byte = packed[i];
        unpacked[2 * i] = TABLE[(byte & 0x0F) as usize];
        unpacked[2 * i + 1] = TABLE[((byte >> 4) & 0x0F) as usize];
    }
}

/// AVX-512 optimized packed F4 unpacker.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn unpack_packed_f4_to_f32_avx512(packed: &[u8], unpacked: &mut [F32]) {
    use core::arch::x86_64::*;
    
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


#![allow(dead_code)]
use super::super::conv::{bf4_to_bf16_bits, f4_to_f32_bits, f8_to_f32_bits};
use crate::types::{Bf16, Bf4, Bf8, F32, F4, F8};
use core::arch::aarch64::*;

#[inline]
pub unsafe fn unpack_bf8_to_bf16(packed: &[Bf8], unpacked: &mut [Bf16]) {
    let len = packed.len().min(unpacked.len());
    let mut i = 0;

    let mask_80 = vdupq_n_u16(0x80);
    let mask_7f = vdupq_n_u16(0x7F);
    let bias_const = vdupq_n_u16(112 << 7);
    let zero_const = vdupq_n_u16(0);

    while i + 8 <= len {
        let ptr = packed.as_ptr().add(i) as *const u8;
        let v_u8 = vld1_u8(ptr);
        let v_u16 = vmovl_u8(v_u8);

        let sign = vshlq_n_u16(vandq_u16(v_u16, mask_80), 8);
        let rest = vshlq_n_u16(vandq_u16(v_u16, mask_7f), 5);

        let is_zero = vceqq_u16(rest, zero_const);
        let bias_diff = vbslq_u16(is_zero, zero_const, bias_const);

        let rest_biased = vaddq_u16(rest, bias_diff);
        let result = vorrq_u16(sign, rest_biased);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut u8;
        vst1q_u8(out_ptr, vreinterpretq_u8_u16(result));

        i += 8;
    }
    for j in i..len {
        let b = packed[j].0 as u16;
        let sign = (b & 0x80) << 8;
        let rest = (b & 0x7f) << 5;
        let bias_diff = if rest == 0 { 0 } else { 112 << 7 };
        unpacked[j] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
    }
}

#[inline]
pub unsafe fn unpack_bf4_to_bf16(packed: &[Bf4], unpacked: &mut [Bf16]) {
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

    let table_lo = vld1q_u8(TABLE_LO.as_ptr());
    let table_hi = vld1q_u8(TABLE_HI.as_ptr());
    let mask_0f = vdupq_n_u8(0x0F);

    while i + 16 <= len {
        let ptr = packed.as_ptr().add(i) as *const u8;
        let v_in = vld1q_u8(ptr);
        let indices = vandq_u8(v_in, mask_0f);

        let res_lo = vqtbl1q_u8(table_lo, indices);
        let res_hi = vqtbl1q_u8(table_hi, indices);

        let zipped = vzipq_u8(res_lo, res_hi);
        let out_lo = zipped.0;
        let out_hi = zipped.1;

        vst1q_u8(unpacked.as_mut_ptr().add(i) as *mut u8, out_lo);
        vst1q_u8(unpacked.as_mut_ptr().add(i + 8) as *mut u8, out_hi);

        i += 16;
    }
    for j in i..len {
        let b = packed[j].0;
        unpacked[j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(b)));
    }
}

#[inline]
pub unsafe fn unpack_bf4_to_bf16_packed(packed: &[u8], unpacked: &mut [Bf16]) {
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

    let table_lo = vld1q_u8(TABLE_LO.as_ptr());
    let table_hi = vld1q_u8(TABLE_HI.as_ptr());
    let mask_0f = vdupq_n_u8(0x0F);

    while i + 16 <= n {
        let ptr = packed.as_ptr().add(i);
        let v = vld1q_u8(ptr);

        let low_nibbles = vandq_u8(v, mask_0f);
        let high_nibbles = vandq_u8(vshrq_n_u8(v, 4), mask_0f);

        let zipped_nibbles = vzipq_u8(low_nibbles, high_nibbles);
        let res_lo = zipped_nibbles.0;
        let res_hi = zipped_nibbles.1;

        let res_lo_lo = vqtbl1q_u8(table_lo, res_lo);
        let res_lo_hi = vqtbl1q_u8(table_hi, res_lo);
        let zipped_out_lo = vzipq_u8(res_lo_lo, res_lo_hi);
        let out_lo0 = zipped_out_lo.0;
        let out_lo1 = zipped_out_lo.1;

        let res_hi_lo = vqtbl1q_u8(table_lo, res_hi);
        let res_hi_hi = vqtbl1q_u8(table_hi, res_hi);
        let zipped_out_hi = vzipq_u8(res_hi_lo, res_hi_hi);
        let out_hi0 = zipped_out_hi.0;
        let out_hi1 = zipped_out_hi.1;

        vst1q_u8(unpacked.as_mut_ptr().add(2 * i) as *mut u8, out_lo0);
        vst1q_u8(unpacked.as_mut_ptr().add(2 * i + 8) as *mut u8, out_lo1);
        vst1q_u8(unpacked.as_mut_ptr().add(2 * i + 16) as *mut u8, out_hi0);
        vst1q_u8(unpacked.as_mut_ptr().add(2 * i + 24) as *mut u8, out_hi1);

        i += 16;
    }
    for j in i..n {
        let byte = packed[j];
        unpacked[2 * j] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits(byte & 0x0F)));
        unpacked[2 * j + 1] = Bf16(half::bf16::from_bits(bf4_to_bf16_bits((byte >> 4) & 0x0F)));
    }
}

#[inline]
pub unsafe fn unpack_f4_to_f32(packed: &[F4], unpacked: &mut [F32]) {
    let len = packed.len().min(unpacked.len());
    let mut i = 0;

    static TABLE_B0: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = (f4_to_f32_bits(idx as u8) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_B1: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = ((f4_to_f32_bits(idx as u8) >> 8) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_B2: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = ((f4_to_f32_bits(idx as u8) >> 16) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_B3: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = ((f4_to_f32_bits(idx as u8) >> 24) & 0xFF) as u8;
            idx += 1;
        }
        t
    };

    let table_b0 = vld1q_u8(TABLE_B0.as_ptr());
    let table_b1 = vld1q_u8(TABLE_B1.as_ptr());
    let table_b2 = vld1q_u8(TABLE_B2.as_ptr());
    let table_b3 = vld1q_u8(TABLE_B3.as_ptr());
    let mask_0f = vdupq_n_u8(0x0F);

    while i + 16 <= len {
        let ptr = packed.as_ptr().add(i) as *const u8;
        let v_in = vld1q_u8(ptr);
        let indices = vandq_u8(v_in, mask_0f);

        let res_b0 = vqtbl1q_u8(table_b0, indices);
        let res_b1 = vqtbl1q_u8(table_b1, indices);
        let res_b2 = vqtbl1q_u8(table_b2, indices);
        let res_b3 = vqtbl1q_u8(table_b3, indices);

        let out_ptr = unpacked.as_mut_ptr().add(i) as *mut u8;
        vst4q_u8(out_ptr, uint8x16x4_t(res_b0, res_b1, res_b2, res_b3));

        i += 16;
    }
    for j in i..len {
        unpacked[j] = F32(f32::from_bits(f4_to_f32_bits(packed[j].0)));
    }
}

#[inline]
pub unsafe fn unpack_f4_to_f32_packed(packed: &[u8], unpacked: &mut [F32]) {
    let len = packed.len();
    let n = len.min(unpacked.len() / 2);
    let mut i = 0;

    static TABLE_B0: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = (f4_to_f32_bits(idx as u8) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_B1: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = ((f4_to_f32_bits(idx as u8) >> 8) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_B2: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = ((f4_to_f32_bits(idx as u8) >> 16) & 0xFF) as u8;
            idx += 1;
        }
        t
    };
    static TABLE_B3: [u8; 16] = {
        let mut t = [0u8; 16];
        let mut idx = 0;
        while idx < 16 {
            t[idx] = ((f4_to_f32_bits(idx as u8) >> 24) & 0xFF) as u8;
            idx += 1;
        }
        t
    };

    let table_b0 = vld1q_u8(TABLE_B0.as_ptr());
    let table_b1 = vld1q_u8(TABLE_B1.as_ptr());
    let table_b2 = vld1q_u8(TABLE_B2.as_ptr());
    let table_b3 = vld1q_u8(TABLE_B3.as_ptr());
    let mask_0f_8 = vdup_n_u8(0x0F);

    while i + 8 <= n {
        let bytes_ptr = packed.as_ptr().add(i);
        let v_8 = vld1_u8(bytes_ptr);

        let low_nibbles = vand_u8(v_8, mask_0f_8);
        let high_nibbles = vand_u8(vshr_n_u8(v_8, 4), mask_0f_8);

        let zipped = vzip_u8(low_nibbles, high_nibbles);
        let indices = vcombine_u8(zipped.0, zipped.1);

        let res_b0 = vqtbl1q_u8(table_b0, indices);
        let res_b1 = vqtbl1q_u8(table_b1, indices);
        let res_b2 = vqtbl1q_u8(table_b2, indices);
        let res_b3 = vqtbl1q_u8(table_b3, indices);

        let out_ptr = unpacked.as_mut_ptr().add(2 * i) as *mut u8;
        vst4q_u8(out_ptr, uint8x16x4_t(res_b0, res_b1, res_b2, res_b3));

        i += 8;
    }
    for j in i..n {
        let byte = packed[j];
        unpacked[2 * j] = F32(f32::from_bits(f4_to_f32_bits(byte & 0x0F)));
        unpacked[2 * j + 1] = F32(f32::from_bits(f4_to_f32_bits((byte >> 4) & 0x0F)));
    }
}

#[inline]
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

    while i + 4 <= len {
        let idx0 = packed[i].0 as usize;
        let idx1 = packed[i + 1].0 as usize;
        let idx2 = packed[i + 2].0 as usize;
        let idx3 = packed[i + 3].0 as usize;

        let val0 = TABLE_BITS[idx0];
        let val1 = TABLE_BITS[idx1];
        let val2 = TABLE_BITS[idx2];
        let val3 = TABLE_BITS[idx3];

        unpacked[i] = F32(f32::from_bits(val0));
        unpacked[i + 1] = F32(f32::from_bits(val1));
        unpacked[i + 2] = F32(f32::from_bits(val2));
        unpacked[i + 3] = F32(f32::from_bits(val3));

        i += 4;
    }
    for j in i..len {
        unpacked[j] = F32(f32::from_bits(TABLE_BITS[packed[j].0 as usize]));
    }
}

//! Bit-level conversion helpers for low-precision formats.

pub(crate) const fn f8_to_f32_bits(bits: u8) -> u32 {
    let sign = (bits & 0x80) as u32;
    let exp = (bits & 0x78) >> 3;
    let mant = bits & 0x07;
    if exp == 0 {
        if mant == 0 {
            sign << 24
        } else {
            if mant >= 4 {
                (sign << 24) | (120 << 23) | (((mant - 4) as u32) << 21)
            } else if mant >= 2 {
                (sign << 24) | (119 << 23) | (((mant - 2) as u32) << 22)
            } else {
                (sign << 24) | (118 << 23)
            }
        }
    } else if exp == 0x0F {
        0x7FC0_0000 | (sign << 24)
    } else {
        let f32_exp = (exp as u32 + 127 - 7) << 23;
        let f32_mant = (mant as u32) << 20;
        (sign << 24) | f32_exp | f32_mant
    }
}

#[allow(dead_code)]
pub(crate) const fn f4_to_f32_bits(bits: u8) -> u32 {
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

#[allow(dead_code)]
pub(crate) const fn bf4_to_bf16_bits(bits: u8) -> u16 {
    let bits = bits & 0x0F;
    let sign = (bits & 0x08) as u32;
    let exp = (bits & 0x06) >> 1;
    let mant = bits & 0x01;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 28
        } else {
            if sign != 0 {
                0xBE00_0000
            } else {
                0x3E00_0000
            }
        }
    } else if exp == 3 {
        if sign != 0 {
            0xFFC0_0000
        } else {
            0x7FC0_0000
        }
    } else {
        let f32_exp = (exp as u32 + 127 - 1) << 23;
        let f32_mant = (mant as u32) << 22;
        (sign << 28) | f32_exp | f32_mant
    };
    (f32_bits >> 16) as u16
}

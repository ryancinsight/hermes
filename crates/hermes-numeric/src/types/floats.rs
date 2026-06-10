/// Transparent wrapper for half::f16.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct F16(pub half::f16);

/// Transparent wrapper for f32.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct F32(pub f32);

/// Transparent wrapper for f64.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct F64(pub f64);

/// Transparent wrapper for half::bf16.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct Bf16(pub half::bf16);

/// Brain Float 8: E5M2 representation.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct Bf8(pub u8);

/// Brain Float 4: E2M1 representation.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct Bf4(pub u8);

/// IEEE-style 8-bit Float: E4M3 representation.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct F8(pub u8);

/// IEEE-style 4-bit Float: E3M0 representation.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd, Debug)]
#[repr(transparent)]
pub struct F4(pub u8);

impl Bf8 {
    /// Convert to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = (bits & 0x80) as u32;
        let exp = (bits & 0x7C) >> 2;
        let mant = bits & 0x03;
        if exp == 0 {
            if mant == 0 {
                f32::from_bits(sign << 24)
            } else {
                let val = (mant as f32) * (1.0 / 16384.0);
                if sign != 0 {
                    -val
                } else {
                    val
                }
            }
        } else if exp == 0x1F {
            if mant == 0 {
                if sign != 0 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            }
        } else {
            let f32_exp = (exp as u32 + 127 - 15) << 23;
            let f32_mant = (mant as u32) << 21;
            f32::from_bits((sign << 24) | f32_exp | f32_mant)
        }
    }

    /// Convert from f32.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let f32_bits = val.to_bits();
        let sign = (f32_bits >> 24) & 0x80;
        if val.is_nan() {
            return Self((sign | 0x7E) as u8);
        }
        if val.is_infinite() {
            return Self((sign | 0x7C) as u8);
        }
        let abs_val = val.abs();
        if abs_val == 0.0 {
            return Self(sign as u8);
        }
        let f32_exp = ((f32_bits >> 23) & 0xFF) as i32;
        let f32_mant = f32_bits & 0x007F_FFFF;
        let exp = f32_exp - 127 + 15;
        if exp <= 0 {
            if exp < -2 {
                Self(sign as u8)
            } else {
                let shift = 1 - exp;
                let mant = (f32_mant | 0x0080_0000) >> (21 + shift);
                Self((sign | mant) as u8)
            }
        } else if exp >= 31 {
            Self((sign | 0x7C) as u8)
        } else {
            let mant = (f32_mant >> 21) & 0x03;
            Self((sign | ((exp as u32) << 2) | mant) as u8)
        }
    }
}

impl Bf4 {
    /// Convert to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0 & 0x0F;
        let sign = (bits & 0x08) as u32;
        let exp = (bits & 0x06) >> 1;
        let mant = bits & 0x01;
        if exp == 0 {
            if mant == 0 {
                f32::from_bits(sign << 28)
            } else {
                let val = (mant as f32) * 0.125;
                if sign != 0 {
                    -val
                } else {
                    val
                }
            }
        } else if exp == 3 {
            f32::NAN
        } else {
            let f32_exp = (exp as u32 + 127 - 1) << 23;
            let f32_mant = (mant as u32) << 22;
            f32::from_bits((sign << 28) | f32_exp | f32_mant)
        }
    }

    /// Convert from f32.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let f32_bits = val.to_bits();
        let sign = (f32_bits >> 28) & 0x08;
        if val.is_nan() {
            return Self((sign | 0x07) as u8);
        }
        if val.is_infinite() {
            return Self((sign | 0x06) as u8);
        }
        let abs_val = val.abs();
        if abs_val == 0.0 {
            return Self(sign as u8);
        }
        let f32_exp = ((f32_bits >> 23) & 0xFF) as i32;
        let f32_mant = f32_bits & 0x007F_FFFF;
        let exp = f32_exp - 127 + 1;
        if exp <= 0 {
            if exp < -1 {
                Self(sign as u8)
            } else {
                let shift = 1 - exp;
                let mant = (f32_mant | 0x0080_0000) >> (22 + shift);
                Self((sign | mant) as u8)
            }
        } else if exp >= 3 {
            Self((sign | 0x06) as u8)
        } else {
            let mant = (f32_mant >> 22) & 0x01;
            Self((sign | ((exp as u32) << 1) | mant) as u8)
        }
    }

    /// Pack two Bf4 values into one byte.
    #[inline]
    pub fn pack_pair(low: Self, high: Self) -> u8 {
        (low.0 & 0x0F) | ((high.0 & 0x0F) << 4)
    }

    /// Unpack one byte into two Bf4 values.
    #[inline]
    pub fn unpack_pair(packed: u8) -> (Self, Self) {
        (Self(packed & 0x0F), Self((packed >> 4) & 0x0F))
    }
}

impl F8 {
    /// Convert to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0;
        let sign = (bits & 0x80) as u32;
        let exp = (bits & 0x78) >> 3;
        let mant = bits & 0x07;
        if exp == 0 {
            if mant == 0 {
                f32::from_bits(sign << 24)
            } else {
                let val = (mant as f32) * (1.0 / 512.0);
                if sign != 0 {
                    -val
                } else {
                    val
                }
            }
        } else if exp == 0x0F {
            f32::NAN
        } else {
            let f32_exp = (exp as u32 + 127 - 7) << 23;
            let f32_mant = (mant as u32) << 20;
            f32::from_bits((sign << 24) | f32_exp | f32_mant)
        }
    }

    /// Convert from f32.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let f32_bits = val.to_bits();
        let sign = (f32_bits >> 24) & 0x80;
        if val.is_nan() {
            return Self((sign | 0x7F) as u8);
        }
        if val.is_infinite() {
            return Self((sign | 0x77) as u8);
        }
        let abs_val = val.abs();
        if abs_val == 0.0 {
            return Self(sign as u8);
        }
        let f32_exp = ((f32_bits >> 23) & 0xFF) as i32;
        let f32_mant = f32_bits & 0x007F_FFFF;
        let exp = f32_exp - 127 + 7;
        if exp <= 0 {
            if exp < -3 {
                Self(sign as u8)
            } else {
                let shift = 1 - exp;
                let mant = (f32_mant | 0x0080_0000) >> (20 + shift);
                Self((sign | mant) as u8)
            }
        } else if exp >= 15 {
            Self((sign | 0x77) as u8)
        } else {
            let mant = (f32_mant >> 20) & 0x07;
            Self((sign | ((exp as u32) << 3) | mant) as u8)
        }
    }
}

impl F4 {
    /// Convert to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let bits = self.0 & 0x0F;
        let sign = (bits & 0x08) as u32;
        let exp = bits & 0x07;
        if exp == 0 {
            f32::from_bits(sign << 28)
        } else if exp == 7 {
            f32::NAN
        } else {
            let f32_exp = (exp as u32 + 127 - 3) << 23;
            f32::from_bits((sign << 28) | f32_exp)
        }
    }

    /// Convert from f32.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        let f32_bits = val.to_bits();
        let sign = (f32_bits >> 28) & 0x08;
        if val.is_nan() {
            return Self((sign | 0x07) as u8);
        }
        if val.is_infinite() {
            return Self((sign | 0x06) as u8);
        }
        let abs_val = val.abs();
        if abs_val == 0.0 {
            return Self(sign as u8);
        }
        let f32_exp = ((f32_bits >> 23) & 0xFF) as i32;
        let exp = f32_exp - 127 + 3;
        if exp <= 0 {
            Self(sign as u8)
        } else if exp >= 7 {
            Self((sign | 0x06) as u8)
        } else {
            Self((sign | (exp as u32)) as u8)
        }
    }

    /// Pack two F4 values into one byte.
    #[inline]
    pub fn pack_pair(low: Self, high: Self) -> u8 {
        (low.0 & 0x0F) | ((high.0 & 0x0F) << 4)
    }

    /// Unpack one byte into two F4 values.
    #[inline]
    pub fn unpack_pair(packed: u8) -> (Self, Self) {
        (Self(packed & 0x0F), Self((packed >> 4) & 0x0F))
    }
}

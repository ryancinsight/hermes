//! Modular-arithmetic kernels for transform workloads.
//!
//! These kernels keep exact residue-field arithmetic in Hermes instead of
//! forcing downstream crates to duplicate butterfly loops around SIMD-facing
//! provider boundaries. Multiplication uses `u128` widening because the
//! numerical contract is exact modular arithmetic, not wrapping arithmetic.

use hermes_simd_core::view::SimdError;

/// Executes one radix-2 NTT butterfly stage in place.
///
/// `data` is partitioned into chunks of `stage_len`; `twiddles` contains one
/// stage twiddle per element in the right half of each chunk. Each butterfly
/// computes:
///
/// ```text
/// left'  = left + right * twiddle (mod modulus)
/// right' = left - right * twiddle (mod modulus)
/// ```
///
/// # Errors
///
/// Returns [`SimdError::LengthMismatch`] when `stage_len` is zero or odd, the
/// data length is not divisible by `stage_len`, or the twiddle count is not
/// `stage_len / 2`. Returns [`SimdError::InvalidModulus`] when `modulus` is
/// zero.
#[inline]
pub fn ntt_butterfly_stage_u64(
    data: &mut [u64],
    stage_len: usize,
    twiddles: &[u64],
    modulus: u64,
) -> Result<(), SimdError> {
    if modulus == 0 {
        return Err(SimdError::InvalidModulus);
    }
    if stage_len == 0
        || !stage_len.is_multiple_of(2)
        || !data.len().is_multiple_of(stage_len)
        || twiddles.len() != stage_len / 2
    {
        return Err(SimdError::LengthMismatch);
    }

    let half = stage_len / 2;
    for chunk in data.chunks_mut(stage_len) {
        let (left, right) = chunk.split_at_mut(half);
        for index in 0..half {
            let lhs = left[index] % modulus;
            let rhs = mod_mul_u64(right[index], twiddles[index], modulus);
            left[index] = mod_add_u64(lhs, rhs, modulus);
            right[index] = mod_sub_u64(lhs, rhs, modulus);
        }
    }
    Ok(())
}

#[inline]
fn mod_mul_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 * rhs as u128) % modulus as u128) as u64
}

#[inline]
fn mod_add_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((lhs as u128 + rhs as u128) % modulus as u128) as u64
}

#[inline]
fn mod_sub_u64(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        modulus - (rhs - lhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_modulus_before_arithmetic() {
        assert_eq!(
            ntt_butterfly_stage_u64(&mut [1, 2], 2, &[1], 0),
            Err(SimdError::InvalidModulus)
        );
    }

    #[test]
    fn rejects_invalid_stage_shapes() {
        let mut data = [1, 2, 3, 4];
        assert_eq!(
            ntt_butterfly_stage_u64(&mut data, 0, &[], 17),
            Err(SimdError::LengthMismatch)
        );
        assert_eq!(
            ntt_butterfly_stage_u64(&mut data, 3, &[1], 17),
            Err(SimdError::LengthMismatch)
        );
        assert_eq!(
            ntt_butterfly_stage_u64(&mut data, 2, &[], 17),
            Err(SimdError::LengthMismatch)
        );
    }

    #[test]
    fn reduces_noncanonical_residues_before_subtraction() {
        let mut data = [5, 7];
        ntt_butterfly_stage_u64(&mut data, 2, &[3], 5).expect("valid NTT shape");
        assert_eq!(data, [1, 4]);
    }
}

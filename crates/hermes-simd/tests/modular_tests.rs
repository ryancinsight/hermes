use hermes_simd::{ntt_butterfly_stage_u64, SimdError};

const MODULUS: u64 = 998_244_353;

fn mod_mul(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((u128::from(lhs) * u128::from(rhs)) % u128::from(modulus)) as u64
}

fn mod_add(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((u128::from(lhs) + u128::from(rhs)) % u128::from(modulus)) as u64
}

fn mod_sub(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    if lhs >= rhs {
        lhs - rhs
    } else {
        modulus - (rhs - lhs)
    }
}

#[test]
fn ntt_butterfly_stage_matches_exact_reference() {
    let mut actual = [1, 2, 3, 4, 5, 6, 7, 8];
    let mut expected = actual;
    let twiddles = [1, 911_660_635];

    for chunk in expected.chunks_mut(4) {
        let (left, right) = chunk.split_at_mut(2);
        for index in 0..2 {
            let lhs = left[index];
            let rhs = mod_mul(right[index], twiddles[index], MODULUS);
            left[index] = mod_add(lhs, rhs, MODULUS);
            right[index] = mod_sub(lhs, rhs, MODULUS);
        }
    }

    ntt_butterfly_stage_u64(&mut actual, 4, &twiddles, MODULUS).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn ntt_butterfly_stage_uses_widened_multiplication() {
    let mut actual = [MODULUS - 1, MODULUS - 2];
    let twiddles = [MODULUS - 1];

    ntt_butterfly_stage_u64(&mut actual, 2, &twiddles, MODULUS).unwrap();

    let rhs = mod_mul(MODULUS - 2, MODULUS - 1, MODULUS);
    assert_eq!(
        actual,
        [
            mod_add(MODULUS - 1, rhs, MODULUS),
            mod_sub(MODULUS - 1, rhs, MODULUS)
        ]
    );
}

#[test]
fn ntt_butterfly_stage_rejects_invalid_shape() {
    let mut data = [1, 2, 3, 4];
    assert_eq!(
        ntt_butterfly_stage_u64(&mut data, 3, &[1], MODULUS),
        Err(SimdError::LengthMismatch)
    );
    assert_eq!(
        ntt_butterfly_stage_u64(&mut data, 4, &[1], MODULUS),
        Err(SimdError::LengthMismatch)
    );
}

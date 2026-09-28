use super::{east_mask, west_mask};

/// Rook sliding attacks via AVX2 Kogge-Stone fill: the four orthogonal
/// directions flood in parallel, one per 64-bit lane of a `__m256i`.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[must_use]
#[inline]
pub unsafe fn kogge_stone_rook_avx2(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_extract_epi64, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_set_epi64x, _mm256_slli_epi64, _mm256_sllv_epi64, _mm256_srlv_epi64,
    };

    let p_scalar = !occupancy;

    // Left shifts: North (+8), East (+1)
    let left_shifts = _mm256_set_epi64x(0, 0, 1, 8);
    let mut g_left = _mm256_set1_epi64x(slider as i64);
    let mut p_left = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    // Right shifts: South (-8), West (-1)
    let right_shifts = _mm256_set_epi64x(0, 0, 1, 8);
    let mut g_right = _mm256_set1_epi64x(slider as i64);
    let mut p_right = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    // Step 0: shift amount = 1
    {
        let left_shift_amt = left_shifts;
        let right_shift_amt = right_shifts;

        let left_mask_vec = _mm256_set_epi64x(0, 0, east_mask(1) as i64, -1);
        let right_mask_vec = _mm256_set_epi64x(0, 0, west_mask(1) as i64, -1);

        let sg = _mm256_and_si256(_mm256_sllv_epi64(g_left, left_shift_amt), left_mask_vec);
        let sp = _mm256_and_si256(_mm256_sllv_epi64(p_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));
        p_left = _mm256_and_si256(p_left, sp);

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        let sp_r = _mm256_and_si256(_mm256_srlv_epi64(p_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
        p_right = _mm256_and_si256(p_right, sp_r);
    }

    // Step 1: shift amount = 2
    {
        let left_shift_amt = _mm256_slli_epi64(left_shifts, 1);
        let right_shift_amt = _mm256_slli_epi64(right_shifts, 1);

        let left_mask_vec = _mm256_set_epi64x(0, 0, east_mask(2) as i64, -1);
        let right_mask_vec = _mm256_set_epi64x(0, 0, west_mask(2) as i64, -1);

        let sg = _mm256_and_si256(_mm256_sllv_epi64(g_left, left_shift_amt), left_mask_vec);
        let sp = _mm256_and_si256(_mm256_sllv_epi64(p_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));
        p_left = _mm256_and_si256(p_left, sp);

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        let sp_r = _mm256_and_si256(_mm256_srlv_epi64(p_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
        p_right = _mm256_and_si256(p_right, sp_r);
    }

    // Step 2: shift amount = 4
    {
        let left_shift_amt = _mm256_slli_epi64(left_shifts, 2);
        let right_shift_amt = _mm256_slli_epi64(right_shifts, 2);

        let left_mask_vec = _mm256_set_epi64x(0, 0, east_mask(4) as i64, -1);
        let right_mask_vec = _mm256_set_epi64x(0, 0, west_mask(4) as i64, -1);

        let sg = _mm256_and_si256(_mm256_sllv_epi64(g_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
    }

    // Shift the final propagated results to get attacks (including blocker, excluding slider)
    let left_shifted = _mm256_and_si256(
        _mm256_sllv_epi64(g_left, left_shifts),
        _mm256_set_epi64x(0, 0, east_mask(1) as i64, -1),
    );
    let right_shifted = _mm256_and_si256(
        _mm256_srlv_epi64(g_right, right_shifts),
        _mm256_set_epi64x(0, 0, west_mask(1) as i64, -1),
    );

    let combined_vec = _mm256_or_si256(left_shifted, right_shifted);
    let val0 = _mm256_extract_epi64(combined_vec, 0) as u64;
    let val1 = _mm256_extract_epi64(combined_vec, 1) as u64;
    val0 | val1
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Bishop sliding attacks via AVX2 Kogge-Stone fill: the four diagonal
/// directions flood in parallel, one per 64-bit lane of a `__m256i`.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[must_use]
#[inline]
pub unsafe fn kogge_stone_bishop_avx2(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::{
        _mm256_and_si256, _mm256_extract_epi64, _mm256_or_si256, _mm256_set1_epi64x,
        _mm256_set_epi64x, _mm256_slli_epi64, _mm256_sllv_epi64, _mm256_srlv_epi64,
    };

    let p_scalar = !occupancy;

    // Left shifts: North-East (+9), North-West (+7)
    let left_shifts = _mm256_set_epi64x(0, 0, 7, 9);
    let mut g_left = _mm256_set1_epi64x(slider as i64);
    let mut p_left = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    // Right shifts: South-East (-7), South-West (-9)
    let right_shifts = _mm256_set_epi64x(0, 0, 9, 7);
    let mut g_right = _mm256_set1_epi64x(slider as i64);
    let mut p_right = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    // Step 0: shift amount = 1
    {
        let left_shift_amt = left_shifts;
        let right_shift_amt = right_shifts;

        let left_mask_vec = _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64);
        let right_mask_vec = _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64);

        let sg = _mm256_and_si256(_mm256_sllv_epi64(g_left, left_shift_amt), left_mask_vec);
        let sp = _mm256_and_si256(_mm256_sllv_epi64(p_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));
        p_left = _mm256_and_si256(p_left, sp);

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        let sp_r = _mm256_and_si256(_mm256_srlv_epi64(p_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
        p_right = _mm256_and_si256(p_right, sp_r);
    }

    // Step 1: shift amount = 2
    {
        let left_shift_amt = _mm256_slli_epi64(left_shifts, 1);
        let right_shift_amt = _mm256_slli_epi64(right_shifts, 1);

        let left_mask_vec = _mm256_set_epi64x(0, 0, west_mask(2) as i64, east_mask(2) as i64);
        let right_mask_vec = _mm256_set_epi64x(0, 0, west_mask(2) as i64, east_mask(2) as i64);

        let sg = _mm256_and_si256(_mm256_sllv_epi64(g_left, left_shift_amt), left_mask_vec);
        let sp = _mm256_and_si256(_mm256_sllv_epi64(p_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));
        p_left = _mm256_and_si256(p_left, sp);

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        let sp_r = _mm256_and_si256(_mm256_srlv_epi64(p_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
        p_right = _mm256_and_si256(p_right, sp_r);
    }

    // Step 2: shift amount = 4
    {
        let left_shift_amt = _mm256_slli_epi64(left_shifts, 2);
        let right_shift_amt = _mm256_slli_epi64(right_shifts, 2);

        let left_mask_vec = _mm256_set_epi64x(0, 0, west_mask(4) as i64, east_mask(4) as i64);
        let right_mask_vec = _mm256_set_epi64x(0, 0, west_mask(4) as i64, east_mask(4) as i64);

        let sg = _mm256_and_si256(_mm256_sllv_epi64(g_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
    }

    // Shift the final propagated results to get attacks (including blocker, excluding slider)
    let left_shifted = _mm256_and_si256(
        _mm256_sllv_epi64(g_left, left_shifts),
        _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64),
    );
    let right_shifted = _mm256_and_si256(
        _mm256_srlv_epi64(g_right, right_shifts),
        _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64),
    );

    let combined_vec = _mm256_or_si256(left_shifted, right_shifted);
    let val0 = _mm256_extract_epi64(combined_vec, 0) as u64;
    let val1 = _mm256_extract_epi64(combined_vec, 1) as u64;
    val0 | val1
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Queen sliding attacks: union of the rook and bishop AVX2 fills.
///
/// # Safety
/// Caller must ensure AVX2 is available.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn kogge_stone_queen_avx2(slider: u64, occupancy: u64) -> u64 {
    // The AVX2 rook and bishop fills each use only lanes 0 and 1 with every
    // other mask lane zeroed, so a "fused" body would perform exactly the two
    // kernels' work with no lane sharing. The union is written directly, as in
    // the scalar and NEON reference backends.
    kogge_stone_rook_avx2(slider, occupancy) | kogge_stone_bishop_avx2(slider, occupancy)
}

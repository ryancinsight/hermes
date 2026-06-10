use super::{east_mask, west_mask};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_rook_avx2(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::*;

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
        let sp = _mm256_and_si256(_mm256_sllv_epi64(p_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));
        p_left = _mm256_and_si256(p_left, sp);

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        let sp_r = _mm256_and_si256(_mm256_srlv_epi64(p_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
        p_right = _mm256_and_si256(p_right, sp_r);
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
#[target_feature(enable = "avx2")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_bishop_avx2(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::*;

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
        let sp = _mm256_and_si256(_mm256_sllv_epi64(p_left, left_shift_amt), left_mask_vec);
        g_left = _mm256_or_si256(g_left, _mm256_and_si256(sg, p_left));
        p_left = _mm256_and_si256(p_left, sp);

        let sg_r = _mm256_and_si256(_mm256_srlv_epi64(g_right, right_shift_amt), right_mask_vec);
        let sp_r = _mm256_and_si256(_mm256_srlv_epi64(p_right, right_shift_amt), right_mask_vec);
        g_right = _mm256_or_si256(g_right, _mm256_and_si256(sg_r, p_right));
        p_right = _mm256_and_si256(p_right, sp_r);
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
#[target_feature(enable = "avx2")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_queen_avx2(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::*;

    let p_scalar = !occupancy;

    // --- Rook Setup ---
    let left_shifts_r = _mm256_set_epi64x(0, 0, 1, 8);
    let mut g_left_r = _mm256_set1_epi64x(slider as i64);
    let mut p_left_r = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    let right_shifts_r = _mm256_set_epi64x(0, 0, 1, 8);
    let mut g_right_r = _mm256_set1_epi64x(slider as i64);
    let mut p_right_r = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    // --- Bishop Setup ---
    let left_shifts_b = _mm256_set_epi64x(0, 0, 7, 9);
    let mut g_left_b = _mm256_set1_epi64x(slider as i64);
    let mut p_left_b = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    let right_shifts_b = _mm256_set_epi64x(0, 0, 9, 7);
    let mut g_right_b = _mm256_set1_epi64x(slider as i64);
    let mut p_right_b = _mm256_set_epi64x(0, 0, p_scalar as i64, p_scalar as i64);

    // --- Step 0 (shift amount = 1) ---
    {
        // Rook mask
        let left_mask_r = _mm256_set_epi64x(0, 0, east_mask(1) as i64, -1);
        let right_mask_r = _mm256_set_epi64x(0, 0, west_mask(1) as i64, -1);
        // Bishop mask
        let left_mask_b = _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64);
        let right_mask_b = _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64);

        // Rook left
        let sg_lr = _mm256_and_si256(_mm256_sllv_epi64(g_left_r, left_shifts_r), left_mask_r);
        let sp_lr = _mm256_and_si256(_mm256_sllv_epi64(p_left_r, left_shifts_r), left_mask_r);
        g_left_r = _mm256_or_si256(g_left_r, _mm256_and_si256(sg_lr, p_left_r));
        p_left_r = _mm256_and_si256(p_left_r, sp_lr);

        // Bishop left
        let sg_lb = _mm256_and_si256(_mm256_sllv_epi64(g_left_b, left_shifts_b), left_mask_b);
        let sp_lb = _mm256_and_si256(_mm256_sllv_epi64(p_left_b, left_shifts_b), left_mask_b);
        g_left_b = _mm256_or_si256(g_left_b, _mm256_and_si256(sg_lb, p_left_b));
        p_left_b = _mm256_and_si256(p_left_b, sp_lb);

        // Rook right
        let sg_rr = _mm256_and_si256(_mm256_srlv_epi64(g_right_r, right_shifts_r), right_mask_r);
        let sp_rr = _mm256_and_si256(_mm256_srlv_epi64(p_right_r, right_shifts_r), right_mask_r);
        g_right_r = _mm256_or_si256(g_right_r, _mm256_and_si256(sg_rr, p_right_r));
        p_right_r = _mm256_and_si256(p_right_r, sp_rr);

        // Bishop right
        let sg_rb = _mm256_and_si256(_mm256_srlv_epi64(g_right_b, right_shifts_b), right_mask_b);
        let sp_rb = _mm256_and_si256(_mm256_srlv_epi64(p_right_b, right_shifts_b), right_mask_b);
        g_right_b = _mm256_or_si256(g_right_b, _mm256_and_si256(sg_rb, p_right_b));
        p_right_b = _mm256_and_si256(p_right_b, sp_rb);
    }

    // --- Step 1 (shift amount = 2) ---
    {
        let left_shift_amt_r = _mm256_slli_epi64(left_shifts_r, 1);
        let right_shift_amt_r = _mm256_slli_epi64(right_shifts_r, 1);
        let left_shift_amt_b = _mm256_slli_epi64(left_shifts_b, 1);
        let right_shift_amt_b = _mm256_slli_epi64(right_shifts_b, 1);

        // Rook mask
        let left_mask_r = _mm256_set_epi64x(0, 0, east_mask(2) as i64, -1);
        let right_mask_r = _mm256_set_epi64x(0, 0, west_mask(2) as i64, -1);
        // Bishop mask
        let left_mask_b = _mm256_set_epi64x(0, 0, west_mask(2) as i64, east_mask(2) as i64);
        let right_mask_b = _mm256_set_epi64x(0, 0, west_mask(2) as i64, east_mask(2) as i64);

        // Rook left
        let sg_lr = _mm256_and_si256(_mm256_sllv_epi64(g_left_r, left_shift_amt_r), left_mask_r);
        let sp_lr = _mm256_and_si256(_mm256_sllv_epi64(p_left_r, left_shift_amt_r), left_mask_r);
        g_left_r = _mm256_or_si256(g_left_r, _mm256_and_si256(sg_lr, p_left_r));
        p_left_r = _mm256_and_si256(p_left_r, sp_lr);

        // Bishop left
        let sg_lb = _mm256_and_si256(_mm256_sllv_epi64(g_left_b, left_shift_amt_b), left_mask_b);
        let sp_lb = _mm256_and_si256(_mm256_sllv_epi64(p_left_b, left_shift_amt_b), left_mask_b);
        g_left_b = _mm256_or_si256(g_left_b, _mm256_and_si256(sg_lb, p_left_b));
        p_left_b = _mm256_and_si256(p_left_b, sp_lb);

        // Rook right
        let sg_rr = _mm256_and_si256(
            _mm256_srlv_epi64(g_right_r, right_shift_amt_r),
            right_mask_r,
        );
        let sp_rr = _mm256_and_si256(
            _mm256_srlv_epi64(p_right_r, right_shift_amt_r),
            right_mask_r,
        );
        g_right_r = _mm256_or_si256(g_right_r, _mm256_and_si256(sg_rr, p_right_r));
        p_right_r = _mm256_and_si256(p_right_r, sp_rr);

        // Bishop right
        let sg_rb = _mm256_and_si256(
            _mm256_srlv_epi64(g_right_b, right_shift_amt_b),
            right_mask_b,
        );
        let sp_rb = _mm256_and_si256(
            _mm256_srlv_epi64(p_right_b, right_shift_amt_b),
            right_mask_b,
        );
        g_right_b = _mm256_or_si256(g_right_b, _mm256_and_si256(sg_rb, p_right_b));
        p_right_b = _mm256_and_si256(p_right_b, sp_rb);
    }

    // --- Step 2 (shift amount = 4) ---
    {
        let left_shift_amt_r = _mm256_slli_epi64(left_shifts_r, 2);
        let right_shift_amt_r = _mm256_slli_epi64(right_shifts_r, 2);
        let left_shift_amt_b = _mm256_slli_epi64(left_shifts_b, 2);
        let right_shift_amt_b = _mm256_slli_epi64(right_shifts_b, 2);

        // Rook mask
        let left_mask_r = _mm256_set_epi64x(0, 0, east_mask(4) as i64, -1);
        let right_mask_r = _mm256_set_epi64x(0, 0, west_mask(4) as i64, -1);
        // Bishop mask
        let left_mask_b = _mm256_set_epi64x(0, 0, west_mask(4) as i64, east_mask(4) as i64);
        let right_mask_b = _mm256_set_epi64x(0, 0, west_mask(4) as i64, east_mask(4) as i64);

        // Rook left
        let sg_lr = _mm256_and_si256(_mm256_sllv_epi64(g_left_r, left_shift_amt_r), left_mask_r);
        g_left_r = _mm256_or_si256(g_left_r, _mm256_and_si256(sg_lr, p_left_r));

        // Bishop left
        let sg_lb = _mm256_and_si256(_mm256_sllv_epi64(g_left_b, left_shift_amt_b), left_mask_b);
        g_left_b = _mm256_or_si256(g_left_b, _mm256_and_si256(sg_lb, p_left_b));

        // Rook right
        let sg_rr = _mm256_and_si256(
            _mm256_srlv_epi64(g_right_r, right_shift_amt_r),
            right_mask_r,
        );
        g_right_r = _mm256_or_si256(g_right_r, _mm256_and_si256(sg_rr, p_right_r));

        // Bishop right
        let sg_rb = _mm256_and_si256(
            _mm256_srlv_epi64(g_right_b, right_shift_amt_b),
            right_mask_b,
        );
        g_right_b = _mm256_or_si256(g_right_b, _mm256_and_si256(sg_rb, p_right_b));
    }

    // Final shifts
    let left_shifted_r = _mm256_and_si256(
        _mm256_sllv_epi64(g_left_r, left_shifts_r),
        _mm256_set_epi64x(0, 0, east_mask(1) as i64, -1),
    );
    let right_shifted_r = _mm256_and_si256(
        _mm256_srlv_epi64(g_right_r, right_shifts_r),
        _mm256_set_epi64x(0, 0, west_mask(1) as i64, -1),
    );

    let left_shifted_b = _mm256_and_si256(
        _mm256_sllv_epi64(g_left_b, left_shifts_b),
        _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64),
    );
    let right_shifted_b = _mm256_and_si256(
        _mm256_srlv_epi64(g_right_b, right_shifts_b),
        _mm256_set_epi64x(0, 0, west_mask(1) as i64, east_mask(1) as i64),
    );

    // Combine all 4 in registers
    let comb_r = _mm256_or_si256(left_shifted_r, right_shifted_r);
    let comb_b = _mm256_or_si256(left_shifted_b, right_shifted_b);
    let combined_all = _mm256_or_si256(comb_r, comb_b);

    let val0 = _mm256_extract_epi64(combined_all, 0) as u64;
    let val1 = _mm256_extract_epi64(combined_all, 1) as u64;
    val0 | val1
}

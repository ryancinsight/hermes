use super::{east_mask, west_mask};

/// Computes Rook attacks using Kogge-Stone vectorized with AVX-512.
///
/// # Safety
/// Caller must ensure that the host CPU supports AVX-512 Foundation features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_rook_avx512(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::*;

    let p_scalar = !occupancy;

    // Lower 4 lanes: left shifts (North, East, 0, 0)
    // Upper 4 lanes: right shifts (South, West, 0, 0)
    let left_shifts = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 1, 8);
    let right_shifts = _mm512_set_epi64(0, 0, 1, 8, 0, 0, 0, 0);

    let mut g = _mm512_set1_epi64(slider as i64);
    let mut p = _mm512_set_epi64(
        0,
        0,
        p_scalar as i64,
        p_scalar as i64,
        0,
        0,
        p_scalar as i64,
        p_scalar as i64,
    );

    let left_mask_1 = east_mask(1) as i64;
    let left_mask_2 = east_mask(2) as i64;
    let left_mask_4 = east_mask(4) as i64;
    let right_mask_1 = west_mask(1) as i64;
    let right_mask_2 = west_mask(2) as i64;
    let right_mask_4 = west_mask(4) as i64;

    let masks = [
        _mm512_set_epi64(0, 0, right_mask_1, -1, 0, 0, left_mask_1, -1),
        _mm512_set_epi64(0, 0, right_mask_2, -1, 0, 0, left_mask_2, -1),
        _mm512_set_epi64(0, 0, right_mask_4, -1, 0, 0, left_mask_4, -1),
    ];

    // Step 0
    {
        let l_shift = left_shifts;
        let r_shift = right_shifts;

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sp_l = _mm512_sllv_epi64(p, l_shift);

        let sg_r = _mm512_srlv_epi64(g, r_shift);
        let sp_r = _mm512_srlv_epi64(p, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let sp = _mm512_or_si512(sp_l, sp_r);

        let masked_sg = _mm512_and_si512(sg, masks[0]);
        let masked_sp = _mm512_and_si512(sp, masks[0]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
        p = _mm512_and_si512(p, masked_sp);
    }

    // Step 1
    {
        let l_shift = _mm512_slli_epi64(left_shifts, 1);
        let r_shift = _mm512_slli_epi64(right_shifts, 1);

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sp_l = _mm512_sllv_epi64(p, l_shift);

        let sg_r = _mm512_srlv_epi64(g, r_shift);
        let sp_r = _mm512_srlv_epi64(p, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let sp = _mm512_or_si512(sp_l, sp_r);

        let masked_sg = _mm512_and_si512(sg, masks[1]);
        let masked_sp = _mm512_and_si512(sp, masks[1]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
        p = _mm512_and_si512(p, masked_sp);
    }

    // Step 2
    {
        let l_shift = _mm512_slli_epi64(left_shifts, 2);
        let r_shift = _mm512_slli_epi64(right_shifts, 2);

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sg_r = _mm512_srlv_epi64(g, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let masked_sg = _mm512_and_si512(sg, masks[2]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
    }

    let final_l = _mm512_and_si512(_mm512_sllv_epi64(g, left_shifts), masks[0]);
    let final_r = _mm512_and_si512(_mm512_srlv_epi64(g, right_shifts), masks[0]);
    let final_res = _mm512_or_si512(final_l, final_r);

    let low = _mm512_castsi512_si256(final_res);
    let high = _mm512_extracti64x4_epi64(final_res, 1);
    let combined_256 = _mm256_or_si256(low, high);

    let val0 = _mm256_extract_epi64(combined_256, 0) as u64;
    let val1 = _mm256_extract_epi64(combined_256, 1) as u64;
    val0 | val1
}

/// Computes Bishop attacks using Kogge-Stone vectorized with AVX-512.
///
/// # Safety
/// Caller must ensure that the host CPU supports AVX-512 Foundation features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_bishop_avx512(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::*;

    let p_scalar = !occupancy;

    // Lower 4 lanes: left shifts (North-East (+9), North-West (+7), 0, 0)
    // Upper 4 lanes: right shifts (South-East (>> 7), South-West (>> 9), 0, 0)
    let left_shifts = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 7, 9);
    let right_shifts = _mm512_set_epi64(0, 0, 9, 7, 0, 0, 0, 0);

    let mut g = _mm512_set1_epi64(slider as i64);
    let mut p = _mm512_set_epi64(
        0,
        0,
        p_scalar as i64,
        p_scalar as i64,
        0,
        0,
        p_scalar as i64,
        p_scalar as i64,
    );

    let east_mask_1 = east_mask(1) as i64;
    let east_mask_2 = east_mask(2) as i64;
    let east_mask_4 = east_mask(4) as i64;
    let west_mask_1 = west_mask(1) as i64;
    let west_mask_2 = west_mask(2) as i64;
    let west_mask_4 = west_mask(4) as i64;

    let masks = [
        _mm512_set_epi64(
            0,
            0,
            west_mask_1,
            east_mask_1,
            0,
            0,
            west_mask_1,
            east_mask_1,
        ),
        _mm512_set_epi64(
            0,
            0,
            west_mask_2,
            east_mask_2,
            0,
            0,
            west_mask_2,
            east_mask_2,
        ),
        _mm512_set_epi64(
            0,
            0,
            west_mask_4,
            east_mask_4,
            0,
            0,
            west_mask_4,
            east_mask_4,
        ),
    ];

    // Step 0
    {
        let l_shift = left_shifts;
        let r_shift = right_shifts;

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sp_l = _mm512_sllv_epi64(p, l_shift);

        let sg_r = _mm512_srlv_epi64(g, r_shift);
        let sp_r = _mm512_srlv_epi64(p, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let sp = _mm512_or_si512(sp_l, sp_r);

        let masked_sg = _mm512_and_si512(sg, masks[0]);
        let masked_sp = _mm512_and_si512(sp, masks[0]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
        p = _mm512_and_si512(p, masked_sp);
    }

    // Step 1
    {
        let l_shift = _mm512_slli_epi64(left_shifts, 1);
        let r_shift = _mm512_slli_epi64(right_shifts, 1);

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sp_l = _mm512_sllv_epi64(p, l_shift);

        let sg_r = _mm512_srlv_epi64(g, r_shift);
        let sp_r = _mm512_srlv_epi64(p, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let sp = _mm512_or_si512(sp_l, sp_r);

        let masked_sg = _mm512_and_si512(sg, masks[1]);
        let masked_sp = _mm512_and_si512(sp, masks[1]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
        p = _mm512_and_si512(p, masked_sp);
    }

    // Step 2
    {
        let l_shift = _mm512_slli_epi64(left_shifts, 2);
        let r_shift = _mm512_slli_epi64(right_shifts, 2);

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sg_r = _mm512_srlv_epi64(g, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let masked_sg = _mm512_and_si512(sg, masks[2]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
    }

    let final_l = _mm512_and_si512(_mm512_sllv_epi64(g, left_shifts), masks[0]);
    let final_r = _mm512_and_si512(_mm512_srlv_epi64(g, right_shifts), masks[0]);
    let final_res = _mm512_or_si512(final_l, final_r);

    let low = _mm512_castsi512_si256(final_res);
    let high = _mm512_extracti64x4_epi64(final_res, 1);
    let combined_256 = _mm256_or_si256(low, high);

    let val0 = _mm256_extract_epi64(combined_256, 0) as u64;
    let val1 = _mm256_extract_epi64(combined_256, 1) as u64;
    val0 | val1
}

/// Computes Queen attacks using Kogge-Stone vectorized with AVX-512 (all 8 directions in parallel).
///
/// # Safety
/// Caller must ensure that the host CPU supports AVX-512 Foundation features.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_queen_avx512(slider: u64, occupancy: u64) -> u64 {
    use core::arch::x86_64::*;

    let p_scalar = !occupancy;

    // Lower 4 lanes: left shifts (North (+8), East (+1), North-East (+9), North-West (+7))
    // Upper 4 lanes: right shifts (South (>> 8), West (>> 1), South-East (>> 7), South-West (>> 9))
    let left_shifts = _mm512_set_epi64(0, 0, 0, 0, 7, 9, 1, 8);
    let right_shifts = _mm512_set_epi64(9, 7, 1, 8, 0, 0, 0, 0);

    let mut g = _mm512_set1_epi64(slider as i64);
    let mut p = _mm512_set1_epi64(p_scalar as i64);

    let east_mask_1 = east_mask(1) as i64;
    let east_mask_2 = east_mask(2) as i64;
    let east_mask_4 = east_mask(4) as i64;
    let west_mask_1 = west_mask(1) as i64;
    let west_mask_2 = west_mask(2) as i64;
    let west_mask_4 = west_mask(4) as i64;

    let masks = [
        _mm512_set_epi64(
            west_mask_1,
            east_mask_1,
            west_mask_1,
            -1,
            west_mask_1,
            east_mask_1,
            east_mask_1,
            -1,
        ),
        _mm512_set_epi64(
            west_mask_2,
            east_mask_2,
            west_mask_2,
            -1,
            west_mask_2,
            east_mask_2,
            east_mask_2,
            -1,
        ),
        _mm512_set_epi64(
            west_mask_4,
            east_mask_4,
            west_mask_4,
            -1,
            west_mask_4,
            east_mask_4,
            east_mask_4,
            -1,
        ),
    ];

    // Step 0
    {
        let l_shift = left_shifts;
        let r_shift = right_shifts;

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sp_l = _mm512_sllv_epi64(p, l_shift);

        let sg_r = _mm512_srlv_epi64(g, r_shift);
        let sp_r = _mm512_srlv_epi64(p, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let sp = _mm512_or_si512(sp_l, sp_r);

        let masked_sg = _mm512_and_si512(sg, masks[0]);
        let masked_sp = _mm512_and_si512(sp, masks[0]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
        p = _mm512_and_si512(p, masked_sp);
    }

    // Step 1
    {
        let l_shift = _mm512_slli_epi64(left_shifts, 1);
        let r_shift = _mm512_slli_epi64(right_shifts, 1);

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sp_l = _mm512_sllv_epi64(p, l_shift);

        let sg_r = _mm512_srlv_epi64(g, r_shift);
        let sp_r = _mm512_srlv_epi64(p, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let sp = _mm512_or_si512(sp_l, sp_r);

        let masked_sg = _mm512_and_si512(sg, masks[1]);
        let masked_sp = _mm512_and_si512(sp, masks[1]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
        p = _mm512_and_si512(p, masked_sp);
    }

    // Step 2
    {
        let l_shift = _mm512_slli_epi64(left_shifts, 2);
        let r_shift = _mm512_slli_epi64(right_shifts, 2);

        let sg_l = _mm512_sllv_epi64(g, l_shift);
        let sg_r = _mm512_srlv_epi64(g, r_shift);

        let sg = _mm512_or_si512(sg_l, sg_r);
        let masked_sg = _mm512_and_si512(sg, masks[2]);

        g = _mm512_or_si512(g, _mm512_and_si512(masked_sg, p));
    }

    let final_l = _mm512_and_si512(_mm512_sllv_epi64(g, left_shifts), masks[0]);
    let final_r = _mm512_and_si512(_mm512_srlv_epi64(g, right_shifts), masks[0]);
    let final_res = _mm512_or_si512(final_l, final_r);

    let low = _mm512_castsi512_si256(final_res);
    let high = _mm512_extracti64x4_epi64(final_res, 1);
    let combined_256 = _mm256_or_si256(low, high);

    let low_128 = _mm256_castsi256_si128(combined_256);
    let high_128 = _mm256_extracti128_si256(combined_256, 1);
    let combined_128 = _mm_or_si128(low_128, high_128);

    let val0 = _mm_cvtsi128_si64(combined_128) as u64;
    let val1 = _mm_cvtsi128_si64(_mm_srli_si128(combined_128, 8)) as u64;
    val0 | val1
}

use super::{east_mask, west_mask};

/// Rook sliding attacks via NEON Kogge-Stone fill: orthogonal directions
/// flood in parallel across 128-bit registers.
///
/// # Safety
/// NEON is mandatory on aarch64; no additional caller requirements.
#[cfg(target_arch = "aarch64")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_rook_neon(slider: u64, occupancy: u64) -> u64 {
    use core::arch::aarch64::*;

    let p_scalar = !occupancy;

    // NEON has 2-lane u64 registers (128-bit).
    // Left shifts: North (+8), East (+1)
    let left_shifts = vsetq_lane_u64(1, vdupq_n_u64(8), 1); // [8, 1]
    let mut g_left = vdupq_n_u64(slider);
    let mut p_left = vsetq_lane_u64(p_scalar, vdupq_n_u64(p_scalar), 1);

    // Right shifts: South (>> 8), West (>> 1)
    let right_shifts = vsetq_lane_u64(1, vdupq_n_u64(8), 1); // [8, 1]
    let mut g_right = vdupq_n_u64(slider);
    let mut p_right = vsetq_lane_u64(p_scalar, vdupq_n_u64(p_scalar), 1);

    let east_mask_1 = east_mask(1);
    let east_mask_2 = east_mask(2);
    let east_mask_4 = east_mask(4);
    let west_mask_1 = west_mask(1);
    let west_mask_2 = west_mask(2);
    let west_mask_4 = west_mask(4);

    let left_masks = [
        vsetq_lane_u64(east_mask_1, vdupq_n_u64(0xFFFF_FFFF_FFFF_FFFF), 1),
        vsetq_lane_u64(east_mask_2, vdupq_n_u64(0xFFFF_FFFF_FFFF_FFFF), 1),
        vsetq_lane_u64(east_mask_4, vdupq_n_u64(0xFFFF_FFFF_FFFF_FFFF), 1),
    ];
    let right_masks = [
        vsetq_lane_u64(west_mask_1, vdupq_n_u64(0xFFFF_FFFF_FFFF_FFFF), 1),
        vsetq_lane_u64(west_mask_2, vdupq_n_u64(0xFFFF_FFFF_FFFF_FFFF), 1),
        vsetq_lane_u64(west_mask_4, vdupq_n_u64(0xFFFF_FFFF_FFFF_FFFF), 1),
    ];

    // Step 0
    {
        let shift_amt = left_shifts;
        let mask_v = left_masks[0];

        let sg = vandq_u64(vshlq_u64(g_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        let sp = vandq_u64(vshlq_u64(p_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        g_left = vorrq_u64(g_left, vandq_u64(sg, p_left));
        p_left = vandq_u64(p_left, sp);

        let right_shift_amt = vnegq_s64(vreinterpretq_s64_u64(shift_amt));
        let r_mask_v = right_masks[0];
        let sg_r = vandq_u64(vshlq_u64(g_right, right_shift_amt), r_mask_v);
        let sp_r = vandq_u64(vshlq_u64(p_right, right_shift_amt), r_mask_v);
        g_right = vorrq_u64(g_right, vandq_u64(sg_r, p_right));
        p_right = vandq_u64(p_right, sp_r);
    }

    // Step 1
    {
        let shift_amt = vshlq_n_u64::<1>(left_shifts);
        let mask_v = left_masks[1];

        let sg = vandq_u64(vshlq_u64(g_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        let sp = vandq_u64(vshlq_u64(p_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        g_left = vorrq_u64(g_left, vandq_u64(sg, p_left));
        p_left = vandq_u64(p_left, sp);

        let right_shift_amt = vnegq_s64(vreinterpretq_s64_u64(shift_amt));
        let r_mask_v = right_masks[1];
        let sg_r = vandq_u64(vshlq_u64(g_right, right_shift_amt), r_mask_v);
        let sp_r = vandq_u64(vshlq_u64(p_right, right_shift_amt), r_mask_v);
        g_right = vorrq_u64(g_right, vandq_u64(sg_r, p_right));
        p_right = vandq_u64(p_right, sp_r);
    }

    // Step 2
    {
        let shift_amt = vshlq_n_u64::<2>(left_shifts);
        let mask_v = left_masks[2];

        let sg = vandq_u64(vshlq_u64(g_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        let sp = vandq_u64(vshlq_u64(p_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        g_left = vorrq_u64(g_left, vandq_u64(sg, p_left));
        p_left = vandq_u64(p_left, sp);

        let right_shift_amt = vnegq_s64(vreinterpretq_s64_u64(shift_amt));
        let r_mask_v = right_masks[2];
        let sg_r = vandq_u64(vshlq_u64(g_right, right_shift_amt), r_mask_v);
        let sp_r = vandq_u64(vshlq_u64(p_right, right_shift_amt), r_mask_v);
        g_right = vorrq_u64(g_right, vandq_u64(sg_r, p_right));
        p_right = vandq_u64(p_right, sp_r);
    }

    let final_left = vandq_u64(
        vshlq_u64(g_left, vreinterpretq_s64_u64(left_shifts)),
        left_masks[0],
    );
    let final_right = vandq_u64(
        vshlq_u64(g_right, vnegq_s64(vreinterpretq_s64_u64(right_shifts))),
        right_masks[0],
    );

    let out_l0 = vgetq_lane_u64(final_left, 0);
    let out_l1 = vgetq_lane_u64(final_left, 1);
    let out_r0 = vgetq_lane_u64(final_right, 0);
    let out_r1 = vgetq_lane_u64(final_right, 1);

    out_l0 | out_l1 | out_r0 | out_r1
}

/// Bishop sliding attacks via NEON Kogge-Stone fill: diagonal directions
/// flood in parallel across 128-bit registers.
///
/// # Safety
/// NEON is mandatory on aarch64; no additional caller requirements.
#[cfg(target_arch = "aarch64")]
#[allow(unused_assignments)]
pub unsafe fn kogge_stone_bishop_neon(slider: u64, occupancy: u64) -> u64 {
    use core::arch::aarch64::*;

    let p_scalar = !occupancy;

    // Left shifts: North-East (+9), North-West (+7)
    let left_shifts = vsetq_lane_u64(7, vdupq_n_u64(9), 1); // [9, 7]
    let mut g_left = vdupq_n_u64(slider);
    let mut p_left = vsetq_lane_u64(p_scalar, vdupq_n_u64(p_scalar), 1);

    // Right shifts: South-East (>> 7), South-West (>> 9)
    let right_shifts = vsetq_lane_u64(9, vdupq_n_u64(7), 1); // [7, 9]
    let mut g_right = vdupq_n_u64(slider);
    let mut p_right = vsetq_lane_u64(p_scalar, vdupq_n_u64(p_scalar), 1);

    let east_mask_1 = east_mask(1);
    let east_mask_2 = east_mask(2);
    let east_mask_4 = east_mask(4);
    let west_mask_1 = west_mask(1);
    let west_mask_2 = west_mask(2);
    let west_mask_4 = west_mask(4);

    let left_masks = [
        vsetq_lane_u64(west_mask_1, vdupq_n_u64(east_mask_1), 1),
        vsetq_lane_u64(west_mask_2, vdupq_n_u64(east_mask_2), 1),
        vsetq_lane_u64(west_mask_4, vdupq_n_u64(east_mask_4), 1),
    ];
    let right_masks = [
        vsetq_lane_u64(west_mask_1, vdupq_n_u64(east_mask_1), 1),
        vsetq_lane_u64(west_mask_2, vdupq_n_u64(east_mask_2), 1),
        vsetq_lane_u64(west_mask_4, vdupq_n_u64(east_mask_4), 1),
    ];

    // Step 0
    {
        let shift_amt = left_shifts;
        let mask_v = left_masks[0];

        let sg = vandq_u64(vshlq_u64(g_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        let sp = vandq_u64(vshlq_u64(p_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        g_left = vorrq_u64(g_left, vandq_u64(sg, p_left));
        p_left = vandq_u64(p_left, sp);

        let right_shift_amt = vnegq_s64(vreinterpretq_s64_u64(right_shifts));
        let r_mask_v = right_masks[0];
        let sg_r = vandq_u64(vshlq_u64(g_right, right_shift_amt), r_mask_v);
        let sp_r = vandq_u64(vshlq_u64(p_right, right_shift_amt), r_mask_v);
        g_right = vorrq_u64(g_right, vandq_u64(sg_r, p_right));
        p_right = vandq_u64(p_right, sp_r);
    }

    // Step 1
    {
        let shift_amt = vshlq_n_u64::<1>(left_shifts);
        let mask_v = left_masks[1];

        let sg = vandq_u64(vshlq_u64(g_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        let sp = vandq_u64(vshlq_u64(p_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        g_left = vorrq_u64(g_left, vandq_u64(sg, p_left));
        p_left = vandq_u64(p_left, sp);

        let right_shift_amt = vnegq_s64(vreinterpretq_s64_u64(vshlq_n_u64::<1>(right_shifts)));
        let r_mask_v = right_masks[1];
        let sg_r = vandq_u64(vshlq_u64(g_right, right_shift_amt), r_mask_v);
        let sp_r = vandq_u64(vshlq_u64(p_right, right_shift_amt), r_mask_v);
        g_right = vorrq_u64(g_right, vandq_u64(sg_r, p_right));
        p_right = vandq_u64(p_right, sp_r);
    }

    // Step 2
    {
        let shift_amt = vshlq_n_u64::<2>(left_shifts);
        let mask_v = left_masks[2];

        let sg = vandq_u64(vshlq_u64(g_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        let sp = vandq_u64(vshlq_u64(p_left, vreinterpretq_s64_u64(shift_amt)), mask_v);
        g_left = vorrq_u64(g_left, vandq_u64(sg, p_left));
        p_left = vandq_u64(p_left, sp);

        let right_shift_amt = vnegq_s64(vreinterpretq_s64_u64(vshlq_n_u64::<2>(right_shifts)));
        let r_mask_v = right_masks[2];
        let sg_r = vandq_u64(vshlq_u64(g_right, right_shift_amt), r_mask_v);
        let sp_r = vandq_u64(vshlq_u64(p_right, right_shift_amt), r_mask_v);
        g_right = vorrq_u64(g_right, vandq_u64(sg_r, p_right));
        p_right = vandq_u64(p_right, sp_r);
    }

    let final_left = vandq_u64(
        vshlq_u64(g_left, vreinterpretq_s64_u64(left_shifts)),
        left_masks[0],
    );
    let final_right = vandq_u64(
        vshlq_u64(g_right, vnegq_s64(vreinterpretq_s64_u64(right_shifts))),
        right_masks[0],
    );

    let out_l0 = vgetq_lane_u64(final_left, 0);
    let out_l1 = vgetq_lane_u64(final_left, 1);
    let out_r0 = vgetq_lane_u64(final_right, 0);
    let out_r1 = vgetq_lane_u64(final_right, 1);

    out_l0 | out_l1 | out_r0 | out_r1
}

/// Queen sliding attacks: union of the rook and bishop NEON fills.
///
/// # Safety
/// NEON is mandatory on aarch64; no additional caller requirements.
#[cfg(target_arch = "aarch64")]
pub unsafe fn kogge_stone_queen_neon(slider: u64, occupancy: u64) -> u64 {
    kogge_stone_rook_neon(slider, occupancy) | kogge_stone_bishop_neon(slider, occupancy)
}

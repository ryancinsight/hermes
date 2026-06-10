//! Hyperbola Quintessence (o-protect) bitboard sliding attack generation.

use hermes_simd_core::bitboard::BitBoardKernel;

/// ZST marker for Hyperbola Quintessence backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hyperbola;

const FILE_RAYS: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        rays[sq] = 0x0101010101010101u64 << (sq & 7);
        sq += 1;
    }
    rays
};

const RANK_RAYS: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        rays[sq] = 0xFFu64 << (sq & 56);
        sq += 1;
    }
    rays
};

const DIAGONAL_RAYS: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let s_file = (sq & 7) as i32;
        let s_rank = (sq >> 3) as i32;
        let mut mask = 0u64;
        let mut f = 0;
        while f < 8 {
            let r = s_rank + (f - s_file);
            if r >= 0 && r < 8 {
                mask |= 1u64 << (r * 8 + f);
            }
            f += 1;
        }
        rays[sq] = mask;
        sq += 1;
    }
    rays
};

const ANTIDIAGONAL_RAYS: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let s_file = (sq & 7) as i32;
        let s_rank = (sq >> 3) as i32;
        let mut mask = 0u64;
        let mut f = 0;
        while f < 8 {
            let r = s_rank - (f - s_file);
            if r >= 0 && r < 8 {
                mask |= 1u64 << (r * 8 + f);
            }
            f += 1;
        }
        rays[sq] = mask;
        sq += 1;
    }
    rays
};

#[inline(always)]
fn file_ray(square: u8) -> u64 {
    FILE_RAYS[square as usize]
}

#[inline(always)]
fn rank_ray(square: u8) -> u64 {
    RANK_RAYS[square as usize]
}

#[inline(always)]
fn diagonal_ray(square: u8) -> u64 {
    DIAGONAL_RAYS[square as usize]
}

#[inline(always)]
fn antidiagonal_ray(square: u8) -> u64 {
    ANTIDIAGONAL_RAYS[square as usize]
}

/// Generates sliding attacks along a ray mask using Hyperbola Quintessence.
#[inline(always)]
pub fn hyperbola_quintessence(square: u8, occupancy: u64, ray_mask: u64) -> u64 {
    let slider = 1u64 << square;
    let o = occupancy & ray_mask;

    // Forward attacks
    let forward = o.wrapping_sub(slider.wrapping_mul(2)) ^ o;

    // Backward attacks (requires bit-reversal)
    let o_rev = o.reverse_bits();
    let slider_rev = slider.reverse_bits();
    let backward = o_rev.wrapping_sub(slider_rev.wrapping_mul(2)) ^ o_rev;

    // Combine and mask to the ray (excluding the slider itself)
    ((forward & ray_mask) | (backward.reverse_bits() & ray_mask)) & !slider
}

impl BitBoardKernel for Hyperbola {
    #[inline]
    unsafe fn rook_attacks(square: u8, occupancy: u64) -> u64 {
        hyperbola_quintessence(square, occupancy, file_ray(square))
            | hyperbola_quintessence(square, occupancy, rank_ray(square))
    }

    #[inline]
    unsafe fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
        hyperbola_quintessence(square, occupancy, diagonal_ray(square))
            | hyperbola_quintessence(square, occupancy, antidiagonal_ray(square))
    }
}

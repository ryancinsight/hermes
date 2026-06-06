//! Pure SWAR bitboard kernels and primitives.

use hermes_simd_core::bitboard::BitBoardKernel;

/// ZST marker for pure SWAR primitives backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swar;

// Directions shift and masks
const FILE_A: u64 = 0x0101010101010101;
const FILE_H: u64 = 0x8080808080808080;

#[inline(always)]
fn shift_n(v: u64) -> u64 { v << 8 }
#[inline(always)]
fn shift_s(v: u64) -> u64 { v >> 8 }
#[inline(always)]
fn shift_e(v: u64) -> u64 { (v << 1) & !FILE_A }
#[inline(always)]
fn shift_w(v: u64) -> u64 { (v >> 1) & !FILE_H }
#[inline(always)]
fn shift_ne(v: u64) -> u64 { (v << 9) & !FILE_A }
#[inline(always)]
fn shift_nw(v: u64) -> u64 { (v << 7) & !FILE_H }
#[inline(always)]
fn shift_se(v: u64) -> u64 { (v >> 7) & !FILE_A }
#[inline(always)]
fn shift_sw(v: u64) -> u64 { (v >> 9) & !FILE_H }

/// Generate sliding attacks using pure SWAR propagation loop in a single direction.
#[inline(always)]
pub fn attack_ray<F>(slider: u64, occupancy: u64, shift_fn: F) -> u64
where
    F: Fn(u64) -> u64,
{
    let mut attacks = 0;
    let mut step = slider;
    for _ in 0..7 {
        step = shift_fn(step);
        attacks |= step;
        if (step & occupancy) != 0 {
            break;
        }
    }
    attacks
}

impl BitBoardKernel for Swar {
    #[inline]
    unsafe fn rook_attacks(square: u8, occupancy: u64) -> u64 {
        let slider = 1u64 << square;
        attack_ray(slider, occupancy, shift_n)
            | attack_ray(slider, occupancy, shift_s)
            | attack_ray(slider, occupancy, shift_e)
            | attack_ray(slider, occupancy, shift_w)
    }

    #[inline]
    unsafe fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
        let slider = 1u64 << square;
        attack_ray(slider, occupancy, shift_ne)
            | attack_ray(slider, occupancy, shift_nw)
            | attack_ray(slider, occupancy, shift_se)
            | attack_ray(slider, occupancy, shift_sw)
    }
}

/// Pure SWAR bitwise utility primitives.
pub struct SwarUtils;

impl SwarUtils {
    /// Count set bits in a 64-bit word using SWAR parallel popcount.
    #[inline(always)]
    pub fn popcount(mut x: u64) -> u32 {
        x -= (x >> 1) & 0x5555555555555555;
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
        x = x.wrapping_mul(0x0101010101010101);
        (x >> 56) as u32
    }

    /// Find the least significant set bit (bit index 0-63) using SWAR trailing zero count.
    #[inline(always)]
    pub fn bit_scan_forward(x: u64) -> u32 {
        x.trailing_zeros()
    }

    /// Find the most significant set bit (bit index 0-63) using SWAR leading zero count.
    #[inline(always)]
    pub fn bit_scan_reverse(x: u64) -> u32 {
        63 - x.leading_zeros()
    }

    /// Computes parallel prefix shift (Kogge-Stone style prefix scan) on a 64-bit mask.
    #[inline(always)]
    pub fn parallel_prefix_shift<F>(mut x: u64, shift_fn: F) -> u64
    where
        F: Fn(u64, usize) -> u64,
    {
        x |= shift_fn(x, 1);
        x |= shift_fn(x, 2);
        x |= shift_fn(x, 4);
        x
    }

    /// Isolates the least significant set bit (all other bits set to zero).
    #[inline(always)]
    pub fn isolate_lsb(x: u64) -> u64 {
        x & x.wrapping_neg()
    }

    /// Clears the least significant set bit.
    #[inline(always)]
    pub fn clear_lsb(x: u64) -> u64 {
        x & (x.wrapping_sub(1))
    }

    /// Isolates the most significant set bit (all other bits set to zero).
    #[inline(always)]
    pub fn isolate_msb(x: u64) -> u64 {
        if x == 0 {
            0
        } else {
            1u64 << (63 - x.leading_zeros())
        }
    }

    /// Computes parallel popcount for each of the 8-bit fields in a 64-bit word.
    #[inline(always)]
    pub fn popcount_8(mut x: u64) -> u64 {
        x = x - ((x >> 1) & 0x5555555555555555);
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    }

    /// Computes parallel popcount for each of the 16-bit fields in a 64-bit word.
    #[inline(always)]
    pub fn popcount_16(mut x: u64) -> u64 {
        x = x - ((x >> 1) & 0x5555555555555555);
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
        (x + (x >> 8)) & 0x00FF00FF00FF00FF
    }

    /// Computes parallel popcount for each of the 32-bit fields in a 64-bit word.
    #[inline(always)]
    pub fn popcount_32(mut x: u64) -> u64 {
        x = x - ((x >> 1) & 0x5555555555555555);
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
        x = (x + (x >> 8)) & 0x00FF00FF00FF00FF;
        (x + (x >> 16)) & 0x0000FFFF0000FFFF
    }

    /// Bit isolation mask: sets all bits from the lowest set bit to bit 0.
    #[inline(always)]
    pub fn blsmsk(x: u64) -> u64 {
        x ^ (x.wrapping_sub(1))
    }

    /// SWAR implementation of parallel bit extract (pext).
    /// Extracts bits from `val` at positions specified by `mask` and packs them contiguously at the LSB.
    #[inline]
    pub fn pext(val: u64, mut mask: u64) -> u64 {
        let mut res = 0;
        let mut shift = 0;
        while mask != 0 {
            let lsb = mask & mask.wrapping_neg();
            if (val & lsb) != 0 {
                res |= 1u64 << shift;
            }
            shift += 1;
            mask ^= lsb;
        }
        res
    }

    /// SWAR implementation of parallel bit deposit (pdep).
    /// Deposits contiguous low bits from `val` to positions specified by `mask`.
    #[inline]
    pub fn pdep(mut val: u64, mut mask: u64) -> u64 {
        let mut res = 0;
        while mask != 0 {
            let lsb = mask & mask.wrapping_neg();
            if (val & 1) != 0 {
                res |= lsb;
            }
            val >>= 1;
            mask ^= lsb;
        }
        res
    }
}


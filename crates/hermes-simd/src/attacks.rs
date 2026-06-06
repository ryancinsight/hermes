use hermes_simd_core::bitboard::BitBoardKernel;
use hermes_simd_intrinsics::Magic;

/// Generate Rook attacks for a square given occupancy using the Magic bitboard backend.
#[inline(always)]
pub fn rook_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::rook_attacks(square, occupancy) }
}

/// Generate Bishop attacks for a square given occupancy using the Magic bitboard backend.
#[inline(always)]
pub fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::bishop_attacks(square, occupancy) }
}

/// Generate Queen attacks for a square given occupancy using the Magic bitboard backend.
#[inline(always)]
pub fn queen_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::queen_attacks(square, occupancy) }
}


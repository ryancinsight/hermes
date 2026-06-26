use hermes_simd_core::bitboard::BitBoardKernel;
use hermes_simd_intrinsics::Magic;

// SAFETY (shared by all three wrappers): the `Magic` backend's attack methods
// perform only computed bit math and bounds-checked table indexing — they carry
// no ISA/target-feature precondition and cannot cause undefined behavior for any
// input (an out-of-range `square` trips an array bounds check and panics). The
// safe wrappers below therefore uphold the `unsafe` trait contract for the
// `Magic` impl without an unverified precondition.

/// Generate Rook attacks for a square given occupancy using the Magic bitboard backend.
///
/// # Panics
/// Panics if `square >= 64` (the magic tables are indexed by `square`).
#[inline(always)]
pub fn rook_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::rook_attacks(square, occupancy) }
}

/// Generate Bishop attacks for a square given occupancy using the Magic bitboard backend.
///
/// # Panics
/// Panics if `square >= 64` (the magic tables are indexed by `square`).
#[inline(always)]
pub fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::bishop_attacks(square, occupancy) }
}

/// Generate Queen attacks for a square given occupancy using the Magic bitboard backend.
///
/// # Panics
/// Panics if `square >= 64` (the magic tables are indexed by `square`).
#[inline(always)]
pub fn queen_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::queen_attacks(square, occupancy) }
}

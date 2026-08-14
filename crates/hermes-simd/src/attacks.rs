use hermes_simd_core::bitboard::BitBoardKernel;
use hermes_simd_intrinsics::Magic;

// SAFETY (shared by all three wrappers): the `Magic` backend's attack methods
// perform only computed bit math and validated table indexing — they carry no
// ISA/target-feature precondition and cannot cause undefined behavior for any
// input. The safe wrappers below therefore uphold the `unsafe` trait contract
// inside the `Magic` implementation without an unverified precondition.

/// Generate Rook attacks for a square given occupancy using the Magic bitboard backend.
///
/// # Panics
/// Panics if `square >= 64` (bitboard squares are `0..64`).
#[inline(always)]
#[must_use]
pub fn rook_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::rook_attacks(square, occupancy) }
}

/// Generate Bishop attacks for a square given occupancy using the Magic bitboard backend.
///
/// # Panics
/// Panics if `square >= 64` (bitboard squares are `0..64`).
#[inline(always)]
#[must_use]
pub fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::bishop_attacks(square, occupancy) }
}

/// Generate Queen attacks for a square given occupancy using the Magic bitboard backend.
///
/// # Panics
/// Panics if `square >= 64` (bitboard squares are `0..64`).
#[inline(always)]
#[must_use]
pub fn queen_attacks(square: u8, occupancy: u64) -> u64 {
    unsafe { <Magic as BitBoardKernel>::queen_attacks(square, occupancy) }
}

//! Hybrid sliding attack generation using SWAR ray-casts and Magic lookups.

use super::magic::Magic;
use super::swar::Swar;
use hermes_simd_core::bitboard::BitBoardKernel;

/// ZST marker for hybrid SWAR and Magic Bitboards backend.
///
/// Optimizes sliding attack generation by avoiding Magic table memory accesses
/// when the occupancy contains 1 or fewer blockers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HybridSwarMagic;

impl BitBoardKernel for HybridSwarMagic {
    #[inline(always)]
    unsafe fn rook_attacks(square: u8, occupancy: u64) -> u64 {
        let mask = super::magic::rook_mask(square);
        let relevant_occupancy = occupancy & mask;
        if relevant_occupancy.count_ones() <= 1 {
            <Swar as BitBoardKernel>::rook_attacks(square, occupancy)
        } else {
            <Magic as BitBoardKernel>::rook_attacks(square, occupancy)
        }
    }

    #[inline(always)]
    unsafe fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
        let mask = super::magic::bishop_mask(square);
        let relevant_occupancy = occupancy & mask;
        if relevant_occupancy.count_ones() <= 1 {
            <Swar as BitBoardKernel>::bishop_attacks(square, occupancy)
        } else {
            <Magic as BitBoardKernel>::bishop_attacks(square, occupancy)
        }
    }
}

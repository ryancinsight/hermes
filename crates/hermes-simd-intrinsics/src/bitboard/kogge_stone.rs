//! Kogge-Stone bitboard sliding attack generation.

use hermes_simd_core::bitboard::BitBoardKernel;
use hermes_simd_macros::runtime_dispatch;

#[cfg(target_arch = "aarch64")]
use crate::Neon;
use crate::Scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::{Avx2, Avx512};

/// Portable scalar Kogge-Stone fill (always available reference backend).
pub mod scalar;

/// AVX2 backend: four flood-fill directions computed per 64-bit lane in parallel.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;

/// AVX-512 backend: eight flood-fill directions in one 512-bit register.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx512;

/// NEON backend: paired flood-fill directions per 128-bit register.
#[cfg(target_arch = "aarch64")]
pub mod neon;

/// ZST marker for direction-parallel vectorized Kogge-Stone backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KoggeStone;

// Shift-step helpers for Kogge-Stone scalar
#[inline(always)]
pub(crate) fn east_mask(s: usize) -> u64 {
    match s {
        1 => 0xFEFE_FEFE_FEFE_FEFE,
        2 => 0xFCFC_FCFC_FCFC_FCFC,
        4 => 0xF0F0_F0F0_F0F0_F0F0,
        _ => 0xFFFF_FFFF_FFFF_FFFF,
    }
}

#[inline(always)]
pub(crate) fn west_mask(s: usize) -> u64 {
    match s {
        1 => 0x7F7F_7F7F_7F7F_7F7F,
        2 => 0x3F3F_3F3F_3F3F_3F3F,
        4 => 0x0F0F_0F0F_0F0F_0F0F,
        _ => 0xFFFF_FFFF_FFFF_FFFF,
    }
}

#[inline(always)]
pub(crate) fn step_n(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s * 8;
    let sg = g << shift;
    let sp = p << shift;
    (g | (sg & p), p & sp)
}

#[inline(always)]
pub(crate) fn step_s(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s * 8;
    let sg = g >> shift;
    let sp = p >> shift;
    (g | (sg & p), p & sp)
}

#[inline(always)]
pub(crate) fn step_e(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s;
    let mask = east_mask(s);
    let sg = (g << shift) & mask;
    let sp = (p << shift) & mask;
    (g | (sg & p), p & sp)
}

#[inline(always)]
pub(crate) fn step_w(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s;
    let mask = west_mask(s);
    let sg = (g >> shift) & mask;
    let sp = (p >> shift) & mask;
    (g | (sg & p), p & sp)
}

#[inline(always)]
pub(crate) fn step_ne(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s * 9;
    let mask = east_mask(s);
    let sg = (g << shift) & mask;
    let sp = (p << shift) & mask;
    (g | (sg & p), p & sp)
}

#[inline(always)]
pub(crate) fn step_nw(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s * 7;
    let mask = west_mask(s);
    let sg = (g << shift) & mask;
    let sp = (p << shift) & mask;
    (g | (sg & p), p & sp)
}

#[inline(always)]
pub(crate) fn step_se(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s * 7;
    let mask = east_mask(s);
    let sg = (g >> shift) & mask;
    let sp = (p >> shift) & mask;
    (g | (sg & p), p & sp)
}

#[inline(always)]
pub(crate) fn step_sw(g: u64, p: u64, s: usize) -> (u64, u64) {
    let shift = s * 9;
    let mask = west_mask(s);
    let sg = (g >> shift) & mask;
    let sp = (p >> shift) & mask;
    (g | (sg & p), p & sp)
}

/// Per-ISA Kogge-Stone sliding-attack fill, one method per attack set.
///
/// The methods are `unsafe` because a vectorized backend's precondition is that
/// its instruction set is present on the host. `#[runtime_dispatch]` selects the
/// marker whose ISA is available and enters that marker's `#[target_feature]`
/// frame, which is what discharges the obligation at each call.
trait KoggeFill {
    /// # Safety
    /// Caller must ensure the backend's ISA is available.
    unsafe fn rook(slider: u64, occupancy: u64) -> u64;

    /// # Safety
    /// Caller must ensure the backend's ISA is available.
    unsafe fn bishop(slider: u64, occupancy: u64) -> u64;

    /// Queen attacks default to the union of the rook and bishop fills.
    ///
    /// # Safety
    /// Caller must ensure the backend's ISA is available.
    unsafe fn queen(slider: u64, occupancy: u64) -> u64 {
        unsafe { Self::rook(slider, occupancy) | Self::bishop(slider, occupancy) }
    }
}

impl KoggeFill for Scalar {
    #[inline]
    unsafe fn rook(slider: u64, occupancy: u64) -> u64 {
        scalar::kogge_stone_rook(slider, occupancy)
    }

    #[inline]
    unsafe fn bishop(slider: u64, occupancy: u64) -> u64 {
        scalar::kogge_stone_bishop(slider, occupancy)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl KoggeFill for Avx2 {
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn rook(slider: u64, occupancy: u64) -> u64 {
        avx2::kogge_stone_rook_avx2(slider, occupancy)
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn bishop(slider: u64, occupancy: u64) -> u64 {
        avx2::kogge_stone_bishop_avx2(slider, occupancy)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl KoggeFill for Avx512 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn rook(slider: u64, occupancy: u64) -> u64 {
        avx512::kogge_stone_rook_avx512(slider, occupancy)
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn bishop(slider: u64, occupancy: u64) -> u64 {
        avx512::kogge_stone_bishop_avx512(slider, occupancy)
    }

    // The AVX-512 queen fill is a genuine two-times kernel — the rook fill alone
    // leaves lanes 2,3,6,7 as masked padding, so the fused body really uses all
    // eight lanes — so it keeps a method of its own instead of the default union.
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn queen(slider: u64, occupancy: u64) -> u64 {
        avx512::kogge_stone_queen_avx512(slider, occupancy)
    }
}

#[cfg(target_arch = "aarch64")]
impl KoggeFill for Neon {
    #[inline]
    unsafe fn rook(slider: u64, occupancy: u64) -> u64 {
        neon::kogge_stone_rook_neon(slider, occupancy)
    }

    #[inline]
    unsafe fn bishop(slider: u64, occupancy: u64) -> u64 {
        neon::kogge_stone_bishop_neon(slider, occupancy)
    }
}

/// Rook attacks for `square`, generated for the widest ISA the host implements.
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
fn dispatch_rook_attacks_kernel<A: KoggeFill>(square: u8, occupancy: u64) -> u64 {
    let slider = 1u64 << square;
    // SAFETY: `A` is the marker for the ISA whose feature frame
    // `#[runtime_dispatch]` selected, so `A::rook`'s ISA precondition holds.
    unsafe { A::rook(slider, occupancy) }
}

/// Bishop attacks for `square`, generated for the widest ISA the host implements.
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
fn dispatch_bishop_attacks_kernel<A: KoggeFill>(square: u8, occupancy: u64) -> u64 {
    let slider = 1u64 << square;
    // SAFETY: as for `dispatch_rook_attacks_kernel`.
    unsafe { A::bishop(slider, occupancy) }
}

/// Queen attacks for `square`, generated for the widest ISA the host implements.
#[runtime_dispatch(avx512f, avx2, neon, scalar)]
fn dispatch_queen_attacks_kernel<A: KoggeFill>(square: u8, occupancy: u64) -> u64 {
    let slider = 1u64 << square;
    // SAFETY: as for `dispatch_rook_attacks_kernel`.
    unsafe { A::queen(slider, occupancy) }
}

impl BitBoardKernel for KoggeStone {
    #[inline(always)]
    fn rook_attacks(square: u8, occupancy: u64) -> u64 {
        dispatch_rook_attacks(square, occupancy)
    }

    #[inline(always)]
    fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
        dispatch_bishop_attacks(square, occupancy)
    }

    #[inline(always)]
    fn queen_attacks(square: u8, occupancy: u64) -> u64 {
        dispatch_queen_attacks(square, occupancy)
    }
}

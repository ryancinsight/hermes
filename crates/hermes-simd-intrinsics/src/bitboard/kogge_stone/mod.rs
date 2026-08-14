//! Kogge-Stone bitboard sliding attack generation.

use hermes_simd_core::bitboard::BitBoardKernel;

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
        1 => 0xFEFEFEFEFEFEFEFE,
        2 => 0xFCFCFCFCFCFCFCFC,
        4 => 0xF0F0F0F0F0F0F0F0,
        _ => 0xFFFFFFFFFFFFFFFF,
    }
}

#[inline(always)]
pub(crate) fn west_mask(s: usize) -> u64 {
    match s {
        1 => 0x7F7F7F7F7F7F7F7F,
        2 => 0x3F3F3F3F3F3F3F3F,
        4 => 0x0F0F0F0F0F0F0F0F,
        _ => 0xFFFFFFFFFFFFFFFF,
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

impl BitBoardKernel for KoggeStone {
    #[inline(always)]
    #[cfg_attr(
        target_arch = "aarch64",
        expect(
            unreachable_code,
            reason = "Architecture-specific returns are cfg-selected before scalar fallback"
        )
    )]
    fn rook_attacks(square: u8, occupancy: u64) -> u64 {
        let slider = 1u64 << square;

        // SAFETY: each ISA-specific fill (`kogge_stone_rook_avx512`/`_avx2`) is
        // `#[target_feature]`-gated and is only reached inside the matching
        // `is_x86_feature_detected!` branch (or the `cfg!(target_feature)` guard in
        // no-std), so its ISA precondition holds at the call. The NEON path is baseline
        // on AArch64, and the scalar fallback is safe. Inputs are `u64` bitboards with
        // no memory access, so no bounds obligation applies.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "std")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    // SAFETY: `kogge_stone_rook_avx512` is `#[target_feature(enable = "avx512f")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx512::kogge_stone_rook_avx512(slider, occupancy) };
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: `kogge_stone_rook_avx2` is `#[target_feature(enable = "avx2")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx2::kogge_stone_rook_avx2(slider, occupancy) };
                }
            }
            #[cfg(not(feature = "std"))]
            {
                if cfg!(target_feature = "avx512f") {
                    // SAFETY: `kogge_stone_rook_avx512` is `#[target_feature(enable = "avx512f")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx512::kogge_stone_rook_avx512(slider, occupancy) };
                }
                if cfg!(target_feature = "avx2") {
                    // SAFETY: `kogge_stone_rook_avx2` is `#[target_feature(enable = "avx2")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx2::kogge_stone_rook_avx2(slider, occupancy) };
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `kogge_stone_rook_neon` requires `neon`, which is baseline on
            // every aarch64 target, so its precondition holds unconditionally here.
            return unsafe { neon::kogge_stone_rook_neon(slider, occupancy) };
        }

        scalar::kogge_stone_rook(slider, occupancy)
    }

    #[inline(always)]
    #[cfg_attr(
        target_arch = "aarch64",
        expect(
            unreachable_code,
            reason = "Architecture-specific returns are cfg-selected before scalar fallback"
        )
    )]
    fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
        let slider = 1u64 << square;

        // SAFETY: each ISA-specific fill (`kogge_stone_bishop_avx512`/`_avx2`) is
        // `#[target_feature]`-gated and is only reached inside the matching
        // `is_x86_feature_detected!` branch (or the `cfg!(target_feature)` guard in
        // no-std), so its ISA precondition holds at the call. The NEON path is baseline
        // on AArch64, and the scalar fallback is safe. Inputs are `u64` bitboards with
        // no memory access, so no bounds obligation applies.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "std")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    // SAFETY: `kogge_stone_bishop_avx512` is `#[target_feature(enable = "avx512f")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx512::kogge_stone_bishop_avx512(slider, occupancy) };
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: `kogge_stone_bishop_avx2` is `#[target_feature(enable = "avx2")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx2::kogge_stone_bishop_avx2(slider, occupancy) };
                }
            }
            #[cfg(not(feature = "std"))]
            {
                if cfg!(target_feature = "avx512f") {
                    // SAFETY: `kogge_stone_bishop_avx512` is `#[target_feature(enable = "avx512f")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx512::kogge_stone_bishop_avx512(slider, occupancy) };
                }
                if cfg!(target_feature = "avx2") {
                    // SAFETY: `kogge_stone_bishop_avx2` is `#[target_feature(enable = "avx2")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx2::kogge_stone_bishop_avx2(slider, occupancy) };
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `kogge_stone_bishop_neon` requires `neon`, which is baseline on
            // every aarch64 target, so its precondition holds unconditionally here.
            return unsafe { neon::kogge_stone_bishop_neon(slider, occupancy) };
        }

        scalar::kogge_stone_bishop(slider, occupancy)
    }

    #[inline(always)]
    #[cfg_attr(
        target_arch = "aarch64",
        expect(
            unreachable_code,
            reason = "Architecture-specific returns are cfg-selected before scalar fallback"
        )
    )]
    fn queen_attacks(square: u8, occupancy: u64) -> u64 {
        let slider = 1u64 << square;

        // SAFETY: each ISA-specific fill (`kogge_stone_queen_avx512`/`_avx2`) is
        // `#[target_feature]`-gated and is only reached inside the matching
        // `is_x86_feature_detected!` branch (or the `cfg!(target_feature)` guard in
        // no-std), so its ISA precondition holds at the call. The NEON path is baseline
        // on AArch64, and the scalar fallback is safe. Inputs are `u64` bitboards with
        // no memory access, so no bounds obligation applies.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "std")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    // SAFETY: `kogge_stone_queen_avx512` is `#[target_feature(enable = "avx512f")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx512::kogge_stone_queen_avx512(slider, occupancy) };
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: `kogge_stone_queen_avx2` is `#[target_feature(enable = "avx2")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx2::kogge_stone_queen_avx2(slider, occupancy) };
                }
            }
            #[cfg(not(feature = "std"))]
            {
                if cfg!(target_feature = "avx512f") {
                    // SAFETY: `kogge_stone_queen_avx512` is `#[target_feature(enable = "avx512f")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx512::kogge_stone_queen_avx512(slider, occupancy) };
                }
                if cfg!(target_feature = "avx2") {
                    // SAFETY: `kogge_stone_queen_avx2` is `#[target_feature(enable = "avx2")]`-gated and this
                    // branch is entered only when that feature is present — probed by
                    // `is_x86_feature_detected!` under std, by `cfg!(target_feature)` otherwise.
                    return unsafe { avx2::kogge_stone_queen_avx2(slider, occupancy) };
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: `kogge_stone_queen_neon` requires `neon`, which is baseline on
            // every aarch64 target, so its precondition holds unconditionally here.
            return unsafe { neon::kogge_stone_queen_neon(slider, occupancy) };
        }

        scalar::kogge_stone_queen(slider, occupancy)
    }
}

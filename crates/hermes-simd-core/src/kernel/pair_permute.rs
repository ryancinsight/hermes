//! Portable stride-`N` decimation and interleave of register arrays at
//! adjacent-lane-pair granularity.
//!
//! Read `N` registers as one flat sequence of lane pairs — interleaved
//! complex samples, one sample per pair. The stride-`N` **decimation** sends
//! flat pair `N q + k` to pair `q` of output `k`: output `k` is the
//! subsequence of pairs congruent to `k` modulo `N`. The **interleave** is its
//! inverse, sending pair `q` of operand `k` to flat pair `N q + k`. On complex
//! data these are the radix-`N` gather and scatter of a mixed-radix transform,
//! and at `N = LANE_COUNT / 2` the decimation is the transpose of a square
//! complex tile.
//!
//! [`BackendKernel::deinterleave_pairs`] and
//! [`BackendKernel::interleave_pairs`] default to [`deinterleave`] and
//! [`interleave`]. A backend overriding either specializes the arities it has
//! a dedicated network for, branching on `N` (a constant, so every
//! instantiation keeps exactly one arm), and forwards the remaining arities
//! here.
//!
//! # Composition
//!
//! For a power of two `N > 2` the decimation is `log2 N` levels of the
//! backend's own two-register decimation: after the level of width `w`, every
//! aligned block of `w` registers holds the stride-`w` decimation of its
//! input, because the two stride-`w/2` decimations in its halves combine
//! pairwise — output `k` of the block is the even half of
//! `(x_k, y_k)` and output `k + w/2` the odd half. The interleave runs the
//! same levels in reverse with the two-register interleave. Other arities,
//! and `N = 2` itself, use scalar lane emulation, so the composition never
//! recurses into the backend's `N = 2` arm through this module.

use super::{BackendKernel, MAX_SIMD_LANES};
use crate::scalar::Scalar;
use core::mem::MaybeUninit;

/// Stride-`N` pair decimation of `regs` through the backend's two-register
/// decimation where `N` is a power of two above two, and scalar lane
/// emulation otherwise.
///
/// The default of [`BackendKernel::deinterleave_pairs`] and the fallback a
/// backend override forwards the arities it does not specialize.
///
/// # Safety
/// The backend's target features must be available.
#[inline(always)]
#[must_use]
pub unsafe fn deinterleave<T: Scalar, B: BackendKernel<T>, const N: usize>(
    regs: [B::Vector; N],
) -> [B::Vector; N] {
    // SAFETY: the caller's feature obligation covers both routes.
    unsafe {
        if N > 2 && N.is_power_of_two() {
            decimate_by_levels::<T, B, N>(regs)
        } else {
            decimate_lanes::<T, B, N>(regs)
        }
    }
}

/// Stride-`N` pair interleave of `regs`, the inverse of [`deinterleave`],
/// through the backend's two-register interleave where `N` is a power of two
/// above two, and scalar lane emulation otherwise.
///
/// The default of [`BackendKernel::interleave_pairs`] and the fallback a
/// backend override forwards the arities it does not specialize.
///
/// # Safety
/// The backend's target features must be available.
#[inline(always)]
#[must_use]
pub unsafe fn interleave<T: Scalar, B: BackendKernel<T>, const N: usize>(
    regs: [B::Vector; N],
) -> [B::Vector; N] {
    // SAFETY: the caller's feature obligation covers both routes.
    unsafe {
        if N > 2 && N.is_power_of_two() {
            interleave_by_levels::<T, B, N>(regs)
        } else {
            interleave_lanes::<T, B, N>(regs)
        }
    }
}

/// Re-types a register array whose length the caller's `N == M` branch has
/// already established, so a backend arm specialized for one arity can name
/// its registers.
///
/// # Panics
/// When `N != M`. The comparison is between constants, so in an `N == M` arm
/// it folds away and no check remains.
#[inline(always)]
#[must_use]
#[track_caller]
pub fn cast_arity<V: Copy, const N: usize, const M: usize>(regs: [V; N]) -> [V; M] {
    assert!(
        N == M,
        "invariant: cast_arity is called only from an `N == M` arm"
    );
    // SAFETY: `N == M`, so `[V; N]` and `[V; M]` are one layout, and `V: Copy`
    // means the read duplicates no ownership.
    unsafe { core::ptr::from_ref(&regs).cast::<[V; M]>().read() }
}

/// `log2 N` levels of the two-register decimation; see the module docs.
///
/// # Safety
/// The backend's target features must be available.
#[inline(always)]
unsafe fn decimate_by_levels<T: Scalar, B: BackendKernel<T>, const N: usize>(
    mut regs: [B::Vector; N],
) -> [B::Vector; N] {
    let mut width = 2;
    while width <= N {
        for block in regs.chunks_exact_mut(width) {
            let (low, high) = block.split_at_mut(width / 2);
            for (x, y) in low.iter_mut().zip(high) {
                // SAFETY: the caller's feature obligation.
                [*x, *y] = unsafe { B::deinterleave_pairs([*x, *y]) };
            }
        }
        width *= 2;
    }
    regs
}

/// The levels of [`decimate_by_levels`] in reverse, each inverted.
///
/// # Safety
/// The backend's target features must be available.
#[inline(always)]
unsafe fn interleave_by_levels<T: Scalar, B: BackendKernel<T>, const N: usize>(
    mut regs: [B::Vector; N],
) -> [B::Vector; N] {
    let mut width = N;
    while width >= 2 {
        for block in regs.chunks_exact_mut(width) {
            let (low, high) = block.split_at_mut(width / 2);
            for (x, y) in low.iter_mut().zip(high) {
                // SAFETY: the caller's feature obligation.
                [*x, *y] = unsafe { B::interleave_pairs([*x, *y]) };
            }
        }
        width /= 2;
    }
    regs
}

/// Scalar emulation of the decimation: output `k`, pair `q` is flat pair
/// `N q + k`.
///
/// # Safety
/// The backend's target features must be available.
#[inline(always)]
unsafe fn decimate_lanes<T: Scalar, B: BackendKernel<T>, const N: usize>(
    regs: [B::Vector; N],
) -> [B::Vector; N] {
    // SAFETY: the caller's feature obligation.
    unsafe { permute_lanes::<T, B, N>(regs, |out, q| q * N + out) }
}

/// Scalar emulation of the interleave: output `j`, pair `q` is flat pair
/// `j P + q` for `P` pairs a register, which is pair `(j P + q) / N` of
/// operand `(j P + q) mod N`.
///
/// # Safety
/// The backend's target features must be available.
#[inline(always)]
unsafe fn interleave_lanes<T: Scalar, B: BackendKernel<T>, const N: usize>(
    regs: [B::Vector; N],
) -> [B::Vector; N] {
    let pairs = B::LANE_COUNT / 2;
    // SAFETY: the caller's feature obligation.
    unsafe {
        permute_lanes::<T, B, N>(regs, |out, q| {
            let flat = out * pairs + q;
            (flat % N) * pairs + flat / N
        })
    }
}

/// Moves whole lane pairs between `N` registers: pair `q` of output `out`
/// takes source pair `source(out, q)`, counted across the concatenated
/// operands.
///
/// # Safety
/// The backend's target features must be available, and `source` must map
/// into `0..N * LANE_COUNT / 2`.
#[inline(always)]
unsafe fn permute_lanes<T: Scalar, B: BackendKernel<T>, const N: usize>(
    regs: [B::Vector; N],
    source: impl Fn(usize, usize) -> usize,
) -> [B::Vector; N] {
    const { B::LANE_BOUND_CHECK };
    let lanes = B::LANE_COUNT;
    debug_assert!(
        lanes.is_multiple_of(2),
        "pair granularity needs whole pairs"
    );
    let pairs = lanes / 2;
    let mut src = [[MaybeUninit::<T>::uninit(); MAX_SIMD_LANES]; N];
    for (buf, reg) in src.iter_mut().zip(regs) {
        // SAFETY: the buffer holds `MAX_SIMD_LANES >= LANE_COUNT` lanes
        // (`LANE_BOUND_CHECK`), and the caller's feature obligation.
        unsafe { B::store_unaligned(buf.as_mut_ptr().cast::<T>(), reg) };
    }
    let mut out = [[MaybeUninit::<T>::uninit(); MAX_SIMD_LANES]; N];
    for (k, buf) in out.iter_mut().enumerate() {
        for q in 0..pairs {
            let flat = source(k, q);
            let (reg, pair) = (flat / pairs, flat % pairs);
            buf[2 * q] = src[reg][2 * pair];
            buf[2 * q + 1] = src[reg][2 * pair + 1];
        }
    }
    // The loads read lanes `0..LANE_COUNT` of each output buffer, all of which
    // the loop above wrote from lanes the stores initialized.
    let mut result = regs;
    for (reg, buf) in result.iter_mut().zip(&out) {
        // SAFETY: as above, and the caller's feature obligation.
        *reg = unsafe { B::load_unaligned(buf.as_ptr().cast::<T>()) };
    }
    result
}

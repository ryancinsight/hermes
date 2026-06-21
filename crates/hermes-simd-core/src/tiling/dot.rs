//! Register-blocked dot product micro-kernel.
//!
//! # Theorem (dependency-chain throughput)
//! A naïve dot loop keeps a single accumulator, so consecutive fused
//! multiply-adds form a loop-carried dependency chain bounded by the FMA latency
//! `L` (≈4 cycles on current x86): throughput is one FMA per `L` cycles
//! regardless of how many FMA issue ports `P` the core has. Splitting the
//! reduction into `TILE_M` independent accumulators
//! `sⱼ ← sⱼ + a_{iM+j}·b_{iM+j}` (`j = 0..TILE_M`) places `TILE_M` mutually
//! independent FMAs in flight per iteration, so throughput rises to
//! `min(P, TILE_M / L)` FMAs/cycle. Saturation therefore needs
//! `TILE_M ≥ P·L`.
//!
//! ## Proof sketch (accuracy is preserved)
//! The tiled result is `Σⱼ sⱼ` where each `sⱼ` sums `⌈n/TILE_M⌉` products. This
//! reassociates the IEEE-754 sum, which is not associative, so the rounded value
//! may differ from the sequential one. The forward error of summing `n` terms is
//! `|Σ̂ − Σ| ≤ (n−1)·ε·Σ|tᵢ| + O(ε²)`; partitioning into `TILE_M` chains of length
//! `n/TILE_M` and one final length-`TILE_M` reduction gives depth
//! `n/TILE_M + TILE_M`, which for fixed `TILE_M` is still `O(n·ε)` — the same
//! asymptotic bound. Hence `TILE_M` is a throughput knob with no accuracy-class
//! regression. Verified empirically by the differential tests in
//! `tiling_tests.rs` (bounded-epsilon equality against the scalar reduction).

use crate::{
    align::Alignment,
    arch::SimdArch,
    kernel::SimdKernel,
    scalar::Scalar,
    view::{SimdError, SimdView},
};

/// Compute `Σᵢ aᵢ·bᵢ` using `TILE_M` independent vector accumulators.
///
/// # Errors
/// [`SimdError::LengthMismatch`] if `a.len() != b.len()`.
#[inline]
pub(super) fn dot_impl<T, Arch, Align, const TILE_M: usize>(
    a: &SimdView<'_, T, Arch, Align>,
    b: &SimdView<'_, T, Arch, Align>,
) -> Result<T, SimdError>
where
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
    T: Scalar,
{
    struct AssertM<const TILE_M: usize>;
    impl<const TILE_M: usize> AssertM<TILE_M> {
        const OK: () = assert!(TILE_M >= 1, "TILE_M must be at least 1");
    }
    let _ = AssertM::<TILE_M>::OK;

    crate::view::check_lengths_equal(a.len(), b.len())?;

    let len = a.len();
    let lane_count = Arch::LANE_COUNT;
    let tile_width = lane_count * TILE_M;
    let tiled_len = (len / tile_width) * tile_width;

    let load = |ptr: *const T| -> Arch::Vector {
        if crate::align::is_aligned_for_arch::<Arch, Align>() {
            unsafe { Arch::load_aligned(ptr) }
        } else {
            unsafe { Arch::load_unaligned(ptr) }
        }
    };

    let mut ptr_a = a.as_slice().as_ptr();
    let mut ptr_b = b.as_slice().as_ptr();

    // TILE_M independent accumulators initialized via mul (not zero+fmadd)
    // to avoid an extra dependency on the zero register.
    let mut accumulators: [Arch::Vector; TILE_M] = {
        let mut arr = [unsafe { Arch::zero() }; TILE_M];
        if tiled_len > 0 {
            for i in 0..TILE_M {
                let va = load(unsafe { ptr_a.add(i * lane_count) });
                let vb = load(unsafe { ptr_b.add(i * lane_count) });
                arr[i] = unsafe { Arch::mul(va, vb) };
            }
            unsafe {
                ptr_a = ptr_a.add(tile_width);
                ptr_b = ptr_b.add(tile_width);
            }
        }
        arr
    };

    if tiled_len > tile_width {
        let iterations = (tiled_len / tile_width) - 1;
        for _ in 0..iterations {
            for i in 0..TILE_M {
                let va = load(unsafe { ptr_a.add(i * lane_count) });
                let vb = load(unsafe { ptr_b.add(i * lane_count) });
                accumulators[i] = unsafe { Arch::fmadd(va, vb, accumulators[i]) };
            }
            unsafe {
                ptr_a = ptr_a.add(tile_width);
                ptr_b = ptr_b.add(tile_width);
            }
        }
    }

    // Horizontal reduce across TILE_M accumulators.
    let mut total = T::ZERO;
    if tiled_len > 0 {
        let mut combined = accumulators[0];
        for i in 1..TILE_M {
            combined = unsafe { Arch::add(combined, accumulators[i]) };
        }
        total = unsafe { Arch::sum_reduce(combined) };
    }

    // Scalar tail.
    let a_slice = a.as_slice();
    let b_slice = b.as_slice();
    for i in tiled_len..len {
        total += a_slice[i] * b_slice[i];
    }

    Ok(total)
}

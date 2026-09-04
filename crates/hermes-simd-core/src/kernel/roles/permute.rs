//! Cross-lane permutation capability facet for a SIMD backend.

use crate::kernel::BackendKernel;
use crate::ops::{ScanMode, ScanOp};
use crate::private::Sealed;
use crate::scalar::Scalar;

use super::storage::SimdStorage;

/// Backend capability for scans, lane permutations, and adjacent shuffles.
pub trait SimdPermute<T: Scalar>: SimdStorage<T> + Sealed {
    /// Performs an inclusive or exclusive intra-register scan.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn scan_vector<Op: ScanOp<T>, SMode: ScanMode>(
        v: Self::Vector,
        carry: T,
    ) -> (Self::Vector, T);

    /// Reverses the lane order.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn reverse(v: Self::Vector) -> Self::Vector;

    /// Interleaves two registers into low and high result registers.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector);

    /// Deinterleaves two registers into even and odd result registers.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector);

    /// Deinterleaves two registers at adjacent-lane-pair granularity into
    /// even-pair and odd-pair result registers.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn deinterleave_pairs(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector);

    /// Reassembles the even-pair and odd-pair registers produced by
    /// [`SimdPermute::deinterleave_pairs`] into the original operand pair.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn interleave_pairs(
        even: Self::Vector,
        odd: Self::Vector,
    ) -> (Self::Vector, Self::Vector);

    /// Splits four registers' adjacent-lane pairs into the four stride-4
    /// subsequences.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn deinterleave_pairs4(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        d: Self::Vector,
    ) -> (Self::Vector, Self::Vector, Self::Vector, Self::Vector);

    /// Splits eight registers' adjacent-lane pairs into the eight stride-8
    /// subsequences.
    ///
    /// The register-width form of the strided gather a mixed-radix transform
    /// performs between passes. See
    /// [`BackendKernel::deinterleave_pairs8`](crate::kernel::BackendKernel::deinterleave_pairs8).
    ///
    /// # Safety
    /// The backend's target features must be available.
    #[expect(
        clippy::too_many_arguments,
        reason = "eight registers is the operation's arity, not a parameter list"
    )]
    unsafe fn deinterleave_pairs8(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        d: Self::Vector,
        e: Self::Vector,
        f: Self::Vector,
        g: Self::Vector,
        h: Self::Vector,
    ) -> [Self::Vector; 8];

    /// Concatenates the two registers' low halves, and their high halves.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn interleave_halves(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector);

    /// Broadcasts one lane pair across the register: `[lo, hi, lo, hi, ...]`.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn splat_pair(lo: T, hi: T) -> Self::Vector;

    /// Concatenates the low half of `a` with the high half of `b`.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn blend_halves(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Swaps each adjacent lane pair.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector;

    /// Swaps each adjacent lane *pair* with its neighbouring pair; a trailing
    /// pair with no neighbour passes through unchanged.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn swap_pairs(v: Self::Vector) -> Self::Vector;

    /// Transposes a square tile of `LANE_COUNT` vectors in place: lane `c`
    /// of row `r` moves to lane `r` of row `c`.
    ///
    /// # Safety
    /// The backend's target features must be available; `tile` must hold
    /// exactly `LANE_COUNT` vectors.
    unsafe fn transpose_square(tile: &mut [Self::Vector]);

    /// Transposes a square tile of interleaved complex registers in place.
    ///
    /// Each vector is one row of `LANE_COUNT / 2` complex samples. Sample
    /// `(r, c)` moves to `(c, r)` while its adjacent real/imaginary lanes stay
    /// paired.
    ///
    /// # Safety
    /// The backend's target features must be available; `tile` must hold
    /// exactly `LANE_COUNT / 2` vectors.
    unsafe fn transpose_interleaved_square(tile: &mut [Self::Vector]);

    /// Duplicates each even lane into its adjacent odd lane.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn dup_even(v: Self::Vector) -> Self::Vector;

    /// Duplicates each odd lane into its adjacent even lane.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector;

    /// Computes alternating fused multiply-add/subtract lanes.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;

    /// Computes alternating fused multiply-subtract/add lanes.
    ///
    /// # Safety
    /// The backend's target features must be available.
    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;
}

impl<T: Scalar, A: BackendKernel<T>> SimdPermute<T> for A {
    unsafe fn scan_vector<Op: ScanOp<T>, SMode: ScanMode>(
        v: Self::Vector,
        carry: T,
    ) -> (Self::Vector, T) {
        <A as BackendKernel<T>>::scan_vector::<Op, SMode>(v, carry)
    }

    unsafe fn reverse(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::reverse(v)
    }

    unsafe fn interleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::interleave(a, b)
    }

    unsafe fn deinterleave(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::deinterleave(a, b)
    }

    unsafe fn deinterleave_pairs(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::deinterleave_pairs(a, b)
    }

    unsafe fn interleave_pairs(
        even: Self::Vector,
        odd: Self::Vector,
    ) -> (Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::interleave_pairs(even, odd)
    }

    unsafe fn deinterleave_pairs4(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        d: Self::Vector,
    ) -> (Self::Vector, Self::Vector, Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::deinterleave_pairs4(a, b, c, d)
    }

    unsafe fn deinterleave_pairs8(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        d: Self::Vector,
        e: Self::Vector,
        f: Self::Vector,
        g: Self::Vector,
        h: Self::Vector,
    ) -> [Self::Vector; 8] {
        <A as BackendKernel<T>>::deinterleave_pairs8(a, b, c, d, e, f, g, h)
    }

    unsafe fn interleave_halves(a: Self::Vector, b: Self::Vector) -> (Self::Vector, Self::Vector) {
        <A as BackendKernel<T>>::interleave_halves(a, b)
    }

    unsafe fn splat_pair(lo: T, hi: T) -> Self::Vector {
        <A as BackendKernel<T>>::splat_pair(lo, hi)
    }

    unsafe fn blend_halves(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::blend_halves(a, b)
    }

    unsafe fn swap_adjacent(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::swap_adjacent(v)
    }

    unsafe fn swap_pairs(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::swap_pairs(v)
    }

    unsafe fn transpose_square(tile: &mut [Self::Vector]) {
        <A as BackendKernel<T>>::transpose_square(tile);
    }

    unsafe fn transpose_interleaved_square(tile: &mut [Self::Vector]) {
        <A as BackendKernel<T>>::transpose_interleaved_square(tile);
    }

    unsafe fn dup_even(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::dup_even(v)
    }

    unsafe fn dup_odd(v: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::dup_odd(v)
    }

    unsafe fn fmaddsub(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::fmaddsub(a, b, c)
    }

    unsafe fn fmsubadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        <A as BackendKernel<T>>::fmsubadd(a, b, c)
    }
}

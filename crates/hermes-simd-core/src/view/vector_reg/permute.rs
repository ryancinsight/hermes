//! Lane permutations: interleave/deinterleave, halves, shifts and transposes.

use super::Vector;
use crate::arch::SimdArch;
use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

impl<T, Arch> Vector<T, Arch>
where
    Arch: SimdArch + SimdKernel<T>,
    T: Scalar,
{
    /// Reverses the lane order.
    #[inline(always)]
    #[must_use]
    pub fn reverse(self) -> Self {
        // SAFETY: constructing `self` proved host support for `Arch`.
        Self::new(unsafe { Arch::reverse(self.raw) })
    }

    /// Interleaves the lanes of `self` and `other`, returning the low and high
    /// halves of the interleaved sequence.
    ///
    /// This is the array-of-structures direction: given planar real and
    /// imaginary vectors it produces interleaved complex samples.
    #[inline(always)]
    #[must_use]
    pub fn interleave(self, other: Self) -> (Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (lo, hi) = unsafe { Arch::interleave(self.raw, other.raw) };
        (Self::new(lo), Self::new(hi))
    }

    /// Deinterleaves two vectors into even-indexed and odd-indexed lanes.
    ///
    /// The inverse of [`Vector::interleave`]: given interleaved complex samples
    /// it produces planar real and imaginary vectors.
    #[inline(always)]
    #[must_use]
    pub fn deinterleave(self, other: Self) -> (Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (even, odd) = unsafe { Arch::deinterleave(self.raw, other.raw) };
        (Self::new(even), Self::new(odd))
    }

    /// Interleaves `self` and `other` within each sub-lane of
    /// [`SUBLANE_LANES`](crate::kernel::SimdPermute::SUBLANE_LANES) lanes.
    ///
    /// The one-instruction form of [`Vector::interleave`] on x86, where the
    /// `unpack` instructions weave within 128-bit sub-lanes; identical to
    /// it where the register is one sub-lane. A kernel that stores its
    /// planar data in sub-lane order gets the interleaved layout from this
    /// alone.
    #[inline(always)]
    #[must_use]
    pub fn interleave_sublanes(self, other: Self) -> (Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (lo, hi) = unsafe { Arch::interleave_sublanes(self.raw, other.raw) };
        (Self::new(lo), Self::new(hi))
    }

    /// Deinterleaves two vectors within each sub-lane, the inverse of
    /// [`Vector::interleave_sublanes`].
    #[inline(always)]
    #[must_use]
    pub fn deinterleave_sublanes(self, other: Self) -> (Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (even, odd) = unsafe { Arch::deinterleave_sublanes(self.raw, other.raw) };
        (Self::new(even), Self::new(odd))
    }

    /// Deinterleaves two vectors at adjacent-lane-pair granularity: reading
    /// `self || other` as a flat sequence of lane pairs, the results hold the
    /// even-indexed and odd-indexed pairs.
    ///
    /// On interleaved complex data this splits a stride-2 complex decimation:
    /// the even and odd complex samples of the concatenated registers, each
    /// with its real/imaginary lanes still adjacent.
    #[inline(always)]
    #[must_use]
    pub fn deinterleave_pairs(self, other: Self) -> (Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (even, odd) = unsafe { Arch::deinterleave_pairs(self.raw, other.raw) };
        (Self::new(even), Self::new(odd))
    }

    /// Reassembles the even-pair and odd-pair vectors produced by
    /// [`Vector::deinterleave_pairs`] into the original operand pair.
    ///
    /// On interleaved complex data this is the radix-2 complex interleave: the
    /// inverse of a stride-2 decimation, run in registers rather than through
    /// a stack buffer.
    #[inline(always)]
    #[must_use]
    pub fn interleave_pairs(self, other: Self) -> (Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (first, second) = unsafe { Arch::interleave_pairs(self.raw, other.raw) };
        (Self::new(first), Self::new(second))
    }

    /// Interleaves three registers' adjacent-lane pairs into the flat
    /// sequence `self0 b0 c0 self1 b1 c1 ...`, three registers long: the
    /// inverse of a stride-3 pair decimation, and the store shape of a
    /// three-arm scatter whose registers hold one group per pair.
    ///
    /// Six shuffles on AVX2 at `f32` (two lane-local unpacks, one pair
    /// shuffle, three half permutes), three half permutes at `f64`, three
    /// half combines on NEON, scalar emulation by default.
    #[inline(always)]
    #[must_use]
    pub fn interleave_pairs3(self, b: Self, c: Self) -> (Self, Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (x, y, z) = unsafe { Arch::interleave_pairs3(self.raw, b.raw, c.raw) };
        (Self::new(x), Self::new(y), Self::new(z))
    }

    /// Interleaves five registers' adjacent-lane pairs into the flat
    /// sequence `self0 b0 c0 d0 e0 self1 b1 c1 d1 e1 ...`, five registers
    /// long: the inverse of a stride-5 pair decimation, and the transpose
    /// that writes `n = 5 m` in natural order after five `m`-point
    /// transforms held one row a register.
    ///
    /// Ten shuffles on AVX2 at `f32` (four lane-local unpacks, two blends,
    /// four half permutes), five half permutes at `f64`, scalar emulation by
    /// default.
    #[inline(always)]
    #[must_use]
    pub fn interleave_pairs5(self, b: Self, c: Self, d: Self, e: Self) -> [Self; 5] {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let [r0, r1, r2, r3, r4] =
            unsafe { Arch::interleave_pairs5(self.raw, b.raw, c.raw, d.raw, e.raw) };
        [
            Self::new(r0),
            Self::new(r1),
            Self::new(r2),
            Self::new(r3),
            Self::new(r4),
        ]
    }

    /// Splits four vectors' adjacent-lane pairs into the four stride-4
    /// subsequences: reading the concatenation as a flat pair sequence,
    /// output `i` holds the pairs congruent to `i` modulo 4.
    ///
    /// On interleaved complex data this is the radix-4 complex decimation of
    /// four registers in one operation.
    #[inline(always)]
    #[must_use]
    pub fn deinterleave_pairs4(self, b: Self, c: Self, d: Self) -> (Self, Self, Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (r0, r1, r2, r3) = unsafe { Arch::deinterleave_pairs4(self.raw, b.raw, c.raw, d.raw) };
        (Self::new(r0), Self::new(r1), Self::new(r2), Self::new(r3))
    }

    /// Splits eight vectors' adjacent-lane pairs into the eight stride-8
    /// subsequences: reading the concatenation as a flat pair sequence,
    /// output `i` holds the pairs congruent to `i` modulo 8.
    ///
    /// On interleaved complex data this is the radix-8 complex decimation of
    /// eight registers in one operation — the movement a mixed-radix
    /// transform performs between passes, and the shuffle a radix-8 Stockham
    /// stage would otherwise route through memory.
    #[inline(always)]
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "eight registers is the operation's arity, not a parameter list"
    )]
    pub fn deinterleave_pairs8(
        self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
    ) -> [Self; 8] {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let raw = unsafe {
            Arch::deinterleave_pairs8(self.raw, b.raw, c.raw, d.raw, e.raw, f.raw, g.raw, h.raw)
        };
        raw.map(Self::new)
    }

    /// Concatenates the low halves of `self` and `other`, and their high
    /// halves: `(self[..n/2] ++ other[..n/2], self[n/2..] ++ other[n/2..])`.
    ///
    /// On interleaved complex data this pairs the first `n/4` samples of each
    /// register, then the last `n/4`: the operand pairing a kernel that packs
    /// two digits of a stride-`n/4` structure into one register needs.
    #[inline(always)]
    #[must_use]
    pub fn interleave_halves(self, other: Self) -> (Self, Self) {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        let (lo, hi) = unsafe { Arch::interleave_halves(self.raw, other.raw) };
        (Self::new(lo), Self::new(hi))
    }

    /// Concatenates the low half of `self` with the high half of `other`:
    /// `self[..n/2] ++ other[n/2..]`.
    ///
    /// The half-granular select. Both halves keep their position, so this is
    /// one in-lane blend where [`Self::interleave_halves`] — which gathers
    /// both operands' *low* halves — costs a cross-lane permute on x86.
    #[inline(always)]
    #[must_use]
    pub fn blend_halves(self, other: Self) -> Self {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        Self::new(unsafe { Arch::blend_halves(self.raw, other.raw) })
    }

    /// The register window `K` lane pairs into `self ++ next`: lanes `2K..`
    /// of `self` followed by lanes `..2K` of `next`.
    ///
    /// On interleaved complex data this is the byte-align at sample
    /// granularity: the window of one register over two consecutive ones,
    /// shifted by `K` samples. One `vperm2f128` (plus one in-lane
    /// `vpalignr` for an odd sample count) on AVX2, one `valignd`/`valignq`
    /// on AVX-512, one `ext` on NEON. `0 < K < LANE_COUNT / 2`, checked per
    /// instantiation.
    #[inline(always)]
    #[must_use]
    pub fn concat_shift_pairs<const K: usize>(self, next: Self) -> Self {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        Self::new(unsafe { Arch::concat_shift_pairs::<K>(self.raw, next.raw) })
    }

    /// [`concat_shift_pairs`](Self::concat_shift_pairs) with the shift a
    /// runtime count: the register window `k` lane pairs into `self ++ next`.
    ///
    /// For kernels generic over the width, whose admissible constants differ
    /// per backend, and for shifts that vary with the data; each backend
    /// selects its constant shuffle by `k`.
    ///
    /// # Panics
    /// Panics in debug builds unless `0 < k < LANE_COUNT / 2`.
    #[inline(always)]
    #[must_use]
    pub fn concat_shift_pairs_at(self, next: Self, k: usize) -> Self {
        // SAFETY: constructing the operand vectors proved host support for `Arch`.
        Self::new(unsafe { Arch::concat_shift_pairs_at(self.raw, next.raw, k) })
    }

    /// Swaps each adjacent lane pair: `[a, b, c, d]` becomes `[b, a, d, c]`.
    ///
    /// On interleaved complex data this exchanges the real and imaginary parts
    /// of every sample, which is the shuffle a complex multiply needs.
    #[inline(always)]
    #[must_use]
    pub fn swap_adjacent(self) -> Self {
        // SAFETY: constructing `self` proved host support for `Arch`.
        Self::new(unsafe { Arch::swap_adjacent(self.raw) })
    }

    /// Swaps each adjacent lane *pair* with its neighbouring pair:
    /// `[a, b, c, d]` becomes `[c, d, a, b]`.
    ///
    /// On interleaved complex data each pair is one sample, so this exchanges
    /// neighbouring complex samples — the operand pairing a distance-one
    /// butterfly needs while held in registers. A trailing pair with no
    /// neighbour passes through unchanged.
    #[inline(always)]
    #[must_use]
    pub fn swap_pairs(self) -> Self {
        // SAFETY: constructing `self` proved host support for `Arch`.
        Self::new(unsafe { Arch::swap_pairs(self.raw) })
    }

    /// Transposes a square tile of `LANE_COUNT` vectors in place: lane `c`
    /// of row `r` moves to lane `r` of row `c`.
    ///
    /// This is the in-register granularity for blocked matrix transposes:
    /// load a `LANE_COUNT x LANE_COUNT` block as rows, transpose here, store
    /// the rows to the exchanged block.
    ///
    /// # Panics
    /// Panics if `tile` does not hold exactly `LANE_COUNT` vectors.
    #[inline(always)]
    pub fn transpose_square(tile: &mut [Self]) {
        assert_eq!(
            tile.len(),
            Arch::LANE_COUNT,
            "tile must hold LANE_COUNT rows"
        );
        // SAFETY: `Vector` is `#[repr(transparent)]` over `Arch::Vector`, so
        // the slice cast preserves layout, and constructing the vectors
        // proved host support for `Arch`.
        unsafe {
            let raw = core::slice::from_raw_parts_mut(
                tile.as_mut_ptr().cast::<Arch::Vector>(),
                tile.len(),
            );
            Arch::transpose_square(raw);
        }
    }

    /// Duplicates each even-indexed lane over its odd neighbour:
    /// `[a, b, c, d]` becomes `[a, a, c, c]`.
    ///
    /// On interleaved complex data this broadcasts the real part of each sample.
    #[inline(always)]
    #[must_use]
    pub fn dup_even(self) -> Self {
        // SAFETY: constructing `self` proved host support for `Arch`.
        Self::new(unsafe { Arch::dup_even(self.raw) })
    }

    /// Duplicates each odd-indexed lane over its even neighbour:
    /// `[a, b, c, d]` becomes `[b, b, d, d]`.
    ///
    /// On interleaved complex data this broadcasts the imaginary part.
    #[inline(always)]
    #[must_use]
    pub fn dup_odd(self) -> Self {
        // SAFETY: constructing `self` proved host support for `Arch`.
        Self::new(unsafe { Arch::dup_odd(self.raw) })
    }
}

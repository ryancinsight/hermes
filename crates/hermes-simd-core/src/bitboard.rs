//! Chess bitboards and sliding attack generation views.

use core::marker::PhantomData;

/// Trait defining the sliding attack generation interface.
///
/// Implemented by backend markers in the intrinsics crate.
///
/// These methods are safe. Unlike [`SimdKernel`](crate::kernel::SimdKernel),
/// whose operations are `#[target_feature]`-gated and therefore carry an ISA
/// precondition, an attack generator either computes with plain integer
/// arithmetic or selects its ISA-specific path by runtime detection behind its
/// own safe surface. Table lookups are bounds-checked, so an out-of-range
/// square panics rather than reading out of bounds.
pub trait BitBoardKernel: Send + Sync + 'static {
    /// Generate Rook attacks for a square given occupancy.
    ///
    /// # Panics
    /// If `square` is not a board index below 64.
    fn rook_attacks(square: u8, occupancy: u64) -> u64;

    /// Generate Bishop attacks for a square given occupancy.
    ///
    /// # Panics
    /// If `square` is not a board index below 64.
    fn bishop_attacks(square: u8, occupancy: u64) -> u64;

    /// Generate Queen attacks for a square given occupancy.
    ///
    /// # Panics
    /// If `square` is not a board index below 64.
    #[inline(always)]
    #[must_use]
    fn queen_attacks(square: u8, occupancy: u64) -> u64 {
        Self::rook_attacks(square, occupancy) | Self::bishop_attacks(square, occupancy)
    }
}

/// A zero-copy newtype family for chess bitboards.
///
/// Parameterized by `Backend` (e.g. `KoggeStone`, `Magic`), `Arch` (SIMD architecture),
/// and reference typestate `Ref` (e.g. `&'a [u64]` or `&'a mut [u64]`).
#[repr(transparent)]
pub struct BitBoardView<'a, Backend, Arch, Ref: 'a = &'a [u64]> {
    ptr: *mut [u64],
    _marker: PhantomData<(&'a u64, Backend, Arch, Ref)>,
}

unsafe impl<'a, Backend, Arch, Ref: 'a> Send for BitBoardView<'a, Backend, Arch, Ref> where Ref: Send
{}
unsafe impl<'a, Backend, Arch, Ref: 'a> Sync for BitBoardView<'a, Backend, Arch, Ref> where Ref: Sync
{}

impl<'a, Backend, Arch> Clone for BitBoardView<'a, Backend, Arch, &'a [u64]> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, Backend, Arch> Copy for BitBoardView<'a, Backend, Arch, &'a [u64]> {}

impl<'a, Backend, Arch> BitBoardView<'a, Backend, Arch, &'a [u64]> {
    /// Create a shared `BitBoardView` over a slice of bitboards.
    #[inline(always)]
    #[must_use]
    pub fn new(data: &'a [u64]) -> Self {
        Self {
            ptr: data as *const [u64] as *mut [u64],
            _marker: PhantomData,
        }
    }
}

impl<'a, Backend, Arch> BitBoardView<'a, Backend, Arch, &'a mut [u64]> {
    /// Create a mutable `BitBoardView` over a slice of bitboards.
    #[inline(always)]
    pub fn new_mut(data: &'a mut [u64]) -> Self {
        Self {
            ptr: data as *mut [u64],
            _marker: PhantomData,
        }
    }

    /// Access the underlying raw mutable slice.
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [u64] {
        // SAFETY: this impl is reachable only for `Ref = &'a mut [u64]`, so the
        // pointer came from `new_mut`'s exclusive borrow and no shared view of
        // the same data can exist. `&mut self` bounds the reborrow to a span in
        // which the view itself is exclusively held, and the returned lifetime
        // is that of `self`, which cannot outlive `'a`.
        unsafe { &mut *self.ptr }
    }

    /// Downgrade exclusive mutable view to a shared view.
    #[inline(always)]
    #[must_use]
    pub fn downgrade(self) -> BitBoardView<'a, Backend, Arch, &'a [u64]> {
        BitBoardView {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<'a, Backend, Arch, Ref: 'a> BitBoardView<'a, Backend, Arch, Ref> {
    /// Access the underlying raw slice of bitboards.
    #[inline(always)]
    #[must_use]
    pub fn as_slice(&self) -> &[u64] {
        // SAFETY: the pointer was derived from a borrow of `'a` — shared in
        // `new`, exclusive in `new_mut` — and the view keeps that borrow alive
        // through its `PhantomData`. Taking `&self` yields a shared reborrow
        // bounded by `self`, which cannot outlive `'a`; when `Ref` is the
        // exclusive typestate, holding `&self` precludes a concurrent
        // `as_slice_mut`, so no aliasing `&mut` exists.
        unsafe { &*self.ptr }
    }
}

impl<'a, Backend, Arch, Ref: 'a> core::ops::Deref for BitBoardView<'a, Backend, Arch, Ref> {
    type Target = [u64];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, Backend, Arch> core::ops::DerefMut for BitBoardView<'a, Backend, Arch, &'a mut [u64]> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
    }
}

impl<'a, Backend, Arch, Ref: 'a> BitBoardView<'a, Backend, Arch, Ref>
where
    Backend: BitBoardKernel,
{
    /// Generate Rook attacks for a square given occupancy.
    #[inline(always)]
    #[must_use]
    pub fn rook_attacks(&self, square: u8, occupancy: u64) -> u64 {
        Backend::rook_attacks(square, occupancy)
    }

    /// Generate Bishop attacks for a square given occupancy.
    #[inline(always)]
    #[must_use]
    pub fn bishop_attacks(&self, square: u8, occupancy: u64) -> u64 {
        Backend::bishop_attacks(square, occupancy)
    }

    /// Generate Queen attacks for a square given occupancy.
    #[inline(always)]
    #[must_use]
    pub fn queen_attacks(&self, square: u8, occupancy: u64) -> u64 {
        Backend::queen_attacks(square, occupancy)
    }

    /// Generate attacks for a batch of squares under a single occupancy bitboard.
    ///
    /// Amortizes loop overhead and permits compiler instruction scheduling / pipelining
    /// by unrolling the attack queries in blocks of 4.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `squares`.
    #[inline]
    pub fn batch_attacks_single_occupancy(
        &self,
        squares: &[u8],
        occupancy: u64,
        out: &mut [u64],
        is_rook: bool,
    ) {
        assert!(
            out.len() >= squares.len(),
            "Output slice too short for batch attacks"
        );

        let len = squares.len();
        let mut i = 0;

        let unroll = (len / 4) * 4;
        while i < unroll {
            let sq0 = squares[i];
            let sq1 = squares[i + 1];
            let sq2 = squares[i + 2];
            let sq3 = squares[i + 3];

            if is_rook {
                out[i] = Backend::rook_attacks(sq0, occupancy);
                out[i + 1] = Backend::rook_attacks(sq1, occupancy);
                out[i + 2] = Backend::rook_attacks(sq2, occupancy);
                out[i + 3] = Backend::rook_attacks(sq3, occupancy);
            } else {
                out[i] = Backend::bishop_attacks(sq0, occupancy);
                out[i + 1] = Backend::bishop_attacks(sq1, occupancy);
                out[i + 2] = Backend::bishop_attacks(sq2, occupancy);
                out[i + 3] = Backend::bishop_attacks(sq3, occupancy);
            }
            i += 4;
        }

        while i < len {
            let sq = squares[i];
            out[i] = if is_rook {
                Backend::rook_attacks(sq, occupancy)
            } else {
                Backend::bishop_attacks(sq, occupancy)
            };
            i += 1;
        }
    }
}

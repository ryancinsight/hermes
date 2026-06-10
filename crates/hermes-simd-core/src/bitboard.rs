//! Chess bitboards and sliding attack generation views.

use core::marker::PhantomData;

/// Trait defining the sliding attack generation interface.
///
/// Implemented by backend markers in the intrinsics crate.
pub trait BitBoardKernel: Send + Sync + 'static {
    /// Generate Rook attacks for a square given occupancy.
    ///
    /// # Safety
    /// Caller must ensure target feature flags are active.
    unsafe fn rook_attacks(square: u8, occupancy: u64) -> u64;

    /// Generate Bishop attacks for a square given occupancy.
    ///
    /// # Safety
    /// Caller must ensure target feature flags are active.
    unsafe fn bishop_attacks(square: u8, occupancy: u64) -> u64;

    /// Generate Queen attacks for a square given occupancy.
    ///
    /// # Safety
    /// Caller must ensure target feature flags are active.
    #[inline(always)]
    unsafe fn queen_attacks(square: u8, occupancy: u64) -> u64 {
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
        unsafe { &mut *self.ptr }
    }

    /// Downgrade exclusive mutable view to a shared view.
    #[inline(always)]
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
    pub fn as_slice(&self) -> &[u64] {
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
    pub fn rook_attacks(&self, square: u8, occupancy: u64) -> u64 {
        unsafe { Backend::rook_attacks(square, occupancy) }
    }

    /// Generate Bishop attacks for a square given occupancy.
    #[inline(always)]
    pub fn bishop_attacks(&self, square: u8, occupancy: u64) -> u64 {
        unsafe { Backend::bishop_attacks(square, occupancy) }
    }

    /// Generate Queen attacks for a square given occupancy.
    #[inline(always)]
    pub fn queen_attacks(&self, square: u8, occupancy: u64) -> u64 {
        unsafe { Backend::queen_attacks(square, occupancy) }
    }

    /// Generate attacks for a batch of squares under a single occupancy bitboard.
    ///
    /// Amortizes loop overhead and permits compiler instruction scheduling / pipelining
    /// by unrolling the attack queries in blocks of 4.
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

            unsafe {
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
            }
            i += 4;
        }

        while i < len {
            let sq = squares[i];
            unsafe {
                out[i] = if is_rook {
                    Backend::rook_attacks(sq, occupancy)
                } else {
                    Backend::bishop_attacks(sq, occupancy)
                };
            }
            i += 1;
        }
    }
}

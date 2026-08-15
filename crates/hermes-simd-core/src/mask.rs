//! Bit-packed lane mask for SIMD predicated operations.
//!
//! `BitMask<N>` stores a predicate for up to 64 SIMD lanes in the bits of a `u64`.
//! Bit `i` of the inner value corresponds to lane `i`; bits `[N..]` are always zero.
//!
//! # Design rationale
//!
//! The prior `mask_from_bools(&[bool])` approach passed a byte-per-element slice on every
//! hot vector call — 8× more memory than necessary, and requires a loop to convert.
//! `BitMask<N>` eliminates this by:
//!
//! - AVX-512: direct `transmute` from `__mmask16` / `__mmask8` (both `u16`/`u8` ⊆ `u64`).
//! - Scalar: `from_bools` is a single bitwise-OR loop; `leading_k` is a const expression.
//! - AVX2 / NEON: `mask_from_bitmask` on `SimdKernel` expands `BitMask<N>` to the native
//!   float blend mask (`__m256` / `uint32x4_t`) once at the entry point.
//!
//! All `BitMask` methods are either `const` or trivially inlineable.

/// Bit-packed predicate mask for exactly `N` SIMD lanes.
///
/// The inner `u64` stores one bit per lane: bit `i` = lane `i` is active.
/// Invariant: `(self.0 >> N) == 0` (high bits always clear).
///
/// `N` must satisfy `N <= 64`. Violations are caught by a const assertion in `ALL_ACTIVE`.
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Clone, Copy, Debug, PartialEq, Eq, Hash))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BitMask<const N: usize>(pub u64);

impl<const N: usize> BitMask<N> {
    /// All `N` lanes active.
    ///
    /// # Panics (compile-time)
    /// Panics at compile time if `N > 64`.
    pub const ALL_ACTIVE: Self = {
        // Const assertion: N must fit in u64.
        assert!(N <= 64, "BitMask<N>: N must be <= 64");
        // (1u64 << 64) would overflow; handle that case with u64::MAX.
        let bits = if N >= 64 {
            u64::MAX
        } else {
            (1u64 << N).wrapping_sub(1)
        };
        Self(bits)
    };

    /// No lanes active.
    pub const NONE_ACTIVE: Self = Self(0);

    /// First `k` lanes active, rest inactive. Clamps `k` to `N`.
    ///
    /// This is a `const fn` so it can be used in const contexts.
    ///
    /// # Examples
    /// ```
    /// use hermes_simd_core::mask::BitMask;
    /// assert_eq!(BitMask::<8>::leading_k(5).0, 0b00011111);
    /// assert_eq!(BitMask::<8>::leading_k(0).0, 0);
    /// assert_eq!(BitMask::<8>::leading_k(8).0, 0xFF);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn leading_k(k: usize) -> Self {
        let k = if k > N { N } else { k };
        let bits = if k >= 64 {
            u64::MAX
        } else {
            (1u64 << k).wrapping_sub(1)
        };
        Self(bits)
    }

    /// Build a mask from a `bool` slice of length `N`.
    ///
    /// Bit `i` is set if `bits[i]` is `true`.
    ///
    /// # Panics
    /// Panics in debug mode if `bits.len() != N`.
    #[inline(always)]
    #[must_use]
    pub fn from_bools(bits: &[bool]) -> Self {
        debug_assert_eq!(
            bits.len(),
            N,
            "BitMask::from_bools: slice length must equal N"
        );
        let mut m = 0u64;
        // Single pass, no branching beyond the loop iterator.
        for (i, &b) in bits.iter().enumerate().take(N) {
            m |= u64::from(b) << i;
        }
        Self(m)
    }

    /// Number of active (set) lanes.
    #[inline(always)]
    #[must_use]
    pub fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns `true` if all `N` lanes are active.
    #[inline(always)]
    #[must_use]
    pub fn is_all_active(self) -> bool {
        self.0 == Self::ALL_ACTIVE.0
    }

    /// Returns `true` if no lanes are active.
    #[inline(always)]
    #[must_use]
    pub fn is_none_active(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if lane `i` is active.
    ///
    /// # Panics
    /// Panics in debug mode if `i >= N`.
    #[inline(always)]
    #[must_use]
    pub fn is_lane_active(self, i: usize) -> bool {
        debug_assert!(i < N, "BitMask::is_lane_active: lane index out of range");
        (self.0 >> i) & 1 == 1
    }

    /// Bitwise AND of two masks.
    #[inline(always)]
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    /// Bitwise OR of two masks.
    #[inline(always)]
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Expand this mask to a `[bool; N]` array.
    ///
    /// Useful for scalar fallbacks and debugging. Not performance-critical.
    #[must_use]
    pub fn to_bools(self) -> [bool; N]
    where
        [bool; N]: Sized,
    {
        core::array::from_fn(|i| (self.0 >> i) & 1 == 1)
    }
}

impl<const N: usize> BitMask<N> {
    /// Convert this `BitMask<N>` to the native hardware mask type for `Arch`.
    ///
    /// Delegates to [`SimdMask::mask_from_bitmask`](crate::kernel::SimdMask::mask_from_bitmask)
    /// using the inner `u64` value.
    /// Zero runtime cost: the compiler inlines this into a single instruction on
    /// AVX-512 (`KMOV`), a vector comparison + blend mask on AVX2, or a bool-array
    /// copy on scalar backends.
    ///
    /// # Safety
    /// Processor must support the target feature of `Arch`.
    ///
    /// # Example
    /// ```rust
    /// use hermes_simd_core::mask::BitMask;
    /// use hermes_simd_core::kernel::SimdStorage;
    /// use hermes_simd_intrinsics::Scalar;
    ///
    /// let bm = BitMask::<4>::leading_k(3);
    /// // SAFETY: `Scalar` has no target-feature precondition.
    /// let native: <Scalar as SimdStorage<f32>>::Mask =
    ///     unsafe { bm.to_native_mask::<f32, Scalar>() };
    ///
    /// assert_eq!(native, [true, true, true, false]);
    /// ```
    #[inline(always)]
    #[must_use]
    pub unsafe fn to_native_mask<T, Arch>(self) -> Arch::Mask
    where
        T: crate::scalar::Scalar,
        Arch: crate::kernel::SimdKernel<T>,
    {
        Arch::mask_from_bitmask(self.0)
    }
}

// ---------------------------------------------------------------------------
// BitMaskIter — active lane index iterator using bit manipulation
// ---------------------------------------------------------------------------

/// Iterator over active lane indices of a [`BitMask<N>`].
///
/// Yields the index (0..N) of each set bit in ascending order.
///
/// # Algorithm
///
/// Uses `u64::trailing_zeros` to jump directly to the next set bit in O(1) per step,
/// then clears that bit with `remaining &= remaining - 1` (Kernighan's bit trick).
/// Total cost is O(popcount), not O(N) — critical for sparse masks.
///
/// # Size
///
/// `size_of::<BitMaskIter<N>>()` is 8 bytes (one `u64`). The `lane` field is removed;
/// position is recovered from `trailing_zeros` each time.
#[derive(Clone, Copy, Debug)]
pub struct BitMaskIter<const N: usize> {
    /// Remaining active bits. Bits are cleared as they are consumed.
    remaining: u64,
}

#[expect(
    clippy::copy_iterator,
    reason = "The iterator is an eight-byte value returned by the copyable BitMask API"
)]
impl<const N: usize> Iterator for BitMaskIter<N> {
    type Item = usize;

    /// Returns the index of the next active lane, or `None` if no lanes remain.
    ///
    /// Clears the lowest set bit after returning its index.
    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        // trailing_zeros gives the position of the lowest set bit.
        let idx = self.remaining.trailing_zeros() as usize;
        // Guard: respect the N-lane bound (high bits of a partial mask could be set
        // only if BitMask invariant is violated, but we check defensively).
        if idx >= N {
            return None;
        }
        // Kernighan's trick: clear the lowest set bit in one instruction.
        self.remaining &= self.remaining.wrapping_sub(1);
        Some(idx)
    }

    /// Returns exact bounds: `(popcount, Some(popcount))`.
    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.remaining.count_ones() as usize;
        (n, Some(n))
    }
}

impl<const N: usize> ExactSizeIterator for BitMaskIter<N> {}

impl<const N: usize> DoubleEndedIterator for BitMaskIter<N> {
    /// Returns the index of the highest active lane remaining.
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        // leading_zeros + bit position = highest set bit.
        let idx = 63 - self.remaining.leading_zeros() as usize;
        if idx >= N {
            return None;
        }
        // Clear the highest set bit.
        self.remaining &= !(1u64 << idx);
        Some(idx)
    }
}

impl<const N: usize> IntoIterator for BitMask<N> {
    type Item = usize;
    type IntoIter = BitMaskIter<N>;

    /// Iterate over active lane indices in ascending order.
    ///
    /// # Example
    /// ```rust
    /// use hermes_simd_core::mask::BitMask;
    ///
    /// let mask = BitMask::<8>::from_bools(&[true, false, true, false, true, false, false, false]);
    /// let indices: Vec<usize> = mask.into_iter().collect();
    ///
    /// assert_eq!(indices, vec![0, 2, 4]);
    /// ```
    #[inline(always)]
    fn into_iter(self) -> BitMaskIter<N> {
        BitMaskIter { remaining: self.0 }
    }
}

impl<const N: usize> BitMask<N> {
    /// Convenience method to iterate active lane indices without consuming.
    ///
    /// Equivalent to `(*self).into_iter()` since `BitMask<N>: Copy`.
    #[inline(always)]
    #[must_use]
    pub fn active_lanes(self) -> BitMaskIter<N> {
        self.into_iter()
    }
}

impl<const N: usize> Default for BitMask<N> {
    #[inline(always)]
    fn default() -> Self {
        Self::NONE_ACTIVE
    }
}

impl<const N: usize> core::ops::BitAnd for BitMask<N> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        self.and(rhs)
    }
}

impl<const N: usize> core::ops::BitOr for BitMask<N> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        self.or(rhs)
    }
}

impl<const N: usize> core::ops::Not for BitMask<N> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0 & Self::ALL_ACTIVE.0)
    }
}

#[cfg(test)]
mod rkyv_tests {
    use super::*;

    #[test]
    // rkyv archived access violates Stacked Borrows inside the dependency;
    // see vec/tests.rs for the rationale and the 0.8.17 re-probe.
    #[cfg_attr(miri, ignore)]
    fn test_bitmask_rkyv() {
        let mask = BitMask::<8>::from_bools(&[true, false, true, true, false, false, true, false]);
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&mask).unwrap();

        let archived = rkyv::access::<rkyv::Archived<BitMask<8>>, rkyv::rancor::Error>(&bytes)
            .expect("validated access");
        let deserialized: BitMask<8> =
            rkyv::deserialize::<_, rkyv::rancor::Error>(archived).unwrap();
        assert_eq!(deserialized, mask);
        assert_eq!(deserialized.0, mask.0);
    }
}

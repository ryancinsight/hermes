//! Bit-packed predicate masks for SIMD operations.
//!
//! `BitMask<N>` stores a predicate for up to 64 SIMD lanes in the bits of a `u64`.
//! Bit `i` of the inner value corresponds to lane `i`; bits `[N..]` are always zero.
//! [`PackedMask`] is its runtime-length counterpart: one bit per element of an
//! arbitrarily long buffer, packed into words, with `<= 64`-lane windows
//! extractable as raw bitmasks for the backends.
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

// ---------------------------------------------------------------------------
// PackedMask — arbitrary-length bit-packed element mask
// ---------------------------------------------------------------------------

/// Number of mask bits stored per word.
const WORD_BITS: usize = 64;

/// Bit-packed predicate mask over an arbitrary number of elements.
///
/// The runtime-length counterpart of [`BitMask`]: where `BitMask<N>` packs one
/// SIMD register's lane predicate (`N <= 64`) into a single word, `PackedMask`
/// packs one bit per element of an arbitrarily long buffer into a word slice —
/// 8× smaller than the byte-per-element `[bool]` representation it replaces,
/// and directly consumable by the SIMD backends: [`PackedMask::lane_bits`]
/// extracts any `<= 64`-lane window as the raw bitmask
/// [`SimdMask::mask_from_bitmask`](crate::kernel::SimdMask::mask_from_bitmask)
/// expands to a native mask, with no per-lane conversion loop.
///
/// Bit `i % 64` of word `i / 64` corresponds to element `i`.
///
/// # Invariants
/// - `words.len() == len.div_ceil(64)` (exact word count).
/// - Bits at positions `>= len` in the final word are zero, so
///   [`PackedMask::popcount`] needs no tail masking.
///
/// Both are established by the validating constructors and preserved by field
/// privacy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PackedMask<W> {
    words: W,
    len: usize,
}

impl PackedMask<Box<[u64]>> {
    /// Bit-pack a `bool` slice: the construction boundary at which the 8×
    /// footprint reduction happens, once.
    ///
    /// # Examples
    /// ```
    /// use hermes_simd_core::mask::PackedMask;
    ///
    /// let mask = PackedMask::from_bools(&[true, false, true]);
    /// assert_eq!(mask.len(), 3);
    /// assert_eq!(mask.popcount(), 2);
    /// assert!(mask.bit(0) && !mask.bit(1) && mask.bit(2));
    /// ```
    #[must_use]
    pub fn from_bools(bits: &[bool]) -> Self {
        let mut words = vec![0u64; bits.len().div_ceil(WORD_BITS)].into_boxed_slice();
        for (i, &b) in bits.iter().enumerate() {
            words[i / WORD_BITS] |= u64::from(b) << (i % WORD_BITS);
        }
        Self {
            words,
            len: bits.len(),
        }
    }
}

impl<W: AsRef<[u64]>> PackedMask<W> {
    /// Wrap pre-packed words as a mask over `len` elements, validating the
    /// representation invariants.
    ///
    /// # Errors
    /// [`SimdError::LengthMismatch`](crate::SimdError::LengthMismatch) when the
    /// word count is not exactly `len.div_ceil(64)`;
    /// [`SimdError::IndexOutOfBounds`](crate::SimdError::IndexOutOfBounds) when
    /// a bit at position `>= len` is set in the final word.
    pub fn new(words: W, len: usize) -> Result<Self, crate::SimdError> {
        let slice = words.as_ref();
        if slice.len() != len.div_ceil(WORD_BITS) {
            return Err(crate::SimdError::LengthMismatch);
        }
        let tail = len % WORD_BITS;
        if tail != 0 && slice[slice.len() - 1] >> tail != 0 {
            return Err(crate::SimdError::IndexOutOfBounds);
        }
        Ok(Self { words, len })
    }

    /// Number of elements the mask covers.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the mask covers no elements.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if element `i` is active.
    ///
    /// # Panics
    /// Panics if `i >= self.len()`.
    #[track_caller]
    #[inline(always)]
    #[must_use]
    pub fn bit(&self, i: usize) -> bool {
        assert!(
            i < self.len,
            "PackedMask::bit: index {i} outside mask length {}",
            self.len
        );
        self.bit_inner(i)
    }

    #[inline(always)]
    pub(crate) fn bit_in_bounds(&self, i: usize) -> bool {
        debug_assert!(i < self.len, "PackedMask::bit_in_bounds: invalid proof");
        self.bit_inner(i)
    }

    #[inline(always)]
    fn bit_inner(&self, i: usize) -> bool {
        (self.words.as_ref()[i / WORD_BITS] >> (i % WORD_BITS)) & 1 == 1
    }

    /// Extract `count <= 64` mask bits starting at element `offset` as a raw
    /// bitmask (bit `k` = element `offset + k`), combining at most two words.
    ///
    /// This is the per-chunk kernel entry point: feed the result to
    /// [`SimdMask::mask_from_bitmask`](crate::kernel::SimdMask::mask_from_bitmask)
    /// to obtain the native lane mask.
    ///
    /// # Panics
    /// Panics if `count > 64`, `offset > self.len()`, or the requested window
    /// extends beyond `self.len()`.
    ///
    /// # Examples
    /// ```
    /// use hermes_simd_core::mask::PackedMask;
    ///
    /// // Element 62 is active; a 4-lane window at offset 61 crosses the
    /// // word boundary and sees it in lane 1.
    /// let mut bits = vec![false; 70];
    /// bits[62] = true;
    /// let mask = PackedMask::from_bools(&bits);
    /// assert_eq!(mask.lane_bits(61, 4), 0b0010);
    /// ```
    #[track_caller]
    #[inline(always)]
    #[must_use]
    pub fn lane_bits(&self, offset: usize, count: usize) -> u64 {
        assert!(
            count <= WORD_BITS,
            "PackedMask::lane_bits: count {count} exceeds 64"
        );
        assert!(
            offset <= self.len,
            "PackedMask::lane_bits: offset {offset} outside mask length {}",
            self.len
        );
        assert!(
            count <= self.len - offset,
            "PackedMask::lane_bits: offset {offset} with count {count} exceeds mask length {}",
            self.len
        );
        self.lane_bits_inner(offset, count)
    }

    #[inline(always)]
    pub(crate) fn lane_bits_in_bounds(&self, offset: usize, count: usize) -> u64 {
        debug_assert!(
            count <= WORD_BITS && offset <= self.len && count <= self.len - offset,
            "PackedMask::lane_bits_in_bounds: invalid proof"
        );
        self.lane_bits_inner(offset, count)
    }

    #[inline(always)]
    fn lane_bits_inner(&self, offset: usize, count: usize) -> u64 {
        if count == 0 {
            return 0;
        }
        let words = self.words.as_ref();
        let word = offset / WORD_BITS;
        let bit = offset % WORD_BITS;
        let mut bits = words[word] >> bit;
        if bit != 0 && bit + count > WORD_BITS {
            bits |= words[word + 1] << (WORD_BITS - bit);
        }
        if count == WORD_BITS {
            bits
        } else {
            bits & (1u64 << count).wrapping_sub(1)
        }
    }

    /// Number of active elements.
    ///
    /// Exact without tail masking because bits `>= len` are zero by invariant.
    #[inline]
    #[must_use]
    pub fn popcount(&self) -> usize {
        self.words
            .as_ref()
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Borrow this mask as a word-slice-backed view (zero-copy).
    #[inline(always)]
    #[must_use]
    pub fn as_view(&self) -> PackedMask<&[u64]> {
        PackedMask {
            words: self.words.as_ref(),
            len: self.len,
        }
    }
}

impl<W: AsRef<[u64]>> From<&PackedMask<W>> for PackedMask<Box<[u64]>> {
    /// Clone the packed words into owned storage.
    #[inline]
    fn from(mask: &PackedMask<W>) -> Self {
        Self {
            words: mask.words.as_ref().into(),
            len: mask.len,
        }
    }
}

#[cfg(test)]
mod packed_mask_tests {
    use super::*;
    use crate::SimdError;

    #[test]
    fn from_bools_round_trips_every_bit() {
        // 70 elements spans two words with a 6-bit tail.
        let bits: Vec<bool> = (0..70).map(|i| i % 3 == 0).collect();
        let mask = PackedMask::from_bools(&bits);
        assert_eq!(mask.len(), 70);
        for (i, &b) in bits.iter().enumerate() {
            assert_eq!(mask.bit(i), b, "bit {i}");
        }
        assert_eq!(mask.popcount(), bits.iter().filter(|&&b| b).count());
    }

    #[test]
    fn empty_mask() {
        let mask = PackedMask::from_bools(&[]);
        assert_eq!(mask.len(), 0);
        assert!(mask.is_empty());
        assert_eq!(mask.popcount(), 0);
        assert_eq!(mask.lane_bits(0, 0), 0);
    }

    #[test]
    fn empty_window_at_logical_end_is_valid() {
        let mask = PackedMask::from_bools(&[true, false, true]);
        assert_eq!(mask.lane_bits(mask.len(), 0), 0);
    }

    #[test]
    #[should_panic(expected = "outside mask length")]
    fn bit_rejects_logical_end() {
        let _ = PackedMask::from_bools(&[true, false, true]).bit(3);
    }

    #[test]
    #[should_panic(expected = "count 65 exceeds 64")]
    fn lane_bits_rejects_more_than_one_word() {
        let _ = PackedMask::from_bools(&[true; 65]).lane_bits(0, 65);
    }

    #[test]
    #[should_panic(expected = "outside mask length")]
    fn lane_bits_rejects_overflow_sized_offset() {
        let _ = PackedMask::from_bools(&[true]).lane_bits(usize::MAX, 1);
    }

    #[test]
    #[should_panic(expected = "offset 2 with count 2 exceeds mask length 3")]
    fn lane_bits_rejects_window_past_logical_end() {
        let _ = PackedMask::from_bools(&[true, false, true]).lane_bits(2, 2);
    }

    #[test]
    #[should_panic(expected = "with count 2 exceeds mask length")]
    fn lane_bits_rejects_near_maximum_window_without_overflow() {
        // The guard rejects the window before storage is observed; constructing
        // this private boundary fixture avoids an impossible usize::MAX-bit allocation.
        let mask = PackedMask {
            words: &[] as &[u64],
            len: usize::MAX,
        };
        let _ = mask.lane_bits(usize::MAX - 1, 2);
    }

    #[test]
    fn all_set_and_all_clear() {
        let set = PackedMask::from_bools(&[true; 130]);
        assert_eq!(set.popcount(), 130);
        assert_eq!(set.lane_bits(120, 10), 0x3FF);
        let clear = PackedMask::from_bools(&[false; 130]);
        assert_eq!(clear.popcount(), 0);
        assert_eq!(clear.lane_bits(0, 64), 0);
    }

    #[test]
    fn lane_bits_crosses_word_boundary() {
        let mut bits = vec![false; 128];
        for i in [60, 63, 64, 67] {
            bits[i] = true;
        }
        let mask = PackedMask::from_bools(&bits);
        // 8-lane window at 60: elements 60..68 -> lanes 0, 3, 4, 7 active.
        assert_eq!(mask.lane_bits(60, 8), 0b1001_1001);
        // Full-word window aligned at 64: elements 64 and 67 -> bits 0 and 3.
        assert_eq!(mask.lane_bits(64, 64), 0b1001);
    }

    #[test]
    fn new_validates_word_count_and_tail() {
        // Exact word count with a clear tail is accepted.
        let mask = PackedMask::new([0b101u64].as_slice(), 3).expect("valid");
        assert_eq!(mask.popcount(), 2);
        // Wrong word count.
        assert_eq!(
            PackedMask::new([0u64; 2].as_slice(), 3).unwrap_err(),
            SimdError::LengthMismatch
        );
        // Set bit beyond `len` in the final word.
        assert_eq!(
            PackedMask::new([0b1000u64].as_slice(), 3).unwrap_err(),
            SimdError::IndexOutOfBounds
        );
    }

    #[test]
    fn view_and_owned_conversion_preserve_value() {
        let mask = PackedMask::from_bools(&[true, false, true, true, false]);
        let view = mask.as_view();
        assert_eq!(view.len(), mask.len());
        assert_eq!(view.popcount(), mask.popcount());
        let owned = PackedMask::from(&view);
        assert_eq!(owned, mask);
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

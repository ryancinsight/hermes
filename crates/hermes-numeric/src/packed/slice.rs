use crate::types::{Bf16, Bf4, F32, F4};

/// Trait for 4-bit types that can be packed two per byte.
pub trait Packable4: Copy + 'static {
    /// The unpacked representation type.
    type Unpacked: Copy + Default;

    /// Pack a low and high element into a single byte.
    fn pack_pair(low: Self, high: Self) -> u8;
    /// Unpack a single byte into a low and high element.
    fn unpack_pair(packed: u8) -> (Self, Self);
    /// Unpack a slice of packed pairs to a slice of unpacked elements.
    fn unpack_slice_packed(packed: &[u8], unpacked: &mut [Self::Unpacked]);
    /// Unpack a single element.
    fn unpack_single(element: Self) -> Self::Unpacked;
}

impl Packable4 for Bf4 {
    type Unpacked = Bf16;

    #[inline(always)]
    fn pack_pair(low: Self, high: Self) -> u8 {
        Bf4::pack_pair(low, high)
    }
    #[inline(always)]
    fn unpack_pair(packed: u8) -> (Self, Self) {
        Bf4::unpack_pair(packed)
    }
    #[inline(always)]
    fn unpack_slice_packed(packed: &[u8], unpacked: &mut [Bf16]) {
        super::unpack::unpack_bf4_to_bf16_packed(packed, unpacked);
    }
    #[inline(always)]
    fn unpack_single(element: Self) -> Bf16 {
        Bf16(half::bf16::from_bits(super::unpack::bf4_to_bf16_bits(
            element.0,
        )))
    }
}

impl Packable4 for F4 {
    type Unpacked = F32;

    #[inline(always)]
    fn pack_pair(low: Self, high: Self) -> u8 {
        F4::pack_pair(low, high)
    }
    #[inline(always)]
    fn unpack_pair(packed: u8) -> (Self, Self) {
        F4::unpack_pair(packed)
    }
    #[inline(always)]
    fn unpack_slice_packed(packed: &[u8], unpacked: &mut [F32]) {
        super::unpack::unpack_f4_to_f32_packed(packed, unpacked);
    }
    #[inline(always)]
    fn unpack_single(element: Self) -> F32 {
        F32(element.to_f32())
    }
}

/// A read-only view over a packed slice of 4-bit values, stored 2 per byte.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Packed4Slice<'a, T: Packable4> {
    pub(crate) data: &'a [u8],
    pub(crate) len: usize,
    pub(crate) _marker: core::marker::PhantomData<T>,
}

impl<'a, T: Packable4> Packed4Slice<'a, T> {
    /// Create a new `Packed4Slice` from packed bytes and logical length.
    /// Returns `None` if the backing buffer is too small for the requested length.
    #[inline]
    pub fn new(data: &'a [u8], len: usize) -> Option<Self> {
        let required_bytes = len.div_ceil(2);
        if data.len() < required_bytes {
            None
        } else {
            Some(Self {
                data,
                len,
                _marker: core::marker::PhantomData,
            })
        }
    }

    /// Returns the logical length (number of elements).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the underlying packed byte slice.
    #[inline(always)]
    pub fn as_packed_slice(&self) -> &'a [u8] {
        self.data
    }

    /// Returns true if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get an element at the given logical index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            None
        } else {
            let byte_idx = index / 2;
            let byte = self.data[byte_idx];
            let (low, high) = T::unpack_pair(byte);
            if index.is_multiple_of(2) {
                Some(low)
            } else {
                Some(high)
            }
        }
    }

    /// Attempts to create a zero-copy sub-slice of the packed slice.
    ///
    /// This is only possible without allocation if the range starts at an even logical index
    /// (aligned to byte boundaries). If the range starts at an odd index, returns `None`.
    #[inline]
    pub fn sub_slice(self, range: core::ops::Range<usize>) -> Option<Self> {
        if range.start > range.end || range.end > self.len {
            return None;
        }
        if range.start.is_multiple_of(2) {
            let byte_start = range.start / 2;
            let byte_end = range.end.div_ceil(2);
            let sub_data = &self.data[byte_start..byte_end];
            let sub_len = range.end - range.start;
            Some(Self {
                data: sub_data,
                len: sub_len,
                _marker: core::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

/// A mutable view over a packed slice of 4-bit values, stored 2 per byte.
pub struct Packed4SliceMut<'a, T: Packable4> {
    pub(crate) data: &'a mut [u8],
    pub(crate) len: usize,
    pub(crate) _marker: core::marker::PhantomData<T>,
}

impl<'a, T: Packable4> Packed4SliceMut<'a, T> {
    /// Create a new `Packed4SliceMut` from packed bytes and logical length.
    /// Returns `None` if the backing buffer is too small for the requested length.
    #[inline]
    pub fn new(data: &'a mut [u8], len: usize) -> Option<Self> {
        let required_bytes = len.div_ceil(2);
        if data.len() < required_bytes {
            None
        } else {
            Some(Self {
                data,
                len,
                _marker: core::marker::PhantomData,
            })
        }
    }

    /// Returns the logical length (number of elements).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the underlying packed byte slice.
    #[inline(always)]
    pub fn as_packed_slice(&self) -> &[u8] {
        self.data
    }

    /// Returns the underlying mutable packed byte slice.
    #[inline(always)]
    pub fn as_packed_slice_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Return a borrowed read-only view of this mutable packed slice.
    #[inline]
    pub fn as_borrowed(&self) -> Packed4Slice<'_, T> {
        Packed4Slice {
            data: self.data,
            len: self.len,
            _marker: core::marker::PhantomData,
        }
    }

    /// Get an element at the given logical index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            None
        } else {
            let byte_idx = index / 2;
            let byte = self.data[byte_idx];
            let (low, high) = T::unpack_pair(byte);
            if index.is_multiple_of(2) {
                Some(low)
            } else {
                Some(high)
            }
        }
    }

    /// Returns true if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set an element at the given logical index.
    #[inline]
    pub fn set(&mut self, index: usize, val: T) {
        if index < self.len {
            let byte_idx = index / 2;
            let byte = self.data[byte_idx];
            let (mut low, mut high) = T::unpack_pair(byte);
            if index.is_multiple_of(2) {
                low = val;
            } else {
                high = val;
            }
            self.data[byte_idx] = T::pack_pair(low, high);
        }
    }

    /// Attempts to create a zero-copy mutable sub-slice of the packed slice.
    ///
    /// This is only possible without allocation if the range starts at an even logical index
    /// (aligned to byte boundaries). If the range starts at an odd index, returns `None`.
    #[inline]
    pub fn sub_slice_mut(self, range: core::ops::Range<usize>) -> Option<Self> {
        if range.start > range.end || range.end > self.len {
            return None;
        }
        if range.start.is_multiple_of(2) {
            let byte_start = range.start / 2;
            let byte_end = range.end.div_ceil(2);
            let sub_data = &mut self.data[byte_start..byte_end];
            let sub_len = range.end - range.start;
            Some(Self {
                data: sub_data,
                len: sub_len,
                _marker: core::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

impl<'a, T: Packable4> Packed4Slice<'a, T> {
    /// Unpack all elements into a destination slice.
    #[inline]
    pub fn unpack(&self, dest: &mut [T::Unpacked]) {
        let n = self.len.min(dest.len());
        let even_len = (n / 2) * 2;
        T::unpack_slice_packed(&self.data[..even_len / 2], &mut dest[..even_len]);
        if n % 2 != 0 {
            if let Some(b) = self.get(n - 1) {
                dest[n - 1] = T::unpack_single(b);
            }
        }
    }
}

impl<'a> Packed4Slice<'a, Bf4> {
    /// Unpack all elements into a destination slice of Bf16.
    #[inline]
    pub fn unpack_to_bf16(&self, dest: &mut [Bf16]) {
        self.unpack(dest);
    }
}

impl<'a> Packed4Slice<'a, F4> {
    /// Unpack all elements into a destination slice of F32.
    #[inline]
    pub fn unpack_to_f32(&self, dest: &mut [F32]) {
        self.unpack(dest);
    }
}

/// Type alias for a read-only view over a packed slice of Bf4 values.
pub type PackedBf4Slice<'a> = Packed4Slice<'a, Bf4>;
/// Type alias for a mutable view over a packed slice of Bf4 values.
pub type PackedBf4SliceMut<'a> = Packed4SliceMut<'a, Bf4>;
/// Type alias for a read-only view over a packed slice of F4 values.
pub type PackedF4Slice<'a> = Packed4Slice<'a, F4>;
/// Type alias for a mutable view over a packed slice of F4 values.
pub type PackedF4SliceMut<'a> = Packed4SliceMut<'a, F4>;

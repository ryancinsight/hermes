use crate::types::{Bf4, F4, Bf16, F32};

/// Trait for 4-bit types that can be packed two per byte.
pub trait Packable4: Copy + 'static {
    /// Pack a low and high element into a single byte.
    fn pack_pair(low: Self, high: Self) -> u8;
    /// Unpack a single byte into a low and high element.
    fn unpack_pair(packed: u8) -> (Self, Self);
}

impl Packable4 for Bf4 {
    #[inline(always)]
    fn pack_pair(low: Self, high: Self) -> u8 {
        Bf4::pack_pair(low, high)
    }
    #[inline(always)]
    fn unpack_pair(packed: u8) -> (Self, Self) {
        Bf4::unpack_pair(packed)
    }
}

impl Packable4 for F4 {
    #[inline(always)]
    fn pack_pair(low: Self, high: Self) -> u8 {
        F4::pack_pair(low, high)
    }
    #[inline(always)]
    fn unpack_pair(packed: u8) -> (Self, Self) {
        F4::unpack_pair(packed)
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
        let required_bytes = (len + 1) / 2;
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
            if index % 2 == 0 {
                Some(low)
            } else {
                Some(high)
            }
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
        let required_bytes = (len + 1) / 2;
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
            if index % 2 == 0 {
                low = val;
            } else {
                high = val;
            }
            self.data[byte_idx] = T::pack_pair(low, high);
        }
    }
}

impl<'a> Packed4Slice<'a, Bf4> {
    /// Unpack all elements into a destination slice of Bf16.
    #[inline]
    pub fn unpack_to_bf16(&self, dest: &mut [Bf16]) {
        let n = self.len.min(dest.len());
        let even_len = (n / 2) * 2;
        super::unpack::unpack_bf4_to_bf16_packed(&self.data[..even_len / 2], &mut dest[..even_len]);
        if n % 2 != 0 {
            if let Some(b) = self.get(n - 1) {
                let b_val = b.0 as u16;
                let sign = (b_val & 0x08) << 12;
                let rest = (b_val & 0x07) << 6;
                let bias_diff = if rest == 0 { 0 } else { 126 << 7 };
                dest[n - 1] = Bf16(half::bf16::from_bits(sign | (rest + bias_diff)));
            }
        }
    }
}

impl<'a> Packed4Slice<'a, F4> {
    /// Unpack all elements into a destination slice of F32.
    #[inline]
    pub fn unpack_to_f32(&self, dest: &mut [F32]) {
        let n = self.len.min(dest.len());
        let even_len = (n / 2) * 2;
        super::unpack::unpack_f4_to_f32_packed(&self.data[..even_len / 2], &mut dest[..even_len]);
        if n % 2 != 0 {
            if let Some(b) = self.get(n - 1) {
                dest[n - 1] = F32(b.to_f32());
            }
        }
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

use super::slice::{Packable4, Packed4Slice, Packed4SliceMut};
use crate::types::{Bf4, F4};
use alloc::vec::Vec;

/// A heap-allocated packed vector of 4-bit values, stored 2 per byte.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed4Vec<T: Packable4> {
    pub(crate) data: Vec<u8>,
    pub(crate) len: usize,
    pub(crate) _marker: core::marker::PhantomData<T>,
}

impl<T: Packable4> Packed4Vec<T> {
    /// Create a new empty `Packed4Vec`.
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            len: 0,
            _marker: core::marker::PhantomData,
        }
    }

    /// Create a new `Packed4Vec` with the given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity.div_ceil(2)),
            len: 0,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns the logical length.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Clear the vector.
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.len = 0;
    }

    /// Push an element to the back of the vector.
    #[inline]
    pub fn push(&mut self, val: T) {
        if self.len.is_multiple_of(2) {
            let byte = T::pack_pair(val, val);
            self.data.push(byte);
        } else {
            let last_idx = self.data.len() - 1;
            let last_byte = self.data[last_idx];
            let (low, _) = T::unpack_pair(last_byte);
            self.data[last_idx] = T::pack_pair(low, val);
        }
        self.len += 1;
    }

    /// Get an element at index.
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

    /// Set an element at index.
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

    /// Access the underlying packed bytes.
    #[inline]
    pub fn as_packed_slice(&self) -> &[u8] {
        &self.data
    }

    /// Access the underlying packed bytes mutably.
    #[inline]
    pub fn as_packed_slice_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Convert to a `Packed4Slice` view.
    #[inline]
    pub fn as_view(&self) -> Packed4Slice<'_, T> {
        Packed4Slice {
            data: &self.data,
            len: self.len,
            _marker: core::marker::PhantomData,
        }
    }

    /// Convert to a `Packed4SliceMut` view.
    #[inline]
    pub fn as_view_mut(&mut self) -> Packed4SliceMut<'_, T> {
        Packed4SliceMut {
            data: &mut self.data,
            len: self.len,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<T: Packable4> Default for Packed4Vec<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over a packed 4-bit slice.
pub struct Packed4Iter<'a, T: Packable4> {
    pub(crate) view: Packed4Slice<'a, T>,
    pub(crate) index: usize,
}

impl<'a, T: Packable4> Iterator for Packed4Iter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let val = self.view.get(self.index);
        if val.is_some() {
            self.index += 1;
        }
        val
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.view.len() - self.index;
        (rem, Some(rem))
    }
}

impl<'a, T: Packable4> ExactSizeIterator for Packed4Iter<'a, T> {}

impl<'a, T: Packable4> IntoIterator for Packed4Slice<'a, T> {
    type Item = T;
    type IntoIter = Packed4Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Packed4Iter {
            view: self,
            index: 0,
        }
    }
}

impl<'a, T: Packable4> core::iter::FusedIterator for Packed4Iter<'a, T> {}

impl<T: Packable4> IntoIterator for Packed4Vec<T> {
    type Item = T;
    type IntoIter = Packed4OwnedIter<T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Packed4OwnedIter {
            vec: self,
            index: 0,
        }
    }
}

/// Owning iterator over a `Packed4Vec`.
pub struct Packed4OwnedIter<T: Packable4> {
    vec: Packed4Vec<T>,
    index: usize,
}

impl<T: Packable4> Iterator for Packed4OwnedIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.index >= self.vec.len() {
            return None;
        }
        let val = self.vec.get(self.index)?;
        self.index += 1;
        Some(val)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.vec.len() - self.index;
        (rem, Some(rem))
    }
}

impl<T: Packable4> ExactSizeIterator for Packed4OwnedIter<T> {}
impl<T: Packable4> core::iter::FusedIterator for Packed4OwnedIter<T> {}

impl<T: Packable4> Extend<T> for Packed4Vec<T> {
    /// Extend the vector from any iterator yielding `T`.
    ///
    /// Uses `size_hint` to pre-allocate the backing byte storage before pushing,
    /// avoiding per-element reallocation when the iterator advertises a lower bound.
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        // Pre-allocate additional byte capacity from the size hint.
        let extra_bytes = lower.div_ceil(2);
        self.data.reserve(extra_bytes);
        for item in iter {
            self.push(item);
        }
    }
}

impl<T: Packable4> FromIterator<T> for Packed4Vec<T> {
    /// Collect any iterator of `T` into a `Packed4Vec<T>`.
    ///
    /// Uses the iterator's `size_hint` to reserve byte capacity before collecting,
    /// so a tight upper bound eliminates all but one allocation.
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = Self::new();
        vec.extend(iter);
        vec
    }
}

/// Type alias for a heap-allocated packed vector of Bf4 values.
pub type PackedBf4Vec = Packed4Vec<Bf4>;
/// Type alias for a heap-allocated packed vector of F4 values.
pub type PackedF4Vec = Packed4Vec<F4>;

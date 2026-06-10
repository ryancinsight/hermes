use super::slice::{Packable4, Packed4Slice};
use super::vec::Packed4Vec;
use crate::types::{Bf4, F4};

/// A Clone-on-Write (CoW) container for packed 4-bit elements.
///
/// Promotes zero-copy operations by borrowing packed byte buffers as read-only
/// `Packed4Slice`s, only upgrading to an owned `Packed4Vec` when mutation is requested.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Packed4Cow<'a, T: Packable4> {
    /// Borrowed read-only packed slice view.
    Borrowed(Packed4Slice<'a, T>),
    /// Owned packed vector.
    Owned(Packed4Vec<T>),
}

impl<'a, T: Packable4> Packed4Cow<'a, T> {
    /// Creates a borrowed `Packed4Cow` wrapping a `Packed4Slice`.
    #[inline(always)]
    pub fn borrowed(view: Packed4Slice<'a, T>) -> Self {
        Self::Borrowed(view)
    }

    /// Creates an owned `Packed4Cow` wrapping a `Packed4Vec`.
    #[inline(always)]
    pub fn owned(vec: Packed4Vec<T>) -> Self {
        Self::Owned(vec)
    }

    /// Returns true when this container is borrowing caller-owned packed bytes.
    #[inline(always)]
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Returns true when this container owns its packed byte storage.
    #[inline(always)]
    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Creates a borrowed `Packed4Cow` from a byte slice and logical length.
    /// Returns `None` if the byte slice is too small for the requested length.
    #[inline]
    pub fn from_packed_slice(data: &'a [u8], len: usize) -> Option<Self> {
        Packed4Slice::new(data, len).map(Self::Borrowed)
    }

    /// Obtains a read-only view of the packed slice.
    #[inline]
    pub fn as_view(&self) -> Packed4Slice<'_, T> {
        match self {
            Self::Borrowed(view) => *view,
            Self::Owned(vec) => vec.as_view(),
        }
    }

    /// Returns the logical length (number of 4-bit elements).
    #[inline(always)]
    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(view) => view.len(),
            Self::Owned(vec) => vec.len(),
        }
    }

    /// Returns true if empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get an element at the given logical index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<T> {
        match self {
            Self::Borrowed(view) => view.get(index),
            Self::Owned(vec) => vec.get(index),
        }
    }

    /// Upgrades the container to the `Owned` variant and returns a mutable reference to the underlying `Packed4Vec`.
    /// Subsequent calls are free (no reallocation).
    #[inline]
    pub fn to_mut(&mut self) -> &mut Packed4Vec<T> {
        if let Self::Borrowed(view) = *self {
            let mut owned = Packed4Vec::with_capacity(view.len());
            owned.data.extend_from_slice(view.data);
            owned.len = view.len;
            *self = Self::Owned(owned);
        }
        match self {
            Self::Owned(ref mut vec) => vec,
            _ => unreachable!(),
        }
    }

    /// Convert into the owned `Packed4Vec` type, allocating if borrowed.
    #[inline]
    pub fn into_owned(self) -> Packed4Vec<T> {
        match self {
            Self::Borrowed(view) => {
                let mut owned = Packed4Vec::with_capacity(view.len());
                owned.data.extend_from_slice(view.data);
                owned.len = view.len;
                owned
            }
            Self::Owned(vec) => vec,
        }
    }

    /// Sets an element at the given logical index, upgrading to owned if currently borrowed.
    #[inline]
    pub fn set(&mut self, index: usize, val: T) {
        let vec = self.to_mut();
        vec.set(index, val);
    }

    /// Attempts to create a zero-copy sub-slice of the packed Clone-on-Write container.
    ///
    /// Returns `Some(Packed4Cow::Borrowed)` if the sub-slice range starts at an even logical index
    /// (aligned to byte boundaries), otherwise returns `None`.
    #[inline]
    pub fn sub_slice(&self, range: core::ops::Range<usize>) -> Option<Packed4Cow<'_, T>> {
        match self {
            Self::Borrowed(view) => view.sub_slice(range).map(Packed4Cow::Borrowed),
            Self::Owned(vec) => vec.as_view().sub_slice(range).map(Packed4Cow::Borrowed),
        }
    }
}

pub enum Packed4CowIntoIter<'a, T: Packable4> {
    /// Iterator over a borrowed packed slice.
    Borrowed(super::vec::Packed4Iter<'a, T>),
    /// Iterator over an owned packed vector.
    Owned {
        /// The underlying packed vector.
        vec: Packed4Vec<T>,
        /// The current logical index.
        index: usize,
    },
}

impl<'a, T: Packable4> Iterator for Packed4CowIntoIter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Borrowed(iter) => iter.next(),
            Self::Owned { vec, index } => {
                let val = vec.get(*index);
                if val.is_some() {
                    *index += 1;
                }
                val
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Borrowed(iter) => iter.size_hint(),
            Self::Owned { vec, index } => {
                let rem = vec.len() - *index;
                (rem, Some(rem))
            }
        }
    }
}

impl<'a, T: Packable4> ExactSizeIterator for Packed4CowIntoIter<'a, T> {}

impl<'a, T: Packable4> IntoIterator for &'a Packed4Cow<'a, T> {
    type Item = T;
    type IntoIter = super::vec::Packed4Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Packed4Cow::Borrowed(view) => view.into_iter(),
            Packed4Cow::Owned(vec) => vec.as_view().into_iter(),
        }
    }
}

impl<'a, T: Packable4> IntoIterator for Packed4Cow<'a, T> {
    type Item = T;
    type IntoIter = Packed4CowIntoIter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Packed4Cow::Borrowed(view) => Packed4CowIntoIter::Borrowed(view.into_iter()),
            Packed4Cow::Owned(vec) => Packed4CowIntoIter::Owned { vec, index: 0 },
        }
    }
}

impl<T: Packable4> Default for Packed4Cow<'static, T> {
    #[inline(always)]
    fn default() -> Self {
        Self::Owned(Packed4Vec::new())
    }
}

/// Type alias for a Clone-on-Write view over packed Bf4 values.
pub type PackedBf4Cow<'a> = Packed4Cow<'a, Bf4>;
/// Type alias for a Clone-on-Write view over packed F4 values.
pub type PackedF4Cow<'a> = Packed4Cow<'a, F4>;

//! Clone-on-Write tensor container backed by [`AlignedVec`].

use crate::align::{Alignment, Unaligned};
use crate::vec::AlignedVec;

use super::error::TensorError;
use super::strides::row_major_strides;
use super::layout::{Layout, RowMajor};
use super::view::TensorView;

// ---------------------------------------------------------------------------
// TensorCow: Clone-on-Write tensor container
// ---------------------------------------------------------------------------

/// A Clone-on-Write (CoW) container for strided tensors.
pub enum TensorCow<'a, T: 'a, const N: usize, L = RowMajor, Align: Alignment = Unaligned> {
    /// Borrowed read-only tensor view.
    Borrowed(TensorView<'a, T, N, L, &'a [T]>),
    /// Owned aligned tensor buffer.
    Owned {
        /// Underlying aligned memory.
        data: AlignedVec<T, Align>,
        /// Logical shape of the tensor.
        shape: [usize; N],
        /// Dimension strides.
        strides: [usize; N],
    },
}

impl<'a, T: Copy + 'a, const N: usize, L: Layout, Align> TensorCow<'a, T, N, L, Align>
where
    Align: Alignment,
{
    /// Create a borrowed `TensorCow` wrapping a `TensorView`.
    #[inline]
    pub fn borrowed(view: TensorView<'a, T, N, L, &'a [T]>) -> Self {
        Self::Borrowed(view)
    }

    /// Create an owned `TensorCow` from an `AlignedVec` and shape.
    #[inline]
    pub fn owned(data: AlignedVec<T, Align>, shape: [usize; N]) -> Self {
        let strides = row_major_strides(shape);
        Self::Owned {
            data,
            shape,
            strides,
        }
    }

    /// Create an owned `TensorCow` with explicit strides.
    #[inline]
    pub fn owned_with_strides(
        data: AlignedVec<T, Align>,
        shape: [usize; N],
        strides: [usize; N],
    ) -> Self {
        Self::Owned {
            data,
            shape,
            strides,
        }
    }

    /// Obtain a read-only view of this tensor.
    #[inline]
    pub fn as_view(&self) -> TensorView<'_, T, N, L, &'_ [T]> {
        match self {
            Self::Borrowed(view) => *view,
            Self::Owned {
                data,
                shape,
                strides,
            } => TensorView::with_strides(data.as_slice(), *shape, *strides)
                .expect("Owned variant stores pre-validated shape and strides"),
        }
    }

    /// Returns the total logical element count.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Borrowed(view) => view.num_elements(),
            Self::Owned { shape, .. } => shape.iter().product(),
        }
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns logical shape.
    #[inline]
    pub fn shape(&self) -> [usize; N] {
        match self {
            Self::Borrowed(view) => view.shape(),
            Self::Owned { shape, .. } => *shape,
        }
    }

    /// Returns tensor strides.
    #[inline]
    pub fn strides(&self) -> [usize; N] {
        match self {
            Self::Borrowed(view) => view.strides(),
            Self::Owned { strides, .. } => *strides,
        }
    }

    /// Returns whether tensor is contiguous row-major.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        match self {
            Self::Borrowed(view) => view.is_contiguous(),
            Self::Owned { shape, strides, .. } => {
                let expected = row_major_strides(*shape);
                *strides == expected
            }
        }
    }

    /// Upgrades to `Owned` if currently borrowed and returns a mutable reference to the `AlignedVec`.
    #[inline]
    pub fn to_mut(&mut self) -> &mut AlignedVec<T, Align> {
        if let Self::Borrowed(view) = *self {
            let owned = AlignedVec::from_slice(view.as_slice());
            *self = Self::Owned {
                data: owned,
                shape: view.shape(),
                strides: view.strides(),
            };
        }
        match self {
            Self::Owned { data, .. } => data,
            _ => unreachable!(),
        }
    }

    /// Converts into the owned `AlignedVec` storage.
    #[inline]
    pub fn into_owned(self) -> AlignedVec<T, Align> {
        match self {
            Self::Borrowed(view) => AlignedVec::from_slice(view.as_slice()),
            Self::Owned { data, .. } => data,
        }
    }

    /// Reshapes the tensor to a different rank `M` without allocation.
    #[inline]
    pub fn reshape<const M: usize>(
        self,
        new_shape: [usize; M],
    ) -> Result<TensorCow<'a, T, M, RowMajor, Align>, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }
        let old_count = self.len();
        let new_count = new_shape.iter().product::<usize>();
        if old_count != new_count {
            return Err(TensorError::ShapeMismatch);
        }
        match self {
            Self::Borrowed(view) => {
                let reshaped = view.reshape(new_shape)?;
                Ok(TensorCow::Borrowed(reshaped))
            }
            Self::Owned { data, .. } => {
                let strides = row_major_strides(new_shape);
                Ok(TensorCow::Owned {
                    data,
                    shape: new_shape,
                    strides,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

impl<'a, T: Clone + 'a, const N: usize, L, Align> Clone for TensorCow<'a, T, N, L, Align>
where
    Align: Alignment,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(view) => Self::Borrowed(*view),
            Self::Owned {
                data,
                shape,
                strides,
            } => Self::Owned {
                data: data.clone(),
                shape: *shape,
                strides: *strides,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Deref
// ---------------------------------------------------------------------------

impl<'a, T: 'a, const N: usize, L, Align> core::ops::Deref for TensorCow<'a, T, N, L, Align>
where
    Align: Alignment,
{
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(view) => view.as_slice(),
            Self::Owned { data, .. } => data.as_slice(),
        }
    }
}

// ---------------------------------------------------------------------------
// PartialEq / Eq
// ---------------------------------------------------------------------------

impl<'a, 'b, T, const N: usize, L1, L2, A1, A2> PartialEq<TensorCow<'b, T, N, L2, A2>>
    for TensorCow<'a, T, N, L1, A1>
where
    T: PartialEq,
    A1: Alignment,
    A2: Alignment,
{
    #[inline]
    fn eq(&self, other: &TensorCow<'b, T, N, L2, A2>) -> bool {
        let s1: &[T] = self;
        let s2: &[T] = other;
        s1 == s2
    }
}

impl<'a, T, const N: usize, L, Align> Eq for TensorCow<'a, T, N, L, Align>
where
    T: Eq,
    Align: Alignment,
{
}

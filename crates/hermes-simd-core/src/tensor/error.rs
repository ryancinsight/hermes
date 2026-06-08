//! Error type for tensor construction and indexing.

/// Error type for tensor construction and indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// The element count derived from `shape` does not fit in `data.len()`.
    ShapeMismatch,
    /// A row or slice index is out of bounds.
    IndexOutOfBounds,
    /// The view is not contiguous and cannot be reshaped without copying.
    NotContiguous,
}

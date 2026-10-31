//! Error type for tensor construction and indexing.

/// Error type for tensor construction and indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorError {
    /// The logical element count or reachable span overflows, or the layout
    /// addresses an element outside the backing storage.
    ShapeMismatch,
    /// A row or slice index is out of bounds.
    IndexOutOfBounds,
    /// The view is not contiguous and cannot be reshaped without copying.
    NotContiguous,
}

//! Private stride and offset helpers shared across tensor modules.

use super::error::TensorError;

/// An empty axis makes the cardinality zero before any product is evaluated.
#[inline]
pub(super) fn checked_element_count(shape: &[usize]) -> Result<usize, TensorError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, &length| {
        count.checked_mul(length).ok_or(TensorError::ShapeMismatch)
    })
}

/// Prove every logical index maps into storage, including repeated positions.
#[inline]
pub(super) fn validate_strides<const N: usize>(
    data_len: usize,
    shape: &[usize; N],
    strides: &[usize; N],
) -> Result<(), TensorError> {
    if checked_element_count(shape)? == 0 {
        return Ok(());
    }
    // Nonnegative strides attain their maximum at the final logical index.
    // Starting at one proves the exclusive storage bound without overflowing
    // a later `maximum_offset + 1`. Rank zero therefore needs one element.
    let span = shape
        .iter()
        .zip(strides)
        .try_fold(1usize, |span, (&length, &stride)| {
            (length - 1)
                .checked_mul(stride)
                .and_then(|offset| span.checked_add(offset))
                .ok_or(TensorError::ShapeMismatch)
        })?;
    if span > data_len {
        return Err(TensorError::ShapeMismatch);
    }
    Ok(())
}

/// Compute row-major strides for `shape`: `strides[i] = ∏_{j=i+1..N} shape[j]`.
#[inline(always)]
pub(super) fn row_major_strides<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut strides = [1usize; N];
    let mut acc = 1usize;
    let mut i = N;
    while i > 0 {
        i -= 1;
        strides[i] = acc;
        // A validated nonempty shape has a representable product. Only
        // unreachable suffixes of empty shapes can saturate here.
        acc = acc.saturating_mul(shape[i]);
    }
    strides
}

/// Compute flat offset: `∑ idx[i] * strides[i]`.
#[inline(always)]
pub(super) fn compute_offset(idx: &[usize], strides: &[usize]) -> usize {
    let mut offset = 0usize;
    for i in 0..idx.len() {
        offset += idx[i] * strides[i];
    }
    offset
}

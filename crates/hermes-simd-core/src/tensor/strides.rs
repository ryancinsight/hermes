//! Private stride and offset helpers shared across tensor modules.

/// Compute row-major strides for `shape`: `strides[i] = ∏_{j=i+1..N} shape[j]`.
#[inline(always)]
pub(super) fn row_major_strides<const N: usize>(shape: [usize; N]) -> [usize; N] {
    let mut strides = [1usize; N];
    let mut acc = 1usize;
    let mut i = N;
    while i > 0 {
        i -= 1;
        strides[i] = acc;
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

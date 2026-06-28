//! Checked matrix-dimension span arithmetic for the register-blocked tiling
//! kernels (SSOT for operand-length validation).
//!
//! The GEMV/GEMM kernels validate that the caller's operand slices are long
//! enough *before* issuing `unsafe` SIMD loads/stores at computed offsets. The
//! required span is a product of caller-supplied `usize` dimensions
//! (`rows·cols`, `(nrows−1)·lda + ncols`), so an adversarial dimension can
//! overflow `usize`. Under `overflow-checks = false` (the release default) the
//! product would wrap to a small value, the length guard would pass spuriously,
//! and the subsequent load would read out of bounds — a memory-safety hole
//! reachable from the public dispatch API (e.g. `lda = usize::MAX`, `nrows = 2`).
//!
//! These helpers compute the span with checked arithmetic and treat overflow as
//! "no slice can satisfy this span": the caller maps [`None`] to
//! [`crate::view::SimdError::LengthMismatch`]. The OOB path is closed in every
//! build profile — correctness no longer depends on `overflow-checks` — and the
//! two GEMV checkers share one authoritative span computation.

/// Span of a row-major sub-matrix: `nrows` rows of `ncols` elements at row
/// stride `lda`. The last row reaches furthest, so the half-open extent is
/// `[0, (nrows−1)·lda + ncols)`. Returns [`None`] on `usize` overflow (an
/// unrepresentable span no allocation can satisfy); `nrows == 0` is the empty
/// span `0`.
#[inline]
pub(super) fn checked_strided_span(nrows: usize, ncols: usize, lda: usize) -> Option<usize> {
    match nrows {
        0 => Some(0),
        n => (n - 1).checked_mul(lda)?.checked_add(ncols),
    }
}

/// Area of a dense `rows × cols` matrix (`rows·cols`), or [`None`] on `usize`
/// overflow.
#[inline]
pub(super) fn checked_area(rows: usize, cols: usize) -> Option<usize> {
    rows.checked_mul(cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_span_matches_unchecked_for_valid_dims() {
        // Packed (lda == ncols) and strided (lda > ncols) cases agree with the
        // direct formula when no overflow occurs.
        assert_eq!(checked_strided_span(4, 6, 6), Some(3 * 6 + 6));
        assert_eq!(checked_strided_span(4, 6, 10), Some(3 * 10 + 6));
        assert_eq!(checked_strided_span(1, 6, 10), Some(6));
        assert_eq!(checked_strided_span(0, 6, 10), Some(0));
    }

    #[test]
    fn strided_span_rejects_overflow() {
        // The exact adversarial input from the dispatch API: a 2-row matrix with a
        // maximal stride. Unchecked, `(2-1)*usize::MAX + ncols` wraps to `ncols-1`
        // and defeats the length guard; checked arithmetic reports overflow.
        assert_eq!(checked_strided_span(2, 6, usize::MAX), None);
        assert_eq!(checked_strided_span(usize::MAX, 1, 2), None);
    }

    #[test]
    fn area_matches_and_rejects_overflow() {
        assert_eq!(checked_area(3, 4), Some(12));
        assert_eq!(checked_area(0, usize::MAX), Some(0));
        assert_eq!(checked_area(usize::MAX, 2), None);
        assert_eq!(checked_area(1 << 40, 1 << 40), None);
    }
}

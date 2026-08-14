//! Packing helpers for the row-major right-hand operand of AMX GEMM.

const TILE_ROWS: usize = 16;

/// Transpose one row-major `K × 16` right-hand panel into an AMX `16 × K` tile.
///
/// AMX dot-product instructions consume both source tiles as rows of dot-product
/// vectors. The public GEMM contract remains row-major `B[K][N]`, so the panel
/// must be repacked at the instruction boundary.
///
/// # Safety
///
/// `source` must point to a row-major matrix containing `TILE_ROWS` columns
/// beginning at `column` and `panel_depth` rows beginning at `depth_offset`.
/// Each source row must contain at least `source_stride` elements, and the
/// destination must have exactly `TILE_ROWS * panel_depth` elements.
pub(crate) unsafe fn pack_rhs_panel<T: Copy>(
    source: *const T,
    source_stride: usize,
    column: usize,
    depth_offset: usize,
    panel_depth: usize,
    destination: &mut [T],
) {
    assert!(panel_depth > 0, "invariant: AMX panel depth is non-zero");
    assert_eq!(
        destination.len(),
        TILE_ROWS * panel_depth,
        "invariant: AMX panel storage matches the configured tile shape"
    );

    for (row, packed_row) in destination.chunks_exact_mut(panel_depth).enumerate() {
        for (depth, value) in packed_row.iter_mut().enumerate() {
            *value = unsafe {
                // SAFETY: The caller's matrix extent and stride guarantee that
                // this transposed panel element is within the source allocation.
                source
                    .add((depth_offset + depth) * source_stride + column + row)
                    .read()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pack_rhs_panel;

    #[test]
    fn rhs_panel_is_transposed_into_amx_row_vectors() {
        let source: Vec<i8> = (0..8 * 24).map(|value| value as i8).collect();
        let mut destination = [0i8; 16 * 8];

        unsafe {
            pack_rhs_panel(source.as_ptr(), 24, 3, 0, 8, &mut destination);
        }

        for row in 0..16 {
            for depth in 0..8 {
                assert_eq!(destination[row * 8 + depth], source[depth * 24 + 3 + row]);
            }
        }
    }
}

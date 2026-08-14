//! Packing helpers for the row-major right-hand operand of AMX GEMM.

/// Pack one row-major right-hand panel into an AMX dot-product tile.
///
/// AMX dot-product instructions consume the right-hand tile in groups of
/// `GROUP` depth elements per output column. The public GEMM contract remains
/// row-major `B[K][N]`, so the panel is repacked at the instruction boundary:
/// TDPBSSD uses four byte elements per group and TDPBF16PS uses two BF16
/// elements per group. The packed tile therefore has `K / GROUP` rows and
/// `N * GROUP` elements per row.
///
/// # Safety
///
/// `source` must point to a row-major matrix containing the destination's
/// output columns beginning at `column` and `panel_depth` rows beginning at
/// `depth_offset`. Each source row must contain at least `source_stride`
/// elements, and the destination must have exactly `panel_depth * N` elements,
/// where `N` is the number of output columns in the panel.
pub(crate) unsafe fn pack_rhs_panel<T: Copy, const GROUP: usize>(
    source: *const T,
    source_stride: usize,
    column: usize,
    depth_offset: usize,
    panel_depth: usize,
    destination: &mut [T],
) {
    assert!(GROUP > 0, "invariant: AMX dot-product group is non-zero");
    assert!(panel_depth > 0, "invariant: AMX panel depth is non-zero");
    assert_eq!(
        panel_depth % GROUP,
        0,
        "invariant: AMX panel depth is divisible by the dot-product group"
    );
    assert_eq!(
        destination.len() % panel_depth,
        0,
        "invariant: AMX packed panel has an integral output width"
    );
    let output_columns = destination.len() / panel_depth;
    assert_eq!(
        destination.len(),
        (panel_depth / GROUP) * output_columns * GROUP,
        "invariant: AMX packed panel matches the configured tile shape"
    );

    for (group, packed_row) in destination
        .chunks_exact_mut(output_columns * GROUP)
        .enumerate()
    {
        for (output_column, packed_group) in packed_row.chunks_exact_mut(GROUP).enumerate() {
            for (lane, value) in packed_group.iter_mut().enumerate() {
                *value = unsafe {
                    // SAFETY: The caller's matrix extent and stride guarantee
                    // that this packed panel element is within the source
                    // allocation.
                    source
                        .add(
                            (depth_offset + group * GROUP + lane) * source_stride
                                + column
                                + output_column,
                        )
                        .read()
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::pack_rhs_panel;

    #[test]
    fn rhs_panel_groups_depth_values_by_output_column() {
        let source: Vec<i8> = (0..8 * 24).map(|value| value as i8).collect();
        let mut destination = [0i8; 2 * 16 * 4];

        unsafe {
            pack_rhs_panel::<_, 4>(source.as_ptr(), 24, 3, 0, 8, &mut destination);
        }

        for group in 0..2 {
            for output_column in 0..16 {
                for lane in 0..4 {
                    assert_eq!(
                        destination[group * 64 + output_column * 4 + lane],
                        source[(group * 4 + lane) * 24 + 3 + output_column]
                    );
                }
            }
        }
    }

    #[test]
    fn rhs_panel_supports_bf16_pair_groups() {
        let source: Vec<u16> = (0..7 * 12).collect();
        let mut destination = [0u16; 3 * 6 * 2];

        unsafe {
            pack_rhs_panel::<_, 2>(source.as_ptr(), 12, 2, 1, 6, &mut destination);
        }

        for group in 0..3 {
            for output_column in 0..6 {
                for lane in 0..2 {
                    assert_eq!(
                        destination[group * 12 + output_column * 2 + lane],
                        source[(1 + group * 2 + lane) * 12 + 2 + output_column]
                    );
                }
            }
        }
    }
}

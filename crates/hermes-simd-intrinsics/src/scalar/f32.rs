//! Fallback scalar f32 kernel.
//!
//! All SIMD operations degenerate to element-wise scalar loops. This is the
//! universal fallback when no hardware SIMD feature is detected.

impl_scalar_float_kernel!(f32, 4, 31, 0xFFFF_FFFF_u32);

#[cfg(test)]
mod tests {
    use crate::Scalar;
    use hermes_simd_core::align::Unaligned;
    use hermes_simd_core::execution::Unmasked;
    use hermes_simd_core::view::SimdView;

    /// The view's extremum scan issues raw-pointer vector loads. `hermes-simd`
    /// holds the behavioural tests, but miri runs only `hermes-simd-core` and
    /// this crate, so driving the scan through a concrete backend here is what
    /// puts those loads under the interpreter.
    /// Located extremum: slice position paired with the stored element.
    type Extremum = Option<(usize, f32)>;

    fn scan(data: &[f32]) -> (Extremum, Extremum) {
        let view = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(data)
            .expect("invariant: any slice is a valid unaligned view");
        (view.argmin(), view.argmax())
    }

    #[test]
    fn extremum_scan_stays_in_bounds_across_lengths() {
        // Lengths straddle the vector body and the scalar tail.
        for len in [0_usize, 1, 3, 4, 5, 7, 8, 9, 16, 17, 33] {
            let data: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();
            let (minimum, maximum) = scan(&data);
            if len == 0 {
                assert_eq!(minimum, None);
                assert_eq!(maximum, None);
            } else {
                assert_eq!(minimum.map(|(index, _)| index), Some(len - 1), "len {len}");
                assert_eq!(maximum.map(|(index, _)| index), Some(0), "len {len}");
            }
        }
    }

    /// The copy-on-write constructors build their output buffer with
    /// `with_capacity` and fill it through a raw pointer. Running them under
    /// miri is what proves every element is written: a missed one leaves
    /// uninitialized memory that the value comparison then reads.
    #[test]
    fn cow_constructors_initialize_every_element() {
        use hermes_simd_core::cow::SimdCow;
        use hermes_simd_core::ops::{Inclusive, ScanAdd};

        // Lengths straddling the 4-lane body leave a scalar tail of every size.
        for len in [0_usize, 1, 3, 4, 5, 7, 8, 9, 17] {
            let src: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
            let cow = SimdCow::<f32, Scalar, Unaligned>::from_slice(&src);

            let scaled = cow.mul_scalar_cow(2.0);
            let expected: Vec<f32> = src.iter().map(|v| v * 2.0).collect();
            assert_eq!(scaled.as_ref(), expected.as_slice(), "mul_scalar len {len}");

            let negated = cow.map_cow(hermes_simd_core::ops::Neg);
            let expected: Vec<f32> = src.iter().map(|v| -v).collect();
            assert_eq!(negated.as_ref(), expected.as_slice(), "map_cow len {len}");

            let filled = SimdCow::<f32, Scalar, Unaligned>::splat_fill(-3.5, len);
            assert_eq!(filled.as_ref(), vec![-3.5_f32; len], "splat_fill len {len}");

            let fused = cow.fma_cow(&cow, &cow).expect("invariant: equal lengths");
            let expected: Vec<f32> = src.iter().map(|v| v * v + v).collect();
            assert_eq!(fused.as_ref(), expected.as_slice(), "fma_cow len {len}");

            let indices: Vec<i32> = (0..len as i32).rev().collect();
            let gathered = cow.gather(&indices).expect("invariant: indices in range");
            let expected: Vec<f32> = indices.iter().map(|&i| src[i as usize]).collect();
            assert_eq!(gathered.as_ref(), expected.as_slice(), "gather len {len}");

            let scanned = cow
                .prefix_scan(ScanAdd, Inclusive)
                .expect("invariant: output length equals input length");
            let mut running = 0.0_f32;
            let expected: Vec<f32> = src
                .iter()
                .map(|v| {
                    running += v;
                    running
                })
                .collect();
            assert_eq!(
                scanned.as_ref(),
                expected.as_slice(),
                "prefix_scan len {len}"
            );
        }
    }

    #[test]
    fn extremum_scan_rejects_nan_in_every_position() {
        for len in [1_usize, 4, 5, 9, 17] {
            for nan_at in 0..len {
                let mut data = vec![1.0_f32; len];
                data[nan_at] = f32::NAN;
                assert_eq!(scan(&data), (None, None), "len {len}, NaN at {nan_at}");
            }
        }
    }

    /// `SpMV` kernels index `x`, `values`, and `col_indices` through raw pointers
    /// with the length bookkeeping stated in their `SAFETY` notes. The
    /// integration tests exercise them on the host SIMD backend, but miri runs
    /// only this crate and `hermes-simd-core`; driving each format through the
    /// `Scalar` backend here puts that pointer arithmetic under the interpreter,
    /// checked against an independent dense reference.
    mod spmv {
        use crate::Scalar;
        use hermes_simd_core::sparse::{
            CsrData, DenseWithMaskData, SellPData, SparseSpMv, SparseView, ValidatedData,
        };

        /// Dense reference: `y[r] += Σ_c A[r][c] * x[c]`, computed from a
        /// row-major dense matrix independent of any sparse kernel.
        fn dense_ref(a: &[f32], x: &[f32], nrows: usize, ncols: usize, y: &mut [f32]) {
            for r in 0..nrows {
                let mut acc = 0.0_f32;
                for c in 0..ncols {
                    acc += a[r * ncols + c] * x[c];
                }
                y[r] += acc;
            }
        }

        #[test]
        fn csr_matches_dense_across_row_lengths() {
            // ncols 24, x[c] = c+1. Row 0 has 22 nonzeros so it spans the
            // 4x-unrolled body [0,16), the SIMD tail [16,20), and the scalar
            // tail [20,22) for LANE_COUNT 4; row 1 has 2 (< LANE_COUNT, wholly
            // scalar); row 2 is empty (the `row_nnz == 0` early-continue).
            let ncols = 24;
            let x: Vec<f32> = (0..ncols).map(|c| c as f32 + 1.0).collect();

            let mut values = Vec::new();
            let mut col_indices = Vec::new();
            let mut row_ptr = vec![0i32];
            let mut dense = vec![0.0_f32; 3 * ncols];

            for c in 0..22i32 {
                let v = (c as f32 + 1.0) * 0.5;
                values.push(v);
                col_indices.push(c);
                dense[c as usize] = v;
            }
            row_ptr.push(values.len() as i32);
            for &(c, v) in &[(5i32, 2.0f32), (17, 3.0)] {
                values.push(v);
                col_indices.push(c);
                dense[ncols + c as usize] = v;
            }
            row_ptr.push(values.len() as i32);
            row_ptr.push(values.len() as i32); // empty row 2

            let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 3, ncols);
            let view = SparseView::<f32, _, Scalar>::from_validated_csr(
                ValidatedData::new(data).expect("fixture validates"),
            );

            let mut y = vec![1.0_f32; 3];
            view.spmv(&x, &mut y);

            let mut want = vec![1.0_f32; 3];
            dense_ref(&dense, &x, 3, ncols, &mut want);
            assert_eq!(y, want);
        }

        #[test]
        fn dense_with_mask_matches_dense() {
            // 2 x 10: masked lanes must contribute zero. 10 columns cover the
            // SIMD body [0,8) and the scalar tail [8,10) at LANE_COUNT 4.
            let (nrows, ncols) = (2, 10);
            let x: Vec<f32> = (0..ncols).map(|c| c as f32 - 3.0).collect();
            let mut values = vec![0.0_f32; nrows * ncols];
            let mut mask = vec![false; nrows * ncols];
            let mut dense = vec![0.0_f32; nrows * ncols];
            for r in 0..nrows {
                for c in 0..ncols {
                    if (r + c) % 3 == 0 {
                        let v = (r * ncols + c) as f32 * 0.25 - 1.0;
                        values[r * ncols + c] = v;
                        mask[r * ncols + c] = true;
                        dense[r * ncols + c] = v;
                    }
                }
            }
            let packed = hermes_simd_core::mask::PackedMask::from_bools(&mask);
            let data = DenseWithMaskData::new(&values[..], packed.as_view(), nrows, ncols);
            let view = SparseView::<f32, _, Scalar>::from_dense_with_mask(data);

            let mut y = vec![0.0_f32; nrows];
            view.spmv(&x, &mut y);

            let mut want = vec![0.0_f32; nrows];
            dense_ref(&dense, &x, nrows, ncols, &mut want);
            assert_eq!(y, want);
        }

        #[test]
        fn sellp_matches_dense() {
            // One slice of C = LANE_COUNT = 4 rows takes the vectorized path.
            // Padding lanes carry col_index 0 with value 0, contributing nothing.
            const C: usize = 4;
            let ncols = 6;
            let x: Vec<f32> = (0..ncols).map(|c| c as f32 + 2.0).collect();
            // Row r keeps two nonzeros at columns r and (r+2)%ncols.
            let mut dense = vec![0.0_f32; C * ncols];
            let col_count = 2usize;
            let mut values = vec![0.0_f32; col_count * C];
            let mut col_indices = vec![0i32; col_count * C];
            for row in 0..C {
                for (k, &c) in [row, (row + 2) % ncols].iter().enumerate() {
                    let v = (row + k + 1) as f32;
                    values[k * C + row] = v;
                    col_indices[k * C + row] = c as i32;
                    dense[row * ncols + c] = v;
                }
            }
            // `slice_ptr` is CSR-like: one offset per slice plus the end.
            let slice_ptr = [0i32, (col_count * C) as i32];
            let slice_col_count = [col_count as i32];
            let data = SellPData::<f32, C>::new(
                &values[..],
                &col_indices[..],
                &slice_ptr[..],
                &slice_col_count[..],
                C,
                ncols,
            );
            let view = SparseView::<f32, _, Scalar>::from_validated_sellp(
                ValidatedData::new(data).expect("fixture validates"),
            );

            let mut y = vec![0.0_f32; C];
            view.spmv(&x, &mut y);

            let mut want = vec![0.0_f32; C];
            dense_ref(&dense, &x, C, ncols, &mut want);
            assert_eq!(y, want);
        }
    }

    /// `SparseOps` elementwise-multiply kernels also index through raw pointers.
    /// Same rationale as the `spmv` module: drive them via the `Scalar` backend
    /// so miri checks the pointer arithmetic, and cover the newly added bounds
    /// guards that keep the CSR gather sound on an unvalidated view.
    mod ops {
        use crate::Scalar;
        use hermes_simd_core::sparse::{CsrData, DenseWithMaskData, SparseOps, SparseView};

        #[test]
        fn csr_elementwise_mul_dense_matches_reference() {
            // `out[k] = values[k] * dense[col_indices[k]]`; row 0 has 6 nonzeros
            // (spanning the 4-lane SIMD body and the 2-element scalar tail), row 1
            // has one. `dense` is the length-`ncols` per-column vector.
            let values = [2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let col_indices = [0i32, 1, 2, 3, 4, 5, 3];
            let row_ptr = [0i32, 6, 7];
            let ncols = 6;
            let dense: Vec<f32> = (0..ncols).map(|c| c as f32 + 1.0).collect();

            let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, ncols);
            let view = SparseView::<f32, _, Scalar>::from_csr(data);

            let mut out = vec![0.0f32; values.len()];
            view.elementwise_mul_dense(&dense, &mut out);

            let want: Vec<f32> = values
                .iter()
                .zip(col_indices.iter())
                .map(|(&v, &c)| v * dense[c as usize])
                .collect();
            assert_eq!(out, want);
        }

        #[test]
        #[should_panic(expected = "structural validation")]
        fn csr_elementwise_mul_dense_rejects_out_of_range_column() {
            // Column index 9 exceeds `ncols`, so the SIMD gather would read out of
            // bounds. The guard must reject it before any unchecked access rather
            // than let safe code reach undefined behavior.
            let values = [1.0f32, 2.0, 3.0, 4.0];
            let col_indices = [0i32, 9, 2, 3];
            let row_ptr = [0i32, 4];
            let dense = [1.0f32; 4];
            let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 1, 4);
            let view = SparseView::<f32, _, Scalar>::from_csr(data);
            let mut out = vec![0.0f32; 4];
            view.elementwise_mul_dense(&dense, &mut out);
        }

        #[test]
        #[should_panic(expected = "dense len")]
        fn csr_elementwise_mul_dense_rejects_short_dense() {
            // Valid indices, but `dense` shorter than `ncols` — the gather would
            // still read past its end, so the length guard must reject it.
            let values = [1.0f32, 2.0, 3.0, 4.0];
            let col_indices = [0i32, 1, 2, 3];
            let row_ptr = [0i32, 4];
            let dense = [1.0f32; 2]; // < ncols = 4
            let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 1, 4);
            let view = SparseView::<f32, _, Scalar>::from_csr(data);
            let mut out = vec![0.0f32; 4];
            view.elementwise_mul_dense(&dense, &mut out);
        }

        #[test]
        fn dense_with_mask_elementwise_mul_matches_reference() {
            // 10 elements span the 4-lane body and a 2-element tail; masked-off
            // lanes must produce zero.
            let len = 10;
            let values: Vec<f32> = (0..len).map(|i| i as f32 + 1.0).collect();
            let mask: Vec<bool> = (0..len).map(|i| i % 3 != 0).collect();
            let dense: Vec<f32> = (0..len).map(|i| (i as f32) * 0.5 - 1.0).collect();

            let packed = hermes_simd_core::mask::PackedMask::from_bools(&mask);
            let data = DenseWithMaskData::new(&values[..], packed.as_view(), 1, len);
            let view = SparseView::<f32, _, Scalar>::from_dense_with_mask(data);

            let mut out = vec![-1.0f32; len];
            view.elementwise_mul_dense(&dense, &mut out);

            let want: Vec<f32> = (0..len)
                .map(|i| if mask[i] { values[i] * dense[i] } else { 0.0 })
                .collect();
            assert_eq!(out, want);
        }

        #[test]
        #[expect(
            clippy::float_cmp,
            reason = "The sparse reduction test compares an exact manufactured sum"
        )]
        fn csr_sum_values_matches_reference() {
            let values = [1.5f32, -2.0, 3.25, 4.0, 5.5];
            let col_indices = [0i32, 1, 2, 0, 1];
            let row_ptr = [0i32, 3, 5];
            let data = CsrData::new(&values[..], &col_indices[..], &row_ptr[..], 2, 3);
            let view = SparseView::<f32, _, Scalar>::from_csr(data);
            assert_eq!(view.sum_values(), values.iter().sum::<f32>());
        }
    }

    /// `reduce` and `zip_reduce` do raw-pointer chunked loads through the
    /// register wrapper. The integration tests cover the host SIMD backend;
    /// this drives them via the `Scalar` backend so miri checks the pointer
    /// arithmetic across the 4×-unrolled body, the single-vector tail, and the
    /// scalar remainder (`Scalar` uses `chunk_size = 16`).
    mod reduce {
        use crate::Scalar;
        use hermes_simd_core::ops::{Dot, Sum};
        use hermes_simd_core::{SimdView, Unaligned, Unmasked};

        fn view(data: &[f32]) -> SimdView<'_, f32, Scalar, Unaligned, Unmasked, &[f32]> {
            SimdView::new(data).expect("scalar view always constructs")
        }

        #[test]
        fn reduce_and_zip_reduce_match_scalar_across_lengths() {
            // 0 and 1 exercise the empty/short guards; 16/17 the unrolled body
            // and its boundary; 38 spans unrolled (32) + simd tail (36) + scalar
            // tail (2).
            for len in [0usize, 1, 4, 15, 16, 17, 32, 38] {
                let a: Vec<f32> = (0..len).map(|i| (i % 7) as f32 - 3.0).collect();
                let b: Vec<f32> = (0..len).map(|i| (i % 5) as f32 + 1.0).collect();

                let got_sum = view(&a).reduce(Sum);
                let want_sum: f32 = a.iter().sum();
                assert!(
                    (got_sum - want_sum).abs() <= 1e-4 * (1.0 + want_sum.abs()),
                    "sum len {len}: {got_sum} vs {want_sum}"
                );

                let got_dot = view(&a)
                    .zip_reduce(&view(&b), Dot)
                    .expect("equal-length views");
                let want_dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
                assert!(
                    (got_dot - want_dot).abs() <= 1e-4 * (1.0 + want_dot.abs()),
                    "dot len {len}: {got_dot} vs {want_dot}"
                );
            }
        }
    }

    /// The register-blocked tiling kernels index `A`/`B`/`C` through raw pointers
    /// at 2D tile offsets. The integration tests cover the host SIMD backend;
    /// this drives dot/gemv/gemv-transpose/gemm via the `Scalar` backend so miri
    /// checks that offset arithmetic against independent scalar references, at a
    /// size that exercises the tiled body plus the row/column remainders.
    mod tiling {
        use crate::Scalar;
        use hermes_simd_core::tiling::{
            tiled_dot, tiled_gemm, tiled_gemv, TilingPolicy, TilingStrategy,
        };
        use hermes_simd_core::{SimdView, Unaligned};

        fn v(data: &[f32]) -> SimdView<'_, f32, Scalar, Unaligned> {
            SimdView::new(data).expect("scalar view")
        }

        fn approx(a: f32, b: f32) -> bool {
            (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()))
        }

        #[test]
        fn tiled_dot_matches_scalar() {
            // 11 = one 8-wide tile (LANE_COUNT 4 * TILE_M 2) + 3-element tail.
            let a: Vec<f32> = (0..11).map(|i| i as f32 * 0.5 - 1.0).collect();
            let b: Vec<f32> = (0..11).map(|i| (11 - i) as f32 * 0.25).collect();
            let got = tiled_dot::<f32, Scalar, Unaligned, 2>(&v(&a), &v(&b)).unwrap();
            let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!(approx(got, want), "dot {got} vs {want}");
        }

        #[test]
        fn tiled_gemv_matches_scalar() {
            // 5 rows (TILE_M 2 → two blocks + 1 cleanup row), 6 cols (lane body 4
            // + 2-col masked tail).
            let (nrows, ncols) = (5, 6);
            let a: Vec<f32> = (0..nrows * ncols).map(|i| (i % 9) as f32 - 4.0).collect();
            let x: Vec<f32> = (0..ncols).map(|i| i as f32 + 1.0).collect();
            let mut y = vec![1.0f32; nrows];
            tiled_gemv::<f32, Scalar, Unaligned, 2>(&v(&a), &v(&x), &mut y, nrows, ncols).unwrap();
            for r in 0..nrows {
                let want = 1.0 + (0..ncols).map(|c| a[r * ncols + c] * x[c]).sum::<f32>();
                assert!(approx(y[r], want), "gemv row {r}: {} vs {want}", y[r]);
            }
        }

        #[test]
        fn gemv_transpose_matches_scalar() {
            let (nrows, ncols) = (4, 6);
            let a: Vec<f32> = (0..nrows * ncols).map(|i| (i % 7) as f32 - 3.0).collect();
            let x: Vec<f32> = (0..nrows).map(|i| i as f32 + 1.0).collect();
            let mut y = vec![0.5f32; ncols];
            <TilingPolicy<2, 1> as TilingStrategy<f32, Scalar, Unaligned>>::gemv_transpose(
                &v(&a),
                &v(&x),
                &mut y,
                nrows,
                ncols,
            )
            .unwrap();
            for c in 0..ncols {
                let want = 0.5 + (0..nrows).map(|r| a[r * ncols + c] * x[r]).sum::<f32>();
                assert!(approx(y[c], want), "gemvT col {c}: {} vs {want}", y[c]);
            }
        }

        #[test]
        fn tiled_gemm_matches_scalar() {
            // 3x5 · 5x6 → 3x6, exercising the register tile plus the column tail.
            let (m, k, n) = (3usize, 5usize, 6usize);
            let a: Vec<f32> = (0..m * k).map(|i| (i % 5) as f32 - 2.0).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i % 4) as f32 - 1.0).collect();
            let mut c = vec![0.25f32; m * n];
            tiled_gemm::<f32, Scalar, Unaligned, 2, 1>(&v(&a), &v(&b), &mut c, m, n, k).unwrap();
            for i in 0..m {
                for j in 0..n {
                    let want = 0.25 + (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum::<f32>();
                    assert!(
                        approx(c[i * n + j], want),
                        "gemm ({i},{j}): {} vs {want}",
                        c[i * n + j]
                    );
                }
            }
        }
    }
}

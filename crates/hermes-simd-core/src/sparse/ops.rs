//! Elementwise operations and value sum/accumulate helpers.
//!
//! # Safety
//!
//! Every kernel call below is `#[target_feature]`-gated and is therefore sound
//! only on a host implementing `Arch`. That holds by construction rather than by
//! inspection: [`SimdView::new`](crate::view::SimdView::new) returns `None` for
//! an architecture the host cannot execute, and the sparse and copy-on-write
//! constructors assert the same condition, so possessing one of these
//! arch-parameterized values *is* the proof. Per-site `SAFETY` comments record
//! only the obligations that go beyond it — pointer provenance, bounds, and
//! alignment.

use super::types::SparseValidate;
use super::{BlockedCoo, Csr, DenseWithMask, SellP, SparseView};
use crate::arch::SimdArch;
use crate::kernel::{
    SimdArith, SimdCompare, SimdGather, SimdLoadStore, SimdMask, SimdReduce, SimdStorage,
};
use crate::scalar::Scalar;
use crate::sparse::spmv::build_index_vector;

/// Unified trait for elementwise and reduction operations on sparse matrices.
pub trait SparseOps<T> {
    /// Compute the sum of all elements stored in the sparse matrix.
    fn sum_values(&self) -> T;

    /// Elementwise multiply the sparse matrix values by corresponding entries
    /// in a dense matrix, writing the results to `out_values`.
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]);
}

impl<T, Arch> SparseOps<T> for SparseView<'_, T, Csr, Arch>
where
    T: Scalar,
    Arch: SimdArch
        + SimdLoadStore<T>
        + SimdArith<T>
        + SimdCompare<T>
        + SimdMask<T>
        + SimdReduce<T>
        + SimdGather<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        if let Some(view) =
            crate::view::SimdView::<T, Arch, crate::align::Unaligned>::new(self.data.values)
        {
            view.reduce(crate::ops::Sum)
        } else {
            T::ZERO
        }
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let lane_count = <Arch as SimdStorage<T>>::LANE_COUNT;

        // SOUNDNESS: the SIMD path below gathers `dense[col_indices[j]]` with an
        // unchecked `Arch::gather`. `SparseView<Csr>` is the *unvalidated* type
        // (constructible from arbitrary `CsrData` via `from_csr`), so nothing
        // otherwise guarantees `col_indices[j] < dense.len()` and the gather
        // could read out of bounds from safe code. Validate the structure
        // (`col_indices[k] < ncols`) via the SSOT checker and require
        // `dense.len() >= ncols`, so every gathered index is in bounds. O(nnz)
        // once per call, matching the SELL-p path in this file.
        d.validate()
            .expect("CSR matrix failed structural validation before elementwise_mul_dense");
        assert!(
            dense.len() >= d.ncols,
            "CSR elementwise_mul_dense: dense len {} < ncols {}",
            dense.len(),
            d.ncols
        );

        for r in 0..d.nrows {
            let start = d.row_ptr[r] as usize;
            let end = d.row_ptr[r + 1] as usize;
            let row_nnz = end - start;
            let vals = &d.values[start..end];
            let cols = &d.col_indices[start..end];
            let out = &mut out_values[start..end];

            let simd_len = (row_nnz / lane_count) * lane_count;
            // SAFETY: `Arch::*` are target-feature kernels (module invariant).
            // The window `[j, j+LANE_COUNT)` stays within `vals`/`cols`/`out` for
            // `j < simd_len <= row_nnz`, and `validate` above proved every
            // `cols[k] < ncols <= dense.len()`, so each gathered `dense[cols[k]]`
            // is in bounds.
            let mut j = 0usize;
            unsafe {
                while j < simd_len {
                    let idx = build_index_vector::<T, Arch>(&cols[j..j + lane_count]);
                    let res_vec = <Arch as SimdArith<T>>::mul(
                        <Arch as SimdLoadStore<T>>::load_unaligned(vals[j..].as_ptr()),
                        <Arch as SimdGather<T>>::gather(dense.as_ptr(), idx),
                    );
                    <Arch as SimdLoadStore<T>>::store_unaligned(out[j..].as_mut_ptr(), res_vec);
                    j += lane_count;
                }
            }
            while j < row_nnz {
                let c = cols[j] as usize;
                out[j] = vals[j] * dense[c];
                j += 1;
            }
        }
    }
}

impl<T, const C: usize, Arch> SparseOps<T> for SparseView<'_, T, SellP<C>, Arch>
where
    T: Scalar,
    Arch: SimdArch
        + SimdLoadStore<T>
        + SimdArith<T>
        + SimdCompare<T>
        + SimdMask<T>
        + SimdReduce<T>
        + SimdGather<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        if let Some(view) =
            crate::view::SimdView::<T, Arch, crate::align::Unaligned>::new(self.data.values)
        {
            view.reduce(crate::ops::Sum)
        } else {
            T::ZERO
        }
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let nslices = d.nslices();
        let lane_count = <Arch as SimdStorage<T>>::LANE_COUNT;

        if lane_count == C {
            // SOUNDNESS: the vectorized path loads `values[offset..]` and stores
            // `out_values[offset..]` as full `C`-lane vectors. Validate SELL-p
            // slice geometry via the SSOT checker (bounds `offset + C <=
            // values.len()`) and require the output to be at least as long as the
            // values array, so both unchecked accesses stay in bounds even for a
            // caller-constructed matrix with `pub` fields.
            use super::types::SparseValidate;
            d.validate()
                .expect("SELL-p matrix failed structural validation before vectorized kernel");
            assert!(
                out_values.len() >= d.values.len(),
                "SELL-p elementwise_mul_dense: out_values len {} < values len {}",
                out_values.len(),
                d.values.len()
            );
            for s in 0..nslices {
                let col_count = d.slice_col_count[s] as usize;
                let start_offset = d.slice_ptr[s] as usize;
                let slice_base_r = s * C;

                for col in 0..col_count {
                    let offset = start_offset + col * C;

                    let mut idx_arr = [0i32; C];
                    let mut mask_arr = [false; C];
                    for row in 0..C {
                        let r = slice_base_r + row;
                        let c = d.col_indices[offset + row] as usize;
                        let in_bounds = r < d.nrows && c < d.ncols;
                        mask_arr[row] = in_bounds;
                        if in_bounds {
                            idx_arr[row] = (r * d.ncols + c) as i32;
                        }
                    }

                    // SAFETY: `Arch::*` are target-feature kernels (module
                    // invariant). `idx_arr`/`mask_arr` hold `C == LANE_COUNT`
                    // valid entries; `mask` is set only where `r < nrows && c <
                    // ncols`, so the masked gather touches `dense` only at those
                    // computed in-bounds indices. `validate` and the output-length
                    // assert above keep `values[offset..offset+C]` and the masked
                    // store into `out_values[offset..]` in bounds.
                    unsafe {
                        let idx = build_index_vector::<T, Arch>(&idx_arr[..C]);
                        let mask = Arch::mask_from_bools(&mask_arr[..C]);
                        let zero_vec = Arch::zero();
                        let dense_vec = Arch::gather_masked(dense.as_ptr(), idx, mask, zero_vec);
                        let res_vec =
                            Arch::mul(Arch::load_unaligned(d.values[offset..].as_ptr()), dense_vec);
                        Arch::masked_store_unaligned(
                            out_values[offset..].as_mut_ptr(),
                            mask,
                            res_vec,
                        );
                    }
                }
            }
        } else {
            for s in 0..nslices {
                let col_count = d.slice_col_count[s] as usize;
                let start_offset = d.slice_ptr[s] as usize;
                for col in 0..col_count {
                    for row in 0..C {
                        let idx = start_offset + col * C + row;
                        let c = d.col_indices[idx] as usize;
                        let r = s * C + row;
                        if r < d.nrows && c < d.ncols {
                            out_values[idx] = d.values[idx] * dense[r * d.ncols + c];
                        }
                    }
                }
            }
        }
    }
}

impl<T, const BM: usize, const BN: usize, Arch> SparseOps<T>
    for SparseView<'_, T, BlockedCoo<BM, BN>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdCompare<T> + SimdMask<T> + SimdReduce<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        if let Some(view) =
            crate::view::SimdView::<T, Arch, crate::align::Unaligned>::new(self.data.blocks)
        {
            view.reduce(crate::ops::Sum)
        } else {
            T::ZERO
        }
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let lane_count = Arch::LANE_COUNT;

        // Bounds for the unchecked SIMD loads/stores below: the dense matrix and
        // the block/output buffers must be large enough, and every block must lie
        // within the `nrows x ncols` dense extent so each `dense[(br+i)*ncols+bc
        // .. +BN]` read stays in bounds. O(nblocks), once per call.
        let block_elems = d.nblocks * BM * BN;
        assert!(
            dense.len() >= d.nrows * d.ncols,
            "dense buffer {} too small for {}x{}",
            dense.len(),
            d.nrows,
            d.ncols
        );
        assert!(
            out_values.len() >= block_elems && d.blocks.len() >= block_elems,
            "block/output buffers too small for {block_elems} block elements"
        );
        for b in 0..d.nblocks {
            let br = d.block_row[b] as usize;
            let bc = d.block_col[b] as usize;
            assert!(
                bc + BN <= d.ncols && br + BM <= d.nrows,
                "BlockedCoo block {b} (row {br}+{BM}, col {bc}+{BN}) exceeds {}x{}",
                d.nrows,
                d.ncols
            );
        }

        if BN == lane_count {
            // SAFETY: `Arch::*` are target-feature kernels (module invariant).
            // The asserts above bound each block within the dense extent and the
            // block/output buffers, so every `LANE_COUNT`-wide load of a block
            // row `blocks[offset..offset+BN]`, the dense window
            // `dense[(br+i)*ncols+bc ..][..BN]`, and the matching store into
            // `out_values` stays in bounds.
            unsafe {
                for b in 0..d.nblocks {
                    let br = d.block_row[b] as usize;
                    let bc = d.block_col[b] as usize;
                    for i in 0..BM {
                        let offset = b * (BM * BN) + i * BN;
                        let dense_idx = (br + i) * d.ncols + bc;
                        let res_vec = Arch::mul(
                            Arch::load_unaligned(d.blocks[offset..].as_ptr()),
                            Arch::load_unaligned(dense[dense_idx..].as_ptr()),
                        );
                        Arch::store_unaligned(out_values[offset..].as_mut_ptr(), res_vec);
                    }
                }
            }
        } else if BN == lane_count * 2 {
            // SAFETY: as the `BN == LANE_COUNT` arm, with each block row and its
            // dense window spanning two `LANE_COUNT` loads; both halves stay
            // within `blocks`/`dense`/`out_values` by the same block-extent and
            // buffer-length asserts.
            unsafe {
                for b in 0..d.nblocks {
                    let br = d.block_row[b] as usize;
                    let bc = d.block_col[b] as usize;
                    for i in 0..BM {
                        let offset = b * (BM * BN) + i * BN;
                        let dense_idx = (br + i) * d.ncols + bc;
                        let res_vec0 = Arch::mul(
                            Arch::load_unaligned(d.blocks[offset..].as_ptr()),
                            Arch::load_unaligned(dense[dense_idx..].as_ptr()),
                        );
                        let res_vec1 = Arch::mul(
                            Arch::load_unaligned(d.blocks[offset + lane_count..].as_ptr()),
                            Arch::load_unaligned(dense[dense_idx + lane_count..].as_ptr()),
                        );
                        Arch::store_unaligned(out_values[offset..].as_mut_ptr(), res_vec0);
                        Arch::store_unaligned(
                            out_values[offset + lane_count..].as_mut_ptr(),
                            res_vec1,
                        );
                    }
                }
            }
        } else {
            for b in 0..d.nblocks {
                let br = d.block_row[b] as usize;
                let bc = d.block_col[b] as usize;
                for i in 0..BM {
                    for j in 0..BN {
                        let idx = b * (BM * BN) + i * BN + j;
                        out_values[idx] = d.blocks[idx] * dense[(br + i) * d.ncols + (bc + j)];
                    }
                }
            }
        }
    }
}

impl<T, Arch> SparseOps<T> for SparseView<'_, T, DenseWithMask, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdMask<T> + SimdReduce<T>,
{
    #[inline]
    fn sum_values(&self) -> T {
        let lane_count = Arch::LANE_COUNT;
        let len = self.data.values.len();
        let simd_len = (len / lane_count) * lane_count;
        assert!(
            self.data.mask.len() >= len,
            "DenseWithMask sum_values: mask covers {} elements, values has {}",
            self.data.mask.len(),
            len
        );

        // SAFETY: `Arch::*` are target-feature kernels (module invariant). Every
        // masked load reads `values[i..i+LANE_COUNT]` for `i < simd_len <= len`,
        // which stays within `values`; `mask.lane_bits` is a safe,
        // self-bounds-checked packed read feeding `mask_from_bitmask`.
        let mut i = 0usize;
        let acc_vec = unsafe {
            let zero_vec = Arch::zero();
            let mut acc_vec = zero_vec;
            while i < simd_len {
                let msk = Arch::mask_from_bitmask(self.data.mask.lane_bits(i, lane_count));
                let v_vec =
                    Arch::masked_load_unaligned(self.data.values[i..].as_ptr(), msk, zero_vec);
                acc_vec = Arch::add(acc_vec, v_vec);
                i += lane_count;
            }
            acc_vec
        };
        // SAFETY: target-feature kernel, covered by the module invariant.
        let mut s = unsafe { Arch::sum_reduce(acc_vec) };
        while i < len {
            if self.data.mask.bit(i) {
                s += self.data.values[i];
            }
            i += 1;
        }
        s
    }

    #[inline]
    fn elementwise_mul_dense(&self, dense: &[T], out_values: &mut [T]) {
        let d = &self.data;
        let len = d.values.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;

        // Bounds for the unchecked loads/stores: a `LANE_COUNT` window at
        // `i < simd_len <= len` must stay within `dense` and `out_values` as
        // well as `values`, and the packed mask must cover every element.
        // `dense` and the output are elementwise-shaped, so require them at
        // least as long as `values`.
        assert!(
            dense.len() >= len && out_values.len() >= len && d.mask.len() >= len,
            "DenseWithMask elementwise_mul_dense: dense {} / out {} / mask {} shorter than values {}",
            dense.len(),
            out_values.len(),
            d.mask.len(),
            len
        );

        // SAFETY: `Arch::*` are target-feature kernels (module invariant). The
        // assert above gives every windowed load/store `[i, i+LANE_COUNT)` room
        // within `dense`, `out_values`, and `values` for `i < simd_len`;
        // `mask.lane_bits` is a safe, self-bounds-checked packed read.
        let mut i = 0usize;
        unsafe {
            let zero_vec = Arch::zero();
            while i < simd_len {
                let msk = Arch::mask_from_bitmask(d.mask.lane_bits(i, lane_count));
                let res_vec = Arch::masked_mul(
                    Arch::load_unaligned(d.values[i..].as_ptr()),
                    Arch::load_unaligned(dense[i..].as_ptr()),
                    msk,
                    zero_vec,
                );
                Arch::store_unaligned(out_values[i..].as_mut_ptr(), res_vec);
                i += lane_count;
            }
        }
        while i < len {
            if d.mask.bit(i) {
                out_values[i] = d.values[i] * dense[i];
            } else {
                out_values[i] = T::ZERO;
            }
            i += 1;
        }
    }
}

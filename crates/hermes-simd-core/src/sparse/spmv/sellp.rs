//! Sliced ELLPACK multiplication.

use super::{build_index_vector, validate_spmv_sizes, SparseSpMv};
use crate::arch::SimdArch;
use crate::kernel::{SimdArith, SimdGather, SimdLoadStore};
use crate::scalar::Scalar;
use crate::sparse::{SellP, SellPData, SparseView, Validated};

fn spmv_scalar<T, const C: usize>(data: &SellPData<'_, T, C>, x: &[T], y: &mut [T])
where
    T: Scalar,
{
    let nslices = data.nslices();
    for slice in 0..nslices {
        let col_count = data.slice_col_count[slice] as usize;
        let start_offset = data.slice_ptr[slice] as usize;
        let mut row_acc = [T::ZERO; C];

        for col in 0..col_count {
            for row in 0..C {
                let idx = start_offset + col * C + row;
                let col_idx = data.col_indices[idx] as usize;
                // SAFETY: Validated<SellP> proves this column lies inside x.
                row_acc[row] += data.values[idx] * unsafe { *x.get_unchecked(col_idx) };
            }
        }

        for row in 0..C {
            let row_idx = slice * C + row;
            if row_idx < y.len() {
                y[row_idx] += row_acc[row];
            }
        }
    }
}

/// Vectorized SELL-p multiplication when `Arch::LANE_COUNT == C`.
///
/// # Safety
/// The host must implement `Arch`. `data` must be validated so all gathered
/// columns and slice windows are in bounds, and `x`/`y` must satisfy its shape.
unsafe fn spmv_vectorized<T, const C: usize, Arch>(data: &SellPData<'_, T, C>, x: &[T], y: &mut [T])
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T>,
{
    assert_eq!(
        Arch::LANE_COUNT,
        C,
        "spmv_vectorized requires Arch::LANE_COUNT == C"
    );
    for slice in 0..data.nslices() {
        let col_count = data.slice_col_count[slice] as usize;
        let start_offset = data.slice_ptr[slice] as usize;
        let mut acc0 = Arch::zero();
        let mut acc1 = Arch::zero();
        let mut acc2 = Arch::zero();
        let mut acc3 = Arch::zero();

        let unroll = (col_count / 4) * 4;
        let mut col = 0;
        while col < unroll {
            let offset = start_offset + col * C;
            // SAFETY: the caller proves the vector-width value and index windows.
            let value_vec = unsafe { Arch::load_unaligned(data.values[offset..].as_ptr()) };
            let index_vec =
                unsafe { build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]) };
            let x_vec = unsafe { Arch::gather(x.as_ptr(), index_vec) };
            acc0 = unsafe { Arch::fmadd(value_vec, x_vec, acc0) };

            let offset = start_offset + (col + 1) * C;
            let value_vec = unsafe { Arch::load_unaligned(data.values[offset..].as_ptr()) };
            let index_vec =
                unsafe { build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]) };
            let x_vec = unsafe { Arch::gather(x.as_ptr(), index_vec) };
            acc1 = unsafe { Arch::fmadd(value_vec, x_vec, acc1) };

            let offset = start_offset + (col + 2) * C;
            let value_vec = unsafe { Arch::load_unaligned(data.values[offset..].as_ptr()) };
            let index_vec =
                unsafe { build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]) };
            let x_vec = unsafe { Arch::gather(x.as_ptr(), index_vec) };
            acc2 = unsafe { Arch::fmadd(value_vec, x_vec, acc2) };

            let offset = start_offset + (col + 3) * C;
            let value_vec = unsafe { Arch::load_unaligned(data.values[offset..].as_ptr()) };
            let index_vec =
                unsafe { build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]) };
            let x_vec = unsafe { Arch::gather(x.as_ptr(), index_vec) };
            acc3 = unsafe { Arch::fmadd(value_vec, x_vec, acc3) };
            col += 4;
        }

        let mut acc = unsafe { Arch::add(Arch::add(acc0, acc1), Arch::add(acc2, acc3)) };
        while col < col_count {
            let offset = start_offset + col * C;
            // SAFETY: the caller proves this complete vector window and gather.
            let value_vec = unsafe { Arch::load_unaligned(data.values[offset..].as_ptr()) };
            let index_vec =
                unsafe { build_index_vector::<T, Arch>(&data.col_indices[offset..offset + C]) };
            let x_vec = unsafe { Arch::gather(x.as_ptr(), index_vec) };
            acc = unsafe { Arch::fmadd(value_vec, x_vec, acc) };
            col += 1;
        }

        let row_idx = slice * C;
        if row_idx + C <= y.len() {
            // SAFETY: the complete C-lane destination lies inside y.
            unsafe {
                let y_ptr = y.as_mut_ptr().add(row_idx);
                let y_vec = Arch::load_unaligned(y_ptr);
                Arch::store_unaligned(y_ptr, Arch::add(y_vec, acc));
            }
        } else {
            let mut temp = [T::ZERO; C];
            // SAFETY: temp has exactly C == LANE_COUNT elements.
            unsafe { Arch::store_unaligned(temp.as_mut_ptr(), acc) };
            for row in 0..y.len() - row_idx {
                y[row_idx + row] += temp[row];
            }
        }
    }
}

impl<T, const C: usize, Arch> SparseSpMv<T> for SparseView<'_, T, Validated<SellP<C>>, Arch>
where
    T: Scalar,
    Arch: SimdArch + SimdLoadStore<T> + SimdArith<T> + SimdGather<T>,
{
    #[inline]
    fn spmv(&self, x: &[T], y: &mut [T]) {
        let data = self.data.storage();
        validate_spmv_sizes(x.len(), y.len(), data.ncols, data.nrows, "SellP");

        if Arch::LANE_COUNT == C {
            // SAFETY: validated storage proves every gather and slice window;
            // the entry checks establish x/y shape compatibility.
            unsafe { spmv_vectorized::<T, C, Arch>(data, x, y) };
        } else {
            spmv_scalar::<T, C>(data, x, y);
        }
    }
}

//! Zero-copy, const-generic N-dimensional strided tensor view.
//!
//! # Design
//!
//! `TensorView<'a, T, const N: usize>` is a rank-`N` view over a borrowed slice.
//! Shape and strides are `[usize; N]` arrays resolved at compile time — the const
//! generic `N` is erased after monomorphization, leaving no runtime overhead vs.
//! a hand-written 2-D or 3-D struct.
//!
//! # Layout Markers
//!
//! Two zero-sized layout markers tag contiguous storage assumptions:
//! - [`RowMajor`] — row-major (C-order) storage; `strides[i] = ∏_{j>i} shape[j]`.
//! - [`ColMajor`] — column-major (Fortran-order) storage.
//!
//! # Zero-Copy Contract
//!
//! - `new(data, shape)` — zero allocation; computes row-major strides from shape.
//! - `with_strides(data, shape, strides)` — zero allocation; caller supplies strides.
//! - `row_view(i)` — returns a `TensorView<'_, T, {N-1}>` sharing the same slice.
//! - `reshape(new_shape)` — returns a new view if the layout is contiguous; no copy.
//! - All `get` / `iter_rows` operations are also zero-copy.
//!
//! # Module Structure
//!
//! | Leaf module       | Contents                                             |
//! |-------------------|------------------------------------------------------|
//! | [`layout`]        | `RowMajor`, `ColMajor` ZSTs; sealed `Layout` trait   |
//! | [`error`]         | `TensorError` enum                                   |
//! | [`view`]          | `TensorView` core + `rank_ops`/`simd_bridge` leaves  |
//! | [`cow`]           | `TensorCow` enum + all impl blocks                   |
//! | `helpers` (priv)  | `row_major_strides`, `compute_offset`                |

pub mod cow;
pub mod error;
mod helpers;
pub mod layout;
pub mod view;

pub use cow::TensorCow;
pub use error::TensorError;
pub use layout::{ColMajor, Layout, RowMajor};
pub use view::TensorView;

#[cfg(test)]
mod tests {
    use super::*;
    use helpers::row_major_strides;

    #[test]
    fn test_row_major_strides_3d() {
        let s = row_major_strides([2usize, 3, 4]);
        assert_eq!(s, [12, 4, 1]);
    }

    #[test]
    fn test_tensor_view_get_2d() {
        let data: Vec<i32> = (0..12).collect();
        let t = TensorView::<i32, 2>::new(&data, [3, 4]).unwrap();
        assert_eq!(t.get([1, 2]).unwrap(), 6);
    }

    #[test]
    fn test_reshape() {
        let data: Vec<i32> = (0..12).collect();
        let t2d = TensorView::<i32, 2>::new(&data, [3, 4]).unwrap();
        let t1d = t2d.reshape([12]).unwrap();
        assert_eq!(t1d.num_elements(), 12);
        assert_eq!(t1d.get([11]).unwrap(), 11);
    }

    #[test]
    fn test_row_view() {
        let data: Vec<f32> = (0..9).map(|x| x as f32).collect();
        let t = TensorView::<f32, 2>::new(&data, [3, 3]).unwrap();
        let row1 = t.row_view(1).unwrap();
        assert_eq!(row1.num_elements(), 3);
        assert_eq!(row1.get([0]).unwrap(), 3.0);
        assert_eq!(row1.get([2]).unwrap(), 5.0);
    }
}

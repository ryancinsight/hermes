//! Generic scalar fallback implementation of `TileMatrixMultiply`.

use crate::Scalar;
use hermes_simd_core::scalar::NumericElement;
use hermes_simd_core::view::TileMatrixMultiply;

impl<Backend, Arch, const M: usize, const N: usize, const K: usize>
    TileMatrixMultiply<i8, i8, i32, Backend, Arch, M, N, K> for Scalar
{
    #[inline]
    unsafe fn tile_matmul(
        c: *mut i32,
        c_stride: usize,
        a: *const i8,
        a_stride: usize,
        b: *const i8,
        b_stride: usize,
    ) {
        for i in 0..M {
            for j in 0..N {
                let mut sum = *c.add(i * c_stride + j);
                for k in 0..K {
                    let val_a = i32::from(*a.add(i * a_stride + k));
                    let val_b = i32::from(*b.add(k * b_stride + j));
                    sum = sum.wrapping_add(val_a * val_b);
                }
                *c.add(i * c_stride + j) = sum;
            }
        }
    }
}

impl<Backend, Arch, const M: usize, const N: usize, const K: usize>
    TileMatrixMultiply<eunomia::Bf16, eunomia::Bf16, eunomia::F32, Backend, Arch, M, N, K>
    for Scalar
{
    #[inline]
    unsafe fn tile_matmul(
        c: *mut eunomia::F32,
        c_stride: usize,
        a: *const eunomia::Bf16,
        a_stride: usize,
        b: *const eunomia::Bf16,
        b_stride: usize,
    ) {
        for i in 0..M {
            for j in 0..N {
                let mut sum = (*c.add(i * c_stride + j)).0;
                for k in 0..K {
                    let val_a = eunomia::FloatElement::to_f32(*a.add(i * a_stride + k));
                    let val_b = eunomia::FloatElement::to_f32(*b.add(k * b_stride + j));
                    sum = val_a.scalar_fmadd(val_b, sum);
                }
                *c.add(i * c_stride + j) = eunomia::F32(sum);
            }
        }
    }
}

impl<Backend, Arch, const M: usize, const N: usize, const K: usize>
    TileMatrixMultiply<eunomia::I8, eunomia::I8, eunomia::I32, Backend, Arch, M, N, K> for Scalar
{
    #[inline]
    unsafe fn tile_matmul(
        c: *mut eunomia::I32,
        c_stride: usize,
        a: *const eunomia::I8,
        a_stride: usize,
        b: *const eunomia::I8,
        b_stride: usize,
    ) {
        for i in 0..M {
            for j in 0..N {
                let mut sum = (*c.add(i * c_stride + j)).0;
                for k in 0..K {
                    let val_a = i32::from((*a.add(i * a_stride + k)).0);
                    let val_b = i32::from((*b.add(k * b_stride + j)).0);
                    sum = sum.wrapping_add(val_a * val_b);
                }
                *c.add(i * c_stride + j) = eunomia::I32(sum);
            }
        }
    }
}

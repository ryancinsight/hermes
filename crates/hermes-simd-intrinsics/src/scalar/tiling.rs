//! Generic scalar fallback implementation of TileMatrixMultiply.

use crate::Scalar;
use hermes_simd_core::scalar::NumericElement;
use hermes_simd_core::view::TileMatrixMultiply;

impl<Backend, Arch, const M: usize, const N: usize, const K: usize>
    TileMatrixMultiply<half::bf16, half::bf16, f32, Backend, Arch, M, N, K> for Scalar
{
    #[inline]
    unsafe fn tile_matmul(
        c: *mut f32,
        c_stride: usize,
        a: *const half::bf16,
        a_stride: usize,
        b: *const half::bf16,
        b_stride: usize,
    ) {
        for i in 0..M {
            for j in 0..N {
                let mut sum = *c.add(i * c_stride + j);
                for k in 0..K {
                    let val_a = (*a.add(i * a_stride + k)).to_f32();
                    let val_b = (*b.add(k * b_stride + j)).to_f32();
                    sum = val_a.scalar_fmadd(val_b, sum);
                }
                *c.add(i * c_stride + j) = sum;
            }
        }
    }
}

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
                    let val_a = *a.add(i * a_stride + k) as i32;
                    let val_b = *b.add(k * b_stride + j) as i32;
                    sum = sum.wrapping_add(val_a * val_b);
                }
                *c.add(i * c_stride + j) = sum;
            }
        }
    }
}

impl<Backend, Arch, const M: usize, const N: usize, const K: usize>
    TileMatrixMultiply<
        hermes_numeric::Bf16,
        hermes_numeric::Bf16,
        hermes_numeric::F32,
        Backend,
        Arch,
        M,
        N,
        K,
    > for Scalar
{
    #[inline]
    unsafe fn tile_matmul(
        c: *mut hermes_numeric::F32,
        c_stride: usize,
        a: *const hermes_numeric::Bf16,
        a_stride: usize,
        b: *const hermes_numeric::Bf16,
        b_stride: usize,
    ) {
        for i in 0..M {
            for j in 0..N {
                let mut sum = (*c.add(i * c_stride + j)).0;
                for k in 0..K {
                    let val_a = (*a.add(i * a_stride + k)).0.to_f32();
                    let val_b = (*b.add(k * b_stride + j)).0.to_f32();
                    sum = val_a.scalar_fmadd(val_b, sum);
                }
                *c.add(i * c_stride + j) = hermes_numeric::F32(sum);
            }
        }
    }
}

impl<Backend, Arch, const M: usize, const N: usize, const K: usize>
    TileMatrixMultiply<
        hermes_numeric::I8,
        hermes_numeric::I8,
        hermes_numeric::I32,
        Backend,
        Arch,
        M,
        N,
        K,
    > for Scalar
{
    #[inline]
    unsafe fn tile_matmul(
        c: *mut hermes_numeric::I32,
        c_stride: usize,
        a: *const hermes_numeric::I8,
        a_stride: usize,
        b: *const hermes_numeric::I8,
        b_stride: usize,
    ) {
        for i in 0..M {
            for j in 0..N {
                let mut sum = (*c.add(i * c_stride + j)).0;
                for k in 0..K {
                    let val_a = (*a.add(i * a_stride + k)).0 as i32;
                    let val_b = (*b.add(k * b_stride + j)).0 as i32;
                    sum = sum.wrapping_add(val_a * val_b);
                }
                *c.add(i * c_stride + j) = hermes_numeric::I32(sum);
            }
        }
    }
}

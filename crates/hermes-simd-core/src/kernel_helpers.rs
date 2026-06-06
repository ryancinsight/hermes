#![allow(dead_code)]

use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

#[inline(always)]
pub unsafe fn generic_binary_op<T, Arch: ?Sized, F>(a: Arch::Vector, b: Arch::Vector, mut op: F) -> Arch::Vector
where
    T: Scalar,
    Arch: SimdKernel<T>,
    F: FnMut(T, T) -> T,
{
    let mut buf_a = [T::ZERO; 128];
    let mut buf_b = [T::ZERO; 128];
    Arch::store_unaligned(buf_a.as_mut_ptr(), a);
    Arch::store_unaligned(buf_b.as_mut_ptr(), b);
    for i in 0..Arch::LANE_COUNT {
        buf_a[i] = op(buf_a[i], buf_b[i]);
    }
    Arch::load_unaligned(buf_a.as_ptr())
}

#[inline(always)]
pub unsafe fn generic_unary_op<T, Arch: ?Sized, F>(a: Arch::Vector, mut op: F) -> Arch::Vector
where
    T: Scalar,
    Arch: SimdKernel<T>,
    F: FnMut(T) -> T,
{
    let mut buf = [T::ZERO; 128];
    Arch::store_unaligned(buf.as_mut_ptr(), a);
    for i in 0..Arch::LANE_COUNT {
        buf[i] = op(buf[i]);
    }
    Arch::load_unaligned(buf.as_ptr())
}

#[inline(always)]
pub unsafe fn generic_blend<T, Arch: ?Sized>(
    mask: Arch::Vector,
    true_val: Arch::Vector,
    false_val: Arch::Vector,
) -> Arch::Vector
where
    T: Scalar,
    Arch: SimdKernel<T>,
{
    let mut buf_mask = [T::ZERO; 128];
    let mut buf_true = [T::ZERO; 128];
    let mut buf_false = [T::ZERO; 128];
    Arch::store_unaligned(buf_mask.as_mut_ptr(), mask);
    Arch::store_unaligned(buf_true.as_mut_ptr(), true_val);
    Arch::store_unaligned(buf_false.as_mut_ptr(), false_val);
    for i in 0..Arch::LANE_COUNT {
        let is_true = buf_mask[i].is_nan() || buf_mask[i].to_f64() != 0.0;
        buf_true[i] = if is_true { buf_true[i] } else { buf_false[i] };
    }
    Arch::load_unaligned(buf_true.as_ptr())
}

#[inline(always)]
pub unsafe fn generic_mask_from_bitmask<T, Arch: ?Sized>(bm: u64) -> Arch::Mask
where
    T: Scalar,
    Arch: SimdKernel<T>,
{
    let mut bools = [false; 64];
    for i in 0..Arch::LANE_COUNT {
        bools[i] = (bm >> i) & 1 == 1;
    }
    Arch::mask_from_bools(&bools[..Arch::LANE_COUNT])
}

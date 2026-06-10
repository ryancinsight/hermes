#![allow(dead_code)]

use crate::kernel::SimdKernel;
use crate::scalar::Scalar;

#[inline(always)]
pub unsafe fn generic_binary_op<T, Arch, F>(a: Arch::Vector, b: Arch::Vector, mut op: F) -> Arch::Vector
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
pub unsafe fn generic_unary_op<T, Arch, F>(a: Arch::Vector, mut op: F) -> Arch::Vector
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
pub unsafe fn generic_blend<T, Arch>(
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
pub unsafe fn generic_mask_from_bitmask<T, Arch>(bm: u64) -> Arch::Mask
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

/// Scalar lane-by-lane alternating FMA used by the `fmaddsub` / `fmsubadd` defaults.
///
/// Even lanes compute `a*b - c` and odd lanes `a*b + c` when `ADD_EVEN == false`
/// (`fmaddsub` semantics); the signs flip per lane parity when `ADD_EVEN == true`
/// (`fmsubadd` semantics).
#[inline(always)]
pub unsafe fn generic_alternating_fma<T, Arch, const ADD_EVEN: bool>(
    a: Arch::Vector,
    b: Arch::Vector,
    c: Arch::Vector,
) -> Arch::Vector
where
    T: Scalar,
    Arch: SimdKernel<T>,
{
    let mut buf_a = [T::ZERO; 128];
    let mut buf_b = [T::ZERO; 128];
    let mut buf_c = [T::ZERO; 128];
    Arch::store_unaligned(buf_a.as_mut_ptr(), a);
    Arch::store_unaligned(buf_b.as_mut_ptr(), b);
    Arch::store_unaligned(buf_c.as_mut_ptr(), c);
    for i in 0..Arch::LANE_COUNT {
        let prod = buf_a[i] * buf_b[i];
        let add = (i & 1 == 1) ^ ADD_EVEN;
        buf_a[i] = if add {
            prod + buf_c[i]
        } else {
            prod - buf_c[i]
        };
    }
    Arch::load_unaligned(buf_a.as_ptr())
}

/// Scalar lane-by-lane horizontal fold used by `min_reduce` and `max_reduce` defaults.
///
/// Stores the vector to a stack buffer, then folds with `op` starting from `identity`.
#[inline(always)]
pub unsafe fn generic_horizontal_reduce<T, Arch>(
    v: Arch::Vector,
    identity: T,
    mut op: impl FnMut(T, T) -> T,
) -> T
where
    T: Scalar,
    Arch: SimdKernel<T>,
{
    let mut buf = [T::ZERO; 128];
    Arch::store_unaligned(buf.as_mut_ptr(), v);
    let mut acc = identity;
    for i in 0..Arch::LANE_COUNT {
        acc = op(acc, buf[i]);
    }
    acc
}

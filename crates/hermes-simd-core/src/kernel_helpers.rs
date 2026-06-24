#![allow(dead_code)]

use crate::kernel::{SimdKernel, MAX_SIMD_LANES};
use crate::scalar::Scalar;

#[inline(always)]
pub unsafe fn generic_binary_op<T, Arch, F>(
    a: Arch::Vector,
    b: Arch::Vector,
    mut op: F,
) -> Arch::Vector
where
    T: Scalar,
    Arch: SimdKernel<T>,
    F: FnMut(T, T) -> T,
{
    const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
    let mut buf_a = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_b = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf_a.as_mut_ptr() as *mut T, a);
    Arch::store_unaligned(buf_b.as_mut_ptr() as *mut T, b);
    for i in 0..Arch::LANE_COUNT {
        let val_a = buf_a[i].assume_init();
        let val_b = buf_b[i].assume_init();
        buf_a[i].write(op(val_a, val_b));
    }
    Arch::load_unaligned(buf_a.as_ptr() as *const T)
}

#[inline(always)]
pub unsafe fn generic_unary_op<T, Arch, F>(a: Arch::Vector, mut op: F) -> Arch::Vector
where
    T: Scalar,
    Arch: SimdKernel<T>,
    F: FnMut(T) -> T,
{
    const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
    let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf.as_mut_ptr() as *mut T, a);
    for i in 0..Arch::LANE_COUNT {
        let val = buf[i].assume_init();
        buf[i].write(op(val));
    }
    Arch::load_unaligned(buf.as_ptr() as *const T)
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
    const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
    let mut buf_mask = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_true = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_false = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf_mask.as_mut_ptr() as *mut T, mask);
    Arch::store_unaligned(buf_true.as_mut_ptr() as *mut T, true_val);
    Arch::store_unaligned(buf_false.as_mut_ptr() as *mut T, false_val);
    for i in 0..Arch::LANE_COUNT {
        let mask_val = buf_mask[i].assume_init();
        let is_true = mask_val.is_nan() || mask_val.to_f64() != 0.0;
        let val = if is_true {
            buf_true[i].assume_init()
        } else {
            buf_false[i].assume_init()
        };
        buf_true[i].write(val);
    }
    Arch::load_unaligned(buf_true.as_ptr() as *const T)
}

#[inline(always)]
pub unsafe fn generic_mask_from_bitmask<T, Arch>(bm: u64) -> Arch::Mask
where
    T: Scalar,
    Arch: SimdKernel<T>,
{
    // The `u64` bitmask and the 64-element `bools` buffer bound this default to
    // 64 lanes; `bm >> i` for `i >= 64` is itself undefined. Checked at compile
    // time per backend.
    const {
        assert!(
            <Arch as SimdKernel<T>>::LANE_COUNT <= u64::BITS as usize,
            "SimdKernel::LANE_COUNT exceeds the 64-lane u64 bitmask width"
        )
    };
    let mut bools = [false; u64::BITS as usize];
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
    const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
    let mut buf_a = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_b = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_c = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf_a.as_mut_ptr() as *mut T, a);
    Arch::store_unaligned(buf_b.as_mut_ptr() as *mut T, b);
    Arch::store_unaligned(buf_c.as_mut_ptr() as *mut T, c);
    for i in 0..Arch::LANE_COUNT {
        let val_a = buf_a[i].assume_init();
        let val_b = buf_b[i].assume_init();
        let val_c = buf_c[i].assume_init();
        let prod = val_a * val_b;
        let add = (i & 1 == 1) ^ ADD_EVEN;
        buf_a[i].write(if add { prod + val_c } else { prod - val_c });
    }
    Arch::load_unaligned(buf_a.as_ptr() as *const T)
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
    const { <Arch as SimdKernel<T>>::LANE_BOUND_CHECK };
    let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf.as_mut_ptr() as *mut T, v);
    let mut acc = identity;
    for i in 0..Arch::LANE_COUNT {
        let val = buf[i].assume_init();
        acc = op(acc, val);
    }
    acc
}

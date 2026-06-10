//! AVX2 f16 emulated SIMD kernel.
//!
//! AVX2 lacks native f16 vector arithmetic instructions (only float-to-half conversion
//! exists under F16C target feature). This implementation provides a documented software-emulated
//! fallback with 16 lanes matching the register size.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::Avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd_core::kernel::SimdKernel;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl SimdKernel<half::f16> for Avx2 {
    type Vector = [half::f16; 16];
    type Mask = [bool; 16];
    type IndexVector = [i32; 16];
    const LANE_COUNT: usize = 16;
    const UNROLL_FACTOR: usize = 4;

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const half::f16) -> Self::Vector {
        let mut v = [half::f16::ZERO; 16];
        core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), 16);
        v
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const half::f16) -> Self::Vector {
        let mut v = [half::f16::ZERO; 16];
        core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), 16);
        v
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut half::f16, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 16);
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut half::f16, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 16);
    }

    #[inline(always)]
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        core::array::from_fn(|i| a[i] + b[i])
    }

    #[inline(always)]
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        core::array::from_fn(|i| a[i] * b[i])
    }

    #[inline(always)]
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        core::array::from_fn(|i| a[i] - b[i])
    }

    #[inline(always)]
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector {
        core::array::from_fn(|i| {
            half::f16::from_f32(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32()))
        })
    }

    #[inline(always)]
    unsafe fn sum_reduce(v: Self::Vector) -> half::f16 {
        v.iter().copied().fold(half::f16::ZERO, |acc, x| acc + x)
    }

    #[inline(always)]
    unsafe fn masked_load_unaligned(
        ptr: *const half::f16,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| if mask[i] { *ptr.add(i) } else { src[i] })
    }

    #[inline(always)]
    unsafe fn masked_store_unaligned(ptr: *mut half::f16, mask: Self::Mask, val: Self::Vector) {
        for i in 0..16 {
            if mask[i] {
                *ptr.add(i) = val[i];
            }
        }
    }

    #[inline(always)]
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| if mask[i] { a[i] + b[i] } else { src[i] })
    }

    #[inline(always)]
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| if mask[i] { a[i] * b[i] } else { src[i] })
    }

    #[inline(always)]
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector {
        core::array::from_fn(|i| {
            if mask[i] {
                half::f16::from_f32(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32()))
            } else {
                c[i]
            }
        })
    }

    #[inline(always)]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> half::f16 {
        let mut s = half::f16::ZERO;
        for i in 0..16 {
            if mask[i] {
                s += v[i];
            }
        }
        s
    }

    #[inline(always)]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut out = [half::f16::ZERO; 16];
        let mut k = 0;
        for i in 0..16 {
            if mask[i] {
                out[k] = src[i];
                k += 1;
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector {
        let mut out = fill;
        let mut k = 0;
        for i in 0..16 {
            if mask[i] {
                out[i] = src[k];
                k += 1;
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn gather(base: *const half::f16, indices: Self::IndexVector) -> Self::Vector {
        core::array::from_fn(|i| *base.add(indices[i] as usize))
    }

    #[inline(always)]
    unsafe fn gather_masked(
        base: *const half::f16,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| {
            if mask[i] {
                *base.add(indices[i] as usize)
            } else {
                src[i]
            }
        })
    }

    #[inline(always)]
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask {
        debug_assert_eq!(bits.len(), 16);
        core::array::from_fn(|i| bits[i])
    }

    #[inline(always)]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        core::array::from_fn(|i| i < k)
    }

    #[inline(always)]
    unsafe fn zero() -> Self::Vector {
        [half::f16::ZERO; 16]
    }

    #[inline(always)]
    unsafe fn splat(val: half::f16) -> Self::Vector {
        [val; 16]
    }

    #[inline(always)]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        let mut m = 0u64;
        for i in 0..16 {
            if mask[i] {
                m |= 1u64 << i;
            }
        }
        m
    }

    #[inline(always)]
    unsafe fn mask_to_vector(mask: Self::Mask) -> Self::Vector {
        core::array::from_fn(|i| {
            if mask[i] {
                half::f16::from_bits(0xFFFF)
            } else {
                half::f16::ZERO
            }
        })
    }
}

//! AVX-512 f16 emulated SIMD kernel.
//!
//! Provides a documented software-emulated fallback for half-precision float arithmetic
//! with 32 lanes on AVX-512 when native FP16 instructions are not present/compiled.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::Avx512;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use eunomia::F16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd_core::kernel::BackendKernel;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl BackendKernel<F16> for Avx512 {
    type Vector = [F16; 32];
    type Mask = [bool; 32];
    type IndexVector = [i32; 32];
    const LANE_COUNT: usize = 32;
    const UNROLL_FACTOR: usize = 4;

    #[inline(always)]
    unsafe fn load_aligned(ptr: *const F16) -> Self::Vector {
        let mut v = [F16::ZERO; 32];
        core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), 32);
        v
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const F16) -> Self::Vector {
        let mut v = [F16::ZERO; 32];
        core::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), 32);
        v
    }

    #[inline(always)]
    unsafe fn store_aligned(ptr: *mut F16, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 32);
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut F16, val: Self::Vector) {
        core::ptr::copy_nonoverlapping(val.as_ptr(), ptr, 32);
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
        core::array::from_fn(|i| F16::from_f32(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32())))
    }

    #[inline(always)]
    unsafe fn sum_reduce(v: Self::Vector) -> F16 {
        v.iter().copied().fold(F16::ZERO, |acc, x| acc + x)
    }

    #[inline(always)]
    unsafe fn masked_load_unaligned(
        ptr: *const F16,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector {
        core::array::from_fn(|i| if mask[i] { *ptr.add(i) } else { src[i] })
    }

    #[inline(always)]
    unsafe fn masked_store_unaligned(ptr: *mut F16, mask: Self::Mask, val: Self::Vector) {
        for i in 0..32 {
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
                F16::from_f32(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32()))
            } else {
                c[i]
            }
        })
    }

    #[inline(always)]
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> F16 {
        let mut s = F16::ZERO;
        for i in 0..32 {
            if mask[i] {
                s += v[i];
            }
        }
        s
    }

    #[inline(always)]
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector {
        let mut out = [F16::ZERO; 32];
        let mut k = 0;
        for i in 0..32 {
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
        for i in 0..32 {
            if mask[i] {
                out[i] = src[k];
                k += 1;
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn gather(base: *const F16, indices: Self::IndexVector) -> Self::Vector {
        core::array::from_fn(|i| *base.add(indices[i] as usize))
    }

    #[inline(always)]
    unsafe fn gather_masked(
        base: *const F16,
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
        debug_assert_eq!(bits.len(), 32);
        core::array::from_fn(|i| bits[i])
    }

    #[inline(always)]
    unsafe fn leading_k_mask(k: usize) -> Self::Mask {
        core::array::from_fn(|i| i < k)
    }

    #[inline(always)]
    unsafe fn zero() -> Self::Vector {
        [F16::ZERO; 32]
    }

    #[inline(always)]
    unsafe fn splat(val: F16) -> Self::Vector {
        [val; 32]
    }

    #[inline(always)]
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64 {
        let mut m = 0u64;
        for i in 0..32 {
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
                F16::from_bits(0xFFFF)
            } else {
                F16::ZERO
            }
        })
    }

    #[inline(always)]
    unsafe fn vector_to_mask(v: Self::Vector) -> Self::Mask {
        // Bit 15 is the sign bit of a binary16 lane; testing it rather than
        // comparing against `F16::ZERO` keeps the all-ones comparison result
        // (a NaN bit pattern) from failing a floating-point equality test.
        core::array::from_fn(|i| (v[i].to_bits() >> 15) != 0)
    }
}

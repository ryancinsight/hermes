//! Low-level SIMD operations trait implemented per architecture and primitive type.
//!
//! # Extension Surface (v2+)
//!
//! New in this iteration:
//! - `sub(a, b)` — elementwise subtraction, required for `Sub` ElementOp strategy.
//! - `mask_from_bitmask(bm)` — convert `BitMask<LANE_COUNT>` to native mask; default
//!   calls `mask_from_bools` via `BitMask::to_bools()`. AVX-512 impls override with direct cast.
//! - `zero()` — returns a vector of zeros; default implementation uses `splat(T::ZERO)`.
//!   Backends may override with an architecture-specific XOR-zero idiom if profiling shows benefit.
//!
//! # Extension Surface (v2)
//!
//! Beyond the base load/store/arithmetic/reduce methods, `SimdKernel` now exposes:
//!
//! - **Masked operations** (`masked_load_unaligned`, `masked_store_unaligned`,
//!   `masked_add`, `masked_mul`, `masked_fmadd`, `masked_sum_reduce`) — predicated
//!   arithmetic using hardware mask registers. The `src` parameter follows AVX-512
//!   merge-masking semantics: lanes where `mask[i] = 0` are taken from `src`.
//!
//! - **Compress / expand** — scatter/gather from/to contiguous storage:
//!   - `compress`: packs selected lanes (mask[i]=1) to low lanes of result.
//!   - `expand`: scatters low lanes of `src` to positions where mask[i]=1.
//!
//! - **Gather** (`gather`, `gather_masked`) — indirect indexed load from a base pointer.
//!
//! - **Mask construction** (`mask_from_bools`, `leading_k_mask`) — build masks from
//!   boolean arrays or lane counts for tail handling.
//!
//! # Architecture Mapping
//!
//! | Method | AVX-512 | AVX2 | NEON | Scalar |
//! |--------|---------|------|------|--------|
//! | `masked_add` | `_mm512_mask_add_ps` | `_mm256_blendv_ps(src,add,mask)` | `vbslq_f32` | loop+if |
//! | `compress` | `_mm512_mask_compressstoreu_ps` | emulated | emulated | loop+if |
//! | `gather` | `_mm512_i32gather_ps` | `_mm256_i32gather_ps` | emulated | loop |

/// Abstract trait defining low-level vector operations.
///
/// Implemented by ZST architecture markers. All methods are `unsafe` — the caller is
/// responsible for ensuring target-feature prerequisites are satisfied. The `#[target_feature]`
/// attribute on each `impl` block ensures the compiler emits the correct machine instruction;
/// calling from a non-gated context requires wrapping in an `unsafe { ... }` block inside
/// a function that is itself gated by `#[target_feature(enable = "...")]`.
pub trait SimdKernel<T: crate::scalar::Scalar>: crate::private::Sealed + Send + Sync + 'static {
    /// The underlying raw register/vector type for this architecture and element type.
    type Vector: Copy + Send + Sync + 'static;

    /// Hardware-native mask type.
    ///
    /// - AVX-512 f32: `__mmask16`
    /// - AVX-512 f64: `__mmask8`
    /// - AVX2 f32: `__m256` (float blend mask)
    /// - AVX2 f64: `__m256d`
    /// - NEON f32: `uint32x4_t`
    /// - NEON f64: `uint64x2_t`
    /// - Scalar f32: `[bool; 4]`
    /// - Scalar f64: `[bool; 2]`
    type Mask: Copy + Send + Sync + 'static;

    /// Integer index vector for gather operations.
    ///
    /// - AVX-512 f32 (16-lane): `__m512i` (16xi32)
    /// - AVX-512 f64 (8-lane): `__m256i` (8xi32)
    /// - AVX2 f32 (8-lane): `__m256i` (8xi32)
    /// - AVX2 f64 (4-lane): `__m128i` (4xi32)
    /// - NEON / Scalar: `[i32; LANE_COUNT]`
    type IndexVector: Copy + Send + Sync + 'static;

    /// Number of primitive elements of type `T` in one `Vector`.
    const LANE_COUNT: usize;

    /// Loop unrolling register accumulation factor to break loop-carried dependency chains.
    const UNROLL_FACTOR: usize = 4;

    // -------------------------------------------------------------------------
    // Load / Store
    // -------------------------------------------------------------------------

    /// Load a vector from an aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for reads and aligned to `LANE_COUNT * size_of::<T>()` bytes.
    unsafe fn load_aligned(ptr: *const T) -> Self::Vector;

    /// Load a vector from an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for reads.
    unsafe fn load_unaligned(ptr: *const T) -> Self::Vector;

    /// Store a vector to an aligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for writes and aligned to `LANE_COUNT * size_of::<T>()` bytes.
    unsafe fn store_aligned(ptr: *mut T, val: Self::Vector);

    /// Store a vector to an unaligned pointer.
    ///
    /// # Safety
    /// `ptr` must be valid for writes.
    unsafe fn store_unaligned(ptr: *mut T, val: Self::Vector);

    // -------------------------------------------------------------------------
    // Dense Arithmetic
    // -------------------------------------------------------------------------

    /// Elementwise addition: `a + b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn add(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Elementwise multiplication: `a * b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mul(a: Self::Vector, b: Self::Vector) -> Self::Vector;

    /// Elementwise subtraction: `a - b`.
    ///
    /// Default: panics with "not implemented" — override in each backend.
    /// Required for `Sub` ElementOp strategy (`zip_cow` / `transform_in_place`).
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn sub(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let _ = (a, b);
        unreachable!("SimdKernel::sub not implemented for this architecture")
    }

    /// Fused multiply-add: `(a * b) + c`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn fmadd(a: Self::Vector, b: Self::Vector, c: Self::Vector) -> Self::Vector;

    /// Horizontal sum of all lanes.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn sum_reduce(v: Self::Vector) -> T;

    // -------------------------------------------------------------------------
    // Masked Load / Store (merge masking: inactive lanes come from `src`)
    // -------------------------------------------------------------------------

    /// Masked load: active lanes loaded from `ptr`, inactive lanes taken from `src`.
    ///
    /// # Safety
    /// `ptr` must be valid for reading `LANE_COUNT` elements. Active lanes determined by `mask`.
    unsafe fn masked_load_unaligned(ptr: *const T, mask: Self::Mask, src: Self::Vector) -> Self::Vector;

    /// Masked store: active lanes written to `ptr`, inactive lanes left unchanged.
    ///
    /// # Safety
    /// `ptr` must be valid for writing `LANE_COUNT` elements.
    unsafe fn masked_store_unaligned(ptr: *mut T, mask: Self::Mask, val: Self::Vector);

    // -------------------------------------------------------------------------
    // Masked Arithmetic (merge masking)
    // -------------------------------------------------------------------------

    /// Masked elementwise add: active lanes compute `a + b`, inactive lanes yield `src`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_add(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Masked elementwise multiply: active lanes compute `a * b`, inactive lanes yield `src`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_mul(
        a: Self::Vector,
        b: Self::Vector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    /// Masked fused multiply-add: active lanes compute `(a * b) + c`, inactive lanes retain `c`.
    ///
    /// The merge source for inactive lanes is the addend `c`, matching AVX-512 semantics
    /// for `_mm512_mask_fmadd_ps(a, mask, b, c)`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_fmadd(
        a: Self::Vector,
        b: Self::Vector,
        c: Self::Vector,
        mask: Self::Mask,
    ) -> Self::Vector;

    /// Masked horizontal sum: only lanes where `mask[i]=1` contribute.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn masked_sum_reduce(v: Self::Vector, mask: Self::Mask) -> T;

    // -------------------------------------------------------------------------
    // Compress / Expand
    // -------------------------------------------------------------------------

    /// Compress: pack selected lanes (where `mask[i]=1`) into the low lanes of the result.
    ///
    /// Unselected high lanes of the result are unspecified.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn compress(src: Self::Vector, mask: Self::Mask) -> Self::Vector;

    /// Expand: scatter the low lanes of `src` into result positions where `mask[i]=1`.
    ///
    /// Result positions where `mask[i]=0` are filled with `fill`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn expand(src: Self::Vector, mask: Self::Mask, fill: Self::Vector) -> Self::Vector;

    // -------------------------------------------------------------------------
    // Gather (indirect indexed load)
    // -------------------------------------------------------------------------

    /// Gather: load `LANE_COUNT` elements at `base + indices[i]` for each lane `i`.
    ///
    /// # Safety
    /// All `base + indices[i]` must be valid for reads.
    unsafe fn gather(base: *const T, indices: Self::IndexVector) -> Self::Vector;

    /// Masked gather: gather active lanes; inactive lanes take value from `src`.
    ///
    /// # Safety
    /// Active `base + indices[i]` must be valid for reads.
    unsafe fn gather_masked(
        base: *const T,
        indices: Self::IndexVector,
        mask: Self::Mask,
        src: Self::Vector,
    ) -> Self::Vector;

    // -------------------------------------------------------------------------
    // Mask Construction Helpers
    // -------------------------------------------------------------------------

    /// Construct a mask from a slice of booleans (length must equal `LANE_COUNT`).
    ///
    /// # Panics
    /// Panics in debug builds if `bits.len() != LANE_COUNT`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_from_bools(bits: &[bool]) -> Self::Mask;

    /// Construct a mask with the first `k` lanes active and the rest inactive.
    ///
    /// If `k >= LANE_COUNT`, all lanes are active. Used for tail handling.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn leading_k_mask(k: usize) -> Self::Mask;

    /// Convert a raw `u64` bitmask to the architecture-native mask type.
    ///
    /// Default: expands to a boolean array then calls `mask_from_bools`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_from_bitmask(bm: u64) -> Self::Mask
    {
        let mut bools = [false; 64];
        for i in 0..Self::LANE_COUNT {
            bools[i] = (bm >> i) & 1 == 1;
        }
        Self::mask_from_bools(&bools[..Self::LANE_COUNT])
    }

    /// Set all lanes to zero.
    ///
    /// Default: delegates to `splat(T::ZERO)`. Backends may override with an
    /// architecture-specific XOR-zero idiom (e.g., `_mm256_xor_ps`) if profiling
    /// shows a register-pressure benefit.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn zero() -> Self::Vector {
        Self::splat(T::ZERO)
    }

    /// Broadcast a scalar value to all lanes.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn splat(val: T) -> Self::Vector;

    /// Elementwise division: `a / b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn div(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = buf_a[i] / buf_b[i];
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise bitwise AND: `a & b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitand(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = buf_a[i].bitand(buf_b[i]);
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise bitwise OR: `a | b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = buf_a[i].bitor(buf_b[i]);
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise bitwise XOR: `a ^ b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn bitxor(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = buf_a[i].bitxor(buf_b[i]);
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise absolute value.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn abs(a: Self::Vector) -> Self::Vector {
        let mut buf = [T::ZERO; 128];
        Self::store_unaligned(buf.as_mut_ptr(), a);
        for i in 0..Self::LANE_COUNT {
            buf[i] = buf[i].abs();
        }
        Self::load_unaligned(buf.as_ptr())
    }

    /// Elementwise minimum of `a` and `b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn min(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] < buf_b[i] { buf_a[i] } else { buf_b[i] };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise maximum of `a` and `b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn max(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] > buf_b[i] { buf_a[i] } else { buf_b[i] };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise square root.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn sqrt(a: Self::Vector) -> Self::Vector {
        let mut buf = [T::ZERO; 128];
        Self::store_unaligned(buf.as_mut_ptr(), a);
        for i in 0..Self::LANE_COUNT {
            buf[i] = buf[i].sqrt();
        }
        Self::load_unaligned(buf.as_ptr())
    }

    /// Elementwise equal: `a == b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_eq(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] == buf_b[i] { T::ALL_ONES } else { T::ZERO };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise not equal: `a != b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_ne(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] != buf_b[i] { T::ALL_ONES } else { T::ZERO };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise less than: `a < b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_lt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] < buf_b[i] { T::ALL_ONES } else { T::ZERO };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise less than or equal: `a <= b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_le(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] <= buf_b[i] { T::ALL_ONES } else { T::ZERO };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise greater than: `a > b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_gt(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] > buf_b[i] { T::ALL_ONES } else { T::ZERO };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise greater than or equal: `a >= b`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn cmp_ge(a: Self::Vector, b: Self::Vector) -> Self::Vector {
        let mut buf_a = [T::ZERO; 128];
        let mut buf_b = [T::ZERO; 128];
        Self::store_unaligned(buf_a.as_mut_ptr(), a);
        Self::store_unaligned(buf_b.as_mut_ptr(), b);
        for i in 0..Self::LANE_COUNT {
            buf_a[i] = if buf_a[i] >= buf_b[i] { T::ALL_ONES } else { T::ZERO };
        }
        Self::load_unaligned(buf_a.as_ptr())
    }

    /// Elementwise blend: select lanes from `true_val` where the sign bit of `mask` is set,
    /// and from `false_val` otherwise.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector {
        let mut buf_mask = [T::ZERO; 128];
        let mut buf_true = [T::ZERO; 128];
        let mut buf_false = [T::ZERO; 128];
        Self::store_unaligned(buf_mask.as_mut_ptr(), mask);
        Self::store_unaligned(buf_true.as_mut_ptr(), true_val);
        Self::store_unaligned(buf_false.as_mut_ptr(), false_val);
        for i in 0..Self::LANE_COUNT {
            let is_true = buf_mask[i].is_nan() || buf_mask[i].to_f64() != 0.0;
            buf_true[i] = if is_true { buf_true[i] } else { buf_false[i] };
        }
        Self::load_unaligned(buf_true.as_ptr())
    }

    /// Elementwise unary negation: `-a`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn neg(a: Self::Vector) -> Self::Vector {
        Self::sub(Self::zero(), a)
    }

    /// Elementwise bitwise NOT: `!a`.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    #[inline(always)]
    unsafe fn bitnot(a: Self::Vector) -> Self::Vector {
        Self::bitxor(a, Self::splat(T::ALL_ONES))
    }

    /// Convert the native mask back to a raw `u64` bitmask.
    ///
    /// # Safety
    /// Processor must support the required target feature.
    unsafe fn mask_to_bitmask(mask: Self::Mask) -> u64;
}
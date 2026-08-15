use crate::kernel::{BackendKernel, MAX_SIMD_LANES};
use crate::scalar::Scalar;

/// Generic merge-masked load default: active lanes (per `mask`) are loaded from
/// `ptr`; inactive lanes keep their value from `src`.
///
/// Backends with a native masked load override [`BackendKernel::masked_load_unaligned`];
/// new backends/types inherit this scalar-emulated default for free. The bitmask
/// shift bounds it to `LANE_COUNT <= MAX_SIMD_LANES`, checked at compile time.
///
/// # Safety
/// `ptr` must be valid for reading `Arch::LANE_COUNT` elements of `T`.
#[inline(always)]
pub unsafe fn generic_masked_load<T, Arch>(
    ptr: *const T,
    mask: Arch::Mask,
    src: Arch::Vector,
) -> Arch::Vector
where
    T: Scalar,
    Arch: BackendKernel<T>,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let bm = Arch::mask_to_bitmask(mask);
    let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    // Seed every lane from `src`, then overwrite the active lanes from memory.
    Arch::store_unaligned(buf.as_mut_ptr().cast::<T>(), src);
    for i in 0..Arch::LANE_COUNT {
        if (bm >> i) & 1 == 1 {
            buf[i].write(*ptr.add(i));
        }
    }
    Arch::load_unaligned(buf.as_ptr().cast::<T>())
}

/// Generic merge-masked store default: active lanes (per `mask`) of `val` are
/// written to `ptr`; inactive lanes of `ptr` are left unchanged.
///
/// Backends with a native masked store override [`BackendKernel::masked_store_unaligned`].
///
/// # Safety
/// `ptr` must be valid for writing the active lanes' elements of `T`.
#[inline(always)]
pub unsafe fn generic_masked_store<T, Arch>(ptr: *mut T, mask: Arch::Mask, val: Arch::Vector)
where
    T: Scalar,
    Arch: BackendKernel<T>,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let bm = Arch::mask_to_bitmask(mask);
    let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf.as_mut_ptr().cast::<T>(), val);
    for i in 0..Arch::LANE_COUNT {
        if (bm >> i) & 1 == 1 {
            *ptr.add(i) = buf[i].assume_init();
        }
    }
}

/// Generic indexed-store (scatter) default: lane `i` of `val` is written to
/// `base + indices[i]`, for the lanes selected by `bitmask`.
///
/// This is the write-side dual of [`BackendKernel::gather`]. Backends with a native
/// scatter instruction (AVX-512 `vscatterdps`/`vscatterdpd`) override
/// [`BackendKernel::scatter`] and [`BackendKernel::scatter_masked`]; every other
/// backend inherits this lane-sequential default, since neither AVX2 nor NEON
/// has a scatter instruction to dispatch to.
///
/// Lanes are written in ascending order, so when `indices` repeats a value the
/// highest active lane holding it wins — the same last-writer-wins rule the
/// hardware instructions document. `T: Scalar` is `Copy`, so no store drops a
/// previous value.
///
/// # Safety
/// - Every active `base + indices[i]` must be valid for writing one `T`.
/// - `Arch::IndexVector` must be `Arch::LANE_COUNT` packed `i32`s, the
///   workspace-wide layout invariant asserted at compile time below.
#[inline(always)]
unsafe fn generic_scatter_bitmask<T, Arch>(
    base: *mut T,
    indices: Arch::IndexVector,
    val: Arch::Vector,
    bitmask: u64,
) where
    T: Scalar,
    Arch: BackendKernel<T>,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    // Compile-time guard binding the soundness condition the SAFETY note relies
    // on: a backend whose `IndexVector` is not `LANE_COUNT` packed `i32`s fails
    // to build rather than reading uninitialized index lanes below.
    const {
        assert!(
            core::mem::size_of::<Arch::IndexVector>()
                == <Arch as BackendKernel<T>>::LANE_COUNT * core::mem::size_of::<i32>(),
            "IndexVector size must equal LANE_COUNT * size_of::<i32>()"
        );
    };

    let mut idx = [0_i32; MAX_SIMD_LANES];
    // The const assert above bounds the write to `LANE_COUNT <= MAX_SIMD_LANES`
    // index slots; the pointer is `i32`-aligned, which is all an unaligned write
    // of the packed-`i32` `IndexVector` requires.
    core::ptr::write_unaligned(idx.as_mut_ptr().cast::<Arch::IndexVector>(), indices);

    let mut lanes = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(lanes.as_mut_ptr().cast::<T>(), val);

    for i in 0..Arch::LANE_COUNT {
        if (bitmask >> i) & 1 == 1 {
            *base.offset(idx[i] as isize) = lanes[i].assume_init();
        }
    }
}

/// Generic unmasked scatter default: every lane of `val` is stored.
///
/// # Safety
/// Carries [`generic_scatter_bitmask`]'s contract with all lanes active.
#[inline(always)]
pub unsafe fn generic_scatter<T, Arch>(base: *mut T, indices: Arch::IndexVector, val: Arch::Vector)
where
    T: Scalar,
    Arch: BackendKernel<T>,
{
    let all_active = if Arch::LANE_COUNT >= u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << Arch::LANE_COUNT) - 1
    };
    generic_scatter_bitmask::<T, Arch>(base, indices, val, all_active);
}

/// Generic masked scatter default: only lanes active in `mask` are stored.
///
/// # Safety
/// Carries [`generic_scatter_bitmask`]'s contract for the active lanes only;
/// inactive lanes' indices are never dereferenced.
#[inline(always)]
pub unsafe fn generic_scatter_masked<T, Arch>(
    base: *mut T,
    indices: Arch::IndexVector,
    mask: Arch::Mask,
    val: Arch::Vector,
) where
    T: Scalar,
    Arch: BackendKernel<T>,
{
    let bitmask = Arch::mask_to_bitmask(mask);
    generic_scatter_bitmask::<T, Arch>(base, indices, val, bitmask);
}

#[inline(always)]
pub unsafe fn generic_binary_op<T, Arch, F>(
    a: Arch::Vector,
    b: Arch::Vector,
    mut op: F,
) -> Arch::Vector
where
    T: Scalar,
    Arch: BackendKernel<T>,
    F: FnMut(T, T) -> T,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let mut buf_a = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_b = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf_a.as_mut_ptr().cast::<T>(), a);
    Arch::store_unaligned(buf_b.as_mut_ptr().cast::<T>(), b);
    for i in 0..Arch::LANE_COUNT {
        let val_a = buf_a[i].assume_init();
        let val_b = buf_b[i].assume_init();
        buf_a[i].write(op(val_a, val_b));
    }
    Arch::load_unaligned(buf_a.as_ptr().cast::<T>())
}

#[inline(always)]
pub unsafe fn generic_unary_op<T, Arch, F>(a: Arch::Vector, mut op: F) -> Arch::Vector
where
    T: Scalar,
    Arch: BackendKernel<T>,
    F: FnMut(T) -> T,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf.as_mut_ptr().cast::<T>(), a);
    for i in 0..Arch::LANE_COUNT {
        let val = buf[i].assume_init();
        buf[i].write(op(val));
    }
    Arch::load_unaligned(buf.as_ptr().cast::<T>())
}

#[inline(always)]
pub unsafe fn generic_blend<T, Arch>(
    mask: Arch::Vector,
    true_val: Arch::Vector,
    false_val: Arch::Vector,
) -> Arch::Vector
where
    T: Scalar,
    Arch: BackendKernel<T>,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let mut buf_mask = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_true = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_false = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf_mask.as_mut_ptr().cast::<T>(), mask);
    Arch::store_unaligned(buf_true.as_mut_ptr().cast::<T>(), true_val);
    Arch::store_unaligned(buf_false.as_mut_ptr().cast::<T>(), false_val);
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
    Arch::load_unaligned(buf_true.as_ptr().cast::<T>())
}

#[inline(always)]
pub unsafe fn generic_mask_from_bitmask<T, Arch>(bm: u64) -> Arch::Mask
where
    T: Scalar,
    Arch: BackendKernel<T>,
{
    // The `u64` bitmask (`bm >> i` is defined only for `i < 64`) and the
    // `bools` buffer bound this default; both are covered by the shared
    // `MAX_SIMD_LANES <= 64` scalar-fallback SSOT, checked at compile time
    // per backend.
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let mut bools = [false; MAX_SIMD_LANES];
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
    Arch: BackendKernel<T>,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let mut buf_a = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_b = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    let mut buf_c = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf_a.as_mut_ptr().cast::<T>(), a);
    Arch::store_unaligned(buf_b.as_mut_ptr().cast::<T>(), b);
    Arch::store_unaligned(buf_c.as_mut_ptr().cast::<T>(), c);
    for i in 0..Arch::LANE_COUNT {
        let val_a = buf_a[i].assume_init();
        let val_b = buf_b[i].assume_init();
        let val_c = buf_c[i].assume_init();
        let prod = val_a * val_b;
        let add = (i & 1 == 1) ^ ADD_EVEN;
        buf_a[i].write(if add { prod + val_c } else { prod - val_c });
    }
    Arch::load_unaligned(buf_a.as_ptr().cast::<T>())
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
    Arch: BackendKernel<T>,
{
    const { <Arch as BackendKernel<T>>::LANE_BOUND_CHECK };
    let mut buf = [core::mem::MaybeUninit::<T>::uninit(); MAX_SIMD_LANES];
    Arch::store_unaligned(buf.as_mut_ptr().cast::<T>(), v);
    let mut acc = identity;
    for i in 0..Arch::LANE_COUNT {
        let val = buf[i].assume_init();
        acc = op(acc, val);
    }
    acc
}

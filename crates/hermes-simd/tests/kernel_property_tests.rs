//! Property tests for `SimdKernel` mask, compress/expand, gather, and
//! tail-mask primitives, exercised per architecture backend.
//!
//! The `Scalar` and SVE-shaped emulated backends always run; AVX2 / AVX-512 run
//! when the host CPU supports them (CI provides at least AVX2 on x86_64 runners
//! and NEON on aarch64 runners via `PreferredArch`-independent explicit
//! markers).

use hermes_simd::{Scalar, SveArch};
use hermes_simd_core::align::Unaligned;
use hermes_simd_core::execution::Unmasked;
use hermes_simd_core::kernel::SimdKernel;
use hermes_simd_core::view::SimdView;
use proptest::prelude::*;

/// Truncate a raw bitmask to the backend's lane count.
fn lane_bits<A: SimdKernel<f32>>(bm: u64) -> u64 {
    bm & ((1u64 << A::LANE_COUNT) - 1)
}

/// `mask_from_bitmask` ∘ `mask_to_bitmask` must be the identity on lane bits.
fn check_bitmask_roundtrip<A: SimdKernel<f32>>(bm: u64) {
    let bm = lane_bits::<A>(bm);
    // SAFETY: caller gates on the required target features for `A`.
    let roundtrip = unsafe { A::mask_to_bitmask(A::mask_from_bitmask(bm)) };
    assert_eq!(
        lane_bits::<A>(roundtrip),
        bm,
        "bitmask round-trip failed for {bm:#b}"
    );
}

/// `expand(compress(v, m), m, fill)` must restore active lanes of `v` and put
/// `fill` in inactive lanes (compress packs active lanes low; expand scatters
/// them back to the same positions).
fn check_compress_expand_identity<A: SimdKernel<f32>>(bm: u64, vals: &[f32]) {
    let lanes = A::LANE_COUNT;
    let bm = lane_bits::<A>(bm);
    let src: Vec<f32> = (0..lanes).map(|i| vals[i % vals.len()]).collect();
    const FILL: f32 = -512.5;

    let mut out = vec![0.0f32; lanes];
    // SAFETY: caller gates on the required target features for `A`; all
    // pointers cover exactly LANE_COUNT elements.
    unsafe {
        let v = A::load_unaligned(src.as_ptr());
        let mask = A::mask_from_bitmask(bm);
        let compressed = A::compress(v, mask);
        let restored = A::expand(compressed, mask, A::splat(FILL));
        A::store_unaligned(out.as_mut_ptr(), restored);
    }

    for (i, &x) in out.iter().enumerate() {
        if (bm >> i) & 1 == 1 {
            assert_eq!(x, src[i], "active lane {i} not restored (mask {bm:#b})");
        } else {
            assert_eq!(x, FILL, "inactive lane {i} not filled (mask {bm:#b})");
        }
    }
}

/// View-level gather must equal the scalar reference for arbitrary in-bounds
/// index permutations (including repeats).
fn check_gather_matches_reference<A>(values: &[f32], indices: &[i32])
where
    A: hermes_simd_core::arch::SimdArch + SimdKernel<f32>,
{
    let view = SimdView::<f32, A, Unaligned, Unmasked, &[f32]>::new(values).unwrap();
    let mut out = vec![0.0f32; indices.len()];
    view.gather(indices, &mut out).unwrap();
    for (k, &idx) in indices.iter().enumerate() {
        assert_eq!(out[k], values[idx as usize], "gather mismatch at {k}");
    }
}

/// `masked_sum_reduce` with `leading_k_mask(k)` must sum exactly the first
/// `min(k, LANE_COUNT)` lanes, including the k = 0 and k > LANE_COUNT bounds.
fn check_leading_k_masked_sum<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let vals: Vec<f32> = (0..lanes).map(|i| (i + 1) as f32).collect();
    for k in 0..=lanes + 2 {
        // SAFETY: caller gates on the required target features for `A`.
        let total = unsafe {
            let v = A::load_unaligned(vals.as_ptr());
            A::masked_sum_reduce(v, A::leading_k_mask(k))
        };
        let expected: f32 = vals[..k.min(lanes)].iter().sum();
        assert_eq!(total, expected, "leading_k_mask({k}) sum mismatch");
    }
}

/// Run every kernel-level check for one backend.
fn check_all_kernel_invariants<A>(bm: u64, vals: &[f32])
where
    A: hermes_simd_core::arch::SimdArch + SimdKernel<f32>,
{
    check_bitmask_roundtrip::<A>(bm);
    check_compress_expand_identity::<A>(bm, vals);
    check_leading_k_masked_sum::<A>();
}

proptest! {
    #[test]
    fn prop_kernel_invariants_all_backends(
        bm in any::<u64>(),
        vals in prop::collection::vec(-1000.0f32..1000.0, 1..32),
    ) {
        check_all_kernel_invariants::<Scalar>(bm, &vals);
        check_all_kernel_invariants::<SveArch>(bm, &vals);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                check_all_kernel_invariants::<hermes_simd::Avx2>(bm, &vals);
            }
            if std::is_x86_feature_detected!("avx512f") {
                check_all_kernel_invariants::<hermes_simd::Avx512>(bm, &vals);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            check_all_kernel_invariants::<hermes_simd::Neon>(bm, &vals);
        }
    }

    #[test]
    fn prop_gather_matches_reference_all_backends(
        (values, indices) in prop::collection::vec(-1000.0f32..1000.0, 1..256)
            .prop_flat_map(|v| {
                let n = v.len();
                (Just(v), prop::collection::vec(0..n as i32, 0..64))
            }),
    ) {
        check_gather_matches_reference::<Scalar>(&values, &indices);
        check_gather_matches_reference::<SveArch>(&values, &indices);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                check_gather_matches_reference::<hermes_simd::Avx2>(&values, &indices);
            }
            if std::is_x86_feature_detected!("avx512f") {
                check_gather_matches_reference::<hermes_simd::Avx512>(&values, &indices);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            check_gather_matches_reference::<hermes_simd::Neon>(&values, &indices);
        }
    }
}

#[test]
fn gather_rejects_out_of_bounds_indices() {
    let values = [1.0f32, 2.0, 3.0];
    let view = SimdView::<f32, Scalar, Unaligned, Unmasked, &[f32]>::new(&values).unwrap();
    let mut out = [0.0f32; 2];
    assert!(matches!(
        view.gather(&[0, 3], &mut out),
        Err(hermes_simd_core::view::SimdError::IndexOutOfBounds)
    ));
    assert!(matches!(
        view.gather(&[-1, 0], &mut out),
        Err(hermes_simd_core::view::SimdError::IndexOutOfBounds)
    ));
}

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

/// View-level scatter must equal the scalar reference for arbitrary in-bounds
/// index sequences, including repeats — where the highest lane writing an index
/// wins, the documented last-writer-wins contract.
fn check_scatter_matches_reference<A>(len: usize, indices: &[i32])
where
    A: hermes_simd_core::arch::SimdArch + SimdKernel<f32>,
{
    let src: Vec<f32> = (0..indices.len()).map(|i| (i + 1) as f32 * 0.5).collect();

    let mut expected = vec![0.0f32; len];
    for (k, &idx) in indices.iter().enumerate() {
        expected[idx as usize] = src[k];
    }

    let mut actual = vec![0.0f32; len];
    {
        let mut view =
            SimdView::<f32, A, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut actual).unwrap();
        view.scatter(indices, &src).unwrap();
    }
    assert_eq!(actual, expected, "scatter mismatch (len {len})");
}

/// Scatter is the write-side inverse of gather: scattering a permutation and
/// gathering back through the same indices must restore the source exactly.
/// This oracle is independent of the element-wise reference loop above.
fn check_gather_scatter_roundtrip<A>(values: &[f32])
where
    A: hermes_simd_core::arch::SimdArch + SimdKernel<f32>,
{
    let n = values.len();
    // A permutation guarantees every destination is written exactly once, so
    // the round-trip is well defined regardless of the duplicate-index rule.
    let perm: Vec<i32> = (0..n).map(|i| ((i * 7 + 3) % n) as i32).collect();
    let perm: Vec<i32> = if perm.iter().collect::<std::collections::HashSet<_>>().len() == n {
        perm
    } else {
        (0..n as i32).rev().collect()
    };

    let mut scattered = vec![0.0f32; n];
    {
        let mut view =
            SimdView::<f32, A, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut scattered).unwrap();
        view.scatter(&perm, values).unwrap();
    }

    let view = SimdView::<f32, A, Unaligned, Unmasked, &[f32]>::new(&scattered).unwrap();
    let mut restored = vec![0.0f32; n];
    view.gather(&perm, &mut restored).unwrap();

    assert_eq!(restored, values, "gather∘scatter is not the identity");
}

/// Cross-lane permutes must match the flat reference reordering, and satisfy
/// their algebraic identities: `reverse` is an involution and `deinterleave` is
/// the exact inverse of `interleave`.
///
/// The reference is written on plain slices, independent of any lane
/// arithmetic in the kernel defaults, so a backend override and the default it
/// replaces are both checked against the same external specification.
fn check_permutes<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let a_vals: Vec<f32> = (0..lanes).map(|i| (i + 1) as f32).collect();
    let b_vals: Vec<f32> = (0..lanes).map(|i| -((i + 1) as f32) * 10.0).collect();

    let mut rev = vec![0.0f32; lanes];
    let mut lo = vec![0.0f32; lanes];
    let mut hi = vec![0.0f32; lanes];
    let mut even = vec![0.0f32; lanes];
    let mut odd = vec![0.0f32; lanes];
    let mut rt_a = vec![0.0f32; lanes];
    let mut rt_b = vec![0.0f32; lanes];
    let mut rev_twice = vec![0.0f32; lanes];

    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        let a = A::load_unaligned(a_vals.as_ptr());
        let b = A::load_unaligned(b_vals.as_ptr());

        A::store_unaligned(rev.as_mut_ptr(), A::reverse(a));
        A::store_unaligned(rev_twice.as_mut_ptr(), A::reverse(A::reverse(a)));

        let (i_lo, i_hi) = A::interleave(a, b);
        A::store_unaligned(lo.as_mut_ptr(), i_lo);
        A::store_unaligned(hi.as_mut_ptr(), i_hi);

        let (d_even, d_odd) = A::deinterleave(a, b);
        A::store_unaligned(even.as_mut_ptr(), d_even);
        A::store_unaligned(odd.as_mut_ptr(), d_odd);

        // Round-trip: deinterleave ∘ interleave == identity.
        let (r_a, r_b) = A::deinterleave(i_lo, i_hi);
        A::store_unaligned(rt_a.as_mut_ptr(), r_a);
        A::store_unaligned(rt_b.as_mut_ptr(), r_b);
    }

    // Reference reversal.
    let mut expected_rev = a_vals.clone();
    expected_rev.reverse();
    assert_eq!(rev, expected_rev, "reverse mismatch ({lanes} lanes)");
    assert_eq!(rev_twice, a_vals, "reverse is not an involution");

    // Reference interleave over the flat 2n-lane sequence.
    let mut flat = Vec::with_capacity(2 * lanes);
    for i in 0..lanes {
        flat.push(a_vals[i]);
        flat.push(b_vals[i]);
    }
    assert_eq!(lo, flat[..lanes], "interleave low half mismatch");
    assert_eq!(hi, flat[lanes..], "interleave high half mismatch");

    // Reference deinterleave over `a` followed by `b`.
    let concat: Vec<f32> = a_vals.iter().chain(b_vals.iter()).copied().collect();
    let expected_even: Vec<f32> = concat.iter().step_by(2).copied().collect();
    let expected_odd: Vec<f32> = concat.iter().skip(1).step_by(2).copied().collect();
    assert_eq!(even, expected_even, "deinterleave even mismatch");
    assert_eq!(odd, expected_odd, "deinterleave odd mismatch");

    assert_eq!(
        rt_a, a_vals,
        "deinterleave∘interleave lost the first operand"
    );
    assert_eq!(
        rt_b, b_vals,
        "deinterleave∘interleave lost the second operand"
    );
}

/// The f64 permute path is a separate monomorphization with its own lane count
/// and, on AVX2, a different instruction (`vpermpd` by immediate rather than
/// `vpermps` by index vector), so it needs its own coverage.
fn check_permutes_f64<A: SimdKernel<f64>>() {
    let lanes = A::LANE_COUNT;
    let a_vals: Vec<f64> = (0..lanes).map(|i| (i + 1) as f64).collect();
    let b_vals: Vec<f64> = (0..lanes).map(|i| -((i + 1) as f64) * 10.0).collect();

    let mut rev = vec![0.0f64; lanes];
    let mut rt_a = vec![0.0f64; lanes];
    let mut rt_b = vec![0.0f64; lanes];

    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        let a = A::load_unaligned(a_vals.as_ptr());
        let b = A::load_unaligned(b_vals.as_ptr());
        A::store_unaligned(rev.as_mut_ptr(), A::reverse(a));
        let (i_lo, i_hi) = A::interleave(a, b);
        let (r_a, r_b) = A::deinterleave(i_lo, i_hi);
        A::store_unaligned(rt_a.as_mut_ptr(), r_a);
        A::store_unaligned(rt_b.as_mut_ptr(), r_b);
    }

    let mut expected_rev = a_vals.clone();
    expected_rev.reverse();
    assert_eq!(rev, expected_rev, "f64 reverse mismatch ({lanes} lanes)");
    assert_eq!(rt_a, a_vals, "f64 round-trip lost the first operand");
    assert_eq!(rt_b, b_vals, "f64 round-trip lost the second operand");
}

#[test]
fn permutes_match_reference_all_backends() {
    check_permutes::<Scalar>();
    check_permutes::<SveArch>();
    check_permutes_f64::<Scalar>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_permutes::<hermes_simd::Avx2>();
            check_permutes_f64::<hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_permutes::<hermes_simd::Avx512>();
            check_permutes_f64::<hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_permutes::<hermes_simd::Neon>();
        check_permutes_f64::<hermes_simd::Neon>();
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

/// Masked merge ops (`masked_load`/`add`/`mul`/`fmadd`/`store`) must merge active
/// lanes (per the mask) with the inactive source. Run across Scalar/SveArch (the
/// scalar-emulated trait defaults) and AVX2/AVX-512 (native overrides), so this is
/// a differential check that the defaults match the native implementations.
/// Small-integer `f32` values keep `a*b+c` exact so native FMA == emulated.
fn check_masked_merge_ops<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let a_vals: Vec<f32> = (0..lanes).map(|i| (i + 1) as f32).collect();
    let b_vals: Vec<f32> = (0..lanes).map(|i| (2 * i + 3) as f32).collect();
    let src_vals: Vec<f32> = (0..lanes).map(|i| -((i + 1) as f32)).collect();
    let k = lanes / 2; // first half active
    let mut buf = vec![0.0f32; lanes];
    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        let a = A::load_unaligned(a_vals.as_ptr());
        let b = A::load_unaligned(b_vals.as_ptr());
        let src = A::load_unaligned(src_vals.as_ptr());
        let mask = A::leading_k_mask(k);

        A::store_unaligned(
            buf.as_mut_ptr(),
            A::masked_load_unaligned(a_vals.as_ptr(), mask, src),
        );
        for i in 0..lanes {
            let want = if i < k { a_vals[i] } else { src_vals[i] };
            assert_eq!(buf[i], want, "masked_load lane {i}");
        }

        A::store_unaligned(buf.as_mut_ptr(), A::masked_add(a, b, mask, src));
        for i in 0..lanes {
            let want = if i < k {
                a_vals[i] + b_vals[i]
            } else {
                src_vals[i]
            };
            assert_eq!(buf[i], want, "masked_add lane {i}");
        }

        A::store_unaligned(buf.as_mut_ptr(), A::masked_mul(a, b, mask, src));
        for i in 0..lanes {
            let want = if i < k {
                a_vals[i] * b_vals[i]
            } else {
                src_vals[i]
            };
            assert_eq!(buf[i], want, "masked_mul lane {i}");
        }

        // masked_fmadd merges inactive lanes from the addend `c` (= src here).
        A::store_unaligned(buf.as_mut_ptr(), A::masked_fmadd(a, b, src, mask));
        for i in 0..lanes {
            let want = if i < k {
                a_vals[i] * b_vals[i] + src_vals[i]
            } else {
                src_vals[i]
            };
            assert_eq!(buf[i], want, "masked_fmadd lane {i}");
        }

        let mut dst = src_vals.clone();
        A::masked_store_unaligned(dst.as_mut_ptr(), mask, a);
        for i in 0..lanes {
            let want = if i < k { a_vals[i] } else { src_vals[i] };
            assert_eq!(dst[i], want, "masked_store lane {i}");
        }
    }
}

/// `vector_to_mask` must invert `mask_to_vector` on lane bits.
fn check_vector_to_mask_roundtrip<A: SimdKernel<f32>>(bm: u64) {
    let bm = lane_bits::<A>(bm);
    // SAFETY: caller gates on the required target features for `A`.
    let roundtrip = unsafe {
        A::mask_to_bitmask(A::vector_to_mask(A::mask_to_vector(A::mask_from_bitmask(
            bm,
        ))))
    };
    assert_eq!(
        lane_bits::<A>(roundtrip),
        bm,
        "vector_to_mask round-trip failed for {bm:#b}"
    );
}

/// `mask_to_bitmask ∘ vector_to_mask ∘ cmp_eq` must report exactly the lanes
/// that compare equal. This is the contract extremum search relies on to locate
/// a match without leaving vector registers.
fn check_vector_to_mask_matches_cmp<A: SimdKernel<f32>>(vals: &[f32]) {
    let lanes = A::LANE_COUNT;
    let a_vals: Vec<f32> = (0..lanes).map(|i| vals[i % vals.len()]).collect();
    // Alternate lanes differ; `|v| < 1000` keeps `v + 1.0` distinct from `v` in
    // f32, so the expected mask is exactly the even lanes.
    let b_vals: Vec<f32> = a_vals
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { v + 1.0 })
        .collect();
    // SAFETY: both buffers hold exactly `LANE_COUNT` elements, so the unaligned
    // loads stay in bounds; caller gates on the required target features for `A`.
    let bm = unsafe {
        let a = A::load_unaligned(a_vals.as_ptr());
        let b = A::load_unaligned(b_vals.as_ptr());
        A::mask_to_bitmask(A::vector_to_mask(A::cmp_eq(a, b)))
    };
    for i in 0..lanes {
        let want = a_vals[i] == b_vals[i];
        let got = (bm >> i) & 1 == 1;
        assert_eq!(got, want, "cmp_eq lane {i}: {} vs {}", a_vals[i], b_vals[i]);
    }
}

/// `cmp_ne` must be the exact lane-wise complement of `cmp_eq`, NaN operands
/// included, because the trait documents it as Rust's `a != b`. An *ordered*
/// hardware not-equal predicate reports a NaN lane as neither equal nor
/// unequal, leaving both results false — the divergence this pins shut.
fn check_cmp_ne_complements_cmp_eq<A: SimdKernel<f32>>(vals: &[f32]) {
    let lanes = A::LANE_COUNT;
    let mut a_vals: Vec<f32> = (0..lanes).map(|i| vals[i % vals.len()]).collect();
    let mut b_vals: Vec<f32> = a_vals
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v } else { v + 1.0 })
        .collect();
    // Lane 0 compares NaN against NaN; lane 1 compares NaN against a finite
    // value. Both must report "not equal".
    a_vals[0] = f32::NAN;
    b_vals[0] = f32::NAN;
    if lanes > 1 {
        a_vals[1] = f32::NAN;
    }

    // SAFETY: both buffers hold exactly `LANE_COUNT` elements, so the unaligned
    // loads stay in bounds; caller gates on the required target features for `A`.
    let (eq, ne) = unsafe {
        let a = A::load_unaligned(a_vals.as_ptr());
        let b = A::load_unaligned(b_vals.as_ptr());
        (
            lane_bits::<A>(A::mask_to_bitmask(A::vector_to_mask(A::cmp_eq(a, b)))),
            lane_bits::<A>(A::mask_to_bitmask(A::vector_to_mask(A::cmp_ne(a, b)))),
        )
    };

    assert_eq!(
        ne,
        lane_bits::<A>(!eq),
        "cmp_ne must be the complement of cmp_eq (eq {eq:#b}, ne {ne:#b})"
    );
    for i in 0..lanes {
        let want = a_vals[i] != b_vals[i];
        assert_eq!(
            (ne >> i) & 1 == 1,
            want,
            "cmp_ne lane {i}: {} vs {}",
            a_vals[i],
            b_vals[i]
        );
    }
}

/// `blend` must take `true_val` exactly on the lanes a canonical mask marks
/// active. The active pattern is `ALL_ONES` — a NaN — so a backend that tests
/// the mask by comparing it against zero rather than by its sign bit
/// misclassifies every active lane under an ordered predicate.
fn check_blend_honors_canonical_mask<A: SimdKernel<f32>>(bm: u64) {
    let lanes = A::LANE_COUNT;
    let bm = lane_bits::<A>(bm);
    let true_vals: Vec<f32> = (0..lanes).map(|i| (i + 1) as f32).collect();
    let false_vals: Vec<f32> = (0..lanes).map(|i| -((i + 1) as f32)).collect();
    let mut out = vec![0.0f32; lanes];

    // SAFETY: every buffer holds exactly `LANE_COUNT` elements; caller gates on
    // the required target features for `A`.
    unsafe {
        let selected = A::load_unaligned(true_vals.as_ptr());
        let rejected = A::load_unaligned(false_vals.as_ptr());
        let mask = A::mask_to_vector(A::mask_from_bitmask(bm));
        A::store_unaligned(out.as_mut_ptr(), A::blend(mask, selected, rejected));
    }

    for (i, &got) in out.iter().enumerate() {
        let want = if (bm >> i) & 1 == 1 {
            true_vals[i]
        } else {
            false_vals[i]
        };
        assert_eq!(got, want, "blend lane {i} (mask {bm:#b})");
    }
}

/// Run every kernel-level check for one backend.
fn check_all_kernel_invariants<A>(bm: u64, vals: &[f32])
where
    A: hermes_simd_core::arch::SimdArch + SimdKernel<f32>,
{
    check_bitmask_roundtrip::<A>(bm);
    check_vector_to_mask_roundtrip::<A>(bm);
    check_vector_to_mask_matches_cmp::<A>(vals);
    check_cmp_ne_complements_cmp_eq::<A>(vals);
    check_blend_honors_canonical_mask::<A>(bm);
    check_compress_expand_identity::<A>(bm, vals);
    check_leading_k_masked_sum::<A>();
    check_masked_merge_ops::<A>();
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

    #[test]
    fn prop_scatter_matches_reference_all_backends(
        (len, indices) in (1usize..256)
            .prop_flat_map(|n| {
                (Just(n), prop::collection::vec(0..n as i32, 0..64))
            }),
    ) {
        check_scatter_matches_reference::<Scalar>(len, &indices);
        check_scatter_matches_reference::<SveArch>(len, &indices);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                check_scatter_matches_reference::<hermes_simd::Avx2>(len, &indices);
            }
            if std::is_x86_feature_detected!("avx512f") {
                check_scatter_matches_reference::<hermes_simd::Avx512>(len, &indices);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            check_scatter_matches_reference::<hermes_simd::Neon>(len, &indices);
        }
    }

    #[test]
    fn prop_gather_scatter_roundtrip_all_backends(
        values in prop::collection::vec(-1000.0f32..1000.0, 1..256),
    ) {
        check_gather_scatter_roundtrip::<Scalar>(&values);
        check_gather_scatter_roundtrip::<SveArch>(&values);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                check_gather_scatter_roundtrip::<hermes_simd::Avx2>(&values);
            }
            if std::is_x86_feature_detected!("avx512f") {
                check_gather_scatter_roundtrip::<hermes_simd::Avx512>(&values);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            check_gather_scatter_roundtrip::<hermes_simd::Neon>(&values);
        }
    }
}

#[test]
fn scatter_rejects_out_of_bounds_indices() {
    let mut values = [1.0f32, 2.0, 3.0];
    let mut view =
        SimdView::<f32, Scalar, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut values).unwrap();
    let src = [9.0f32, 9.0];
    assert!(matches!(
        view.scatter(&[0, 3], &src),
        Err(hermes_simd_core::view::SimdError::IndexOutOfBounds)
    ));
    assert!(matches!(
        view.scatter(&[-1, 0], &src),
        Err(hermes_simd_core::view::SimdError::IndexOutOfBounds)
    ));
    // The rejected calls wrote nothing.
    assert_eq!(values, [1.0f32, 2.0, 3.0]);
}

#[test]
fn scatter_rejects_short_source() {
    let mut values = [1.0f32, 2.0, 3.0];
    let mut view =
        SimdView::<f32, Scalar, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut values).unwrap();
    let src = [9.0f32];
    assert!(matches!(
        view.scatter(&[0, 1], &src),
        Err(hermes_simd_core::view::SimdError::InsufficientInputLength)
    ));
    assert_eq!(values, [1.0f32, 2.0, 3.0]);
}

/// Duplicate indices resolve last-writer-wins, matching the hardware scatter
/// rule on both the native AVX-512 path and the lane-sequential fallback.
#[test]
fn scatter_duplicate_indices_take_the_highest_lane() {
    let mut values = [0.0f32; 4];
    {
        let mut view =
            SimdView::<f32, Scalar, Unaligned, Unmasked, &mut [f32]>::new_mut(&mut values).unwrap();
        view.scatter(&[1, 1, 1, 2], &[10.0f32, 20.0, 30.0, 40.0])
            .unwrap();
    }
    assert_eq!(values, [0.0, 30.0, 40.0, 0.0]);
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

/// `recip_sqrt` must reach full native precision on every backend — it is a
/// full-precision `1/√x`, not a reduced-accuracy fast approximation. Inputs are
/// deliberately **not** perfect squares so an under-refined seed (a single Newton
/// step from a low-bit `rsqrt` estimate) is exposed rather than converging exactly
/// by luck (the trap the old perfect-square tests fell into).
///
/// Derived relative bounds (regression tripwires, not fitted):
/// - f32: a hardware `rsqrt` seed (≥12-bit on x86, 8-bit on NEON) refined by Newton
///   steps to ≥23 bits, then rounded — worst case the x86 12-bit seed + one step is
///   ≈2 ulp; `8·f32::EPSILON` (≈9.5e-7) covers the Newton-step rounding. A backend
///   left at a single 8-bit-seed step (≈1.5e-2 *…* 1.5e-5) fails this.
/// - f64: correctly-rounded hardware `sqrt` + divide ≈1 ulp; `4·f64::EPSILON`
///   (≈8.9e-16). The old rsqrt-seed paths (≈6e-8 .. 1.5e-5) fail this.
fn check_recip_sqrt_f32<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let inputs: Vec<f32> = (0..lanes).map(|i| 0.3 + 1.7 * i as f32).collect();
    let mut out = vec![0.0f32; lanes];
    // SAFETY: caller gates on the required target features for `A`; buffers cover
    // exactly LANE_COUNT elements and all inputs are strictly positive.
    unsafe {
        A::store_unaligned(
            out.as_mut_ptr(),
            A::recip_sqrt(A::load_unaligned(inputs.as_ptr())),
        );
    }
    let tol = 8.0 * f64::from(f32::EPSILON);
    for (&y, &x) in out.iter().zip(inputs.iter()) {
        let want = 1.0_f64 / f64::from(x).sqrt();
        let rel = (f64::from(y) - want).abs() / want;
        assert!(
            rel <= tol,
            "f32 recip_sqrt: x={x} got={y} want={want} rel={rel:e}"
        );
    }
}

fn check_recip_sqrt_f64<A: SimdKernel<f64>>() {
    let lanes = A::LANE_COUNT;
    let inputs: Vec<f64> = (0..lanes).map(|i| 0.3 + 1.7 * i as f64).collect();
    let mut out = vec![0.0f64; lanes];
    // SAFETY: as above; inputs strictly positive.
    unsafe {
        A::store_unaligned(
            out.as_mut_ptr(),
            A::recip_sqrt(A::load_unaligned(inputs.as_ptr())),
        );
    }
    let tol = 4.0 * f64::EPSILON;
    for (&y, &x) in out.iter().zip(inputs.iter()) {
        let want = 1.0_f64 / x.sqrt();
        let rel = (y - want).abs() / want;
        assert!(
            rel <= tol,
            "f64 recip_sqrt: x={x} got={y} want={want} rel={rel:e}"
        );
    }
}

#[test]
fn recip_sqrt_is_full_precision_all_backends() {
    check_recip_sqrt_f32::<Scalar>();
    check_recip_sqrt_f64::<Scalar>();
    check_recip_sqrt_f32::<SveArch>();
    check_recip_sqrt_f64::<SveArch>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_recip_sqrt_f32::<hermes_simd::Avx2>();
            check_recip_sqrt_f64::<hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_recip_sqrt_f32::<hermes_simd::Avx512>();
            check_recip_sqrt_f64::<hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_recip_sqrt_f32::<hermes_simd::Neon>();
        check_recip_sqrt_f64::<hermes_simd::Neon>();
    }
}

//! Property tests for the public SIMD kernel facets: mask, compress/expand,
//! gather, and
//! tail-mask primitives, exercised per architecture backend.
//!
//! The `Scalar` and SVE-shaped emulated backends always run; AVX2 / AVX-512 run
//! when the host CPU supports them (CI provides at least AVX2 on `x86_64` runners
//! and NEON on aarch64 runners via `PreferredArch`-independent explicit
//! markers).

#![expect(
    clippy::items_after_statements,
    reason = "The property harness keeps compile-time lane constants beside the exercised kernel"
)]
#![expect(
    clippy::float_cmp,
    reason = "These property checks compare exact lane values from manufactured inputs"
)]

use hermes_simd::{Scalar, SveArch};
use hermes_simd_core::align::Unaligned;
use hermes_simd_core::execution::Unmasked;
use hermes_simd_core::kernel::{SimdArith, SimdKernel, SimdReduce};
use hermes_simd_core::scalar::Scalar as ScalarElement;
use hermes_simd_core::view::SimdView;
use proptest::prelude::*;

/// Truncate a raw bitmask to the backend's lane count.
fn lane_bits<A: SimdKernel<f32>>(bm: u64) -> u64 {
    bm & ((1u64 << A::LANE_COUNT) - 1)
}

fn reduce_through_role<A: SimdReduce<f32>>(vector: A::Vector) -> f32 {
    // SAFETY: the caller supplies a vector created by the same sealed backend;
    // the role facet preserves the backend's existing ISA contract.
    unsafe { A::sum_reduce(vector) }
}

#[test]
fn reduction_role_facet_preserves_backend_contract() {
    let vector = unsafe { <Scalar as SimdArith<f32>>::splat(2.0) };
    assert_eq!(reduce_through_role::<Scalar>(vector), 8.0);
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

/// `mask_from_bitmask` (overridden with register-only expansions on the
/// native backends) must construct exactly the mask `mask_from_bools` (the
/// generic default's substrate) builds from the same bits: identical bitmask
/// round-trip and identical canonical all-ones/all-zero vector lanes.
fn check_mask_from_bitmask_matches_bools<A: SimdKernel<f32>>(bm: u64) {
    let lanes = A::LANE_COUNT;
    let bm = lane_bits::<A>(bm);
    let bools: Vec<bool> = (0..lanes).map(|i| (bm >> i) & 1 == 1).collect();
    let mut from_bitmask = vec![0.0f32; lanes];
    let mut from_bools = vec![0.0f32; lanes];
    // SAFETY: caller gates on the required target features for `A`; both
    // output buffers hold exactly `LANE_COUNT` elements.
    let (bm_bitmask, bm_bools) = unsafe {
        let a = A::mask_from_bitmask(bm);
        let b = A::mask_from_bools(&bools);
        A::store_unaligned(from_bitmask.as_mut_ptr(), A::mask_to_vector(a));
        A::store_unaligned(from_bools.as_mut_ptr(), A::mask_to_vector(b));
        (A::mask_to_bitmask(a), A::mask_to_bitmask(b))
    };
    assert_eq!(
        lane_bits::<A>(bm_bitmask),
        bm,
        "mask_from_bitmask round-trip diverged for {bm:#b}"
    );
    assert_eq!(
        bm_bitmask, bm_bools,
        "mask_from_bitmask and mask_from_bools disagree for {bm:#b}"
    );
    for i in 0..lanes {
        assert_eq!(
            from_bitmask[i].to_bits(),
            from_bools[i].to_bits(),
            "mask_to_vector lane {i} diverged between bitmask and bools construction ({bm:#b})"
        );
    }
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
    let mut pair_even = vec![0.0f32; lanes];
    let mut pair_odd = vec![0.0f32; lanes];
    let mut quad = vec![vec![0.0f32; lanes]; 4];

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

        let (p_even, p_odd) = A::deinterleave_pairs(a, b);
        A::store_unaligned(pair_even.as_mut_ptr(), p_even);
        A::store_unaligned(pair_odd.as_mut_ptr(), p_odd);

        let (q0, q1, q2, q3) = A::deinterleave_pairs4(a, b, a, b);
        A::store_unaligned(quad[0].as_mut_ptr(), q0);
        A::store_unaligned(quad[1].as_mut_ptr(), q1);
        A::store_unaligned(quad[2].as_mut_ptr(), q2);
        A::store_unaligned(quad[3].as_mut_ptr(), q3);
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

    // Reference pair deinterleave: alternating adjacent-lane pairs of the
    // concatenation, each pair's lanes kept adjacent.
    let pairs: Vec<&[f32]> = concat.chunks_exact(2).collect();
    let expected_pair_even: Vec<f32> = pairs
        .iter()
        .step_by(2)
        .flat_map(|p| p.iter().copied())
        .collect();
    let expected_pair_odd: Vec<f32> = pairs
        .iter()
        .skip(1)
        .step_by(2)
        .flat_map(|p| p.iter().copied())
        .collect();
    assert_eq!(
        pair_even, expected_pair_even,
        "deinterleave_pairs even mismatch"
    );
    assert_eq!(
        pair_odd, expected_pair_odd,
        "deinterleave_pairs odd mismatch"
    );

    // Reference stride-4 pair split over `a || b || a || b`.
    let concat4: Vec<f32> = concat.iter().chain(concat.iter()).copied().collect();
    let pairs4: Vec<&[f32]> = concat4.chunks_exact(2).collect();
    for lane_class in 0..4usize {
        let expected: Vec<f32> = pairs4
            .iter()
            .skip(lane_class)
            .step_by(4)
            .flat_map(|p| p.iter().copied())
            .collect();
        assert_eq!(
            quad[lane_class], expected,
            "deinterleave_pairs4 output {lane_class} mismatch"
        );
    }

    assert_eq!(
        rt_a, a_vals,
        "deinterleave∘interleave lost the first operand"
    );
    assert_eq!(
        rt_b, b_vals,
        "deinterleave∘interleave lost the second operand"
    );
}

/// `transpose_square` must move lane `c` of row `r` to lane `r` of row `c`,
/// checked against a flat index-coded reference, and be an involution. Run
/// both precisions per backend so every native network and the stack-capture
/// default meet one specification.
fn transpose_fixture_value<T: From<u16>>(row: usize, column: usize) -> T {
    let encoded = row
        .checked_mul(100)
        .and_then(|base| base.checked_add(column))
        .expect("fixture coordinates must fit in usize");
    T::from(u16::try_from(encoded).expect("fixture value must fit in u16"))
}

fn check_transpose_square<T, A>()
where
    T: hermes_simd_core::Scalar + PartialEq + core::fmt::Debug + From<u16>,
    A: SimdKernel<T>,
{
    let lanes = A::LANE_COUNT;
    // Row r, lane c holds 100 * r + c, so any misrouting is visible.
    let rows: Vec<Vec<T>> = (0..lanes)
        .map(|r| (0..lanes).map(|c| transpose_fixture_value(r, c)).collect())
        .collect();

    let mut out = rows.clone();
    let mut twice = rows.clone();
    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        let mut tile: Vec<A::Vector> = rows
            .iter()
            .map(|row| A::load_unaligned(row.as_ptr()))
            .collect();
        A::transpose_square(&mut tile);
        for (row, dst) in tile.iter().zip(out.iter_mut()) {
            A::store_unaligned(dst.as_mut_ptr(), *row);
        }
        A::transpose_square(&mut tile);
        for (row, dst) in tile.iter().zip(twice.iter_mut()) {
            A::store_unaligned(dst.as_mut_ptr(), *row);
        }
    }

    for r in 0..lanes {
        for c in 0..lanes {
            assert_eq!(
                out[r][c],
                transpose_fixture_value(c, r),
                "transpose mismatch at ({r}, {c}) with {lanes} lanes"
            );
        }
        assert_eq!(twice[r], rows[r], "transpose is not an involution");
    }
}

/// The interleaved-complex transpose operates on `LANE_COUNT / 2` rows and
/// moves complete adjacent scalar pairs as one sample.
fn check_transpose_interleaved_square<T, A>()
where
    T: hermes_simd_core::Scalar + PartialEq + core::fmt::Debug + From<u16>,
    A: SimdKernel<T>,
{
    let lanes = A::LANE_COUNT;
    let samples = lanes / 2;
    let rows: Vec<Vec<T>> = (0..samples)
        .map(|row| {
            (0..samples)
                .flat_map(|column| {
                    [
                        transpose_fixture_value(row, column),
                        transpose_fixture_value(row + samples, column),
                    ]
                })
                .collect()
        })
        .collect();
    let mut out = rows.clone();
    let mut twice = rows.clone();

    // SAFETY: caller gates on the required target features for `A`; each row
    // holds one complete vector and the tile has exactly `LANE_COUNT / 2`
    // rows.
    unsafe {
        let mut tile: Vec<A::Vector> = rows
            .iter()
            .map(|row| A::load_unaligned(row.as_ptr()))
            .collect();
        A::transpose_interleaved_square(&mut tile);
        for (row, dst) in tile.iter().zip(out.iter_mut()) {
            A::store_unaligned(dst.as_mut_ptr(), *row);
        }
        A::transpose_interleaved_square(&mut tile);
        for (row, dst) in tile.iter().zip(twice.iter_mut()) {
            A::store_unaligned(dst.as_mut_ptr(), *row);
        }
    }

    for row in 0..samples {
        for column in 0..samples {
            assert_eq!(
                out[row][2 * column],
                rows[column][2 * row],
                "complex transpose real lane mismatch at ({row}, {column})"
            );
            assert_eq!(
                out[row][2 * column + 1],
                rows[column][2 * row + 1],
                "complex transpose imaginary lane mismatch at ({row}, {column})"
            );
        }
        assert_eq!(
            twice[row], rows[row],
            "complex transpose is not an involution"
        );
    }
}

/// `transpose_square` is pure data movement: it must relocate lane *bit
/// patterns* without perturbing them. The index-coded law above manufactures
/// small positive integers, which a network that leaked an operand through an
/// arithmetic or NaN-canonicalizing instruction would still satisfy. These
/// fixtures instead carry signalling and quiet NaNs, negative zero, and
/// denormals, so any such leak shows up as a changed bit pattern — the exact
/// sense in which each backend override must equal the generic default, which
/// moves lanes through `store`/`load` and therefore cannot perturb them.
fn check_transpose_square_bit_exact<T, A>(cell: impl Fn(usize, usize) -> T, bits: impl Fn(T) -> u64)
where
    T: hermes_simd_core::Scalar,
    A: SimdKernel<T>,
{
    let lanes = A::LANE_COUNT;
    let rows: Vec<Vec<T>> = (0..lanes)
        .map(|r| (0..lanes).map(|c| cell(r, c)).collect())
        .collect();
    let mut out = rows.clone();

    // SAFETY: caller gates on the required target features for `A`; every row
    // holds exactly `LANE_COUNT` elements, and `tile` exactly `LANE_COUNT` rows.
    unsafe {
        let mut tile: Vec<A::Vector> = rows
            .iter()
            .map(|row| A::load_unaligned(row.as_ptr()))
            .collect();
        A::transpose_square(&mut tile);
        for (row, dst) in tile.iter().zip(out.iter_mut()) {
            A::store_unaligned(dst.as_mut_ptr(), *row);
        }
    }

    for r in 0..lanes {
        for c in 0..lanes {
            assert_eq!(
                bits(out[r][c]),
                bits(rows[c][r]),
                "transpose perturbed the bit pattern at ({r}, {c}) with {lanes} lanes"
            );
        }
    }
}

/// Adversarial bit classes, tagged with a unique per-cell payload in the low
/// mantissa bits. OR-ing the tag preserves each class: an all-ones exponent
/// with a non-zero mantissa stays a NaN, and a zero exponent stays a denormal.
fn transpose_bits_f32(row: usize, column: usize) -> f32 {
    const CLASSES: [u32; 4] = [0x7F80_0001, 0x8000_0000, 0x0000_0001, 0xFFC0_0000];
    let tag = u32::try_from(row * 16 + column).expect("lane index fits in u32");
    f32::from_bits(CLASSES[(row + column) % CLASSES.len()] | (tag & 0xFFFF))
}

fn transpose_bits_f64(row: usize, column: usize) -> f64 {
    const CLASSES: [u64; 4] = [
        0x7FF0_0000_0000_0001,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0xFFF8_0000_0000_0000,
    ];
    let tag = u64::try_from(row * 16 + column).expect("lane index fits in u64");
    f64::from_bits(CLASSES[(row + column) % CLASSES.len()] | (tag & 0xFFFF))
}

#[test]
fn transpose_square_is_bit_exact_all_backends() {
    check_transpose_square_bit_exact::<f32, Scalar>(transpose_bits_f32, |v| u64::from(v.to_bits()));
    check_transpose_square_bit_exact::<f64, Scalar>(transpose_bits_f64, f64::to_bits);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_transpose_square_bit_exact::<f32, hermes_simd::Avx2>(transpose_bits_f32, |v| {
                u64::from(v.to_bits())
            });
            check_transpose_square_bit_exact::<f64, hermes_simd::Avx2>(
                transpose_bits_f64,
                f64::to_bits,
            );
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_transpose_square_bit_exact::<f32, hermes_simd::Avx512>(transpose_bits_f32, |v| {
                u64::from(v.to_bits())
            });
            check_transpose_square_bit_exact::<f64, hermes_simd::Avx512>(
                transpose_bits_f64,
                f64::to_bits,
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_transpose_square_bit_exact::<f32, hermes_simd::Neon>(transpose_bits_f32, |v| {
            u64::from(v.to_bits())
        });
        check_transpose_square_bit_exact::<f64, hermes_simd::Neon>(
            transpose_bits_f64,
            f64::to_bits,
        );
    }
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

    let mut pair_even = vec![0.0f64; lanes];
    let mut pair_odd = vec![0.0f64; lanes];
    let mut quad = vec![vec![0.0f64; lanes]; 4];

    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        let a = A::load_unaligned(a_vals.as_ptr());
        let b = A::load_unaligned(b_vals.as_ptr());
        A::store_unaligned(rev.as_mut_ptr(), A::reverse(a));
        let (i_lo, i_hi) = A::interleave(a, b);
        let (r_a, r_b) = A::deinterleave(i_lo, i_hi);
        A::store_unaligned(rt_a.as_mut_ptr(), r_a);
        A::store_unaligned(rt_b.as_mut_ptr(), r_b);

        let (p_even, p_odd) = A::deinterleave_pairs(a, b);
        A::store_unaligned(pair_even.as_mut_ptr(), p_even);
        A::store_unaligned(pair_odd.as_mut_ptr(), p_odd);

        let (q0, q1, q2, q3) = A::deinterleave_pairs4(a, b, a, b);
        A::store_unaligned(quad[0].as_mut_ptr(), q0);
        A::store_unaligned(quad[1].as_mut_ptr(), q1);
        A::store_unaligned(quad[2].as_mut_ptr(), q2);
        A::store_unaligned(quad[3].as_mut_ptr(), q3);
    }

    let mut expected_rev = a_vals.clone();
    expected_rev.reverse();
    assert_eq!(rev, expected_rev, "f64 reverse mismatch ({lanes} lanes)");
    assert_eq!(rt_a, a_vals, "f64 round-trip lost the first operand");
    assert_eq!(rt_b, b_vals, "f64 round-trip lost the second operand");

    let concat: Vec<f64> = a_vals.iter().chain(b_vals.iter()).copied().collect();
    let pairs: Vec<&[f64]> = concat.chunks_exact(2).collect();
    let expected_pair_even: Vec<f64> = pairs
        .iter()
        .step_by(2)
        .flat_map(|p| p.iter().copied())
        .collect();
    let expected_pair_odd: Vec<f64> = pairs
        .iter()
        .skip(1)
        .step_by(2)
        .flat_map(|p| p.iter().copied())
        .collect();
    assert_eq!(
        pair_even, expected_pair_even,
        "f64 deinterleave_pairs even mismatch"
    );
    assert_eq!(
        pair_odd, expected_pair_odd,
        "f64 deinterleave_pairs odd mismatch"
    );

    let concat4: Vec<f64> = concat.iter().chain(concat.iter()).copied().collect();
    let pairs4: Vec<&[f64]> = concat4.chunks_exact(2).collect();
    for lane_class in 0..4usize {
        let expected: Vec<f64> = pairs4
            .iter()
            .skip(lane_class)
            .step_by(4)
            .flat_map(|p| p.iter().copied())
            .collect();
        assert_eq!(
            quad[lane_class], expected,
            "f64 deinterleave_pairs4 output {lane_class} mismatch"
        );
    }
}

/// The half concatenation must match the flat reference: the low halves of
/// `a` then `b`, and the high halves of `a` then `b`.
fn check_splat_pair<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let (lo, hi) = (1.5_f32, -2.25_f32);
    let mut out = vec![0.0f32; lanes];

    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        A::store_unaligned(out.as_mut_ptr(), A::splat_pair(lo, hi));
    }

    let expected: Vec<f32> = (0..lanes)
        .map(|i: usize| if i.is_multiple_of(2) { lo } else { hi })
        .collect();
    assert_eq!(out, expected, "splat_pair mismatch ({lanes} lanes)");
}

fn check_splat_pair_f64<A: SimdKernel<f64>>() {
    let lanes = A::LANE_COUNT;
    let (lo, hi) = (0.125_f64, -7.5_f64);
    let mut out = vec![0.0f64; lanes];

    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        A::store_unaligned(out.as_mut_ptr(), A::splat_pair(lo, hi));
    }

    let expected: Vec<f64> = (0..lanes)
        .map(|i: usize| if i.is_multiple_of(2) { lo } else { hi })
        .collect();
    assert_eq!(out, expected, "splat_pair f64 mismatch ({lanes} lanes)");
}

fn check_interleave_halves<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let half = lanes / 2;
    let a_vals: Vec<f32> = (0..lanes).map(|i| (i + 1) as f32).collect();
    let b_vals: Vec<f32> = (0..lanes).map(|i| -((i + 1) as f32) * 10.0).collect();
    let mut lo = vec![0.0f32; lanes];
    let mut hi = vec![0.0f32; lanes];

    // SAFETY: caller gates on the required target features for `A`.
    unsafe {
        let a = A::load_unaligned(a_vals.as_ptr());
        let b = A::load_unaligned(b_vals.as_ptr());
        let (h_lo, h_hi) = A::interleave_halves(a, b);
        A::store_unaligned(lo.as_mut_ptr(), h_lo);
        A::store_unaligned(hi.as_mut_ptr(), h_hi);
    }

    let expected_lo: Vec<f32> = a_vals[..half]
        .iter()
        .chain(&b_vals[..half])
        .copied()
        .collect();
    let expected_hi: Vec<f32> = a_vals[half..]
        .iter()
        .chain(&b_vals[half..])
        .copied()
        .collect();
    assert_eq!(
        lo, expected_lo,
        "interleave_halves low mismatch ({lanes} lanes)"
    );
    assert_eq!(
        hi, expected_hi,
        "interleave_halves high mismatch ({lanes} lanes)"
    );
}

#[test]
fn permutes_match_reference_all_backends() {
    check_permutes::<Scalar>();
    check_interleave_halves::<Scalar>();
    check_splat_pair::<Scalar>();
    check_splat_pair_f64::<Scalar>();
    check_transpose_square::<f32, Scalar>();
    check_transpose_square::<f64, Scalar>();
    check_transpose_interleaved_square::<f32, Scalar>();
    check_transpose_interleaved_square::<f64, Scalar>();
    check_permutes::<SveArch>();
    check_interleave_halves::<SveArch>();
    check_splat_pair::<SveArch>();
    check_splat_pair_f64::<SveArch>();
    check_permutes_f64::<Scalar>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_permutes::<hermes_simd::Avx2>();
            check_interleave_halves::<hermes_simd::Avx2>();
            check_splat_pair::<hermes_simd::Avx2>();
            check_splat_pair_f64::<hermes_simd::Avx2>();
            check_transpose_square::<f32, hermes_simd::Avx2>();
            check_transpose_square::<f64, hermes_simd::Avx2>();
            check_transpose_interleaved_square::<f32, hermes_simd::Avx2>();
            check_transpose_interleaved_square::<f64, hermes_simd::Avx2>();
            check_permutes_f64::<hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_permutes::<hermes_simd::Avx512>();
            check_interleave_halves::<hermes_simd::Avx512>();
            check_splat_pair::<hermes_simd::Avx512>();
            check_splat_pair_f64::<hermes_simd::Avx512>();
            check_transpose_square::<f32, hermes_simd::Avx512>();
            check_transpose_square::<f64, hermes_simd::Avx512>();
            check_transpose_interleaved_square::<f32, hermes_simd::Avx512>();
            check_transpose_interleaved_square::<f64, hermes_simd::Avx512>();
            check_permutes_f64::<hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_permutes::<hermes_simd::Neon>();
        check_interleave_halves::<hermes_simd::Neon>();
        check_splat_pair::<hermes_simd::Neon>();
        check_splat_pair_f64::<hermes_simd::Neon>();
        check_transpose_square::<f32, hermes_simd::Neon>();
        check_transpose_square::<f64, hermes_simd::Neon>();
        check_transpose_interleaved_square::<f32, hermes_simd::Neon>();
        check_transpose_interleaved_square::<f64, hermes_simd::Neon>();
        check_permutes_f64::<hermes_simd::Neon>();
    }
}

/// `masked_sum_reduce` with `leading_k_mask(k)` must sum exactly the first
/// `min(k, LANE_COUNT)` lanes, including the k = 0 and k > `LANE_COUNT` bounds.
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

/// Partial masked memory must access active lanes only, merge inactive loads
/// from `src`, and leave inactive stores and adjacent canaries unchanged.
fn check_partial_masked_memory<T, A>()
where
    T: ScalarElement,
    A: hermes_simd_core::arch::SimdArch + SimdKernel<T>,
{
    let lanes = <A as hermes_simd::SimdStorage<T>>::LANE_COUNT;
    let src_values: Vec<T> = (0..lanes)
        .map(|lane| {
            T::cast_from(
                -1000 - i32::try_from(lane).expect("invariant: a SIMD lane index fits in i32"),
            )
        })
        .collect();
    let stored_values: Vec<T> = (0..lanes)
        .map(|lane| {
            T::cast_from(
                100 + i32::try_from(lane).expect("invariant: a SIMD lane index fits in i32"),
            )
        })
        .collect();
    let mut loaded = vec![T::ZERO; lanes];

    for valid_lanes in 0..=lanes {
        let data: Vec<T> = (0..valid_lanes)
            .map(|lane| T::cast_from(10 + lane as i32))
            .collect();
        let valid_mask = if valid_lanes == u64::BITS as usize {
            u64::MAX
        } else {
            (1_u64 << valid_lanes) - 1
        };
        let alternating = 0x5555_5555_5555_5555_u64 & valid_mask;
        let highest = valid_lanes.checked_sub(1).map_or(0, |lane| 1_u64 << lane);

        for mask_bits in [valid_mask, alternating, highest] {
            let sentinel = T::cast_from(-77);
            let mut destination = vec![sentinel; valid_lanes + 2];

            // SAFETY: the caller gates backend support. Every active bit is
            // below `valid_lanes`; the input and destination expose exactly
            // that prefix. Full-width source/output/value buffers are used only
            // for register construction and observation.
            unsafe {
                let mask = A::mask_from_bitmask(mask_bits);
                let src = A::load_unaligned(src_values.as_ptr());
                let value = A::masked_load_partial(data.as_ptr(), valid_lanes, mask, src);
                A::store_unaligned(loaded.as_mut_ptr(), value);

                let stored = A::load_unaligned(stored_values.as_ptr());
                A::masked_store_partial(destination.as_mut_ptr().add(1), valid_lanes, mask, stored);
            }

            for lane in 0..lanes {
                let expected = if (mask_bits >> lane) & 1 == 1 {
                    data[lane]
                } else {
                    src_values[lane]
                };
                assert_eq!(
                    loaded[lane], expected,
                    "partial load lane {lane}, valid_lanes {valid_lanes}, mask {mask_bits:#x}"
                );
            }
            assert_eq!(destination[0], sentinel, "prefix canary changed");
            assert_eq!(
                destination[valid_lanes + 1],
                sentinel,
                "suffix canary changed"
            );
            for lane in 0..valid_lanes {
                let expected = if (mask_bits >> lane) & 1 == 1 {
                    stored_values[lane]
                } else {
                    sentinel
                };
                assert_eq!(
                    destination[lane + 1],
                    expected,
                    "partial store lane {lane}, valid_lanes {valid_lanes}, mask {mask_bits:#x}"
                );
            }
        }
    }
}

#[test]
fn partial_masked_memory_conforms_on_supported_backends() {
    check_partial_masked_memory::<eunomia::F16, Scalar>();
    check_partial_masked_memory::<f32, Scalar>();
    check_partial_masked_memory::<f64, Scalar>();
    check_partial_masked_memory::<f32, SveArch>();
    check_partial_masked_memory::<f64, SveArch>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_partial_masked_memory::<eunomia::F16, hermes_simd::Avx2>();
            check_partial_masked_memory::<f32, hermes_simd::Avx2>();
            check_partial_masked_memory::<f64, hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_partial_masked_memory::<eunomia::F16, hermes_simd::Avx512>();
            check_partial_masked_memory::<f32, hermes_simd::Avx512>();
            check_partial_masked_memory::<f64, hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_partial_masked_memory::<f32, hermes_simd::Neon>();
        check_partial_masked_memory::<f64, hermes_simd::Neon>();
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

/// Adversarial *non-canonical* mask lane patterns: values whose sign bit and
/// whose nonzero-ness disagree, so a backend testing "differs from zero" (or
/// bit-splicing the raw mask, as NEON `vbsl` would) diverges from the
/// documented sign-bit selection.
const NON_CANONICAL_MASK_PATTERNS: [f32; 8] = [
    2.0,                         // nonzero, sign clear → inactive
    -0.0,                        // zero, sign set → active
    0.0,                         // zero, sign clear → inactive
    f32::from_bits(0x7fc0_0000), // positive NaN → inactive
    f32::from_bits(0xffc0_0000), // negative NaN → active
    -3.5,                        // ordinary negative (not all-ones) → active
    f32::from_bits(!0),          // canonical ALL_ONES → active
    f32::from_bits(0x0000_0001), // positive subnormal → inactive
];

/// `blend` must select by the mask lane's *sign bit* alone — the documented
/// contract — even when the mask is non-canonical. A nonzero-or-NaN test
/// (scalar emulation) or a bitwise splice (NEON `vbsl` on the raw mask)
/// diverges on every pattern above whose sign and nonzero-ness disagree.
fn check_blend_sign_bit_semantics<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let mask_vals: Vec<f32> = (0..lanes)
        .map(|i| NON_CANONICAL_MASK_PATTERNS[i % NON_CANONICAL_MASK_PATTERNS.len()])
        .collect();
    let true_vals: Vec<f32> = (0..lanes).map(|i| (i + 1) as f32).collect();
    let false_vals: Vec<f32> = (0..lanes).map(|i| -((i + 1) as f32)).collect();
    let mut out = vec![0.0f32; lanes];

    // SAFETY: every buffer holds exactly `LANE_COUNT` elements; caller gates on
    // the required target features for `A`.
    unsafe {
        let mask = A::load_unaligned(mask_vals.as_ptr());
        let selected = A::load_unaligned(true_vals.as_ptr());
        let rejected = A::load_unaligned(false_vals.as_ptr());
        A::store_unaligned(out.as_mut_ptr(), A::blend(mask, selected, rejected));
    }

    for (i, &got) in out.iter().enumerate() {
        let want = if mask_vals[i].is_sign_negative() {
            true_vals[i]
        } else {
            false_vals[i]
        };
        assert_eq!(
            got,
            want,
            "blend lane {i} must select by sign bit (mask {:#010x})",
            mask_vals[i].to_bits()
        );
    }
}

/// `Vector::to_bitmask` documents "(sign bits)": each bit must equal the
/// mask lane's sign bit, non-canonical lanes included.
fn check_vector_to_bitmask_sign_bit<A>()
where
    A: hermes_simd_core::arch::SimdArch + SimdKernel<f32>,
{
    let lanes = A::LANE_COUNT;
    let mask_vals: Vec<f32> = (0..lanes)
        .map(|i| NON_CANONICAL_MASK_PATTERNS[i % NON_CANONICAL_MASK_PATTERNS.len()])
        .collect();
    let vector = hermes_simd::Vector::<f32, A>::load_unaligned_from_slice(&mask_vals)
        .expect("mask buffer holds one register");
    let bm = vector.to_bitmask().0;
    for (i, &val) in mask_vals.iter().enumerate() {
        assert_eq!(
            (bm >> i) & 1 == 1,
            val.is_sign_negative(),
            "to_bitmask lane {i} must report the sign bit (value {:#010x})",
            val.to_bits()
        );
    }
}

/// Uniform fused multiply-subtract must match the scalar single-rounding
/// contract lane for lane. Inputs stay finite and bounded, so exact bit
/// equality is the correct oracle for every native and emulated backend.
fn check_fmsub_matches_scalar<A: SimdKernel<f32>>(vals: &[f32]) {
    let lanes = A::LANE_COUNT;
    let a: Vec<f32> = (0..lanes).map(|i| vals[i % vals.len()]).collect();
    let b: Vec<f32> = (0..lanes)
        .map(|i| vals[(i * 5 + 1) % vals.len()] * 0.03125)
        .collect();
    let c: Vec<f32> = (0..lanes)
        .map(|i| vals[(i * 7 + 2) % vals.len()] * 0.015_625)
        .collect();
    let mut out = vec![0.0; lanes];

    // SAFETY: caller gates on the target features for `A`, and all buffers
    // contain exactly one complete lane group.
    unsafe {
        let result = A::fmsub(
            A::load_unaligned(a.as_ptr()),
            A::load_unaligned(b.as_ptr()),
            A::load_unaligned(c.as_ptr()),
        );
        A::store_unaligned(out.as_mut_ptr(), result);
    }

    for i in 0..lanes {
        assert_eq!(
            out[i].to_bits(),
            a[i].mul_add(b[i], -c[i]).to_bits(),
            "fmsub lane {i} must preserve the fused single-rounding contract"
        );
    }
}

/// `fmaddsub`/`fmsubadd` must preserve the fused single-rounding contract
/// lane for lane: `a*b ∓ c` with one rounding and an exactly negated addend,
/// per lane parity. A multiply-then-add default would round twice and
/// diverge from the hardware `vfmaddsub` backends on rounding-sensitive
/// inputs like these.
fn check_alternating_fma_fused<A: SimdKernel<f32>>(vals: &[f32]) {
    let lanes = A::LANE_COUNT;
    let a: Vec<f32> = (0..lanes).map(|i| vals[i % vals.len()]).collect();
    let b: Vec<f32> = (0..lanes)
        .map(|i| vals[(i * 5 + 1) % vals.len()] * 0.03125)
        .collect();
    let c: Vec<f32> = (0..lanes)
        .map(|i| vals[(i * 7 + 2) % vals.len()] * 0.015_625)
        .collect();
    let mut maddsub = vec![0.0; lanes];
    let mut msubadd = vec![0.0; lanes];

    // SAFETY: caller gates on the target features for `A`, and all buffers
    // contain exactly one complete lane group.
    unsafe {
        let av = A::load_unaligned(a.as_ptr());
        let bv = A::load_unaligned(b.as_ptr());
        let cv = A::load_unaligned(c.as_ptr());
        A::store_unaligned(maddsub.as_mut_ptr(), A::fmaddsub(av, bv, cv));
        A::store_unaligned(msubadd.as_mut_ptr(), A::fmsubadd(av, bv, cv));
    }

    for i in 0..lanes {
        let (want_maddsub, want_msubadd) = if i % 2 == 0 {
            (a[i].mul_add(b[i], -c[i]), a[i].mul_add(b[i], c[i]))
        } else {
            (a[i].mul_add(b[i], c[i]), a[i].mul_add(b[i], -c[i]))
        };
        assert_eq!(
            maddsub[i].to_bits(),
            want_maddsub.to_bits(),
            "fmaddsub lane {i} must be fused with an exactly negated addend"
        );
        assert_eq!(
            msubadd[i].to_bits(),
            want_msubadd.to_bits(),
            "fmsubadd lane {i} must be fused with an exactly negated addend"
        );
    }
}

/// The f64 operation used by planar FFT kernels must preserve the same fused,
/// single-rounding contract on every backend that this host can execute.
fn check_fmsub_f64_matches_scalar<A: SimdKernel<f64>>() {
    let lanes = <A as hermes_simd::SimdStorage<f64>>::LANE_COUNT;
    let seeds = [
        -17.25_f64,
        -0.125,
        f64::from_bits(0x3ff0_0000_0000_0001),
        3.75,
        257.5,
    ];
    let a: Vec<f64> = (0..lanes).map(|i| seeds[i % seeds.len()]).collect();
    let b: Vec<f64> = (0..lanes)
        .map(|i| seeds[(i * 3 + 1) % seeds.len()] * 0.03125)
        .collect();
    let c: Vec<f64> = (0..lanes)
        .map(|i| seeds[(i * 2 + 2) % seeds.len()] * 0.015_625)
        .collect();
    let mut out = vec![0.0; lanes];

    // SAFETY: the caller invokes this helper only for backends supported by
    // the current host, and each buffer contains one complete lane group.
    unsafe {
        let result = A::fmsub(
            A::load_unaligned(a.as_ptr()),
            A::load_unaligned(b.as_ptr()),
            A::load_unaligned(c.as_ptr()),
        );
        A::store_unaligned(out.as_mut_ptr(), result);
    }

    for i in 0..lanes {
        assert_eq!(
            out[i].to_bits(),
            a[i].mul_add(b[i], -c[i]).to_bits(),
            "f64 fmsub lane {i} must preserve the fused single-rounding contract"
        );
    }
}

#[test]
fn fmsub_f64_matches_scalar_on_supported_backends() {
    check_fmsub_f64_matches_scalar::<Scalar>();
    check_fmsub_f64_matches_scalar::<SveArch>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_fmsub_f64_matches_scalar::<hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_fmsub_f64_matches_scalar::<hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_fmsub_f64_matches_scalar::<hermes_simd::Neon>();
    }
}

fn check_neg_preserves_bits_f32<A: SimdKernel<f32>>() {
    let seeds = [
        0.0_f32,
        -0.0,
        1.25,
        -7.5,
        f32::from_bits(0x7fc1_2345),
        f32::INFINITY,
    ];
    let values: Vec<f32> = (0..A::LANE_COUNT)
        .map(|index| seeds[index % seeds.len()])
        .collect();
    let mut output = vec![0.0; A::LANE_COUNT];
    // SAFETY: the caller gates on host support, and both buffers contain one
    // complete lane group for `A`.
    unsafe {
        let vector = A::load_unaligned(values.as_ptr());
        A::store_unaligned(output.as_mut_ptr(), A::neg(vector));
    }
    for (actual, source) in output.iter().zip(&values) {
        assert_eq!(actual.to_bits(), source.to_bits() ^ 0x8000_0000);
    }
}

fn check_neg_preserves_bits_f64<A: SimdKernel<f64>>() {
    let lanes = <A as hermes_simd::SimdStorage<f64>>::LANE_COUNT;
    let seeds = [
        0.0_f64,
        -0.0,
        1.25,
        -7.5,
        f64::from_bits(0x7ff8_0000_0001_2345),
        f64::INFINITY,
    ];
    let values: Vec<f64> = (0..lanes).map(|index| seeds[index % seeds.len()]).collect();
    let mut output = vec![0.0; lanes];
    // SAFETY: the caller gates on host support, and both buffers contain one
    // complete lane group for `A`.
    unsafe {
        let vector = A::load_unaligned(values.as_ptr());
        A::store_unaligned(output.as_mut_ptr(), A::neg(vector));
    }
    for (actual, source) in output.iter().zip(&values) {
        assert_eq!(actual.to_bits(), source.to_bits() ^ 0x8000_0000_0000_0000);
    }
}

#[test]
fn neg_preserves_sign_bit_on_supported_backends() {
    check_neg_preserves_bits_f32::<Scalar>();
    check_neg_preserves_bits_f32::<SveArch>();
    check_neg_preserves_bits_f64::<Scalar>();
    check_neg_preserves_bits_f64::<SveArch>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_neg_preserves_bits_f32::<hermes_simd::Avx2>();
            check_neg_preserves_bits_f64::<hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_neg_preserves_bits_f32::<hermes_simd::Avx512>();
            check_neg_preserves_bits_f64::<hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_neg_preserves_bits_f32::<hermes_simd::Neon>();
        check_neg_preserves_bits_f64::<hermes_simd::Neon>();
    }
}

/// Deterministic adversarial run of the sign-bit mask-contract checks on
/// every backend the host can execute: the mask-active criterion is defined
/// once (sign bit), so scalar-emulated and native backends must agree on the
/// non-canonical patterns.
#[test]
fn blend_and_to_bitmask_honor_sign_bit_on_supported_backends() {
    check_blend_sign_bit_semantics::<Scalar>();
    check_blend_sign_bit_semantics::<SveArch>();
    check_vector_to_bitmask_sign_bit::<Scalar>();
    check_vector_to_bitmask_sign_bit::<SveArch>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_blend_sign_bit_semantics::<hermes_simd::Avx2>();
            check_vector_to_bitmask_sign_bit::<hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_blend_sign_bit_semantics::<hermes_simd::Avx512>();
            check_vector_to_bitmask_sign_bit::<hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_blend_sign_bit_semantics::<hermes_simd::Neon>();
        check_vector_to_bitmask_sign_bit::<hermes_simd::Neon>();
    }
}

/// Run every kernel-level check for one backend.
fn check_all_kernel_invariants<A>(bm: u64, vals: &[f32])
where
    A: hermes_simd_core::arch::SimdArch + SimdKernel<f32>,
{
    check_bitmask_roundtrip::<A>(bm);
    check_mask_from_bitmask_matches_bools::<A>(bm);
    check_vector_to_mask_roundtrip::<A>(bm);
    check_vector_to_mask_matches_cmp::<A>(vals);
    check_cmp_ne_complements_cmp_eq::<A>(vals);
    check_blend_honors_canonical_mask::<A>(bm);
    check_fmsub_matches_scalar::<A>(vals);
    check_alternating_fma_fused::<A>(vals);
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

/// Elementwise rounding must match the plain-slice reference bit-exactly on
/// every backend: `round` (ties to the even neighbor), `floor`, `ceil`, and
/// `trunc`, plus the NaN/±Inf/signed-zero contract. The reference is written on
/// plain `f32` scalars, independent of any lane arithmetic in the kernel
/// defaults, so a native override and the default it replaces are both checked
/// against the same external specification.
///
/// The case list covers every contract edge: exact halfway values (in both
/// signs, where the tie must resolve to the even neighbor), values straddling a
/// tie by the smallest representable margin, large magnitudes whose fractional
/// part is representable, magnitudes far beyond the integer range (identity),
/// subnormals, infinities, and a quiet NaN. The `-0.5..` half-grid sweep pushes
/// tie resolution across the whole lane count.
fn check_rounding_f32<A: SimdKernel<f32>>() {
    let lanes = A::LANE_COUNT;
    let mut cases: Vec<f32> = vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.5,
        -3.5,
        0.49,
        -0.49,
        1.499_999_9,
        1.500_000_1,
        -1.499_999_9,
        -1.500_000_1,
        1_048_576.5,
        -1_048_576.5,
        1.0e20,
        3.0e38,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for i in 0..(3 * lanes) {
        cases.push((i as f32) * 0.5 - (1.5 * lanes as f32));
    }

    let mut floor_out = vec![0.0f32; lanes];
    let mut ceil_out = vec![0.0f32; lanes];
    let mut round_out = vec![0.0f32; lanes];
    let mut trunc_out = vec![0.0f32; lanes];

    for chunk in cases.chunks(lanes) {
        let mut v = vec![0.0f32; lanes];
        v[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: caller gates on the required target features for `A`; every
        // buffer covers exactly `LANE_COUNT` elements, so the unaligned loads
        // and stores stay in bounds.
        unsafe {
            let x = A::load_unaligned(v.as_ptr());
            A::store_unaligned(floor_out.as_mut_ptr(), A::floor(x));
            A::store_unaligned(ceil_out.as_mut_ptr(), A::ceil(x));
            A::store_unaligned(round_out.as_mut_ptr(), A::round(x));
            A::store_unaligned(trunc_out.as_mut_ptr(), A::trunc(x));
        }
        for i in 0..lanes {
            let y = v[i];
            // Bitwise equality: `==` treats NaN as unequal and merges ±0,
            // hiding exactly the special-value behavior this contract pins.
            assert_eq!(
                floor_out[i].to_bits(),
                y.floor().to_bits(),
                "f32 floor({y:e})"
            );
            assert_eq!(ceil_out[i].to_bits(), y.ceil().to_bits(), "f32 ceil({y:e})");
            assert_eq!(
                round_out[i].to_bits(),
                f32::round_ties_even(y).to_bits(),
                "f32 round({y:e})"
            );
            assert_eq!(
                trunc_out[i].to_bits(),
                y.trunc().to_bits(),
                "f32 trunc({y:e})"
            );
        }
    }
}

/// The f64 rounding path is a separate monomorphization (4 lanes on AVX2,
/// `vroundpd` instead of `vroundps`), so it needs its own coverage, including
/// a tie at the top of the exact-representable range.
fn check_rounding_f64<A: SimdKernel<f64>>() {
    let lanes = A::LANE_COUNT;
    let mut cases: Vec<f64> = vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        2.5,
        -2.5,
        4_503_599_627_370_497.5, // 2^52 + 0.5: tie at the integer-range edge
        -4_503_599_627_370_497.5,
        1.0e20,
        1.0e300,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    for i in 0..(3 * lanes) {
        cases.push((i as f64) * 0.5 - (1.5 * lanes as f64));
    }

    let mut floor_out = vec![0.0f64; lanes];
    let mut ceil_out = vec![0.0f64; lanes];
    let mut round_out = vec![0.0f64; lanes];
    let mut trunc_out = vec![0.0f64; lanes];

    for chunk in cases.chunks(lanes) {
        let mut v = vec![0.0f64; lanes];
        v[..chunk.len()].copy_from_slice(chunk);
        // SAFETY: caller gates on the required target features for `A`; every
        // buffer covers exactly `LANE_COUNT` elements.
        unsafe {
            let x = A::load_unaligned(v.as_ptr());
            A::store_unaligned(floor_out.as_mut_ptr(), A::floor(x));
            A::store_unaligned(ceil_out.as_mut_ptr(), A::ceil(x));
            A::store_unaligned(round_out.as_mut_ptr(), A::round(x));
            A::store_unaligned(trunc_out.as_mut_ptr(), A::trunc(x));
        }
        for i in 0..lanes {
            let y = v[i];
            assert_eq!(
                floor_out[i].to_bits(),
                y.floor().to_bits(),
                "f64 floor({y:e})"
            );
            assert_eq!(ceil_out[i].to_bits(), y.ceil().to_bits(), "f64 ceil({y:e})");
            assert_eq!(
                round_out[i].to_bits(),
                f64::round_ties_even(y).to_bits(),
                "f64 round({y:e})"
            );
            assert_eq!(
                trunc_out[i].to_bits(),
                y.trunc().to_bits(),
                "f64 trunc({y:e})"
            );
        }
    }
}

#[test]
fn rounding_matches_reference_all_backends() {
    check_rounding_f32::<Scalar>();
    check_rounding_f64::<Scalar>();
    check_rounding_f32::<SveArch>();
    check_rounding_f64::<SveArch>();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            check_rounding_f32::<hermes_simd::Avx2>();
            check_rounding_f64::<hermes_simd::Avx2>();
        }
        if std::is_x86_feature_detected!("avx512f") {
            check_rounding_f32::<hermes_simd::Avx512>();
            check_rounding_f64::<hermes_simd::Avx512>();
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_rounding_f32::<hermes_simd::Neon>();
        check_rounding_f64::<hermes_simd::Neon>();
    }
}

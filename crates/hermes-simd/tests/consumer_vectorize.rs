//! Consumer-shaped conformance for [`hermes_simd::vectorize`].
//!
//! This file stands in for a downstream crate. It is deliberately written the
//! way a consumer would write it, and the crate-level `forbid(unsafe_code)`
//! below is the assertion that matters: if reaching Hermes' lane surface from
//! outside required `unsafe`, this file would not compile. Before `vectorize`
//! existed, a consumer wanting per-ISA code for its own kernel had to write
//! `#[target_feature]` trampolines, which it cannot do here.
//!
//! The kernel exercised is a radix-2 butterfly over planar real and imaginary
//! planes — the shape that motivated the entry, and one that needs the fused
//! multiply-add rather than only elementwise arithmetic.

#![forbid(unsafe_code)]

use hermes_simd::{
    BitMask, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, TargetId,
};

#[path = "consumer_vectorize/exact_lanes.rs"]
mod exact_lanes;

/// Input planes for one butterfly stage.
struct Planes {
    re0: Vec<f32>,
    im0: Vec<f32>,
    re1: Vec<f32>,
    im1: Vec<f32>,
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
}

/// Output planes: `out0.re`, `out0.im`, `out1.re`, `out1.im`.
struct Butterflies {
    o0r: Vec<f32>,
    o0i: Vec<f32>,
    o1r: Vec<f32>,
    o1i: Vec<f32>,
}

impl Planes {
    /// Deterministic, mutually incommensurate tones, so no two planes are
    /// accidentally equal and a swapped-plane defect cannot pass.
    fn new(n: usize) -> Self {
        let tone = |scale: f32, phase: f32| -> Vec<f32> {
            (0..n).map(|i| (scale * i as f32 + phase).sin()).collect()
        };
        Self {
            re0: tone(0.017, 0.0),
            im0: tone(0.031, 0.4),
            re1: tone(0.023, 0.9),
            im1: tone(0.041, 1.3),
            tw_re: tone(0.011, 2.1),
            tw_im: tone(0.037, 2.7),
        }
    }

    fn len(&self) -> usize {
        self.re0.len()
    }

    /// The same butterfly evaluated one element at a time.
    fn reference(&self) -> Butterflies {
        let n = self.len();
        let mut out = Butterflies {
            o0r: vec![0.0; n],
            o0i: vec![0.0; n],
            o1r: vec![0.0; n],
            o1i: vec![0.0; n],
        };
        for j in 0..n {
            let (tr, ti) = self.twiddled(j);
            out.o0r[j] = self.re0[j] + tr;
            out.o0i[j] = self.im0[j] + ti;
            out.o1r[j] = self.re0[j] - tr;
            out.o1i[j] = self.im0[j] - ti;
        }
        out
    }

    /// `w * b` at element `j`, each component one fused multiply-add.
    fn twiddled(&self, j: usize) -> (f32, f32) {
        (
            self.tw_re[j].mul_add(self.re1[j], -(self.tw_im[j] * self.im1[j])),
            self.tw_re[j].mul_add(self.im1[j], self.tw_im[j] * self.re1[j]),
        )
    }
}

/// One radix-2 decimation-in-time butterfly stage over planar f32 data.
///
/// For each `k`: `out0 = in0 + w * in1`, `out1 = in0 - w * in1`, in complex
/// arithmetic with real and imaginary parts held in separate planes.
struct ButterflyStage<'a>(&'a Planes);

impl LaneKernel<f32> for ButterflyStage<'_> {
    type Output = Butterflies;

    fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) -> Self::Output {
        let p = self.0;
        let n = p.len();
        let mut out = Butterflies {
            o0r: vec![0.0; n],
            o0i: vec![0.0; n],
            o1r: vec![0.0; n],
            o1i: vec![0.0; n],
        };

        let mut chunks = simd.io_chunks(
            [
                p.re0.as_slice(),
                p.im0.as_slice(),
                p.re1.as_slice(),
                p.im1.as_slice(),
                p.tw_re.as_slice(),
                p.tw_im.as_slice(),
            ],
            [
                out.o0r.as_mut_slice(),
                out.o0i.as_mut_slice(),
                out.o1r.as_mut_slice(),
                out.o1i.as_mut_slice(),
            ],
        );
        for ([a_re, a_im, b_re, b_im, w_re, w_im], [mut o0r, mut o0i, mut o1r, mut o1i]) in
            &mut chunks
        {
            let a_re = a_re.load();
            let a_im = a_im.load();
            let b_re = b_re.load();
            let b_im = b_im.load();
            let w_re = w_re.load();
            let w_im = w_im.load();
            let t_re = w_re.mul_add(b_re, -(w_im * b_im));
            let t_im = w_re.mul_add(b_im, w_im * b_re);
            o0r.store(a_re + t_re);
            o0i.store(a_im + t_im);
            o1r.store(a_re - t_re);
            o1i.store(a_im - t_im);
        }

        // Scalar tail, same arithmetic, so the comparison covers every element
        // rather than only the vectorized prefix.
        let ([re0, im0, re1, im1, tw_re, tw_im], [o0r, o0i, o1r, o1i]) = chunks.into_remainders();
        for j in 0..re0.len() {
            let tr = tw_re[j].mul_add(re1[j], -(tw_im[j] * im1[j]));
            let ti = tw_re[j].mul_add(im1[j], tw_im[j] * re1[j]);
            o0r[j] = re0[j] + tr;
            o0i[j] = im0[j] + ti;
            o1r[j] = re0[j] - tr;
            o1i[j] = im0[j] - ti;
        }
        out
    }
}

/// The whole point: a consumer kernel under `forbid(unsafe_code)` reaches the
/// lane surface, and its result matches the scalar reference on every element.
///
/// Lengths straddle every lane count Hermes ships (1, 4, 8, 16), so each case
/// exercises a different vector-body-plus-tail split.
#[test]
fn vectorized_butterfly_matches_scalar_reference() {
    for n in [0usize, 1, 3, 7, 8, 15, 16, 17, 64, 129] {
        let planes = Planes::new(n);
        let got = hermes_simd::vectorize(ButterflyStage(&planes));
        let want = planes.reference();
        // Reduction order is identical here — the same operations in the same
        // sequence, only wider — so equality is the correct oracle and a
        // tolerance would hide a real lane-placement defect.
        assert_eq!(got.o0r, want.o0r, "out0.re mismatch at n={n}");
        assert_eq!(got.o0i, want.o0i, "out0.im mismatch at n={n}");
        assert_eq!(got.o1r, want.o1r, "out1.re mismatch at n={n}");
        assert_eq!(got.o1i, want.o1i, "out1.im mismatch at n={n}");
    }
}

/// The dispatch ladder must land on a backend this host can execute.
///
/// A ladder that fell through to an unsupported target would fault rather than
/// return, so reaching the assertion at all is part of the check; the assertion
/// pins that the scalar backend is always available as the ladder's floor.
#[test]
fn dispatch_selects_a_host_supported_backend() {
    assert!(
        TargetId::supported_on_host().contains(&TargetId::Scalar),
        "the scalar backend is the dispatch floor and must always be supported"
    );
    let planes = Planes::new(32);
    let _ = hermes_simd::vectorize(ButterflyStage(&planes));
}

/// A lane-parallel reduction through the entry agrees with a sequential sum
/// within the bound reassociation allows.
#[test]
fn dispatched_reduction_agrees_with_sequential_sum() {
    struct Sum<'a>(&'a [f32]);

    impl LaneKernel<f32> for Sum<'_> {
        type Output = f32;

        fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) -> f32 {
            let mut acc = simd.zero();
            let view = simd.view(self.0);
            let mut chunks = view.simd_chunks();
            for chunk in &mut chunks {
                acc = acc + chunk.load();
            }
            acc.sum_reduce() + chunks.remainder().iter().sum::<f32>()
        }
    }

    let data: Vec<f32> = (0..256).map(|i| (0.019 * i as f32).sin()).collect();
    let dispatched = hermes_simd::vectorize(Sum(&data));
    let sequential: f32 = data.iter().sum();

    // Lane-parallel accumulation reassociates the additions, so bitwise
    // equality is not a valid oracle. The bound is the standard O(log n * u)
    // growth for tree-shaped accumulation over 256 f32 values, scaled by the
    // magnitude sum, with room for the differing lane counts across backends.
    let bound = 64.0 * f32::EPSILON * data.iter().map(|v| v.abs()).sum::<f32>();
    assert!(
        (dispatched - sequential).abs() <= bound,
        "dispatched sum {dispatched} differs from sequential {sequential} by more than {bound}"
    );
}

/// Token-scoped constants and masks must not force a consumer to re-enter a
/// checked standalone constructor or use unsafe code inside its lane kernel.
#[test]
fn capability_constructs_constants_and_masks() {
    struct Constructors;

    impl LaneKernel<f64> for Constructors {
        type Output = (Vec<f64>, u64);

        fn call<A: SimdArch + SimdKernel<f64>>(self, simd: Simd<f64, A>) -> Self::Output {
            let lanes = <A as SimdStorage<f64>>::LANE_COUNT;
            let valid_bits = if lanes == 64 {
                u64::MAX
            } else {
                (1_u64 << lanes) - 1
            };
            let expected_bits = 0xA5A5_A5A5_A5A5_A5A5 & valid_bits;
            let actual_bits = simd
                .mask_from_bitmask(BitMask::<64>(expected_bits))
                .to_bitmask()
                .0;

            let mut constants = vec![0.0; lanes];
            (simd.splat(2.0) + simd.zero())
                .store_unaligned_to_slice(&mut constants)
                .expect("output has exactly one complete lane group");
            (constants, actual_bits)
        }
    }

    let (constants, bits) = hermes_simd::vectorize(Constructors);
    assert!(constants
        .iter()
        .all(|&value| value.to_bits() == 2.0_f64.to_bits()));
    let valid_bits = if constants.len() == 64 {
        u64::MAX
    } else {
        (1_u64 << constants.len()) - 1
    };
    assert_eq!(bits, 0xA5A5_A5A5_A5A5_A5A5 & valid_bits);
}

fn assert_pairwise_sums(actual: &[f32], left: &[f32], right: &[f32]) {
    for ((actual, left), right) in actual.iter().zip(left).zip(right) {
        assert_eq!(actual.to_bits(), (*left + *right).to_bits());
    }
}

fn assert_filled_with(actual: &[f32], expected: f32) {
    assert!(actual
        .iter()
        .all(|value| value.to_bits() == expected.to_bits()));
}

#[derive(Clone, Copy)]
enum IoIteration {
    Complete,
    OneChunk,
}

struct IoExercise<'a> {
    left: &'a [f32],
    right: &'a [f32],
    output: &'a mut [f32],
    iteration: IoIteration,
}

impl LaneKernel<f32> for IoExercise<'_> {
    type Output = (usize, usize, usize, [usize; 2], usize);

    fn call<A: SimdArch + SimdKernel<f32>>(self, simd: Simd<f32, A>) -> Self::Output {
        let mut chunks = simd.io_chunks([self.left, self.right], [self.output]);
        let initial = chunks.len();
        match self.iteration {
            IoIteration::Complete => {
                for ([left, right], [mut output]) in &mut chunks {
                    output.store(left.load() + right.load());
                }
            }
            IoIteration::OneChunk => {
                if let Some(([left, right], [mut output])) = chunks.next() {
                    output.store(left.load() + right.load());
                }
            }
        }
        let remaining = chunks.chunks_remaining();
        let ([left, right], [output]) = chunks.into_remainders();
        let tail_lengths = [left.len(), right.len()];
        let output_tail_length = output.len();
        if let IoIteration::Complete = self.iteration {
            let common = left.len().min(right.len()).min(output.len());
            for index in 0..common {
                output[index] = left[index] + right[index];
            }
        }
        (
            <A as SimdStorage<f32>>::LANE_COUNT,
            initial,
            remaining,
            tail_lengths,
            output_tail_length,
        )
    }
}

fn exercise_io_chunks(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    iteration: IoIteration,
) -> (usize, usize, usize, [usize; 2], usize) {
    hermes_simd::vectorize(IoExercise {
        left,
        right,
        output,
        iteration,
    })
}

/// Planar I/O chunking must use one shortest-plane limit, expose exact iterator
/// length, preserve mutable writes, and return every unprocessed suffix.
#[test]
fn io_chunks_preserve_unequal_planes_and_tails() {
    let left: Vec<f32> = (0..70).map(|index| index as f32 + 0.25).collect();
    let right: Vec<f32> = (0..67).map(|index| 2.0 * index as f32 - 0.5).collect();
    let mut output = vec![-1_000.0; 69];
    let (lanes, initial, remaining, tails, output_tail) =
        exercise_io_chunks(&left, &right, &mut output, IoIteration::Complete);
    let vectorized = (67 / lanes) * lanes;
    assert_eq!(initial, 67 / lanes);
    assert_eq!(remaining, 0);
    assert_eq!(tails, [70 - vectorized, 67 - vectorized]);
    assert_eq!(output_tail, 69 - vectorized);
    assert_pairwise_sums(&output[..67], &left[..67], &right);
    assert_filled_with(&output[67..], -1_000.0);

    let mut zero_output = [-1_000.0; 2];
    let zero = exercise_io_chunks(
        &[],
        &[2.0, 3.0, 4.0],
        &mut zero_output,
        IoIteration::Complete,
    );
    assert_eq!(zero, (lanes, 0, 0, [0, 3], 2));
    assert_filled_with(&zero_output, -1_000.0);

    let short_len = lanes.saturating_sub(1);
    let short_left = vec![1.0; short_len];
    let short_right = vec![2.0; short_len + 2];
    let mut short_output = vec![-1_000.0; short_len + 1];
    let short = exercise_io_chunks(
        &short_left,
        &short_right,
        &mut short_output,
        IoIteration::Complete,
    );
    assert_eq!(
        short,
        (lanes, 0, 0, [short_len, short_len + 2], short_len + 1)
    );
    assert_filled_with(&short_output[..short_len], 3.0);
    assert_eq!(short_output[short_len].to_bits(), (-1_000.0_f32).to_bits());

    let mut early_output = vec![-1_000.0; 69];
    let early = exercise_io_chunks(&left, &right, &mut early_output, IoIteration::OneChunk);
    assert_eq!(
        early,
        (
            lanes,
            67 / lanes,
            67 / lanes - 1,
            [70 - lanes, 67 - lanes],
            69 - lanes
        )
    );
    assert_pairwise_sums(&early_output[..lanes], &left[..lanes], &right[..lanes]);
    assert_filled_with(&early_output[lanes..], -1_000.0);
}

/// A consumer kernel generic over the scalar type must reach the entry.
///
/// This is the case `LaneScalar` exists for, and it is a regression test rather
/// than a coverage box: the dispatch ladder names concrete backends, so without
/// that trait a `T`-generic caller has to prove `Avx2: SimdKernel<T>` and the
/// rest itself, and the whole cfg-gated backend list leaks into its signature.
/// Apollo hit exactly this as the first consumer. If `LaneScalar` is ever
/// simplified away, this file stops compiling.
#[test]
fn generic_consumer_kernel_compiles_and_runs() {
    struct Doubling<'a, T: LaneScalar>(&'a mut [T]);

    impl<T> LaneKernel<T> for Doubling<'_, T>
    where
        T: LaneScalar + core::ops::Add<Output = T>,
    {
        type Output = ();

        fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
            let view = simd.view_mut(self.0);
            let mut chunks = view.simd_chunks_mut();
            for mut chunk in &mut chunks {
                let vector = chunk.load();
                chunk.store(vector + vector);
            }
            for value in chunks.into_remainder() {
                *value = *value + *value;
            }
        }
    }

    // The caller is itself generic, which is the shape that failed before
    // `LaneScalar`: it never names a backend.
    fn double_all<T>(data: &mut [T])
    where
        T: LaneScalar + core::ops::Add<Output = T>,
    {
        hermes_simd::vectorize(Doubling(data));
    }

    // Doubling a small integer-valued float is exact in binary floating point,
    // so equality is the correct oracle here rather than a tolerance. The
    // comparison goes through slices because `assert_eq!` on float *arrays* is
    // what `clippy::float_cmp` flags, and silencing that lint would suppress it
    // for the cases where it is right.
    let mut a32 = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    double_all(&mut a32);
    assert_eq!(a32.as_slice(), [2.0f32, 4.0, 6.0, 8.0, 10.0].as_slice());

    let mut a64 = [1.0f64, 2.0, 3.0];
    double_all(&mut a64);
    assert_eq!(a64.as_slice(), [2.0f64, 4.0, 6.0].as_slice());

    let mut abf16 = [
        eunomia::Bf16::from_f32(1.0),
        eunomia::Bf16::from_f32(2.0),
        eunomia::Bf16::from_f32(3.0),
    ];
    double_all(&mut abf16);
    assert_eq!(
        abf16.map(eunomia::Bf16::to_bits),
        [
            eunomia::Bf16::from_f32(2.0).to_bits(),
            eunomia::Bf16::from_f32(4.0).to_bits(),
            eunomia::Bf16::from_f32(6.0).to_bits(),
        ]
    );
}

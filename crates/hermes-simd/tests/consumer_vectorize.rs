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

use hermes_simd::{LaneKernel, LaneScalar, SimdArch, SimdKernel, SimdStorage, TargetId, Vector};

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

    fn call<A: SimdArch + SimdKernel<f32>>(self) -> Self::Output {
        let p = self.0;
        let n = p.len();
        let lanes = <A as SimdStorage<f32>>::LANE_COUNT;
        let mut out = Butterflies {
            o0r: vec![0.0; n],
            o0i: vec![0.0; n],
            o1r: vec![0.0; n],
            o1i: vec![0.0; n],
        };

        let load = |s: &[f32], r: core::ops::Range<usize>| {
            Vector::<f32, A>::load_unaligned_from_slice(&s[r]).unwrap()
        };

        let mut i = 0;
        while i + lanes <= n {
            let r = i..i + lanes;
            let a_re = load(&p.re0, r.clone());
            let a_im = load(&p.im0, r.clone());
            let b_re = load(&p.re1, r.clone());
            let b_im = load(&p.im1, r.clone());
            let w_re = load(&p.tw_re, r.clone());
            let w_im = load(&p.tw_im, r.clone());

            let t_re = w_re.mul_add(b_re, -(w_im * b_im));
            let t_im = w_re.mul_add(b_im, w_im * b_re);

            let store = |v: Vector<f32, A>, dst: &mut [f32]| {
                v.store_unaligned_to_slice(dst).unwrap();
            };
            store(a_re + t_re, &mut out.o0r[r.clone()]);
            store(a_im + t_im, &mut out.o0i[r.clone()]);
            store(a_re - t_re, &mut out.o1r[r.clone()]);
            store(a_im - t_im, &mut out.o1i[r]);
            i += lanes;
        }

        // Scalar tail, same arithmetic, so the comparison covers every element
        // rather than only the vectorized prefix.
        for j in i..n {
            let (tr, ti) = p.twiddled(j);
            out.o0r[j] = p.re0[j] + tr;
            out.o0i[j] = p.im0[j] + ti;
            out.o1r[j] = p.re0[j] - tr;
            out.o1i[j] = p.im0[j] - ti;
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

        fn call<A: SimdArch + SimdKernel<f32>>(self) -> f32 {
            let lanes = <A as SimdStorage<f32>>::LANE_COUNT;
            let mut acc = Vector::<f32, A>::zero();
            let mut i = 0;
            while i + lanes <= self.0.len() {
                let chunk =
                    Vector::<f32, A>::load_unaligned_from_slice(&self.0[i..i + lanes]).unwrap();
                acc = acc + chunk;
                i += lanes;
            }
            acc.sum_reduce() + self.0[i..].iter().sum::<f32>()
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

        fn call<A: SimdArch + SimdKernel<T>>(self) {
            let lanes = <A as SimdStorage<T>>::LANE_COUNT;
            let n = self.0.len();
            let mut i = 0;
            while i + lanes <= n {
                let span = i..i + lanes;
                let v = Vector::<T, A>::load_unaligned_from_slice(&self.0[span.clone()]).unwrap();
                (v + v).store_unaligned_to_slice(&mut self.0[span]).unwrap();
                i += lanes;
            }
            for j in i..n {
                self.0[j] = self.0[j] + self.0[j];
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
}

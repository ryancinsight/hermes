//! Exact-lane consumer dispatch: the cost of the scalar fallback's frame.
//!
//! `vectorize_lanes::<N, T>` falls back to the portable scalar backend when no
//! ISA backend carries exactly `N` lanes of `T`. Whether that fallback body
//! runs inside an AVX2+FMA `#[target_feature]` frame is invisible to every
//! other instrument in this suite: the backend, the lane count, and every
//! value are identical either way, so only instruction selection moves. No
//! existing group reaches the exact-lane entry at all.
//!
//! The workload is a short serial `mul_add` chain per loaded chunk. The scalar
//! backend implements `fmadd` with `f32::mul_add`, which is a `fmaf` library
//! call outside an FMA frame and a single instruction inside one, so an
//! FMA-dense body is what separates the two codegen outcomes — the same
//! density the FFT butterflies that reach this path in practice carry.

use criterion::{black_box, BenchmarkId, Criterion, Throughput};
use hermes_simd::{
    vectorize_lanes, LaneKernel, LaneScalar, Simd, SimdArch, SimdKernel, SimdStorage, Vector,
};

/// Fused multiply-adds per loaded chunk.
///
/// Four keeps the chain long enough that the multiply-add dominates the load
/// and store around it, and short enough that the working set, not the
/// dependency chain, still sets the size sweep's shape.
const CHAIN: usize = 4;

struct FmaChain<'a, T: LaneScalar> {
    a: &'a [T],
    b: &'a [T],
    out: &'a mut [T],
}

impl<T: LaneScalar> LaneKernel<T> for FmaChain<'_, T> {
    type Output = ();

    #[inline(always)]
    fn call<A: SimdArch + SimdKernel<T>>(self, simd: Simd<T, A>) {
        let lanes = <A as SimdStorage<T>>::LANE_COUNT;
        let chunks = self.a.len() / lanes;
        let a = simd.view(self.a);
        let b = simd.view(self.b);
        let mut out = simd.view_mut(self.out);
        for chunk in 0..chunks {
            let va = Vector::from_view_chunk(&a, chunk);
            let vb = Vector::from_view_chunk(&b, chunk);
            let mut acc = vb;
            for _ in 0..CHAIN {
                acc = va.mul_add(vb, acc);
            }
            acc.store_to_view_chunk(&mut out, chunk);
        }
    }
}

/// Benchmarks the `LANES`-wide exact-lane entry for one scalar.
///
/// `LANES` is the caller's requested width, not the host's widest: the point
/// of the entry is that a consumer whose schedule fixes its register width
/// gets exactly that width or nothing.
pub fn bench<const LANES: usize, T>(
    criterion: &mut Criterion,
    group_name: &str,
    a_value: T,
    b_value: T,
) where
    T: LaneScalar + Copy + 'static,
{
    let probe_a = [a_value; LANES];
    let probe_b = [b_value; LANES];
    let mut probe_out = [a_value; LANES];
    if vectorize_lanes::<LANES, T, _>(FmaChain {
        a: &probe_a,
        b: &probe_b,
        out: &mut probe_out,
    })
    .is_none()
    {
        return;
    }

    let mut group = super::group::configured(criterion, group_name);
    for &size in &[256usize, 1024, 4096] {
        let a = vec![a_value; size];
        let b = vec![b_value; size];
        let mut out = vec![a_value; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("vectorize_lanes", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    vectorize_lanes::<LANES, T, _>(FmaChain {
                        a: black_box(&a),
                        b: black_box(&b),
                        out: black_box(out.as_mut_slice()),
                    })
                    .expect("invariant: exact-lane capability preflight succeeded");
                });
            },
        );
        black_box(out.as_slice());
    }
    group.finish();
}

//! Streaming (non-temporal) store feasibility experiment.
//!
//! Answers the open question "are `_mm256_stream_ps` non-temporal stores worth a
//! `store_streaming` kernel seam?" empirically before productionizing. For a
//! write-only, out-of-last-level-cache elementwise op (`out = a + b`), a normal
//! store triggers a read-for-ownership (write-allocate) that fetches the
//! destination line only to overwrite it — a non-temporal store skips it,
//! cutting store-side memory traffic. The two rows differ **only** in the store
//! instruction (both do unaligned AVX2 loads + add); the gap is the NT benefit
//! on this microarchitecture. Buffers are 64-byte aligned (NT stores fault
//! otherwise) and sized well past L3 so the effect is visible.
//!
//! Not a regression gate — a one-shot experiment. If NT wins materially the
//! result justifies a `SimdLoadStore::store_streaming` seam + a size-gated path in
//! the elementwise kernels; if not, it is a recorded negative result.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd_core::{align::Aligned, AlignedVec};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn aligned_filled(len: usize, f: impl Fn(usize) -> f32) -> AlignedVec<f32, Aligned<64>> {
    let mut v = AlignedVec::<f32, Aligned<64>>::with_capacity(len);
    for i in 0..len {
        v.push(f(i));
    }
    v
}

/// `out[i] = a[i] + b[i]` with normal aligned stores (write-allocate / RFO).
///
/// # Safety
/// `avx2` must be supported; `a`/`b`/`out` are 64-byte aligned, length `len`, a
/// multiple of 8.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_regular(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_store_ps};
    let mut i = 0;
    while i < len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_store_ps(out.add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
}

/// `out[i] = a[i] + b[i]` with non-temporal stores + a trailing `sfence`.
///
/// # Safety
/// As [`add_regular`]; additionally the caller must not read `out` before the
/// `sfence` (issued here) retires.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_streaming(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_stream_ps, _mm_sfence};
    let mut i = 0;
    while i < len {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        _mm256_stream_ps(out.add(i), _mm256_add_ps(va, vb));
        i += 8;
    }
    _mm_sfence();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn bench_streaming(c: &mut Criterion) {
    if !std::is_x86_feature_detected!("avx2") {
        eprintln!("skipping streaming_bench: host lacks avx2");
        return;
    }
    let mut group = c.benchmark_group("Elementwise add store policy");

    // 16 Mi f32 = 64 MiB per buffer (192 MiB working set) — past any consumer L3,
    // so the write path is DRAM-bound and the RFO the NT store avoids is visible.
    let len = 1usize << 24;
    let a = aligned_filled(len, |i| (i % 97) as f32 * 0.5);
    let b = aligned_filled(len, |i| (i % 89) as f32 * 0.25);
    let mut out = aligned_filled(len, |_| 0.0);

    group.throughput(Throughput::Bytes((len * 4 * 3) as u64)); // 2 reads + 1 write

    group.bench_with_input(BenchmarkId::new("regular", len), &len, |bencher, _| {
        bencher.iter(|| unsafe { add_regular(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len) });
    });
    group.bench_with_input(BenchmarkId::new("streaming", len), &len, |bencher, _| {
        bencher.iter(|| unsafe { add_streaming(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len) });
    });
    group.finish();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
criterion_group!(benches, bench_streaming);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
criterion_main!(benches);

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}

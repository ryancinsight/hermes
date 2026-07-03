//! Register-blocking / tiling benchmark: tiled GEMM, batch sensitivity, and context switch pressure.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use half::bf16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use hermes_simd::AmxSupport;
use hermes_simd::{gemm, tiled_gemm, Scalar, TileMatrixMultiply};

fn bench_tiled_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tiled GEMM f32");

    for &size in &[32usize, 64] {
        let a: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.0001).collect();
        let b: Vec<f32> = (0..size * size)
            .map(|i| (size * size - i) as f32 * 0.0001)
            .collect();
        let mut out = vec![0.0f32; size * size];

        group.throughput(Throughput::Elements((size * size * size) as u64));

        group.bench_with_input(BenchmarkId::new("tiled_gemm", size), &size, |bencher, _| {
            bencher.iter(|| {
                tiled_gemm(&a, &b, &mut out, size, size, size).unwrap();
            })
        });
    }
    group.finish();
}

fn bench_bf16_gemm_batch_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("BF16 GEMM Batch Sensitivity");

    // Test a range of matrix sizes to show the crossover threshold where AMX becomes highly profitable
    for &size in &[8usize, 16, 32, 64, 128] {
        let a = vec![bf16::from_f32(1.0); size * size];
        let b = vec![bf16::from_f32(2.0); size * size];
        let mut out = vec![0.0f32; size * size];

        group.throughput(Throughput::Elements((size * size * size) as u64));

        // Baseline: generic scalar fallback
        group.bench_with_input(
            BenchmarkId::new("scalar_fallback", size),
            &size,
            |bencher, _| {
                bencher.iter(|| unsafe {
                    let mut i = 0;
                    while i + 16 <= size {
                        let mut j = 0;
                        while j + 16 <= size {
                            let mut kk = 0;
                            while kk + 32 <= size {
                                <Scalar as TileMatrixMultiply<
                                    bf16,
                                    bf16,
                                    f32,
                                    Scalar,
                                    Scalar,
                                    16,
                                    16,
                                    32,
                                >>::tile_matmul(
                                    out.as_mut_ptr().add(i * size + j),
                                    size,
                                    a.as_ptr().add(i * size + kk),
                                    size,
                                    b.as_ptr().add(kk * size + j),
                                    size,
                                );
                                kk += 32;
                            }
                            j += 16;
                        }
                        i += 16;
                    }
                    // remainder
                    let bound = (size / 16) * 16;
                    let k_bound = (size / 32) * 32;
                    for r in 0..size {
                        for col in 0..size {
                            if r >= bound || col >= bound {
                                let mut sum = 0.0f32;
                                for kk in 0..size {
                                    sum += a[r * size + kk].to_f32() * b[kk * size + col].to_f32();
                                }
                                out[r * size + col] += sum;
                            } else if k_bound < size {
                                let mut sum = 0.0f32;
                                for kk in k_bound..size {
                                    sum += a[r * size + kk].to_f32() * b[kk * size + col].to_f32();
                                }
                                out[r * size + col] += sum;
                            }
                        }
                    }
                })
            },
        );

        // Dynamic dispatch: AVX-512 or AMX depending on CPU and batch size heuristics
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| unsafe {
                gemm::<bf16, bf16, f32>(size, size, size, &a, size, &b, size, &mut out, size)
                    .unwrap();
            })
        });
    }
    group.finish();
}

/// Square int8 GEMM over 16×16×64 tiles with the scalar remainder — one body
/// shared by every forced-backend bench row so rows differ only in the tile
/// kernel under measurement.
///
/// # Safety
/// Caller must ensure the CPU supports `Arch`'s ISA (bench rows gate on the
/// matching `is_x86_feature_detected!`) and that `a`, `b`, `out` are
/// `size * size` element buffers.
unsafe fn forced_backend_int8_gemm<Arch>(size: usize, a: &[i8], b: &[i8], out: &mut [i32])
where
    Arch: TileMatrixMultiply<i8, i8, i32, Arch, Arch, 16, 16, 64>,
{
    let mut i = 0;
    while i + 16 <= size {
        let mut j = 0;
        while j + 16 <= size {
            let mut kk = 0;
            while kk + 64 <= size {
                Arch::tile_matmul(
                    out.as_mut_ptr().add(i * size + j),
                    size,
                    a.as_ptr().add(i * size + kk),
                    size,
                    b.as_ptr().add(kk * size + j),
                    size,
                );
                kk += 64;
            }
            j += 16;
        }
        i += 16;
    }
    // remainder
    let bound = (size / 16) * 16;
    let k_bound = (size / 64) * 64;
    for r in 0..size {
        for col in 0..size {
            if r >= bound || col >= bound {
                let mut sum = 0i32;
                for kk in 0..size {
                    sum = sum.wrapping_add((a[r * size + kk] as i32) * (b[kk * size + col] as i32));
                }
                out[r * size + col] += sum;
            } else if k_bound < size {
                let mut sum = 0i32;
                for kk in k_bound..size {
                    sum = sum.wrapping_add((a[r * size + kk] as i32) * (b[kk * size + col] as i32));
                }
                out[r * size + col] += sum;
            }
        }
    }
}

fn bench_int8_gemm_batch_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("INT8 GEMM Batch Sensitivity");

    for &size in &[8usize, 16, 32, 64, 128] {
        let a = vec![1i8; size * size];
        let b = vec![2i8; size * size];
        let mut out = vec![0i32; size * size];

        group.throughput(Throughput::Elements((size * size * size) as u64));

        // Baseline: generic scalar fallback
        group.bench_with_input(
            BenchmarkId::new("scalar_fallback", size),
            &size,
            |bencher, _| {
                bencher
                    .iter(|| unsafe { forced_backend_int8_gemm::<Scalar>(size, &a, &b, &mut out) })
            },
        );

        // Forced 256-bit AVX-VNNI tiles (client CPUs without AVX-512).
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avxvnni") {
            group.bench_with_input(
                BenchmarkId::new("avx_vnni_tiles", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| unsafe {
                        forced_backend_int8_gemm::<hermes_simd::AvxVnni>(size, &a, &b, &mut out)
                    })
                },
            );
        }

        // Dynamic dispatch: AMX / AVX-512 VNNI / AVX-VNNI / scalar by CPU and
        // batch-size heuristics.
        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |bencher, _| {
            bencher.iter(|| unsafe {
                gemm::<i8, i8, i32>(size, size, size, &a, size, &b, size, &mut out, size).unwrap();
            })
        });
    }
    group.finish();
}

fn bench_amx_context_switch_pressure(c: &mut Criterion) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !<bf16 as AmxSupport>::has_amx() {
        return;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    return;

    let mut group = c.benchmark_group("AMX Context Switch Pressure");
    let size = 64usize;
    let a = vec![bf16::from_f32(1.0); size * size];
    let b = vec![bf16::from_f32(2.0); size * size];
    let mut out = vec![0.0f32; size * size];

    group.throughput(Throughput::Elements((size * size * size) as u64));

    // Scenario 1: Reusable AmxSession (static config)
    group.bench_function("reusable_session", |bencher| {
        bencher.iter(|| unsafe {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if <bf16 as AmxSupport>::has_amx() {
                    let config = hermes_simd::AmxConfig::new_uniform(16, 64);
                    let _session = hermes_simd::AmxSession::new(&config);
                    gemm::<bf16, bf16, f32>(size, size, size, &a, size, &b, size, &mut out, size)
                        .unwrap();
                    return;
                }
            }
            gemm::<bf16, bf16, f32>(size, size, size, &a, size, &b, size, &mut out, size).unwrap();
        })
    });

    // Scenario 2: Configuring/releasing tile state on every iteration (no reuse / raw call)
    group.bench_function("raw_call_config_release", |bencher| {
        bencher.iter(|| unsafe {
            gemm::<bf16, bf16, f32>(size, size, size, &a, size, &b, size, &mut out, size).unwrap();
        })
    });

    // Scenario 3: yielding thread to simulate heavy OS context switch pressure
    group.bench_function("with_thread_yield", |bencher| {
        bencher.iter(|| unsafe {
            gemm::<bf16, bf16, f32>(size, size, size, &a, size, &b, size, &mut out, size).unwrap();
            std::thread::yield_now();
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tiled_gemm,
    bench_bf16_gemm_batch_sensitivity,
    bench_int8_gemm_batch_sensitivity,
    bench_amx_context_switch_pressure
);
criterion_main!(benches);

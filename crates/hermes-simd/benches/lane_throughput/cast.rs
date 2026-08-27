//! Equal-width numeric conversion against an AVX2-native ceiling and Fearless SIMD.

use criterion::Criterion;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use std::hint::black_box;

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256i, _mm256_add_ps, _mm256_cvtepi32_ps, _mm256_cvttps_epi32, _mm256_loadu_ps,
        _mm256_loadu_si256, _mm256_setzero_ps, _mm256_setzero_si256, _mm256_storeu_ps,
        _mm256_storeu_si256, _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256i, _mm256_add_ps, _mm256_cvtepi32_ps, _mm256_cvttps_epi32, _mm256_loadu_ps,
        _mm256_loadu_si256, _mm256_setzero_ps, _mm256_setzero_si256, _mm256_storeu_ps,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    use criterion::{BenchmarkId, Criterion, Throughput};
    use fearless_simd::prelude::{SimdBase, SimdFloat, SimdInt};
    use fearless_simd::{Avx2 as FearlessAvx2, Level, Simd as FearlessSimd};
    use hermes_simd::{Avx2, SimdArch, SimdStorage, Vector};

    use super::super::SCALAR_LENGTHS;

    const AVX2_LANES: usize = 8;

    #[target_feature(enable = "avx2")]
    unsafe fn hermes_float_to_int(source: &[f32], destination: &mut [i32]) -> [i32; AVX2_LANES] {
        let mut checksum = Vector::<i32, Avx2>::zero();
        for (source, destination) in source
            .chunks_exact(AVX2_LANES)
            .zip(destination.chunks_exact_mut(AVX2_LANES))
        {
            // SAFETY: the benchmark checks AVX2 support before calling this
            // kernel, and each exact chunk contains one complete vector.
            unsafe {
                let output = Vector::<f32, Avx2>::load_unaligned(source.as_ptr()).cast::<i32>();
                checksum ^= output;
                output.store_unaligned(destination.as_mut_ptr());
            }
        }
        checksum.to_array()
    }

    #[target_feature(enable = "avx2")]
    unsafe fn hermes_int_to_float(source: &[i32], destination: &mut [f32]) -> [f32; AVX2_LANES] {
        let mut checksum = Vector::<f32, Avx2>::zero();
        for (source, destination) in source
            .chunks_exact(AVX2_LANES)
            .zip(destination.chunks_exact_mut(AVX2_LANES))
        {
            // SAFETY: the benchmark checks AVX2 support before calling this
            // kernel, and each exact chunk contains one complete vector.
            unsafe {
                let output = Vector::<i32, Avx2>::load_unaligned(source.as_ptr()).cast::<f32>();
                checksum += output;
                output.store_unaligned(destination.as_mut_ptr());
            }
        }
        checksum.to_array()
    }

    #[target_feature(enable = "avx2")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "AVX2 storeu accepts an i32-aligned destination"
    )]
    unsafe fn native_float_to_int(source: &[f32], destination: &mut [i32]) -> [i32; AVX2_LANES] {
        let mut checksum = _mm256_setzero_si256();
        for (source, destination) in source
            .chunks_exact(AVX2_LANES)
            .zip(destination.chunks_exact_mut(AVX2_LANES))
        {
            // SAFETY: the function's target feature covers these intrinsics;
            // exact chunks prove one unaligned 256-bit load and store valid.
            unsafe {
                let input = _mm256_loadu_ps(source.as_ptr());
                let output = _mm256_cvttps_epi32(input);
                checksum = _mm256_xor_si256(checksum, output);
                _mm256_storeu_si256(destination.as_mut_ptr().cast::<__m256i>(), output);
            }
        }
        let mut result = [0; AVX2_LANES];
        // SAFETY: `result` contains exactly one vector and storeu permits its
        // four-byte alignment.
        unsafe { _mm256_storeu_si256(result.as_mut_ptr().cast::<__m256i>(), checksum) };
        result
    }

    #[target_feature(enable = "avx2")]
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "AVX2 loadu accepts an i32-aligned source"
    )]
    unsafe fn native_int_to_float(source: &[i32], destination: &mut [f32]) -> [f32; AVX2_LANES] {
        let mut checksum = _mm256_setzero_ps();
        for (source, destination) in source
            .chunks_exact(AVX2_LANES)
            .zip(destination.chunks_exact_mut(AVX2_LANES))
        {
            // SAFETY: the function's target feature covers these intrinsics;
            // exact chunks prove one unaligned 256-bit load and store valid.
            unsafe {
                let input = _mm256_loadu_si256(source.as_ptr().cast::<__m256i>());
                let output = _mm256_cvtepi32_ps(input);
                checksum = _mm256_add_ps(checksum, output);
                _mm256_storeu_ps(destination.as_mut_ptr(), output);
            }
        }
        let mut result = [0.0; AVX2_LANES];
        // SAFETY: `result` contains exactly one vector and storeu permits its
        // four-byte alignment.
        unsafe { _mm256_storeu_ps(result.as_mut_ptr(), checksum) };
        result
    }

    #[target_feature(enable = "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,fxsr,lzcnt,movbe,popcnt,xsave")]
    unsafe fn fearless_float_to_int(
        simd: FearlessAvx2,
        source: &[f32],
        destination: &mut [i32],
    ) -> [i32; AVX2_LANES] {
        let zero = [0; AVX2_LANES];
        let mut checksum = <FearlessAvx2 as FearlessSimd>::i32s::from_slice(simd, &zero);
        for (source, destination) in source
            .chunks_exact(AVX2_LANES)
            .zip(destination.chunks_exact_mut(AVX2_LANES))
        {
            let input = <FearlessAvx2 as FearlessSimd>::f32s::from_slice(simd, source);
            let output: <FearlessAvx2 as FearlessSimd>::i32s = input.to_int_precise();
            checksum ^= output;
            output.store_slice(destination);
        }
        let mut result = [0; AVX2_LANES];
        checksum.store_slice(&mut result);
        result
    }

    #[target_feature(enable = "avx2,bmi1,bmi2,cmpxchg16b,f16c,fma,fxsr,lzcnt,movbe,popcnt,xsave")]
    unsafe fn fearless_int_to_float(
        simd: FearlessAvx2,
        source: &[i32],
        destination: &mut [f32],
    ) -> [f32; AVX2_LANES] {
        let zero = [0.0; AVX2_LANES];
        let mut checksum = <FearlessAvx2 as FearlessSimd>::f32s::from_slice(simd, &zero);
        for (source, destination) in source
            .chunks_exact(AVX2_LANES)
            .zip(destination.chunks_exact_mut(AVX2_LANES))
        {
            let input = <FearlessAvx2 as FearlessSimd>::i32s::from_slice(simd, source);
            let output: <FearlessAvx2 as FearlessSimd>::f32s = input.to_float();
            checksum += output;
            output.store_slice(destination);
        }
        let mut result = [0.0; AVX2_LANES];
        checksum.store_slice(&mut result);
        result
    }

    fn integer_checksum(values: &[i32]) -> [i32; AVX2_LANES] {
        let mut checksum = [0; AVX2_LANES];
        for chunk in values.chunks_exact(AVX2_LANES) {
            for (lane, value) in checksum.iter_mut().zip(chunk) {
                *lane ^= value;
            }
        }
        checksum
    }

    fn float_checksum(values: &[f32]) -> [f32; AVX2_LANES] {
        let mut checksum = [0.0; AVX2_LANES];
        for chunk in values.chunks_exact(AVX2_LANES) {
            for (lane, value) in checksum.iter_mut().zip(chunk) {
                *lane += value;
            }
        }
        checksum
    }

    fn assert_float_checksum_eq(actual: [f32; AVX2_LANES], expected: [f32; AVX2_LANES]) {
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }

    fn float_inputs(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let bounded = u16::try_from(index % 2_047).expect("bounded index fits u16");
                f32::from(bounded) - 1_023.75
            })
            .collect()
    }

    fn int_inputs(len: usize) -> Vec<i32> {
        (0..len)
            .map(|index| {
                let bounded = i32::try_from(index % 1_048_573).expect("bounded index fits i32");
                bounded - 524_286
            })
            .collect()
    }

    fn bench_float_to_int(c: &mut Criterion, simd: FearlessAvx2) {
        let mut group = c.benchmark_group("cast_f32_to_i32");
        for &len in SCALAR_LENGTHS {
            group.throughput(Throughput::Elements(
                u64::try_from(len).expect("benchmark length fits u64"),
            ));
            let source = float_inputs(len);
            let expected = source
                .iter()
                .copied()
                .map(|value| value as i32)
                .collect::<Vec<_>>();
            let mut hermes = vec![0; len];
            let mut native = vec![0; len];
            let mut fearless = vec![0; len];
            let expected_checksum = integer_checksum(&expected);

            // SAFETY: `bench` establishes AVX2 support before entering this group.
            let hermes_checksum = unsafe { hermes_float_to_int(&source, &mut hermes) };
            // SAFETY: `bench` establishes AVX2 support before entering this group.
            let native_checksum = unsafe { native_float_to_int(&source, &mut native) };
            // SAFETY: `Level::as_avx2` established the x86-64-v3 feature set.
            let fearless_checksum = unsafe { fearless_float_to_int(simd, &source, &mut fearless) };
            assert_eq!(hermes, expected);
            assert_eq!(native, expected);
            assert_eq!(fearless, expected);
            assert_eq!(hermes_checksum, expected_checksum);
            assert_eq!(native_checksum, expected_checksum);
            assert_eq!(fearless_checksum, expected_checksum);

            group.bench_function(BenchmarkId::new("hermes_public", len), |bencher| {
                bencher.iter(|| {
                    // SAFETY: `bench` establishes AVX2 support before this group.
                    black_box(unsafe {
                        hermes_float_to_int(black_box(&source), black_box(&mut hermes))
                    })
                });
            });
            group.bench_function(BenchmarkId::new("native_avx2", len), |bencher| {
                bencher.iter(|| {
                    // SAFETY: `bench` establishes AVX2 support before this group.
                    black_box(unsafe {
                        native_float_to_int(black_box(&source), black_box(&mut native))
                    })
                });
            });
            group.bench_function(BenchmarkId::new("fearless_simd", len), |bencher| {
                bencher.iter(|| {
                    // SAFETY: `Level::as_avx2` established the x86-64-v3 feature set.
                    black_box(unsafe {
                        fearless_float_to_int(simd, black_box(&source), black_box(&mut fearless))
                    })
                });
            });
        }
        group.finish();
    }

    fn bench_int_to_float(c: &mut Criterion, simd: FearlessAvx2) {
        let mut group = c.benchmark_group("cast_i32_to_f32");
        for &len in SCALAR_LENGTHS {
            group.throughput(Throughput::Elements(
                u64::try_from(len).expect("benchmark length fits u64"),
            ));
            let source = int_inputs(len);
            let expected = source
                .iter()
                .copied()
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            let mut hermes = vec![0.0; len];
            let mut native = vec![0.0; len];
            let mut fearless = vec![0.0; len];
            let expected_checksum = float_checksum(&expected);

            // SAFETY: `bench` establishes AVX2 support before entering this group.
            let hermes_checksum = unsafe { hermes_int_to_float(&source, &mut hermes) };
            // SAFETY: `bench` establishes AVX2 support before entering this group.
            let native_checksum = unsafe { native_int_to_float(&source, &mut native) };
            // SAFETY: `Level::as_avx2` established the x86-64-v3 feature set.
            let fearless_checksum = unsafe { fearless_int_to_float(simd, &source, &mut fearless) };
            assert_eq!(hermes, expected);
            assert_eq!(native, expected);
            assert_eq!(fearless, expected);
            assert_float_checksum_eq(hermes_checksum, expected_checksum);
            assert_float_checksum_eq(native_checksum, expected_checksum);
            assert_float_checksum_eq(fearless_checksum, expected_checksum);

            group.bench_function(BenchmarkId::new("hermes_public", len), |bencher| {
                bencher.iter(|| {
                    // SAFETY: `bench` establishes AVX2 support before this group.
                    black_box(unsafe {
                        hermes_int_to_float(black_box(&source), black_box(&mut hermes))
                    })
                });
            });
            group.bench_function(BenchmarkId::new("native_avx2", len), |bencher| {
                bencher.iter(|| {
                    // SAFETY: `bench` establishes AVX2 support before this group.
                    black_box(unsafe {
                        native_int_to_float(black_box(&source), black_box(&mut native))
                    })
                });
            });
            group.bench_function(BenchmarkId::new("fearless_simd", len), |bencher| {
                bencher.iter(|| {
                    // SAFETY: `Level::as_avx2` established the x86-64-v3 feature set.
                    black_box(unsafe {
                        fearless_int_to_float(simd, black_box(&source), black_box(&mut fearless))
                    })
                });
            });
        }
        group.finish();
    }

    pub(super) fn bench(c: &mut Criterion) {
        if !Avx2::is_runtime_supported() {
            return;
        }
        let Some(simd) = Level::new().as_avx2() else {
            return;
        };
        assert_eq!(<Avx2 as SimdStorage<f32>>::LANE_COUNT, AVX2_LANES);
        assert_eq!(<Avx2 as SimdStorage<i32>>::LANE_COUNT, AVX2_LANES);
        assert_eq!(<FearlessAvx2 as FearlessSimd>::f32s::N, AVX2_LANES);
        assert_eq!(<FearlessAvx2 as FearlessSimd>::i32s::N, AVX2_LANES);

        bench_float_to_int(c, simd);
        bench_int_to_float(c, simd);
    }
}

pub(super) fn bench(c: &mut Criterion) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    x86::bench(c);

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let _ = c;
}

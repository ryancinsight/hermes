#![expect(
    clippy::float_cmp,
    reason = "The manufactured products and zero imaginary lanes are exactly representable"
)]

use hermes_simd::{
    real_mul_to_interleaved_complex, real_mul_to_interleaved_complex_runtime, PreferredArch,
    Scalar, SimdError,
};
#[cfg(not(miri))]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(not(miri))]
use std::cell::Cell;

#[cfg(not(miri))]
thread_local! {
    static ALLOCATION_COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}

#[cfg(not(miri))]
fn record_allocation() {
    let _ = ALLOCATION_COUNT.try_with(|count| {
        if let Some(current) = count.get() {
            count.set(Some(current + 1));
        }
    });
}

#[cfg(not(miri))]
fn count_allocations<R>(operation: impl FnOnce() -> R) -> (R, usize) {
    ALLOCATION_COUNT.with(|count| count.set(Some(0)));
    let result = operation();
    let count = ALLOCATION_COUNT
        .with(|count| count.replace(None))
        .unwrap_or(0);
    (result, count)
}

#[cfg(not(miri))]
struct CountingAllocator;

// SAFETY: every operation forwards its exact pointer and layout contract to
// `System`; the thread-local counter only observes allocation calls.
#[cfg(not(miri))]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[cfg(not(miri))]
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn expected_f64(input: &[f64], factors: &[f64]) -> Vec<f64> {
    input
        .iter()
        .zip(factors)
        .flat_map(|(&sample, &factor)| [sample * factor, 0.0])
        .collect()
}

#[test]
fn real_mul_to_interleaved_complex_covers_full_and_ragged_vectors() {
    for len in [0, 1, 3, 4, 7, 8, 17, 33] {
        let input: Vec<f64> = (0..len).map(|index| index as f64 - 7.0).collect();
        let factors: Vec<f64> = (0..len)
            .map(|index| (index % 5) as f64 * 0.25 - 0.5)
            .collect();
        let expected = expected_f64(&input, &factors);

        let mut scalar = vec![f64::NAN; len * 2];
        real_mul_to_interleaved_complex::<f64, Scalar>(&input, &factors, &mut scalar).unwrap();
        assert_eq!(scalar, expected, "scalar length {len}");

        let mut preferred = vec![f64::NAN; len * 2];
        real_mul_to_interleaved_complex::<f64, PreferredArch>(&input, &factors, &mut preferred)
            .unwrap();
        assert_eq!(preferred, expected, "preferred length {len}");

        let mut runtime = vec![f64::NAN; len * 2];
        real_mul_to_interleaved_complex_runtime(&input, &factors, &mut runtime).unwrap();
        assert_eq!(runtime, expected, "runtime length {len}");
    }
}

#[test]
fn real_mul_to_interleaved_complex_preserves_native_f32_arithmetic() {
    let input = [1.5_f32, -2.0, 3.25, -4.5, 0.125, 8.0, -16.0, 32.0, 7.0];
    let factors = [2.0_f32, -0.5, 4.0, 0.25, -8.0, 0.125, -0.0625, 0.5, -3.0];
    let mut output = [f32::NAN; 18];

    real_mul_to_interleaved_complex_runtime(&input, &factors, &mut output).unwrap();

    for (index, (&sample, &factor)) in input.iter().zip(&factors).enumerate() {
        assert_eq!(output[index * 2], sample * factor);
        assert_eq!(output[index * 2 + 1].to_bits(), 0.0_f32.to_bits());
    }
}

#[test]
fn real_mul_to_interleaved_complex_rejects_before_mutation() {
    let input = [1.0_f64, 2.0, 3.0];
    let factors = [4.0_f64, 5.0, 6.0];

    let mut short_output = [91.0_f64; 5];
    assert_eq!(
        real_mul_to_interleaved_complex_runtime(&input, &factors, &mut short_output),
        Err(SimdError::LengthMismatch)
    );
    assert_eq!(short_output, [91.0; 5]);

    let mut output = [73.0_f64; 6];
    assert_eq!(
        real_mul_to_interleaved_complex_runtime(&input, &factors[..2], &mut output),
        Err(SimdError::LengthMismatch)
    );
    assert_eq!(output, [73.0; 6]);
}

#[test]
#[cfg(not(miri))]
fn real_mul_to_interleaved_complex_first_and_warm_calls_allocate_nothing() {
    let input = vec![1.25_f64; 1_027];
    let factors = vec![-0.5_f64; input.len()];
    let mut output = vec![f64::NAN; input.len() * 2];

    let (first, first_allocations) = count_allocations(|| {
        real_mul_to_interleaved_complex_runtime(&input, &factors, &mut output)
    });
    assert_eq!(first, Ok(()));
    assert_eq!(first_allocations, 0);

    let (warm, warm_allocations) = count_allocations(|| {
        real_mul_to_interleaved_complex_runtime(&input, &factors, &mut output)
    });
    assert_eq!(warm, Ok(()));
    assert_eq!(warm_allocations, 0);
    assert_eq!(output[0], -0.625);
    assert_eq!(output[1].to_bits(), 0.0_f64.to_bits());
    assert_eq!(output[output.len() - 2], -0.625);
    assert_eq!(output[output.len() - 1].to_bits(), 0.0_f64.to_bits());
}

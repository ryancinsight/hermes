//! Exact-lane dispatch contract tests.

use core::cell::Cell;

use hermes_simd::{
    vectorize_lanes, LaneKernel, Simd, SimdArch, SimdKernel, SimdScalar, SimdStorage, TargetId,
};

struct SelectionProbe<'a>(&'a Cell<usize>);

impl<T: SimdScalar> LaneKernel<T> for SelectionProbe<'_> {
    type Output = (&'static str, usize);

    fn call<A: SimdArch + SimdKernel<T>>(self, _: Simd<T, A>) -> Self::Output {
        self.0.set(self.0.get() + 1);
        (A::NAME, <A as SimdStorage<T>>::LANE_COUNT)
    }
}

#[test]
fn unavailable_width_does_not_invoke_kernel() {
    let calls = Cell::new(0);
    assert_eq!(vectorize_lanes::<0, f64, _>(SelectionProbe(&calls)), None);
    assert_eq!(vectorize_lanes::<3, f64, _>(SelectionProbe(&calls)), None);
    assert_eq!(calls.get(), 0);
}

#[test]
fn portable_width_invokes_kernel_once() {
    let calls = Cell::new(0);
    let actual = vectorize_lanes::<2, f64, _>(SelectionProbe(&calls));
    assert_eq!(actual.map(|(_, lanes)| lanes), Some(2));
    assert_eq!(calls.get(), 1);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn four_f64_lanes_select_avx2_even_when_avx512_exists() {
    let calls = Cell::new(0);
    let actual = vectorize_lanes::<4, f64, _>(SelectionProbe(&calls));
    if TargetId::Avx2.is_supported() {
        assert_eq!(actual, Some(("avx2", 4)));
        assert_eq!(calls.get(), 1);
    } else {
        assert_eq!(actual, None);
        assert_eq!(calls.get(), 0);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn four_f32_lanes_select_portable_backend() {
    let calls = Cell::new(0);
    assert_eq!(
        vectorize_lanes::<4, f32, _>(SelectionProbe(&calls)),
        Some(("scalar", 4))
    );
    assert_eq!(calls.get(), 1);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn four_f32_lanes_select_neon() {
    let calls = Cell::new(0);
    assert_eq!(
        vectorize_lanes::<4, f32, _>(SelectionProbe(&calls)),
        Some(("neon", 4))
    );
    assert_eq!(calls.get(), 1);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn unavailable_four_f64_lanes_do_not_substitute_neon() {
    let calls = Cell::new(0);
    assert_eq!(vectorize_lanes::<4, f64, _>(SelectionProbe(&calls)), None);
    assert_eq!(calls.get(), 0);
}

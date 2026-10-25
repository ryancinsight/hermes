//! Downstream implementations retain the marker contracts after bound factoring.

use hermes_simd_core::align::{is_aligned_for_arch, Alignment};
use hermes_simd_core::arch::{IsaFamily, SimdArch};
use hermes_simd_core::kernel::{
    SimdArith, SimdBitwise, SimdCompare, SimdLoadStore, SimdMask, SimdReduce,
};
use hermes_simd_core::ops::elementwise::ElementOp;
use hermes_simd_core::ops::reduction::{ReductionOp, Sum};
use hermes_simd_core::ops::scan::{ScanMode, ScanOp};
use hermes_simd_core::ops::unary::UnaryOp;
use hermes_simd_core::private::Sealed;
use tracing_subscriber::fmt::format::FmtSpan;

#[derive(Clone, Copy)]
struct ConsumerMarker;

impl Sealed for ConsumerMarker {}

impl Alignment for ConsumerMarker {
    const ALIGNMENT: Option<usize> = Some(16);
    const IS_ALIGNED: bool = true;
    const ALIGN_BYTES: usize = 16;
}

impl SimdArch for ConsumerMarker {
    const NAME: &'static str = "consumer-marker";
    const REGISTER_WIDTH_BITS: u32 = 128;
    const ISA_FAMILY: IsaFamily = IsaFamily::Other;
    const FMA_THROUGHPUT_HINT: u32 = 1;

    fn is_runtime_supported() -> bool {
        false
    }
}

impl ScanMode for ConsumerMarker {
    const IS_INCLUSIVE: bool = true;
}

impl ScanOp<f32> for ConsumerMarker {
    fn identity() -> f32 {
        0.0
    }

    fn combine(a: f32, b: f32) -> f32 {
        a + b
    }
}

impl ElementOp<f32> for ConsumerMarker {
    unsafe fn apply<Arch: SimdArith<f32> + SimdBitwise<f32>>(
        self,
        a: Arch::Vector,
        b: Arch::Vector,
    ) -> Arch::Vector {
        // SAFETY: the caller provides the target-feature guarantee required by this trait.
        unsafe { Arch::add(a, b) }
    }

    fn apply_scalar(self, a: f32, b: f32) -> f32 {
        a + b
    }
}

impl UnaryOp<f32> for ConsumerMarker {
    unsafe fn apply<Arch: SimdArith<f32> + SimdBitwise<f32>>(
        self,
        v: Arch::Vector,
    ) -> Arch::Vector {
        v
    }

    fn apply_scalar(self, a: f32) -> f32 {
        a
    }
}

impl ReductionOp<f32> for ConsumerMarker {
    unsafe fn accumulate<
        Arch: SimdLoadStore<f32> + SimdArith<f32> + SimdCompare<f32> + SimdMask<f32> + SimdReduce<f32>,
    >(
        acc: Arch::Vector,
        v: Arch::Vector,
    ) -> Arch::Vector {
        // SAFETY: the caller supplies the same target-feature guarantee as Sum requires.
        unsafe { <Sum as ReductionOp<f32>>::accumulate::<Arch>(acc, v) }
    }

    unsafe fn finalize<
        Arch: SimdLoadStore<f32> + SimdArith<f32> + SimdCompare<f32> + SimdMask<f32> + SimdReduce<f32>,
    >(
        acc: Arch::Vector,
    ) -> f32 {
        // SAFETY: the caller supplies the same target-feature guarantee as Sum requires.
        unsafe { <Sum as ReductionOp<f32>>::finalize::<Arch>(acc) }
    }

    fn identity_scalar() -> f32 {
        0.0
    }

    fn scalar_combine(a: f32, b: f32) -> f32 {
        a + b
    }
}

fn old_op_bounds<T: Sealed + Copy + 'static>(value: T) -> T {
    value
}

fn old_arch_bounds<T: Sealed + Send + Sync + Copy + Clone + 'static>(value: T) -> T {
    value
}

// Check each implication separately so another marker trait cannot supply its bounds.
fn scan_bounds<T: ScanOp<f32>>(value: T) -> T {
    old_op_bounds(value)
}
fn mode_bounds<T: ScanMode>(value: T) -> T {
    old_op_bounds(value)
}
fn reduction_bounds<T: ReductionOp<f32>>(value: T) -> T {
    old_op_bounds(value)
}
fn element_bounds<T: ElementOp<f32>>(value: T) -> T {
    old_op_bounds(value)
}
fn unary_bounds<T: UnaryOp<f32>>(value: T) -> T {
    old_op_bounds(value)
}
fn arch_bounds<T: SimdArch>(value: T) -> T {
    old_arch_bounds(value)
}
fn alignment_bounds<T: Alignment>(value: T) -> T {
    old_arch_bounds(value)
}

#[test]
fn downstream_marker_implementations_keep_bounds_and_values() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .without_time()
        .with_span_events(FmtSpan::FULL)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("one subscriber in the isolated marker test binary");
    let _test = tracing::info_span!("marker_supertraits").entered();
    let arrange = tracing::info_span!("arrange").entered();
    let values = [1.0_f32, 2.0, 4.0];
    let marker = ConsumerMarker;
    drop(arrange);

    let act = tracing::info_span!("act").entered();
    let scan = values.into_iter().fold(
        <ConsumerMarker as ScanOp<f32>>::identity(),
        <ConsumerMarker as ScanOp<f32>>::combine,
    );
    let reduction = values.into_iter().fold(
        <ConsumerMarker as ReductionOp<f32>>::identity_scalar(),
        <ConsumerMarker as ReductionOp<f32>>::scalar_accumulate,
    );
    let element = <ConsumerMarker as ElementOp<f32>>::apply_scalar(marker, 0.5, 1.5);
    let unary = <ConsumerMarker as UnaryOp<f32>>::apply_scalar(marker, -2.5);
    let aligned = is_aligned_for_arch::<ConsumerMarker, ConsumerMarker>();
    let _ = (
        scan_bounds(marker),
        mode_bounds(marker),
        reduction_bounds(marker),
        element_bounds(marker),
        unary_bounds(marker),
        arch_bounds(marker),
        alignment_bounds(marker),
    );
    drop(act);

    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(scan.to_bits(), 7.0_f32.to_bits());
    assert_eq!(reduction.to_bits(), 7.0_f32.to_bits());
    assert_eq!(element.to_bits(), 2.0_f32.to_bits());
    assert_eq!(unary.to_bits(), (-2.5_f32).to_bits());
    assert!(aligned);
    const { assert!(<ConsumerMarker as ScanMode>::IS_INCLUSIVE) };
    assert!(!ConsumerMarker::is_runtime_supported());
}

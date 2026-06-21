//! Typestate markers for statically and dynamically guaranteed slice alignment.
//!
//! Provides types that represent memory alignment bounds at the type level, allowing
//! SIMD operations to dispatch to faster aligned loading instructions (`load_aligned`)
//! instead of unaligned loading instructions safely.

/// Trait representing a memory alignment guarantee.
///
/// Implemented by Zero-Sized Types (ZSTs) representing the alignment layout of the slice.
pub trait Alignment: crate::private::Sealed + Send + Sync + 'static + Copy + Clone {
    /// The alignment boundary in bytes, if statically guaranteed.
    ///
    /// Set to `Some(N)` for aligned views where `N` is a power of two, or `None` if
    /// there is no static alignment guarantee (i.e. `Unaligned`).
    const ALIGNMENT: Option<usize>;

    /// Whether static alignment is guaranteed.
    const IS_ALIGNED: bool;

    /// The alignment boundary in bytes (0 if unaligned).
    const ALIGN_BYTES: usize;
}

/// A static alignment guarantee of `A` bytes.
///
/// Guaranteed at compile time to represent a power-of-two alignment boundary.
/// If `A` is not a power of two, it will fail to compile due to a const assertion check.
///
/// # Examples
///
/// ```rust
/// use hermes_simd_core::align::{Aligned, Alignment};
///
/// assert_eq!(Aligned::<32>::ALIGNMENT, Some(32));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aligned<const A: usize>;

impl<const A: usize> Aligned<A> {
    const _CHECK_POWER_OF_TWO: () = {
        assert!(
            A.is_power_of_two(),
            "Alignment boundary must be a power of two"
        );
    };
}

/// No static alignment guarantee.
///
/// Represents raw slice memory that might start at any byte boundary. Dispatches
/// to unaligned memory access operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unaligned;

impl<const A: usize> Alignment for Aligned<A> {
    const ALIGNMENT: Option<usize> = {
        // Evaluate the const assertion when the trait is implemented.
        let _ = Self::_CHECK_POWER_OF_TWO;
        Some(A)
    };
    const IS_ALIGNED: bool = true;
    const ALIGN_BYTES: usize = A;
}

impl Alignment for Unaligned {
    const ALIGNMENT: Option<usize> = None;
    const IS_ALIGNED: bool = false;
    const ALIGN_BYTES: usize = 0;
}

impl<const A: usize> crate::private::Sealed for Aligned<A> {}
impl crate::private::Sealed for Unaligned {}

/// Helper to check if the alignment `Align` is sufficient for architecture `Arch` vector register width.
#[inline(always)]
pub fn is_aligned_for_arch<Arch: crate::arch::SimdArch, Align: Alignment>() -> bool {
    if !Align::IS_ALIGNED {
        return false;
    }
    let req_align = Arch::REGISTER_WIDTH_BITS as usize / 8;
    Align::ALIGN_BYTES >= req_align
}

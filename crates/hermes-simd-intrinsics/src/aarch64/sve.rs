//! AArch64 SVE (Scalable Vector Extension) architecture marker.
//!
//! Stable Rust does not yet expose production-ready SVE vector register types
//! for Hermes' generic [`SimdKernel`](hermes_simd_core::kernel::SimdKernel)
//! contract. This module therefore provides an honest portable implementation;
//! native hardware capability is probed separately and never confused with the
//! emulated execution path:
//! [`crate::SveArch`] is callable and value-preserving through lane-emulated
//! monomorphized arrays, while the native SVE intrinsic backend remains tracked
//! as a separate backlog item.
//!
//! The emulated lane counts model a 512-bit SVE vector shape (`16xf32`,
//! `8xf64`) so downstream generic code can compile and execute without a
//! unavailable hardware branch. This is not a hardware-SVE performance claim.

use hermes_simd_core::arch::{IsaFamily, SimdArch};

/// AArch64 SVE architecture ZST marker.
///
/// The current stable implementation uses the same real lane-emulated kernel
/// family as other non-native `(architecture, scalar)` pairs. Native SVE
/// intrinsics can replace this implementation once stable Rust exposes the
/// required scalable-vector types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SveArch;

impl SveArch {
    /// Returns whether the current process can execute native AArch64 SVE.
    ///
    /// The stable Hermes backend remains lane-emulated, so this capability is
    /// intentionally separate from [`SimdArch::is_runtime_supported`]: the
    /// latter reports that the emulated backend is safe to construct on every
    /// host, while this probe reports hardware capability only.
    #[inline]
    pub fn is_native_hardware_supported() -> bool {
        #[cfg(all(target_arch = "aarch64", feature = "std"))]
        {
            std::arch::is_aarch64_feature_detected!("sve")
        }
        #[cfg(all(target_arch = "aarch64", not(feature = "std")))]
        {
            cfg!(target_feature = "sve")
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }
}

impl SimdArch for SveArch {
    const NAME: &'static str = "sve-emulated";
    const REGISTER_WIDTH_BITS: u32 = 512;
    const ISA_FAMILY: IsaFamily = IsaFamily::AArch64;
    const FMA_THROUGHPUT_HINT: u32 = 4;

    #[inline]
    fn is_runtime_supported() -> bool {
        // This marker executes the lane-emulated implementation, not native
        // SVE instructions; native hardware capability is exposed separately.
        true
    }
}

crate::impl_emulated_kernel!(SveArch, f32, 16, cfg(all()));
crate::impl_emulated_kernel!(SveArch, f64, 8, cfg(all()));

#[cfg(test)]
mod tests {
    use super::SveArch;
    use hermes_simd_core::kernel::SimdKernel;

    #[test]
    fn emulated_sve_f32_kernel_is_value_semantic() {
        // Runtime SVE detection is intentionally independent from compile-time
        // `target_feature`: a host may support SVE even when this stable,
        // lane-emulated crate was not compiled with `+sve`.
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!SveArch::is_native_hardware_supported());
        let lhs = [1.0_f32; 16];
        let rhs = [2.0_f32; 16];
        let addend = [3.0_f32; 16];
        let mask = [
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, true, false,
        ];

        // SAFETY: the emulated SVE backend has no hardware target-feature
        // precondition; all pointers are valid for the lane count used.
        unsafe {
            let product = <SveArch as SimdKernel<f32>>::fmadd(lhs, rhs, addend);
            assert_eq!(product, [5.0; 16]);
            assert_eq!(
                <SveArch as SimdKernel<f32>>::masked_sum_reduce(product, mask),
                40.0
            );

            let compact = <SveArch as SimdKernel<f32>>::compress(product, mask);
            assert_eq!(&compact[..8], &[5.0; 8]);

            let expanded = <SveArch as SimdKernel<f32>>::expand(compact, mask, [0.0; 16]);
            assert_eq!(
                expanded,
                [5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0]
            );
        }
    }

    #[test]
    fn emulated_sve_f64_kernel_loads_gathers_and_stores() {
        let source = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let indices = [7, 6, 5, 4, 3, 2, 1, 0];
        let mut out = [0.0_f64; 8];
        let mask = [true, true, false, false, true, true, false, false];

        // SAFETY: the emulated SVE backend has no hardware target-feature
        // precondition; source and out are valid for eight f64 lanes.
        unsafe {
            let loaded = <SveArch as SimdKernel<f64>>::load_unaligned(source.as_ptr());
            assert_eq!(<SveArch as SimdKernel<f64>>::sum_reduce(loaded), 36.0);

            let gathered = <SveArch as SimdKernel<f64>>::gather(source.as_ptr(), indices);
            assert_eq!(gathered, [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

            <SveArch as SimdKernel<f64>>::masked_store_unaligned(out.as_mut_ptr(), mask, gathered);
        }

        assert_eq!(out, [8.0, 7.0, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0]);
    }
}

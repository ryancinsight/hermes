//! High-performance, zero-overhead SIMD abstraction library.
//!
//! # Architecture
//!
//! `hermes-simd` is the public facade for the hermes-simd workspace:
//!
//! - [`hermes_simd_core`] — core abstractions, traits, views
//! - [`hermes_simd_intrinsics`] — architecture-specific kernels
//! - [`hermes_simd_macros`] — proc-macro code generation
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `std` (default) | Enables runtime CPU feature detection |
//! | `avx512` | Build hint: compile with `-C target-feature=+avx512f` |
//! | `sparse` | Enable `SparseView` and SpMV kernels |
//! | `tiling` | Enable const-generic tiled dot product |
//! | `macros` | Enable proc-macro code generation helpers |
//! | `bytemuck` | Enable safe type casting via `bytemuck` |
//! | `portable-simd` | Enable nightly `std::simd` backend |
//!
//! # Usage Examples
//!
//! **Dense sum (runtime dispatch):**
//! ```rust
//! use hermes_simd::sum;
//! let data = vec![1.0f32; 1024];
//! assert_eq!(sum(&data), 1024.0);
//! ```
//!
//! **Masked dot product:**
//! ```rust
//! use hermes_simd::masked_dot;
//! let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let b = vec![1.0f32; 5];
//! let mask = vec![true, false, true, false, true];
//! assert_eq!(masked_dot(&a, &b, &mask).unwrap(), 9.0); // 1+3+5
//! ```

#![warn(missing_docs)]
#![allow(unused_unsafe)]

// Re-export core types
pub use hermes_simd_core::{
    arch::SimdArch,
    align::{Alignment, Aligned, Unaligned},
    execution::{ExecutionMode, Unmasked, Masked},
    kernel::SimdKernel,
    view::{SimdView, SimdError, TileView, TileMatrixMultiply, Vector, Mask},
    vec::AlignedVec,
    cow::{SimdCow, ArchivedSimdCow, SimdCowResolver},
    mask::BitMask,
    compute::ComputeView,
    bitboard::{BitBoardView, BitBoardKernel},
    scalar::{FloatElement, Scalar as SimdScalar},
    current_numa_node, refresh_numa_node, verify_numa_locality, NumaBinding, NumaTopologyService, numa_node_count, numa_node_distance,
};

// Re-export sparse types
pub use hermes_simd_core::sparse::{
    SparseFormat, SparseView,
    Csr, SellP, BlockedCoo, DenseWithMask,
    CsrData, SellPData, BlockedCooData, DenseWithMaskData,
};

// Re-export tiling
pub use hermes_simd_core::tiling::{tiled_dot, TilingPolicy, TilingStrategy, tiled_gemv};

// Re-export concrete ZST architecture markers
pub use hermes_simd_intrinsics::{
    Scalar, Avx2, Avx512, Neon,
    Swar, KoggeStone, Hyperbola, Magic, HybridSwarMagic, SwarUtils,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use hermes_simd_intrinsics::{AmxBf16, AmxInt8, AmxConfig, AmxSession, AmxBatchSession};

pub use hermes_numeric::{
    F16, F32, F64, Bf16, Bf8, Bf4, F8, F4, I8, I16, I32,
    Packable4, Packed4Slice, Packed4SliceMut,
    PackedBf4Slice, PackedBf4SliceMut, PackedF4Slice, PackedF4SliceMut,
    Packed4Vec, Packed4Iter, PackedBf4Vec, PackedF4Vec,
    Packed4Cow, PackedBf4Cow, PackedF4Cow,
};

/// Runtime CPU feature detection utilities.
pub mod cpu;
/// Chess board attack generation kernels using bitboards and SWAR.
pub mod attacks;
/// Dynamic dispatcher choosing optimal backends based on hardware/layout.
pub mod dispatcher;

/// Tiled matrix multiplication dispatch and kernel interfaces.
pub mod tile_matmul;

/// Runtime-dispatched SIMD abstractions and dynamic facade.
pub mod dispatch;

pub use cpu::{AmxSupport, Avx512Support};
pub use attacks::{rook_attacks, bishop_attacks, queen_attacks};
pub use tile_matmul::{TiledGemm, gemm, dispatch_tile_matmul, unpack_int4};
pub use dispatcher::{AdaptiveDispatcher, DispatchDecision};

// Re-export the generic dispatch operations. These monomorphize at call sites:
// calling `sum::<f32>(data)` produces the f32 specialization.
pub use dispatch::{
    // Core trait — sealed; implemented for f32 and f64.
    SimdOps,
    // Generic free functions — the primary public API.
    sum,
    dot,
    elementwise_mul,
    masked_sum,
    masked_dot,
    masked_add,
    tiled_gemm,
    // Sparse operations — generic entry points.
    spmv_csr,
    spmv_bcoo4x4,
    spmv_bcoo8x8,
    spmv_dense_masked,
    spmv_sellp4,
    spmv_sellp8,
};

/// Target-specific, runtime-dispatched SIMD view wrapper.
pub enum DispatchedView<'a, T, Align = Unaligned, Mode = Unmasked, Ref = &'a [T]>
where
    Align: hermes_simd_core::align::Alignment,
    Mode: hermes_simd_core::execution::ExecutionMode,
    Ref: core::ops::Deref<Target = [T]>,
{
    /// AVX-512 architecture target.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx512(SimdView<'a, T, Avx512, Align, Mode, Ref>),
    /// AVX2 architecture target.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(SimdView<'a, T, Avx2, Align, Mode, Ref>),
    /// NEON architecture target.
    #[cfg(target_arch = "aarch64")]
    Neon(SimdView<'a, T, Neon, Align, Mode, Ref>),
    /// Fallback scalar target.
    Scalar(SimdView<'a, T, Scalar, Align, Mode, Ref>),
}

/// Dispatches a shared slice into the best matching `DispatchedView` based on runtime CPU feature detection.
#[inline]
pub fn dispatch_view<'a, T, Align>(data: &'a [T]) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return SimdView::<T, Avx512, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Avx512);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return SimdView::<T, Avx2, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdView::<T, Neon, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Neon);
    }
    SimdView::<T, Scalar, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Scalar)
}

/// Dispatches a mutable slice into the best matching `DispatchedView` based on runtime CPU feature detection.
#[inline]
pub fn dispatch_view_mut<'a, T, Align>(data: &'a mut [T]) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a mut [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx512f") {
            return SimdView::<T, Avx512, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Avx512);
        }
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return SimdView::<T, Avx2, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Avx2);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdView::<T, Neon, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Neon);
    }
    SimdView::<T, Scalar, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Scalar)
}

/// Extension trait for `SimdCow` providing vector-register level operations.
pub trait SimdCowExt<T: SimdScalar, Arch: SimdArch + SimdKernel<T>, Align: Alignment> {
    /// In-place vector-level transformation.
    ///
    /// Promotes `self` to owned if borrowed (zero-copy upgrade), then applies
    /// the function `f` elementwise to each SIMD vector chunk.
    fn transform_vectors<F>(&mut self, f: F)
    where
        F: FnMut(Vector<T, Arch>) -> Vector<T, Arch>;
}

impl<'a, T, Arch, Align> SimdCowExt<T, Arch, Align> for SimdCow<'a, T, Arch, Align>
where
    T: SimdScalar,
    Arch: SimdArch + SimdKernel<T>,
    Align: Alignment,
{
    #[inline]
    fn transform_vectors<F>(&mut self, mut f: F)
    where
        F: FnMut(Vector<T, Arch>) -> Vector<T, Arch>,
    {
        let owned_vec = self.to_mut();
        let len = owned_vec.len();
        let lane_count = Arch::LANE_COUNT;
        let simd_len = (len / lane_count) * lane_count;
        let slice = owned_vec.as_mut_slice();

        let mut i = 0;
        while i < simd_len {
            unsafe {
                let ptr = slice.as_mut_ptr().add(i);
                let vec = if Align::IS_ALIGNED {
                    Vector::load_aligned(ptr)
                } else {
                    Vector::load_unaligned(ptr)
                };
                let res = f(vec);
                if Align::IS_ALIGNED {
                    res.store_aligned(ptr);
                } else {
                    res.store_unaligned(ptr);
                }
            }
            i += lane_count;
        }

        if simd_len < len {
            let mut tail_buf = [T::ZERO; 128];
            unsafe {
                for idx in simd_len..len {
                    tail_buf[idx - simd_len] = slice[idx];
                }
                let vec = Vector::load_unaligned(tail_buf.as_ptr());
                let res = f(vec);
                res.store_unaligned(tail_buf.as_mut_ptr());
                for idx in simd_len..len {
                    slice[idx] = tail_buf[idx - simd_len];
                }
            }
        }
    }
}

/// Extension trait for packed Bf4 Clone-on-Write containers.
pub trait PackedBf4CowExt<'a> {
    /// Unpack Bf4 elements directly to a `SimdCow` of `Bf16` with zero intermediate allocations.
    fn unpack_to_bf16_cow<Arch, Align>(&self) -> SimdCow<'static, Bf16, Arch, Align>
    where
        Arch: SimdArch,
        Align: Alignment;
}

/// Extension trait for packed F4 Clone-on-Write containers.
pub trait PackedF4CowExt<'a> {
    /// Unpack F4 elements directly to a `SimdCow` of `F32` with zero intermediate allocations.
    fn unpack_to_f32_cow<Arch, Align>(&self) -> SimdCow<'static, F32, Arch, Align>
    where
        Arch: SimdArch,
        Align: Alignment;
}

impl<'a> PackedBf4CowExt<'a> for Packed4Cow<'a, Bf4> {
    #[inline]
    fn unpack_to_bf16_cow<Arch, Align>(&self) -> SimdCow<'static, Bf16, Arch, Align>
    where
        Arch: SimdArch,
        Align: Alignment,
    {
        let len = self.len();
        let mut dest = AlignedVec::with_capacity(len);
        unsafe {
            dest.set_len(len);
        }
        self.as_view().unpack_to_bf16(dest.as_mut_slice());
        SimdCow::Owned(dest)
    }
}

impl<'a> PackedF4CowExt<'a> for Packed4Cow<'a, F4> {
    #[inline]
    fn unpack_to_f32_cow<Arch, Align>(&self) -> SimdCow<'static, F32, Arch, Align>
    where
        Arch: SimdArch,
        Align: Alignment,
    {
        let len = self.len();
        let mut dest = AlignedVec::with_capacity(len);
        unsafe {
            dest.set_len(len);
        }
        self.as_view().unpack_to_f32(dest.as_mut_slice());
        SimdCow::Owned(dest)
    }
}

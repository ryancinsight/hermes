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

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![allow(
    unused_unsafe,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::manual_memcpy
)]

extern crate alloc;

pub use hermes_simd_core::{
    align::{Aligned, Alignment, Unaligned},
    arch::SimdArch,
    bitboard::{BitBoardKernel, BitBoardView},
    // ComputeView extension
    compute::{ComputeReduce, ComputeView},
    cow::{ArchivedPacked4Cow, ArchivedSimdCow, Packed4CowResolver, SimdCow, SimdCowResolver},
    current_numa_node,
    execution::{ExecutionMode, Masked, Unmasked},
    // Chunk iterators
    iter::{SimdChunks, SimdChunksMut, ZipChunks, ZipChunksMut},
    kernel::SimdKernel,
    mask::BitMask,
    numa_node_count,
    numa_node_distance,
    refresh_numa_node,
    scalar::{CastFrom, CastTo, FloatElement, Scalar as SimdScalar},
    vec::AlignedVec,
    verify_numa_locality,
    view::{Mask, SimdError, SimdView, TileMatrixMultiply, TileView, Vector},
    // Unary strategy ZSTs
    Abs,
    Add,
    BitAnd,
    BitOr,
    BitXor,
    Clamp,
    Div,
    Dot,
    ElementOp,
    Exclusive,
    // Extended strategy ZSTs (v2)
    FmaAdd,
    Inclusive,
    Mul,
    Neg,
    NumaBinding,
    NumaTopologyService,
    Product,
    // Operation strategy ZSTs and sealed traits — zero-cost, erased at monomorphization.
    ReductionOp,
    ScanAdd,
    ScanMax,
    ScanMin,
    ScanMode,
    ScanMul,
    // Scan strategy ZSTs
    ScanOp,
    Sqrt,
    Sub,
    Sum,
    UnaryOp,
};

// Re-export sparse types
pub use hermes_simd_core::sparse::{
    BlockedCoo,
    BlockedCooData,
    // Format-to-owned-storage mapping for Cow containers
    CowFormat,
    Csr,
    CsrData,
    DenseWithMask,
    DenseWithMaskData,
    OwnedBlockedCoo,
    // Owned heap-backed sparse storage types
    OwnedCsr,
    OwnedDenseWithMask,
    OwnedSellP,
    SellP,
    SellPData,
    // Generic Clone-on-Write sparse container
    SparseCow,
    SparseFormat,
    SparseOps,
    SparseSpMv,
    SparseView,
};

// Re-export tiling
pub use hermes_simd_core::tiling::{tiled_dot, tiled_gemv, TilingPolicy, TilingStrategy};

// Re-export tensor views
pub use hermes_simd_core::tensor::{ColMajor, RowMajor, TensorCow, TensorError, TensorView};

// Re-export concrete ZST architecture markers
pub use hermes_simd_intrinsics::{
    Avx2, Avx512, HybridSwarMagic, Hyperbola, KoggeStone, Magic, Neon, Scalar, SveArch, Swar,
    SwarUtils,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use hermes_simd_intrinsics::{AmxBatchSession, AmxBf16, AmxConfig, AmxInt8, AmxSession};

pub use hermes_numeric::{
    Bf16, Bf4, Bf8, Packable4, Packed4Cow, Packed4Iter, Packed4Slice, Packed4SliceMut, Packed4Vec,
    PackedBf4Cow, PackedBf4Slice, PackedBf4SliceMut, PackedBf4Vec, PackedF4Cow, PackedF4Slice,
    PackedF4SliceMut, PackedF4Vec, F16, F32, F4, F64, F8, I16, I32, I8,
};

// Re-export monomorphized vector register types and PreferredArch
pub use hermes_simd_types::{
    MaskBf16, MaskBf4, MaskBf8, MaskF16, MaskF32, MaskF4, MaskF64, MaskF8, MaskI16, MaskI32,
    MaskI8, PreferredArch, ScalarBf16, ScalarBf4, ScalarBf8, ScalarF16, ScalarF32, ScalarF4,
    ScalarF64, ScalarF8, ScalarI16, ScalarI32, ScalarI8, ScalarMaskF32, ScalarMaskF64, SimdBf16,
    SimdBf4, SimdBf8, SimdF16, SimdF32, SimdF4, SimdF64, SimdF8, SimdI16, SimdI32, SimdI8,
    SimdMaskBf16, SimdMaskBf4, SimdMaskBf8, SimdMaskF16, SimdMaskF32, SimdMaskF4, SimdMaskF64,
    SimdMaskF8, SimdMaskI16, SimdMaskI32, SimdMaskI8, VectorBf16, VectorBf4, VectorBf8, VectorF16,
    VectorF32, VectorF4, VectorF64, VectorF8, VectorI16, VectorI32, VectorI8,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use hermes_simd_types::{
    Avx2Bf16, Avx2Bf4, Avx2Bf8, Avx2F16, Avx2F32, Avx2F4, Avx2F64, Avx2F8, Avx2I16, Avx2I32,
    Avx2I8, Avx2MaskBf16, Avx2MaskF16, Avx2MaskF32, Avx2MaskF64, Avx512Bf16, Avx512Bf4, Avx512Bf8,
    Avx512F16, Avx512F32, Avx512F4, Avx512F64, Avx512F8, Avx512I16, Avx512I32, Avx512I8,
    Avx512MaskBf16, Avx512MaskF16, Avx512MaskF32, Avx512MaskF64,
};

#[cfg(target_arch = "aarch64")]
pub use hermes_simd_types::{
    NeonBf16, NeonBf4, NeonBf8, NeonF16, NeonF32, NeonF4, NeonF64, NeonF8, NeonI16, NeonI32,
    NeonI8, NeonMaskBf16, NeonMaskF16, NeonMaskF32, NeonMaskF64,
};

/// Chess board attack generation kernels using bitboards and SWAR.
pub mod attacks;
/// Runtime CPU feature detection utilities.
pub mod cpu;
/// Dynamic dispatcher choosing optimal backends based on hardware/layout.
pub mod dispatcher;

/// Tiled matrix multiplication dispatch and kernel interfaces.
pub mod tile_matmul;

/// Runtime-dispatched SIMD abstractions and dynamic facade.
pub mod dispatch;
/// Explicit runtime target tokens and forced dispatch helpers.
pub mod target;

pub use attacks::{bishop_attacks, queen_attacks, rook_attacks};
pub use cpu::{AmxSupport, Avx512Support};
pub use dispatcher::{AdaptiveDispatcher, DispatchDecision};
pub use target::{dispatch_view_mut_to, dispatch_view_to, TargetId};
pub use tile_matmul::{dispatch_tile_matmul, gemm, unpack_int4, TiledGemm};

// Re-export the generic dispatch operations. These monomorphize at call sites:
// calling `sum::<f32>(data)` produces the f32 specialization.
pub use dispatch::{
    abs_max,
    abs_sum,
    argmax,
    argmin,
    axpy,
    axpy_rows,
    axpy_rows_batch,
    dot,
    elementwise_add,
    elementwise_div,
    elementwise_mul,
    elementwise_sub,
    gemv,
    gemv_transpose,
    interleaved_complex_dot,
    interleaved_complex_dot_runtime,
    interleaved_complex_mul_assign,
    interleaved_complex_mul_assign_runtime,
    masked_add,
    masked_dot,
    masked_sum,
    max,
    min,
    ntt_butterfly_stage_u64,
    scale,
    spmv_bcoo,
    // Sparse operations — generic entry points.
    spmv_csr,
    spmv_dense_masked,
    spmv_sellp,
    // Generic free functions — the primary public API.
    sum,
    tiled_gemm,
    // Core trait — sealed; implemented for f32 and f64.
    SimdOps,
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
#[allow(unreachable_code)]
pub fn dispatch_view<'a, T, Align>(
    data: &'a [T],
) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a [T]>::new(data)
                    .map(DispatchedView::Avx512);
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a [T]>::new(data)
                    .map(DispatchedView::Avx2);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a [T]>::new(data)
                    .map(DispatchedView::Avx512);
            }
            if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a [T]>::new(data)
                    .map(DispatchedView::Avx2);
            }
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
#[allow(unreachable_code)]
pub fn dispatch_view_mut<'a, T, Align>(
    data: &'a mut [T],
) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a mut [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a mut [T]>::new_mut(data)
                    .map(DispatchedView::Avx512);
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a mut [T]>::new_mut(data)
                    .map(DispatchedView::Avx2);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a mut [T]>::new_mut(data)
                    .map(DispatchedView::Avx512);
            }
            if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a mut [T]>::new_mut(data)
                    .map(DispatchedView::Avx2);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return SimdView::<T, Neon, Align, Unmasked, &'a mut [T]>::new_mut(data)
            .map(DispatchedView::Neon);
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

/// Extension trait for packed Clone-on-Write containers to support zero-copy unpacking directly into `SimdCow`.
pub trait Packed4CowExt<'a, T: Packable4> {
    /// Unpack packed elements directly to a `SimdCow` of wider precision with zero intermediate allocations.
    fn unpack_to_cow<Arch, Align>(&self) -> SimdCow<'static, T::Unpacked, Arch, Align>
    where
        Arch: SimdArch,
        Align: Alignment;
}

impl<'a, T: Packable4> Packed4CowExt<'a, T> for Packed4Cow<'a, T> {
    #[inline]
    fn unpack_to_cow<Arch, Align>(&self) -> SimdCow<'static, T::Unpacked, Arch, Align>
    where
        Arch: SimdArch,
        Align: Alignment,
    {
        let len = self.len();
        let mut dest = AlignedVec::with_capacity(len);
        unsafe {
            dest.set_len(len);
        }
        let view = self.as_view();
        let n = view.len().min(dest.len());
        let even_len = (n / 2) * 2;
        T::unpack_slice_packed(
            &view.as_packed_slice()[..even_len / 2],
            &mut dest[..even_len],
        );
        if n % 2 != 0 {
            if let Some(b) = view.get(n - 1) {
                dest[n - 1] = T::unpack_single(b);
            }
        }
        SimdCow::Owned(dest)
    }
}

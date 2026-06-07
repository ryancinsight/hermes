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
#![warn(missing_docs)]
#![allow(
    unused_unsafe,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::manual_memcpy
)]

extern crate alloc;

pub use hermes_simd_core::{
    arch::SimdArch,
    align::{Alignment, Aligned, Unaligned},
    execution::{ExecutionMode, Unmasked, Masked},
    kernel::SimdKernel,
    view::{SimdView, SimdError, TileView, TileMatrixMultiply, Vector, Mask},
    vec::AlignedVec,
    cow::{SimdCow, ArchivedSimdCow, SimdCowResolver, ArchivedPacked4Cow, Packed4CowResolver},
    mask::BitMask,
    bitboard::{BitBoardView, BitBoardKernel},
    scalar::{FloatElement, Scalar as SimdScalar, CastFrom, CastTo},
    current_numa_node, refresh_numa_node, verify_numa_locality, NumaBinding, NumaTopologyService, numa_node_count, numa_node_distance,
    // Operation strategy ZSTs and sealed traits — zero-cost, erased at monomorphization.
    ReductionOp, ElementOp, UnaryOp,
    Sum, Dot, Mul, Add, Sub, Div, BitAnd, BitOr, BitXor,
    // Unary strategy ZSTs
    Abs, Neg, Sqrt, Clamp,
    // Scan strategy ZSTs
    ScanOp, ScanMode, ScanAdd, ScanMul, ScanMin, ScanMax, Inclusive, Exclusive,
    // Extended strategy ZSTs (v2)
    FmaAdd, Product,
    // Chunk iterators
    iter::{SimdChunks, ZipChunks, ZipChunksMut, SimdChunksMut},
    // ComputeView extension
    compute::{ComputeView, ComputeReduce},
};

// Re-export sparse types
pub use hermes_simd_core::sparse::{
    SparseFormat, SparseView, SparseSpMv, SparseOps,
    Csr, SellP, BlockedCoo, DenseWithMask,
    CsrData, SellPData, BlockedCooData, DenseWithMaskData,
    // Clone-on-Write sparse containers
    SparseCow, CsrCow, SellPCow, BlockedCooCow, DenseWithMaskCow,
    // Owned heap-backed sparse storage types
    OwnedCsr, OwnedSellP, OwnedBlockedCoo, OwnedDenseWithMask,
};

// Re-export tiling
pub use hermes_simd_core::tiling::{tiled_dot, TilingPolicy, TilingStrategy, tiled_gemv};

// Re-export tensor views, softmax, matmul, norms, and layer norm
pub use hermes_simd_core::tensor::{
    TensorView, TensorCow, TensorError, RowMajor, ColMajor,
};
pub use hermes_simd_core::tensor::softmax::{
    softmax_inplace, softmax as softmax_alloc,
    softmax_2d_rows_inplace, softmax_2d_rows,
};
pub use hermes_simd_core::tensor::ops::{
    matmul, batch_matmul, matmul_to, batch_matmul_to, DefaultTilePolicy,
};
pub use hermes_simd_core::tensor::norm::{
    norm_l1, norm_l2, norm_linf, normalize_l2_inplace, row_norms_l2, SquaredSum,
};
pub use hermes_simd_core::tensor::layer_norm::{
    layer_norm_inplace, layer_norm,
};

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

// Re-export monomorphized vector register types and PreferredArch
pub use hermes_simd_types::{
    PreferredArch,
    VectorF32, VectorF64, VectorF16, VectorBf16, VectorBf8, VectorBf4, VectorF8, VectorF4, VectorI8, VectorI16, VectorI32,
    MaskF32, MaskF64, MaskF16, MaskBf16, MaskBf8, MaskBf4, MaskF8, MaskF4, MaskI8, MaskI16, MaskI32,
    SimdF32, SimdF64, SimdF16, SimdBf16, SimdBf8, SimdBf4, SimdF8, SimdF4, SimdI8, SimdI16, SimdI32,
    SimdMaskF32, SimdMaskF64, SimdMaskF16, SimdMaskBf16, SimdMaskBf8, SimdMaskBf4, SimdMaskF8, SimdMaskF4, SimdMaskI8, SimdMaskI16, SimdMaskI32,
    ScalarF32, ScalarF64, ScalarF16, ScalarBf16, ScalarBf8, ScalarBf4, ScalarF8, ScalarF4, ScalarI8, ScalarI16, ScalarI32,
    ScalarMaskF32, ScalarMaskF64,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use hermes_simd_types::{
    Avx2F32, Avx2F64, Avx2F16, Avx2Bf16, Avx2Bf8, Avx2Bf4, Avx2F8, Avx2F4, Avx2I8, Avx2I16, Avx2I32,
    Avx2MaskF32, Avx2MaskF64, Avx2MaskF16, Avx2MaskBf16,
    Avx512F32, Avx512F64, Avx512F16, Avx512Bf16, Avx512Bf8, Avx512Bf4, Avx512F8, Avx512F4, Avx512I8, Avx512I16, Avx512I32,
    Avx512MaskF32, Avx512MaskF64, Avx512MaskF16, Avx512MaskBf16,
};

#[cfg(target_arch = "aarch64")]
pub use hermes_simd_types::{
    NeonF32, NeonF64, NeonF16, NeonBf16, NeonBf8, NeonBf4, NeonF8, NeonF4, NeonI8, NeonI16, NeonI32,
    NeonMaskF32, NeonMaskF64, NeonMaskF16, NeonMaskBf16,
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
    min,
    max,
    scale,
    argmin,
    argmax,
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
#[allow(unreachable_code)]
pub fn dispatch_view<'a, T, Align>(data: &'a [T]) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Avx512);
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Avx2);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Avx512);
            }
            if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a [T]>::new(data).map(DispatchedView::Avx2);
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
pub fn dispatch_view_mut<'a, T, Align>(data: &'a mut [T]) -> Option<DispatchedView<'a, T, Align, Unmasked, &'a mut [T]>>
where
    T: FloatElement,
    Align: Alignment,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "std")]
        {
            if std::is_x86_feature_detected!("avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Avx512);
            }
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Avx2);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            if cfg!(target_feature = "avx512f") {
                return SimdView::<T, Avx512, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Avx512);
            }
            if cfg!(target_feature = "avx2") && cfg!(target_feature = "fma") {
                return SimdView::<T, Avx2, Align, Unmasked, &'a mut [T]>::new_mut(data).map(DispatchedView::Avx2);
            }
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

/// Extension trait for packed Clone-on-Write containers to support zero-copy unpacking directly into `SimdCow`.
pub trait Packed4CowExt<'a, T: Packable4> {
    /// Unpack packed elements directly to a `SimdCow` of wider precision with zero intermediate allocations.
    fn unpack_to_cow<Arch, Align>(&self) -> SimdCow<'static, T::Unpacked, Arch, Align>
    where
        Arch: SimdArch,
        Align: Alignment;
}

// Private helper trait to dispatch unpacking to the best hardware backends
pub(crate) trait HardwareUnpack: Packable4 {
    fn hardware_unpack(packed: &[u8], unpacked: &mut [Self::Unpacked]);
}

impl HardwareUnpack for Bf4 {
    #[inline(always)]
    fn hardware_unpack(packed: &[u8], unpacked: &mut [Bf16]) {
        #[cfg(target_arch = "x86_64")]
        {
            hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_packed_bf4_to_bf16(packed, unpacked);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            <Self as Packable4>::unpack_slice_packed(packed, unpacked);
        }
    }
}

impl HardwareUnpack for F4 {
    #[inline(always)]
    fn hardware_unpack(packed: &[u8], unpacked: &mut [F32]) {
        #[cfg(target_arch = "x86_64")]
        {
            hermes_simd_intrinsics::x86_64::avx512_tiling::unpack_packed_f4_to_f32(packed, unpacked);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            <Self as Packable4>::unpack_slice_packed(packed, unpacked);
        }
    }
}

impl<'a, T: Packable4 + HardwareUnpack> Packed4CowExt<'a, T> for Packed4Cow<'a, T> {
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
        T::hardware_unpack(&view.as_packed_slice()[..even_len / 2], &mut dest[..even_len]);
        if n % 2 != 0 {
            if let Some(b) = view.get(n - 1) {
                dest[n - 1] = T::unpack_single(b);
            }
        }
        SimdCow::Owned(dest)
    }
}

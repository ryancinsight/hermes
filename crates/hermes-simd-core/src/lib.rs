//! Core abstractions for `hermes-simd`.
//!
//! # Module Organization
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`arch`] | `SimdArch` marker trait with architecture constants |
//! | [`align`] | `Alignment`, `Aligned<N>`, `Unaligned` typestates |
//! | [`execution`] | `ExecutionMode`, `Unmasked`, `Masked` ZSTs |
//! | [`kernel`] | `SimdKernel<T>` trait — full SIMD operation surface |
//! | [`scalar`] | `Scalar` sealed element trait |
//! | [`mask`] | `BitMask<N>` bit-packed lane mask |
//! | [`ops`] | `ReductionOp<T>`, `ElementOp<T>` ZST strategies |
//! | [`view`] | `SimdView`, `SimdError` — safe typed slice views |
//! | [`tiling`] | Const-generic tiled dot product and `TilingPolicy` |
//! | [`sparse`] | `SparseView`, format ZSTs, data structs, SpMV kernels |
//! | [`vec`] | `AlignedVec` — heap-allocated aligned vector |
//! | [`cow`] | `SimdCow` — SIMD-aware copy-on-write |
//! | [`tensor`] | N-D tensor views, GEMM, softmax, LayerNorm, Attention |

#![cfg_attr(not(feature = "std"), no_std)]

#![warn(missing_docs)]
#![allow(
    clippy::needless_range_loop,
    clippy::let_unit_value,
    clippy::manual_div_ceil,
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::unit_arg
)]

extern crate alloc;

pub mod arch;
pub mod align;
pub mod execution;
pub mod kernel;
mod kernel_helpers;
pub mod scalar;
pub mod mask;
pub mod ops;
pub mod view;
pub mod tiling;
pub mod sparse;
pub mod vec;
pub mod cow;
pub mod iter;
pub mod compute;
pub mod bitboard;
pub mod numa;
pub mod tensor;

/// Hidden private module containing the sealed trait supertrait.
#[doc(hidden)]
pub mod private {
    pub trait Sealed {}
}

// Re-exports for ergonomic use
pub use arch::{SimdArch, IsaFamily};
pub use align::{Alignment, Aligned, Unaligned};
pub use execution::{ExecutionMode, Unmasked, Masked};
pub use kernel::SimdKernel;
pub use scalar::{Scalar, FloatElement, NumericElement};
pub use mask::BitMask;
pub use ops::{
    ReductionOp, ElementOp, UnaryOp, Sum, Dot, Mul, Add, Sub, Div, BitAnd, BitOr, BitXor, Min, Max,
    Abs, Neg, Sqrt, Clamp, ScanOp, ScanMode, ScanAdd, ScanMul, ScanMin, ScanMax, Inclusive, Exclusive,
    FmaAdd, Product,
};
pub use view::{SimdView, SimdError, TileView, TileMatrixMultiply, Vector, Mask};
pub use tiling::{tiled_dot, TilingPolicy, TilingStrategy, tiled_gemv, tiled_gemm};
pub use sparse::{
    SparseFormat, SparseShape, SparseView,
    Csr, SellP, BlockedCoo, DenseWithMask,
    CsrData, SellPData, BlockedCooData, DenseWithMaskData,
};
pub use vec::AlignedVec;
pub use cow::{SimdCow, ArchivedSimdCow, SimdCowResolver, ArchivedPacked4Cow, Packed4CowResolver};
pub use iter::{SimdChunks, ZipChunks, SimdChunksMut};
pub use compute::ComputeView;
pub use bitboard::{BitBoardView, BitBoardKernel};
pub use numa::{
    NumaAllocator, MnemosyneNumaAllocator, current_numa_node, refresh_numa_node,
    verify_numa_locality, NumaBinding, NumaTopologyService, numa_node_count, numa_node_distance,
};
pub use tensor::{
    TensorView, TensorError, RowMajor, ColMajor,
    softmax::{softmax_inplace, softmax, softmax_2d_rows_inplace, softmax_2d_rows},
    layer_norm::{layer_norm_inplace, layer_norm},
    attention::{attention, batch_attention},
    ops::{matmul, matmul_to, batch_matmul, batch_matmul_to, DefaultTilePolicy},
    norm::{norm_l1, norm_l2, norm_linf, normalize_l2_inplace, row_norms_l2, SquaredSum},
};

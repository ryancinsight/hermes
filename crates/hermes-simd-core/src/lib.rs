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
//! | [`vec`](mod@crate::vec) | `AlignedVec` — heap-allocated aligned vector |
//! | [`cow`] | `SimdCow` — SIMD-aware copy-on-write |
//! | [`tensor`] | N-D tensor views, GEMM, softmax, LayerNorm, Attention |

#![cfg_attr(not(feature = "std"), no_std)]
// Lint policy is inherited from the workspace table (`[lints] workspace = true`).
#![cfg_attr(test, allow(clippy::unwrap_used))]

extern crate alloc;

pub mod align;
pub mod arch;
pub mod bitboard;
pub mod compute;
pub mod cow;
pub mod execution;
pub mod iter;
pub mod kernel;
mod kernel_helpers;
pub mod mask;
pub mod numa;
pub mod ops;
pub mod scalar;
pub mod sparse;
pub mod tensor;
pub mod tiling;
pub mod vec;
pub mod view;

/// Hidden private module containing the sealed trait supertrait.
#[doc(hidden)]
pub mod private {
    pub trait Sealed {}
}

// Re-exports for ergonomic use
pub use align::{Aligned, Alignment, Unaligned};
pub use arch::{IsaFamily, SimdArch};
pub use bitboard::{BitBoardKernel, BitBoardView};
pub use compute::ComputeView;
pub use cow::{ArchivedPacked4Cow, ArchivedSimdCow, Packed4CowResolver, SimdCow, SimdCowResolver};
pub use execution::{ExecutionMode, Masked, Unmasked};
pub use iter::{SimdChunks, SimdChunksMut, ZipChunks};
pub use kernel::SimdKernel;
pub use mask::BitMask;
pub use numa::{
    current_numa_node, refresh_numa_node, verify_numa_locality, MnemosyneNumaAllocator,
    NumaAllocator, NumaBinding,
};
pub use ops::{
    Abs, AbsMax, AbsSum, Add, BitAnd, BitOr, BitXor, Ceil, Clamp, Div, Dot, ElementOp, Exclusive,
    Floor, FmaAdd, Inclusive, Max, Min, Mul, Neg, Popcount, Product, RecipSqrt, ReductionOp, Round,
    ScanAdd, ScanMax, ScanMin, ScanMode, ScanMul, ScanOp, Sqrt, Sub, Sum, Trunc, UnaryOp,
};
pub use scalar::{FloatElement, NumericElement, RoundTiesEven, Scalar};
pub use sparse::{
    BlockedCoo, BlockedCooData, Csr, CsrData, DenseWithMask, DenseWithMaskData, SellP, SellPData,
    SparseFormat, SparseShape, SparseView, Validated, ValidatedData,
};
pub use tensor::{ColMajor, RowMajor, TensorCow, TensorError, TensorView};
pub use tiling::{tiled_dot, tiled_gemm, tiled_gemv, TilingPolicy, TilingStrategy};
pub use vec::AlignedVec;
pub use view::{Mask, SimdError, SimdView, TileMatrixMultiply, TileView, Vector};

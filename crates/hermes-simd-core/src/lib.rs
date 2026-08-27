//! Core abstractions for `hermes-simd`.
//!
//! # Module Organization
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`arch`] | `SimdArch` marker trait with architecture constants |
//! | [`align`] | `Alignment`, `Aligned<N>`, `Unaligned` typestates |
//! | [`execution`] | `ExecutionMode`, `Unmasked`, `Masked` ZSTs |
//! | [`kernel`] | `SimdKernel<T>` aggregate and operation-family facets |
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
// These are library-only and so cannot live in that package-scoped table: a
// panic, a write to the process's stdio, or an undocumented item is a defect in
// library code and routine (or irrelevant) in a test, bench, or example.
#![deny(missing_docs)]
#![deny(clippy::unwrap_used, clippy::print_stdout, clippy::print_stderr)]
#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::print_stdout, clippy::print_stderr)
)]

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
pub use iter::{SimdChunks, SimdChunksMut, SimdIoChunks, ZipChunks};
pub use kernel::{
    BackendKernel, SimdArith, SimdBitwise, SimdCompare, SimdGather, SimdKernel, SimdLoadStore,
    SimdMask, SimdPermute, SimdReduce, SimdStorage,
};
pub use mask::{BitMask, PackedMask};
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
pub use view::{Mask, Simd, SimdChunk, SimdError, SimdView, TileMatrixMultiply, TileView, Vector};

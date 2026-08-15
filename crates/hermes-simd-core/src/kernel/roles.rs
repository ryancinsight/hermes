//! Operation-family capability facets for the sealed backend kernel.

mod arithmetic;
mod bitwise;
mod compare;
mod gather;
mod load_store;
mod mask;
mod permute;
mod reduce;
mod storage;

pub use arithmetic::SimdArith;
pub use bitwise::SimdBitwise;
pub use compare::SimdCompare;
pub use gather::SimdGather;
pub use load_store::SimdLoadStore;
pub use mask::SimdMask;
pub use permute::SimdPermute;
pub use reduce::SimdReduce;
pub use storage::SimdStorage;

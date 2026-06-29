#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

//! Foundational numeric types, traits, and mixed-precision helpers for the hermes ecosystem.
//! Single Source of Truth (SSOT) for all numeric representations.

extern crate alloc;

mod casts;
mod impls;
mod ops;
mod packed;
mod traits;
mod types;

// Re-export core traits
pub use traits::{CastFrom, CastTo, FloatElement, NumericElement};

// Re-export wrapper types
pub use types::{Bf16, Bf4, Bf8, Complex, F16, F32, F4, F64, F8, I16, I32, I8};

// Re-export packed layout structures and functions
pub use packed::{
    unpack_bf4_to_bf16, unpack_bf4_to_bf16_packed, unpack_bf8_to_bf16, unpack_f4_to_f32,
    unpack_f4_to_f32_packed, unpack_f8_to_f32, Packable4, Packed4Cow, Packed4Iter, Packed4Slice,
    Packed4SliceMut, Packed4Vec, PackedBf4Cow, PackedBf4Slice, PackedBf4SliceMut, PackedBf4Vec,
    PackedF4Cow, PackedF4Slice, PackedF4SliceMut, PackedF4Vec,
};

#[cfg(feature = "rkyv")]
pub use packed::{ArchivedPacked4Cow, ArchivedPacked4Vec, Packed4CowResolver, Packed4VecResolver};

#[cfg(target_arch = "x86_64")]
pub use packed::unsafe_intrinsics;

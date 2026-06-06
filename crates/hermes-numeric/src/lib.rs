#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! Foundational numeric types, traits, and mixed-precision helpers for the hermes ecosystem.
//! Single Source of Truth (SSOT) for all numeric representations.

extern crate alloc;

mod traits;
mod types;
mod ops;
mod impls;
mod casts;
mod packed;

// Re-export core traits
pub use traits::{NumericElement, FloatElement, CastFrom, CastTo};

// Re-export wrapper types
pub use types::{F16, F32, F64, Bf16, Bf8, Bf4, F8, F4, I8, I16, I32};

// Re-export packed layout structures and functions
pub use packed::{
    Packable4, Packed4Slice, Packed4SliceMut,
    PackedBf4Slice, PackedBf4SliceMut, PackedF4Slice, PackedF4SliceMut,
    Packed4Vec, Packed4Iter, PackedBf4Vec, PackedF4Vec,
    Packed4Cow, PackedBf4Cow, PackedF4Cow,
    unpack_bf8_to_bf16, unpack_bf4_to_bf16, unpack_bf4_to_bf16_packed,
    unpack_f4_to_f32, unpack_f4_to_f32_packed, unpack_f8_to_f32,
};

#[cfg(feature = "rkyv")]
pub use packed::{
    ArchivedPacked4Cow, ArchivedPacked4Vec, Packed4CowResolver, Packed4VecResolver,
};

#[cfg(target_arch = "x86_64")]
pub use packed::unsafe_intrinsics;

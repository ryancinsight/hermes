mod slice;
mod vec;
mod unpack;
mod cow;

pub use slice::{
    Packable4, Packed4Slice, Packed4SliceMut,
    PackedBf4Slice, PackedBf4SliceMut, PackedF4Slice, PackedF4SliceMut,
};
pub use vec::{Packed4Vec, Packed4Iter, PackedBf4Vec, PackedF4Vec};
pub use unpack::{
    unpack_bf8_to_bf16, unpack_bf4_to_bf16, unpack_bf4_to_bf16_packed,
    unpack_f4_to_f32, unpack_f4_to_f32_packed, unpack_f8_to_f32,
};
pub use cow::{Packed4Cow, PackedBf4Cow, PackedF4Cow};

#[cfg(feature = "rkyv")]
pub mod rkyv;

#[cfg(feature = "rkyv")]
pub use self::rkyv::{
    ArchivedPacked4Cow, ArchivedPacked4Vec, Packed4CowResolver, Packed4VecResolver,
};

#[cfg(target_arch = "x86_64")]
pub use unpack::unsafe_intrinsics;

//! rkyv zero-copy serialization support for AlignedVec.

//! Zero-copy serialization support for aligned vectors using `rkyv`.

use super::AlignedVec;
use crate::align::Alignment;
use rkyv::munge::munge;
use rkyv::rancor::Fallible;
use rkyv::ser::{Allocator, Writer};
use rkyv::{Place, Portable};

/// Archived representation of an `AlignedVec` used by `rkyv` zero-copy serialization.
///
/// Transparent over `ArchivedVec`, so portability and byte validity reduce to
/// the element type's; both are derived rather than asserted by hand.
#[derive(Portable, rkyv::bytecheck::CheckBytes)]
#[bytecheck(crate = rkyv::bytecheck)]
#[repr(transparent)]
pub struct ArchivedAlignedVec<T> {
    pub(crate) elements: rkyv::vec::ArchivedVec<T>,
}

/// Resolver type for `AlignedVec` serialization.
pub struct AlignedVecResolver {
    pub(crate) elements_resolver: rkyv::vec::VecResolver,
}

impl<T, Align> rkyv::Archive for AlignedVec<T, Align>
where
    T: rkyv::Archive,
    Align: Alignment,
{
    type Archived = ArchivedAlignedVec<T::Archived>;
    type Resolver = AlignedVecResolver;

    #[inline]
    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        // 0.8 projects the field through `Place`, so the offset the 0.7
        // implementation computed by hand cannot drift from the layout.
        munge!(let ArchivedAlignedVec { elements } = out);
        rkyv::vec::ArchivedVec::resolve_from_slice(
            self.as_slice(),
            resolver.elements_resolver,
            elements,
        );
    }
}

impl<T, Align, S> rkyv::Serialize<S> for AlignedVec<T, Align>
where
    T: rkyv::Serialize<S> + rkyv::Archive,
    Align: Alignment,
    S: Fallible + Allocator + Writer + ?Sized,
{
    #[inline]
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let elements_resolver =
            rkyv::vec::ArchivedVec::serialize_from_slice(self.as_slice(), serializer)?;
        Ok(AlignedVecResolver { elements_resolver })
    }
}

// The archived element type is not the native one -- `Archived<i32>` is
// `i32_le` -- so the target element is a distinct parameter. The 0.7 bound
// `T: Deserialize<T, D>` only compiled because the two coincided under native
// endianness; naming them separately is what makes this correct rather than
// accidentally well-typed.
impl<T, U, Align, D> rkyv::Deserialize<AlignedVec<U, Align>, D> for ArchivedAlignedVec<T>
where
    T: rkyv::Deserialize<U, D>,
    Align: Alignment,
    D: Fallible + ?Sized,
{
    #[inline]
    fn deserialize(&self, deserializer: &mut D) -> Result<AlignedVec<U, Align>, D::Error> {
        let slice = self.elements.as_slice();
        let mut v = AlignedVec::with_capacity(slice.len());
        for x in slice {
            v.push(x.deserialize(deserializer)?);
        }
        Ok(v)
    }
}

//! rkyv zero-copy serialization support for AlignedVec.

//! Zero-copy serialization support for aligned vectors using `rkyv`.

use super::AlignedVec;
use crate::align::Alignment;

#[repr(transparent)]
/// Archived representation of an `AlignedVec` used by `rkyv` zero-copy serialization.
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
    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        let out_elements = core::ptr::addr_of_mut!((*out).elements);
        rkyv::vec::ArchivedVec::resolve_from_slice(
            self.as_slice(),
            pos,
            resolver.elements_resolver,
            out_elements,
        );
    }
}

impl<T, Align, S> rkyv::Serialize<S> for AlignedVec<T, Align>
where
    T: rkyv::Serialize<S> + rkyv::Archive,
    Align: Alignment,
    S: rkyv::Fallible + rkyv::ser::Serializer + ?Sized,
    [T]: rkyv::SerializeUnsized<S>,
{
    #[inline]
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let elements_resolver =
            rkyv::vec::ArchivedVec::serialize_from_slice(self.as_slice(), serializer)?;
        Ok(AlignedVecResolver { elements_resolver })
    }
}

impl<T, Align, D> rkyv::Deserialize<AlignedVec<T, Align>, D> for ArchivedAlignedVec<T>
where
    T: rkyv::Archive,
    T: rkyv::Deserialize<T, D>,
    Align: Alignment,
    D: rkyv::Fallible + ?Sized,
{
    #[inline]
    fn deserialize(&self, deserializer: &mut D) -> Result<AlignedVec<T, Align>, D::Error> {
        let slice = self.elements.as_slice();
        let mut v = AlignedVec::with_capacity(slice.len());
        for x in slice {
            v.push(x.deserialize(deserializer)?);
        }
        Ok(v)
    }
}

//! Zero-copy serialization support for Clone-on-Write SIMD containers using `rkyv`.

use super::SimdCow;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::vec::AlignedVec;
use crate::view::SimdView;
use rkyv::munge::munge;
use rkyv::rancor::Fallible;
use rkyv::ser::{Allocator, Writer};
use rkyv::{Place, Portable};

/// Archived representation of a `SimdCow` used by `rkyv` zero-copy serialization.
///
/// `Portable` and `CheckBytes` are derived rather than asserted: the wrapper is
/// transparent over `ArchivedVec`, so both properties reduce to the element
/// type's, and the derive is what enforces that rather than a hand-written
/// claim.
#[derive(Portable, rkyv::bytecheck::CheckBytes)]
#[bytecheck(crate = rkyv::bytecheck)]
#[repr(transparent)]
pub struct ArchivedSimdCow<T> {
    pub(crate) elements: rkyv::vec::ArchivedVec<T>,
}

/// Resolver type for `SimdCow` serialization.
pub struct SimdCowResolver {
    pub(crate) elements_resolver: rkyv::vec::VecResolver,
}

impl<T, Arch, Align> rkyv::Archive for SimdCow<'_, T, Arch, Align>
where
    T: rkyv::Archive,
    Arch: SimdArch,
    Align: Alignment,
{
    type Archived = ArchivedSimdCow<T::Archived>;
    type Resolver = SimdCowResolver;

    #[inline]
    fn resolve(&self, resolver: Self::Resolver, out: Place<Self::Archived>) {
        // 0.8 projects the field through `Place`, so the offset the 0.7
        // implementation computed by hand cannot drift from the layout.
        munge!(let ArchivedSimdCow { elements } = out);
        rkyv::vec::ArchivedVec::resolve_from_slice(&self[..], resolver.elements_resolver, elements);
    }
}

impl<T, Arch, Align, S> rkyv::Serialize<S> for SimdCow<'_, T, Arch, Align>
where
    T: rkyv::Serialize<S> + rkyv::Archive,
    Arch: SimdArch,
    Align: Alignment,
    S: Fallible + Allocator + Writer + ?Sized,
{
    #[inline]
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let elements_resolver =
            rkyv::vec::ArchivedVec::serialize_from_slice(&self[..], serializer)?;
        Ok(SimdCowResolver { elements_resolver })
    }
}

// See `ArchivedAlignedVec`: the archived and native element types are distinct
// in 0.8, so the deserialization target is its own parameter.
impl<T, U, Arch, Align, D> rkyv::Deserialize<SimdCow<'static, U, Arch, Align>, D>
    for ArchivedSimdCow<T>
where
    T: rkyv::Deserialize<U, D>,
    Arch: SimdArch,
    Align: Alignment,
    D: Fallible + ?Sized,
{
    #[inline]
    fn deserialize(
        &self,
        deserializer: &mut D,
    ) -> Result<SimdCow<'static, U, Arch, Align>, D::Error> {
        let slice = self.elements.as_slice();
        let mut v = AlignedVec::with_capacity(slice.len());
        for x in slice {
            v.push(x.deserialize(deserializer)?);
        }
        Ok(SimdCow::Owned(v))
    }
}

impl<T> ArchivedSimdCow<T> {
    /// Returns the length of the archived vector.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns `true` if the archived vector is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Access the archived elements as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.elements.as_slice()
    }

    /// Zero-copy conversion of the archived `SimdCow` to a borrowed `SimdCow`.
    ///
    /// # Safety
    /// The alignment of the underlying archived memory must satisfy `Align`.
    #[inline]
    #[must_use]
    pub unsafe fn as_borrowed<Arch, Align>(&self) -> Option<SimdCow<'_, T, Arch, Align>>
    where
        Arch: SimdArch,
        Align: Alignment,
    {
        let slice = self.elements.as_slice();
        let view = SimdView::new(slice)?;
        Some(SimdCow::Borrowed(view))
    }
}

impl<T> core::ops::Deref for ArchivedSimdCow<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.elements.as_slice()
    }
}

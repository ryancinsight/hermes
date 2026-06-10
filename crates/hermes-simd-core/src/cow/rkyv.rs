//! Zero-copy serialization support for Clone-on-Write SIMD containers using `rkyv`.

use super::SimdCow;
use crate::align::Alignment;
use crate::arch::SimdArch;
use crate::vec::AlignedVec;
use crate::view::SimdView;

#[repr(transparent)]
/// Archived representation of a `SimdCow` used by `rkyv` zero-copy serialization.
pub struct ArchivedSimdCow<T> {
    pub(crate) elements: rkyv::vec::ArchivedVec<T>,
}

/// Resolver type for `SimdCow` serialization.
pub struct SimdCowResolver {
    pub(crate) elements_resolver: rkyv::vec::VecResolver,
}

impl<'a, T, Arch, Align> rkyv::Archive for SimdCow<'a, T, Arch, Align>
where
    T: rkyv::Archive,
    Arch: SimdArch,
    Align: Alignment,
{
    type Archived = ArchivedSimdCow<T::Archived>;
    type Resolver = SimdCowResolver;

    #[inline]
    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        let out_elements = core::ptr::addr_of_mut!((*out).elements);
        rkyv::vec::ArchivedVec::resolve_from_slice(
            &self[..],
            pos,
            resolver.elements_resolver,
            out_elements,
        );
    }
}

impl<'a, T, Arch, Align, S> rkyv::Serialize<S> for SimdCow<'a, T, Arch, Align>
where
    T: rkyv::Serialize<S> + rkyv::Archive,
    Arch: SimdArch,
    Align: Alignment,
    S: rkyv::Fallible + rkyv::ser::Serializer + ?Sized,
    [T]: rkyv::SerializeUnsized<S>,
{
    #[inline]
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let elements_resolver =
            rkyv::vec::ArchivedVec::serialize_from_slice(&self[..], serializer)?;
        Ok(SimdCowResolver { elements_resolver })
    }
}

impl<T, Arch, Align, D> rkyv::Deserialize<SimdCow<'static, T, Arch, Align>, D>
    for ArchivedSimdCow<T>
where
    T: rkyv::Archive,
    T: rkyv::Deserialize<T, D>,
    Arch: SimdArch,
    Align: Alignment,
    D: rkyv::Fallible + ?Sized,
{
    #[inline]
    fn deserialize(
        &self,
        deserializer: &mut D,
    ) -> Result<SimdCow<'static, T, Arch, Align>, D::Error> {
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
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Returns `true` if the archived vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Access the archived elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.elements.as_slice()
    }

    /// Zero-copy conversion of the archived SimdCow to a borrowed SimdCow.
    ///
    /// # Safety
    /// The alignment of the underlying archived memory must satisfy `Align`.
    #[inline]
    pub unsafe fn as_borrowed<'a, Arch, Align>(&'a self) -> Option<SimdCow<'a, T, Arch, Align>>
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

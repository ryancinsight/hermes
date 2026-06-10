//! rkyv zero-copy serialization support for packed 4-bit containers.

#![allow(clippy::let_unit_value, clippy::unit_arg)]

use crate::packed::cow::Packed4Cow;
use crate::packed::slice::{Packable4, Packed4Slice};
use crate::packed::vec::Packed4Vec;
use rkyv::Deserialize;

#[repr(C)]
/// Archived representation of a `Packed4Cow` for zero-copy deserialization.
pub struct ArchivedPacked4Cow<T: Packable4> {
    pub(crate) data: rkyv::vec::ArchivedVec<u8>,
    pub(crate) len: rkyv::Archived<usize>,
    pub(crate) _marker: core::marker::PhantomData<T>,
}

/// Resolver type for `Packed4Cow`.
pub struct Packed4CowResolver {
    pub(crate) data_resolver: rkyv::vec::VecResolver,
    pub(crate) len_resolver: rkyv::Resolver<usize>,
}

impl<'a, T: Packable4> rkyv::Archive for Packed4Cow<'a, T> {
    type Archived = ArchivedPacked4Cow<T>;
    type Resolver = Packed4CowResolver;

    #[inline]
    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        let view = self.as_view();

        let out_data = core::ptr::addr_of_mut!((*out).data);
        rkyv::vec::ArchivedVec::resolve_from_slice(
            view.as_packed_slice(),
            pos + core::mem::offset_of!(ArchivedPacked4Cow<T>, data),
            resolver.data_resolver,
            out_data,
        );

        let out_len = core::ptr::addr_of_mut!((*out).len);
        view.len().resolve(
            pos + core::mem::offset_of!(ArchivedPacked4Cow<T>, len),
            resolver.len_resolver,
            out_len,
        );
    }
}

impl<'a, T: Packable4, S> rkyv::Serialize<S> for Packed4Cow<'a, T>
where
    S: rkyv::Fallible + rkyv::ser::Serializer + rkyv::ser::ScratchSpace + ?Sized,
{
    #[inline]
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let view = self.as_view();
        let data_resolver =
            rkyv::vec::ArchivedVec::serialize_from_slice(view.as_packed_slice(), serializer)?;
        let len_resolver = view.len().serialize(serializer)?;
        Ok(Packed4CowResolver {
            data_resolver,
            len_resolver,
        })
    }
}

impl<T: Packable4, D> rkyv::Deserialize<Packed4Cow<'static, T>, D> for ArchivedPacked4Cow<T>
where
    D: rkyv::Fallible + ?Sized,
{
    #[inline]
    fn deserialize(&self, deserializer: &mut D) -> Result<Packed4Cow<'static, T>, D::Error> {
        let bytes = self.data.as_slice();
        let len: usize = self.len.deserialize(deserializer)?;

        let mut vec = Packed4Vec::with_capacity(len);
        vec.data.extend_from_slice(bytes);
        vec.len = len;

        Ok(Packed4Cow::Owned(vec))
    }
}

impl<T: Packable4> ArchivedPacked4Cow<T> {
    /// Returns the logical length of the archived packed container.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.deserialize(&mut rkyv::Infallible).unwrap()
    }

    /// Returns `true` if the archived packed container is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Zero-copy conversion of the archived container to a borrowed `Packed4Cow`.
    #[inline]
    pub fn as_borrowed<'a>(&'a self) -> Option<Packed4Cow<'a, T>> {
        let len = self.len();
        Packed4Slice::new(self.data.as_slice(), len).map(Packed4Cow::Borrowed)
    }
}

#[repr(C)]
/// Archived representation of a `Packed4Vec`.
pub struct ArchivedPacked4Vec<T: Packable4> {
    pub(crate) data: rkyv::vec::ArchivedVec<u8>,
    pub(crate) len: rkyv::Archived<usize>,
    pub(crate) _marker: core::marker::PhantomData<T>,
}

/// Resolver type for `Packed4Vec`.
pub struct Packed4VecResolver {
    pub(crate) data_resolver: rkyv::vec::VecResolver,
    pub(crate) len_resolver: rkyv::Resolver<usize>,
}

impl<T: Packable4> rkyv::Archive for Packed4Vec<T> {
    type Archived = ArchivedPacked4Vec<T>;
    type Resolver = Packed4VecResolver;

    #[inline]
    unsafe fn resolve(&self, pos: usize, resolver: Self::Resolver, out: *mut Self::Archived) {
        let out_data = core::ptr::addr_of_mut!((*out).data);
        rkyv::vec::ArchivedVec::resolve_from_slice(
            self.as_packed_slice(),
            pos + core::mem::offset_of!(ArchivedPacked4Vec<T>, data),
            resolver.data_resolver,
            out_data,
        );

        let out_len = core::ptr::addr_of_mut!((*out).len);
        self.len.resolve(
            pos + core::mem::offset_of!(ArchivedPacked4Vec<T>, len),
            resolver.len_resolver,
            out_len,
        );
    }
}

impl<T: Packable4, S> rkyv::Serialize<S> for Packed4Vec<T>
where
    S: rkyv::Fallible + rkyv::ser::Serializer + rkyv::ser::ScratchSpace + ?Sized,
{
    #[inline]
    fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
        let data_resolver =
            rkyv::vec::ArchivedVec::serialize_from_slice(self.as_packed_slice(), serializer)?;
        let len_resolver = self.len.serialize(serializer)?;
        Ok(Packed4VecResolver {
            data_resolver,
            len_resolver,
        })
    }
}

impl<T: Packable4, D> rkyv::Deserialize<Packed4Vec<T>, D> for ArchivedPacked4Vec<T>
where
    D: rkyv::Fallible + ?Sized,
{
    #[inline]
    fn deserialize(&self, deserializer: &mut D) -> Result<Packed4Vec<T>, D::Error> {
        let bytes = self.data.as_slice();
        let len: usize = self.len.deserialize(deserializer)?;

        let mut vec = Packed4Vec::with_capacity(len);
        vec.data.extend_from_slice(bytes);
        vec.len = len;

        Ok(vec)
    }
}

impl<T: Packable4> ArchivedPacked4Vec<T> {
    /// Returns the logical length of the archived vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.deserialize(&mut rkyv::Infallible).unwrap()
    }

    /// Returns `true` if the archived vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert the archived vector to a borrowed `Packed4Slice` view.
    #[inline]
    pub fn as_view(&self) -> Packed4Slice<'_, T> {
        Packed4Slice {
            data: self.data.as_slice(),
            len: self.len(),
            _marker: core::marker::PhantomData,
        }
    }
}

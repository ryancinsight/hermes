//! Custom aligned vector allocation for zero-copy aligned SIMD memory access.

extern crate alloc;

use crate::align::Alignment;
use crate::numa::NumaAllocator;
use crate::view::SimdView;
use core::alloc::Layout;
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};

#[cfg(not(feature = "mnemosyne-memory"))]
use alloc::alloc::{alloc, dealloc};

/// Zero-copy serialization support for aligned vectors using `rkyv`.
pub mod rkyv;
#[cfg(test)]
mod tests;

/// A heap-allocated vector with statically guaranteed memory alignment layout.
///
/// Underpinned by custom memory allocation using `core::alloc::Layout` to ensure that
/// loading elements into SIMD registers does not trigger alignment faults.
pub struct AlignedVec<T, Align: Alignment> {
    ptr: *mut T,
    len: usize,
    cap: usize,
    node: Option<u32>,
    alloc_align: u32,
    _marker: PhantomData<(T, Align)>,
}

unsafe impl<T: Send, Align: Alignment> Send for AlignedVec<T, Align> {}
unsafe impl<T: Sync, Align: Alignment> Sync for AlignedVec<T, Align> {}

impl<T, Align> AlignedVec<T, Align>
where
    Align: Alignment,
{
    #[inline(always)]
    fn layout_for_capacity(capacity: usize, align: usize) -> Layout {
        let size = capacity
            .checked_mul(core::mem::size_of::<T>())
            .expect("Capacity overflow");
        Layout::from_size_align(size, align).unwrap()
    }

    /// Create a new empty `AlignedVec` with no allocation.
    #[inline]
    pub fn new() -> Self {
        Self {
            ptr: core::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            cap: 0,
            node: None,
            alloc_align: if Align::IS_ALIGNED {
                Align::ALIGN_BYTES as u32
            } else {
                core::mem::align_of::<T>() as u32
            },
            _marker: PhantomData,
        }
    }

    /// Create a new `AlignedVec` with space allocated for `capacity` elements
    /// satisfying the alignment boundary constraints.
    pub fn with_capacity(capacity: usize) -> Self {
        let default_align = if Align::IS_ALIGNED {
            Align::ALIGN_BYTES as u32
        } else {
            core::mem::align_of::<T>() as u32
        };
        if core::mem::size_of::<T>() == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: usize::MAX,
                node: None,
                alloc_align: default_align,
                _marker: PhantomData,
            };
        }
        if capacity == 0 {
            return Self::new();
        }

        let align = if Align::IS_ALIGNED {
            Align::ALIGN_BYTES
        } else {
            core::mem::align_of::<T>()
        };
        let layout = Self::layout_for_capacity(capacity, align);

        #[cfg(feature = "mnemosyne-memory")]
        let ptr =
            unsafe { core::alloc::GlobalAlloc::alloc(&mnemosyne::Mnemosyne, layout) as *mut T };
        #[cfg(not(feature = "mnemosyne-memory"))]
        let ptr = unsafe { alloc(layout) as *mut T };

        if ptr.is_null() {
            alloc::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr,
            len: 0,
            cap: capacity,
            node: None,
            alloc_align: align as u32,
            _marker: PhantomData,
        }
    }

    /// Create a new `AlignedVec` with space allocated for `capacity` elements
    /// on the specified NUMA node.
    pub fn with_capacity_numa(capacity: usize, node: u32) -> Self {
        let default_align = if Align::IS_ALIGNED {
            Align::ALIGN_BYTES as u32
        } else {
            core::mem::align_of::<T>() as u32
        };
        if core::mem::size_of::<T>() == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: usize::MAX,
                node: Some(node),
                alloc_align: default_align,
                _marker: PhantomData,
            };
        }
        if capacity == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: 0,
                node: Some(node),
                alloc_align: default_align,
                _marker: PhantomData,
            };
        }

        let align = if Align::IS_ALIGNED {
            Align::ALIGN_BYTES
        } else {
            core::mem::align_of::<T>()
        };
        let layout = Self::layout_for_capacity(capacity, align);

        let allocator = crate::numa::MnemosyneNumaAllocator;
        let ptr = unsafe { allocator.alloc_on_node(layout, node) as *mut T };
        if ptr.is_null() {
            alloc::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr,
            len: 0,
            cap: capacity,
            node: Some(node),
            alloc_align: align as u32,
            _marker: PhantomData,
        }
    }

    /// Appends an element to the back of the vector.
    pub fn push(&mut self, value: T) {
        if core::mem::size_of::<T>() == 0 {
            // No allocation or writes needed for zero-sized types.
            // Bypasses drop of `value` when returning.
            core::mem::forget(value);
            self.len = self.len.checked_add(1).expect("Length overflow");
            self.cap = usize::MAX;
            return;
        }
        if self.len == self.cap {
            self.grow();
        }
        unsafe {
            core::ptr::write(self.ptr.add(self.len), value);
            self.len += 1;
        }
    }

    /// Returns the number of elements in the vector.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the vector contains no elements.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the vector.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns a raw pointer to the vector's buffer.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a raw mutable pointer to the vector's buffer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Forcefully sets the length of the vector without initializing elements.
    ///
    /// # Safety
    ///
    /// The elements up to `new_len` must be initialized.
    #[inline(always)]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap);
        self.len = new_len;
    }

    /// Accesses the elements as an immutable slice.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Accesses the elements as a mutable slice.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    /// Copy `src` into a new `AlignedVec` in a single allocation.
    ///
    /// # Performance
    ///
    /// Exactly one call to the allocator (`with_capacity(src.len())`) followed by
    /// one `copy_nonoverlapping` of `src.len()` elements. Zero intermediate allocations.
    /// The returned vec is fully owned and has `len == cap == src.len()`.
    #[inline]
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Copy,
    {
        let n = src.len();
        if n == 0 {
            return Self::new();
        }
        if core::mem::size_of::<T>() == 0 {
            let mut v = Self::new();
            v.len = n;
            v.cap = usize::MAX;
            return v;
        }
        let mut v = Self::with_capacity(n);
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), v.ptr, n);
            v.len = n;
        }
        v
    }

    /// Clone elements from `src` into a new `AlignedVec` in a single allocation.
    ///
    /// # Performance
    ///
    /// Exactly one call to the allocator (`with_capacity(src.len())`) followed by
    /// cloning elements into place sequentially.
    #[inline]
    pub fn from_slice_clone(src: &[T]) -> Self
    where
        T: Clone,
    {
        let n = src.len();
        if n == 0 {
            return Self::new();
        }
        if core::mem::size_of::<T>() == 0 {
            let mut v = Self::new();
            v.len = n;
            v.cap = usize::MAX;
            return v;
        }
        let mut v = Self::with_capacity(n);
        for i in 0..n {
            unsafe {
                core::ptr::write(v.ptr.add(i), src[i].clone());
                v.len = i + 1;
            }
        }
        v
    }

    /// Obtains a compile-time safe immutable `SimdView` over the vector's buffer.
    #[inline(always)]
    pub fn view<'a, Arch>(
        &'a self,
    ) -> SimdView<'a, T, Arch, Align, crate::execution::Unmasked, &'a [T]>
    where
        Arch: crate::arch::SimdArch,
    {
        SimdView::new(self.as_slice()).unwrap()
    }

    /// Obtains a compile-time safe mutable `SimdView` over the vector's buffer.
    #[inline(always)]
    pub fn view_mut<'a, Arch>(
        &'a mut self,
    ) -> SimdView<'a, T, Arch, Align, crate::execution::Unmasked, &'a mut [T]>
    where
        Arch: crate::arch::SimdArch,
    {
        SimdView::new_mut(self.as_mut_slice()).unwrap()
    }

    /// Converts this `AlignedVec` to another alignment layout type-safely, without checking
    /// if the pointer satisfies the new alignment's constraints.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the underlying memory address satisfies the alignment
    /// boundary constraints of `NewAlign`.
    #[inline(always)]
    pub unsafe fn into_alignment_unchecked<NewAlign: Alignment>(self) -> AlignedVec<T, NewAlign> {
        let md = core::mem::ManuallyDrop::new(self);
        AlignedVec {
            ptr: md.ptr,
            len: md.len,
            cap: md.cap,
            node: md.node,
            alloc_align: md.alloc_align,
            _marker: PhantomData,
        }
    }

    /// Converts this `AlignedVec` to an unaligned layout, stripping the alignment guarantee zero-cost.
    #[inline(always)]
    pub fn into_unaligned(self) -> AlignedVec<T, crate::align::Unaligned> {
        unsafe { self.into_alignment_unchecked() }
    }

    /// Attempts to cast this `AlignedVec` to a stricter alignment constraint.
    /// Returns `Some` if the pointer satisfies the alignment requirement of `NewAlign`, otherwise `None`.
    #[inline]
    pub fn try_into_alignment<NewAlign: Alignment>(self) -> Option<AlignedVec<T, NewAlign>> {
        if NewAlign::IS_ALIGNED {
            let addr = self.as_ptr() as usize;
            if addr % NewAlign::ALIGN_BYTES == 0 {
                unsafe { Some(self.into_alignment_unchecked()) }
            } else {
                None
            }
        } else {
            unsafe { Some(self.into_alignment_unchecked()) }
        }
    }

    fn layout_for(&self, capacity: usize) -> Layout {
        Self::layout_for_capacity(capacity, self.alloc_align as usize)
    }

    fn grow(&mut self) {
        if core::mem::size_of::<T>() == 0 {
            self.cap = usize::MAX;
            return;
        }

        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap.checked_mul(2).expect("Capacity overflow")
        };
        let new_layout = self.layout_for(new_cap);

        let old_ptr = self.ptr;
        let new_ptr = if self.cap == 0 {
            if let Some(node) = self.node {
                let allocator = crate::numa::MnemosyneNumaAllocator;
                unsafe { allocator.alloc_on_node(new_layout, node) as *mut T }
            } else {
                #[cfg(feature = "mnemosyne-memory")]
                unsafe {
                    core::alloc::GlobalAlloc::alloc(&mnemosyne::Mnemosyne, new_layout) as *mut T
                }
                #[cfg(not(feature = "mnemosyne-memory"))]
                unsafe {
                    alloc(new_layout) as *mut T
                }
            }
        } else {
            let old_layout = self.layout_for(self.cap);
            unsafe {
                if let Some(node) = self.node {
                    let allocator = crate::numa::MnemosyneNumaAllocator;
                    allocator.realloc_on_node(self.ptr as *mut u8, old_layout, new_layout, node)
                        as *mut T
                } else {
                    #[cfg(feature = "mnemosyne-memory")]
                    let ptr = core::alloc::GlobalAlloc::realloc(
                        &mnemosyne::Mnemosyne,
                        self.ptr as *mut u8,
                        old_layout,
                        new_layout.size(),
                    ) as *mut T;
                    #[cfg(not(feature = "mnemosyne-memory"))]
                    let ptr =
                        alloc::alloc::realloc(self.ptr as *mut u8, old_layout, new_layout.size())
                            as *mut T;
                    ptr
                }
            }
        };

        if new_ptr.is_null() {
            alloc::alloc::handle_alloc_error(new_layout);
        }

        if self.node.is_none() && self.cap > 0 && new_ptr != old_ptr {
            crate::numa::locality::bump_alloc_generation();
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
    }
}

impl<T, Align: Alignment> Deref for AlignedVec<T, Align> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, Align: Alignment> DerefMut for AlignedVec<T, Align> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

struct DeallocGuard<T, Align: Alignment> {
    ptr: *mut T,
    cap: usize,
    node: Option<u32>,
    alloc_align: u32,
    _marker: PhantomData<(T, Align)>,
}

impl<T, Align: Alignment> Drop for DeallocGuard<T, Align> {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.cap > 0 {
            crate::numa::locality::bump_alloc_generation();
            unsafe {
                let layout = AlignedVec::<T, Align>::layout_for_capacity(
                    self.cap,
                    self.alloc_align as usize,
                );
                if let Some(node) = self.node {
                    let allocator = crate::numa::MnemosyneNumaAllocator;
                    allocator.dealloc_on_node(self.ptr as *mut u8, layout, node);
                } else {
                    #[cfg(feature = "mnemosyne-memory")]
                    core::alloc::GlobalAlloc::dealloc(
                        &mnemosyne::Mnemosyne,
                        self.ptr as *mut u8,
                        layout,
                    );
                    #[cfg(not(feature = "mnemosyne-memory"))]
                    dealloc(self.ptr as *mut u8, layout);
                }
            }
        }
    }
}

impl<T, Align: Alignment> Drop for AlignedVec<T, Align> {
    fn drop(&mut self) {
        if core::mem::size_of::<T>() == 0 {
            if self.len > 0 {
                unsafe {
                    core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(
                        self.ptr, self.len,
                    ));
                }
            }
            return;
        }
        if !self.ptr.is_null() && self.cap > 0 {
            let ptr = self.ptr;
            let cap = self.cap;
            let len = self.len;
            let alloc_align = self.alloc_align;

            self.ptr = core::ptr::null_mut();
            self.cap = 0;
            self.len = 0;

            let _guard: DeallocGuard<T, Align> = DeallocGuard {
                ptr,
                cap,
                node: self.node,
                alloc_align,
                _marker: PhantomData,
            };
            unsafe {
                core::ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(ptr, len));
            }
        }
    }
}

impl<T: Clone, Align: Alignment> Clone for AlignedVec<T, Align> {
    fn clone(&self) -> Self {
        if core::mem::size_of::<T>() == 0 {
            let mut new_vec = Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: usize::MAX,
                node: self.node,
                alloc_align: self.alloc_align,
                _marker: PhantomData,
            };
            for val in self.as_slice() {
                new_vec.push(val.clone());
            }
            return new_vec;
        }
        let mut new_vec = if let Some(node) = self.node {
            Self::with_capacity_numa(self.len, node)
        } else {
            Self::with_capacity(self.len)
        };
        for i in 0..self.len {
            unsafe {
                let val = (*self.ptr.add(i)).clone();
                core::ptr::write(new_vec.ptr.add(i), val);
                new_vec.len = i + 1;
            }
        }
        new_vec
    }
}

impl<T, Align: Alignment> Default for AlignedVec<T, Align> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: core::fmt::Debug, Align: Alignment> core::fmt::Debug for AlignedVec<T, Align> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl<T: PartialEq, Align1: Alignment, Align2: Alignment> PartialEq<AlignedVec<T, Align2>>
    for AlignedVec<T, Align1>
{
    #[inline]
    fn eq(&self, other: &AlignedVec<T, Align2>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq, Align: Alignment> Eq for AlignedVec<T, Align> {}

impl<T: PartialEq, Align: Alignment> PartialEq<[T]> for AlignedVec<T, Align> {
    #[inline]
    fn eq(&self, other: &[T]) -> bool {
        self.as_slice() == other
    }
}

impl<T: PartialEq, Align: Alignment> PartialEq<AlignedVec<T, Align>> for [T] {
    #[inline]
    fn eq(&self, other: &AlignedVec<T, Align>) -> bool {
        self == other.as_slice()
    }
}

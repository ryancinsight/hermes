//! Custom aligned vector allocation for zero-copy aligned SIMD memory access.

extern crate alloc;

use core::alloc::Layout;
use core::ops::{Deref, DerefMut};
use core::marker::PhantomData;
use crate::align::Alignment;
use crate::view::SimdView;
use crate::numa::NumaAllocator;

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
    _marker: PhantomData<(T, Align)>,
}

unsafe impl<T: Send, Align: Alignment> Send for AlignedVec<T, Align> {}
unsafe impl<T: Sync, Align: Alignment> Sync for AlignedVec<T, Align> {}

impl<T, Align> AlignedVec<T, Align>
where
    Align: Alignment,
{
    /// Create a new empty `AlignedVec` with no allocation.
    #[inline]
    pub fn new() -> Self {
        Self {
            ptr: core::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            cap: 0,
            node: None,
            _marker: PhantomData,
        }
    }

    /// Create a new `AlignedVec` with space allocated for `capacity` elements
    /// satisfying the alignment boundary constraints.
    pub fn with_capacity(capacity: usize) -> Self {
        if core::mem::size_of::<T>() == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: usize::MAX,
                node: None,
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
        let size = capacity.checked_mul(core::mem::size_of::<T>())
            .expect("Capacity overflow");
        let layout = Layout::from_size_align(size, align).unwrap();

        let ptr = unsafe { alloc(layout) as *mut T };
        if ptr.is_null() {
            alloc::alloc::handle_alloc_error(layout);
        }

        Self {
            ptr,
            len: 0,
            cap: capacity,
            node: None,
            _marker: PhantomData,
        }
    }

    /// Create a new `AlignedVec` with space allocated for `capacity` elements
    /// on the specified NUMA node.
    pub fn with_capacity_numa(capacity: usize, node: u32) -> Self {
        if core::mem::size_of::<T>() == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: usize::MAX,
                node: Some(node),
                _marker: PhantomData,
            };
        }
        if capacity == 0 {
            return Self {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: 0,
                node: Some(node),
                _marker: PhantomData,
            };
        }

        let align = if Align::IS_ALIGNED {
            Align::ALIGN_BYTES
        } else {
            core::mem::align_of::<T>()
        };
        let size = capacity.checked_mul(core::mem::size_of::<T>())
            .expect("Capacity overflow");
        let layout = Layout::from_size_align(size, align).unwrap();

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

    /// Obtains a compile-time safe immutable `SimdView` over the vector's buffer.
    #[inline(always)]
    pub fn view<'a, Arch>(&'a self) -> SimdView<'a, T, Arch, Align, crate::execution::Unmasked, &'a [T]>
    where
        Arch: crate::arch::SimdArch,
    {
        SimdView::new(self.as_slice()).unwrap()
    }

    /// Obtains a compile-time safe mutable `SimdView` over the vector's buffer.
    #[inline(always)]
    pub fn view_mut<'a, Arch>(&'a mut self) -> SimdView<'a, T, Arch, Align, crate::execution::Unmasked, &'a mut [T]>
    where
        Arch: crate::arch::SimdArch,
    {
        SimdView::new_mut(self.as_mut_slice()).unwrap()
    }

    fn layout_for(&self, capacity: usize) -> Layout {
        let size = capacity.checked_mul(core::mem::size_of::<T>())
            .expect("Capacity overflow");
        let align = if Align::IS_ALIGNED {
            Align::ALIGN_BYTES
        } else {
            core::mem::align_of::<T>()
        };
        Layout::from_size_align(size, align).unwrap()
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

        let new_ptr = if self.cap == 0 {
            if let Some(node) = self.node {
                let allocator = crate::numa::MnemosyneNumaAllocator;
                unsafe { allocator.alloc_on_node(new_layout, node) as *mut T }
            } else {
                unsafe { alloc(new_layout) as *mut T }
            }
        } else {
            let old_layout = self.layout_for(self.cap);
            unsafe {
                let new_p = if let Some(node) = self.node {
                    let allocator = crate::numa::MnemosyneNumaAllocator;
                    allocator.alloc_on_node(new_layout, node) as *mut T
                } else {
                    alloc(new_layout) as *mut T
                };
                if !new_p.is_null() {
                    core::ptr::copy_nonoverlapping(self.ptr, new_p, self.len);
                    if let Some(node) = self.node {
                        let allocator = crate::numa::MnemosyneNumaAllocator;
                        allocator.dealloc_on_node(self.ptr as *mut u8, old_layout, node);
                    } else {
                        dealloc(self.ptr as *mut u8, old_layout);
                    }
                }
                new_p
            }
        };

        if new_ptr.is_null() {
            alloc::alloc::handle_alloc_error(new_layout);
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

impl<T, Align: Alignment> Drop for AlignedVec<T, Align> {
    fn drop(&mut self) {
        if core::mem::size_of::<T>() == 0 {
            if self.len > 0 {
                unsafe {
                    core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.ptr, self.len));
                }
            }
            return;
        }
        if !self.ptr.is_null() && self.cap > 0 {
            unsafe {
                core::ptr::drop_in_place(core::slice::from_raw_parts_mut(self.ptr, self.len));
                let layout = self.layout_for(self.cap);
                if let Some(node) = self.node {
                    let allocator = crate::numa::MnemosyneNumaAllocator;
                    allocator.dealloc_on_node(self.ptr as *mut u8, layout, node);
                } else {
                    dealloc(self.ptr as *mut u8, layout);
                }
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
            }
        }
        new_vec.len = self.len;
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

impl<T: PartialEq, Align1: Alignment, Align2: Alignment> PartialEq<AlignedVec<T, Align2>> for AlignedVec<T, Align1> {
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

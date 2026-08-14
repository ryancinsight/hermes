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
        Layout::from_size_align(size, align)
            .expect("align is power-of-2, size validated by checked_mul")
    }

    /// Create a new empty `AlignedVec` with no allocation.
    #[inline]
    #[must_use]
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
    #[must_use]
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
            unsafe { core::alloc::GlobalAlloc::alloc(&mnemosyne::Mnemosyne, layout).cast::<T>() };
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
    #[must_use]
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
        let ptr = unsafe { allocator.alloc_on_node(layout, node).cast::<T>() };
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
    ///
    /// # Panics
    ///
    /// Panics if the length of a zero-sized element vector would overflow
    /// `usize`.
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

    /// Reserve capacity for at least `additional` more elements beyond `len` in
    /// a single reallocation; a no-op when capacity already suffices.
    ///
    /// Growth is geometric — the new capacity is at least double the old — so a
    /// sequence of `reserve`/`push` calls keeps amortized O(1) append while a
    /// one-shot `reserve(n)` before a bulk append performs exactly one
    /// allocation instead of the `⌈log₂ n⌉` reallocations a push loop incurs.
    ///
    /// # Panics
    /// If `len + additional` overflows `usize`, or on allocator failure.
    pub fn reserve(&mut self, additional: usize) {
        if core::mem::size_of::<T>() == 0 {
            self.cap = usize::MAX;
            return;
        }
        let needed = self.len.checked_add(additional).expect("Capacity overflow");
        if needed <= self.cap {
            return;
        }
        // Grow to at least `needed`, but never less than doubling, so a bulk
        // reserve honors the exact target while incremental reserves preserve
        // the geometric-growth amortization.
        let new_cap = needed.max(self.cap.saturating_mul(2)).max(4);
        self.grow_to(new_cap);
    }

    /// Append every element of `src` in a single reserve + `copy_nonoverlapping`.
    ///
    /// For `T: Copy` this is the bulk counterpart of [`push`](Self::push): one
    /// allocation sized to fit (via [`reserve`](Self::reserve)) then one
    /// contiguous memcpy, versus a per-element push loop's repeated bounds check
    /// and geometric reallocations.
    pub fn extend_from_slice(&mut self, src: &[T])
    where
        T: Copy,
    {
        self.reserve(src.len());
        if src.is_empty() {
            return;
        }
        // SAFETY: `reserve(src.len())` guaranteed `cap ≥ len + src.len()`, so the
        // destination `[len, len + src.len())` is within the allocation and
        // disjoint from `src` (distinct allocations). `T: Copy` ⇒ no drop/overlap
        // hazard. For a ZST the copy is a no-op on the dangling-but-aligned ptr.
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(self.len), src.len());
            self.len += src.len();
        }
    }

    /// Returns the number of elements in the vector.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the vector contains no elements.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the vector.
    #[inline(always)]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns a raw pointer to the vector's buffer.
    #[inline(always)]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a raw mutable pointer to the vector's buffer.
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns the reserved-but-uninitialized tail `[len, capacity)` as a slice
    /// of `MaybeUninit<T>`.
    ///
    /// A routine that fills this slice and then advances the length with
    /// [`set_len`](Self::set_len) initializes each element exactly once with no
    /// intervening zero-fill — the memory-efficient alternative to reserving,
    /// zeroing, and overwriting. Immediately after `with_capacity(n)` the length
    /// is zero, so this covers the whole `n`-element allocation.
    #[inline(always)]
    pub fn spare_capacity_mut(&mut self) -> &mut [core::mem::MaybeUninit<T>] {
        // SAFETY: `ptr` is valid for `cap` elements and `len <= cap`, so the
        // `[len, cap)` region lies within the allocation. `MaybeUninit<T>` has
        // the same layout as `T` and imposes no initialization invariant, so a
        // mutable slice of it over reserved capacity is sound even though those
        // elements are not yet initialized.
        unsafe {
            core::slice::from_raw_parts_mut(
                self.ptr.add(self.len).cast::<core::mem::MaybeUninit<T>>(),
                self.cap - self.len,
            )
        }
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
    #[must_use]
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
    ///
    /// # Panics
    ///
    /// Panics if the vector's internal alignment invariant is violated.
    #[inline(always)]
    #[must_use]
    pub fn view<Arch>(&self) -> SimdView<'_, T, Arch, Align, crate::execution::Unmasked, &[T]>
    where
        Arch: crate::arch::SimdArch,
    {
        SimdView::new(self.as_slice())
            .expect("AlignedVec guarantees aligned buffer of sufficient length")
    }

    /// Obtains a compile-time safe mutable `SimdView` over the vector's buffer.
    ///
    /// # Panics
    ///
    /// Panics if the vector's internal alignment invariant is violated.
    #[inline(always)]
    pub fn view_mut<Arch>(
        &mut self,
    ) -> SimdView<'_, T, Arch, Align, crate::execution::Unmasked, &mut [T]>
    where
        Arch: crate::arch::SimdArch,
    {
        SimdView::new_mut(self.as_mut_slice())
            .expect("AlignedVec guarantees aligned buffer of sufficient length")
    }

    /// Converts this `AlignedVec` to another alignment layout type-safely, without checking
    /// if the pointer satisfies the new alignment's constraints.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the underlying memory address satisfies the alignment
    /// boundary constraints of `NewAlign`.
    #[inline(always)]
    #[must_use]
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
    #[must_use]
    pub fn into_unaligned(self) -> AlignedVec<T, crate::align::Unaligned> {
        unsafe { self.into_alignment_unchecked() }
    }

    /// Attempts to cast this `AlignedVec` to a stricter alignment constraint.
    /// Returns `Some` if the pointer satisfies the alignment requirement of `NewAlign`, otherwise `None`.
    #[inline]
    #[must_use]
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
        self.grow_to(new_cap);
    }

    /// Reallocate the backing storage to exactly `new_cap` elements in a single
    /// allocator call, preserving the existing `len` initialized elements.
    ///
    /// SSOT for capacity growth: [`grow`] (geometric doubling for `push`) and
    /// [`reserve`](Self::reserve) (grow to an explicit target) both delegate here
    /// so the NUMA / mnemosyne / global-allocator branch logic lives once.
    ///
    /// # Panics
    /// On allocator failure (`handle_alloc_error`) or capacity-layout overflow.
    /// Caller guarantees `T` is not a ZST and `new_cap > self.cap`.
    fn grow_to(&mut self, new_cap: usize) {
        let new_layout = self.layout_for(new_cap);

        let old_ptr = self.ptr;
        let new_ptr = if self.cap == 0 {
            if let Some(node) = self.node {
                let allocator = crate::numa::MnemosyneNumaAllocator;
                unsafe { allocator.alloc_on_node(new_layout, node).cast::<T>() }
            } else {
                #[cfg(feature = "mnemosyne-memory")]
                unsafe {
                    core::alloc::GlobalAlloc::alloc(&mnemosyne::Mnemosyne, new_layout).cast::<T>()
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
                    allocator
                        .realloc_on_node(self.ptr.cast::<u8>(), old_layout, new_layout, node)
                        .cast::<T>()
                } else {
                    #[cfg(feature = "mnemosyne-memory")]
                    let ptr = core::alloc::GlobalAlloc::realloc(
                        &mnemosyne::Mnemosyne,
                        self.ptr.cast::<u8>(),
                        old_layout,
                        new_layout.size(),
                    )
                    .cast::<T>();
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
                    allocator.dealloc_on_node(self.ptr.cast::<u8>(), layout, node);
                } else {
                    #[cfg(feature = "mnemosyne-memory")]
                    core::alloc::GlobalAlloc::dealloc(
                        &mnemosyne::Mnemosyne,
                        self.ptr.cast::<u8>(),
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

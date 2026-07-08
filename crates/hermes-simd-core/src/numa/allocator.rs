use crate::numa::binding::NumaBinding;
use core::alloc::Layout;

/// Trait representing a NUMA-aware memory allocator.
pub trait NumaAllocator: Send + Sync {
    /// Allocate memory on a specific NUMA node.
    ///
    /// # Safety
    /// - `layout` must have non-zero size.
    unsafe fn alloc_on_node(&self, layout: Layout, node: u32) -> *mut u8;

    /// Deallocate memory previously allocated by this allocator.
    ///
    /// # Safety
    /// - `ptr` must have been allocated by this allocator with `layout` on `node`.
    unsafe fn dealloc_on_node(&self, ptr: *mut u8, layout: Layout, node: u32);

    /// Reallocate memory previously allocated by this allocator on a specific NUMA node.
    ///
    /// # Safety
    /// - `ptr` must have been allocated by this allocator with `layout` on `node`.
    /// - `new_layout` must have non-zero size.
    unsafe fn realloc_on_node(
        &self,
        ptr: *mut u8,
        layout: Layout,
        new_layout: Layout,
        node: u32,
    ) -> *mut u8 {
        let new_ptr = self.alloc_on_node(new_layout, node);
        if !new_ptr.is_null() {
            let copy_size = core::cmp::min(layout.size(), new_layout.size());
            core::ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);
            self.dealloc_on_node(ptr, layout, node);
        }
        new_ptr
    }
}

/// NUMA-associated allocator backed by Mnemosyne when `mnemosyne-memory` is enabled.
pub struct MnemosyneNumaAllocator;

impl NumaAllocator for MnemosyneNumaAllocator {
    unsafe fn alloc_on_node(&self, layout: Layout, node: u32) -> *mut u8 {
        #[cfg(feature = "mnemosyne-memory")]
        {
            let _binding = NumaBinding::bind(node);
            unsafe { core::alloc::GlobalAlloc::alloc(&mnemosyne::Mnemosyne, layout) }
        }
        #[cfg(not(feature = "mnemosyne-memory"))]
        {
            let _binding = NumaBinding::bind(node);
            alloc::alloc::alloc(layout)
        }
    }

    unsafe fn dealloc_on_node(&self, ptr: *mut u8, layout: Layout, _node: u32) {
        super::locality::bump_alloc_generation();
        #[cfg(feature = "mnemosyne-memory")]
        {
            // No NUMA binding: Mnemosyne routes a free by the pointer's owning
            // segment, not the calling thread's node, so binding here only adds
            // an affinity round-trip with no placement effect.
            unsafe {
                core::alloc::GlobalAlloc::dealloc(&mnemosyne::Mnemosyne, ptr, layout);
            }
        }
        #[cfg(not(feature = "mnemosyne-memory"))]
        {
            let _ = _node;
            alloc::alloc::dealloc(ptr, layout);
        }
    }

    unsafe fn realloc_on_node(
        &self,
        ptr: *mut u8,
        layout: Layout,
        new_layout: Layout,
        node: u32,
    ) -> *mut u8 {
        #[cfg(feature = "mnemosyne-memory")]
        {
            let _binding = NumaBinding::bind(node);
            let new_ptr = unsafe {
                core::alloc::GlobalAlloc::realloc(
                    &mnemosyne::Mnemosyne,
                    ptr,
                    layout,
                    new_layout.size(),
                )
            };
            if !new_ptr.is_null() && new_ptr != ptr {
                super::locality::bump_alloc_generation();
            }
            new_ptr
        }
        #[cfg(not(feature = "mnemosyne-memory"))]
        {
            let new_ptr = self.alloc_on_node(new_layout, node);
            if !new_ptr.is_null() {
                let copy_size = core::cmp::min(layout.size(), new_layout.size());
                core::ptr::copy_nonoverlapping(ptr, new_ptr, copy_size);
                self.dealloc_on_node(ptr, layout, node);
            }
            new_ptr
        }
    }
}

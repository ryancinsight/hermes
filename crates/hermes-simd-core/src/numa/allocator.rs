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

/// Default system NUMA allocator that hooks into Mnemosyne or uses platform APIs.
pub struct MnemosyneNumaAllocator;

impl NumaAllocator for MnemosyneNumaAllocator {
    unsafe fn alloc_on_node(&self, layout: Layout, node: u32) -> *mut u8 {
        #[cfg(feature = "mnemosyne-memory")]
        {
            let _binding = NumaBinding::bind(node);
            unsafe { core::alloc::GlobalAlloc::alloc(&mnemosyne::Mnemosyne, layout) }
        }
        #[cfg(all(
            not(feature = "mnemosyne-memory"),
            target_os = "linux",
            feature = "libnuma"
        ))]
        {
            #[link(name = "numa")]
            extern "C" {
                fn numa_alloc_onnode(size: usize, node: i32) -> *mut u8;
            }
            numa_alloc_onnode(layout.size(), node as i32)
        }
        #[cfg(all(not(feature = "mnemosyne-memory"), target_os = "windows"))]
        {
            extern "system" {
                fn GetCurrentProcess() -> *mut core::ffi::c_void;
                fn VirtualAllocExNuma(
                    hProcess: *mut core::ffi::c_void,
                    lpAddress: *mut core::ffi::c_void,
                    dwSize: usize,
                    flAllocationType: u32,
                    flProtect: u32,
                    nndPreferred: u32,
                ) -> *mut core::ffi::c_void;
            }
            const MEM_COMMIT: u32 = 0x00001000;
            const MEM_RESERVE: u32 = 0x00002000;
            const PAGE_READWRITE: u32 = 0x04;

            let ptr = VirtualAllocExNuma(
                GetCurrentProcess(),
                core::ptr::null_mut(),
                layout.size(),
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
                node,
            );
            if ptr.is_null() {
                core::ptr::null_mut()
            } else {
                ptr as *mut u8
            }
        }
        #[cfg(not(any(
            feature = "mnemosyne-memory",
            all(target_os = "linux", feature = "libnuma"),
            target_os = "windows"
        )))]
        {
            let _ = node;
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
        #[cfg(all(
            not(feature = "mnemosyne-memory"),
            target_os = "linux",
            feature = "libnuma"
        ))]
        {
            #[link(name = "numa")]
            extern "C" {
                fn numa_free(ptr: *mut u8, size: usize);
            }
            numa_free(ptr, layout.size());
        }
        #[cfg(all(not(feature = "mnemosyne-memory"), target_os = "windows"))]
        {
            extern "system" {
                fn GetCurrentProcess() -> *mut core::ffi::c_void;
                fn VirtualFreeEx(
                    hProcess: *mut core::ffi::c_void,
                    lpAddress: *mut core::ffi::c_void,
                    dwSize: usize,
                    dwFreeType: u32,
                ) -> i32;
            }
            const MEM_RELEASE: u32 = 0x00008000;
            let _res = VirtualFreeEx(
                GetCurrentProcess(),
                ptr as *mut core::ffi::c_void,
                0,
                MEM_RELEASE,
            );
        }
        #[cfg(not(any(
            feature = "mnemosyne-memory",
            all(target_os = "linux", feature = "libnuma"),
            target_os = "windows"
        )))]
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

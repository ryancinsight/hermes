//! NUMA-aware memory allocation and thread affinity interfaces.
//!
//! Provides explicit node placement capabilities for Windows (VirtualAllocExNuma)
//! and Linux (numa_alloc_onnode) architectures with standard allocator fallback paths,
//! alongside thread affinity pinning and memory residency verification.
//!
//! # `libnuma` feature
//!
//! The Linux node-placement paths call into `libnuma` and therefore require
//! linking `-lnuma`. They are gated behind the off-by-default `libnuma` cargo
//! feature so plain Linux builds carry no shared-library dependency; without
//! the feature every API degrades to the documented portable fallback
//! (standard allocator, single-node topology, fixed remote distance).
//! Linux paths that only need libc (`sched_getcpu`, `mincore`, `move_pages`)
//! remain active unconditionally.

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
}

/// Default system NUMA allocator that hooks into Mnemosyne or uses platform APIs.
pub struct MnemosyneNumaAllocator;

impl NumaAllocator for MnemosyneNumaAllocator {
    unsafe fn alloc_on_node(&self, layout: Layout, node: u32) -> *mut u8 {
        #[cfg(all(target_os = "linux", feature = "libnuma"))]
        {
            // On Linux we can use numa_alloc_onnode.
            #[link(name = "numa")]
            extern "C" {
                fn numa_alloc_onnode(size: usize, node: i32) -> *mut u8;
            }
            let ptr = numa_alloc_onnode(layout.size(), node as i32);
            if ptr.is_null() {
                alloc::alloc::alloc(layout)
            } else {
                ptr
            }
        }
        #[cfg(target_os = "windows")]
        {
            // On Windows we can use VirtualAllocExNuma.
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
                alloc::alloc::alloc(layout)
            } else {
                ptr as *mut u8
            }
        }
        #[cfg(not(any(all(target_os = "linux", feature = "libnuma"), target_os = "windows")))]
        {
            let _ = node;
            alloc::alloc::alloc(layout)
        }
    }

    unsafe fn dealloc_on_node(&self, ptr: *mut u8, layout: Layout, _node: u32) {
        #[cfg(all(target_os = "linux", feature = "libnuma"))]
        {
            #[link(name = "numa")]
            extern "C" {
                fn numa_free(ptr: *mut u8, size: usize);
            }
            numa_free(ptr, layout.size());
        }
        #[cfg(target_os = "windows")]
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
            let res = VirtualFreeEx(
                GetCurrentProcess(),
                ptr as *mut core::ffi::c_void,
                0,
                MEM_RELEASE,
            );
            if res == 0 {
                alloc::alloc::dealloc(ptr, layout);
            }
        }
        #[cfg(not(any(all(target_os = "linux", feature = "libnuma"), target_os = "windows")))]
        {
            let _ = _node;
            alloc::alloc::dealloc(ptr, layout);
        }
    }
}

/// Returns the index of the NUMA node the current thread is executing on.
pub fn current_numa_node() -> Option<u32> {
    #[cfg(all(target_os = "linux", feature = "libnuma"))]
    {
        #[link(name = "numa")]
        extern "C" {
            fn sched_getcpu() -> i32;
            fn numa_node_of_cpu(cpu: i32) -> i32;
        }
        unsafe {
            let cpu = sched_getcpu();
            if cpu >= 0 {
                let node = numa_node_of_cpu(cpu);
                if node >= 0 {
                    return Some(node as u32);
                }
            }
            None
        }
    }
    #[cfg(target_os = "windows")]
    {
        extern "system" {
            fn GetCurrentProcessorNumber() -> u32;
            fn GetNumaProcessorNode(processor: u8, node_number: *mut u8) -> i32;
        }
        unsafe {
            let cpu = GetCurrentProcessorNumber() as u8;
            let mut node = 0u8;
            if GetNumaProcessorNode(cpu, &mut node) != 0 {
                Some(node as u32)
            } else {
                None
            }
        }
    }
    #[cfg(not(any(all(target_os = "linux", feature = "libnuma"), target_os = "windows")))]
    {
        None
    }
}

/// Refreshes and returns the current NUMA node index for the executing thread.
pub fn refresh_numa_node() -> Option<u32> {
    current_numa_node()
}

/// Returns the total number of NUMA nodes configured in the system.
pub fn numa_node_count() -> u32 {
    NumaTopologyService::total_nodes()
}

/// Retrieve the NUMA distance between two NUMA nodes.
/// Returns 10 if node_a == node_b, and queries the OS distance tables or returns 20 as remote fallback.
pub fn numa_node_distance(node_a: u32, node_b: u32) -> u32 {
    if node_a == node_b {
        return 10;
    }
    #[cfg(all(target_os = "linux", feature = "libnuma"))]
    {
        #[link(name = "numa")]
        extern "C" {
            fn numa_distance(node1: i32, node2: i32) -> i32;
        }
        let dist = unsafe { numa_distance(node_a as i32, node_b as i32) };
        if dist > 0 {
            return dist as u32;
        }
    }
    20
}

/// Verify if the physical memory backing a pointer range is resident on a specific node.
/// Topology service to query NUMA nodes and logical processors.
pub struct NumaTopologyService;

impl NumaTopologyService {
    /// Query the current CPU/processor index.
    pub fn current_cpu() -> Option<u32> {
        #[cfg(target_os = "linux")]
        {
            extern "C" {
                fn sched_getcpu() -> i32;
            }
            let cpu = unsafe { sched_getcpu() };
            if cpu >= 0 {
                Some(cpu as u32)
            } else {
                None
            }
        }
        #[cfg(target_os = "windows")]
        {
            extern "system" {
                fn GetCurrentProcessorNumber() -> u32;
            }
            Some(unsafe { GetCurrentProcessorNumber() })
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            None
        }
    }

    /// Query the current NUMA node ID.
    pub fn current_node() -> Option<u32> {
        current_numa_node()
    }

    /// Get total number of NUMA nodes in the system.
    pub fn total_nodes() -> u32 {
        #[cfg(all(target_os = "linux", feature = "libnuma"))]
        {
            #[link(name = "numa")]
            extern "C" {
                fn numa_num_configured_nodes() -> i32;
            }
            let count = unsafe { numa_num_configured_nodes() };
            if count > 0 {
                count as u32
            } else {
                1
            }
        }
        #[cfg(target_os = "windows")]
        {
            extern "system" {
                fn GetNumaHighestNodeNumber(highest_node_number: *mut u32) -> i32;
            }
            let mut highest = 0;
            if unsafe { GetNumaHighestNodeNumber(&mut highest) } != 0 {
                highest + 1
            } else {
                1
            }
        }
        #[cfg(not(any(all(target_os = "linux", feature = "libnuma"), target_os = "windows")))]
        {
            1
        }
    }

    /// Query the distance between node_a and node_b.
    pub fn node_distance(node_a: u32, node_b: u32) -> u32 {
        numa_node_distance(node_a, node_b)
    }
}

/// Verify if the physical memory backing a pointer range is resident on a specific node.
pub fn verify_numa_locality(ptr: *const u8, size: usize, expected_node: u32) -> bool {
    let _ = size;
    let segment_ptr = (ptr as usize) & !(2 * 1024 * 1024 - 1);

    #[cfg(target_os = "windows")]
    unsafe {
        use core::ffi::c_void;
        #[repr(C)]
        struct MemoryBasicInformation {
            base_address: *mut c_void,
            allocation_base: *mut c_void,
            allocation_protect: u32,
            partition_id: u16,
            region_size: usize,
            state: u32,
            protect: u32,
            type_: u32,
        }
        extern "system" {
            fn VirtualQuery(
                lpAddress: *const c_void,
                lpBuffer: *mut MemoryBasicInformation,
                dwLength: usize,
            ) -> usize;
        }
        let mut info = core::mem::zeroed::<MemoryBasicInformation>();
        let res_size = VirtualQuery(
            segment_ptr as *const c_void,
            &mut info,
            core::mem::size_of::<MemoryBasicInformation>(),
        );
        if res_size > 0 && info.state == 0x1000 {
            // MEM_COMMIT
            let owner_ptr = (segment_ptr + 8) as *const *const c_void;
            let owner = *owner_ptr;
            if !owner.is_null() {
                let node_ptr = (segment_ptr + 60) as *const u32;
                let node = *node_ptr;
                return node == expected_node;
            }
        }
    }

    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn mincore(addr: *mut core::ffi::c_void, length: usize, vec: *mut u8) -> i32;
            fn move_pages(
                pid: i32,
                count: usize,
                pages: *const *mut core::ffi::c_void,
                nodes: *const i32,
                status: *mut i32,
                flags: i32,
            ) -> i32;
        }
        let mut vec = 0u8;
        let res = mincore(segment_ptr as *mut core::ffi::c_void, 4096, &mut vec);
        if res == 0 && (vec & 1) != 0 {
            let owner_ptr = (segment_ptr + 8) as *const *const core::ffi::c_void;
            let owner = *owner_ptr;
            if !owner.is_null() {
                let node_ptr = (segment_ptr + 60) as *const u32;
                let node = *node_ptr;
                return node == expected_node;
            }
        }

        // Fallback to standard move_pages check if not a Mnemosyne segment or if owner check is skipped
        let page_size = 4096;
        let start_page = (ptr as usize) & !(page_size - 1);
        let end_page = ((ptr as usize) + size + page_size - 1) & !(page_size - 1);
        let pages_count = (end_page - start_page) / page_size;
        if pages_count == 0 {
            return true;
        }
        let mut pages = alloc::vec![core::ptr::null_mut(); pages_count];
        for i in 0..pages_count {
            pages[i] = (start_page + i * page_size) as *mut core::ffi::c_void;
        }
        let mut status = alloc::vec![0i32; pages_count];
        if move_pages(
            0,
            pages_count,
            pages.as_ptr(),
            core::ptr::null(),
            status.as_mut_ptr(),
            0,
        ) >= 0
        {
            return status.iter().all(|&node| node == expected_node as i32);
        }
    }

    // Default fallback
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = size;
        let _ = expected_node;
    }
    true
}

/// RAII scope guard that binds the current thread to a specific NUMA node.
pub struct NumaBinding {
    #[cfg(all(target_os = "linux", feature = "libnuma"))]
    old_mask: *mut core::ffi::c_void,
    #[cfg(target_os = "windows")]
    old_mask: usize,
}

impl NumaBinding {
    /// Bind the current thread to the specified NUMA node.
    pub fn bind(node: u32) -> Self {
        #[cfg(all(target_os = "linux", feature = "libnuma"))]
        {
            #[link(name = "numa")]
            extern "C" {
                fn numa_allocate_nodemask() -> *mut core::ffi::c_void;
                fn numa_bitmask_setbit(mask: *mut core::ffi::c_void, bit: u32);
                fn numa_bind(mask: *mut core::ffi::c_void);
                fn numa_get_run_node_mask() -> *mut core::ffi::c_void;
                fn numa_bitmask_free(mask: *mut core::ffi::c_void);
            }
            unsafe {
                let old = numa_get_run_node_mask();
                let mask = numa_allocate_nodemask();
                if !mask.is_null() {
                    numa_bitmask_setbit(mask, node);
                    numa_bind(mask);
                    numa_bitmask_free(mask);
                }
                Self { old_mask: old }
            }
        }
        #[cfg(target_os = "windows")]
        {
            extern "system" {
                fn GetCurrentThread() -> *mut core::ffi::c_void;
                fn SetThreadAffinityMask(
                    hThread: *mut core::ffi::c_void,
                    dwThreadAffinityMask: usize,
                ) -> usize;
                fn GetNumaNodeProcessorMask(node: u8, processor_mask: *mut u64) -> i32;
            }
            unsafe {
                let thread = GetCurrentThread();
                let mut mask = 0u64;
                if GetNumaNodeProcessorMask(node as u8, &mut mask) != 0 && mask != 0 {
                    let old = SetThreadAffinityMask(thread, mask as usize);
                    Self { old_mask: old }
                } else {
                    Self { old_mask: 0 }
                }
            }
        }
        #[cfg(not(any(all(target_os = "linux", feature = "libnuma"), target_os = "windows")))]
        {
            let _ = node;
            Self {}
        }
    }
}

#[cfg(any(all(target_os = "linux", feature = "libnuma"), target_os = "windows"))]
impl Drop for NumaBinding {
    fn drop(&mut self) {
        #[cfg(all(target_os = "linux", feature = "libnuma"))]
        {
            if !self.old_mask.is_null() {
                #[link(name = "numa")]
                extern "C" {
                    fn numa_bind(mask: *mut core::ffi::c_void);
                    fn numa_bitmask_free(mask: *mut core::ffi::c_void);
                }
                unsafe {
                    numa_bind(self.old_mask);
                    numa_bitmask_free(self.old_mask);
                }
            }
        }
        #[cfg(target_os = "windows")]
        {
            if self.old_mask != 0 {
                extern "system" {
                    fn GetCurrentThread() -> *mut core::ffi::c_void;
                    fn SetThreadAffinityMask(
                        hThread: *mut core::ffi::c_void,
                        dwThreadAffinityMask: usize,
                    ) -> usize;
                }
                unsafe {
                    SetThreadAffinityMask(GetCurrentThread(), self.old_mask);
                }
            }
        }
    }
}

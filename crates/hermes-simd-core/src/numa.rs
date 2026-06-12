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
        #[cfg(all(not(feature = "mnemosyne-memory"), target_os = "windows"))]
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
        #[cfg(feature = "mnemosyne-memory")]
        {
            let _binding = NumaBinding::bind(_node);
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
}

/// Returns the index of the NUMA node the current thread is executing on.
///
/// Delegates to the themis topology SSOT. `None` means the platform did not
/// report a node — never a fabricated node 0.
pub fn current_numa_node() -> Option<u32> {
    themis::try_current_numa_node().map(|node| node.get())
}

/// Refreshes and returns the current NUMA node index for the executing thread.
pub fn refresh_numa_node() -> Option<u32> {
    current_numa_node()
}

/// Returns the total number of NUMA nodes configured in the system.
pub fn numa_node_count() -> u32 {
    NumaTopologyService::total_nodes()
}

/// Process-cached themis topology snapshot: detection reads sysfs / Windows
/// topology tables once; every later query borrows the cached result.
#[cfg(feature = "std")]
fn topology() -> Option<&'static themis::CpuTopology> {
    static TOPOLOGY: std::sync::OnceLock<Option<themis::CpuTopology>> = std::sync::OnceLock::new();
    TOPOLOGY.get_or_init(themis::CpuTopology::detect).as_ref()
}

/// Retrieve the NUMA distance between two NUMA nodes.
///
/// Returns the themis topology distance-table entry; when no topology is
/// available (detection failed or `no_std`) it falls back to the documented
/// 10 local / 20 remote convention.
pub fn numa_node_distance(node_a: u32, node_b: u32) -> u32 {
    #[cfg(feature = "std")]
    {
        use themis::NumaNodeId;
        if let Some(topology) = topology() {
            return topology.distance(NumaNodeId::new(node_a), NumaNodeId::new(node_b));
        }
    }
    if node_a == node_b {
        10
    } else {
        20
    }
}

/// Topology service to query NUMA nodes and logical processors.
///
/// Detection is delegated to themis (`CpuTopology`, `try_current_numa_node`,
/// `current_processor`) — the stack topology SSOT; this type keeps the
/// SIMD-facing query surface stable.
pub struct NumaTopologyService;

impl NumaTopologyService {
    /// Query the current CPU/processor index.
    pub fn current_cpu() -> Option<u32> {
        themis::current_processor()
    }

    /// Query the current NUMA node ID.
    pub fn current_node() -> Option<u32> {
        current_numa_node()
    }

    /// Get total number of NUMA nodes in the system.
    pub fn total_nodes() -> u32 {
        #[cfg(feature = "std")]
        {
            topology().map_or(1, |t| {
                u32::try_from(t.numa_nodes().len().max(1)).unwrap_or(u32::MAX)
            })
        }
        #[cfg(not(feature = "std"))]
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

        // Fallback to a move_pages residency query if not a Mnemosyne segment.
        // move_pages(2) is wrapped by libnuma, not libc, so this refinement is
        // only available with the `libnuma` feature; without it the function
        // reports locality optimistically (documented portable fallback).
        #[cfg(feature = "libnuma")]
        {
            #[link(name = "numa")]
            extern "C" {
                fn move_pages(
                    pid: i32,
                    count: usize,
                    pages: *const *mut core::ffi::c_void,
                    nodes: *const i32,
                    status: *mut i32,
                    flags: i32,
                ) -> i32;
            }
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

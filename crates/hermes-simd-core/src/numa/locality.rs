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

#[cfg(feature = "std")]
#[derive(Copy, Clone)]
struct CacheEntry {
    ptr_start: usize,
    ptr_end: usize,
    node: u32,
    generation: u64,
    local: bool,
}

#[cfg(feature = "std")]
struct LocalityCache {
    entries: [Option<CacheEntry>; 16],
    next_idx: usize,
}

#[cfg(feature = "std")]
impl LocalityCache {
    const fn new() -> Self {
        Self {
            entries: [None; 16],
            next_idx: 0,
        }
    }
}

#[cfg(feature = "std")]
thread_local! {
    static LOCALITY_CACHE: core::cell::RefCell<LocalityCache> = const {
        core::cell::RefCell::new(LocalityCache::new())
    };
}

#[cfg(feature = "std")]
#[repr(align(64))]
struct CacheAlignedAtomicU64(core::sync::atomic::AtomicU64);

#[cfg(feature = "std")]
static ALLOC_GENERATION: CacheAlignedAtomicU64 =
    CacheAlignedAtomicU64(core::sync::atomic::AtomicU64::new(0));

/// Bump the global allocation generation counter.
///
/// This invalidates thread-local locality cache entries, preventing stale cache hits
/// when virtual memory addresses are deallocated and subsequently reallocated.
#[inline]
pub fn bump_alloc_generation() {
    #[cfg(feature = "std")]
    ALLOC_GENERATION
        .0
        .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
}

/// Returns the current global allocation generation counter.
#[inline]
pub fn get_alloc_generation() -> u64 {
    #[cfg(feature = "std")]
    {
        ALLOC_GENERATION
            .0
            .load(core::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "std"))]
    {
        0
    }
}

/// Verify if the physical memory backing a pointer range is resident on a specific node.
pub fn verify_numa_locality(ptr: *const u8, size: usize, expected_node: u32) -> bool {
    #[cfg(feature = "std")]
    {
        let ptr_val = ptr as usize;
        let gen = get_alloc_generation();

        let cached = LOCALITY_CACHE.with(|cache| {
            let cache_ref = cache.borrow();
            let end_val = ptr_val.saturating_add(size);
            for entry in cache_ref.entries.iter().flatten() {
                if entry.node == expected_node
                    && entry.generation == gen
                    && ptr_val >= entry.ptr_start
                    && end_val <= entry.ptr_end
                {
                    return Some(entry.local);
                }
            }
            None
        });

        if let Some(local) = cached {
            return local;
        }
    }

    let local = verify_numa_locality_os(ptr, size, expected_node);

    #[cfg(feature = "std")]
    {
        let ptr_val = ptr as usize;
        let gen = get_alloc_generation();
        LOCALITY_CACHE.with(|cache| {
            let mut cache_mut = cache.borrow_mut();
            let idx = cache_mut.next_idx;
            let end_val = ptr_val.saturating_add(size);
            cache_mut.entries[idx] = Some(CacheEntry {
                ptr_start: ptr_val,
                ptr_end: end_val,
                node: expected_node,
                generation: gen,
                local,
            });
            cache_mut.next_idx = (idx + 1) % 16;
        });
    }

    local
}

fn verify_numa_locality_os(ptr: *const u8, size: usize, expected_node: u32) -> bool {
    #[cfg(target_os = "windows")]
    unsafe {
        use core::ffi::c_void;
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct PsapiWorkingSetExInformation {
            virtual_address: *mut c_void,
            virtual_attributes: usize,
        }
        extern "system" {
            fn GetCurrentProcess() -> *mut c_void;
            fn K32QueryWorkingSetEx(hProcess: *mut c_void, pv: *mut c_void, cb: u32) -> i32;
        }
        let page_size = 4096;
        let start_page = (ptr as usize) & !(page_size - 1);
        let end_page = ((ptr as usize) + size + page_size - 1) & !(page_size - 1);
        let pages_count = (end_page - start_page) / page_size;
        if pages_count == 0 {
            return true;
        }

        const CHUNK_SIZE: usize = 64;
        let mut info_arr = [PsapiWorkingSetExInformation {
            virtual_address: core::ptr::null_mut(),
            virtual_attributes: 0,
        }; CHUNK_SIZE];

        let mut checked = 0;
        while checked < pages_count {
            let chunk_len = core::cmp::min(pages_count - checked, CHUNK_SIZE);
            for i in 0..chunk_len {
                info_arr[i].virtual_address =
                    (start_page + (checked + i) * page_size) as *mut c_void;
                info_arr[i].virtual_attributes = 0;
            }
            let cb = (chunk_len * core::mem::size_of::<PsapiWorkingSetExInformation>()) as u32;
            let res = K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info_arr.as_mut_ptr() as *mut c_void,
                cb,
            );
            if res != 0 {
                for i in 0..chunk_len {
                    let flags = info_arr[i].virtual_attributes;
                    let valid = (flags & 1) != 0;
                    if valid {
                        let node = (flags >> 16) & 0x3F;
                        if node as u32 != expected_node {
                            return false;
                        }
                    }
                }
            } else {
                return false;
            }
            checked += chunk_len;
        }
        true
    }

    #[cfg(target_os = "linux")]
    unsafe {
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

            const CHUNK_SIZE: usize = 64;
            let mut pages_arr = [core::ptr::null_mut(); CHUNK_SIZE];
            let mut status_arr = [0i32; CHUNK_SIZE];

            let mut checked = 0;
            while checked < pages_count {
                let chunk_len = core::cmp::min(pages_count - checked, CHUNK_SIZE);
                for i in 0..chunk_len {
                    pages_arr[i] =
                        (start_page + (checked + i) * page_size) as *mut core::ffi::c_void;
                }
                let res = move_pages(
                    0,
                    chunk_len,
                    pages_arr.as_ptr(),
                    core::ptr::null(),
                    status_arr.as_mut_ptr(),
                    0,
                );
                if res >= 0 {
                    for i in 0..chunk_len {
                        let node = status_arr[i];
                        if node >= 0 && node as u32 != expected_node {
                            return false;
                        }
                    }
                } else {
                    return false;
                }
                checked += chunk_len;
            }
            return true;
        }
        #[cfg(not(feature = "libnuma"))]
        {
            let _ = ptr;
            let _ = size;
            let _ = expected_node;
            true
        }
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        let _ = ptr;
        let _ = size;
        let _ = expected_node;
        true
    }
}

use crate::numa::locality::current_numa_node;

/// RAII scope guard that binds the current thread to a specific NUMA node.
pub struct NumaBinding {
    #[cfg(all(target_os = "linux", feature = "libnuma"))]
    old_mask: *mut core::ffi::c_void,
    #[cfg(target_os = "windows")]
    old_mask: usize,
}

impl NumaBinding {
    /// Bind the current thread to the specified NUMA node.
    #[must_use]
    pub fn bind(node: u32) -> Self {
        if current_numa_node() == Some(node) {
            #[cfg(all(target_os = "linux", feature = "libnuma"))]
            {
                return Self {
                    old_mask: core::ptr::null_mut(),
                };
            }
            #[cfg(target_os = "windows")]
            {
                return Self { old_mask: 0 };
            }
            #[cfg(not(any(all(target_os = "linux", feature = "libnuma"), target_os = "windows")))]
            {
                return Self {};
            }
        }

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

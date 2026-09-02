//! Linux backend for exact processor binding.
//!
//! `sched_getaffinity` / `sched_setaffinity` on the calling thread and
//! `sched_getcpu`, declared directly: three calls do not earn a dependency,
//! and the `cpu_set_t` layout is fixed by the C library ABI.

use super::{ProcessorBindingError, ProcessorIndex};

/// glibc and musl `cpu_set_t`: `CPU_SETSIZE` (1024) bits as 16 machine words.
///
/// Declared here rather than through a bindings crate for the same reason
/// the Windows module declares kernel32 directly: three calls do not earn
/// a dependency, and the layout is fixed by the C library ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub(super) struct CpuSet {
    bits: [u64; 16],
}

const _: () = assert!(core::mem::size_of::<CpuSet>() == 128);
const CPU_SETSIZE: u32 = 128 * 8;
/// `pid` 0 names the calling thread for the scheduler affinity calls.
const CALLING_THREAD: i32 = 0;

impl CpuSet {
    const fn empty() -> Self {
        Self { bits: [0; 16] }
    }

    pub(super) fn contains(&self, processor: u32) -> bool {
        processor < CPU_SETSIZE
            && self.bits[(processor / 64) as usize] & (1u64 << (processor % 64)) != 0
    }

    fn single(processor: u32) -> Option<Self> {
        (processor < CPU_SETSIZE).then(|| {
            let mut set = Self::empty();
            set.bits[(processor / 64) as usize] |= 1u64 << (processor % 64);
            set
        })
    }
}

unsafe extern "C" {
    fn sched_getaffinity(pid: i32, cpusetsize: usize, mask: *mut CpuSet) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const CpuSet) -> i32;
    fn sched_getcpu() -> i32;
    fn __errno_location() -> *mut i32;
}

fn last_error() -> u32 {
    // SAFETY: `__errno_location` returns the calling thread's errno slot,
    // which is always valid and only read here.
    let code = unsafe { *__errno_location() };
    u32::try_from(code).unwrap_or(0)
}

fn platform_error(operation: &'static str) -> ProcessorBindingError {
    ProcessorBindingError::Platform {
        operation,
        code: last_error(),
    }
}

pub(super) fn current_processor() -> Result<ProcessorIndex, ProcessorBindingError> {
    // SAFETY: `sched_getcpu` takes no arguments and only reads scheduler
    // state for the calling thread.
    let cpu = unsafe { sched_getcpu() };
    u32::try_from(cpu)
        .map(ProcessorIndex::new)
        .map_err(|_| platform_error("query current processor"))
}

pub(super) fn thread_affinity() -> Result<CpuSet, ProcessorBindingError> {
    let mut set = CpuSet::empty();
    // SAFETY: `set` is a live, correctly sized `cpu_set_t` for the length
    // passed alongside it, and the call writes only within it.
    let rc =
        unsafe { sched_getaffinity(CALLING_THREAD, core::mem::size_of::<CpuSet>(), &raw mut set) };
    if rc != 0 {
        return Err(platform_error("query current thread affinity"));
    }
    Ok(set)
}

fn apply(set: &CpuSet, operation: &'static str) -> Result<(), ProcessorBindingError> {
    // SAFETY: `set` is a live, correctly sized `cpu_set_t` read only for the
    // duration of the call.
    let rc = unsafe {
        sched_setaffinity(
            CALLING_THREAD,
            core::mem::size_of::<CpuSet>(),
            &raw const *set,
        )
    };
    if rc != 0 {
        return Err(platform_error(operation));
    }
    Ok(())
}

/// Bind the calling thread to `processor`; returns the affinity to restore.
///
/// A processor outside the thread's current allowed set — including any
/// index past `CPU_SETSIZE` — is rejected before any mutation, so the
/// caller's affinity is untouched on error.
pub(super) fn bind(processor: ProcessorIndex) -> Result<CpuSet, ProcessorBindingError> {
    let previous = thread_affinity()?;
    let requested = CpuSet::single(processor.get())
        .filter(|_| previous.contains(processor.get()))
        .ok_or(ProcessorBindingError::ProcessorUnavailable { processor })?;
    apply(&requested, "bind current thread")?;
    Ok(previous)
}

pub(super) fn restore(previous: &CpuSet) -> Result<(), ProcessorBindingError> {
    apply(previous, "restore current thread")
}

/// Best-effort restore for `Drop`: a destructor cannot report failure and
/// must not panic, so an error here is deliberately dropped.
pub(super) fn restore_on_drop(previous: &CpuSet) {
    let _ = restore(previous);
}

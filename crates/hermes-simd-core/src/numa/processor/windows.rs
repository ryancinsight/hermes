//! Windows backend for exact processor binding.
//!
//! Group-aware: a logical processor is a (group, number) pair, flattened
//! as `group * 64 + number` to match the stack's topology numbering, and
//! bound with `SetThreadGroupAffinity` through the shared affinity guard.

use super::{ProcessorBindingError, ProcessorIndex};
use crate::numa::affinity::{self, GetActiveProcessorCount, GetActiveProcessorGroupCount};
use themis::ProcessorGroupAffinity;

pub(super) use crate::numa::affinity::{restore_on_drop, GroupAffinity};

const WINDOWS_PROCESSORS_PER_GROUP: u32 = 64;

#[repr(C)]
struct ProcessorNumber {
    group: u16,
    number: u8,
    reserved: u8,
}

const _: () = assert!(core::mem::size_of::<ProcessorNumber>() == 4);

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetCurrentProcessorNumberEx(processor_number: *mut ProcessorNumber);
}

pub(super) fn current_processor() -> ProcessorIndex {
    let mut processor = ProcessorNumber {
        group: 0,
        number: 0,
        reserved: 0,
    };
    // SAFETY: the pointer names a writable, correctly laid-out
    // `PROCESSOR_NUMBER`; the API has no failure result.
    unsafe { GetCurrentProcessorNumberEx(core::ptr::addr_of_mut!(processor)) };
    ProcessorIndex::new(
        u32::from(processor.group) * WINDOWS_PROCESSORS_PER_GROUP + u32::from(processor.number),
    )
}

pub(super) fn bind(processor: ProcessorIndex) -> Result<GroupAffinity, ProcessorBindingError> {
    let requested = ProcessorGroupAffinity::from_processor(processor.get())
        .ok_or_else(|| unavailable(processor))?;
    let group = requested.group();

    // SAFETY: these parameter-free queries have no pointer obligations.
    let group_count = unsafe { GetActiveProcessorGroupCount() };
    if group_count == 0 {
        return Err(platform_error("query active processor groups"));
    }
    if group >= group_count {
        return Err(unavailable(processor));
    }

    // SAFETY: `group` is below the queried active group count.
    let active_count = unsafe { GetActiveProcessorCount(group) };
    if active_count == 0 {
        return Err(platform_error("query active processors in group"));
    }
    if requested.mask().trailing_zeros() >= active_count {
        return Err(unavailable(processor));
    }

    // Validation above proves the requested single-bit group affinity is
    // active, so a failure here is a platform fault rather than a bad mask.
    affinity::bind(&GroupAffinity::new(group, requested.mask()))
        .ok_or_else(|| platform_error("bind current thread"))
}

pub(super) fn restore(previous: &GroupAffinity) -> Result<(), ProcessorBindingError> {
    // `previous` was initialized by a successful bind on this same thread,
    // and `ProcessorBinding` is not Send, so restoration cannot move to
    // another thread.
    affinity::bind(previous)
        .map(|_| ())
        .ok_or_else(|| platform_error("restore current thread"))
}

const fn unavailable(processor: ProcessorIndex) -> ProcessorBindingError {
    ProcessorBindingError::ProcessorUnavailable { processor }
}

fn platform_error(operation: &'static str) -> ProcessorBindingError {
    ProcessorBindingError::Platform {
        operation,
        code: affinity::last_error(),
    }
}

#[cfg(test)]
pub(super) fn thread_affinity() -> Result<GroupAffinity, ProcessorBindingError> {
    affinity::thread_affinity().ok_or_else(|| platform_error("query current thread affinity"))
}

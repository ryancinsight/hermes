//! Windows processor-group affinity primitives shared by the NUMA guards.
//!
//! Windows expresses thread affinity as a `GROUP_AFFINITY`: a processor group
//! paired with a mask naming processors *within* that group. Both affinity
//! guards in this module — the exact-processor guard and the NUMA-node guard —
//! mutate affinity through the same `SetThreadGroupAffinity` contract, so the
//! layout and the `kernel32` declarations live here once. Two independent
//! `#[repr(C)]` declarations of one operating-system structure can drift apart
//! silently, and a second copy of the affinity contract is the fork ADR 021
//! records as the defect to avoid.

/// Windows `GROUP_AFFINITY`: a processor group and a mask within it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub(super) struct GroupAffinity {
    /// Processors selected within [`GroupAffinity::group`].
    pub(super) mask: usize,
    /// Processor group the mask is interpreted in.
    pub(super) group: u16,
    reserved: [u16; 3],
}

impl GroupAffinity {
    /// Affinity naming no processors, used as platform output storage.
    pub(super) const ZERO: Self = Self::new(0, 0);

    /// Construct an affinity naming `mask` within `group`.
    pub(super) const fn new(group: u16, mask: usize) -> Self {
        Self {
            mask,
            group,
            reserved: [0; 3],
        }
    }
}

const _: () = assert!(core::mem::size_of::<GroupAffinity>() == core::mem::size_of::<usize>() + 8);

#[link(name = "kernel32")]
unsafe extern "system" {
    pub(super) fn GetActiveProcessorCount(group_number: u16) -> u32;
    pub(super) fn GetActiveProcessorGroupCount() -> u16;
    pub(super) fn GetCurrentThread() -> *mut core::ffi::c_void;
    pub(super) fn GetLastError() -> u32;
    pub(super) fn SetThreadGroupAffinity(
        thread: *mut core::ffi::c_void,
        group_affinity: *const GroupAffinity,
        previous_group_affinity: *mut GroupAffinity,
    ) -> i32;
    #[cfg(test)]
    fn GetThreadGroupAffinity(
        thread: *mut core::ffi::c_void,
        group_affinity: *mut GroupAffinity,
    ) -> i32;
}

/// Bind the calling thread to `requested`, returning the affinity it replaced.
///
/// Returns `None` when the platform rejects the request. Callers validate the
/// group and mask against the live processor inventory first; this is the
/// single site that mutates affinity.
pub(super) fn bind(requested: &GroupAffinity) -> Option<GroupAffinity> {
    let mut previous = GroupAffinity::ZERO;
    // SAFETY: both affinity pointers name valid, correctly laid-out
    // `GROUP_AFFINITY` storage for the duration of the call, and the current
    // thread pseudo-handle is valid in the calling thread.
    let bound = unsafe {
        SetThreadGroupAffinity(
            GetCurrentThread(),
            core::ptr::from_ref(requested),
            core::ptr::addr_of_mut!(previous),
        )
    };
    (bound != 0).then_some(previous)
}

/// Restore `previous` on the calling thread, discarding the platform result.
///
/// `Drop` cannot surface a platform error; guards that need that signal call
/// their own fallible restore path instead.
pub(super) fn restore_on_drop(previous: &GroupAffinity) {
    // SAFETY: `previous` was produced by a successful bind on this same thread.
    // Neither guard is `Send`, so restoration cannot move to another thread.
    unsafe {
        SetThreadGroupAffinity(
            GetCurrentThread(),
            core::ptr::from_ref(previous),
            core::ptr::null_mut(),
        );
    }
}

/// Capture the calling thread's last-error value after a failed platform call.
pub(super) fn last_error() -> u32 {
    // SAFETY: the query has no pointer obligations and reads thread-local state.
    unsafe { GetLastError() }
}

/// Mask of the processors active in `group`, or `None` when `group` is absent.
///
/// This is the validation both guards apply before mutating affinity: it proves
/// the group exists on this host and bounds the mask to processors the group
/// actually has, so a stale or fabricated processor set cannot reach the
/// platform call.
///
/// Gated on `std`: the NUMA-node guard is its only consumer, and that guard
/// needs the topology detection `std` activates. The exact-processor guard
/// validates a single index and reports typed errors, so it keeps its own
/// checks rather than collapsing into this mask.
#[cfg(feature = "std")]
pub(super) fn active_mask(group: u16) -> Option<usize> {
    // SAFETY: parameter-free query with no pointer obligations.
    let group_count = unsafe { GetActiveProcessorGroupCount() };
    if group_count == 0 || group >= group_count {
        return None;
    }
    // SAFETY: `group` is below the queried active group count.
    let active = unsafe { GetActiveProcessorCount(group) };
    match active {
        0 => None,
        active if active >= usize::BITS => Some(usize::MAX),
        _ => Some((1usize << active) - 1),
    }
}

/// Query the calling thread's current group affinity.
#[cfg(test)]
pub(super) fn thread_affinity() -> Option<GroupAffinity> {
    let mut affinity = GroupAffinity::ZERO;
    // SAFETY: the output pointer names writable `GROUP_AFFINITY` storage.
    let queried =
        unsafe { GetThreadGroupAffinity(GetCurrentThread(), core::ptr::addr_of_mut!(affinity)) };
    (queried != 0).then_some(affinity)
}

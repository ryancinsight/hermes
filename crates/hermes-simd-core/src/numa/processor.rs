use core::{fmt, marker::PhantomData};

/// Operating-system logical processor index.
///
/// On Windows, indices use the same stable flattening as Hermes' topology
/// dependency: `processor_group * 64 + processor_number`. Construction does
/// not query the host; [`ProcessorBinding::bind`] validates availability before
/// changing the calling thread's affinity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ProcessorIndex(u32);

impl ProcessorIndex {
    /// Construct an operating-system logical processor index.
    #[must_use]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Return the underlying operating-system index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Query the logical processor currently executing this thread.
    ///
    /// # Errors
    ///
    /// Returns [`ProcessorBindingError::UnsupportedPlatform`] when Hermes has
    /// no exact-processor backend for the target.
    pub fn current() -> Result<Self, ProcessorBindingError> {
        #[cfg(target_os = "windows")]
        {
            Ok(windows::current_processor())
        }
        #[cfg(target_os = "linux")]
        {
            linux::current_processor()
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            Err(ProcessorBindingError::UnsupportedPlatform)
        }
    }
}

/// Failure to query, bind, or restore exact processor affinity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProcessorBindingError {
    /// The target has no Hermes exact-processor backend.
    UnsupportedPlatform,
    /// The requested logical processor is not active on this host.
    ProcessorUnavailable {
        /// Rejected processor index.
        processor: ProcessorIndex,
    },
    /// An operating-system affinity operation failed.
    Platform {
        /// Platform operation that failed.
        operation: &'static str,
        /// Operating-system error code captured at the failure site.
        code: u32,
    },
}

impl fmt::Display for ProcessorBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                formatter.write_str("exact processor binding is unsupported on this target")
            }
            Self::ProcessorUnavailable { processor } => write!(
                formatter,
                "logical processor {} is not active on this host",
                processor.get()
            ),
            Self::Platform { operation, code } => write!(
                formatter,
                "processor-affinity operation {operation} failed with operating-system error {code}"
            ),
        }
    }
}

impl core::error::Error for ProcessorBindingError {}

/// Thread-bound guard for exact logical-processor affinity.
///
/// Construction binds the calling thread to one processor. Scope exit restores
/// the complete prior affinity. The guard is neither `Send` nor `Sync` because
/// moving it to another thread would restore affinity on the wrong thread.
/// Use [`Self::restore`] when restoration failure must be observed explicitly;
/// [`Drop`] is the panic-free unwind fallback.
///
/// ```compile_fail
/// use hermes_simd_core::{ProcessorBinding, ProcessorIndex};
///
/// let guard = ProcessorBinding::bind(ProcessorIndex::new(0)).unwrap();
/// std::thread::spawn(move || drop(guard));
/// ```
#[must_use = "dropping the guard immediately restores the previous affinity"]
pub struct ProcessorBinding {
    processor: ProcessorIndex,
    #[cfg(target_os = "windows")]
    previous: windows::GroupAffinity,
    #[cfg(target_os = "linux")]
    previous: linux::CpuSet,
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    active: bool,
    thread_bound: PhantomData<*mut ()>,
}

impl ProcessorBinding {
    /// Bind the calling thread to `processor`.
    ///
    /// Validation completes before the platform affinity mutation. An invalid
    /// processor therefore leaves the calling thread unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ProcessorBindingError::ProcessorUnavailable`] for an inactive
    /// index, [`ProcessorBindingError::UnsupportedPlatform`] when the target
    /// has no backend, or [`ProcessorBindingError::Platform`] when the operating
    /// system rejects the query or bind operation.
    pub fn bind(processor: ProcessorIndex) -> Result<Self, ProcessorBindingError> {
        #[cfg(target_os = "windows")]
        {
            let previous = windows::bind(processor)?;
            Ok(Self {
                processor,
                previous,
                active: true,
                thread_bound: PhantomData,
            })
        }
        #[cfg(target_os = "linux")]
        {
            let previous = linux::bind(processor)?;
            Ok(Self {
                processor,
                previous,
                active: true,
                thread_bound: PhantomData,
            })
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            let _ = processor;
            Err(ProcessorBindingError::UnsupportedPlatform)
        }
    }

    /// Return the processor requested when the guard was constructed.
    #[must_use]
    pub const fn processor(&self) -> ProcessorIndex {
        self.processor
    }

    /// Restore the calling thread's prior affinity before scope exit.
    ///
    /// A successful call makes subsequent calls and [`Drop`] no-ops.
    ///
    /// # Errors
    ///
    /// Returns [`ProcessorBindingError::Platform`] when Windows rejects the
    /// saved group affinity. The guard remains active so the caller may retry;
    /// [`Drop`] also makes one final panic-free restoration attempt.
    pub fn restore(&mut self) -> Result<(), ProcessorBindingError> {
        #[cfg(target_os = "windows")]
        {
            if self.active {
                windows::restore(&self.previous)?;
                self.active = false;
            }
            Ok(())
        }
        #[cfg(target_os = "linux")]
        {
            if self.active {
                linux::restore(&self.previous)?;
                self.active = false;
            }
            Ok(())
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            Err(ProcessorBindingError::UnsupportedPlatform)
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for ProcessorBinding {
    fn drop(&mut self) {
        if self.active {
            windows::restore_on_drop(&self.previous);
            self.active = false;
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for ProcessorBinding {
    fn drop(&mut self) {
        if self.active {
            linux::restore_on_drop(&self.previous);
            self.active = false;
        }
    }
}

#[cfg(target_os = "windows")]
mod windows {
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
}

#[cfg(target_os = "linux")]
mod linux {
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
        let rc = unsafe {
            sched_getaffinity(CALLING_THREAD, core::mem::size_of::<CpuSet>(), &raw mut set)
        };
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
}

#[cfg(all(test, target_os = "linux"))]
mod linux_processor_tests {
    use super::{linux, ProcessorBinding, ProcessorBindingError, ProcessorIndex};

    #[test]
    #[cfg_attr(
        miri,
        ignore = "sched_* are foreign scheduler calls miri cannot execute"
    )]
    fn exact_binding_reports_processor_and_restores_on_drop() {
        let before = linux::thread_affinity().expect("current affinity must be queryable");
        let processor = ProcessorIndex::current().expect("Linux supports processor queries");
        {
            let binding =
                ProcessorBinding::bind(processor).expect("current processor must be bindable");
            assert_eq!(binding.processor(), processor);
            std::thread::yield_now();
            assert_eq!(ProcessorIndex::current(), Ok(processor));
            let exact = linux::thread_affinity().expect("bound affinity must be queryable");
            assert!(exact.contains(processor.get()));
            assert!(
                (0..1024u32).filter(|&p| exact.contains(p)).count() == 1,
                "the bound set must hold exactly the requested processor"
            );
        }
        assert_eq!(
            linux::thread_affinity().expect("restored affinity must be queryable"),
            before
        );
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "sched_* are foreign scheduler calls miri cannot execute"
    )]
    fn explicit_restore_is_observable_and_idempotent() {
        let before = linux::thread_affinity().expect("current affinity must be queryable");
        let processor = ProcessorIndex::current().expect("Linux supports processor queries");
        let mut binding =
            ProcessorBinding::bind(processor).expect("current processor must be bindable");
        binding.restore().expect("saved affinity must restore");
        assert_eq!(
            linux::thread_affinity().expect("restored affinity must be queryable"),
            before
        );
        binding
            .restore()
            .expect("restoring an inactive guard must be a no-op");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "sched_* are foreign scheduler calls miri cannot execute"
    )]
    fn unavailable_processor_rejects_before_affinity_mutation() {
        let before = linux::thread_affinity().expect("current affinity must be queryable");
        let processor = ProcessorIndex::new(u32::MAX);
        assert!(matches!(
            ProcessorBinding::bind(processor),
            Err(ProcessorBindingError::ProcessorUnavailable { processor: rejected })
                if rejected == processor
        ));
        assert_eq!(
            linux::thread_affinity().expect("unchanged affinity must be queryable"),
            before
        );
    }
}

#[cfg(all(test, target_os = "windows"))]
mod processor_tests {
    use super::{windows, ProcessorBinding, ProcessorBindingError, ProcessorIndex};

    #[test]
    fn exact_binding_reports_processor_and_restores_on_drop() {
        let before = windows::thread_affinity().expect("current affinity must be queryable");
        let processor = ProcessorIndex::current().expect("Windows supports processor queries");

        {
            let binding =
                ProcessorBinding::bind(processor).expect("current processor must be bindable");
            assert_eq!(binding.processor(), processor);
            std::thread::yield_now();
            assert_eq!(ProcessorIndex::current(), Ok(processor));

            let exact = windows::thread_affinity().expect("bound affinity must be queryable");
            let expected = themis::ProcessorGroupAffinity::from_processor(processor.get())
                .expect("a live Windows processor has a native affinity mask");
            assert_eq!(exact.group, expected.group());
            assert_eq!(exact.mask, expected.mask());
        }

        assert_eq!(
            windows::thread_affinity().expect("restored affinity must be queryable"),
            before
        );
    }

    #[test]
    fn explicit_restore_is_observable_and_idempotent() {
        let before = windows::thread_affinity().expect("current affinity must be queryable");
        let processor = ProcessorIndex::current().expect("Windows supports processor queries");
        let mut binding =
            ProcessorBinding::bind(processor).expect("current processor must be bindable");

        binding.restore().expect("saved affinity must restore");
        assert_eq!(
            windows::thread_affinity().expect("restored affinity must be queryable"),
            before
        );
        binding
            .restore()
            .expect("restoring an inactive guard must be a no-op");
    }

    #[test]
    fn unavailable_processor_rejects_before_affinity_mutation() {
        let before = windows::thread_affinity().expect("current affinity must be queryable");
        let processor = ProcessorIndex::new(u32::MAX);

        assert!(matches!(
            ProcessorBinding::bind(processor),
            Err(ProcessorBindingError::ProcessorUnavailable { processor: rejected })
                if rejected == processor
        ));
        assert_eq!(
            windows::thread_affinity().expect("unchanged affinity must be queryable"),
            before
        );
    }
}

#[cfg(all(test, not(any(target_os = "windows", target_os = "linux"))))]
mod unsupported_processor_tests {
    use super::{ProcessorBinding, ProcessorBindingError, ProcessorIndex};

    #[test]
    fn exact_binding_is_explicitly_unsupported() {
        assert!(matches!(
            ProcessorIndex::current(),
            Err(ProcessorBindingError::UnsupportedPlatform)
        ));
        assert!(matches!(
            ProcessorBinding::bind(ProcessorIndex::new(0)),
            Err(ProcessorBindingError::UnsupportedPlatform)
        ));
    }
}

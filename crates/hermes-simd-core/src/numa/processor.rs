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
        #[cfg(not(target_os = "windows"))]
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
    #[cfg(target_os = "windows")]
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
        #[cfg(not(target_os = "windows"))]
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
        #[cfg(not(target_os = "windows"))]
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

#[cfg(target_os = "windows")]
mod windows {
    use super::{ProcessorBindingError, ProcessorIndex};

    const PROCESSORS_PER_GROUP: u32 = 64;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    #[repr(C)]
    pub(super) struct GroupAffinity {
        pub(super) mask: usize,
        pub(super) group: u16,
        reserved: [u16; 3],
    }

    #[repr(C)]
    struct ProcessorNumber {
        group: u16,
        number: u8,
        reserved: u8,
    }

    const _: () =
        assert!(core::mem::size_of::<GroupAffinity>() == core::mem::size_of::<usize>() + 8);
    const _: () = assert!(core::mem::size_of::<ProcessorNumber>() == 4);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetActiveProcessorCount(group_number: u16) -> u32;
        fn GetActiveProcessorGroupCount() -> u16;
        fn GetCurrentProcessorNumberEx(processor_number: *mut ProcessorNumber);
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn GetLastError() -> u32;
        fn SetThreadGroupAffinity(
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
            u32::from(processor.group) * PROCESSORS_PER_GROUP + u32::from(processor.number),
        )
    }

    pub(super) fn bind(processor: ProcessorIndex) -> Result<GroupAffinity, ProcessorBindingError> {
        let raw_group = processor.get() / PROCESSORS_PER_GROUP;
        let group = u16::try_from(raw_group).map_err(|_| unavailable(processor))?;
        let number = processor.get() % PROCESSORS_PER_GROUP;

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
        if number >= active_count {
            return Err(unavailable(processor));
        }
        let Some(mask) = 1usize.checked_shl(number) else {
            return Err(unavailable(processor));
        };

        let requested = GroupAffinity {
            mask,
            group,
            reserved: [0; 3],
        };
        let mut previous = GroupAffinity {
            mask: 0,
            group: 0,
            reserved: [0; 3],
        };
        // SAFETY: both affinity pointers are valid for the call; the current
        // thread pseudo-handle is valid in the calling thread, and validation
        // above proves the requested single-bit group affinity is active.
        let bound = unsafe {
            SetThreadGroupAffinity(
                GetCurrentThread(),
                core::ptr::addr_of!(requested),
                core::ptr::addr_of_mut!(previous),
            )
        };
        if bound == 0 {
            Err(platform_error("bind current thread"))
        } else {
            Ok(previous)
        }
    }

    pub(super) fn restore(previous: &GroupAffinity) -> Result<(), ProcessorBindingError> {
        // SAFETY: `previous` was initialized by a successful bind on this same
        // thread. `ProcessorBinding` is not Send, so restoration cannot move to
        // another thread.
        let restored = unsafe {
            SetThreadGroupAffinity(
                GetCurrentThread(),
                core::ptr::from_ref(previous),
                core::ptr::null_mut(),
            )
        };
        if restored == 0 {
            Err(platform_error("restore current thread"))
        } else {
            Ok(())
        }
    }

    pub(super) fn restore_on_drop(previous: &GroupAffinity) {
        // SAFETY: same invariant as `restore`. Drop cannot return a platform
        // error; callers that require that signal use `ProcessorBinding::restore`.
        unsafe {
            SetThreadGroupAffinity(
                GetCurrentThread(),
                core::ptr::from_ref(previous),
                core::ptr::null_mut(),
            );
        }
    }

    const fn unavailable(processor: ProcessorIndex) -> ProcessorBindingError {
        ProcessorBindingError::ProcessorUnavailable { processor }
    }

    fn platform_error(operation: &'static str) -> ProcessorBindingError {
        // SAFETY: captures the calling thread's last-error value immediately
        // after the failed platform call.
        let code = unsafe { GetLastError() };
        ProcessorBindingError::Platform { operation, code }
    }

    #[cfg(test)]
    pub(super) fn thread_affinity() -> Result<GroupAffinity, ProcessorBindingError> {
        let mut affinity = GroupAffinity {
            mask: 0,
            group: 0,
            reserved: [0; 3],
        };
        // SAFETY: the output pointer names writable `GROUP_AFFINITY` storage.
        let queried = unsafe {
            GetThreadGroupAffinity(GetCurrentThread(), core::ptr::addr_of_mut!(affinity))
        };
        if queried == 0 {
            Err(platform_error("query current thread affinity"))
        } else {
            Ok(affinity)
        }
    }
}

#[cfg(all(test, target_os = "windows"))]
mod processor_tests {
    use super::{windows, ProcessorBinding, ProcessorBindingError, ProcessorIndex};

    const PROCESSORS_PER_GROUP: u32 = 64;

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
            assert_eq!(
                exact.group,
                u16::try_from(processor.get() / PROCESSORS_PER_GROUP)
                    .expect("Windows processor groups fit u16")
            );
            assert_eq!(
                exact.mask,
                1usize << (processor.get() % PROCESSORS_PER_GROUP)
            );
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

#[cfg(all(test, not(target_os = "windows")))]
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

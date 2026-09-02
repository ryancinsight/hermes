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
mod windows;

#[cfg(target_os = "linux")]
mod linux;

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

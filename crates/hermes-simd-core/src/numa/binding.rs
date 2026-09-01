use crate::numa::locality::current_numa_node;

/// How completely a [`NumaBinding`] guard expresses its node's processor set.
///
/// Node membership is a topology fact; confining a thread to it is a platform
/// mechanism, and the two do not always have the same reach. Windows binds one
/// processor group per call, so a node whose processors span several groups
/// cannot be expressed whole. This type reports that shortfall instead of
/// letting the guard look complete.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NumaBindingCoverage {
    /// The guard changed no affinity.
    ///
    /// Either the thread already ran on the requested node, the node is absent
    /// from the topology, or the target has no NUMA affinity backend.
    Unbound,
    /// The thread is confined to exactly the processors reported for the node.
    Complete,
    /// The thread is confined to a proper subset of the node's processors.
    ///
    /// On Windows a single `SetThreadGroupAffinity` call names one processor
    /// group, so a node spanning several groups binds to the group holding the
    /// largest share of it. The thread stays on the node; it is confined more
    /// tightly than the node's full processor set.
    Partial {
        /// Processor group the thread was confined to.
        bound_group: u16,
        /// Processors of the node the thread was confined to.
        bound_processors: u32,
        /// Processors the topology reports for the node.
        node_processors: u32,
        /// Processor groups the node's processors span.
        node_groups: u32,
    },
}

/// RAII scope guard that binds the current thread to a specific NUMA node.
///
/// The node's processor set comes from the topology dependency
/// ([`themis`](https://docs.rs/themis-topology)), which is the single source of
/// truth for which processors belong to a node. Hermes owns only the mechanism
/// that applies it to the calling thread.
pub struct NumaBinding {
    #[cfg(all(target_os = "linux", feature = "libnuma"))]
    old_mask: *mut core::ffi::c_void,
    #[cfg(all(target_os = "windows", feature = "std"))]
    previous: Option<windows::GroupAffinity>,
    coverage: NumaBindingCoverage,
}

impl NumaBinding {
    /// Bind the current thread to the specified NUMA node.
    ///
    /// Binding is best-effort: a node the topology does not report, or a
    /// platform without a NUMA affinity backend, leaves affinity untouched.
    /// Inspect [`NumaBinding::coverage`] to distinguish a complete binding from
    /// a partial one or from no binding at all — the guard never reports
    /// success it did not achieve.
    #[must_use]
    pub fn bind(node: u32) -> Self {
        if current_numa_node() == Some(node) {
            return Self::unbound();
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
                if mask.is_null() {
                    return Self {
                        old_mask: old,
                        coverage: NumaBindingCoverage::Unbound,
                    };
                }
                numa_bitmask_setbit(mask, node);
                numa_bind(mask);
                numa_bitmask_free(mask);
                Self {
                    old_mask: old,
                    // `numa_bind` takes a node mask, so the whole node binds in
                    // one call; libnuma has no per-group shortfall.
                    coverage: NumaBindingCoverage::Complete,
                }
            }
        }
        #[cfg(all(target_os = "windows", feature = "std"))]
        {
            match windows::bind_node(node) {
                Some((previous, coverage)) => Self {
                    previous: Some(previous),
                    coverage,
                },
                None => Self::unbound(),
            }
        }
        #[cfg(not(any(
            all(target_os = "linux", feature = "libnuma"),
            all(target_os = "windows", feature = "std")
        )))]
        {
            let _ = node;
            Self::unbound()
        }
    }

    /// Report how completely this guard expresses its node's processor set.
    #[must_use]
    pub const fn coverage(&self) -> NumaBindingCoverage {
        self.coverage
    }

    /// A guard that changed no affinity and restores nothing on drop.
    const fn unbound() -> Self {
        Self {
            #[cfg(all(target_os = "linux", feature = "libnuma"))]
            old_mask: core::ptr::null_mut(),
            #[cfg(all(target_os = "windows", feature = "std"))]
            previous: None,
            coverage: NumaBindingCoverage::Unbound,
        }
    }
}

#[cfg(any(
    all(target_os = "linux", feature = "libnuma"),
    all(target_os = "windows", feature = "std")
))]
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
        #[cfg(all(target_os = "windows", feature = "std"))]
        {
            if let Some(previous) = self.previous {
                windows::restore_on_drop(&previous);
            }
        }
    }
}

#[cfg(all(target_os = "windows", feature = "std"))]
mod windows {
    use super::NumaBindingCoverage;
    use crate::numa::affinity::active_mask;
    use themis::{ProcessorAffinityGroups, ProcessorGroupAffinity};

    pub(super) use crate::numa::affinity::{restore_on_drop, GroupAffinity};

    /// Choose the processor group holding the largest share of `processors`.
    ///
    /// `processors` are the topology's group-flattened identifiers
    /// (`group * 64 + number`). One `SetThreadGroupAffinity` call names exactly
    /// one group, so a node spanning several groups cannot be expressed whole;
    /// the largest share is the closest expressible confinement. Ties resolve to
    /// the lowest group number, so the choice is deterministic rather than
    /// dependent on the order the topology happened to list processors in.
    ///
    /// Returns `None` for an empty processor list or a processor identity that
    /// the native Windows group-mask representation cannot express.
    pub(super) fn select_group(processors: &[u32]) -> Option<(ProcessorGroupAffinity, u32)> {
        let affinities = ProcessorAffinityGroups::from_processors(processors.iter().copied());
        if !affinities.is_complete() {
            return None;
        }
        let groups = u32::try_from(affinities.groups().len()).ok()?;
        Some((affinities.largest_group()?, groups))
    }

    /// Confine the calling thread to the processors reported for `node`.
    ///
    /// Returns the replaced affinity and the coverage achieved, or `None` when
    /// the node is absent from the topology or the platform rejects the bind.
    pub(super) fn bind_node(node: u32) -> Option<(GroupAffinity, NumaBindingCoverage)> {
        let topology = themis::CpuTopology::detect()?;
        let numa_node = topology
            .numa_nodes()
            .iter()
            .find(|candidate| candidate.id.get() == node)?;
        let node_processors = u32::try_from(numa_node.processors.len()).ok()?;
        let (selection, node_groups) = select_group(&numa_node.processors)?;

        let group = selection.group();
        let mask = selection.mask() & active_mask(group)?;
        if mask == 0 {
            return None;
        }

        let previous = crate::numa::affinity::bind(&GroupAffinity::new(group, mask))?;

        let bound_processors = mask.count_ones();
        let coverage = if node_groups == 1 && bound_processors == node_processors {
            NumaBindingCoverage::Complete
        } else {
            NumaBindingCoverage::Partial {
                bound_group: group,
                bound_processors,
                node_processors,
                node_groups,
            }
        };
        Some((previous, coverage))
    }
}

#[cfg(all(test, target_os = "windows", feature = "std"))]
mod binding_tests {
    use super::{windows, NumaBinding, NumaBindingCoverage};
    use crate::numa::affinity::{active_mask, thread_affinity};

    /// The defect this module exists to prevent, in its pure form.
    ///
    /// `GetNumaNodeProcessorMask` returns one group-0 mask, so a node whose
    /// processors live in group 1 came back either empty or wrong. The topology
    /// flattens as `group * 64 + number`, so ids 64..=66 are group 1 — a group
    /// the legacy query had no way to name.
    #[test]
    fn multi_group_node_selects_the_group_the_legacy_query_could_not_name() {
        let processors = [0u32, 1, 64, 65, 66];

        let (selection, groups) =
            windows::select_group(&processors).expect("a non-empty node selects");

        assert_eq!(selection.group(), 1);
        assert_eq!(selection.mask(), 0b111);
        assert_eq!(groups, 2);
    }

    #[test]
    fn single_group_node_selects_every_processor() {
        let (selection, groups) =
            windows::select_group(&[0u32, 1, 2, 3]).expect("a non-empty node selects");

        assert_eq!(selection.group(), 0);
        assert_eq!(selection.mask(), 0b1111);
        assert_eq!(groups, 1);
    }

    #[test]
    fn group_selection_is_deterministic_under_ties() {
        // Equal populations in groups 2 and 5; the lowest group wins, whichever
        // order the topology listed them in.
        let ascending = windows::select_group(&[128u32, 129, 320, 321]).expect("selects");
        let descending = windows::select_group(&[321u32, 320, 129, 128]).expect("selects");

        assert_eq!(ascending, descending);
        assert_eq!(ascending.0.group(), 2);
        assert_eq!(ascending.0.mask(), 0b11);
        assert_eq!(ascending.1, 2);
    }

    #[test]
    fn empty_node_selects_nothing() {
        assert_eq!(windows::select_group(&[]), None);
    }

    #[test]
    fn unrepresentable_node_processor_rejects_the_whole_selection() {
        assert_eq!(windows::select_group(&[0, u32::MAX]), None);
    }

    /// End-to-end against the live host: the mask the guard applies is the one
    /// the topology reports, and the previous affinity is restored.
    #[test]
    fn node_binding_matches_the_topology_table_and_restores() {
        let topology = themis::CpuTopology::detect().expect("Windows reports a CPU topology");
        let current = themis::try_current_numa_node().expect("the current node is queryable");
        let numa_node = topology
            .numa_nodes()
            .iter()
            .find(|candidate| candidate.id == current)
            .expect("the current node appears in the topology table");
        let (selection, node_groups) =
            windows::select_group(&numa_node.processors).expect("a live node reports processors");
        let group = selection.group();
        let node_processors =
            u32::try_from(numa_node.processors.len()).expect("a node's processor count fits u32");
        // Derived from the topology table and the live group inventory, both
        // read here independently of the guard.
        let expected_mask =
            selection.mask() & active_mask(group).expect("the node's group is active");
        assert_ne!(expected_mask, 0);

        let before = thread_affinity().expect("current affinity is queryable");
        let (previous, coverage) =
            windows::bind_node(current.get()).expect("the current node must be bindable");
        let bound = thread_affinity().expect("bound affinity is queryable");

        assert_eq!(bound.group, group);
        assert_eq!(
            bound.mask, expected_mask,
            "the applied mask must be exactly the node's processors in this group"
        );
        let expected_coverage = if node_groups == 1 && expected_mask.count_ones() == node_processors
        {
            NumaBindingCoverage::Complete
        } else {
            NumaBindingCoverage::Partial {
                bound_group: group,
                bound_processors: expected_mask.count_ones(),
                node_processors,
                node_groups,
            }
        };
        assert_eq!(coverage, expected_coverage);

        windows::restore_on_drop(&previous);
        assert_eq!(
            thread_affinity().expect("restored affinity is queryable"),
            before
        );
    }

    #[test]
    fn binding_the_current_node_changes_no_affinity() {
        let current = themis::try_current_numa_node().expect("the current node is queryable");
        let before = thread_affinity().expect("current affinity is queryable");

        let binding = NumaBinding::bind(current.get());

        assert_eq!(binding.coverage(), NumaBindingCoverage::Unbound);
        assert_eq!(
            thread_affinity().expect("affinity is queryable"),
            before,
            "an already-local thread must not be re-confined"
        );
    }

    #[test]
    fn absent_node_leaves_affinity_untouched() {
        let before = thread_affinity().expect("current affinity is queryable");

        let binding = NumaBinding::bind(u32::MAX);

        assert_eq!(binding.coverage(), NumaBindingCoverage::Unbound);
        assert_eq!(
            thread_affinity().expect("affinity is queryable"),
            before,
            "a node the topology does not report must not change affinity"
        );
    }
}

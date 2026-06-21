#[cfg(feature = "std")]
fn topology() -> Option<&'static themis::CpuTopology> {
    static TOPOLOGY: std::sync::OnceLock<Option<themis::CpuTopology>> = std::sync::OnceLock::new();
    TOPOLOGY.get_or_init(themis::CpuTopology::detect).as_ref()
}

/// Returns the total number of NUMA nodes configured in the system.
pub fn numa_node_count() -> u32 {
    NumaTopologyService::total_nodes()
}

/// Retrieve the NUMA distance between two NUMA nodes.
pub fn numa_node_distance(node_a: u32, node_b: u32) -> u32 {
    #[cfg(feature = "std")]
    {
        use themis::NumaNodeId;
        if let Some(t) = topology() {
            return t.distance(NumaNodeId::new(node_a), NumaNodeId::new(node_b));
        }
    }
    if node_a == node_b {
        10
    } else {
        20
    }
}

/// Topology service to query NUMA nodes and logical processors.
pub struct NumaTopologyService;

impl NumaTopologyService {
    /// Query the current CPU/processor index.
    pub fn current_cpu() -> Option<u32> {
        themis::current_processor()
    }

    /// Query the current NUMA node ID.
    pub fn current_node() -> Option<u32> {
        crate::numa::locality::current_numa_node()
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

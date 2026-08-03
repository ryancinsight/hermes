/// 64-byte AMX tile configuration structure (TILECFG).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmxConfig {
    /// Palette ID (0 = init, 1 = Palette 1)
    pub palette_id: u8,
    /// Start row for recovery after interrupt / context switch
    pub start_row: u8,
    /// Reserved bytes
    pub reserved: [u8; 14],
    /// Column widths in bytes for each of the 8 tiles
    pub cols_b: [u16; 8],
    /// Reserved bytes
    pub reserved2: [u8; 16],
    /// Number of rows for each of the 8 tiles
    pub rows: [u8; 8],
    /// Reserved bytes
    pub reserved3: [u8; 8],
}

impl AmxConfig {
    /// Creates a palette 1 configuration where all tiles have the specified rows and byte columns.
    #[inline]
    pub fn new_uniform(r: u8, c_bytes: u16) -> Self {
        Self {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 14],
            cols_b: [c_bytes; 8],
            reserved2: [0; 16],
            rows: [r; 8],
            reserved3: [0; 8],
        }
    }

    /// Creates a custom palette 1 configuration with row/col sizes for each of the 8 tiles.
    #[inline]
    pub fn new_custom(rows: [u8; 8], cols_b: [u16; 8]) -> Self {
        Self {
            palette_id: 1,
            start_row: 0,
            reserved: [0; 14],
            cols_b,
            reserved2: [0; 16],
            rows,
            reserved3: [0; 8],
        }
    }

    /// Generate adaptive tile config based on dynamic matrix dimensions.
    #[inline]
    pub fn for_dimensions(m: usize, n: usize, k: usize, element_size: usize) -> Self {
        let r_a = m.min(16) as u8;
        let c_a_bytes = (k * element_size).min(64) as u16;
        let r_b = k.min(64 / element_size) as u8;
        let c_b_bytes = (n * element_size).min(64) as u16;
        let r_c = m.min(16) as u8;
        let c_c_bytes = (n * 4).min(64) as u16; // Accumulation in 32-bit (F32/I32)

        let mut rows = [0; 8];
        let mut cols_b = [0; 8];

        // Tile 0 (A): M rows x K cols
        rows[0] = r_a;
        cols_b[0] = c_a_bytes;
        // Tile 1 (B): K rows x N cols
        rows[1] = r_b;
        cols_b[1] = c_b_bytes;
        // Tile 2 (C): M rows x N cols
        rows[2] = r_c;
        cols_b[2] = c_c_bytes;

        // Auxiliary registers for register blocking
        rows[3] = r_c;
        cols_b[3] = c_c_bytes;
        rows[4] = r_c;
        cols_b[4] = c_c_bytes;
        rows[5] = r_c;
        cols_b[5] = c_c_bytes;
        rows[6] = r_b;
        cols_b[6] = c_b_bytes;
        rows[7] = r_b;
        cols_b[7] = c_b_bytes;

        Self::new_custom(rows, cols_b)
    }
}

/// Unaligned thread-local copy of the active tile configuration.
///
/// `AmxConfig` itself requires 64-byte alignment for `ldtilecfg`. Windows'
/// native thread-local storage does not guarantee that over-alignment, so the
/// session state stores the same bytes in a naturally aligned representation
/// and keeps the aligned type only at the instruction boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveAmxConfig {
    palette_id: u8,
    start_row: u8,
    reserved: [u8; 14],
    cols_b: [u16; 8],
    reserved2: [u8; 16],
    rows: [u8; 8],
    reserved3: [u8; 8],
}

const _: () = assert!(core::mem::size_of::<ActiveAmxConfig>() == 64);
const _: () = assert!(core::mem::align_of::<ActiveAmxConfig>() <= 2);

impl From<&AmxConfig> for ActiveAmxConfig {
    fn from(config: &AmxConfig) -> Self {
        Self {
            palette_id: config.palette_id,
            start_row: config.start_row,
            reserved: config.reserved,
            cols_b: config.cols_b,
            reserved2: config.reserved2,
            rows: config.rows,
            reserved3: config.reserved3,
        }
    }
}

#[cfg(feature = "std")]
thread_local! {
    pub(crate) static ACTIVE_CONFIG: core::cell::Cell<Option<ActiveAmxConfig>> = const { core::cell::Cell::new(None) };
    pub(crate) static SESSION_DEPTH: core::cell::Cell<usize> = core::cell::Cell::new(0);
}

#[cfg(not(feature = "std"))]
pub(crate) struct DummyThreadLocal<T> {
    cell: core::cell::Cell<T>,
}

#[cfg(not(feature = "std"))]
impl<T> DummyThreadLocal<T> {
    const fn new(val: T) -> Self {
        Self {
            cell: core::cell::Cell::new(val),
        }
    }

    #[inline(always)]
    fn with<R, F: FnOnce(&core::cell::Cell<T>) -> R>(&self, f: F) -> R {
        f(&self.cell)
    }
}

#[cfg(not(feature = "std"))]
unsafe impl<T> Sync for DummyThreadLocal<T> {}

#[cfg(not(feature = "std"))]
pub(crate) static ACTIVE_CONFIG: DummyThreadLocal<Option<ActiveAmxConfig>> =
    DummyThreadLocal::new(None);

#[cfg(not(feature = "std"))]
pub(crate) static SESSION_DEPTH: DummyThreadLocal<usize> = DummyThreadLocal::new(0);

//! Fancy Magic Bitboards sliding attack generation with lazy table initialization.

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicUsize, Ordering};

/// Minimal `no_std`-compatible once-initialization cell used for the lazily
/// built magic attack tables (spin-based; initialization races run `f` once).
pub struct OnceLock<T> {
    state: AtomicUsize, // 0 = uninitialized, 1 = initializing, 2 = initialized
    value: UnsafeCell<Option<T>>,
}

unsafe impl<T: Send + Sync> Sync for OnceLock<T> {}
unsafe impl<T: Send> Send for OnceLock<T> {}

impl<T> OnceLock<T> {
    /// Create an empty, uninitialized cell.
    pub const fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
            value: UnsafeCell::new(None),
        }
    }

    /// Return the stored value, running `f` exactly once to initialize it;
    /// concurrent callers spin until initialization completes.
    pub fn get_or_init<F>(&self, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        if self.state.load(Ordering::Acquire) == 2 {
            return unsafe {
                (*self.value.get())
                    .as_ref()
                    .expect("OnceLock: state==2 implies value is initialized")
            };
        }

        loop {
            let current = self.state.load(Ordering::Acquire);
            if current == 2 {
                break;
            }
            if current == 0 {
                if self
                    .state
                    // Winner acquires no shared data on the 0->1 claim (it next
                    // *writes* the value and publishes it via `store(2, Release)`),
                    // so `Relaxed` success ordering is sufficient.
                    .compare_exchange_weak(0, 1, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    unsafe {
                        *self.value.get() = Some(f());
                    }
                    self.state.store(2, Ordering::Release);
                    break;
                }
            } else {
                core::hint::spin_loop();
            }
        }
        unsafe {
            (*self.value.get())
                .as_ref()
                .expect("OnceLock: state==2 implies value is initialized")
        }
    }
}

use hermes_simd_core::bitboard::BitBoardKernel;

/// ZST marker for Fancy Magic Bitboards backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Magic;
use super::swar::attack_ray;

// Helper functions for directional shifting
#[inline(always)]
fn shift_n(v: u64) -> u64 {
    v << 8
}
#[inline(always)]
fn shift_s(v: u64) -> u64 {
    v >> 8
}
#[inline(always)]
fn shift_e(v: u64) -> u64 {
    (v << 1) & 0xFEFEFEFEFEFEFEFE
}
#[inline(always)]
fn shift_w(v: u64) -> u64 {
    (v >> 1) & 0x7F7F7F7F7F7F7F7F
}
#[inline(always)]
fn shift_ne(v: u64) -> u64 {
    (v << 9) & 0xFEFEFEFEFEFEFEFE
}
#[inline(always)]
fn shift_nw(v: u64) -> u64 {
    (v << 7) & 0x7F7F7F7F7F7F7F7F
}
#[inline(always)]
fn shift_se(v: u64) -> u64 {
    (v >> 7) & 0xFEFEFEFEFEFEFEFE
}
#[inline(always)]
fn shift_sw(v: u64) -> u64 {
    (v >> 9) & 0x7F7F7F7F7F7F7F7F
}

/// Construct Rook occupancy mask (excluding edges).
#[inline]
pub fn rook_mask(sq: u8) -> u64 {
    let mut mask = 0u64;
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    for i in 1..7 {
        if i != r {
            mask |= 1u64 << (i * 8 + f);
        }
        if i != f {
            mask |= 1u64 << (r * 8 + i);
        }
    }
    mask
}

/// Construct Bishop occupancy mask (excluding edges).
#[inline]
pub fn bishop_mask(sq: u8) -> u64 {
    let mut mask = 0u64;
    let r = (sq / 8) as i32;
    let f = (sq % 8) as i32;
    for i in 1..7 {
        let nr = r + i;
        let nf = f + i;
        if nr < 7 && nf < 7 {
            mask |= 1u64 << (nr * 8 + nf);
        }
        let nr = r + i;
        let nf = f - i;
        if nr < 7 && nf > 0 {
            mask |= 1u64 << (nr * 8 + nf);
        }
        let nr = r - i;
        let nf = f + i;
        if nr > 0 && nf < 7 {
            mask |= 1u64 << (nr * 8 + nf);
        }
        let nr = r - i;
        let nf = f - i;
        if nr > 0 && nf > 0 {
            mask |= 1u64 << (nr * 8 + nf);
        }
    }
    mask
}

/// Generate occupancy pattern for a mask and index.
fn get_occupancy(index: usize, mask: u64) -> u64 {
    let mut occupancy = 0u64;
    let mut m = mask;
    let mut i = index;
    while m != 0 {
        let lsb = m & m.wrapping_neg();
        m ^= lsb;
        if (i & 1) != 0 {
            occupancy |= lsb;
        }
        i >>= 1;
    }
    occupancy
}

const ROOK_MAGICS: [u64; 64] = [
    612507691268440096,
    612507278951579776,
    72092778695245832,
    13907124514132861184,
    36037595259731970,
    4683752412853567490,
    144116304901620224,
    72064193267807488,
    422213022908800,
    2324420636851601920,
    5066833117843584,
    18437160864616448,
    1191905823245468288,
    73746496243761160,
    217298750773725188,
    1162491709658234966,
    18014948810555456,
    360571644730802304,
    4503737068946432,
    1154057300386783488,
    2252351717508096,
    2306969184032063552,
    722409925791711888,
    8070457129321923844,
    140881369776129,
    2305930988397535232,
    727366526348300288,
    2882338984544960769,
    2314854612957921664,
    1226109398745743488,
    18016614716868624,
    3940933141856388,
    70438008914056,
    1441292755694272521,
    13935298335742103616,
    650771245936148992,
    8800404767760,
    4611826764514067456,
    18304738365800706,
    2392641597604096,
    18014675537002496,
    72708505458589696,
    2306124621734543376,
    7318633399844880,
    288239172530012164,
    1125908513570880,
    423793149607938,
    72128522269097987,
    5080060654553923648,
    108368149504100864,
    9228018641916657792,
    621506644450222336,
    1155173871364606080,
    288934080773488768,
    11530377299149456384,
    6918392146485592576,
    88510740570371,
    5782639788609832065,
    144396738350809281,
    157913028229072929,
    5765170767461884162,
    562954517088258,
    2307021690107888132,
    2815026809283586,
];

const BISHOP_MAGICS: [u64; 64] = [
    2900320361214181632,
    4521240153325862,
    307415763168788480,
    577591189911371776,
    9307640807030792,
    299076021126434,
    4648803916316672,
    8368260145767256068,
    297246444856279360,
    9295447291819266081,
    4416308641792,
    81073727902580769,
    720580351445188672,
    2314992114330763778,
    5718564289912896,
    612666572850601984,
    18159602805933060,
    94575605329300608,
    2778721004624613648,
    2251853534609920,
    9259682319860499464,
    72198366053867524,
    4612249038727946256,
    2311648435029673024,
    297246406417580304,
    2253998907228672,
    1127016866841633,
    4644474689028384,
    3207833970406400000,
    142938675888640,
    2265548827396352,
    369437041106157696,
    608023367451217920,
    9224515546128126208,
    126127195025834048,
    70403640918528,
    657526662304433160,
    155391783633690752,
    18586217575221400,
    595040317506339856,
    2451251326177329220,
    2305915887359631364,
    288511954241199106,
    4611690700075958400,
    316745583693888,
    6926537335024648708,
    5480885148919529568,
    16141208930453029254,
    2883465404411281498,
    576976457634414720,
    986410176759168,
    702579169524056576,
    4693455495865565184,
    90297529081856,
    4618466725024890896,
    580965460101579777,
    2305988728998461956,
    565254270747148,
    9223372451327787272,
    72057733633020163,
    1127034055164428,
    5225583234828469520,
    904169302867347712,
    2612237321755557952,
];

// Flat lookup tables and offsets
struct MagicTable {
    rook_table: alloc::vec::Vec<u64>,
    bishop_table: alloc::vec::Vec<u64>,
    rook_offsets: [usize; 64],
    bishop_offsets: [usize; 64],
    rook_magics: [u64; 64],
    bishop_magics: [u64; 64],
}

static MAGIC_DATA: OnceLock<MagicTable> = OnceLock::new();

fn get_magic_data() -> &'static MagicTable {
    MAGIC_DATA.get_or_init(|| {
        let mut rook_offsets = [0; 64];
        let mut bishop_offsets = [0; 64];

        let mut rook_total_size = 0;
        let mut bishop_total_size = 0;

        for sq in 0..64 {
            rook_offsets[sq] = rook_total_size;
            let rook_pop = rook_mask(sq as u8).count_ones() as usize;
            rook_total_size += 1 << rook_pop;

            bishop_offsets[sq] = bishop_total_size;
            let bishop_pop = bishop_mask(sq as u8).count_ones() as usize;
            bishop_total_size += 1 << bishop_pop;
        }

        let mut rook_table = alloc::vec![0u64; rook_total_size];
        let mut bishop_table = alloc::vec![0u64; bishop_total_size];

        // Populate tables using static precomputed magics
        for sq in 0..64 {
            // Rook
            let r_mask = rook_mask(sq as u8);
            let r_pop = r_mask.count_ones() as usize;
            let r_num_patterns = 1 << r_pop;
            let r_magic = ROOK_MAGICS[sq];
            let r_shift = 64 - r_pop;
            let r_offset = rook_offsets[sq];

            for idx in 0..r_num_patterns {
                let occ = get_occupancy(idx, r_mask);
                let slider = 1u64 << sq;
                let att = attack_ray(slider, occ, shift_n)
                    | attack_ray(slider, occ, shift_s)
                    | attack_ray(slider, occ, shift_e)
                    | attack_ray(slider, occ, shift_w);
                let hash = ((occ.wrapping_mul(r_magic)) >> r_shift) as usize;
                rook_table[r_offset + hash] = att;
            }

            // Bishop
            let b_mask = bishop_mask(sq as u8);
            let b_pop = b_mask.count_ones() as usize;
            let b_num_patterns = 1 << b_pop;
            let b_magic = BISHOP_MAGICS[sq];
            let b_shift = 64 - b_pop;
            let b_offset = bishop_offsets[sq];

            for idx in 0..b_num_patterns {
                let occ = get_occupancy(idx, b_mask);
                let slider = 1u64 << sq;
                let att = attack_ray(slider, occ, shift_ne)
                    | attack_ray(slider, occ, shift_nw)
                    | attack_ray(slider, occ, shift_se)
                    | attack_ray(slider, occ, shift_sw);
                let hash = ((occ.wrapping_mul(b_magic)) >> b_shift) as usize;
                bishop_table[b_offset + hash] = att;
            }
        }

        MagicTable {
            rook_table,
            bishop_table,
            rook_offsets,
            bishop_offsets,
            rook_magics: ROOK_MAGICS,
            bishop_magics: BISHOP_MAGICS,
        }
    })
}

impl BitBoardKernel for Magic {
    #[inline]
    fn rook_attacks(square: u8, occupancy: u64) -> u64 {
        let table = get_magic_data();
        let mask = rook_mask(square);
        let pop = mask.count_ones() as usize;
        let magic = table.rook_magics[square as usize];
        let shift = 64 - pop;
        let offset = table.rook_offsets[square as usize];
        let idx = (((occupancy & mask).wrapping_mul(magic)) >> shift) as usize;
        table.rook_table[offset + idx]
    }

    #[inline]
    fn bishop_attacks(square: u8, occupancy: u64) -> u64 {
        let table = get_magic_data();
        let mask = bishop_mask(square);
        let pop = mask.count_ones() as usize;
        let magic = table.bishop_magics[square as usize];
        let shift = 64 - pop;
        let offset = table.bishop_offsets[square as usize];
        let idx = (((occupancy & mask).wrapping_mul(magic)) >> shift) as usize;
        table.bishop_table[offset + idx]
    }
}

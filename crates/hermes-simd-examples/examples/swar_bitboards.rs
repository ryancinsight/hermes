//! Example showcasing chess bitboards, sliding attack generation backends,
//! and pure SWAR (SIMD Within A Register) bitwise primitives in `hermes-simd`.
//!
//! This example demonstrates:
//! 1. Rook, Bishop, and Queen sliding attack queries on different squares and occupancies.
//! 2. Choosing between different sliding attack backends (`KoggeStone`, `Hyperbola`, `Magic`, `HybridSwarMagic`).
//! 3. Batch attack generation using the `BitBoardView`.
//! 4. Pure SWAR utility primitives in `SwarUtils`.

use hermes_simd::{
    BitBoardKernel, BitBoardView, HybridSwarMagic, Hyperbola, KoggeStone, Magic, SwarUtils,
};

fn main() {
    println!("=== Hermes SIMD SWAR Bitboards Example ===");

    // Let's set up a mock chess board state.
    // Occupancy bitboard representing blockers on the board.
    // Let's place blocker pieces at:
    // - c3 (square 18)
    // - f3 (square 21)
    // - d6 (square 43)
    // - d2 (square 10)
    let c3 = 18;
    let f3 = 21;
    let d6 = 43;
    let d2 = 10;
    let occupancy = (1u64 << c3) | (1u64 << f3) | (1u64 << d6) | (1u64 << d2);

    // Let's place a sliding rook or bishop at d3 (square 19).
    let d3 = 19;

    println!("\nOccupancy bitboard (blockers at c3, f3, d6, d2):");
    print_bitboard(occupancy);

    println!("\nSlider position (d3):");
    print_bitboard(1u64 << d3);

    // 1. Sliding attacks using the default Magic Bitboard backend
    println!("\nRook attacks from d3 using Magic Bitboards:");
    let rook_magic = <Magic as BitBoardKernel>::rook_attacks(d3, occupancy);
    print_bitboard(rook_magic);

    println!("\nBishop attacks from d3 using Magic Bitboards:");
    let bishop_magic = <Magic as BitBoardKernel>::bishop_attacks(d3, occupancy);
    print_bitboard(bishop_magic);

    // 2. Comparing different backends
    println!("\nComparing sliding attack backends...");
    let r_swar = <HybridSwarMagic as BitBoardKernel>::rook_attacks(d3, occupancy);
    let r_ks = <KoggeStone as BitBoardKernel>::rook_attacks(d3, occupancy);
    let r_hyp = <Hyperbola as BitBoardKernel>::rook_attacks(d3, occupancy);

    assert_eq!(rook_magic, r_swar);
    assert_eq!(rook_magic, r_ks);
    assert_eq!(rook_magic, r_hyp);
    println!(
        "  [OK] All Rook attack backends (Magic, KoggeStone, Hyperbola, Hybrid) matched perfectly."
    );

    let b_swar = <HybridSwarMagic as BitBoardKernel>::bishop_attacks(d3, occupancy);
    let b_ks = <KoggeStone as BitBoardKernel>::bishop_attacks(d3, occupancy);
    let b_hyp = <Hyperbola as BitBoardKernel>::bishop_attacks(d3, occupancy);

    assert_eq!(bishop_magic, b_swar);
    assert_eq!(bishop_magic, b_ks);
    assert_eq!(bishop_magic, b_hyp);
    println!("  [OK] All Bishop attack backends matched perfectly.");

    // 3. Batch attacks with BitBoardView
    println!("\nBatch attack queries using BitBoardView:");
    let squares = [d3, c3, f3, d6];
    let mut rook_attacks_out = [0u64; 4];

    // Create a view over the output array
    let data = [0u64; 4];
    let view = BitBoardView::<Magic, hermes_simd::Scalar>::new(&data);
    view.batch_attacks_single_occupancy(&squares, occupancy, &mut rook_attacks_out, true);

    for (i, sq) in squares.iter().enumerate() {
        let rank = sq / 8 + 1;
        let file = (b'a' + (sq % 8)) as char;
        println!(
            "  Rook attacks from {}{} (sq {}): popcount = {}",
            file,
            rank,
            sq,
            rook_attacks_out[i].count_ones()
        );
    }

    // 4. Pure SWAR primitives
    println!("\nDemonstrating Pure SWAR Primitives (SwarUtils):");
    let val = 0b101100u64; // bits at index 2, 3, 5 (decimal 44)
    println!("  Input value: {:#b} (decimal {})", val, val);

    let lsb = SwarUtils::isolate_lsb(val);
    println!("  Isolate LSB: {:#b} (decimal {})", lsb, lsb);
    assert_eq!(lsb, 4);

    let cleared_lsb = SwarUtils::clear_lsb(val);
    println!(
        "  Clear LSB:   {:#b} (decimal {})",
        cleared_lsb, cleared_lsb
    );
    assert_eq!(cleared_lsb, 40);

    let msb = SwarUtils::isolate_msb(val);
    println!("  Isolate MSB: {:#b} (decimal {})", msb, msb);
    assert_eq!(msb, 32);

    let pop = SwarUtils::popcount(val);
    println!("  Popcount:    {}", pop);
    assert_eq!(pop, 3);

    let scan_f = SwarUtils::bit_scan_forward(val);
    println!("  Bit scan forward:  index {}", scan_f);
    assert_eq!(scan_f, 2);

    let scan_r = SwarUtils::bit_scan_reverse(val);
    println!("  Bit scan reverse:  index {}", scan_r);
    assert_eq!(scan_r, 5);

    // Popcount-8 (parallel byte-wise popcount)
    let bytes_val = 0x01_03_07_0f_00_ff_55_aa_u64;
    let byte_pops = SwarUtils::popcount_8(bytes_val);
    println!("  Byte-wise popcount of {:#018x}:", bytes_val);
    for b in 0..8 {
        let original_byte = (bytes_val >> (b * 8)) & 0xff;
        let pop_count = (byte_pops >> (b * 8)) & 0xff;
        println!(
            "    Byte {}: {:#010b} has popcount {}",
            b, original_byte, pop_count
        );
    }
}

/// Helper function to print chess bitboard to stdout.
fn print_bitboard(b: u64) {
    for rank in (0..8).rev() {
        print!("  {} |", rank + 1);
        for file in 0..8 {
            let square = rank * 8 + file;
            if (b & (1u64 << square)) != 0 {
                print!(" X");
            } else {
                print!(" .");
            }
        }
        println!();
    }
    println!("    -----------------");
    println!("      a b c d e f g h");
}

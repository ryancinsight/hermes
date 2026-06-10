use hermes_simd::*;

#[test]
fn test_bitboard_backends_match() {
    let occ = 0x0000_1234_5678_abcd;
    for sq in 0..64 {
        unsafe {
            let r_swar = <Swar as BitBoardKernel>::rook_attacks(sq, occ);
            let r_ks = <KoggeStone as BitBoardKernel>::rook_attacks(sq, occ);
            let r_hyp = <Hyperbola as BitBoardKernel>::rook_attacks(sq, occ);
            let r_magic = <Magic as BitBoardKernel>::rook_attacks(sq, occ);
            assert_eq!(r_swar, r_ks, "Rook swar vs ks mismatch at sq {}", sq);
            assert_eq!(r_swar, r_hyp, "Rook swar vs hyp mismatch at sq {}", sq);
            assert_eq!(r_swar, r_magic, "Rook swar vs magic mismatch at sq {}", sq);

            let b_swar = <Swar as BitBoardKernel>::bishop_attacks(sq, occ);
            let b_ks = <KoggeStone as BitBoardKernel>::bishop_attacks(sq, occ);
            let b_hyp = <Hyperbola as BitBoardKernel>::bishop_attacks(sq, occ);
            let b_magic = <Magic as BitBoardKernel>::bishop_attacks(sq, occ);
            assert_eq!(b_swar, b_ks, "Bishop swar vs ks mismatch at sq {}", sq);
            assert_eq!(b_swar, b_hyp, "Bishop swar vs hyp mismatch at sq {}", sq);
            assert_eq!(
                b_swar, b_magic,
                "Bishop swar vs magic mismatch at sq {}",
                sq
            );

            let q_swar = <Swar as BitBoardKernel>::queen_attacks(sq, occ);
            let q_ks = <KoggeStone as BitBoardKernel>::queen_attacks(sq, occ);
            let q_hyp = <Hyperbola as BitBoardKernel>::queen_attacks(sq, occ);
            let q_magic = <Magic as BitBoardKernel>::queen_attacks(sq, occ);
            assert_eq!(q_swar, q_ks, "Queen swar vs ks mismatch at sq {}", sq);
            assert_eq!(q_swar, q_hyp, "Queen swar vs hyp mismatch at sq {}", sq);
            assert_eq!(q_swar, q_magic, "Queen swar vs magic mismatch at sq {}", sq);
        }
    }
}

#[test]
fn test_hybrid_swar_magic_matching() {
    let test_occupancies = [
        0,
        1u64 << 12,
        (1u64 << 12) | (1u64 << 34),
        0x0000_1234_5678_abcd,
    ];
    for occ in test_occupancies {
        for sq in 0..64 {
            unsafe {
                let r_swar = <Swar as BitBoardKernel>::rook_attacks(sq, occ);
                let r_magic = <Magic as BitBoardKernel>::rook_attacks(sq, occ);
                let r_hybrid = <HybridSwarMagic as BitBoardKernel>::rook_attacks(sq, occ);
                assert_eq!(
                    r_hybrid, r_swar,
                    "Rook hybrid vs swar mismatch at sq {}, occ {}",
                    sq, occ
                );
                assert_eq!(
                    r_hybrid, r_magic,
                    "Rook hybrid vs magic mismatch at sq {}, occ {}",
                    sq, occ
                );

                let b_swar = <Swar as BitBoardKernel>::bishop_attacks(sq, occ);
                let b_magic = <Magic as BitBoardKernel>::bishop_attacks(sq, occ);
                let b_hybrid = <HybridSwarMagic as BitBoardKernel>::bishop_attacks(sq, occ);
                assert_eq!(
                    b_hybrid, b_swar,
                    "Bishop hybrid vs swar mismatch at sq {}, occ {}",
                    sq, occ
                );
                assert_eq!(
                    b_hybrid, b_magic,
                    "Bishop hybrid vs magic mismatch at sq {}, occ {}",
                    sq, occ
                );

                let q_swar = <Swar as BitBoardKernel>::queen_attacks(sq, occ);
                let q_magic = <Magic as BitBoardKernel>::queen_attacks(sq, occ);
                let q_hybrid = <HybridSwarMagic as BitBoardKernel>::queen_attacks(sq, occ);
                assert_eq!(
                    q_hybrid, q_swar,
                    "Queen hybrid vs swar mismatch at sq {}, occ {}",
                    sq, occ
                );
                assert_eq!(
                    q_hybrid, q_magic,
                    "Queen hybrid vs magic mismatch at sq {}, occ {}",
                    sq, occ
                );
            }
        }
    }
}

#[test]
fn test_swar_utils_primitives() {
    let val = 44u64; // 0b101100
    assert_eq!(SwarUtils::isolate_lsb(val), 4);
    assert_eq!(SwarUtils::clear_lsb(val), 40);
    assert_eq!(SwarUtils::isolate_msb(val), 32);
    assert_eq!(SwarUtils::popcount(val), 3);
    assert_eq!(SwarUtils::bit_scan_forward(val), 2);
    assert_eq!(SwarUtils::bit_scan_reverse(val), 5);

    let multi_byte = 0x01_03_07_0f_00_ff_55_aa_u64;
    let p8 = SwarUtils::popcount_8(multi_byte);
    assert_eq!(p8 & 0xff, 0xaa_u64.count_ones() as u64);
    assert_eq!((p8 >> 8) & 0xff, 0x55_u64.count_ones() as u64);
    assert_eq!((p8 >> 16) & 0xff, 0xff_u64.count_ones() as u64);
    assert_eq!((p8 >> 24) & 0xff, 0x00_u64.count_ones() as u64);
}

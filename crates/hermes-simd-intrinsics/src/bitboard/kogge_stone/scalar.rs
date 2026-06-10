use super::{step_e, step_n, step_ne, step_nw, step_s, step_se, step_sw, step_w};

/// Computes Rook attacks using Kogge-Stone.
#[inline]
pub fn kogge_stone_rook(slider: u64, occupancy: u64) -> u64 {
    let p = !occupancy;

    // North
    let (mut gn, mut pn) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_n(gn, pn, s);
        gn = g;
        pn = pr;
    }
    let attacks_n = gn << 8;

    // South
    let (mut gs, mut ps) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_s(gs, ps, s);
        gs = g;
        ps = pr;
    }
    let attacks_s = gs >> 8;

    // East
    let (mut ge, mut pe) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_e(ge, pe, s);
        ge = g;
        pe = pr;
    }
    let attacks_e = (ge << 1) & 0xFEFEFEFEFEFEFEFE;

    // West
    let (mut gw, mut pw) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_w(gw, pw, s);
        gw = g;
        pw = pr;
    }
    let attacks_w = (gw >> 1) & 0x7F7F7F7F7F7F7F7F;

    attacks_n | attacks_s | attacks_e | attacks_w
}

/// Computes Bishop attacks using Kogge-Stone.
#[inline]
pub fn kogge_stone_bishop(slider: u64, occupancy: u64) -> u64 {
    let p = !occupancy;

    // North-East
    let (mut gne, mut pne) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_ne(gne, pne, s);
        gne = g;
        pne = pr;
    }
    let attacks_ne = (gne << 9) & 0xFEFEFEFEFEFEFEFE;

    // North-West
    let (mut gnw, mut pnw) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_nw(gnw, pnw, s);
        gnw = g;
        pnw = pr;
    }
    let attacks_nw = (gnw << 7) & 0x7F7F7F7F7F7F7F7F;

    // South-East
    let (mut gse, mut pse) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_se(gse, pse, s);
        gse = g;
        pse = pr;
    }
    let attacks_se = (gse >> 7) & 0xFEFEFEFEFEFEFEFE;

    // South-West
    let (mut gsw, mut psw) = (slider, p);
    for s in [1, 2, 4] {
        let (g, pr) = step_sw(gsw, psw, s);
        gsw = g;
        psw = pr;
    }
    let attacks_sw = (gsw >> 9) & 0x7F7F7F7F7F7F7F7F;

    attacks_ne | attacks_nw | attacks_se | attacks_sw
}

/// Computes Queen attacks using Kogge-Stone.
#[inline]
pub fn kogge_stone_queen(slider: u64, occupancy: u64) -> u64 {
    kogge_stone_rook(slider, occupancy) | kogge_stone_bishop(slider, occupancy)
}

//! Slot masks and polynomial index lists for the prime-field Keccak sum-check instances.

use agg_fpkeccak::{lane, lane_xy, pi_inv, Absorbed, KeccakLayout, COLS, LANES, LANE_BITS};

use crate::manifest::Column;

/// Which sum-check stage terminal point a Keccak reduction targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeccakPoint {
    /// The `d_K`-suffix of the stage-1 terminal point `r^(1)`.
    R1,
    /// The `d_K`-suffix of the stage-2 terminal point `r^(2)`.
    R2,
}

/// Live-slot bitmask (`t = 16 q + s < 24 * perms`) of slot group `q`.
#[inline]
#[must_use]
pub fn live_mask(layout: &KeccakLayout, q: usize) -> u16 {
    layout.live(0, q)
}

/// Inner-slot bitmask (`t < 24 * perms` and not a boundary block absorption `t = 24 m`, `m >= 1`)
/// of slot group `q`.
#[inline]
#[must_use]
pub fn inner_mask(layout: &KeccakLayout, q: usize) -> u16 {
    layout.inner(0, q)
}

/// Inner-slot bitmask excluding `t = 0` (`s = 0` of `q = 0`): all inner rounds `t >= 1` where
/// `A_in = iota(chi(Prev))`.
#[inline]
#[must_use]
pub fn inner_after_t0_mask(layout: &KeccakLayout, q: usize) -> u16 {
    let m = layout.inner(0, q);
    if q == 0 {
        m & !1_u16
    } else {
        m
    }
}

/// Slot bitmask of `t = 0` (`s = 0` of `q = 0`).
pub const T0_MASK: u16 = 1;

/// Boundary-slot bitmask (`t = 24 m`, `1 <= m < perms`) of slot group `q`.
#[inline]
#[must_use]
pub fn boundary_mask(layout: &KeccakLayout, q: usize) -> u16 {
    layout.boundary(0, q)
}

/// Output-slot bitmask (`t = 24 * perms - 1`, in `q = Q - 1`).
#[inline]
#[must_use]
pub fn output_mask(layout: &KeccakLayout) -> u16 {
    layout.output(0, layout.slot_groups - 1)
}

/// The `perms - 1` slot groups `q` that contain a boundary block-absorption round `t = 24 m`
/// (`1 <= m < perms`), in increasing `q` order (`[1, 3, 4, 6, 7, 9]` for `perms = 7`, and
/// `[1, 3, 4, 6, 7, 9, 10, 12]` for `perms = 9`).
#[must_use]
pub fn boundary_slot_groups(layout: &KeccakLayout) -> Vec<u8> {
    (0..layout.slot_groups)
        .filter(|&q| boundary_mask(layout, q) != 0)
        .map(|q| q as u8)
        .collect()
}

/// The `A'` columns in the final slot group `q = Q - 1` whose `KeccakRot` virtual lanes are read
/// by `k_out[0..L]`, in canonical lane order `lane(x, y) = x + 5 * y`.
///
/// For `(x, y) == (0, 0)` (where `rot(0, 0) == 0`), `KeccakRot` holds `A'[(0, 0), Q - 1] xor RC_23`
/// at the output slot `s_out` for `k_out[0]`, while `k_out` consumers of un-XORed `B[0, 0]` read
/// `Column::KeccakA { x: 0, y: 0, q: Q - 1 }` directly.
#[must_use]
pub fn output_rot_columns(layout: &KeccakLayout) -> Vec<Column> {
    let q_last = (layout.slot_groups - 1) as u8;
    let mut seen = [false; LANES];
    for out_lane in 0..layout.output_lanes {
        let (x, y) = lane_xy(out_lane);
        for d in 0..3 {
            let (src_x, src_y) = pi_inv((x + d) % COLS, y);
            let l = lane(src_x, src_y);
            if (src_x, src_y) == (0, 0) {
                if out_lane == 0 && d == 0 {
                    seen[l] = true;
                }
            } else {
                seen[l] = true;
            }
        }
    }
    (0..LANES)
        .filter(|&l| seen[l])
        .map(|l| {
            let (x, y) = lane_xy(l);
            Column::KeccakA {
                x: x as u8,
                y: y as u8,
                q: q_last,
            }
        })
        .collect()
}

/// True if lane `(x, y)` in permutation `m` absorbs `w1Encode(w1')` bits (`Absorbed::W1Bit`).
#[must_use]
pub fn perm_lane_has_w1(layout: &KeccakLayout, m: usize, x: usize, y: usize) -> bool {
    (0..LANE_BITS).any(|z| matches!(layout.wire(m, x, y, z), Absorbed::W1Bit { .. }))
}

/// True if lane `(x, y)` in permutation `m` absorbs `mu` bits (`Absorbed::Mu`).
#[must_use]
pub fn perm_lane_has_mu(layout: &KeccakLayout, m: usize, x: usize, y: usize) -> bool {
    (0..LANE_BITS).any(|z| matches!(layout.wire(m, x, y, z), Absorbed::Mu(_)))
}

/// The `(x, y)` lanes at `t = 0` (`m = 0`, `q = 0, s = 0`) that absorb `w1` bits, in canonical
/// `lane(x, y)` order (`lanes 8..17` for every set: `mu` fills lanes `0..8` exactly).
#[must_use]
pub fn t0_w1_lanes(layout: &KeccakLayout) -> Vec<(u8, u8)> {
    (0..LANES)
        .map(lane_xy)
        .filter(|&(x, y)| perm_lane_has_w1(layout, 0, x, y))
        .map(|(x, y)| (x as u8, y as u8))
        .collect()
}

/// The `(x, y)` lanes at `t = 0` (`m = 0`, `q = 0, s = 0`) that absorb `mu` bits, in canonical
/// `lane(x, y)` order (`lanes 0..8`).
#[must_use]
pub fn t0_mu_lanes(layout: &KeccakLayout) -> Vec<(u8, u8)> {
    (0..LANES)
        .map(lane_xy)
        .filter(|&(x, y)| perm_lane_has_mu(layout, 0, x, y))
        .map(|(x, y)| (x as u8, y as u8))
        .collect()
}

/// All `C[x, q]` columns in canonical `(x, q)` order (`x in 0..5` outer, `q in 0..Q` inner).
#[must_use]
pub fn c_columns(layout: &KeccakLayout) -> Vec<Column> {
    (0..COLS as u8)
        .flat_map(|x| (0..layout.slot_groups as u8).map(move |q| Column::KeccakC { x, q }))
        .collect()
}

/// All `C'[x, q]` columns in canonical `(x, q)` order (`x in 0..5` outer, `q in 0..Q` inner).
#[must_use]
pub fn c_prime_columns(layout: &KeccakLayout) -> Vec<Column> {
    (0..COLS as u8)
        .flat_map(|x| (0..layout.slot_groups as u8).map(move |q| Column::KeccakCPrime { x, q }))
        .collect()
}

/// All `A'[(x, y), q]` columns in canonical `(lane(x, y), q)` order (`lane in 0..25` outer,
/// `q in 0..Q` inner).
#[must_use]
pub fn a_prime_columns(layout: &KeccakLayout) -> Vec<Column> {
    (0..LANES)
        .flat_map(|l| {
            let (x, y) = lane_xy(l);
            (0..layout.slot_groups as u8).map(move |q| Column::KeccakA {
                x: x as u8,
                y: y as u8,
                q,
            })
        })
        .collect()
}

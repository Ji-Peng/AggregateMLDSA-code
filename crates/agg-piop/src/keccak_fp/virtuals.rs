//! Prover-side virtual lane materialization, polynomial constraint terms, and verifier-side
//! public multilinear evaluators for the prime-field Keccak sum-check instances.

use aerie_falcon::protocol::algebra::eq_table;
use agg_fpkeccak::{
    lane, lane_xy, rot, xor2, Absorbed, KeccakLayout, KeccakPublics, KeccakTables, COLS, LANES,
    LANE_BITS, RC, RECORD_CELLS_PER_POLY, ROUNDS_PER_PERM, SLOTS,
};
use agg_witness::keccak::sponge_trace_for;
use agg_witness::params::{dispatch_set, N as POLY_N, TR_BYTES};
use jolt_field::Zero;
use rayon::prelude::*;

use crate::keccak_fp::masks::{boundary_slot_groups, output_rot_columns, t0_w1_lanes};
use crate::kernel::{Lane, Term};
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::source_tables::{Cells, MainTables};
use crate::stage::LaneId;
use crate::OpeningField as E;

/// `bool_a` / `bool_c`: `v(v - 1) = v^2 - v` (degree 2).
#[must_use]
pub fn bool_term(lane: usize) -> Term {
    Term::Monomials(vec![(-1, vec![lane]), (1, vec![lane, lane])])
}

/// `xor3(a, b, c) = a + b + c - 2ab - 2ac - 2bc + 4abc` in the group lanes `a, b, c`.
#[must_use]
pub fn xor3_monomials(a: usize, b: usize, c: usize) -> Vec<(i64, Vec<usize>)> {
    vec![
        (1, vec![a]),
        (1, vec![b]),
        (1, vec![c]),
        (-2, vec![a, b]),
        (-2, vec![a, c]),
        (-2, vec![b, c]),
        (4, vec![a, b, c]),
    ]
}

/// `theta_c`: `C'[x, z] - xor3(C[x, z], C[x - 1, z], C[x + 1, z - 1])` (degree 3).
#[must_use]
pub fn theta_c_term(c_prime: usize, c_x: usize, c_xm1: usize, c_xp1_rot1: usize) -> Term {
    Term::Monomials(vec![
        (1, vec![c_prime]),
        (-1, vec![c_x]),
        (-1, vec![c_xm1]),
        (-1, vec![c_xp1_rot1]),
        (2, vec![c_x, c_xm1]),
        (2, vec![c_x, c_xp1_rot1]),
        (2, vec![c_xm1, c_xp1_rot1]),
        (-4, vec![c_x, c_xm1, c_xp1_rot1]),
    ])
}

/// `parity`: `d(d - 2)(d - 4) = d^3 - 6 d^2 + 8 d` where `d = sum_{y < 5} A'[x, y] - C'[x]` (degree 3).
#[must_use]
pub fn parity_term(a_col: [usize; COLS], c_prime: usize) -> Term {
    let vars = [a_col[0], a_col[1], a_col[2], a_col[3], a_col[4], c_prime];
    let signs: [i64; 6] = [1, 1, 1, 1, 1, -1];
    let mut monomials = Vec::with_capacity(56);
    // 8 * d
    for i in 0..6 {
        monomials.push((8 * signs[i], vec![vars[i]]));
    }
    // -6 * d^2
    for i in 0..6 {
        monomials.push((-6, vec![vars[i], vars[i]]));
        for j in (i + 1)..6 {
            monomials.push((-12 * signs[i] * signs[j], vec![vars[i], vars[j]]));
        }
    }
    // d^3
    for i in 0..6 {
        monomials.push((signs[i], vec![vars[i], vars[i], vars[i]]));
        for j in 0..6 {
            if i != j {
                monomials.push((3 * signs[j], vec![vars[i], vars[i], vars[j]]));
            }
        }
        for j in (i + 1)..6 {
            for k in (j + 1)..6 {
                monomials.push((
                    6 * signs[i] * signs[j] * signs[k],
                    vec![vars[i], vars[j], vars[k]],
                ));
            }
        }
    }
    Term::Monomials(monomials)
}

/// `chi(b0, b1, b2) = xor2(b0, (1 - b1) b2) = b0 + b2 - b1 b2 - 2 b0 b2 + 2 b0 b1 b2` (degree 3).
#[must_use]
pub fn chi_monomials(b0: usize, b1: usize, b2: usize) -> Vec<(i64, Vec<usize>)> {
    vec![
        (1, vec![b0]),
        (1, vec![b2]),
        (-1, vec![b1, b2]),
        (-2, vec![b0, b2]),
        (2, vec![b0, b1, b2]),
    ]
}

/// `k_link` on inner rounds `t >= 1`: `xor3(A', C, C') - chi(b0, b1, b2)` (degree 3).
#[must_use]
pub fn link_inner_term(
    a_prime: usize,
    c: usize,
    c_prime: usize,
    b0: usize,
    b1: usize,
    b2: usize,
) -> Term {
    let mut m = xor3_monomials(a_prime, c, c_prime);
    for (coeff, lanes) in chi_monomials(b0, b1, b2) {
        m.push((-coeff, lanes));
    }
    Term::Monomials(m)
}

/// `k_link` at `t = 0` on a lane absorbing `mu` or `Blk`: `xor3(A', C, C') - input` (degree 3).
#[must_use]
pub fn link_t0_input_term(a_prime: usize, c: usize, c_prime: usize, input: usize) -> Term {
    let mut m = xor3_monomials(a_prime, c, c_prime);
    m.push((-1, vec![input]));
    Term::Monomials(m)
}

/// `k_link` at `t = 0` on a capacity lane (`A_0 = 0`): `xor3(A', C, C')` (degree 3).
#[must_use]
pub fn link_t0_zero_term(a_prime: usize, c: usize, c_prime: usize) -> Term {
    Term::Monomials(xor3_monomials(a_prime, c, c_prime))
}

/// `k_out[lane]` on the output slot: `chi(b0, b1, b2) - c_tilde` (degree 3).
#[must_use]
pub fn out_term(b0: usize, b1: usize, b2: usize, c_tilde: usize) -> Term {
    let mut m = chi_monomials(b0, b1, b2);
    m.push((-1, vec![c_tilde]));
    Term::Monomials(m)
}

/// `k_link_bnd` on boundary slots (`t = 24 m`, `1 <= m < perms`):
/// `xor3(A', C, C') - xor2(chi(b0, b1, b2), blk)` (degree 4).
#[must_use]
pub fn link_bnd_term(
    a_prime: usize,
    c: usize,
    c_prime: usize,
    b0: usize,
    b1: usize,
    b2: usize,
    blk: usize,
) -> Term {
    let mut m = xor3_monomials(a_prime, c, c_prime);
    // -chi(b0, b1, b2) - blk + 2 * chi(b0, b1, b2) * blk
    for (coeff, lanes) in chi_monomials(b0, b1, b2) {
        m.push((-coeff, lanes.clone()));
        let mut with_blk = lanes;
        with_blk.push(blk);
        m.push((2 * coeff, with_blk));
    }
    m.push((-1, vec![blk]));
    Term::Monomials(m)
}

/// Extracts `(mu_i, w1Encode(w1'_i))` per live record from `mu_list` and `main`.
#[must_use]
pub fn extract_keccak_records(
    manifest: &CommitmentManifest,
    main: &MainTables,
    mu_list: &[[u8; TR_BYTES]],
) -> Vec<([u8; TR_BYTES], Vec<u8>)> {
    let n = manifest.records();
    assert_eq!(mu_list.len(), n);
    let shape = manifest.shape();
    let lane_cells = manifest.lane_cells();
    let bit_cols: Vec<Vec<&[i8]>> = (0..shape.w1_bits)
        .map(|t| main.bit_columns(manifest, t))
        .collect();
    let w1_bytes_len = (shape.k * POLY_N * shape.w1_bits) / 8;
    let mut out = Vec::with_capacity(n);
    for (i, &mu) in mu_list.iter().enumerate() {
        let mut w1_bytes = vec![0_u8; w1_bytes_len];
        for rho in 0..shape.k {
            let (poly_lane, base) = shape.row_lane(rho);
            for j in 0..POLY_N {
                let cell = i * lane_cells + base + j;
                for t in 0..shape.w1_bits {
                    let bit = (bit_cols[t][poly_lane][cell] & 1) as u8;
                    let bit_idx = (rho * POLY_N + j) * shape.w1_bits + t;
                    w1_bytes[bit_idx / 8] |= bit << (bit_idx % 8);
                }
            }
        }
        out.push((mu, w1_bytes));
    }
    out
}

/// Expanded `i8` columns for the committed and virtual Keccak tables of one batch.
#[derive(Clone, Debug)]
pub struct KeccakProverTables {
    layout: KeccakLayout,
    n_star: usize,
    cells_per_poly: usize,
    /// Expanded `i8` cells of all `layout.num_polys()` committed polynomials (`A'`, `C`, `C'`).
    committed: Vec<i8>,
    /// `Rot(C[x, q])`: `C[x, q]` rotated by 1 bit in `z` (`(z - 1) mod 64`), indexed by `x * Q + q`.
    c_rot: Vec<Vec<i8>>,
    /// `Rot(A'[x, y, Q - 1])` for each column in `output_rot_columns(&layout)`.
    a_rot_out_cols: Vec<Column>,
    a_rot_out: Vec<Vec<i8>>,
    /// `Prev[(x, y), q]`: `Rot(A'[(x, y), q])` shifted by 1 slot (`s - 1`, or `s = 15` of `q - 1`),
    /// indexed by `lane(x, y) * Q + q`.
    a_prev: Vec<Vec<i8>>,
    /// `PrevIota(q)`: `Prev[(0, 0), q] xor RC_{(16q + s - 1) mod 24}`, indexed by `q in 0..Q`.
    a_prev_iota: Vec<Vec<i8>>,
    /// `Blk[(x, y), 0]` at `t = 0` for each `(x, y)` in `t0_w1_lanes(&layout)`.
    blk_t0_lanes: Vec<(u8, u8)>,
    blk_t0: Vec<Vec<i8>>,
    /// `Blk[(x, y), q]` for each boundary slot group `q` in `boundary_slot_groups(&layout)` and
    /// `lane(x, y) in 0..25`, indexed by `bnd_idx * 25 + lane(x, y)`.
    bnd_qs: Vec<u8>,
    blk_bnd: Vec<Vec<i8>>,
    /// Public `Mu(lane)` (`lane in 0..8`) at `q = 0, s = 0`.
    mu: Vec<Vec<i8>>,
    /// Public `CTilde(out_lane)` (`out_lane in 0..L`) at `q = Q - 1, s = s_out`.
    c_tilde: Vec<Vec<i8>>,
}

impl KeccakProverTables {
    /// Materializes the expanded `i8` committed and virtual Keccak tables from `tables` and
    /// `publics`.
    #[must_use]
    pub fn build(tables: &KeccakTables, publics: &KeccakPublics) -> Self {
        let layout = tables.layout().clone();
        let n_star = tables.n_star();
        let q_count = layout.slot_groups;
        let cells_per_poly = RECORD_CELLS_PER_POLY * n_star;
        let committed = tables.expand_i8(0..layout.num_polys());

        let poly_slice = |poly: usize| -> &[i8] {
            &committed[poly * cells_per_poly..(poly + 1) * cells_per_poly]
        };

        // 1. C rotated by 1 in z: C[x, q]((z - 1) mod 64, s, i).
        let c_rot: Vec<Vec<i8>> = (0..COLS * q_count)
            .into_par_iter()
            .map(|idx| {
                let (x, q) = (idx / q_count, idx % q_count);
                let src = poly_slice(layout.c_poly(x, q));
                let mut out = vec![0_i8; cells_per_poly];
                for word_idx in 0..(SLOTS * n_star) {
                    let base = word_idx * LANE_BITS;
                    for z in 0..LANE_BITS {
                        out[base + z] = src[base + ((z + LANE_BITS - 1) % LANE_BITS)];
                    }
                }
                out
            })
            .collect();

        // 2. Output Rot(A') at q = Q - 1.
        let a_rot_out_cols = output_rot_columns(&layout);
        let s_out = (layout.rounds - 1) % SLOTS;
        let a_rot_out: Vec<Vec<i8>> = a_rot_out_cols
            .par_iter()
            .map(|&col| {
                let Column::KeccakA { x, y, q } = col else {
                    unreachable!("output_rot_columns returns KeccakA")
                };
                let (x, y, q) = (usize::from(x), usize::from(y), usize::from(q));
                let src = poly_slice(layout.a_prime_poly(x, y, q));
                let mut out = vec![0_i8; cells_per_poly];
                if (x, y) == (0, 0) {
                    let rc = RC[ROUNDS_PER_PERM - 1];
                    for i in 0..n_star {
                        for s in 0..SLOTS {
                            let base = (i * SLOTS + s) * LANE_BITS;
                            for z in 0..LANE_BITS {
                                let a_val = i64::from(src[base + z]);
                                let rc_bit = if s == s_out { ((rc >> z) & 1) as i64 } else { 0 };
                                out[base + z] = xor2(a_val, rc_bit) as i8;
                            }
                        }
                    }
                } else {
                    let r_off = rot(x, y) as usize;
                    for word_idx in 0..(SLOTS * n_star) {
                        let base = word_idx * LANE_BITS;
                        for z in 0..LANE_BITS {
                            out[base + z] = src[base + ((z + LANE_BITS - r_off) % LANE_BITS)];
                        }
                    }
                }
                out
            })
            .collect();

        // 3. Prev[(x, y), q]: Rot(A') shifted by 1 slot (s - 1, or s = 15 of q - 1 when s = 0).
        let a_prev: Vec<Vec<i8>> = (0..LANES * q_count)
            .into_par_iter()
            .map(|idx| {
                let (l, q) = (idx / q_count, idx % q_count);
                let (x, y) = lane_xy(l);
                let r_off = rot(x, y) as usize;
                let cur = poly_slice(layout.a_prime_poly(x, y, q));
                let prev_q = (q >= 1).then(|| poly_slice(layout.a_prime_poly(x, y, q - 1)));
                let mut out = vec![0_i8; cells_per_poly];
                for i in 0..n_star {
                    for s in 0..SLOTS {
                        let dst_base = (i * SLOTS + s) * LANE_BITS;
                        if s >= 1 {
                            let src_base = (i * SLOTS + (s - 1)) * LANE_BITS;
                            for z in 0..LANE_BITS {
                                out[dst_base + z] =
                                    cur[src_base + ((z + LANE_BITS - r_off) % LANE_BITS)];
                            }
                        } else if let Some(prev) = prev_q {
                            let src_base = (i * SLOTS + (SLOTS - 1)) * LANE_BITS;
                            for z in 0..LANE_BITS {
                                out[dst_base + z] =
                                    prev[src_base + ((z + LANE_BITS - r_off) % LANE_BITS)];
                            }
                        }
                    }
                }
                out
            })
            .collect();

        // 4. PrevIota(q): Prev[(0, 0), q] xor RC_{(16q + s - 1) mod 24}.
        let a_prev_iota: Vec<Vec<i8>> = (0..q_count)
            .into_par_iter()
            .map(|q| {
                let src = &a_prev[lane(0, 0) * q_count + q];
                let mut out = vec![0_i8; cells_per_poly];
                for i in 0..n_star {
                    for s in 0..SLOTS {
                        let base = (i * SLOTS + s) * LANE_BITS;
                        let t = layout.time(q, s);
                        let rc = if t >= 1 && t < layout.rounds {
                            RC[(t - 1) % ROUNDS_PER_PERM]
                        } else {
                            0
                        };
                        for z in 0..LANE_BITS {
                            let prev_val = i64::from(src[base + z]);
                            let rc_bit = ((rc >> z) & 1) as i64;
                            out[base + z] = xor2(prev_val, rc_bit) as i8;
                        }
                    }
                }
                out
            })
            .collect();

        // 5. Blk at t = 0 (q = 0, s = 0) for the w1 lanes.
        let blk_t0_lanes = t0_w1_lanes(&layout);
        let blk_t0: Vec<Vec<i8>> = blk_t0_lanes
            .par_iter()
            .map(|&(x, y)| {
                let (x, y) = (usize::from(x), usize::from(y));
                let mut out = vec![0_i8; cells_per_poly];
                for i in 0..n_star {
                    let base = (i * SLOTS) * LANE_BITS; // s = 0
                    for z in 0..LANE_BITS {
                        if matches!(
                            layout.wire(0, x, y, z),
                            Absorbed::W1Bit { .. } | Absorbed::Pad(_)
                        ) {
                            out[base + z] = publics.absorbed_bit(i, 0, x, y, z) as i8;
                        }
                    }
                }
                out
            })
            .collect();

        // 6. Blk at the boundary slot groups (t = 24 m, 1 <= m < perms).
        let bnd_qs = boundary_slot_groups(&layout);
        let blk_bnd: Vec<Vec<i8>> = (0..bnd_qs.len() * LANES)
            .into_par_iter()
            .map(|idx| {
                let (bnd_idx, l) = (idx / LANES, idx % LANES);
                let q = usize::from(bnd_qs[bnd_idx]);
                let (x, y) = lane_xy(l);
                let mut out = vec![0_i8; cells_per_poly];
                for s in 0..SLOTS {
                    let t = layout.time(q, s);
                    if layout.is_boundary(t) {
                        let m = t / ROUNDS_PER_PERM;
                        for i in 0..n_star {
                            let base = (i * SLOTS + s) * LANE_BITS;
                            for z in 0..LANE_BITS {
                                if matches!(
                                    layout.wire(m, x, y, z),
                                    Absorbed::W1Bit { .. } | Absorbed::Pad(_)
                                ) {
                                    out[base + z] = publics.absorbed_bit(i, m, x, y, z) as i8;
                                }
                            }
                        }
                    }
                }
                out
            })
            .collect();

        // 7. Public Mu(lane_idx) at q = 0, s = 0 (lanes 0..8).
        let mu: Vec<Vec<i8>> = (0..8_usize)
            .into_par_iter()
            .map(|l| {
                let (x, y) = lane_xy(l);
                let mut out = vec![0_i8; cells_per_poly];
                for i in 0..n_star {
                    let base = (i * SLOTS) * LANE_BITS; // s = 0
                    for z in 0..LANE_BITS {
                        out[base + z] = publics.absorbed_bit(i, 0, x, y, z) as i8;
                    }
                }
                out
            })
            .collect();

        // 8. Public CTilde(out_lane) at q = Q - 1, s = s_out.
        let c_tilde: Vec<Vec<i8>> = (0..layout.output_lanes)
            .into_par_iter()
            .map(|out_lane| {
                let mut out = vec![0_i8; cells_per_poly];
                for i in 0..n_star {
                    let base = (i * SLOTS + s_out) * LANE_BITS;
                    for z in 0..LANE_BITS {
                        out[base + z] = publics.c_tilde_bit(i, out_lane, z) as i8;
                    }
                }
                out
            })
            .collect();

        Self {
            layout,
            n_star,
            cells_per_poly,
            committed,
            c_rot,
            a_rot_out_cols,
            a_rot_out,
            a_prev,
            a_prev_iota,
            blk_t0_lanes,
            blk_t0,
            bnd_qs,
            blk_bnd,
            mu,
            c_tilde,
        }
    }

    /// The parameter-set Keccak layout.
    #[inline]
    #[must_use]
    pub const fn layout(&self) -> &KeccakLayout {
        &self.layout
    }

    /// Padded batch size `N*`.
    #[inline]
    #[must_use]
    pub const fn n_star(&self) -> usize {
        self.n_star
    }

    /// The `i8` cells of a committed Keccak column (`KeccakA`, `KeccakC`, `KeccakCPrime`).
    #[must_use]
    pub fn column_slice(&self, column: Column) -> &[i8] {
        let poly = match column {
            Column::KeccakA { x, y, q } => {
                self.layout.a_prime_poly(usize::from(x), usize::from(y), usize::from(q))
            }
            Column::KeccakC { x, q } => self.layout.c_poly(usize::from(x), usize::from(q)),
            Column::KeccakCPrime { x, q } => {
                self.layout.c_prime_poly(usize::from(x), usize::from(q))
            }
            other => panic!("{other:?} is not a Keccak column"),
        };
        &self.committed[poly * self.cells_per_poly..(poly + 1) * self.cells_per_poly]
    }

    /// The `Cells::Narrow` views of one Keccak commitment group (`KeccakA(y)`, `KeccakC`,
    /// `KeccakCPrime`) in manifest order.
    #[must_use]
    pub fn group_cells(&self, role: GroupRole) -> Vec<Cells<'_>> {
        let group_idx = match role {
            GroupRole::KeccakA(y) => usize::from(y),
            GroupRole::KeccakC => 5,
            GroupRole::KeccakCPrime => 6,
            other => panic!("{other:?} is not a Keccak group"),
        };
        let range = self.layout.group_poly_range(group_idx);
        range
            .map(|poly| {
                Cells::Narrow(
                    &self.committed[poly * self.cells_per_poly..(poly + 1) * self.cells_per_poly],
                )
            })
            .collect()
    }

    /// The `i8` cells of `LaneId::KeccakRot(column)`.
    #[must_use]
    pub fn rot_slice(&self, column: Column) -> &[i8] {
        match column {
            Column::KeccakC { x, q } => {
                &self.c_rot[usize::from(x) * self.layout.slot_groups + usize::from(q)]
            }
            Column::KeccakA { .. } => {
                let pos = self
                    .a_rot_out_cols
                    .iter()
                    .position(|&c| c == column)
                    .unwrap_or_else(|| panic!("KeccakRot({column:?}) is not in output_rot_columns"));
                &self.a_rot_out[pos]
            }
            other => panic!("KeccakRot is not defined for {other:?}"),
        }
    }

    /// The `i8` cells of `LaneId::KeccakPrev(column)`.
    #[must_use]
    pub fn prev_slice(&self, column: Column) -> &[i8] {
        let Column::KeccakA { x, y, q } = column else {
            panic!("KeccakPrev is only defined on KeccakA, got {column:?}");
        };
        let l = lane(usize::from(x), usize::from(y));
        &self.a_prev[l * self.layout.slot_groups + usize::from(q)]
    }

    /// The `i8` cells of `LaneId::KeccakPrevIota(q)`.
    #[must_use]
    pub fn prev_iota_slice(&self, q: u8) -> &[i8] {
        &self.a_prev_iota[usize::from(q)]
    }

    /// The `i8` cells of `LaneId::KeccakBlk { x, y, q }`.
    #[must_use]
    pub fn blk_slice(&self, x: u8, y: u8, q: u8) -> &[i8] {
        if q == 0 {
            let pos = self
                .blk_t0_lanes
                .iter()
                .position(|&pair| pair == (x, y))
                .unwrap_or_else(|| panic!("KeccakBlk({x}, {y}, 0) is not a t = 0 w1 lane"));
            &self.blk_t0[pos]
        } else {
            let bnd_idx = self
                .bnd_qs
                .iter()
                .position(|&bq| bq == q)
                .unwrap_or_else(|| panic!("q = {q} is not a boundary slot group"));
            let l = lane(usize::from(x), usize::from(y));
            &self.blk_bnd[bnd_idx * LANES + l]
        }
    }

    /// The `i8` cells of `LaneId::KeccakMu(lane_idx)`.
    #[must_use]
    pub fn mu_slice(&self, lane_idx: u8) -> &[i8] {
        &self.mu[usize::from(lane_idx)]
    }

    /// The `i8` cells of `LaneId::KeccakCTilde(out_lane)`.
    #[must_use]
    pub fn c_tilde_slice(&self, out_lane: u8) -> &[i8] {
        &self.c_tilde[usize::from(out_lane)]
    }

    /// Kernel [`Lane`] for a committed Keccak column.
    #[must_use]
    pub fn column_lane(&self, column: Column) -> Lane<'_> {
        Lane::column(self.column_slice(column))
    }

    /// Kernel [`Lane`] for a committed or virtual Keccak table (`None` for non-Keccak ids).
    #[must_use]
    pub fn lane(&self, id: LaneId) -> Option<Lane<'_>> {
        let cells = match id {
            LaneId::Column(
                column @ (Column::KeccakA { .. }
                | Column::KeccakC { .. }
                | Column::KeccakCPrime { .. }),
            ) => self.column_slice(column),
            LaneId::KeccakRot(column) => self.rot_slice(column),
            LaneId::KeccakPrev(column) => self.prev_slice(column),
            LaneId::KeccakPrevIota(q) => self.prev_iota_slice(q),
            LaneId::KeccakBlk { x, y, q } => self.blk_slice(x, y, q),
            LaneId::KeccakMu(lane_idx) => self.mu_slice(lane_idx),
            LaneId::KeccakCTilde(out_lane) => self.c_tilde_slice(out_lane),
            _ => return None,
        };
        Some(Lane::column(cells))
    }
}

/// The commitment hash `c~ = SHAKE256(0^{64 + |w1Encode|})` of a padded record (`i >= N`).
#[must_use]
pub fn zero_message_c_tilde(layout: &KeccakLayout) -> Vec<u8> {
    let zero_mu = [0_u8; TR_BYTES];
    let zero_w1 = vec![0_u8; layout.w1_encode_bytes];
    dispatch_set!(layout.set, |P| {
        sponge_trace_for::<P>(&zero_mu, &zero_w1)
            .c_tilde
            .as_ref()
            .to_vec()
    })
}

/// Verifier evaluation of the public `KeccakMu(lane_idx)` table at a Keccak-domain point
/// `r_k = (r_z, r_s, r_i)`.
#[must_use]
pub fn mu_mle_at(mu_records: &[[u8; TR_BYTES]], r_k: &[E], lane_idx: usize) -> E {
    let eq_z = eq_table(&r_k[..6]);
    let eq_s0 = eq_table(&r_k[6..10])[0];
    let eq_i = eq_table(&r_k[10..]);
    let sum_i = mu_records
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (i, mu)| {
            let word = u64::from_le_bytes(
                mu[8 * lane_idx..8 * (lane_idx + 1)]
                    .try_into()
                    .expect("8-byte lane"),
            );
            let mut dot = E::zero();
            for z in 0..LANE_BITS {
                if (word >> z) & 1 == 1 {
                    dot += eq_z[z];
                }
            }
            acc + eq_i[i] * dot
        });
    eq_s0 * sum_i
}

/// Verifier evaluation of the public `KeccakCTilde(out_lane)` table at a Keccak-domain point
/// `r_k = (r_z, r_s, r_i)` (including padded records `i >= N`, which carry the zero-message `c~`).
#[must_use]
pub fn c_tilde_mle_at(
    layout: &KeccakLayout,
    c_tilde_records: &[Vec<u8>],
    r_k: &[E],
    out_lane: usize,
) -> E {
    let eq_z = eq_table(&r_k[..6]);
    let s_out = (layout.rounds - 1) % SLOTS;
    let eq_s_out = eq_table(&r_k[6..10])[s_out];
    let eq_i = eq_table(&r_k[10..]);
    let word_dot = |bytes: &[u8]| -> E {
        let word = u64::from_le_bytes(
            bytes[8 * out_lane..8 * (out_lane + 1)]
                .try_into()
                .expect("8-byte lane"),
        );
        let mut dot = E::zero();
        for z in 0..LANE_BITS {
            if (word >> z) & 1 == 1 {
                dot += eq_z[z];
            }
        }
        dot
    };
    let live_sum = c_tilde_records
        .iter()
        .enumerate()
        .fold(E::zero(), |acc, (i, ct)| acc + eq_i[i] * word_dot(ct));
    let pad_sum = if c_tilde_records.len() < eq_i.len() {
        let pad_ct = zero_message_c_tilde(layout);
        let pad_dot = word_dot(&pad_ct);
        let pad_eq = eq_i[c_tilde_records.len()..]
            .iter()
            .copied()
            .fold(E::zero(), |a, b| a + b);
        pad_eq * pad_dot
    } else {
        E::zero()
    };
    eq_s_out * (live_sum + pad_sum)
}

/// Verifier evaluation of the round-constant offset of `Rot[(0, 0), Q - 1]` (output slot `s_out`,
/// round `r = 23`) at a Keccak-domain point `r_k = (r_z, r_s, r_i)`.
#[must_use]
pub fn rc_out_mle_at(layout: &KeccakLayout, r_k: &[E]) -> E {
    let eq_z = eq_table(&r_k[..6]);
    let s_out = (layout.rounds - 1) % SLOTS;
    let eq_s_out = eq_table(&r_k[6..10])[s_out];
    let rc = RC[ROUNDS_PER_PERM - 1];
    let z_sum = (0..LANE_BITS)
        .filter(|&z| (rc >> z) & 1 == 1)
        .fold(E::zero(), |acc, z| acc + eq_z[z]);
    eq_s_out * z_sum
}

/// Verifier evaluation of the round-constant offset of `PrevIota(q)` at a Keccak-domain point
/// `r_k = (r_z, r_s, r_i)`.
#[must_use]
pub fn rc_prev_mle_at(layout: &KeccakLayout, q: usize, r_k: &[E]) -> E {
    let eq_z = eq_table(&r_k[..6]);
    let eq_s = eq_table(&r_k[6..10]);
    let mut total = E::zero();
    for s in 0..SLOTS {
        let t = layout.time(q, s);
        if t >= 1 && t < layout.rounds {
            let rc = RC[(t - 1) % ROUNDS_PER_PERM];
            let z_sum = (0..LANE_BITS)
                .filter(|&z| (rc >> z) & 1 == 1)
                .fold(E::zero(), |acc, z| acc + eq_z[z]);
            total += eq_s[s] * z_sum;
        }
    }
    total
}

/// Verifier evaluation of the constant padding-bit offset (`0x1F` or `0x80` in permutation
/// `m = perms - 1`) of `Blk[(x, y), q]` at a Keccak-domain point `r_k = (r_z, r_s, r_i)`.
#[must_use]
pub fn pad_blk_mle_at(layout: &KeccakLayout, x: usize, y: usize, q: usize, r_k: &[E]) -> E {
    let eq_z = eq_table(&r_k[..6]);
    let eq_s = eq_table(&r_k[6..10]);
    let mut total = E::zero();
    for s in 0..SLOTS {
        let t = layout.time(q, s);
        if (q == 0 && s == 0) || layout.is_boundary(t) {
            let m = t / ROUNDS_PER_PERM;
            let z_sum = (0..LANE_BITS)
                .filter(|&z| matches!(layout.wire(m, x, y, z), Absorbed::Pad(1)))
                .fold(E::zero(), |acc, z| acc + eq_z[z]);
            total += eq_s[s] * z_sum;
        }
    }
    total
}

/// Helper for ML-DSA-44 tests that only have `MainTables` and `mu_list`: builds `(KeccakTables, KeccakPublics)`.
#[must_use]
pub fn build_keccak_witness(
    manifest: &CommitmentManifest,
    main: &MainTables,
    mu_list: &[[u8; TR_BYTES]],
) -> (KeccakTables, KeccakPublics) {
    let layout = KeccakLayout::for_set(manifest.set());
    let records = extract_keccak_records(manifest, main, mu_list);
    let n_star = manifest.padded_records();
    let tables = KeccakTables::build(&layout, &records, n_star);
    let publics = KeccakPublics::from_records(&layout, &records, n_star);
    (tables, publics)
}

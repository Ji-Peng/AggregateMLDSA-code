//! Stage-1 and stage-2 `GroupSpec` builders for the prime-field Keccak constraints.

use agg_fpkeccak::{lane, lane_xy, pi_inv, KeccakLayout, COLS, LANES};

use crate::keccak_fp::masks::{
    a_prime_columns, boundary_mask, boundary_slot_groups, c_columns, inner_after_t0_mask,
    live_mask, output_mask, perm_lane_has_mu, perm_lane_has_w1, T0_MASK,
};
use crate::keccak_fp::virtuals::{
    bool_term, link_bnd_term, link_inner_term, link_t0_input_term, link_t0_zero_term, out_term,
    parity_term, theta_c_term,
};
use crate::kernel::Term;
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::stage::{EqPoint, GroupSpec, InstanceSpec, LaneId};

const KECCAK_DOMAIN: GroupRole = GroupRole::KeccakA(0);

fn stage1_eq(mask: u16) -> EqPoint {
    if mask == 0xFFFF {
        EqPoint::Tau
    } else {
        EqPoint::TauMasked(mask)
    }
}

fn stage2_eq(mask: u16) -> EqPoint {
    if mask == 0xFFFF {
        EqPoint::R1
    } else {
        EqPoint::R1Masked(mask)
    }
}

fn group_for(groups: &mut Vec<GroupSpec>, eq: EqPoint) -> &mut GroupSpec {
    if let Some(pos) = groups.iter().position(|g| g.eq == eq) {
        return &mut groups[pos];
    }
    groups.push(GroupSpec {
        domain: KECCAK_DOMAIN,
        eq,
        lanes: Vec::new(),
        instances: Vec::new(),
    });
    groups.last_mut().expect("just pushed")
}

fn intern_lane(group: &mut GroupSpec, id: LaneId) -> usize {
    if let Some(pos) = group.lanes.iter().position(|&l| l == id) {
        pos
    } else {
        let pos = group.lanes.len();
        group.lanes.push(id);
        pos
    }
}

fn prev_chi_lanes(group: &mut GroupSpec, x: u8, y: u8, q: u8) -> [usize; 3] {
    std::array::from_fn(|d| {
        let (src_x, src_y) = pi_inv((usize::from(x) + d) % COLS, usize::from(y));
        let id = if d == 0 && (x, y) == (0, 0) {
            LaneId::KeccakPrevIota(q)
        } else {
            LaneId::KeccakPrev(Column::KeccakA {
                x: src_x as u8,
                y: src_y as u8,
                q,
            })
        };
        intern_lane(group, id)
    })
}

/// Builds the stage-1 Keccak zero-check groups (`k_bool_a`, `k_bool_c`, `k_theta_c`, `k_parity`,
/// `k_link`, `k_out`) starting at batching index `gamma0`, returning `(groups, used_gammas)`.
#[must_use]
pub fn stage1_groups(manifest: &CommitmentManifest, gamma0: usize) -> (Vec<GroupSpec>, usize) {
    let layout = KeccakLayout::for_set(manifest.set());
    let mut groups = Vec::new();
    let mut gamma = gamma0;

    // 1. k_bool_a per (x, y, q) on live slots.
    for col in a_prime_columns(&layout) {
        let Column::KeccakA { q, .. } = col else {
            unreachable!("a_prime_columns")
        };
        let g = group_for(&mut groups, stage1_eq(live_mask(&layout, usize::from(q))));
        let a = intern_lane(g, LaneId::Column(col));
        g.instances.push(InstanceSpec::new(gamma, vec![bool_term(a)]));
        gamma += 1;
    }

    // 2. k_bool_c per (x, q) on live slots.
    for col in c_columns(&layout) {
        let Column::KeccakC { q, .. } = col else {
            unreachable!("c_columns")
        };
        let g = group_for(&mut groups, stage1_eq(live_mask(&layout, usize::from(q))));
        let c = intern_lane(g, LaneId::Column(col));
        g.instances.push(InstanceSpec::new(gamma, vec![bool_term(c)]));
        gamma += 1;
    }

    // 3. k_theta_c per (x, q) on live slots.
    for x in 0..COLS as u8 {
        let xm1 = (x + COLS as u8 - 1) % COLS as u8;
        let xp1 = (x + 1) % COLS as u8;
        for q in 0..layout.slot_groups as u8 {
            let g = group_for(&mut groups, stage1_eq(live_mask(&layout, usize::from(q))));
            let cp = intern_lane(g, LaneId::Column(Column::KeccakCPrime { x, q }));
            let cx = intern_lane(g, LaneId::Column(Column::KeccakC { x, q }));
            let cxm1 = intern_lane(g, LaneId::Column(Column::KeccakC { x: xm1, q }));
            let cxp1_rot = intern_lane(g, LaneId::KeccakRot(Column::KeccakC { x: xp1, q }));
            g.instances
                .push(InstanceSpec::new(gamma, vec![theta_c_term(cp, cx, cxm1, cxp1_rot)]));
            gamma += 1;
        }
    }

    // 4. k_parity per (x, q) on live slots.
    for x in 0..COLS as u8 {
        for q in 0..layout.slot_groups as u8 {
            let g = group_for(&mut groups, stage1_eq(live_mask(&layout, usize::from(q))));
            let a_col: [usize; COLS] =
                std::array::from_fn(|y| intern_lane(g, LaneId::Column(Column::KeccakA { x, y: y as u8, q })));
            let cp = intern_lane(g, LaneId::Column(Column::KeccakCPrime { x, q }));
            g.instances
                .push(InstanceSpec::new(gamma, vec![parity_term(a_col, cp)]));
            gamma += 1;
        }
    }

    // 5. k_link per (x, y, q) on inner slots (with t = 0 as a second part on T0_MASK when q = 0).
    for col in a_prime_columns(&layout) {
        let Column::KeccakA { x, y, q } = col else {
            unreachable!("a_prime_columns")
        };
        let mask_inner = inner_after_t0_mask(&layout, usize::from(q));
        let g = group_for(&mut groups, stage1_eq(mask_inner));
        let a = intern_lane(g, LaneId::Column(col));
        let c = intern_lane(g, LaneId::Column(Column::KeccakC { x, q }));
        let cp = intern_lane(g, LaneId::Column(Column::KeccakCPrime { x, q }));
        let [b0, b1, b2] = prev_chi_lanes(g, x, y, q);
        g.instances.push(InstanceSpec::new(
            gamma,
            vec![link_inner_term(a, c, cp, b0, b1, b2)],
        ));

        if q == 0 {
            let g0 = group_for(&mut groups, stage1_eq(T0_MASK));
            let a0 = intern_lane(g0, LaneId::Column(col));
            let c0 = intern_lane(g0, LaneId::Column(Column::KeccakC { x, q: 0 }));
            let cp0 = intern_lane(g0, LaneId::Column(Column::KeccakCPrime { x, q: 0 }));
            let (ux, uy) = (usize::from(x), usize::from(y));
            let term: Term = if perm_lane_has_mu(&layout, 0, ux, uy) {
                let mu = intern_lane(g0, LaneId::KeccakMu(lane(ux, uy) as u8));
                link_t0_input_term(a0, c0, cp0, mu)
            } else if perm_lane_has_w1(&layout, 0, ux, uy) {
                let blk = intern_lane(g0, LaneId::KeccakBlk { x, y, q: 0 });
                link_t0_input_term(a0, c0, cp0, blk)
            } else {
                link_t0_zero_term(a0, c0, cp0)
            };
            g0.instances.push(InstanceSpec::new(gamma, vec![term]));
        }
        gamma += 1;
    }

    // 6. k_out per output lane (0..L) on the output slot of q = Q - 1.
    let q_last = (layout.slot_groups - 1) as u8;
    let out_mask = output_mask(&layout);
    for out_lane in 0..layout.output_lanes {
        let (x, y) = lane_xy(out_lane);
        let g = group_for(&mut groups, stage1_eq(out_mask));
        let [b0, b1, b2]: [usize; 3] = std::array::from_fn(|d| {
            let (src_x, src_y) = pi_inv((x + d) % COLS, y);
            let col = Column::KeccakA {
                x: src_x as u8,
                y: src_y as u8,
                q: q_last,
            };
            let id = if (src_x, src_y) == (0, 0) && !(out_lane == 0 && d == 0) {
                LaneId::Column(col)
            } else {
                LaneId::KeccakRot(col)
            };
            intern_lane(g, id)
        });
        let ct = intern_lane(g, LaneId::KeccakCTilde(out_lane as u8));
        g.instances
            .push(InstanceSpec::new(gamma, vec![out_term(b0, b1, b2, ct)]));
        gamma += 1;
    }

    (groups, gamma - gamma0)
}

/// Builds the stage-2 Keccak boundary-link zero-check groups (`k_link_bnd` at `t = 24 m`,
/// `1 <= m < P`) starting at batching index `gamma0`, returning `(groups, used_gammas)`.
#[must_use]
pub fn stage2_groups(manifest: &CommitmentManifest, gamma0: usize) -> (Vec<GroupSpec>, usize) {
    let layout = KeccakLayout::for_set(manifest.set());
    let mut groups = Vec::new();
    let mut gamma = gamma0;

    for q in boundary_slot_groups(&layout) {
        let mask = boundary_mask(&layout, usize::from(q));
        let g = group_for(&mut groups, stage2_eq(mask));
        for l in 0..LANES {
            let (ux, uy) = lane_xy(l);
            let (x, y) = (ux as u8, uy as u8);
            let a = intern_lane(g, LaneId::Column(Column::KeccakA { x, y, q }));
            let c = intern_lane(g, LaneId::Column(Column::KeccakC { x, q }));
            let cp = intern_lane(g, LaneId::Column(Column::KeccakCPrime { x, q }));
            let [b0, b1, b2] = prev_chi_lanes(g, x, y, q);
            let blk = intern_lane(g, LaneId::KeccakBlk { x, y, q });
            g.instances.push(InstanceSpec::new(
                gamma,
                vec![link_bnd_term(a, c, cp, b0, b1, b2, blk)],
            ));
            gamma += 1;
        }
    }

    (groups, gamma - gamma0)
}

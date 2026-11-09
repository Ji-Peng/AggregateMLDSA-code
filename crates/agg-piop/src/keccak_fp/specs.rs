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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashSet};

    use agg_fpkeccak::{Absorbed, KeccakPublics, KeccakTables, LANE_BITS, ROUNDS_PER_PERM, SLOTS};
    use agg_witness::gen::SplitMix64;
    use agg_witness::params::{Set, TR_BYTES};

    use super::*;
    use crate::keccak_fp::virtuals::KeccakProverTables;
    use crate::manifest::Relation;

    fn manifest(set: Set) -> CommitmentManifest {
        let manifest = CommitmentManifest::for_set_fp(set, 8, Relation::default()).unwrap();
        assert_eq!(manifest.d_sig(), 3);
        manifest
    }

    /// The instances of `groups` by gamma index, each with the lanes of its group.
    fn by_gamma(groups: &[GroupSpec]) -> BTreeMap<usize, Vec<(EqPoint, Vec<LaneId>)>> {
        let mut out: BTreeMap<usize, Vec<(EqPoint, Vec<LaneId>)>> = BTreeMap::new();
        for g in groups {
            assert_eq!(g.domain, KECCAK_DOMAIN);
            for inst in &g.instances {
                let lanes = inst
                    .terms
                    .iter()
                    .flat_map(|t| match t {
                        Term::Bool(l) => vec![*l],
                        Term::Monomials(m) => m.iter().flat_map(|(_, ls)| ls.clone()).collect(),
                        other => panic!("unexpected Keccak term {other:?}"),
                    })
                    .map(|l| g.lanes[l])
                    .collect();
                out.entry(inst.gamma).or_default().push((g.eq, lanes));
            }
        }
        out
    }

    /// Stage 1 holds `25 Q` `k_bool_a`, `5 Q` each of `k_bool_c`, `k_theta_c`, `k_parity`,
    /// `25 Q` `k_link` (with a second part at `t = 0` for `q = 0`) and `L` `k_out`; stage 2
    /// `25 (P - 1)` `k_link_bnd`, one per lane and boundary slot group.
    #[test]
    fn instance_counts_per_set() {
        for (set, perms, q_count, out_lanes) in [(Set::S44, 7, 11, 4), (Set::S65, 7, 11, 6), (Set::S87, 9, 14, 8)] {
            let manifest = manifest(set);
            let layout = KeccakLayout::for_set(set);
            assert_eq!((layout.perms, layout.slot_groups, layout.output_lanes), (perms, q_count, out_lanes));
            assert_eq!(manifest.d_keccak(), 13);
            assert_eq!(manifest.d_stage(), 13, "{set:?}: d_stage = d_K");
            assert_eq!(manifest.d_main(), if set == Set::S44 { 13 } else { 11 });

            let gamma0 = 5;
            let (groups, used) = stage1_groups(&manifest, gamma0);
            assert_eq!(used, 65 * q_count + out_lanes, "{set:?}");
            let gammas = by_gamma(&groups);
            assert_eq!(gammas.keys().copied().collect::<Vec<_>>(), (gamma0..gamma0 + used).collect::<Vec<_>>());
            let range = |start: usize, len: usize| (gamma0 + start..gamma0 + start + len).map(|g| &gammas[&g]);
            let q = q_count;
            // k_bool_a, k_bool_c, k_theta_c, k_parity: one part each on the live slots.
            for parts in range(0, 40 * q) {
                assert_eq!(parts.len(), 1);
                assert!(matches!(parts[0].0, EqPoint::Tau | EqPoint::TauMasked(_)));
            }
            // k_link: a second part at t = 0 for the 25 lanes of q = 0.
            let links: Vec<_> = range(40 * q, 25 * q).collect();
            assert_eq!(links.iter().filter(|p| p.len() == 2).count(), 25);
            assert!(links.iter().all(|p| p.len() <= 2));
            let t0 = links.iter().flat_map(|p| p.iter()).filter(|part| part.0 == EqPoint::TauMasked(T0_MASK)).count();
            assert_eq!(t0, 25);
            // k_out: the output slot and one public c~ lane each.
            for (lane_idx, parts) in range(65 * q, out_lanes).enumerate() {
                assert_eq!(parts.len(), 1);
                assert_eq!(parts[0].0, EqPoint::TauMasked(output_mask(&layout)));
                assert!(parts[0].1.contains(&LaneId::KeccakCTilde(lane_idx as u8)));
            }
            // Degree 4 with the equality factor.
            assert!(groups.iter().all(|g| g.instances.iter().all(|i| i.terms.iter().all(|t| t.degree() < 4))));

            let (groups2, used2) = stage2_groups(&manifest, 0);
            assert_eq!(used2, 25 * (perms - 1), "{set:?}");
            let gammas2 = by_gamma(&groups2);
            assert_eq!(gammas2.len(), used2);
            let bnd = boundary_slot_groups(&layout);
            assert_eq!(bnd.len(), perms - 1);
            let mut blocks: Vec<u8> = Vec::new();
            for parts in gammas2.values() {
                assert_eq!(parts.len(), 1);
                assert!(matches!(parts[0].0, EqPoint::R1Masked(_)));
                let blk: HashSet<u8> =
                    parts[0].1.iter().filter_map(|l| if let LaneId::KeccakBlk { q, .. } = l { Some(*q) } else { None }).collect();
                let blk: Vec<u8> = blk.into_iter().collect();
                assert_eq!(blk.len(), 1, "one block lane per boundary link");
                blocks.push(blk[0]);
            }
            for q in &bnd {
                assert_eq!(blocks.iter().filter(|&&b| b == *q).count(), 25);
            }
            // Degree 5 with the equality factor.
            assert!(groups2.iter().all(|g| g.instances.iter().all(|i| i.terms.iter().all(|t| t.degree() < 5))));
        }
    }

    /// Every `KeccakBlk` lane of stages 1 and 2 holds, at its absorption slot, the bits the
    /// set's sponge wiring puts there (`w1Encode` bits and padding, `mu` and capacity read as
    /// zero) and zero at every other slot; together they read every `w1Encode` bit once. The
    /// first block splits into whole `mu` lanes and whole `w1` lanes.
    #[test]
    fn block_lanes_resolve_through_the_wiring() {
        let mut rng = SplitMix64::new(0xB10C_0065);
        for set in Set::ALL {
            let manifest = manifest(set);
            let layout = KeccakLayout::for_set(set);
            let n_star = manifest.padded_records();
            let records: Vec<([u8; TR_BYTES], Vec<u8>)> = (0..6)
                .map(|_| {
                    let mu: [u8; TR_BYTES] = std::array::from_fn(|_| rng.next_u64() as u8);
                    (mu, (0..layout.w1_encode_bytes).map(|_| rng.next_u64() as u8).collect())
                })
                .collect();
            let witness = KeccakTables::build(&layout, &records, n_star);
            let publics = KeccakPublics::from_records(&layout, &records, n_star);
            let keccak = KeccakProverTables::build(&witness, &publics);

            for l in 0..LANES {
                let (x, y) = lane_xy(l);
                let (mu, w1) = (perm_lane_has_mu(&layout, 0, x, y), perm_lane_has_w1(&layout, 0, x, y));
                assert!(!(mu && w1), "{set:?}: lane {l} of the first block mixes mu and w1");
                assert_eq!(mu, l < 8);
                if !mu && !w1 {
                    assert!((0..LANE_BITS).all(|z| layout.wire(0, x, y, z) == Absorbed::Zero));
                }
            }

            let (s1, _) = stage1_groups(&manifest, 0);
            let (s2, _) = stage2_groups(&manifest, 0);
            let blocks: HashSet<LaneId> = s1
                .iter()
                .chain(&s2)
                .flat_map(|g| g.lanes.iter().copied())
                .filter(|l| matches!(l, LaneId::KeccakBlk { .. }))
                .collect();
            let mut seen = HashSet::new();
            for &id in &blocks {
                let LaneId::KeccakBlk { x, y, q } = id else { unreachable!() };
                let cells = keccak.blk_slice(x, y, q);
                let (ux, uy, uq) = (usize::from(x), usize::from(y), usize::from(q));
                for s in 0..SLOTS {
                    let t = layout.time(uq, s);
                    let m = (t == 0 || layout.is_boundary(t)).then_some(t / ROUNDS_PER_PERM);
                    for z in 0..LANE_BITS {
                        let wire = m.map(|m| layout.wire(m, ux, uy, z));
                        if let Some(Absorbed::W1Bit { t: bit, j, rho }) = wire {
                            assert!(seen.insert((bit, j, rho)), "{set:?}: bit read twice");
                        }
                        let read = matches!(wire, Some(Absorbed::W1Bit { .. } | Absorbed::Pad(_)));
                        for i in 0..n_star {
                            let want = match m {
                                Some(m) if read => publics.absorbed_bit(i, m, ux, uy, z),
                                _ => 0,
                            };
                            assert_eq!(i64::from(cells[z + LANE_BITS * s + 1024 * i]), want, "{set:?} {id:?}");
                        }
                    }
                }
            }
            assert_eq!(seen.len(), layout.k * 256 * layout.w1_bits, "{set:?}");
        }
    }
    /// Every lane id of the stage-1 and stage-2 specs reads the same cells from the packed
    /// words as from the expanded `i8` tables, for 8 and 64 records of every set.
    #[test]
    fn fp_bits_lanes_match_i8() {
        let mut rng = SplitMix64::new(0xB175_1A9E);
        for set in Set::ALL {
            for n in [8, 64] {
                let manifest = CommitmentManifest::for_set_fp(set, n, Relation::default()).unwrap();
                let layout = KeccakLayout::for_set(set);
                let n_star = manifest.padded_records();
                let records: Vec<([u8; TR_BYTES], Vec<u8>)> = (0..n)
                    .map(|_| {
                        let mu: [u8; TR_BYTES] = std::array::from_fn(|_| rng.next_u64() as u8);
                        (mu, (0..layout.w1_encode_bytes).map(|_| rng.next_u64() as u8).collect())
                    })
                    .collect();
                let witness = std::sync::Arc::new(KeccakTables::build(&layout, &records, n_star));
                let publics = KeccakPublics::from_records(&layout, &records, n_star);
                let wide = KeccakProverTables::build_shared(&witness, &publics, false);
                let packed = KeccakProverTables::build_shared(&witness, &publics, true);
                let ids: HashSet<LaneId> = [stage1_groups(&manifest, 0).0, stage2_groups(&manifest, 0).0]
                    .iter()
                    .flatten()
                    .flat_map(|g| g.lanes.iter().copied())
                    .collect();
                let mut packed_lanes = 0;
                for id in ids {
                    let (a, b) = (wide.lane(id).expect("Keccak lane"), packed.lane(id).expect("Keccak lane"));
                    packed_lanes += usize::from(matches!(b, crate::kernel::Lane::Source(crate::kernel::Source::Bits(_))));
                    assert_eq!(a.len(), b.len(), "{set:?} {n} {id:?}");
                    assert_eq!(a.len(), 1024 * n_star);
                    for cell in 0..a.len() {
                        assert_eq!(a.value(cell), b.value(cell), "{set:?} {n} {id:?} cell {cell}");
                    }
                }
                assert!(packed_lanes > 0, "{set:?} {n}: the packed source serves the committed lanes");
            }
        }
    }
}

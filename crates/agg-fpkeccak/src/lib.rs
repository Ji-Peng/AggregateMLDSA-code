//! Prime-field (`F_p`) Keccak-f[1600] witness layout, bit-packed trace tables, `i8`
//! expansion, and Boolean-domain constraint evaluator for ML-DSA commitment-hash chains.

// Index loops mirror the FIPS 202 Keccak-f[1600] pseudocode and index multiple arrays at once.
#![allow(clippy::needless_range_loop)]

pub mod constraints;
pub mod layout;
pub mod trace;

pub use constraints::{
    evaluate, evaluate_all, xor2, xor3, ConstraintFamily, KeccakPublics, RecordPublic, Violation,
};
pub use layout::{
    lane, lane_xy, pi, pi_inv, rot, slot_of, state_bit, state_bit_xyz, time, Absorbed,
    KeccakLayout, PolyFamily, SlotMasks, COLS, KECCAK_GROUPS, LANES, LANE_BITS, LANE_BITS_LOG2,
    RC, RECORD_CELLS_PER_POLY, ROT, ROUNDS_PER_PERM, SLOTS, SLOTS_LOG2, STATE_BITS,
    TOTAL_LANE_FAMILIES,
};
pub use trace::{rotated, KeccakTables, RecordTrace};

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use agg_witness::gen::SplitMix64;
    use agg_witness::keccak::{sponge_blocks_for, RATE, RATE_BITS};
    use agg_witness::params::{dispatch_set, Set, N, TR_BYTES};
    use sha3::digest::{ExtendableOutput, Update, XofReader};

    use super::*;

    fn shake256_ref(mu: &[u8; TR_BYTES], w1_bytes: &[u8], out_len: usize) -> Vec<u8> {
        let mut h = sha3::Shake256::default();
        h.update(mu);
        h.update(w1_bytes);
        let mut out = vec![0u8; out_len];
        h.finalize_xof().read(&mut out);
        out
    }

    fn random_records(
        rng: &mut SplitMix64,
        layout: &KeccakLayout,
        count: usize,
    ) -> Vec<([u8; TR_BYTES], Vec<u8>)> {
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let mut mu = [0u8; TR_BYTES];
            let mut w1 = vec![0u8; layout.w1_encode_bytes];
            rng.fill(&mut mu);
            rng.fill(&mut w1);
            records.push((mu, w1));
        }
        records
    }

    #[test]
    fn layout_geometry_and_masks_all_sets() {
        for set in Set::ALL {
            let layout = KeccakLayout::for_set(set);
            match set {
                Set::S44 => {
                    assert_eq!(layout.perms, 7);
                    assert_eq!(layout.slots, 16);
                    assert_eq!(layout.slot_groups, 11);
                    assert_eq!(layout.rounds, 168);
                    assert_eq!(layout.output_lanes, 4);
                    assert_eq!(layout.c_tilde_bytes, 32);
                    assert_eq!(layout.w1_encode_bytes, 768);
                }
                Set::S65 => {
                    assert_eq!(layout.perms, 7);
                    assert_eq!(layout.slots, 16);
                    assert_eq!(layout.slot_groups, 11);
                    assert_eq!(layout.rounds, 168);
                    assert_eq!(layout.output_lanes, 6);
                    assert_eq!(layout.c_tilde_bytes, 48);
                    assert_eq!(layout.w1_encode_bytes, 768);
                }
                Set::S87 => {
                    assert_eq!(layout.perms, 9);
                    assert_eq!(layout.slots, 16);
                    assert_eq!(layout.slot_groups, 14);
                    assert_eq!(layout.rounds, 216);
                    assert_eq!(layout.output_lanes, 8);
                    assert_eq!(layout.c_tilde_bytes, 64);
                    assert_eq!(layout.w1_encode_bytes, 1024);
                }
            }

            // Check pi and pi_inv are exact inverses on 0..5 x 0..5.
            for y in 0..COLS {
                for x in 0..COLS {
                    let (px, py) = layout.pi(x, y);
                    assert_eq!(layout.pi_inv(px, py), (x, y));
                }
            }

            // Check masks across all (lane, q) and slots s.
            let mut live_count = 0usize;
            let mut inner_count = 0usize;
            let mut boundary_count = 0usize;
            let mut output_count = 0usize;
            for q in 0..layout.slot_groups {
                let m = layout.masks(0, q);
                for l in 1..LANES {
                    assert_eq!(layout.masks(l, q), m);
                }
                assert_eq!(m.inner | m.boundary, m.live);
                assert_eq!(m.inner & m.boundary, 0);
                for s in 0..SLOTS {
                    let t = layout.time(q, s);
                    assert_eq!(layout.slot_of(t), (q, s));
                    assert_eq!(m.is_live(s), t < layout.rounds);
                    assert_eq!(
                        m.is_boundary(s),
                        t > 0 && t < layout.rounds && t.is_multiple_of(ROUNDS_PER_PERM)
                    );
                    assert_eq!(m.is_inner(s), m.is_live(s) && !m.is_boundary(s));
                    assert_eq!(m.is_output(s), t + 1 == layout.rounds);
                    if m.is_live(s) {
                        live_count += 1;
                    }
                    if m.is_inner(s) {
                        inner_count += 1;
                    }
                    if m.is_boundary(s) {
                        boundary_count += 1;
                    }
                    if m.is_output(s) {
                        output_count += 1;
                    }
                }
            }
            assert_eq!(live_count, layout.rounds);
            assert_eq!(boundary_count, layout.perms - 1);
            assert_eq!(inner_count, layout.rounds - (layout.perms - 1));
            assert_eq!(output_count, 1);
        }
    }

    // (a) For 44, 65, 87 and 16 random records, evaluate returns no violation and the derived
    // c~ equals sha3::Shake256(mu || w1_bytes) of the right length.
    #[test]
    fn honest_records_evaluate_clean_and_match_shake256() {
        let mut rng = SplitMix64::new(0xA11_7E57);
        for set in Set::ALL {
            let layout = KeccakLayout::for_set(set);
            let records = random_records(&mut rng, &layout, 16);
            let tables = KeccakTables::build(&layout, &records, 16);
            let publics = KeccakPublics::from_records(&layout, &records, 16);

            let violations = evaluate(&tables, &publics);
            assert!(
                violations.is_empty(),
                "set {set}: expected 0 violations on 16 honest records, got {:?}",
                &violations[..violations.len().min(8)]
            );

            for (i, (mu, w1)) in records.iter().enumerate() {
                let expected = shake256_ref(mu, w1, layout.c_tilde_bytes);
                assert_eq!(tables.c_tilde(i), expected.as_slice());
                assert_eq!(publics.records()[i].c_tilde, expected);
            }
        }
    }

    // (b) Flipping one A' bit at a random (lane, t, z) produces violations only in the expected
    // families (bool_a or link/parity of that round and the link of round t+1) and never zero
    // violations; flipping a C bit, a C' bit, a b_t input bit and a c~ output bit likewise.
    #[test]
    fn bit_flips_and_non_boolean_cells_produce_exact_expected_violations() {
        let mut rng = SplitMix64::new(0xFA11_C0DE);
        for set in Set::ALL {
            let layout = KeccakLayout::for_set(set);
            let records = random_records(&mut rng, &layout, 4);
            let base_tables = KeccakTables::build(&layout, &records, 4);
            let base_publics = KeccakPublics::from_records(&layout, &records, 4);

            // 1. Flip one A' bit at random live (lane, t, z, record).
            for trial in 0..24 {
                let rec = rng.below(4) as usize;
                let lane_idx = rng.below(LANES as u64) as usize;
                let (x, y) = lane_xy(lane_idx);
                // Cover t = 0, inner rounds, boundary rounds, and the final round.
                let t = match trial {
                    0 => 0,
                    1 => ROUNDS_PER_PERM,
                    2 => layout.rounds - 1,
                    _ => rng.below(layout.rounds as u64) as usize,
                };
                let z = rng.below(LANE_BITS as u64) as usize;

                let mut tables = base_tables.clone();
                tables.flip_a_prime_bit(lane_idx, t, z, rec);
                let v = evaluate(&tables, &base_publics);
                assert!(!v.is_empty(), "A' flip at ({lane_idx}, {t}, {z}) had 0 violations");

                // Must include Parity(x, t, z, rec) and Link(lane_idx, t, z, rec).
                assert!(v.contains(&Violation {
                    family: ConstraintFamily::Parity,
                    lane: x,
                    t,
                    z,
                    record: rec,
                }));
                assert!(v.contains(&Violation {
                    family: ConstraintFamily::Link,
                    lane: lane_idx,
                    t,
                    z,
                    record: rec,
                }));

                let (bx, by) = layout.pi(x, y);
                let bz = (z + layout.rot(x, y) as usize) % LANE_BITS;
                if t + 1 < layout.rounds {
                    // Must include Link(lane(bx, by), t + 1, bz, rec).
                    assert!(v.contains(&Violation {
                        family: ConstraintFamily::Link,
                        lane: lane(bx, by),
                        t: t + 1,
                        z: bz,
                        record: rec,
                    }));
                }

                for viol in &v {
                    assert_eq!(viol.record, rec);
                    match viol.family {
                        ConstraintFamily::Parity => {
                            assert_eq!((viol.lane, viol.t, viol.z), (x, t, z));
                        }
                        ConstraintFamily::Link => {
                            if viol.t == t {
                                assert_eq!((viol.lane, viol.z), (lane_idx, z));
                            } else {
                                assert_eq!(viol.t, t + 1);
                                assert_eq!(viol.z, bz);
                                let (_vx, vy) = lane_xy(viol.lane);
                                assert_eq!(vy, by);
                            }
                        }
                        ConstraintFamily::Out => {
                            assert_eq!(t, layout.rounds - 1);
                            assert_eq!(viol.t, layout.rounds - 1);
                            assert_eq!(viol.z, bz);
                        }
                        other => panic!("unexpected family {other:?} on A' bit flip"),
                    }
                }
            }

            // 2. Non-Boolean A' cell override triggers BoolA at that exact cell.
            {
                let rec = rng.below(4) as usize;
                let lane_idx = rng.below(LANES as u64) as usize;
                let t = rng.below(layout.rounds as u64) as usize;
                let z = rng.below(LANE_BITS as u64) as usize;
                let mut tables = base_tables.clone();
                tables.set_a_prime_cell(lane_idx, t, z, rec, 2);
                let v = evaluate(&tables, &base_publics);
                let bool_a_viols: Vec<_> = v
                    .iter()
                    .copied()
                    .filter(|x| x.family == ConstraintFamily::BoolA)
                    .collect();
                assert_eq!(
                    bool_a_viols,
                    vec![Violation {
                        family: ConstraintFamily::BoolA,
                        lane: lane_idx,
                        t,
                        z,
                        record: rec,
                    }]
                );
            }

            // 3. Flip one C bit at random live (x, t, z, record).
            for _ in 0..16 {
                let rec = rng.below(4) as usize;
                let x = rng.below(COLS as u64) as usize;
                let t = rng.below(layout.rounds as u64) as usize;
                let z = rng.below(LANE_BITS as u64) as usize;

                let mut tables = base_tables.clone();
                tables.flip_c_bit(x, t, z, rec);
                let v = evaluate(&tables, &base_publics);
                // Exactly 3 ThetaC violations and 5 Link violations in round t.
                assert_eq!(v.len(), 8);
                let mut expected = BTreeSet::new();
                expected.insert(Violation {
                    family: ConstraintFamily::ThetaC,
                    lane: x,
                    t,
                    z,
                    record: rec,
                });
                expected.insert(Violation {
                    family: ConstraintFamily::ThetaC,
                    lane: (x + 1) % COLS,
                    t,
                    z,
                    record: rec,
                });
                expected.insert(Violation {
                    family: ConstraintFamily::ThetaC,
                    lane: (x + COLS - 1) % COLS,
                    t,
                    z: (z + 1) % LANE_BITS,
                    record: rec,
                });
                for y in 0..COLS {
                    expected.insert(Violation {
                        family: ConstraintFamily::Link,
                        lane: lane(x, y),
                        t,
                        z,
                        record: rec,
                    });
                }
                let actual: BTreeSet<_> = v.into_iter().collect();
                assert_eq!(actual, expected);
            }

            // 4. Non-Boolean C cell override triggers BoolC at that exact cell.
            {
                let rec = rng.below(4) as usize;
                let x = rng.below(COLS as u64) as usize;
                let t = rng.below(layout.rounds as u64) as usize;
                let z = rng.below(LANE_BITS as u64) as usize;
                let mut tables = base_tables.clone();
                tables.set_c_cell(x, t, z, rec, -1);
                let v = evaluate(&tables, &base_publics);
                let bool_c_viols: Vec<_> = v
                    .iter()
                    .copied()
                    .filter(|v| v.family == ConstraintFamily::BoolC)
                    .collect();
                assert_eq!(
                    bool_c_viols,
                    vec![Violation {
                        family: ConstraintFamily::BoolC,
                        lane: x,
                        t,
                        z,
                        record: rec,
                    }]
                );
            }

            // 5. Flip one C' bit at random live (x, t, z, record).
            for _ in 0..16 {
                let rec = rng.below(4) as usize;
                let x = rng.below(COLS as u64) as usize;
                let t = rng.below(layout.rounds as u64) as usize;
                let z = rng.below(LANE_BITS as u64) as usize;

                let mut tables = base_tables.clone();
                tables.flip_c_prime_bit(x, t, z, rec);
                let v = evaluate(&tables, &base_publics);
                // Exactly 1 ThetaC + 1 Parity + 5 Link = 7 violations in round t.
                assert_eq!(v.len(), 7);
                let mut expected = BTreeSet::new();
                expected.insert(Violation {
                    family: ConstraintFamily::ThetaC,
                    lane: x,
                    t,
                    z,
                    record: rec,
                });
                expected.insert(Violation {
                    family: ConstraintFamily::Parity,
                    lane: x,
                    t,
                    z,
                    record: rec,
                });
                for y in 0..COLS {
                    expected.insert(Violation {
                        family: ConstraintFamily::Link,
                        lane: lane(x, y),
                        t,
                        z,
                        record: rec,
                    });
                }
                let actual: BTreeSet<_> = v.into_iter().collect();
                assert_eq!(actual, expected);
            }

            // 6. Flip one b_t input bit at random (t_bit, j, rho, record).
            for _ in 0..16 {
                let rec = rng.below(4) as usize;
                let t_bit = rng.below(layout.w1_bits as u64) as usize;
                let j = rng.below(N as u64) as usize;
                let rho = rng.below(layout.k as u64) as usize;

                let mut publics = base_publics.clone();
                publics.flip_w1_bit(rec, t_bit, j, rho);
                let v = evaluate(&base_tables, &publics);

                let slot = dispatch_set!(set, |P| {
                    agg_witness::keccak::w1_bit_slot_for::<P>(t_bit, j, rho)
                });
                let (x, y, z) = state_bit_xyz(slot.state_bit);
                assert_eq!(
                    v,
                    vec![Violation {
                        family: ConstraintFamily::Link,
                        lane: lane(x, y),
                        t: slot.perm * ROUNDS_PER_PERM,
                        z,
                        record: rec,
                    }]
                );
            }

            // 7. Flip one c~ output bit at random (lane, z, record).
            for _ in 0..16 {
                let rec = rng.below(4) as usize;
                let out_lane = rng.below(layout.output_lanes as u64) as usize;
                let z = rng.below(LANE_BITS as u64) as usize;

                let mut publics = base_publics.clone();
                publics.flip_c_tilde_bit(rec, out_lane, z);
                let v = evaluate(&base_tables, &publics);
                assert_eq!(
                    v,
                    vec![Violation {
                        family: ConstraintFamily::Out,
                        lane: out_lane,
                        t: layout.rounds - 1,
                        z,
                        record: rec,
                    }]
                );
            }
        }
    }

    // (c) expand_i8 of a packed word matches bit extraction.
    #[test]
    fn expand_i8_matches_bit_extraction_and_group_ranges() {
        let mut rng = SplitMix64::new(0xE89A_4D18);
        for set in Set::ALL {
            let layout = KeccakLayout::for_set(set);
            let records = random_records(&mut rng, &layout, 5);
            let n_star = 8;
            let tables = KeccakTables::build(&layout, &records, n_star);

            for g in 0..KECCAK_GROUPS {
                let range = layout.group_poly_range(g);
                let expanded = tables.expand_i8(range.clone());
                assert_eq!(expanded.len(), range.len() * RECORD_CELLS_PER_POLY * n_star);

                for (offset, poly) in range.enumerate() {
                    let poly_base = offset * RECORD_CELLS_PER_POLY * n_star;
                    for i in 0..n_star {
                        for s in 0..SLOTS {
                            let word = tables.word(poly, s, i);
                            let t = layout.time(poly % layout.slot_groups, s);
                            if t >= layout.rounds {
                                assert_eq!(word, 0, "dead slot must be 0");
                            }
                            for z in 0..LANE_BITS {
                                let cell = z + 64 * s + 1024 * i;
                                let expected_bit = ((word >> z) & 1) as i8;
                                assert_eq!(expanded[poly_base + cell], expected_bit);
                            }
                        }
                    }
                }
            }

            // Also verify rotated(word, rot) bit correspondence.
            for _ in 0..64 {
                let w = rng.next_u64();
                let r = rng.below(64) as u32;
                let rw = rotated(w, r);
                for z in 0..64 {
                    let src_z = (z + 64 - (r as usize % 64)) % 64;
                    assert_eq!((rw >> z) & 1, (w >> src_z) & 1);
                }
            }
        }
    }

    // (d) Padded records evaluate clean with the zero-message c~.
    #[test]
    fn padded_records_evaluate_clean_with_zero_message_c_tilde() {
        let mut rng = SplitMix64::new(0x9AD0_2EC0);
        for set in Set::ALL {
            let layout = KeccakLayout::for_set(set);
            let live = 5;
            let n_star = 8;
            let records = random_records(&mut rng, &layout, live);
            let tables = KeccakTables::build(&layout, &records, n_star);
            let publics = KeccakPublics::from_records(&layout, &records, n_star);

            let violations = evaluate(&tables, &publics);
            assert!(
                violations.is_empty(),
                "set {set}: padded batch had violations: {:?}",
                violations
            );

            let zero_mu = [0u8; TR_BYTES];
            let zero_w1 = vec![0u8; layout.w1_encode_bytes];
            let expected_zero_c_tilde = shake256_ref(&zero_mu, &zero_w1, layout.c_tilde_bytes);

            for i in live..n_star {
                assert_eq!(tables.c_tilde(i), expected_zero_c_tilde.as_slice());
                assert_eq!(publics.records()[i].c_tilde, expected_zero_c_tilde);
            }
        }
    }

    // (e) The wiring is a bijection from the 6,144 (44, 65) or 8,192 (87) w1 bits onto block
    // positions and matches sponge_blocks_for.
    #[test]
    fn wiring_is_bijection_and_matches_sponge_blocks_for() {
        let mut rng = SplitMix64::new(0xB1EC_7104);
        for set in Set::ALL {
            let layout = KeccakLayout::for_set(set);
            let expected_w1_bits = layout.k * N * layout.w1_bits;
            assert_eq!(
                expected_w1_bits,
                match set {
                    Set::S44 | Set::S65 => 6144,
                    Set::S87 => 8192,
                }
            );

            let mut seen_w1 = BTreeSet::new();
            let mut seen_mu = BTreeSet::new();
            let mut w1_positions = 0usize;
            let mut mu_positions = 0usize;

            for m in 0..layout.perms {
                for y in 0..COLS {
                    for x in 0..COLS {
                        for z in 0..LANE_BITS {
                            let st_bit = state_bit(x, y, z);
                            let w = layout.wire(m, x, y, z);
                            if st_bit >= RATE_BITS {
                                assert_eq!(
                                    w,
                                    Absorbed::Zero,
                                    "capacity bit {st_bit} in perm {m} must be Zero"
                                );
                            }
                            match w {
                                Absorbed::W1Bit { t, j, rho } => {
                                    assert!(t < layout.w1_bits && j < N && rho < layout.k);
                                    assert!(
                                        seen_w1.insert((t, j, rho)),
                                        "duplicate W1Bit ({t}, {j}, {rho})"
                                    );
                                    w1_positions += 1;
                                }
                                Absorbed::Mu(k) => {
                                    assert_eq!(m, 0);
                                    assert!(k < TR_BYTES * 8);
                                    assert!(seen_mu.insert(k), "duplicate Mu({k})");
                                    mu_positions += 1;
                                }
                                Absorbed::Pad(bit) => {
                                    assert_eq!(m, layout.perms - 1);
                                    assert!(st_bit < RATE_BITS);
                                    assert!(bit <= 1);
                                }
                                Absorbed::Zero => {}
                            }
                        }
                    }
                }
            }

            assert_eq!(w1_positions, expected_w1_bits);
            assert_eq!(seen_w1.len(), expected_w1_bits);
            assert_eq!(mu_positions, TR_BYTES * 8);
            assert_eq!(seen_mu.len(), TR_BYTES * 8);

            // Reconstruct sponge_blocks_for from layout.wire for random inputs.
            for _ in 0..8 {
                let mut mu = [0u8; TR_BYTES];
                let mut w1_bytes = vec![0u8; layout.w1_encode_bytes];
                rng.fill(&mut mu);
                rng.fill(&mut w1_bytes);

                let expected_blocks =
                    dispatch_set!(set, |P| sponge_blocks_for::<P>(&mu, &w1_bytes));
                let mut rebuilt = vec![[0u8; RATE]; layout.perms];

                for m in 0..layout.perms {
                    for y in 0..COLS {
                        for x in 0..COLS {
                            for z in 0..LANE_BITS {
                                let st_bit = state_bit(x, y, z);
                                let bit = layout
                                    .wire(m, x, y, z)
                                    .eval(layout.w1_bits, &mu, &w1_bytes);
                                if st_bit < RATE_BITS {
                                    rebuilt[m][st_bit / 8] |= bit << (st_bit % 8);
                                } else {
                                    assert_eq!(bit, 0);
                                }
                            }
                        }
                    }
                }

                assert_eq!(rebuilt, expected_blocks);
            }
        }
    }

    /// The honest trace of real signatures: `mu` from the public instance, `w1Encode(w1')`
    /// from the derived witness, and the public `c~` taken from the signature itself (not
    /// recomputed), so a clean evaluation shows the output lanes equal the signed `c~`.
    fn real_records_evaluate_clean<P: agg_witness::ParamSet>(seed: u64) {
        use agg_witness::derive::derive_for;
        use agg_witness::gen::records_for;
        use agg_witness::instance::PublicInstance;

        let layout = KeccakLayout::for_set(P::SET);
        let (live, n_star) = (3, 4);
        let records = records_for::<P>(seed, live);
        let instance = PublicInstance::<P>::from_records_for(&records).expect("instance");
        let witnesses: Vec<([u8; TR_BYTES], Vec<u8>)> = instance
            .records
            .iter()
            .zip(&records)
            .map(|(p, r)| {
                let w = derive_for::<P>(&p.key, &p.mu, &r.sig).expect("the signature is accepted");
                assert_eq!(w.mu, p.mu);
                (w.mu, w.w1_bytes.as_ref().to_vec())
            })
            .collect();
        let tables = KeccakTables::build(&layout, &witnesses, n_star);
        let mut publics = KeccakPublics::from_records(&layout, &witnesses, n_star);
        for (i, r) in records.iter().enumerate() {
            let signed = &r.sig[..layout.c_tilde_bytes];
            assert_eq!(tables.c_tilde(i), signed, "{}: record {i}", P::SET);
            publics.records_mut()[i].c_tilde = signed.to_vec();
        }
        let violations = evaluate(&tables, &publics);
        assert!(violations.is_empty(), "{}: {:?}", P::SET, &violations[..violations.len().min(8)]);

        // A changed signed c~ byte is caught by the output constraint only.
        publics.records_mut()[1].c_tilde[layout.c_tilde_bytes - 1] ^= 0x10;
        let violations = evaluate(&tables, &publics);
        assert_eq!(violations.len(), 1, "{}", P::SET);
        let v = violations[0];
        assert_eq!((v.family, v.lane, v.z, v.record), (ConstraintFamily::Out, layout.output_lanes - 1, 60, 1));
    }

    #[test]
    fn real_records_of_every_set_evaluate_clean_against_the_signed_c_tilde() {
        real_records_evaluate_clean::<agg_witness::Set44>(0x44);
        real_records_evaluate_clean::<agg_witness::Set65>(0x65);
        real_records_evaluate_clean::<agg_witness::Set87>(0x87);
    }
}

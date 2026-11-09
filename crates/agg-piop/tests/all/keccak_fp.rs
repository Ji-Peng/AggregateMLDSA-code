//! Commit and batched-opening round trip of the seven `HashField::Fp` Keccak groups
//! (`KeccakA(0..4)`, `KeccakC`, `KeccakCPrime`) together with the lattice groups
//! (`G_main16`, `G_main4`, `G_k`, `G_m`) for ML-DSA-44 at `N = 64` and `N = 1024`.

use crate::common::{self, mle_i64, point, Batch};

use agg_fpkeccak::{evaluate, KeccakLayout, KeccakPublics, KeccakTables, KECCAK_GROUPS};
use agg_pcs_config::commit::{evaluate_signed_table, GroupClaim, GroupCommitment};
use agg_piop::claims::{PreStageClaims, Stage1Claims};
use agg_piop::keccak_fp::specs::{stage1_groups, stage2_groups};
use agg_piop::keccak_fp::virtuals::{zero_message_c_tilde, KeccakProverTables};
use agg_piop::kernel::{Lane, Source, Term};
use agg_piop::manifest::{Column, CommitmentManifest, DigitValue, GroupRole, GROUP_COUNT_FP};
use agg_piop::pcs::{group_points_fp, verifier_for, TableCommitter};
use agg_piop::source_tables::SourceTables;
use agg_piop::stage::{
    check_terminal, prove_stage, prove_stage_unchecked, replay_stage, EqPoint, GroupSpec,
    InstanceSpec, Points, StageError, StageSpec,
};
use agg_piop::stage1::{keccak_claim_lanes, resolve_keccak, KeccakPublicInputs};
use agg_piop::tables::ProverTables;
use agg_piop::transcript::{
    commitment_site, terminal_site, Transcript, BRIDGE_SEED_SITE, FINAL_VECTORS_SITE,
    FINAL_VECTOR_BYTES, FLOCK_COMMIT_SITE, FLOCK_SEEDS_SITE, FLOCK_SHARES_SITE, KAPPA1_SITE,
    PRESTAGE_SITE, TAU_SITE,
};
use agg_piop::virtual_polys::{compose_table, w1_table, MainValues};
use agg_piop::{stage1, stage2, BaseField, OpeningField};
use agg_witness::gen::SplitMix64;
use agg_witness::params::{Set, K, N as POLY_N, TR_BYTES, W1_BITS, W1_ENCODE_BYTES};
use jolt_field::{ExtField, Ring, Zero};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// Extract `(mu_i, w1Encode(w1'_i))` per live record from `batch`.
fn keccak_inputs_from_batch(batch: &Batch) -> Vec<([u8; TR_BYTES], Vec<u8>)> {
    let bit_lanes = batch.main.bit_lanes(&batch.manifest);
    let n = batch.records.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mu = batch.instance.records[i].mu;
        let mut w1_bytes = vec![0_u8; W1_ENCODE_BYTES];
        let rec_base = i * (K * POLY_N);
        for rho in 0..K {
            for j in 0..POLY_N {
                let cell = rec_base + rho * POLY_N + j;
                for t in 0..W1_BITS {
                    let bit = bit_lanes[t][cell] as u8;
                    debug_assert!(bit <= 1);
                    let bit_idx = (rho * POLY_N + j) * W1_BITS + t;
                    w1_bytes[bit_idx / 8] |= bit << (bit_idx % 8);
                }
            }
        }
        out.push((mu, w1_bytes));
    }
    out
}

fn check_main_virtual_claims(
    batch: &Batch,
    fp_manifest: &CommitmentManifest,
    main_point: &[OpeningField],
    values: &[Vec<OpeningField>],
) {
    let tables = batch.tables();
    let main = MainValues::from_claims(fp_manifest, |column: Column| {
        let (role, position) = fp_manifest.locate(column);
        values[role.index_in(fp_manifest.hash_field())][position]
    });
    let composed = |value: DigitValue| compose_table(&value.layout(), &tables.digits(value));
    assert_eq!(main.z, mle_i64(&composed(fp_manifest.z_value()), main_point));
    assert_eq!(main.u, mle_i64(&composed(DigitValue::U), main_point));
    assert_eq!(main.v, mle_i64(&composed(DigitValue::V), main_point));
    let w1 = w1_table(&batch.main.bit_lanes(&batch.manifest));
    assert_eq!(main.w1, mle_i64(&w1, main_point));
}

fn commit_and_open_fp(n_own: usize, n_shared: usize, seed: u64, check_helper_parity: bool) {
    let batch = Batch::build(common::records(seed, n_own, n_shared), seed);
    let n_records = batch.records.len();
    let manifest = CommitmentManifest::for_batch_fp(n_records).expect("fp manifest");
    let n_star = manifest.padded_records();

    let k_layout = KeccakLayout::for_set(Set::S44);
    let keccak_inputs = keccak_inputs_from_batch(&batch);
    let keccak_tables = KeccakTables::build(&k_layout, &keccak_inputs, n_star);
    let keccak_publics = KeccakPublics::from_records(&k_layout, &keccak_inputs, n_star);
    assert!(
        evaluate(&keccak_tables, &keccak_publics).is_empty(),
        "honest ML-DSA-44 records must satisfy all Fp Keccak constraints"
    );
    for i in 0..n_records {
        assert_eq!(
            keccak_tables.c_tilde(i),
            batch.instance.records[i].c_tilde.as_slice()
        );
    }

    let committer = TableCommitter::new(manifest.clone()).expect("fp committer setup");
    let [main16, main] = committer.commit_round1(&batch.main).expect("round 1 main");

    let mut rng = SplitMix64::new(seed ^ 0xF9_4ECC_A4);
    let r3 = point(&mut rng, manifest.d_main());
    let pi_m = point(&mut rng, 16);
    let points = group_points_fp(&manifest, &r3, &pi_m).expect("fp group points");

    // Expand and commit the seven Keccak groups one at a time via PcsProver::commit_group,
    // and evaluate every polynomial at its group's point with evaluate_signed_table.
    let mut keccak_commitments: Vec<GroupCommitment> = Vec::with_capacity(KECCAK_GROUPS);
    let mut keccak_claims: Vec<GroupClaim> = Vec::with_capacity(KECCAK_GROUPS);
    for g in 0..KECCAK_GROUPS {
        let role = GroupRole::KECCAK[g];
        let wire_idx = role.index_in(manifest.hash_field());
        let descriptor = manifest.group(role);
        let cells_per_poly = descriptor.logical_cells();
        let flat = keccak_tables.expand_i8(k_layout.group_poly_range(g));
        let group_tables: Vec<Vec<i8>> = flat
            .chunks_exact(cells_per_poly)
            .map(|chunk| chunk.to_vec())
            .collect();
        assert_eq!(group_tables.len(), descriptor.polynomial_count());

        let pt = &points[wire_idx];
        let values: Vec<OpeningField> = group_tables
            .iter()
            .map(|poly| evaluate_signed_table(poly, pt))
            .collect();
        keccak_claims.push(GroupClaim {
            point: pt.clone(),
            values,
        });

        let committed = committer
            .prover()
            .commit_group(role, &group_tables)
            .expect("commit Keccak group");
        keccak_commitments.push(committed);
    }

    if check_helper_parity {
        let via_helper = committer
            .commit_keccak(&keccak_tables)
            .expect("bounded commit_keccak");
        for (a, b) in keccak_commitments.iter().zip(&via_helper) {
            assert_eq!(
                a.serialized().expect("serialize"),
                b.serialized().expect("serialize"),
                "commit_keccak must match one-at-a-time commit_group"
            );
        }
    }

    let proj_cells = batch.bridge.group(&batch.manifest, GroupRole::Proj);
    let mult_cells = batch.bridge.group(&batch.manifest, GroupRole::Mult);
    let [proj, mult] = committer
        .commit_round2_fp(&proj_cells, &mult_cells)
        .expect("round 2 fp");

    // Assemble the 11 group commitments and claims in wire order:
    // G_main16, G_main4, KeccakA[0..4], KeccakC, KeccakCPrime, G_k, G_m.
    let eval_cells = |cells_list: &[agg_piop::source_tables::Cells<'_>], pt: &[OpeningField]| {
        GroupClaim {
            point: pt.to_vec(),
            values: cells_list.iter().map(|c| c.evaluate(pt)).collect(),
        }
    };
    let mut claims = Vec::with_capacity(GROUP_COUNT_FP);
    claims.push(eval_cells(&batch.main.wide_columns(), &points[0]));
    let main_narrow: Vec<_> = batch
        .main
        .columns()
        .iter()
        .map(|c| agg_piop::source_tables::Cells::Narrow(c))
        .collect();
    claims.push(eval_cells(&main_narrow, &points[1]));
    claims.extend(keccak_claims);
    claims.push(eval_cells(&proj_cells, &points[9]));
    claims.push(eval_cells(&mult_cells, &points[10]));
    assert_eq!(claims.len(), GROUP_COUNT_FP);
    assert_eq!(
        claims.iter().map(|c| c.values.len()).sum::<usize>(),
        manifest.total_polynomials()
    );

    let values: Vec<Vec<OpeningField>> = claims.iter().map(|c| c.values.clone()).collect();
    check_main_virtual_claims(&batch, &manifest, &points[0], &values);

    let mut all_commitments = Vec::with_capacity(GROUP_COUNT_FP);
    all_commitments.push(main16);
    all_commitments.push(main);
    all_commitments.extend(keccak_commitments);
    all_commitments.push(proj);
    all_commitments.push(mult);
    let public_commitments: Vec<_> = all_commitments
        .iter()
        .map(|c| c.commitment().clone())
        .collect();

    let session = [0x4b; 32];
    let proof = committer
        .open_claims(all_commitments, &claims, &session)
        .expect("batched fp opening");
    eprintln!(
        "Fp N = {n_records} (N* = {n_star}): {} claims across {} groups, opening proof {} bytes",
        claims.iter().map(|c| c.values.len()).sum::<usize>(),
        claims.len(),
        proof.len()
    );

    let verifier = verifier_for(&manifest).expect("fp verifier setup");
    verifier
        .verify(&public_commitments, &claims, &proof, &session)
        .expect("seed-derived verifier accepts fp opening");
    committer
        .verifier()
        .expect("prover verifier")
        .verify(&public_commitments, &claims, &proof, &session)
        .expect("prover-derived verifier accepts fp opening");

    // Negative tests: altering a claim in any group is rejected.
    for group_idx in [0, 1, 2, 5, 7, 8, 9, 10] {
        let mut wrong = claims.clone();
        wrong[group_idx].values[0] += OpeningField::lift_base(BaseField::from_u64(1));
        assert!(
            verifier
                .verify(&public_commitments, &wrong, &proof, &session)
                .is_err(),
            "altered claim in group {group_idx} must be rejected"
        );
    }
    assert!(verifier
        .verify(&public_commitments, &claims, &proof, &[0xb4; 32])
        .is_err());
    let mut swapped = public_commitments.clone();
    swapped.swap(2, 7);
    assert!(verifier
        .verify(&swapped, &claims, &proof, &session)
        .is_err());
}

#[test]
fn fp_keccak_and_lattice_groups_commit_and_open_at_64() {
    on_large_stack(|| commit_and_open_fp(50, 14, 0x64, true));
}

#[test]
fn fp_keccak_and_lattice_groups_commit_and_open_at_1024() {
    on_large_stack(|| commit_and_open_fp(900, 124, 0x400, false));
}

fn lane_cell_i64(lane: &Lane<'_>, cell: usize) -> i64 {
    match lane {
        Lane::Source(Source::Bytes { cells, map }) => map[cells[cell] as u8 as usize],
        Lane::Source(Source::Ints(cells)) => i64::from(cells[cell]),
        other => panic!("unexpected non-integer Keccak lane: {other:?}"),
    }
}

fn eval_term_i64(term: &Term, vals: &[i64]) -> i64 {
    match term {
        Term::Product(lanes) => lanes.iter().fold(1_i64, |acc, &l| acc * vals[l]),
        Term::Bool(l) => vals[*l] * (vals[*l] - 1),
        Term::Affine {
            weight,
            lanes,
            constant,
        } => {
            let affine = lanes
                .iter()
                .fold(*constant, |acc, &(l, c)| acc + c * vals[l]);
            match weight {
                Some(w) => vals[*w] * affine,
                None => affine,
            }
        }
        Term::Monomials(monomials) => monomials.iter().fold(0_i64, |acc, (c, m)| {
            acc + *c * m.iter().fold(1_i64, |p, &l| p * vals[l])
        }),
        other => panic!("unsupported term in Keccak spec: {other:?}"),
    }
}

fn count_nonzero_terms(groups: &[GroupSpec], prover_tables: &KeccakProverTables) -> usize {
    let n_star = prover_tables.n_star();
    let mut nonzero = 0;
    for group in groups {
        let mask = group.eq.mask().unwrap_or(0xFFFF);
        let lanes: Vec<Lane<'_>> = group
            .lanes
            .iter()
            .map(|&id| prover_tables.lane(id).expect("Keccak lane"))
            .collect();
        let mut vals = vec![0_i64; lanes.len()];
        for i in 0..n_star {
            for s in 0..16 {
                if (mask >> s) & 1 == 0 {
                    continue;
                }
                let base = (i * 16 + s) * 64;
                for z in 0..64 {
                    let cell = base + z;
                    for (dst, lane) in vals.iter_mut().zip(&lanes) {
                        *dst = lane_cell_i64(lane, cell);
                    }
                    for instance in &group.instances {
                        for term in &instance.terms {
                            if eval_term_i64(term, &vals) != 0 {
                                nonzero += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    nonzero
}

#[test]
fn keccak_specs_count_and_degree() {
    let manifest = CommitmentManifest::for_batch_fp(8).expect("fp manifest");
    let keccak_domain = Column::KeccakA { x: 0, y: 0, q: 0 }.domain();

    let gamma0_s1 = 7;
    let (s1_groups, s1_used) = stage1_groups(&manifest, gamma0_s1);
    assert_eq!(s1_used, 719);
    let mut s1_seen = vec![0_usize; s1_used];
    for group in &s1_groups {
        assert_eq!(group.domain, keccak_domain);
        assert_ne!(group.eq.mask(), Some(0));
        assert_ne!(group.eq, EqPoint::TauMasked(0xFFFF));
        for &lane in &group.lanes {
            assert_eq!(lane.domain(), keccak_domain);
        }
        for instance in &group.instances {
            assert!(
                instance.gamma >= gamma0_s1 && instance.gamma < gamma0_s1 + s1_used,
                "stage-1 gamma {} outside [{}..{})",
                instance.gamma,
                gamma0_s1,
                gamma0_s1 + s1_used
            );
            s1_seen[instance.gamma - gamma0_s1] += 1;
            for term in &instance.terms {
                assert!(term.degree() <= 3, "stage-1 term degree {} > 3", term.degree());
            }
        }
    }
    assert!(s1_seen.iter().all(|&c| c == 1 || c == 2));
    assert_eq!(s1_seen.iter().filter(|&&c| c == 2).count(), 25);

    let (s1_zero, s1_count) = stage1_groups(&manifest, 0);
    StageSpec {
        stage: 1,
        degree: 4,
        gamma_count: s1_count,
        groups: s1_zero,
    }
    .validate()
    .expect("stage-1 Keccak spec validates");

    let gamma0_s2 = 11;
    let (s2_groups, s2_used) = stage2_groups(&manifest, gamma0_s2);
    assert_eq!(s2_used, 150);
    let mut s2_seen = vec![0_usize; s2_used];
    for group in &s2_groups {
        assert_eq!(group.domain, keccak_domain);
        assert_ne!(group.eq.mask(), Some(0));
        assert_ne!(group.eq, EqPoint::R1Masked(0xFFFF));
        for &lane in &group.lanes {
            assert_eq!(lane.domain(), keccak_domain);
        }
        for instance in &group.instances {
            assert!(
                instance.gamma >= gamma0_s2 && instance.gamma < gamma0_s2 + s2_used,
                "stage-2 gamma {} outside [{}..{})",
                instance.gamma,
                gamma0_s2,
                gamma0_s2 + s2_used
            );
            s2_seen[instance.gamma - gamma0_s2] += 1;
            for term in &instance.terms {
                assert!(term.degree() <= 4, "stage-2 term degree {} > 4", term.degree());
            }
        }
    }
    assert!(s2_seen.iter().all(|&c| c == 1));

    let (s2_zero, s2_count) = stage2_groups(&manifest, 0);
    StageSpec {
        stage: 2,
        degree: 5,
        gamma_count: s2_count,
        groups: s2_zero,
    }
    .validate()
    .expect("stage-2 Keccak spec validates");
}

#[test]
fn keccak_specs_vanish_on_the_boolean_domain() {
    let k_layout = KeccakLayout::for_set(Set::S44);
    let zero_ct = zero_message_c_tilde(&k_layout);

    for (n_own, n_shared, seed) in [(6, 2, 0xA11_u64), (4, 1, 0xB22_u64)] {
        let batch = Batch::build(common::records(seed, n_own, n_shared), seed);
        let n_records = batch.records.len();
        let manifest = CommitmentManifest::for_batch_fp(n_records).expect("fp manifest");
        let n_star = manifest.padded_records();
        let (s1_groups, _) = stage1_groups(&manifest, 0);
        let (s2_groups, _) = stage2_groups(&manifest, 0);

        let inputs = keccak_inputs_from_batch(&batch);
        let keccak_tables = KeccakTables::build(&k_layout, &inputs, n_star);
        let keccak_publics = KeccakPublics::from_records(&k_layout, &inputs, n_star);
        for i in n_records..n_star {
            assert_eq!(keccak_tables.c_tilde(i), zero_ct.as_slice());
        }

        let prover_tables = KeccakProverTables::build(&keccak_tables, &keccak_publics);
        assert_eq!(count_nonzero_terms(&s1_groups, &prover_tables), 0);
        assert_eq!(count_nonzero_terms(&s2_groups, &prover_tables), 0);

        if n_records == n_star {
            // Flip one A' bit at a live slot (t = 5 < 168): both the spec terms and
            // agg_fpkeccak::evaluate report violations.
            let mut live_flipped = keccak_tables.clone();
            live_flipped.flip_a_prime_bit(3, 5, 19, 1);
            assert!(!evaluate(&live_flipped, &keccak_publics).is_empty());
            let bad_prover = KeccakProverTables::build(&live_flipped, &keccak_publics);
            let bad_nonzero = count_nonzero_terms(&s1_groups, &bad_prover)
                + count_nonzero_terms(&s2_groups, &bad_prover);
            assert!(bad_nonzero > 0, "live-slot bit flip must make some instance nonzero");

            // Flip one A' bit in a dead slot (t = 168 >= 168): unconstrained, stays 0.
            let mut dead_flipped = keccak_tables.clone();
            dead_flipped.flip_a_prime_bit(3, 168, 19, 1);
            assert!(evaluate(&dead_flipped, &keccak_publics).is_empty());
            let dead_prover = KeccakProverTables::build(&dead_flipped, &keccak_publics);
            assert_eq!(count_nonzero_terms(&s1_groups, &dead_prover), 0);
            assert_eq!(count_nonzero_terms(&s2_groups, &dead_prover), 0);
        }
    }
}

fn stages_branch_transcript(manifest: &CommitmentManifest) -> (Transcript, Points) {
    let mut trunk = Transcript::new(manifest, &manifest.binding_bytes()).expect("transcript");
    for role in [GroupRole::Main16, GroupRole::Main] {
        trunk.absorb(commitment_site(role), 1, &[0]).expect("round 1");
    }
    trunk
        .absorb_items(FLOCK_COMMIT_SITE, &[vec![]])
        .expect("flock commit");
    trunk
        .challenge_bytes(BRIDGE_SEED_SITE, 1)
        .expect("bridge seed");
    trunk.challenge_psi().expect("psi");
    for role in [
        GroupRole::Level1,
        GroupRole::Level2,
        GroupRole::Level3,
        GroupRole::Proj,
        GroupRole::Mult,
    ] {
        trunk.absorb(commitment_site(role), 1, &[0]).expect("round 2");
    }
    trunk
        .challenge_bytes(FLOCK_SEEDS_SITE, 1)
        .expect("flock seeds");
    trunk
        .absorb(FINAL_VECTORS_SITE, 1, &[0; FINAL_VECTOR_BYTES])
        .expect("final vectors");
    trunk
        .absorb_items(FLOCK_SHARES_SITE, &[vec![]])
        .expect("flock shares");
    let (_lookup, mut stages) = trunk.fork().expect("fork");
    let tau = stages
        .challenge_extensions(TAU_SITE, manifest.d_stage())
        .expect("tau");
    stages
        .absorb_extensions(
            PRESTAGE_SITE,
            &vec![OpeningField::zero(); PreStageClaims::len(manifest)],
        )
        .expect("prestage");
    (
        stages,
        Points {
            tau,
            ..Points::default()
        },
    )
}

#[test]
fn keccak_stage12_groups_prove_and_replay() {
    on_large_stack(|| {
        let seed = 0x512_u64;
        let batch = Batch::build(common::records(seed, 6, 2), seed);
        let n_records = batch.records.len();
        assert_eq!(n_records, 8);
        let manifest = CommitmentManifest::for_batch_fp(n_records).expect("fp manifest");
        let n_star = manifest.padded_records();
        assert_eq!(n_star, 8);

        let k_layout = KeccakLayout::for_set(Set::S44);
        let inputs = keccak_inputs_from_batch(&batch);
        let keccak_tables = KeccakTables::build(&k_layout, &inputs, n_star);
        let keccak_publics = KeccakPublics::from_records(&k_layout, &inputs, n_star);
        let prover_tables = KeccakProverTables::build(&keccak_tables, &keccak_publics);

        let mu_list: Vec<[u8; TR_BYTES]> = batch.instance.records.iter().map(|r| r.mu).collect();
        let c_tilde_list: Vec<Vec<u8>> = batch
            .instance
            .records
            .iter()
            .map(|r| r.c_tilde.to_vec())
            .collect();
        let public_inputs = KeccakPublicInputs {
            mu: &mu_list,
            c_tilde: &c_tilde_list,
        };

        // 1. Keccak-only groups from stage1_groups(m, 0) and stage2_groups(m, 0) through
        // prove_stage, replay_stage and check_terminal with resolve_keccak.
        let s1_gamma_count = stage1::gamma_count(&manifest);
        let (mut s1_groups, s1_used) = stage1_groups(&manifest, 0);
        assert_eq!(s1_used, 719);
        for gamma in s1_used..s1_gamma_count {
            s1_groups[0].instances.push(InstanceSpec::new(gamma, Vec::new()));
        }
        let s1_spec = StageSpec {
            stage: 1,
            degree: stage1::DEGREE,
            gamma_count: s1_gamma_count,
            groups: s1_groups,
        };
        s1_spec.validate().expect("stage 1 Keccak spec validates");

        let s2_gamma_count = stage2::gamma_count(&manifest);
        let (mut s2_groups, s2_used) = stage2_groups(&manifest, 0);
        assert_eq!(s2_used, 150);
        for gamma in s2_used..s2_gamma_count {
            s2_groups[0].instances.push(InstanceSpec::new(gamma, Vec::new()));
        }
        let s2_spec = StageSpec {
            stage: 2,
            degree: stage2::DEGREE,
            gamma_count: s2_gamma_count,
            groups: s2_groups,
        };
        s2_spec.validate().expect("stage 2 Keccak spec validates");

        let s1_zero_claims = vec![OpeningField::zero(); s1_spec.gamma_count];
        let s2_zero_claims = vec![OpeningField::zero(); s2_spec.gamma_count];

        let (mut prover_tx, mut prover_points) = stages_branch_transcript(&manifest);
        let (mut verifier_tx, mut verifier_points) = stages_branch_transcript(&manifest);

        let s1_run = prove_stage(
            &s1_spec,
            &manifest,
            &prover_points,
            &s1_zero_claims,
            |id| prover_tables.lane(id).expect("Keccak lane"),
            &mut prover_tx,
        )
        .expect("honest stage 1 Keccak prove");
        let s1_keccak_claims: Vec<OpeningField> = keccak_claim_lanes(&s1_spec)
            .into_iter()
            .map(|id| s1_run.value(&s1_spec, id))
            .collect();
        let s1_replay = replay_stage(
            &s1_spec,
            &manifest,
            &s1_zero_claims,
            &s1_run.messages,
            &mut verifier_tx,
        )
        .expect("honest stage 1 Keccak replay");
        check_terminal(&s1_spec, &manifest, &verifier_points, &s1_replay, |id| {
            resolve_keccak(
                &manifest,
                &s1_spec,
                &s1_replay.point,
                &s1_keccak_claims,
                Some(public_inputs),
                id,
            )
        })
        .expect("honest stage 1 Keccak terminal check accepts");

        // Advance both transcripts to stage 2 using the stage-1 terminal point r^(1).
        prover_points.r1 = s1_run.point.clone();
        verifier_points.r1 = s1_replay.point.clone();
        let dummy_s1_wire = Stage1Claims {
            y1: Vec::new(),
            y2: Vec::new(),
            b: vec![OpeningField::zero(); W1_BITS],
            keccak: s1_keccak_claims.clone(),
        }
        .to_vec();
        prover_tx
            .absorb_extensions(&terminal_site(1), &dummy_s1_wire)
            .expect("prover absorb stage 1 terminal");
        verifier_tx
            .absorb_extensions(&terminal_site(1), &dummy_s1_wire)
            .expect("verifier absorb stage 1 terminal");
        prover_tx
            .absorb_extensions(KAPPA1_SITE, &[])
            .expect("prover absorb empty kappa1");
        verifier_tx
            .absorb_extensions(KAPPA1_SITE, &[])
            .expect("verifier absorb empty kappa1");

        let s2_run = prove_stage(
            &s2_spec,
            &manifest,
            &prover_points,
            &s2_zero_claims,
            |id| prover_tables.lane(id).expect("Keccak lane"),
            &mut prover_tx,
        )
        .expect("honest stage 2 Keccak prove");
        let s2_keccak_claims: Vec<OpeningField> = keccak_claim_lanes(&s2_spec)
            .into_iter()
            .map(|id| s2_run.value(&s2_spec, id))
            .collect();
        let s2_replay = replay_stage(
            &s2_spec,
            &manifest,
            &s2_zero_claims,
            &s2_run.messages,
            &mut verifier_tx,
        )
        .expect("honest stage 2 Keccak replay");
        check_terminal(&s2_spec, &manifest, &verifier_points, &s2_replay, |id| {
            resolve_keccak(
                &manifest,
                &s2_spec,
                &s2_replay.point,
                &s2_keccak_claims,
                None,
                id,
            )
        })
        .expect("honest stage 2 Keccak terminal check accepts");

        // Negative test: flip one A' bit at a live boundary slot (t = 24, live in stage 1 and
        // a boundary slot in stage 2). Both prove_stage and prove_stage_unchecked +
        // check_terminal must reject with StageError::Terminal.
        let mut flipped = keccak_tables.clone();
        flipped.flip_a_prime_bit(3, 24, 19, 1);
        let bad_prover_tables = KeccakProverTables::build(&flipped, &keccak_publics);

        let (mut checked_tx, _) = stages_branch_transcript(&manifest);
        assert!(matches!(
            prove_stage(
                &s1_spec,
                &manifest,
                &prover_points,
                &s1_zero_claims,
                |id| bad_prover_tables.lane(id).expect("Keccak lane"),
                &mut checked_tx,
            ),
            Err(StageError::Terminal(1))
        ));

        let (mut bad_p_tx, _) = stages_branch_transcript(&manifest);
        let (mut bad_v_tx, _) = stages_branch_transcript(&manifest);
        let bad_s1_run = prove_stage_unchecked(
            &s1_spec,
            &manifest,
            &prover_points,
            &s1_zero_claims,
            |id| bad_prover_tables.lane(id).expect("Keccak lane"),
            &mut bad_p_tx,
        )
        .expect("unchecked stage 1 prove");
        let bad_s1_claims: Vec<OpeningField> = keccak_claim_lanes(&s1_spec)
            .into_iter()
            .map(|id| bad_s1_run.value(&s1_spec, id))
            .collect();
        let bad_s1_replay = replay_stage(
            &s1_spec,
            &manifest,
            &s1_zero_claims,
            &bad_s1_run.messages,
            &mut bad_v_tx,
        )
        .expect("stage 1 replay of tampered proof");
        assert!(matches!(
            check_terminal(&s1_spec, &manifest, &verifier_points, &bad_s1_replay, |id| {
                resolve_keccak(
                    &manifest,
                    &s1_spec,
                    &bad_s1_replay.point,
                    &bad_s1_claims,
                    Some(public_inputs),
                    id,
                )
            }),
            Err(StageError::Terminal(1))
        ));

        // Advance the tampered transcripts to stage 2 and verify stage 2 also rejects.
        bad_p_tx
            .absorb_extensions(&terminal_site(1), &dummy_s1_wire)
            .expect("absorb terminal 1");
        bad_v_tx
            .absorb_extensions(&terminal_site(1), &dummy_s1_wire)
            .expect("absorb terminal 1");
        bad_p_tx
            .absorb_extensions(KAPPA1_SITE, &[])
            .expect("absorb empty kappa1");
        bad_v_tx
            .absorb_extensions(KAPPA1_SITE, &[])
            .expect("absorb empty kappa1");

        let bad_s2_run = prove_stage_unchecked(
            &s2_spec,
            &manifest,
            &prover_points,
            &s2_zero_claims,
            |id| bad_prover_tables.lane(id).expect("Keccak lane"),
            &mut bad_p_tx,
        )
        .expect("unchecked stage 2 prove");
        let bad_s2_claims: Vec<OpeningField> = keccak_claim_lanes(&s2_spec)
            .into_iter()
            .map(|id| bad_s2_run.value(&s2_spec, id))
            .collect();
        let bad_s2_replay = replay_stage(
            &s2_spec,
            &manifest,
            &s2_zero_claims,
            &bad_s2_run.messages,
            &mut bad_v_tx,
        )
        .expect("stage 2 replay of tampered proof");
        assert!(matches!(
            check_terminal(&s2_spec, &manifest, &verifier_points, &bad_s2_replay, |id| {
                resolve_keccak(
                    &manifest,
                    &s2_spec,
                    &bad_s2_replay.point,
                    &bad_s2_claims,
                    None,
                    id,
                )
            }),
            Err(StageError::Terminal(2))
        ));

        // 2. Full stage1::{prove, verify} and stage2::{prove, verify} drivers under Fp with
        // ProverTables::set_keccak (lattice + Keccak groups together).
        let source_tables = SourceTables {
            manifest: &manifest,
            main: &batch.main,
            bridge: &batch.bridge,
        };
        let mut full_tables = ProverTables::new(source_tables, &batch.matrices);
        full_tables.set_keccak(&prover_tables);
        let dummy_pre = PreStageClaims {
            y2: Vec::new(),
            kappa2: Vec::new(),
            kappa3: OpeningField::zero(),
        };
        let (mut full_p_tx, mut full_p_pts) = stages_branch_transcript(&manifest);
        let (mut full_v_tx, mut full_v_pts) = stages_branch_transcript(&manifest);
        let (s1_proof, s1_out) = stage1::prove(
            &manifest,
            &full_tables,
            &full_p_pts,
            &dummy_pre,
            OpeningField::zero(),
            &mut full_p_tx,
        )
        .expect("full Fp stage 1 prove");
        let s1_verified = stage1::verify(
            &manifest,
            &batch.matrices,
            &full_v_pts,
            &dummy_pre,
            OpeningField::zero(),
            Some(public_inputs),
            &s1_proof,
            &mut full_v_tx,
        )
        .expect("full Fp stage 1 verify");
        assert_eq!(s1_out, s1_verified);

        full_p_pts.r1 = s1_out.point.clone();
        full_v_pts.r1 = s1_verified.point.clone();
        let (s2_proof, s2_out) =
            stage2::prove(&manifest, &full_tables, &full_p_pts, &s1_out, &mut full_p_tx)
                .expect("full Fp stage 2 prove");
        let s2_verified = stage2::verify(
            &manifest,
            &batch.matrices,
            &full_v_pts,
            &s1_verified,
            &s2_proof,
            &mut full_v_tx,
        )
        .expect("full Fp stage 2 verify");
        assert_eq!(s2_out, s2_verified);

        // 3. Padded batch (N = 6, N* = 8): exercises the zero_message_c_tilde padding path in
        // resolve_keccak through stage1::{prove, verify}.
        let pad_batch = Batch::build(common::records(0x516, 4, 2), 0x516);
        let pad_manifest = CommitmentManifest::for_batch_fp(pad_batch.records.len()).expect("fp manifest");
        assert_eq!((pad_manifest.records(), pad_manifest.padded_records()), (6, 8));
        let pad_inputs = keccak_inputs_from_batch(&pad_batch);
        let pad_ktables = KeccakTables::build(&k_layout, &pad_inputs, 8);
        let pad_kpublics = KeccakPublics::from_records(&k_layout, &pad_inputs, 8);
        let pad_kprover = KeccakProverTables::build(&pad_ktables, &pad_kpublics);
        let pad_mu: Vec<[u8; TR_BYTES]> = pad_batch.instance.records.iter().map(|r| r.mu).collect();
        let pad_ct: Vec<Vec<u8>> = pad_batch.instance.records.iter().map(|r| r.c_tilde.to_vec()).collect();
        let pad_pub = KeccakPublicInputs {
            mu: &pad_mu,
            c_tilde: &pad_ct,
        };
        let pad_source = SourceTables {
            manifest: &pad_manifest,
            main: &pad_batch.main,
            bridge: &pad_batch.bridge,
        };
        let mut pad_prover_tables = ProverTables::new(pad_source, &pad_batch.matrices);
        pad_prover_tables.set_keccak(&pad_kprover);
        let (mut pad_p_tx, pad_p_pts) = stages_branch_transcript(&pad_manifest);
        let (mut pad_v_tx, pad_v_pts) = stages_branch_transcript(&pad_manifest);
        let (pad_s1_proof, pad_s1_out) = stage1::prove(
            &pad_manifest,
            &pad_prover_tables,
            &pad_p_pts,
            &dummy_pre,
            OpeningField::zero(),
            &mut pad_p_tx,
        )
        .expect("padded Fp stage 1 prove");
        let pad_s1_verified = stage1::verify(
            &pad_manifest,
            &pad_batch.matrices,
            &pad_v_pts,
            &dummy_pre,
            OpeningField::zero(),
            Some(pad_pub),
            &pad_s1_proof,
            &mut pad_v_tx,
        )
        .expect("padded Fp stage 1 verify");
        assert_eq!(pad_s1_out, pad_s1_verified);
    });
}

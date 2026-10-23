//! The three sum-check stages on a real batch of 256 ML-DSA-44 signatures: every
//! identity's claimed sum against an independent slow evaluator, prover and verifier
//! round trips, and tampered witnesses rejected at the stated stage (spec Section 13).
#![allow(clippy::needless_range_loop)]

mod common;

use aerie_falcon::protocol::algebra::{eq_table, evaluate_mle, lift_table};
use agg_piop::claims::{Kappa1Claims, PreStageClaims};
use agg_piop::kernel::LeafPoly;
use agg_piop::manifest::{Column, CommitmentManifest, DigitValue, GroupRole};
use agg_piop::public_data::PublicData;
use agg_piop::range::{digits, hi_map, lo_map, square_map};
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::stage::{suffix, Points, StageError};
use agg_piop::tables::ProverTables;
use agg_piop::transcript::{
    commitment_site, Transcript, BRIDGE_SEED_SITE, FINAL_VECTORS_SITE, FLOCK_COMMIT_SITE,
    FLOCK_PROOFS_SITE, FLOCK_SEEDS_SITE, PRESTAGE_SITE, TAU_SITE,
};
use agg_piop::virtual_polys::{compose_table, w1_table};
use agg_piop::weights::{level1_weight_table, level2_weight_table, level3_weight_tables, ring_weight_table};
use agg_piop::{stage1, stage2, stage3, OpeningField as E};
use agg_witness::crt::az_integer;
use agg_witness::padding::main_index;
use agg_witness::params::{GAMMA2, K, N, Q, W1_BITS};
use common::Batch;
use jolt_field::{One, Ring, Zero};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// A transcript driven through events 0 to 7 with fixed dummy payloads, for isolated
/// stage tests (prover and verifier replay the same prefix).
fn transcript_at_tau(manifest: &CommitmentManifest) -> (Transcript, Vec<E>) {
    let segments = manifest.segments();
    let mut t = Transcript::new(manifest, &manifest.binding_bytes()).unwrap();
    t.absorb(commitment_site(GroupRole::Main), 1, &[1]).unwrap();
    t.absorb_items(FLOCK_COMMIT_SITE, &vec![vec![]; segments]).unwrap();
    t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap();
    for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3] {
        t.absorb(commitment_site(role), 1, &[2]).unwrap();
    }
    t.challenge_bytes(FLOCK_SEEDS_SITE, segments).unwrap();
    t.absorb(FINAL_VECTORS_SITE, segments, &vec![0; 512 * segments]).unwrap();
    t.absorb_items(FLOCK_PROOFS_SITE, &vec![vec![]; segments]).unwrap();
    let tau = t.challenge_extensions(TAU_SITE, manifest.d_main()).unwrap();
    (t, tau)
}

/// Independent slow evaluators of the stage sums over the Boolean domains.
struct Slow<'a> {
    manifest: &'a CommitmentManifest,
    tables: SourceTables<'a>,
    public: PublicData,
    y1: [Vec<i64>; W1_BITS],
}

impl<'a> Slow<'a> {
    fn new(batch: &'a Batch, tables: SourceTables<'a>) -> Self {
        let y1 = agg_piop::virtual_polys::parity1_tables(tables, &batch.matrices.m1);
        Self {
            manifest: &batch.manifest,
            tables,
            public: PublicData::from_instance(&batch.instance),
            y1,
        }
    }

    fn column(&self, column: Column) -> Vec<E> {
        lift_table(self.tables.column(column))
    }

    fn mapped(&self, column: Column, map: &[i64; 256]) -> Vec<E> {
        self.tables
            .column(column)
            .iter()
            .map(|&z| E::from_i64(map[z as u8 as usize]))
            .collect()
    }

    fn eq(&self, point: &[E], domain: GroupRole) -> Vec<E> {
        eq_table(&suffix(point, self.manifest.group(domain).logical_vars))
    }

    fn dot(weights: &[E], values: &[E]) -> E {
        weights
            .iter()
            .zip(values)
            .fold(E::zero(), |acc, (&w, &v)| acc + w * v)
    }

    /// `sum_x eq(point_dom, x) f(x)`.
    fn eq_sum(&self, point: &[E], domain: GroupRole, f: &[E]) -> E {
        Self::dot(&self.eq(point, domain), f)
    }
}

fn digit_value(value: DigitValue, digit: u8) -> Column {
    Column::Digit { value, digit }
}

/// Stage 1: every claimed sum equals the slow evaluator's; honest tables give the
/// zero-checks zero; the terminal claims are the extensions of the virtual tables.
fn check_stage1(batch: &Batch, slow: &Slow<'_>, prover: &ProverTables<'_>, points: &Points, prestage: &PreStageClaims, y_at_tau3: E) -> stage1::Stage1Output {
    let manifest = &batch.manifest;
    let geom = batch.main.geometry();
    let tau = &points.tau;
    let claims = stage1::claimed_sums(prestage, y_at_tau3);
    // Zero-checks and products on honest tables.
    for t in 0..W1_BITS {
        let y1: Vec<E> = lift_table(&slow.y1[t]);
        let bool_y1: Vec<E> = y1.iter().map(|&y| y * (y - E::one())).collect();
        assert_eq!(slow.eq_sum(tau, GroupRole::Level1, &bool_y1), E::zero());
        let b = slow.column(Column::W1Bit(t as u8));
        let bool_b: Vec<E> = b.iter().map(|&y| y * (y - E::one())).collect();
        assert_eq!(slow.eq_sum(tau, GroupRole::Main, &bool_b), E::zero());
        let y2 = slow.column(Column::Parity2(t as u8));
        let bool_y2: Vec<E> = y2.iter().map(|&y| y * (y - E::one())).collect();
        assert_eq!(slow.eq_sum(tau, GroupRole::Level2, &bool_y2), E::zero());
        // level2[t]: sum w_2 y1_t = y2_t(tau_2) + 2 kappa2_t(tau_2).
        let w2 = level2_weight_table(&geom, &batch.matrices.m2, &suffix(tau, geom.d2() as usize));
        assert_eq!(Slow::dot(&w2, &y1), claims[stage1::gamma_level2(t)]);
    }
    let bits: Vec<Vec<E>> = (0..W1_BITS).map(|t| slow.column(Column::W1Bit(t as u8))).collect();
    let range: Vec<E> = (0..geom.main_len())
        .map(|x| bits[5][x] * (bits[4][x] + bits[3][x] * bits[2][x]))
        .collect();
    assert_eq!(slow.eq_sum(tau, GroupRole::Main, &range), E::zero());
    // level3: sum_t sum w_{3,t} y2_t = Y(tau_3) + 2 kappa3(tau_3).
    let w3 = level3_weight_tables(&geom, &batch.matrices.m3, &suffix(tau, geom.d3() as usize));
    let level3 = (0..W1_BITS).fold(E::zero(), |acc, t| {
        acc + Slow::dot(&w3[t], &slow.column(Column::Parity2(t as u8)))
    });
    assert_eq!(level3, claims[stage1::GAMMA_LEVEL3]);
    for digit in agg_piop::range::base16_digits(manifest) {
        let lo = slow.mapped(digit.column, lo_map());
        let hi = slow.mapped(digit.column, hi_map());
        let product: Vec<E> = lo.iter().zip(&hi).map(|(&a, &b)| a * b).collect();
        assert_eq!(slow.eq_sum(tau, digit.group, &product), E::zero(), "{}", digit.column.name());
    }

    let (mut pt, _) = transcript_at_tau(manifest);
    pt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    let (proof, output) = stage1::prove(manifest, prover, points, prestage, y_at_tau3, &mut pt).unwrap();
    assert_eq!(proof.rounds.len(), manifest.d_main());
    assert!(proof.rounds.iter().all(|m| m.len() == 4));
    // Terminal claims are the multilinear extensions at r^(1).
    let r1 = &output.point;
    for t in 0..W1_BITS {
        assert_eq!(output.claims.y1[t], evaluate_mle(&lift_table(&slow.y1[t]), &suffix(r1, geom.d1() as usize)));
        assert_eq!(output.claims.y2[t], evaluate_mle(&slow.column(Column::Parity2(t as u8)), &suffix(r1, geom.d2() as usize)));
        assert_eq!(output.claims.b[t], evaluate_mle(&bits[t], r1));
    }
    for (k, digit) in agg_piop::range::base16_digits(manifest).iter().enumerate() {
        let local = suffix(r1, manifest.group(digit.group).logical_vars);
        assert_eq!(output.claims.products[k][0], evaluate_mle(&slow.mapped(digit.column, lo_map()), &local));
        assert_eq!(output.claims.products[k][1], evaluate_mle(&slow.mapped(digit.column, hi_map()), &local));
    }
    // The verifier accepts, and rejects a changed claim or message.
    let (mut vt, _) = transcript_at_tau(manifest);
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    let verified = stage1::verify(manifest, &batch.matrices, points, prestage, y_at_tau3, &proof, &mut vt).unwrap();
    assert_eq!(verified, output);
    let mut bad = proof.clone();
    bad.claims.b[2] += E::one();
    let (mut vt, _) = transcript_at_tau(manifest);
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    assert!(matches!(
        stage1::verify(manifest, &batch.matrices, points, prestage, y_at_tau3, &bad, &mut vt),
        Err(StageError::Terminal(1))
    ));
    let mut bad = proof.clone();
    bad.rounds[3][1] += E::one();
    let (mut vt, _) = transcript_at_tau(manifest);
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    assert!(matches!(
        stage1::verify(manifest, &batch.matrices, points, prestage, y_at_tau3, &bad, &mut vt),
        Err(StageError::Terminal(1))
    ));
    output
}

/// Stage 2: the leaves vanish on honest digits, `def_y1` sums to (7.5), and the
/// terminal claims are the extensions of `P_Z`.
fn check_stage2(batch: &Batch, slow: &Slow<'_>, prover: &ProverTables<'_>, points: &Points, prestage: &PreStageClaims, stage1: &stage1::Stage1Output) -> stage2::Stage2Output {
    let manifest = &batch.manifest;
    let geom = batch.main.geometry();
    let r1 = &points.r1;
    let kappa1 = Kappa1Claims::evaluate(manifest, slow.tables, r1);
    let claims = stage2::claimed_sums(&stage1.claims, &kappa1);
    let w1 = level1_weight_table(&geom, &batch.matrices.m1, &suffix(r1, geom.d1() as usize));
    for t in 0..W1_BITS {
        let b = slow.column(Column::W1Bit(t as u8));
        assert_eq!(Slow::dot(&w1, &b), claims[stage2::gamma_def_y1(t)], "def_y1[{t}]");
    }
    let all = digits(manifest);
    for digit in all.iter() {
        let p = slow.mapped(digit.column, square_map());
        let leaf = |poly: LeafPoly| -> Vec<E> { p.iter().map(|&v| poly.evaluate(v)).collect() };
        match digit.base {
            16 => {
                let k16 = agg_piop::range::base16_digits(manifest)
                    .iter()
                    .position(|d| d == digit)
                    .unwrap();
                let (lo, hi) = stage2::gamma_leaf16(k16);
                assert_eq!(slow.eq_sum(r1, digit.group, &leaf(LeafPoly::Q8)), claims[lo]);
                assert_eq!(slow.eq_sum(r1, digit.group, &leaf(LeafPoly::Q16Hi)), claims[hi]);
                assert_eq!(claims[lo], stage1.claims.products[k16][0], "{}", digit.column.name());
                assert_eq!(claims[hi], stage1.claims.products[k16][1]);
            }
            8 => assert_eq!(slow.eq_sum(r1, digit.group, &leaf(LeafPoly::Q8)), E::zero()),
            4 => assert_eq!(slow.eq_sum(r1, digit.group, &leaf(LeafPoly::Q4)), E::zero()),
            _ => unreachable!(),
        }
    }
    let (mut pt, _) = transcript_at_tau(manifest);
    pt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    skip_stage(&mut pt, 1, stage1, manifest);
    let (proof, output) = stage2::prove(manifest, prover, points, stage1, &mut pt).unwrap();
    assert!(proof.rounds.iter().all(|m| m.len() == 5));
    assert_eq!(proof.kappa1, kappa1);
    let r2 = &output.point;
    for t in 0..W1_BITS {
        assert_eq!(output.claims.b[t], evaluate_mle(&slow.column(Column::W1Bit(t as u8)), r2));
    }
    for (k, digit) in all.iter().enumerate() {
        let local = suffix(r2, manifest.group(digit.group).logical_vars);
        assert_eq!(output.claims.squares[k], evaluate_mle(&slow.mapped(digit.column, square_map()), &local));
    }
    let (mut vt, _) = transcript_at_tau(manifest);
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    skip_stage(&mut vt, 1, stage1, manifest);
    let verified = stage2::verify(manifest, &batch.matrices, points, stage1, &proof, &mut vt).unwrap();
    assert_eq!(verified, output);
    let mut bad = proof.clone();
    bad.claims.squares[7] += E::one();
    let (mut vt, _) = transcript_at_tau(manifest);
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    skip_stage(&mut vt, 1, stage1, manifest);
    assert!(matches!(
        stage2::verify(manifest, &batch.matrices, points, stage1, &bad, &mut vt),
        Err(StageError::Terminal(2))
    ));
    output
}

/// Drive a transcript through the registry events of a finished stage with the
/// messages of `output`-compatible dummy payloads (the values do not matter for the
/// isolated tests of later stages, only the event shapes).
fn skip_stage(t: &mut Transcript, stage: usize, stage1: &stage1::Stage1Output, manifest: &CommitmentManifest) {
    use agg_piop::transcript::{gamma_site, round_sites, terminal_site, STAGE_DEGREES, STAGE_GAMMAS, STAGE_TERMINALS};
    if stage == 2 {
        t.absorb_extensions(agg_piop::transcript::KAPPA1_SITE, &vec![E::zero(); 18]).unwrap();
    }
    t.challenge_extensions(&gamma_site(stage), STAGE_GAMMAS[stage - 1]).unwrap();
    for round in 0..manifest.d_main() {
        let (m, c) = round_sites(stage, round);
        t.absorb_extensions(&m, &vec![E::zero(); STAGE_DEGREES[stage - 1]]).unwrap();
        t.challenge_extensions(&c, 1).unwrap();
    }
    let _ = stage1;
    t.absorb_extensions(&terminal_site(stage), &vec![E::zero(); STAGE_TERMINALS[stage - 1]]).unwrap();
}

/// Stage 3: the ring sum equals `T(r^(2))`, the links and reductions hold, and the
/// terminal claims are the committed extensions at `r^(3)`.
fn check_stage3(batch: &Batch, slow: &Slow<'_>, prover: &ProverTables<'_>, points: &Points, prestage: &PreStageClaims, stage1: &stage1::Stage1Output, stage2: &stage2::Stage2Output) -> stage3::Stage3Output {
    let manifest = &batch.manifest;
    let geom = batch.main.geometry();
    let r2 = &points.r2;
    // Ring (7.6) by the slow evaluator: sum w_A z - sum eq (2 gamma2 w1' + u + q v) = T(r2).
    let wa = ring_weight_table(&slow.public, r2);
    let z = lift_table(&compose_table(&DigitValue::Z.layout(), &slow.tables.digits(DigitValue::Z)));
    let u = compose_table(&DigitValue::U.layout(), &slow.tables.digits(DigitValue::U));
    let v = compose_table(&DigitValue::V.layout(), &slow.tables.digits(DigitValue::V));
    let w1 = w1_table(&batch.main.bit_lanes(manifest));
    let high: Vec<E> = (0..geom.main_len())
        .map(|x| E::from_i64(2 * GAMMA2 * w1[x] + u[x] + Q * v[x]))
        .collect();
    let ring = Slow::dot(&wa, &z) - slow.eq_sum(r2, GroupRole::Main, &high);
    let t_at_r2 = slow.public.t_evaluation(r2);
    assert_eq!(ring, t_at_r2);
    // And (A z) itself: sum eq(r2, x) (A z)(x) = sum w_A z.
    let mut az_sum = E::zero();
    let eq_r2 = eq_table(r2);
    let z_table = compose_table(&DigitValue::Z.layout(), &slow.tables.digits(DigitValue::Z));
    for i in 0..geom.n {
        let mut zi = [[0_i64; N]; 4];
        for (s, row) in zi.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = z_table[main_index(j, s, i)];
            }
        }
        let az = az_integer(&batch.instance.records[i].key, &zi);
        for rho in 0..K {
            for j in 0..N {
                az_sum += eq_r2[main_index(j, rho, i)] * E::from_i128(az[rho][j]);
            }
        }
    }
    assert_eq!(az_sum, Slow::dot(&wa, &z));
    // Links: sum eq(r2_Z, x) Z (Z + 1) = P_Z(r2_Z).
    for (k, digit) in digits(manifest).iter().enumerate() {
        let col = slow.column(digit.column);
        let link: Vec<E> = col.iter().map(|&c| c * (c + E::one())).collect();
        assert_eq!(slow.eq_sum(r2, digit.group, &link), stage2.claims.squares[k]);
    }
    let (mut pt, _) = transcript_at_tau(manifest);
    pt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    skip_stage(&mut pt, 1, stage1, manifest);
    skip_stage(&mut pt, 2, stage1, manifest);
    let (proof, output) = stage3::prove(manifest, prover, &slow.public, points, prestage, stage1, stage2, &mut pt).unwrap();
    assert!(proof.rounds.iter().all(|m| m.len() == 3));
    let r3 = &output.point;
    for group in manifest.groups() {
        let local = suffix(r3, group.logical_vars);
        for &column in &group.columns {
            assert_eq!(output.claims.column(manifest, column), evaluate_mle(&slow.column(column), &local), "{}", column.name());
        }
    }
    let (mut vt, _) = transcript_at_tau(manifest);
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    skip_stage(&mut vt, 1, stage1, manifest);
    skip_stage(&mut vt, 2, stage1, manifest);
    let verified = stage3::verify(manifest, &slow.public, points, prestage, stage1, stage2, &proof, &mut vt).unwrap();
    assert_eq!(verified, output);
    let mut bad = proof.clone();
    bad.claims.0[0] += E::one();
    let (mut vt, _) = transcript_at_tau(manifest);
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    skip_stage(&mut vt, 1, stage1, manifest);
    skip_stage(&mut vt, 2, stage1, manifest);
    assert!(matches!(
        stage3::verify(manifest, &slow.public, points, prestage, stage1, stage2, &bad, &mut vt),
        Err(StageError::Terminal(3))
    ));
    output
}

fn batch_256() -> Batch {
    Batch::build(common::records(0x5a6e, 200, 56), 0x5a6e)
}

#[test]
fn stages_match_slow_evaluators_on_256_signatures() {
    on_large_stack(|| {
        let batch = batch_256();
        let manifest = &batch.manifest;
        let tables = batch.tables();
        let slow = Slow::new(&batch, tables);
        let (_, tau) = transcript_at_tau(manifest);
        let mut points = Points { tau: tau.clone(), ..Points::default() };
        let prestage = PreStageClaims::evaluate(manifest, tables, &tau);
        let y: Vec<i8> = batch.bridge.public_y().iter().map(|&b| b as i8).collect();
        let y_at_tau3 = evaluate_mle(&lift_table(&y), &suffix(&tau, manifest.group(GroupRole::Level3).logical_vars));
        let mut prover = ProverTables::new(tables, &batch.matrices);
        prover.set_tau(&batch.matrices, &tau);
        let stage1 = check_stage1(&batch, &slow, &prover, &points, &prestage, y_at_tau3);
        points.r1 = stage1.point.clone();
        prover.set_r1(&batch.matrices, &points.r1);
        let stage2 = check_stage2(&batch, &slow, &prover, &points, &prestage, &stage1);
        points.r2 = stage2.point.clone();
        prover.set_r2(&slow.public, &points.r2);
        let _ = check_stage3(&batch, &slow, &prover, &points, &prestage, &stage1, &stage2);
    });
}

/// Which stage rejects a tampered witness.
fn rejecting_stage(batch: &Batch, main: &MainTables) -> Option<usize> {
    let manifest = &batch.manifest;
    let bridge = BridgeGroupTables::build(manifest, main, &batch.matrices).unwrap();
    let tables = SourceTables { manifest, main, bridge: &bridge };
    let public = PublicData::from_instance(&batch.instance);
    let (mut pt, tau) = transcript_at_tau(manifest);
    let (mut vt, _) = transcript_at_tau(manifest);
    let mut points = Points { tau: tau.clone(), ..Points::default() };
    let prestage = PreStageClaims::evaluate(manifest, tables, &tau);
    pt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
    let y: Vec<i8> = bridge.public_y().iter().map(|&b| b as i8).collect();
    let y_at_tau3 = evaluate_mle(&lift_table(&y), &suffix(&tau, manifest.group(GroupRole::Level3).logical_vars));
    let mut prover = ProverTables::new(tables, &batch.matrices);
    prover.set_tau(&batch.matrices, &tau);
    let (proof1, out1) = stage1::prove_unchecked(manifest, &prover, &points, &prestage, y_at_tau3, &mut pt).unwrap();
    if stage1::verify(manifest, &batch.matrices, &points, &prestage, y_at_tau3, &proof1, &mut vt).is_err() {
        return Some(1);
    }
    points.r1 = out1.point.clone();
    prover.set_r1(&batch.matrices, &points.r1);
    let (proof2, out2) = stage2::prove_unchecked(manifest, &prover, &points, &out1, &mut pt).unwrap();
    if stage2::verify(manifest, &batch.matrices, &points, &out1, &proof2, &mut vt).is_err() {
        return Some(2);
    }
    points.r2 = out2.point.clone();
    prover.set_r2(&public, &points.r2);
    let (proof3, _) = stage3::prove_unchecked(manifest, &prover, &public, &points, &prestage, &out1, &out2, &mut pt).unwrap();
    if stage3::verify(manifest, &public, &points, &prestage, &out1, &out2, &proof3, &mut vt).is_err() {
        return Some(3);
    }
    None
}

#[test]
fn tampered_witnesses_fail_at_the_stated_stage() {
    on_large_stack(|| {
        // 50 live records keep the tampering runs short (N* = 64, d_main = 16).
        let batch = Batch::build(common::records(0x7a, 40, 10), 0x7a);
        let manifest = &batch.manifest;
        assert_eq!(rejecting_stage(&batch, &batch.main), None, "honest tables verify");
        let cell = main_index(17, 2, 5);
        let mutate = |column: Column, f: &dyn Fn(i8) -> i8| {
            let mut main = batch.main.clone();
            let (_, position) = manifest.locate(column);
            let value = main.columns()[position][cell];
            main.columns_mut()[position][cell] = f(value);
            main
        };
        // One coefficient of z changed (digit stays in its alphabet): only the ring fails.
        let z0 = digit_value(DigitValue::Z, 0);
        let z = mutate(z0, &|d| if d < 7 { d + 1 } else { d - 1 });
        assert_eq!(rejecting_stage(&batch, &z), Some(3));
        // One residual off by q: the quotient digit v_0 changed by one.
        let v = mutate(digit_value(DigitValue::V, 0), &|d| if d < 7 { d + 1 } else { d - 1 });
        assert_eq!(rejecting_stage(&batch, &v), Some(3));
        // One bit flipped: w1' changes, the ring fails (w1' < 44 is kept: flip b_0).
        let b = mutate(Column::W1Bit(0), &|d| 1 - d);
        assert_eq!(rejecting_stage(&batch, &b), Some(3));
        // A non-Boolean bit fails the Booleanity check of stage 1.
        let two = mutate(Column::W1Bit(1), &|_| 2);
        assert_eq!(rejecting_stage(&batch, &two), Some(1));
        // w1' = 44 or more fails range_w1 in stage 1: set b_5 = 1, b_3 = b_2 = 1.
        let mut w44 = batch.main.clone();
        for t in [2, 3, 5] {
            let (_, position) = manifest.locate(Column::W1Bit(t));
            w44.columns_mut()[position][cell] = 1;
        }
        for t in [0, 1, 4] {
            let (_, position) = manifest.locate(Column::W1Bit(t));
            w44.columns_mut()[position][cell] = 0;
        }
        assert_eq!(rejecting_stage(&batch, &w44), Some(1));
        // A base-16 digit outside its alphabet fails b16_prod in stage 1.
        let wide = mutate(digit_value(DigitValue::U, 1), &|_| 9);
        assert_eq!(rejecting_stage(&batch, &wide), Some(1));
        // A base-4 digit outside its alphabet passes stage 1 and fails leaf4 in stage 2.
        let base4 = mutate(digit_value(DigitValue::Z, 4), &|_| 3);
        assert_eq!(rejecting_stage(&batch, &base4), Some(2));
        // A base-8 digit outside its alphabet fails leaf8 in stage 2.
        let base8 = mutate(digit_value(DigitValue::U, 4), &|_| -6);
        assert_eq!(rejecting_stage(&batch, &base8), Some(2));
    });
}

#[test]
fn tampered_bridge_tables_fail_in_stage_one() {
    on_large_stack(|| {
        let batch = Batch::build(common::records(0x7b, 20, 4), 0x7b);
        let manifest = &batch.manifest;
        let public = PublicData::from_instance(&batch.instance);
        let run = |bridge: &BridgeGroupTables| -> Option<usize> {
            let tables = SourceTables { manifest, main: &batch.main, bridge };
            let (mut pt, tau) = transcript_at_tau(manifest);
            let (mut vt, _) = transcript_at_tau(manifest);
            let points = Points { tau: tau.clone(), ..Points::default() };
            let prestage = PreStageClaims::evaluate(manifest, tables, &tau);
            pt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
            vt.absorb_extensions(PRESTAGE_SITE, &prestage.to_vec()).unwrap();
            let y: Vec<i8> = bridge.public_y().iter().map(|&b| b as i8).collect();
            let y_at_tau3 = evaluate_mle(&lift_table(&y), &suffix(&tau, manifest.group(GroupRole::Level3).logical_vars));
            let mut prover = ProverTables::new(tables, &batch.matrices);
            prover.set_tau(&batch.matrices, &tau);
            let (proof1, _) = stage1::prove_unchecked(manifest, &prover, &points, &prestage, y_at_tau3, &mut pt).unwrap();
            stage1::verify(manifest, &batch.matrices, &points, &prestage, y_at_tau3, &proof1, &mut vt).err().map(|_| 1)
        };
        let _ = &public;
        assert_eq!(run(&batch.bridge), None);
        // A level-1 carry digit changed: y^(1) leaves {0, 1} (bool_y1) and level2 breaks.
        let mut carry = batch.bridge.clone();
        let column = &mut carry.group_mut(GroupRole::Level1)[0];
        column[3] = if column[3] < 7 { column[3] + 1 } else { column[3] - 1 };
        assert_eq!(run(&carry), Some(1));
        // A level-2 parity flipped: level2 (7.2) and level3 (7.3) break.
        let mut parity = batch.bridge.clone();
        parity.group_mut(GroupRole::Level2)[2][5] ^= 1;
        assert_eq!(run(&parity), Some(1));
        // A final-vector bit flipped: level3 breaks.
        let mut y = batch.bridge.to_witness_tables();
        y.y[11] ^= 1;
        let flipped = BridgeGroupTables::from_tables(manifest, y).unwrap();
        assert_eq!(run(&flipped), Some(1));
    });
}

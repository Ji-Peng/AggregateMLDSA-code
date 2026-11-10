//! ML-DSA-65 and ML-DSA-87 end to end without the opening: the F_p side (round-1 and
//! round-2 tables, the PIOP of lookup and stages) and the Keccak side (one tied part over the
//! batch, its sponge records read back from the committed bits) on one transcript that
//! starts from the mode-A instance record with the set's work plan. The tied part's
//! commitment is absorbed before the bridge seed, its final vector is the F_p side's public
//! `Y` and is absorbed at the final-vector step, and the part's Flock seed is the span seed
//! of the transcript's Flock seed and final-vector record. The verifier builds its public
//! data from `(pk, M, c~)` only. The full mode-A driver (`Aggregator`, `AggregateVerifier`:
//! the Akita opening and the wire format) runs at the end, with tampered witnesses and
//! aggregates. Relation H-FIPS also runs on the synthetic records of hint weight exactly omega
//! (`agg_witness::gen::hint_corner_record_for`) for all three sets, ML-DSA-44 included. Run
//! with `--release`.

use agg_bridge::seed::derive_matrices_for;
use agg_bridge::Shape;
use agg_cli::mode_a::{keccak_plan, span_seed, sponge_records_for};
use agg_keccak::prover::{KeccakError, Profile};
use agg_keccak::sponge::sponge_gate_for;
use agg_keccak::tied::TiedPart;
use agg_piop::binding::{instance_record, public_records_binding, FlockProfile};
use agg_piop::lane_public::{LaneKeyCache, LanePublicData, LaneStreamedPublic};
use agg_piop::manifest::{CommitmentManifest, GroupRole, Relation};
use agg_witness::params::Set;
use agg_piop::pow::PowPlan;
use agg_piop::prover::prove_piop_lanes;
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::transcript::{
    commitment_site, flock_record_digest, Transcript, BRIDGE_SEED_SITE, FINAL_VECTORS_SITE,
    FLOCK_COMMIT_SITE, FLOCK_SEEDS_SITE, FLOCK_SHARES_SITE,
};
use agg_piop::verifier::{verify_piop_lanes, VerifyTimings};
use agg_piop::wire::pack_final_vector;
use agg_witness::instance::PublicInstance;
use agg_witness::padding::Geometry;
use agg_witness::params::{ParamSet, Set44, Set65, Set87};

/// What the transcript prefix hands to the provers: `sd`, `Psi`, and after the final-vector
/// step the transcript itself and the part's Flock seed.
struct Prefix {
    sd: [u8; 32],
    psi: Vec<u8>,
}

/// The prefix up to `Psi`: round-1 commitments (dummy payloads), the tied part's commitment
/// at the Flock commitment site, the bridge seed.
fn to_psi(t: &mut Transcript, flock_commitment: &[u8]) -> Prefix {
    t.absorb(commitment_site(GroupRole::Main16), 1, &[1]).unwrap();
    t.absorb(commitment_site(GroupRole::Main), 1, &[1]).unwrap();
    t.absorb_items(FLOCK_COMMIT_SITE, &[flock_commitment.to_vec()]).unwrap();
    let sd: [u8; 32] = t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap().try_into().expect("32 bytes");
    let psi = t.challenge_psi().unwrap();
    Prefix { sd, psi }
}

/// The round-2 commitments (dummy payloads), the Flock seed, the final vector `y` and empty
/// share items; returns the part's span seed.
fn to_stages(t: &mut Transcript, y: &[u8]) -> [u8; 32] {
    for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Proj, GroupRole::Mult] {
        t.absorb(commitment_site(role), 1, &[2]).unwrap();
    }
    let fs: [u8; 32] = t.challenge_bytes(FLOCK_SEEDS_SITE, 1).unwrap().try_into().expect("32 bytes");
    let packed = pack_final_vector(y);
    t.absorb(FINAL_VECTORS_SITE, 1, &packed).unwrap();
    t.absorb_items(FLOCK_SHARES_SITE, &[Vec::new()]).unwrap();
    span_seed(&[fs], &flock_record_digest(&[packed], &[Vec::new()]))
}

fn end_to_end<P: ParamSet, const C: usize>(seed: u64, n: usize)
where
    P::CTilde: Sync,
{
    let records = agg_witness::gen::records_for::<P>(seed, n);
    let instance = PublicInstance::<P>::from_records_for(&records).expect("instance");
    let manifest = CommitmentManifest::for_set(P::SET, n, Relation::HFips).expect("manifest");
    assert_eq!(manifest.segments(), 1);
    let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
    let main = MainTables::build_for::<P>(&manifest, &instance, &signatures).expect("round 1 tables");

    // The Keccak side's records are the w1Encode of the committed bits, so their c~ is the
    // signatures' (FIPS 204 Algorithm 8, line 12).
    let flock_records = sponge_records_for(&instance, &manifest, &main);
    for (f, r) in flock_records.iter().zip(&instance.records) {
        assert_eq!(f.c_tilde(), r.c_tilde);
    }
    let gate = sponge_gate_for(P::SET);
    let geometry = Geometry::new(n).unwrap();
    let d = geometry.d_seg();
    let part = TiedPart::new_span(&gate, n, d, 0, 1, Profile::Fast);
    let committed = part.commit(&flock_records);
    let commitment = committed.commitment().clone();

    // The instance record of mode A with the set's plan (the tied part's tie and chain sites).
    let inputs: Vec<(&[u8], &[u8], P::CTilde)> = records
        .iter()
        .map(|r| (r.pk.as_slice(), r.msg.as_slice(), P::c_tilde_from_slice(&r.sig)))
        .collect();
    let keccak = keccak_plan(&gate, &geometry, 13, true, FlockProfile::Fast);
    assert_eq!(keccak.parts.len(), 1);
    assert_eq!(keccak.parts[0], part.work());
    let plan = PowPlan::new(&manifest, FlockProfile::Fast, &keccak);
    let record = instance_record(&manifest, &public_records_binding(&inputs), FlockProfile::Fast, &plan.digest());

    // The prover.
    let mut t = Transcript::new(&manifest, &record).expect("prover transcript");
    let prefix = to_psi(&mut t, &commitment.to_bytes());
    let shape = Shape::production(P::SET);
    let matrices = derive_matrices_for(&shape, &prefix.sd);
    let bridge = BridgeGroupTables::build(&manifest, &main, &matrices, &prefix.psi).expect("round 2 tables");
    let y = bridge.public_y().to_vec();
    let ys = part.final_vectors(&committed, &matrices);
    assert_eq!(ys, vec![y.clone()], "the Keccak side's Y is the F_p side's");
    let fs = to_stages(&mut t, &y);
    let public = LanePublicData::from_instance(&instance);
    let tables = SourceTables { manifest: &manifest, main: &main, bridge: &bridge };
    let (proof, _) = prove_piop_lanes(&manifest, tables, &matrices, &public, &mut t, false).expect("PIOP prover");
    let session = t.pcs_session().expect("session");
    t.ensure_complete().expect("every event");
    let work = t.work_stream();
    let flock_proof = part.prove_with_final_vectors(committed, prefix.sd, &matrices, ys.clone(), &fs).expect("tied prover");
    eprintln!(
        "{} end to end at {n}: tied proof {} B ({:.1} B/sig)",
        P::NAME,
        flock_proof.bytes.len(),
        flock_proof.bytes.len() as f64 / n as f64
    );

    // The verifier: (pk, M, c~), the record, the nonce stream, the commitment and the proofs.
    let mut cache = LaneKeyCache::new(P::SET);
    let lane_data = LaneStreamedPublic::prepare::<P>(&inputs, Some(&cache)).expect("public data");
    let mut v = Transcript::replay(&manifest, &record, &work).expect("verifier transcript");
    let vprefix = to_psi(&mut v, &commitment.to_bytes());
    assert_eq!(vprefix.sd, prefix.sd);
    let vmatrices = derive_matrices_for(&shape, &vprefix.sd);
    let vfs = to_stages(&mut v, &y);
    let mut timings = VerifyTimings::default();
    verify_piop_lanes(&manifest, &vmatrices, &y, &proof, &mut v, &mut timings, |r2, r3, gammas| {
        (lane_data.t_evaluations(r2), lane_data.ring_weights_at(r2, r3, gammas, Some(&mut cache)))
    })
    .outcome
    .expect("PIOP verifies");
    assert_eq!(v.pcs_session().expect("session"), session);
    v.ensure_complete().expect("every event");
    let mu: Vec<[u8; 64]> = instance.records.iter().map(|r| r.mu).collect();
    let c_tilde: Vec<[u8; C]> = inputs.iter().map(|(_, _, c)| c.as_ref().try_into().expect("c~")).collect();
    part.verify(&mu, &c_tilde, &commitment, vprefix.sd, &vmatrices, &ys, &vfs, &flock_proof)
        .expect("tied part verifies");

    // The other final vector (a bit of Y flipped) fails the Keccak side; another c~ fails it
    // too (the Flock transcript of the sponge's statement).
    let mut bad = ys.clone();
    bad[0][7] ^= 1;
    assert!(part.verify(&mu, &c_tilde, &commitment, vprefix.sd, &vmatrices, &bad, &vfs, &flock_proof).is_err());
    let mut other = c_tilde.clone();
    other[n / 2][C - 1] ^= 1;
    assert!(part.verify(&mu, &other, &commitment, vprefix.sd, &vmatrices, &ys, &vfs, &flock_proof).is_err());
    // A tampered committed bit (lane 3 of record 1): Y and the sponge follow the tampered
    // record, the published c~ is the signature's, so the sponge's public claim fails.
    let mut tampered = flock_records.clone();
    tampered[1].w1_encode.as_mut()[10] ^= 0x80;
    let bad_part = part.commit_parts(part.honest_parts(&tampered).0, part.honest_parts(&tampered).1, &flock_records);
    let bad_commitment = bad_part.commitment().clone();
    let bad_ys = part.final_vectors(&bad_part, &vmatrices);
    assert_ne!(bad_ys, ys);
    let bad_proof = part.prove_with_final_vectors(bad_part, prefix.sd, &matrices, bad_ys.clone(), &fs).expect("prove");
    assert!(matches!(
        part.verify(&mu, &c_tilde, &bad_commitment, vprefix.sd, &vmatrices, &bad_ys, &vfs, &bad_proof),
        Err(KeccakError::Tie("public words"))
    ));
}

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and PIOP proofs; run with --release")]
fn ml_dsa_65_end_to_end_without_the_opening_at_1k() {
    on_large_stack(|| end_to_end::<Set65, 48>(0x6501, 1024));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and PIOP proofs; run with --release")]
fn ml_dsa_87_end_to_end_without_the_opening_at_1k() {
    on_large_stack(|| end_to_end::<Set87, 64>(0x8701, 1024));
}

/// The full mode-A driver for a set with lanes: `Aggregator` (instance, tables, Flock side,
/// PIOP, Akita opening, wire bytes) and `AggregateVerifier` from `(pk, M)` only, uncached
/// and with a warm `LaneKeyCache`; then tampered witnesses and aggregates are rejected.
fn full_driver<P: ParamSet>(seed: u64, n: usize, other: Set) {
    use agg_cli::mode_a::{AggregateHooks, AggregateVerifier, Aggregator, AggregatorConfig, KeyCacheMut};
    use agg_piop::manifest::Column;
    use agg_piop::stage::StageError;
    use agg_piop::verifier::VerifyError;
    use agg_witness::padding::main_index_for;

    let records = agg_witness::gen::records_shared_keys_for::<P>(seed, n, n / 4);
    let signed: Vec<(&[u8], &[u8], &[u8])> =
        records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice())).collect();
    let statements: Vec<(&[u8], &[u8])> = records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice())).collect();
    let config = AggregatorConfig { set: P::SET, ..AggregatorConfig::default() };
    let aggregator = Aggregator::new(config);
    let aggregate = aggregator.aggregate(&signed).expect("aggregate");
    assert_eq!(aggregate.proof.set, P::SET);
    assert_eq!(aggregate.report.size.c_tilde, n * P::C_TILDE_BYTES);
    for (c, r) in aggregate.c_tilde().iter().zip(&records) {
        assert_eq!(c.as_slice(), &r.sig[..P::C_TILDE_BYTES]);
    }
    eprintln!(
        "{} full driver at {n}: {} B ({:.1} B/sig)",
        P::NAME,
        aggregate.bytes.len(),
        aggregate.report.bytes_per_signature()
    );
    // Deterministic: a second run gives the same bytes.
    assert_eq!(aggregator.aggregate(&signed).expect("again").bytes, aggregate.bytes);

    let verifier = AggregateVerifier::for_set(P::SET);
    let report = verifier.verify(&statements, &aggregate.bytes, None);
    assert!(report.accepted(), "{report}");
    let mut cache = LaneKeyCache::new(P::SET);
    let cold = verifier.verify_with(&statements, &aggregate.bytes, KeyCacheMut::Lanes(&mut cache));
    assert!(cold.accepted(), "{cold}");
    assert_eq!(cache.len(), n / 4);
    let warm = verifier.verify_with(&statements, &aggregate.bytes, KeyCacheMut::Lanes(&mut cache));
    assert_eq!(warm.report.outcome.as_ref().expect("warm").expanded_keys, 0);

    // The verifier's set, never the header's: another set's verifier, and a rewritten set
    // nibble, are rejected.
    let wrong = AggregateVerifier::for_set(other).verify(&statements, &aggregate.bytes, None);
    assert!(matches!(wrong.report.outcome, Err(VerifyError::ParamSet { .. })), "{wrong}");
    let as_44 = AggregateVerifier::new().verify(&statements, &aggregate.bytes, None);
    assert!(!as_44.accepted(), "{as_44}");
    let mut renamed = aggregate.bytes.clone();
    renamed[2] ^= 0x30;
    assert!(!verifier.verify(&statements, &renamed, None).accepted());
    // Another c~ in the list, another message, a flipped byte of the opening.
    let mut other_c = aggregate.proof.clone();
    other_c.c_tilde[n / 2][3] ^= 1;
    assert!(!verifier.verify(&statements, &other_c.to_bytes(), None).accepted());
    let mut moved = statements.clone();
    let msg = b"another message".to_vec();
    moved[1].1 = &msg;
    assert!(!verifier.verify(&moved, &aggregate.bytes, None).accepted());
    let mut flipped = aggregate.bytes.clone();
    let last = flipped.len() - 100;
    flipped[last] ^= 1;
    assert!(!verifier.verify(&statements, &flipped, None).accepted());
    // H-exact: its own relation byte; each relation's verifier rejects the other's
    // aggregate, and relation byte 0 is unknown.
    let exact = Aggregator::new(AggregatorConfig { relation: Relation::HExact, ..config })
        .aggregate(&signed)
        .expect("H-exact aggregate");
    assert_eq!(exact.proof.relation, Relation::HExact);
    assert_eq!(exact.bytes[2] & 0x0F, Relation::HExact.id());
    let exact_verifier = AggregateVerifier::for_set(P::SET).relation(Relation::HExact);
    let report = exact_verifier.verify(&statements, &exact.bytes, None);
    assert!(report.accepted(), "{report}");
    let crossed = exact_verifier.verify(&statements, &aggregate.bytes, None);
    assert!(matches!(crossed.report.outcome, Err(VerifyError::Relation { .. })), "{crossed}");
    let crossed = verifier.verify(&statements, &exact.bytes, None);
    assert!(matches!(crossed.report.outcome, Err(VerifyError::Relation { .. })), "{crossed}");
    let mut relabelled = exact.bytes.clone();
    relabelled[2] &= 0xF0;
    assert!(!verifier.verify(&statements, &relabelled, None).accepted());
    // H-FIPS (the default): its own layout (the hint columns), accepted under H-FIPS only.
    let fips = Aggregator::new(AggregatorConfig { relation: Relation::HFips, ..config })
        .aggregate(&signed)
        .expect("H-FIPS aggregate");
    assert_eq!(fips.bytes[2] & 0x0F, Relation::HFips.id());
    assert_eq!(fips.bytes, aggregate.bytes, "H-FIPS is the default relation");
    let fips_verifier = AggregateVerifier::for_set(P::SET).relation(Relation::HFips);
    let report = fips_verifier.verify(&statements, &fips.bytes, None);
    assert!(report.accepted(), "{report}");
    let crossed = exact_verifier.verify(&statements, &fips.bytes, None);
    assert!(matches!(crossed.report.outcome, Err(VerifyError::Relation { .. })), "{crossed}");
    let mut relabelled = fips.bytes.clone();
    relabelled[2] = (relabelled[2] & 0xF0) | Relation::HExact.id();
    assert!(!exact_verifier.verify(&statements, &relabelled, None).accepted());
    eprintln!(
        "{} H-FIPS at {n}: {} B (H-exact {} B)",
        P::NAME,
        fips.bytes.len(),
        exact.bytes.len()
    );

    // Tampered witnesses (the prover's self-checks off): a G_main16 cell and a G_main4 cell.
    let wide = |_: &CommitmentManifest, t: &mut MainTables| {
        let c = &mut t.wide_mut()[17];
        *c = if *c < (1 << 15) - 1 { *c + 1 } else { *c - 1 };
    };
    let narrow = |_: &CommitmentManifest, t: &mut MainTables| t.columns_mut()[0][17] ^= 1;
    for (name, tamper) in [("G_main16 cell", &wide as &dyn Fn(&CommitmentManifest, &mut MainTables)), ("G_main4 cell", &narrow)] {
        let hooks = AggregateHooks { tamper_main: Some(tamper), ..AggregateHooks::default() };
        let bad = aggregator.aggregate_with(&signed, &hooks).expect("tampered aggregate");
        let report = verifier.verify(&statements, &bad.bytes, None);
        assert!(!report.accepted(), "{name}: {report}");
    }
    // A committed bit flipped with the round-2 tables (and Y) of the honest bits: the Flock
    // part accepts, and stage 1 rejects a committed bit that disagrees with the bridge tables.
    let flipped = |m: &CommitmentManifest, t: &mut MainTables| {
        let (role, position) = m.locate(Column::W1Bit(0, 1));
        assert_eq!(role, GroupRole::Main);
        t.columns_mut()[position][main_index_for::<P>(17, 0, 5)] ^= 1;
    };
    let hooks = AggregateHooks { tamper_main: Some(&flipped), stale_bridge: true, ..AggregateHooks::default() };
    let bad = aggregator.aggregate_with(&signed, &hooks).expect("stale round-2 aggregate");
    let report = verifier.verify(&statements, &bad.bytes, None);
    assert!(
        matches!(report.report.outcome, Err(VerifyError::Stage(StageError::Terminal(1)))),
        "committed bit flipped, stale round 2: {report}"
    );
    assert!(report.parts.iter().all(|p| p.accepted), "the Flock part accepts: {report}");
}

/// The default prover's layout at small batches: segment 0's item names the canonical layout
/// of the batch (one part of the full segment), and a default verifier accepts the aggregate.
fn default_layout_round_trip<P: ParamSet>(seed: u64, n: usize) {
    use agg_cli::mode_a::{canonical_layout, AggregateVerifier, Aggregator, AggregatorConfig, FlockLayout};
    use agg_cli::wire::SegmentCommitments;

    let records = agg_witness::gen::records_shared_keys_for::<P>(seed, n, n.div_ceil(4));
    let signed: Vec<(&[u8], &[u8], &[u8])> =
        records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice())).collect();
    let statements: Vec<(&[u8], &[u8])> = records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice())).collect();
    let aggregate = Aggregator::new(AggregatorConfig { set: P::SET, ..AggregatorConfig::default() })
        .aggregate(&signed)
        .expect("aggregate");
    let item = SegmentCommitments::from_bytes(&aggregate.proof.flock_commitments[0]).expect("item");
    let geometry = Geometry::new(n).unwrap();
    assert_eq!(
        FlockLayout { span: item.span, d_part: item.d_part },
        canonical_layout(&geometry),
        "{} at N = {n}",
        P::NAME
    );
    assert_eq!((item.d_part, item.commitments.len()), (geometry.d_seg(), 1), "{} at N = {n}", P::NAME);
    let report = AggregateVerifier::for_set(P::SET).verify_with(
        &statements,
        &aggregate.bytes,
        agg_cli::mode_a::KeyCacheMut::None,
    );
    assert!(report.accepted(), "{} at N = {n}: {report}", P::NAME);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and Akita proofs; run with --release")]
fn default_layout_verifies_by_default_at_small_batches_of_every_set() {
    on_large_stack(|| {
        for n in [1, 9, 40] {
            default_layout_round_trip::<Set44>(0x1a44 + n as u64, n);
        }
        // 17 records: the smallest lane batch of ML-DSA-65 and ML-DSA-87.
        for n in [17, 33] {
            default_layout_round_trip::<Set65>(0x1a65 + n as u64, n);
            default_layout_round_trip::<Set87>(0x1a87 + n as u64, n);
        }
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and Akita proofs; run with --release")]
fn ml_dsa_65_full_driver_with_the_opening() {
    on_large_stack(|| full_driver::<Set65>(0x6511, 256, Set::S87));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and Akita proofs; run with --release")]
fn ml_dsa_87_full_driver_with_the_opening() {
    on_large_stack(|| full_driver::<Set87>(0x8711, 256, Set::S65));
}

/// Mode A under relation H-FIPS on 15 honest records and the weight-omega records of seeds 1 and
/// 2 (17 records, the smallest lane batch of ML-DSA-65 and -87). Each corner record has the coefficient
/// `w' = q - gamma2, h = 0` and the `e = -1` wrap to `w1' = m - 1 = 15`. The integer checker
/// accepts their H-FIPS witnesses, extraction returns the signatures byte for byte, the
/// aggregate verifies, and a hint slack moved off zero is rejected by the weight identity.
fn h_fips_corner_records<P: ParamSet>(seed: u64) {
    use agg_cli::mode_a::{AggregateHooks, AggregateVerifier, Aggregator, AggregatorConfig};
    use agg_piop::stage::StageError;
    use agg_piop::verifier::VerifyError;
    use agg_witness::derive::derive_for;
    use agg_witness::gen::{self, HINT_ZERO};
    use agg_witness::hfips::{check_batch_for, HFipsRecordOf};

    let mut records = gen::records_for::<P>(seed, 15);
    records.push(gen::hint_corner_record_for::<P>(1));
    records.push(gen::hint_corner_record_for::<P>(2));
    let instance = PublicInstance::<P>::from_records_for(&records).expect("instance");
    let witnesses: Vec<HFipsRecordOf<P>> = instance
        .records
        .iter()
        .zip(&records)
        .map(|(p, r)| {
            let w = derive_for::<P>(&p.key, &p.mu, &r.sig).expect("an accepted signature");
            HFipsRecordOf::<P>::from_witness(&w).expect("H-FIPS witness")
        })
        .collect();
    let rep = check_batch_for::<P>(&instance, &witnesses);
    assert!(rep.is_ok(), "{:?}", rep.failures);
    for i in [15, 16] {
        let w = &witnesses[i];
        assert_eq!(w.hint_weight(), P::OMEGA as i64, "record {i} has hint weight omega");
        assert_eq!(w.u_h, 0);
        assert_eq!(w.w1()[HINT_ZERO.0][HINT_ZERO.1], P::M_W - 1, "record {i}: the wrap lands on m - 1");
        assert_eq!(w.extract_signature(&instance.records[i].c_tilde).expect("extract"), records[i].sig);
    }

    let signed: Vec<(&[u8], &[u8], &[u8])> =
        records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice(), r.sig.as_slice())).collect();
    let statements: Vec<(&[u8], &[u8])> = records.iter().map(|r| (r.pk.as_slice(), r.msg.as_slice())).collect();
    let config = AggregatorConfig { set: P::SET, relation: Relation::HFips, ..AggregatorConfig::default() };
    let aggregator = Aggregator::new(config);
    let aggregate = aggregator.aggregate(&signed).expect("H-FIPS aggregate");
    assert_eq!(aggregate.bytes[2] & 0x0F, Relation::HFips.id());
    let verifier = AggregateVerifier::for_set(P::SET).relation(Relation::HFips);
    let report = verifier.verify(&statements, &aggregate.bytes, None);
    assert!(report.accepted(), "{report}");
    eprintln!("{} H-FIPS with weight-omega records: {} B", P::NAME, aggregate.bytes.len());

    // u_h = 1 on a weight-omega record claims weight omega - 1: stage 3 rejects it.
    let slack = |_: &CommitmentManifest, t: &mut MainTables| t.hint_slack_mut()[16] += 1;
    let hooks = AggregateHooks { tamper_main: Some(&slack), ..AggregateHooks::default() };
    let bad = aggregator.aggregate_with(&signed, &hooks).expect("tampered aggregate");
    let report = verifier.verify(&statements, &bad.bytes, None);
    assert!(
        matches!(report.report.outcome, Err(VerifyError::Stage(StageError::Terminal(3)))),
        "u_h = 1: {report}"
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and Akita proofs; run with --release")]
fn ml_dsa_44_h_fips_weight_omega_records_in_mode_a() {
    on_large_stack(|| h_fips_corner_records::<Set44>(0x44f));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and Akita proofs; run with --release")]
fn ml_dsa_65_h_fips_weight_omega_records_in_mode_a() {
    on_large_stack(|| h_fips_corner_records::<Set65>(0x65f));
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock and Akita proofs; run with --release")]
fn ml_dsa_87_h_fips_weight_omega_records_in_mode_a() {
    on_large_stack(|| h_fips_corner_records::<Set87>(0x87f));
}

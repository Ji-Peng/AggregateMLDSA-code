//! ML-DSA-65 and ML-DSA-87 end to end without the opening: the F_p side (round-1 and
//! round-2 tables, the PIOP of lookup and stages) and the Keccak side (one tied part over the
//! batch, its sponge records read back from the committed bits) on one transcript that
//! starts from the mode-A instance record with the set's work plan. The tied part's
//! commitment is absorbed before the bridge seed, its final vector is the F_p side's public
//! `Y` and is absorbed at the final-vector step, and the part's Flock seed is the span seed
//! of the transcript's Flock seed and final-vector record. The verifier builds its public
//! data from `(pk, M, c~)` only. What the full mode-A driver adds (the Akita opening and the
//! wire format) is not exercised here. Run with `--release`.

use agg_bridge::seed::derive_matrices_for;
use agg_bridge::Shape;
use agg_cli::mode_a::{keccak_plan, span_seed, sponge_records_for};
use agg_keccak::prover::{KeccakError, Profile};
use agg_keccak::sponge::sponge_gate_for;
use agg_keccak::tied::TiedPart;
use agg_piop::binding::{instance_record, public_records_binding, FlockProfile};
use agg_piop::lane_public::{LaneKeyCache, LanePublicData};
use agg_piop::manifest::{CommitmentManifest, GroupRole, Relation};
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
use agg_witness::params::{ParamSet, Set65, Set87};

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
    let manifest = CommitmentManifest::for_set(P::SET, n, Relation::Hu).expect("manifest");
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
    let lane_data = LanePublicData::prepare::<P>(&inputs, Some(&mut cache)).expect("public data");
    let mut v = Transcript::replay(&manifest, &record, &work).expect("verifier transcript");
    let vprefix = to_psi(&mut v, &commitment.to_bytes());
    assert_eq!(vprefix.sd, prefix.sd);
    let vmatrices = derive_matrices_for(&shape, &vprefix.sd);
    let vfs = to_stages(&mut v, &y);
    let mut timings = VerifyTimings::default();
    verify_piop_lanes(&manifest, &vmatrices, &y, &proof, &mut v, &mut timings, |r2, r3, gammas| {
        (lane_data.t_evaluations(r2), lane_data.ring_weights_at(r2, r3, gammas))
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

//! Shared helpers of the agg-piop integration tests.
#![allow(dead_code)]

use aerie_falcon::protocol::algebra::{evaluate_mle, lift_table};
use agg_piop::lane_public::LanePublicData;
use agg_piop::lookup::Table;
use agg_piop::manifest::{CommitmentManifest, GroupRole, Relation, LOOKUP_SHIFT};
use agg_piop::prover::{prove_piop, prove_piop_lanes, PiopProof, PiopRun, ProveError};
use agg_piop::public_data::PublicData;
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::stage::Points;
use agg_piop::stage3::Stage3Output;
use agg_piop::transcript::{
    commitment_site, Transcript, BRIDGE_SEED_SITE, FINAL_VECTORS_SITE, FLOCK_COMMIT_SITE,
    FLOCK_SEEDS_SITE, FLOCK_SHARES_SITE,
};
use agg_piop::verifier::{verify_piop, verify_piop_lanes, VerifyError, VerifyTimings};
use agg_piop::weights::ring_weight_at;
use agg_piop::{BaseField, OpeningField};
use agg_witness::bridge::{BridgeMatrices, SparseMatrix};
use agg_witness::gen::{self, Record, SplitMix64};
use agg_witness::instance::PublicInstance;
use agg_witness::params::ParamSet;
use jolt_field::{ExtField, Ring};

/// A pseudo-random element of `K`.
pub fn field(rng: &mut SplitMix64) -> OpeningField {
    OpeningField::from_base_slice(&[
        BaseField::from_u64(rng.next_u64()),
        BaseField::from_u64(rng.next_u64()),
    ])
}

/// A pseudo-random point of `K^len`.
pub fn point(rng: &mut SplitMix64, len: usize) -> Vec<OpeningField> {
    (0..len).map(|_| field(rng)).collect()
}

/// The Boolean point of `index` (least significant coordinate first).
pub fn boolean_point(index: usize, len: usize) -> Vec<OpeningField> {
    (0..len)
        .map(|bit| OpeningField::from_u64(((index >> bit) & 1) as u64))
        .collect()
}

/// Multilinear extension of an integer table.
pub fn mle_i64(table: &[i64], point: &[OpeningField]) -> OpeningField {
    evaluate_mle(&lift_table(table), point)
}

/// Own-key and shared-key honest records (`n_own + n_shared` in total).
pub fn records(seed: u64, n_own: usize, n_shared: usize) -> Vec<Record> {
    let mut records = gen::records(seed, n_own);
    records.extend(gen::records_shared_keys(seed + 1, n_shared, 3));
    records
}

/// Everything a prover holds after both commitment rounds' tables are built.
pub struct Batch {
    pub records: Vec<Record>,
    pub instance: PublicInstance,
    pub manifest: CommitmentManifest,
    pub main: MainTables,
    pub matrices: BridgeMatrices,
    pub bridge: BridgeGroupTables,
}

impl Batch {
    pub fn build(records: Vec<Record>, seed: u64) -> Self {
        Self::build_as(Relation::Hu, records, seed)
    }

    /// [`Batch::build`] under `relation`.
    pub fn build_as(relation: Relation, records: Vec<Record>, seed: u64) -> Self {
        let instance = PublicInstance::from_records(&records).expect("instance");
        let manifest = CommitmentManifest::for_relation(records.len(), relation).expect("manifest");
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let main = MainTables::build(&manifest, &instance, &signatures).expect("round 1 tables");
        let matrices = BridgeMatrices::random_for_tests(seed);
        let psi = transcript_psi(&manifest);
        let bridge = BridgeGroupTables::build(&manifest, &main, &matrices, &psi).expect("round 2 tables");
        Self {
            records,
            instance,
            manifest,
            main,
            matrices,
            bridge,
        }
    }

    pub fn tables(&self) -> SourceTables<'_> {
        SourceTables {
            manifest: &self.manifest,
            main: &self.main,
            bridge: &self.bridge,
        }
    }
}

/// The transcript of [`transcript_after_commitments`] up to and including the bridge seed.
fn transcript_to_seed(manifest: &CommitmentManifest) -> Transcript {
    drive_to_seed(manifest, Transcript::new(manifest, &manifest.binding_bytes()).unwrap())
}

/// Absorb the dummy round-1 and Flock commitments and draw the bridge seed.
fn drive_to_seed(manifest: &CommitmentManifest, mut t: Transcript) -> Transcript {
    let segments = manifest.segments();
    t.absorb(commitment_site(GroupRole::Main16), 1, &[1]).unwrap();
    t.absorb(commitment_site(GroupRole::Main), 1, &[1]).unwrap();
    t.absorb_items(FLOCK_COMMIT_SITE, &vec![vec![]; segments]).unwrap();
    t.challenge_bytes(BRIDGE_SEED_SITE, 1).unwrap();
    t
}

/// The projection matrix `Psi` that [`transcript_after_commitments`] draws (the dummy
/// prefix fixes it), so the isolated tests build the round-2 tables under the transcript's
/// own `Psi`.
pub fn transcript_psi(manifest: &CommitmentManifest) -> Vec<u8> {
    transcript_to_seed(manifest).challenge_psi().unwrap()
}

/// A transcript driven through the transcript through the final-vector step (the final vectors and empty share items) with
/// fixed dummy payloads, for isolated PIOP tests (prover and verifier replay the same
/// prefix; the join follows the stages).
pub fn transcript_after_commitments(manifest: &CommitmentManifest) -> Transcript {
    let mut t = transcript_to_seed(manifest);
    t.challenge_psi().unwrap();
    drive_after_psi(manifest, &mut t);
    t
}

/// The prefix of [`transcript_after_commitments`] on the instance record `record`: a
/// prover's transcript (`work = None`, its nonces searched) or a verifier's replaying the
/// nonce stream `work`. Returns the transcript after the final-vector step and `Psi`.
pub fn transcript_on_record(
    manifest: &CommitmentManifest,
    record: &[u8],
    work: Option<&[u8]>,
) -> Result<(Transcript, Vec<u8>), agg_piop::transcript::TranscriptError> {
    let t = match work {
        None => Transcript::new(manifest, record)?,
        Some(work) => Transcript::replay(manifest, record, work)?,
    };
    let mut t = drive_to_seed(manifest, t);
    let psi = t.challenge_psi()?;
    drive_after_psi(manifest, &mut t);
    Ok((t, psi))
}

/// The dummy round-2 commitments, the Flock seeds, zero final vectors and empty share items.
fn drive_after_psi(manifest: &CommitmentManifest, t: &mut Transcript) {
    let segments = manifest.segments();
    for role in [GroupRole::Level1, GroupRole::Level2, GroupRole::Level3, GroupRole::Proj, GroupRole::Mult] {
        t.absorb(commitment_site(role), 1, &[2]).unwrap();
    }
    t.challenge_bytes(FLOCK_SEEDS_SITE, segments).unwrap();
    t.absorb(FINAL_VECTORS_SITE, segments, &vec![0; 512 * segments]).unwrap();
    t.absorb_items(FLOCK_SHARES_SITE, &vec![vec![]; segments]).unwrap();
}

/// The PIOP part (lookup and stages) of `tables`, honest (`unchecked = false`) or with the
/// prover's self-checks skipped.
pub fn prove_tables(
    batch: &Batch,
    tables: SourceTables<'_>,
    unchecked: bool,
) -> Result<(PiopProof, PiopRun), ProveError> {
    let public = PublicData::from_instance(&batch.instance);
    let mut transcript = transcript_after_commitments(&batch.manifest);
    prove_piop(&batch.manifest, tables, &batch.matrices, &public, &mut transcript, unchecked)
}

/// Verify the PIOP part against the public `Y` of `y_table` (the opening is not run: the
/// caller compares the returned claims with the tables).
pub fn verify_tables(
    batch: &Batch,
    y_table: &[u8],
    proof: &PiopProof,
) -> Result<(Points, Stage3Output, Vec<OpeningField>), VerifyError> {
    let public = PublicData::from_instance(&batch.instance);
    let mut transcript = transcript_after_commitments(&batch.manifest);
    let mut timings = VerifyTimings::default();
    verify_piop(
        &batch.manifest,
        &batch.matrices,
        y_table,
        proof,
        &mut transcript,
        &mut timings,
        |r2, r3| (public.t_evaluation(r2), ring_weight_at(&public, r2, r3)),
    )
    .outcome
}

/// The multiplicities of `T_16` over the in-table looked-up cells only (a cheating
/// prover's best count when a cell is out of the table).
pub fn lenient_multiplicities(tables: SourceTables<'_>) -> Vec<i32> {
    let mut m = vec![0_i32; Table::T16.len()];
    for stack in tables.lookup_stacks().unwrap() {
        for &stored in stack.stacked() {
            let f = i64::from(stored) + i64::from(LOOKUP_SHIFT);
            if (0..1 << 16).contains(&f) {
                m[f as usize] += 1;
            }
        }
    }
    m
}

/// The PIOP part of a batch of any parameter set: its tables and the prover and verifier
/// of the lookup and the stages.
pub trait PiopBatch {
    fn manifest(&self) -> &CommitmentManifest;
    fn main(&self) -> &MainTables;
    fn bridge(&self) -> &BridgeGroupTables;
    fn matrices(&self) -> &BridgeMatrices;
    /// Prove the PIOP part of `tables` (honest, or with the self-checks skipped).
    fn prove(&self, tables: SourceTables<'_>, unchecked: bool) -> Result<(PiopProof, PiopRun), ProveError>;
    /// Verify the PIOP part against the public `Y` of `y_table`.
    fn verify(
        &self,
        y_table: &[u8],
        proof: &PiopProof,
    ) -> Result<(Points, Stage3Output, Vec<OpeningField>), VerifyError>;

    fn tables(&self) -> SourceTables<'_> {
        SourceTables {
            manifest: self.manifest(),
            main: self.main(),
            bridge: self.bridge(),
        }
    }
}

impl PiopBatch for Batch {
    fn manifest(&self) -> &CommitmentManifest {
        &self.manifest
    }
    fn main(&self) -> &MainTables {
        &self.main
    }
    fn bridge(&self) -> &BridgeGroupTables {
        &self.bridge
    }
    fn matrices(&self) -> &BridgeMatrices {
        &self.matrices
    }
    fn prove(&self, tables: SourceTables<'_>, unchecked: bool) -> Result<(PiopProof, PiopRun), ProveError> {
        prove_tables(self, tables, unchecked)
    }
    fn verify(
        &self,
        y_table: &[u8],
        proof: &PiopProof,
    ) -> Result<(Points, Stage3Output, Vec<OpeningField>), VerifyError> {
        verify_tables(self, y_table, proof)
    }
}

/// Test matrices of the bridge shape of `manifest`'s parameter set.
pub fn bridge_matrices_for(manifest: &CommitmentManifest, seed: u64) -> BridgeMatrices {
    let shape = agg_bridge::Shape::production(manifest.set());
    let level = |t: usize| shape.level(t);
    let matrix = |t: usize| SparseMatrix::random_for_tests(seed ^ t as u64, level(t).rows, level(t).cols, level(t).k);
    BridgeMatrices {
        m1: matrix(1),
        m2: matrix(2),
        m3: matrix(3),
    }
}

/// A batch of a parameter set with polynomial lanes (ML-DSA-65, ML-DSA-87), relation H-u.
pub struct LaneBatch {
    pub records: Vec<Record>,
    pub manifest: CommitmentManifest,
    pub main: MainTables,
    pub matrices: BridgeMatrices,
    pub bridge: BridgeGroupTables,
    pub public: LanePublicData,
}

impl LaneBatch {
    /// `n_own` records with their own keys and `n_shared` over three shared keys.
    pub fn build<P: ParamSet>(seed: u64, n_own: usize, n_shared: usize) -> Self {
        let mut records = gen::records_for::<P>(seed, n_own);
        records.extend(gen::records_shared_keys_for::<P>(seed + 1, n_shared, 3));
        let instance = PublicInstance::<P>::from_records_for(&records).expect("instance");
        let manifest = CommitmentManifest::for_set(P::SET, records.len(), Relation::Hu).expect("manifest");
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let main = MainTables::build_for::<P>(&manifest, &instance, &signatures).expect("round 1 tables");
        let matrices = bridge_matrices_for(&manifest, seed);
        let psi = transcript_psi(&manifest);
        let bridge = BridgeGroupTables::build(&manifest, &main, &matrices, &psi).expect("round 2 tables");
        Self {
            public: LanePublicData::from_instance(&instance),
            records,
            manifest,
            main,
            matrices,
            bridge,
        }
    }
}

impl PiopBatch for LaneBatch {
    fn manifest(&self) -> &CommitmentManifest {
        &self.manifest
    }
    fn main(&self) -> &MainTables {
        &self.main
    }
    fn bridge(&self) -> &BridgeGroupTables {
        &self.bridge
    }
    fn matrices(&self) -> &BridgeMatrices {
        &self.matrices
    }
    fn prove(&self, tables: SourceTables<'_>, unchecked: bool) -> Result<(PiopProof, PiopRun), ProveError> {
        let mut transcript = transcript_after_commitments(&self.manifest);
        prove_piop_lanes(&self.manifest, tables, &self.matrices, &self.public, &mut transcript, unchecked)
    }
    fn verify(
        &self,
        y_table: &[u8],
        proof: &PiopProof,
    ) -> Result<(Points, Stage3Output, Vec<OpeningField>), VerifyError> {
        let mut transcript = transcript_after_commitments(&self.manifest);
        let mut timings = VerifyTimings::default();
        verify_piop_lanes(
            &self.manifest,
            &self.matrices,
            y_table,
            proof,
            &mut transcript,
            &mut timings,
            |r2, r3, gammas| (self.public.t_evaluations(r2), self.public.ring_weights_at(r2, r3, gammas)),
        )
        .outcome
    }
}

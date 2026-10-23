//! Shared helpers of the agg-piop integration tests.
#![allow(dead_code)]

use aerie_falcon::protocol::algebra::{evaluate_mle, lift_table};
use agg_piop::manifest::CommitmentManifest;
use agg_piop::source_tables::{BridgeGroupTables, MainTables, SourceTables};
use agg_piop::{BaseField, OpeningField};
use agg_witness::bridge::BridgeMatrices;
use agg_witness::gen::{self, Record, SplitMix64};
use agg_witness::instance::PublicInstance;
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

/// Multilinear extension of an `i8` table.
pub fn mle_i8(table: &[i8], point: &[OpeningField]) -> OpeningField {
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
        let instance = PublicInstance::from_records(&records).expect("instance");
        let manifest = CommitmentManifest::for_batch(records.len()).expect("manifest");
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let main = MainTables::build(&manifest, &instance, &signatures).expect("round 1 tables");
        let matrices = BridgeMatrices::random_for_tests(seed);
        let bridge = BridgeGroupTables::build(&manifest, &main, &matrices).expect("round 2 tables");
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

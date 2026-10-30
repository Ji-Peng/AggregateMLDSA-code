//! The Fp side of the parity bridge inside the 256-signature prover run (`docs/bridge.md`):
//! matrices derived from the transcript's `sd` by the shared
//! `agg_bridge::seed` on both sides, the honest round-2 tables against the independent
//! `agg-witness` reference and the GF(2) side, the verifier's bridge weights against the
//! prover's weight tables at the transcript's terminal points, and bridge tampering that the
//! stage-1 identities reject.

use crate::common;

use agg_bridge::seed::derive_matrices;
use agg_bridge::{gf2_walker, parity, weights as bridge_weights, Shape};
use agg_piop::binding::FlockProfile;
use agg_piop::manifest::Column;
use agg_piop::manifest::{CommitmentManifest, GroupRole, Relation};
use agg_piop::prover::{
    main_tables, prove, prove_with_tables, BridgeDerivation, BridgeTamper, FixedMatrices, NoKeccak,
    ProveHooks, ProverInputs, SeededMatrices,
};
use agg_piop::source_tables::{BridgeGroupTables, MainTables};
use agg_piop::stage::{suffix, StageError};
use agg_piop::stage1::BridgeWeightsAt;
use agg_piop::verifier::{verify, NoKeccakVerifier, VerifyError};
use agg_piop::weights::{
    level1_weight_at, level1_weight_table, level2_weight_table, level3_weight_tables,
};
use agg_piop::wire::{unpack_final_vector, AggregateProof};
use agg_piop::OpeningField as E;
use agg_witness::bridge::{self as reference, BridgeMatrices};
use agg_witness::gen::Record;
use agg_witness::instance::PublicInstance;
use agg_witness::padding::Geometry;
use agg_witness::params::C_TILDE_BYTES;
use jolt_field::Zero;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

fn mle(table: &[E], point: &[E]) -> E {
    bridge_weights::eq_table(point)
        .iter()
        .zip(table)
        .fold(E::zero(), |acc, (&w, &v)| acc + w * v)
}

struct Fixture {
    records: Vec<Record>,
    instance: PublicInstance,
}

impl Fixture {
    fn new(seed: u64, own: usize, shared: usize) -> Self {
        let records = common::records(seed, own, shared);
        let instance = PublicInstance::from_records(&records).unwrap();
        Self { records, instance }
    }

    fn inputs(&self) -> Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> {
        self.records
            .iter()
            .map(|r| {
                let mut ct = [0_u8; C_TILDE_BYTES];
                ct.copy_from_slice(&r.sig[..C_TILDE_BYTES]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect()
    }

    fn public(&self) -> Vec<(&[u8], &[u8])> {
        self.records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect()
    }

    fn signatures(&self) -> Vec<&[u8]> {
        self.records.iter().map(|r| r.sig.as_slice()).collect()
    }

    fn verify(&self, bytes: &[u8], bridge: &impl BridgeDerivation) -> Result<(), VerifyError> {
        verify(
            &self.public(),
            bytes,
            bridge,
            &NoKeccakVerifier,
            FlockProfile::None,
            Relation::Hu,
            None,
        )
        .map(|_| ())
    }

    /// A prover run over `main` with unchecked stages and an optional round-2 change.
    fn prove_tampered(
        &self,
        main: &MainTables,
        bridge: &impl BridgeDerivation,
        tamper: Option<BridgeTamper<'_>>,
    ) -> Vec<u8> {
        let inputs = self.inputs();
        let signatures = self.signatures();
        let prover_inputs = ProverInputs {
            instance: &self.instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        };
        let manifest = CommitmentManifest::for_batch(self.records.len()).unwrap();
        let hooks = ProveHooks {
            unchecked: true,
            tamper_bridge: tamper,
            swap_join: false,
        };
        prove_with_tables(
            &prover_inputs,
            &manifest,
            main,
            bridge,
            &NoKeccak,
            FlockProfile::None,
            &hooks,
        )
        .expect("prove")
        .proof
        .to_bytes()
    }
}

#[test]
fn bridge_identities_in_the_256_signature_run_with_seed_derived_matrices() {
    on_large_stack(|| {
        let fixture = Fixture::new(0xb81d, 200, 56);
        let inputs = fixture.inputs();
        let signatures = fixture.signatures();
        let prover_inputs = ProverInputs {
            instance: &fixture.instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        };
        let output = prove(
            &prover_inputs,
            &SeededMatrices,
            &NoKeccak,
            FlockProfile::None,
        )
        .expect("prove");
        let bytes = output.proof.to_bytes();
        fixture.verify(&bytes, &SeededMatrices).expect("verify");

        // Both sides derive the same matrices from the transcript's sd (agg_bridge::seed).
        let matrices = derive_matrices(&output.bridge_seed);
        let (manifest, main) = main_tables(&prover_inputs).unwrap();
        let geom = Geometry::new(256).unwrap();
        let lanes = main.bit_lanes(&manifest);

        // The honest round-2 tables: agg-bridge equals the agg-witness reference, the
        // identities hold at every Boolean point, and the published Y is the GF(2) composite.
        let tables = BridgeGroupTables::build(&manifest, &main, &matrices, &agg_piop::projection::test_psi(1)).unwrap();
        let expected = reference::compute(&geom, &lanes, &matrices).unwrap();
        assert_eq!(tables.carry_values(&manifest, 1), expected.kappa1.to_vec());
        assert_eq!(tables.carry_values(&manifest, 2), expected.kappa2.to_vec());
        assert_eq!(tables.carry_values(&manifest, 3), vec![expected.kappa3.clone()]);
        for (ours, theirs) in tables.parities().iter().zip(&expected.y2) {
            assert!(ours.iter().zip(theirs).all(|(&a, &b)| i64::from(a) == b));
        }
        assert!(tables.public_y().iter().zip(&expected.y).all(|(&a, &b)| i64::from(a) == b));
        let witness = parity::compute(&Shape::PRODUCTION, &lanes, &matrices).unwrap();
        parity::check(&Shape::PRODUCTION, &lanes, &matrices, &witness).unwrap();
        let proof = AggregateProof::from_bytes(&bytes).unwrap();
        let published = unpack_final_vector(&proof.final_vectors[0]);
        assert_eq!(published, tables.public_y());
        assert_eq!(
            published,
            reference::gf2_composite(&geom, &lanes, &matrices)
        );
        let layout = gf2_walker::SegmentLayout::production(geom.d_seg());
        let mut packed = vec![0_u64; layout.input_bits() / 64];
        for (t, lane) in lanes.iter().enumerate() {
            for (p, &bit) in lane.iter().enumerate() {
                packed[layout.column(t, p) / 64] |=
                    u64::from(bit as u8) << (layout.column(t, p) % 64);
            }
        }
        assert_eq!(
            gf2_walker::composite(&layout, &matrices, &packed),
            published
        );

        // The verifier's bridge weights at the terminal points equal the multilinear
        // extensions of the prover's weight tables.
        let (points, _r3) = &output.points;
        let vars = |role| manifest.group(role).logical_vars;
        let (tau2, tau3) = (
            suffix(&points.tau, vars(GroupRole::Level2)),
            suffix(&points.tau, vars(GroupRole::Level3)),
        );
        let (r1_1, r1_2) = (
            suffix(&points.r1, vars(GroupRole::Level1)),
            suffix(&points.r1, vars(GroupRole::Level2)),
        );
        let at = BridgeWeightsAt::evaluate(&manifest, &geom, &matrices, &points.tau, &points.r1);
        assert_eq!(
            at.level2,
            mle(&level2_weight_table(&geom, &matrices.m2, &tau2), &r1_1)
        );
        let w3 = level3_weight_tables(&geom, &matrices.m3, &tau3);
        for (t, table) in w3.iter().enumerate() {
            assert_eq!(at.level3[t], mle(table, &r1_2), "lane {t}");
        }
        let w1 = level1_weight_table(&geom, &matrices.m1, &r1_1);
        assert_eq!(
            level1_weight_at(&geom, &matrices.m1, &r1_1, &points.r2),
            mle(&w1, &points.r2)
        );
    });
}

#[test]
fn bridge_tampering_is_rejected_at_stage_one() {
    on_large_stack(|| {
        let fixture = Fixture::new(0xb81e, 40, 8);
        let inputs = fixture.inputs();
        let signatures = fixture.signatures();
        let prover_inputs = ProverInputs {
            instance: &fixture.instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        };
        let (manifest, main) = main_tables(&prover_inputs).unwrap();
        // Matrices fixed in advance (test stand-in), so that a hook can rebuild stale tables.
        let fixed = FixedMatrices(BridgeMatrices::random_for_tests(0xb81e));
        // The stage whose terminal check rejects, or None when the proof verifies.
        let run = |main: &MainTables, tamper: Option<BridgeTamper<'_>>| -> Option<usize> {
            match fixture.verify(&fixture.prove_tampered(main, &fixed, tamper), &fixed) {
                Ok(()) => None,
                Err(VerifyError::Stage(StageError::Terminal(stage))) => Some(stage),
                Err(other) => panic!("rejected outside the stage terminals: {other:?}"),
            }
        };
        assert_eq!(run(&main, None), None);

        // A flipped final-vector bit, committed through the transcript: level3 breaks.
        let flip_y: BridgeTamper<'_> = &|_, tables| tables.public_y_mut()[11] ^= 1;
        assert_eq!(run(&main, Some(flip_y)), Some(1));
        // A flipped level-2 parity: level2 and level3 break.
        let flip_y2: BridgeTamper<'_> = &|_, tables| tables.parities_mut()[2][5] ^= 1;
        assert_eq!(run(&main, Some(flip_y2)), Some(1));
        // A level-1 carry moved inside T_16 (m recounted): y^(1) leaves {0, 1} (bool_y1).
        let carry1: BridgeTamper<'_> = &|manifest, tables| {
            tables.carries_mut(1)[3] += 1;
            tables.recount(manifest, &main).unwrap();
        };
        assert_eq!(run(&main, Some(carry1)), Some(1));
        // A level-3 carry moved inside T_16: level3 breaks.
        let carry3: BridgeTamper<'_> = &|manifest, tables| {
            tables.carries_mut(3)[0] += 1;
            tables.recount(manifest, &main).unwrap();
        };
        assert_eq!(run(&main, Some(carry3)), Some(1));
        // A committed bit flipped in G_main under the stale honest round-2 tables: the level-1
        // parities of the committed bits leave {0, 1} (bool_y1, def_y1).
        let honest = BridgeGroupTables::build(&manifest, &main, &fixed.0, &agg_piop::projection::test_psi(1)).unwrap();
        let stale: BridgeTamper<'_> = &|_, tables| *tables = honest.clone();
        let mut flipped = main.clone();
        let (_, position) = manifest.locate(Column::W1Bit(2));
        flipped.columns_mut()[position][1000] ^= 1;
        assert_eq!(run(&flipped, Some(stale)), Some(1));
    });
}

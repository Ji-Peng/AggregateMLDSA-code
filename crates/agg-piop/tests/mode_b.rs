//! Mode B end to end (spec Sections 11, 13, 14): `prove_mode_b` / `verify_mode_b` round
//! trips at 256 and 1,024 signatures with the seed-derived matrices of spec 9.1
//! (`SeededMatrices`), the proof size and the verifier's step timings printed, corrupted
//! aggregates rejected, and every witness tampering of `tests/stages.rs` pushed through the
//! full prover (unchecked stages) and rejected by the verifier.
#![allow(clippy::needless_range_loop)]

mod common;

use agg_piop::binding::FlockProfile;
use agg_piop::manifest::{Column, CommitmentManifest, DigitValue, GroupRole};
use agg_piop::mode_b::{prove_mode_b, public_bits, verify_mode_b, PublicBits, PublicBitsVerifier};
use agg_piop::prover::{
    main_tables, prove_with_tables, BridgeDerivation, BridgeTamper, FixedMatrices, ProveHooks,
    ProverInputs, ProverOutput, SeededMatrices,
};
use agg_piop::public_data::KeyCache;
use agg_piop::source_tables::{BridgeGroupTables, MainTables};
use agg_piop::stage::StageError;
use agg_piop::verifier::{
    verify_timed, BitsContext, ConfigurationError, KeccakVerifier, SegmentContext, VerifyError,
    VerifyReport,
};
use agg_bridge::seed::derive_matrices;
use agg_witness::bridge::{BridgeMatrices, J1, K1};
use agg_witness::digits::U_DIGITS;
use agg_witness::gen::Record;
use agg_witness::instance::PublicInstance;
use agg_witness::native::commitment_hash;
use agg_witness::padding::main_index;
use agg_witness::params::{C_TILDE_BYTES, GAMMA2};

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

/// The honest inputs of a batch.
struct Fixture {
    records: Vec<Record>,
    instance: PublicInstance,
}

impl Fixture {
    fn new(seed: u64, own: usize, shared: usize) -> Self {
        let records = common::records(seed, own, shared);
        let instance = PublicInstance::from_records(&records).expect("instance");
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

    fn signatures(&self) -> Vec<&[u8]> {
        self.records.iter().map(|r| r.sig.as_slice()).collect()
    }

    fn public(&self) -> Vec<(&[u8], &[u8])> {
        self.records
            .iter()
            .map(|r| (r.pk.as_slice(), r.msg.as_slice()))
            .collect()
    }

    fn prove(&self, bridge: &impl BridgeDerivation) -> ProverOutput {
        let inputs = self.inputs();
        let signatures = self.signatures();
        prove_mode_b(
            &ProverInputs {
                instance: &self.instance,
                records: &inputs,
                signatures: &signatures,
            },
            bridge,
        )
        .expect("prove")
    }
}

fn report_line(label: &str, report: &VerifyReport) {
    eprintln!(
        "{label}: accepted {}, {}",
        report.accepted(),
        report.timings
    );
}

/// An honest round trip with the matrices derived from `sd` (spec 9.1): proof size,
/// verifier timings uncached and cached, and the public bits satisfying the hash relation.
fn round_trip(
    n_own: usize,
    n_shared: usize,
    seed: u64,
) -> (Fixture, SeededMatrices, ProverOutput, Vec<u8>) {
    let fixture = Fixture::new(seed, n_own, n_shared);
    let n = n_own + n_shared;
    let bridge = SeededMatrices;
    let output = fixture.prove(&bridge);
    let bytes = output.proof.to_bytes();
    let size = output.proof.size();
    assert_eq!(size.total(), bytes.len());
    eprintln!(
        "N = {n}: aggregate {} bytes = header {} + c~ {} + public bits {} + Akita commitments {} \
         + final vectors {} + Keccak proofs {} + PIOP {} + opening {}",
        bytes.len(),
        size.header,
        size.c_tilde,
        size.keccak_commitments,
        size.akita_commitments,
        size.final_vectors,
        size.keccak_proofs,
        size.piop,
        size.opening
    );
    eprintln!("N = {n}: prover {}", output.timings);
    for i in [0, n / 2, n - 1] {
        let bits = public_bits(&output.proof, i).expect("record bits");
        let record = &fixture.instance.records[i];
        assert_eq!(
            commitment_hash(&record.mu, bits.try_into().unwrap()),
            record.c_tilde
        );
    }
    assert!(public_bits(&output.proof, n).is_none());

    let public = fixture.public();
    let uncached = verify_mode_b(&public, &bytes, &bridge, None);
    report_line(&format!("N = {n}: verifier uncached"), &uncached);
    let verified = uncached.outcome.expect("verify");
    assert_eq!(verified.records, n);
    assert_eq!(verified.bridge_seed, output.bridge_seed);
    assert_eq!(verified.session, output.session);
    let mut cache = KeyCache::default();
    let first = verify_mode_b(&public, &bytes, &bridge, Some(&mut cache));
    assert_eq!(
        first
            .outcome
            .expect("verify, filling the cache")
            .expanded_keys,
        verified.expanded_keys
    );
    let cached = verify_mode_b(&public, &bytes, &bridge, Some(&mut cache));
    report_line(&format!("N = {n}: verifier cached keys"), &cached);
    assert_eq!(
        cached
            .outcome
            .expect("verify with cached keys")
            .expanded_keys,
        0
    );
    drop(public);
    (fixture, bridge, output, bytes)
}

fn rejected(report: VerifyReport) -> VerifyError {
    report.outcome.expect_err("the verifier must reject")
}

#[test]
fn mode_b_round_trip_at_256_signatures_and_corrupted_aggregates_are_rejected() {
    on_large_stack(|| {
        let (fixture, bridge, output, bytes) = round_trip(200, 56, 0xb256);
        let public = fixture.public();
        let verify = |proof: &agg_piop::wire::AggregateProof| {
            verify_mode_b(&public, &proof.to_bytes(), &bridge, None)
        };

        // The public bits: a flipped bit breaks the hash, a value of 44 or more the range.
        let mut flipped = output.proof.clone();
        flipped.flock_commitments[0][5 * 768 + 300] ^= 0x10;
        assert!(matches!(rejected(verify(&flipped)), VerifyError::Keccak(0)));
        let mut wide = output.proof.clone();
        wide.flock_commitments[0][9 * 768] |= 0x3f;
        assert!(matches!(rejected(verify(&wide)), VerifyError::Keccak(0)));
        let mut short = output.proof.clone();
        short.flock_commitments[0].truncate(768 * 255);
        assert!(matches!(rejected(verify(&short)), VerifyError::Keccak(0)));
        // A changed c~ fails the hash; a changed final vector fails the projection.
        let mut c_tilde = output.proof.clone();
        c_tilde.c_tilde[17][3] ^= 1;
        assert!(matches!(rejected(verify(&c_tilde)), VerifyError::Keccak(0)));
        let mut final_vector = output.proof.clone();
        final_vector.final_vectors[0][100] ^= 4;
        assert!(matches!(
            rejected(verify(&final_vector)),
            VerifyError::Keccak(0)
        ));
        let mut keccak_proof = output.proof.clone();
        keccak_proof.flock_proofs[0] = vec![0];
        assert!(matches!(
            rejected(verify(&keccak_proof)),
            VerifyError::Keccak(0)
        ));

        // PIOP and opening corruptions.
        let mut prestage = output.proof.clone();
        prestage.prestage.kappa3[1] = prestage.prestage.kappa3[0];
        assert!(matches!(
            rejected(verify(&prestage)),
            VerifyError::Stage(StageError::Terminal(1))
        ));
        let mut round = output.proof.clone();
        round.stage2.rounds[3][1] = round.stage2.rounds[3][0];
        assert!(matches!(
            rejected(verify(&round)),
            VerifyError::Stage(StageError::Terminal(2))
        ));
        let mut stage3 = output.proof.clone();
        stage3.stage3.rounds[0][0] = stage3.stage3.rounds[0][1];
        assert!(matches!(
            rejected(verify(&stage3)),
            VerifyError::Stage(StageError::Terminal(3))
        ));
        let mut claim = output.proof.clone();
        claim.stage3.claims.0[20] = claim.stage3.claims.0[21];
        assert!(matches!(
            rejected(verify(&claim)),
            VerifyError::Stage(StageError::Terminal(3))
        ));
        let mut opening = output.proof.clone();
        let last = opening.opening.len() - 1;
        opening.opening[last] ^= 0x20;
        assert!(matches!(
            rejected(verify(&opening)),
            VerifyError::Opening(_)
        ));

        // Records, profile and framing.
        let mut swapped = public.clone();
        swapped.swap(10, 11);
        assert!(!verify_mode_b(&swapped, &bytes, &bridge, None).accepted());
        assert!(matches!(
            rejected(verify_mode_b(&public[1..], &bytes, &bridge, None)),
            VerifyError::RecordCount
        ));
        // Event 0 binds the profile: under profile 0 the transcript, hence sd and the
        // seed-derived matrices, change, and the projection of the public bits no longer
        // matches Y_sigma.
        let other_profile = verify_timed(
            &public,
            &bytes,
            &bridge,
            &AsProfile(FlockProfile::None),
            FlockProfile::None,
            None,
        );
        assert!(matches!(rejected(other_profile), VerifyError::Keccak(0)));
        // A Keccak verifier for another profile is refused before the proof is read.
        let mismatch = verify_timed(
            &public,
            &bytes,
            &bridge,
            &PublicBitsVerifier,
            FlockProfile::None,
            None,
        );
        assert!(matches!(
            rejected(mismatch),
            VerifyError::Configuration(ConfigurationError::ProfileMismatch { .. })
        ));

        // The matrices are those of the transcript's sd: the same matrices as a fixed
        // stand-in verify, the matrices of any other seed (or test matrices) do not.
        let verified_seed = verify_mode_b(&public, &bytes, &bridge, None)
            .outcome
            .expect("verify")
            .bridge_seed;
        assert_eq!(verified_seed, output.bridge_seed);
        let same = FixedMatrices(derive_matrices(&output.bridge_seed));
        assert!(verify_mode_b(&public, &bytes, &same, None).accepted());
        let mut other_seed = output.bridge_seed;
        other_seed[0] ^= 1;
        let other_seed = FixedMatrices(derive_matrices(&other_seed));
        assert!(matches!(
            rejected(verify_mode_b(&public, &bytes, &other_seed, None)),
            VerifyError::Keccak(0)
        ));
        let other = FixedMatrices(BridgeMatrices::random_for_tests(0x99));
        assert!(matches!(
            rejected(verify_mode_b(&public, &bytes, &other, None)),
            VerifyError::Keccak(0)
        ));
        assert!(matches!(
            rejected(verify_mode_b(
                &public,
                &bytes[..bytes.len() - 1],
                &bridge,
                None
            )),
            VerifyError::Wire(_)
        ));
    });
}

#[test]
#[cfg_attr(debug_assertions, ignore = "1,024 signatures: run with --release")]
fn mode_b_round_trip_at_1024_signatures() {
    on_large_stack(|| {
        let _ = round_trip(1000, 24, 0xb1024);
    });
}

/// The prover run on tampered tables (unchecked stages) with the honest public bits.
fn prove_tampered(
    fixture: &Fixture,
    bridge: &FixedMatrices,
    main: &MainTables,
    tamper_bridge: Option<BridgeTamper<'_>>,
) -> Result<Vec<u8>, String> {
    let inputs = fixture.inputs();
    let signatures = fixture.signatures();
    let prover_inputs = ProverInputs {
        instance: &fixture.instance,
        records: &inputs,
        signatures: &signatures,
    };
    let (manifest, honest) = main_tables(&prover_inputs).map_err(|e| e.to_string())?;
    let side = PublicBits::from_tables(&manifest, &honest);
    let hooks = ProveHooks {
        unchecked: true,
        tamper_bridge,
    };
    prove_with_tables(
        &prover_inputs,
        &manifest,
        main,
        bridge,
        &side,
        FlockProfile::PublicBits,
        &hooks,
    )
    .map(|output| output.proof.to_bytes())
    .map_err(|e| e.to_string())
}

/// Mutable access to one `G_main` cell.
fn cell<'a>(
    manifest: &CommitmentManifest,
    main: &'a mut MainTables,
    column: Column,
    x: usize,
) -> &'a mut i8 {
    let (role, position) = manifest.locate(column);
    assert_eq!(role, GroupRole::Main);
    &mut main.columns_mut()[position][x]
}

fn digit(value: DigitValue, digit: u8) -> Column {
    Column::Digit { value, digit }
}

#[test]
fn every_stage_tampering_is_rejected_end_to_end() {
    on_large_stack(|| {
        // 50 live records (N* = 64, d_main = 16) keep the 12 prover runs short.
        let fixture = Fixture::new(0x7a, 40, 10);
        let bridge = FixedMatrices(BridgeMatrices::random_for_tests(0x7a));
        let inputs = fixture.inputs();
        let signatures = fixture.signatures();
        let (manifest, honest) = main_tables(&ProverInputs {
            instance: &fixture.instance,
            records: &inputs,
            signatures: &signatures,
        })
        .unwrap();
        let public = fixture.public();
        let verify = |bytes: &[u8]| verify_mode_b(&public, bytes, &bridge, None);
        let honest_bytes = prove_tampered(&fixture, &bridge, &honest, None).unwrap();
        assert!(
            verify(&honest_bytes).accepted(),
            "the unchecked prover is honest on honest tables"
        );

        let x = main_index(17, 2, 5);
        let tampered = |f: &dyn Fn(&mut MainTables)| {
            let mut main = honest.clone();
            f(&mut main);
            main
        };
        let step = |d: i8| if d < 7 { d + 1 } else { d - 1 };
        let expect = |label: &str, main: &MainTables, check: &dyn Fn(&VerifyError) -> bool| {
            let bytes = prove_tampered(&fixture, &bridge, main, None).expect(label);
            let error = rejected(verify(&bytes));
            eprintln!("{label}: rejected with {error}");
            assert!(check(&error), "{label}: {error:?}");
        };
        let at_stage = |s: usize| move |e: &VerifyError| matches!(e, VerifyError::Stage(StageError::Terminal(t)) if *t == s);

        // A z digit or a v digit changed inside its alphabet: the ring identity (stage 3).
        expect(
            "z digit",
            &tampered(&|m| {
                *cell(&manifest, m, digit(DigitValue::Z, 0), x) =
                    step(*cell(&manifest, m, digit(DigitValue::Z, 0), x))
            }),
            &at_stage(3),
        );
        expect(
            "v digit",
            &tampered(&|m| {
                *cell(&manifest, m, digit(DigitValue::V, 0), x) =
                    step(*cell(&manifest, m, digit(DigitValue::V, 0), x))
            }),
            &at_stage(3),
        );
        // A committed bit flipped: committed and public bits differ, so the bridge final
        // vector no longer matches the projection of the public bits.
        expect(
            "flipped bit",
            &tampered(&|m| *cell(&manifest, m, Column::W1Bit(0), x) ^= 1),
            &|e| matches!(e, VerifyError::Keccak(0)),
        );
        // A non-Boolean bit (0 -> 2 keeps every parity): Booleanity in stage 1.
        let even = (0..1024)
            .map(|p| main_index(p % 256, p / 256, 7))
            .find(|&y| *cell(&manifest, &mut honest.clone(), Column::W1Bit(1), y) == 0)
            .unwrap();
        expect(
            "bit = 2",
            &tampered(&|m| *cell(&manifest, m, Column::W1Bit(1), even) = 2),
            &at_stage(1),
        );
        // w1' = 44 in the committed bits only: the projection; in both copies: the range.
        let w44 = tampered(&|m| {
            for (t, bit) in [0, 0, 1, 1, 0, 1].into_iter().enumerate() {
                *cell(&manifest, m, Column::W1Bit(t as u8), x) = bit;
            }
        });
        expect("w1' = 44", &w44, &|e| matches!(e, VerifyError::Keccak(0)));
        // A base-16 digit outside its alphabet: the width-4 commitment cannot hold 9.
        let wide = tampered(&|m| *cell(&manifest, m, digit(DigitValue::U, 1), x) = 9);
        match prove_tampered(&fixture, &bridge, &wide, None) {
            Err(error) => eprintln!("base-16 digit 9: the prover's commitment fails ({error})"),
            Ok(bytes) => {
                let error = rejected(verify(&bytes));
                eprintln!("base-16 digit 9: rejected with {error}");
                assert!(at_stage(1)(&error), "{error:?}");
            }
        }
        // Base-4 and base-8 digits outside their alphabets: the leaves of stage 2.
        expect(
            "base-4 digit",
            &tampered(&|m| *cell(&manifest, m, digit(DigitValue::Z, 4), x) = 3),
            &at_stage(2),
        );
        expect(
            "base-8 digit",
            &tampered(&|m| *cell(&manifest, m, digit(DigitValue::U, 4), x) = -6),
            &at_stage(2),
        );

        // Round-2 tampering: a level-1 carry digit, a level-2 parity, a final-vector bit.
        let run_bridge = |label: &str,
                          tamper: &dyn Fn(&CommitmentManifest, &mut BridgeGroupTables),
                          check: &dyn Fn(&VerifyError) -> bool| {
            let bytes = prove_tampered(&fixture, &bridge, &honest, Some(tamper)).expect(label);
            let error = rejected(verify(&bytes));
            eprintln!("{label}: rejected with {error}");
            assert!(check(&error), "{label}: {error:?}");
        };
        run_bridge(
            "level-1 carry digit",
            &|_, b| {
                let column = &mut b.group_mut(GroupRole::Level1)[0];
                column[3] = step(column[3]);
            },
            &at_stage(1),
        );
        run_bridge(
            "level-2 parity",
            &|_, b| b.group_mut(GroupRole::Level2)[2][5] ^= 1,
            &at_stage(1),
        );
        run_bridge(
            "final-vector bit",
            &|m, b| {
                let mut tables = b.to_witness_tables();
                tables.y[11] ^= 1;
                *b = BridgeGroupTables::from_tables(m, tables).unwrap();
            },
            &|e| matches!(e, VerifyError::Keccak(0)),
        );
    });
}

/// The mode-B checks under another declared profile.
struct AsProfile(FlockProfile);

impl KeccakVerifier for AsProfile {
    fn profile(&self) -> FlockProfile {
        self.0
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        PublicBitsVerifier.verify(segment)
    }

    fn check_bits(&self, bits: &BitsContext<'_>) -> bool {
        PublicBitsVerifier.check_bits(bits)
    }
}

/// The projection predicate alone (check 4), without the direct bit check (check 5).
struct ProjectionOnly;

impl KeccakVerifier for ProjectionOnly {
    fn profile(&self) -> FlockProfile {
        FlockProfile::PublicBits
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        PublicBitsVerifier.verify(segment)
    }
}

#[test]
fn committed_bits_other_than_the_public_ones_are_caught_by_the_bit_check() {
    on_large_stack(|| {
        // Fixed matrices with two equal M_1 columns let a prover flip b_0 at both positions
        // without changing any parity, and u absorbs the change of w1' in the ring identity:
        // every PIOP check passes, the projection of the public bits matches, and only the
        // direct check of b_t(r^(3)) against the public bits rejects (deviation P9).
        let fixture = Fixture::new(0x5b, 20, 4);
        let inputs = fixture.inputs();
        let signatures = fixture.signatures();
        let (manifest, honest) = main_tables(&ProverInputs {
            instance: &fixture.instance,
            records: &inputs,
            signatures: &signatures,
        })
        .unwrap();
        let read =
            |m: &MainTables, column: Column, x: usize| m.columns()[manifest.locate(column).1][x];
        let u_at = |m: &MainTables, x: usize| {
            let digits: Vec<i64> = (0..5)
                .map(|k| i64::from(read(m, digit(DigitValue::U, k), x)))
                .collect();
            U_DIGITS.compose(&digits)
        };
        // Two positions of record 3 whose u can absorb -2 gamma_2 (w1' bit 0 flipped).
        let shifted = |x: usize| {
            let delta = if read(&honest, Column::W1Bit(0), x) == 0 {
                1
            } else {
                -1
            };
            let u = u_at(&honest, x) - 2 * GAMMA2 * delta;
            (-(1 << 18)..(1 << 18)).contains(&u).then_some(u)
        };
        let positions: Vec<(usize, i64)> = (0..1024)
            .map(|p| main_index(p % 256, p / 256, 3))
            .filter_map(|x| shifted(x).map(|u| (x, u)))
            .take(2)
            .collect();
        assert_eq!(positions.len(), 2);
        let mut tampered = honest.clone();
        for &(x, u) in &positions {
            *cell(&manifest, &mut tampered, Column::W1Bit(0), x) ^= 1;
            for (k, d) in U_DIGITS.decompose(u).into_iter().enumerate() {
                *cell(&manifest, &mut tampered, digit(DigitValue::U, k as u8), x) = d as i8;
            }
        }
        let mut matrices = BridgeMatrices::random_for_tests(0x5c);
        let (z1, z2) = (positions[0].0 & (J1 - 1), positions[1].0 & (J1 - 1));
        let column: Vec<u32> = matrices.m1.column(z1).to_vec();
        matrices.m1.entries[z2 * K1..(z2 + 1) * K1].copy_from_slice(&column);
        let bridge = FixedMatrices(matrices);

        let bytes = prove_tampered(&fixture, &bridge, &tampered, None).expect("prove");
        let public = fixture.public();
        let report = verify_mode_b(&public, &bytes, &bridge, None);
        assert!(
            matches!(report.outcome, Err(VerifyError::Bits)),
            "{:?}",
            report.outcome
        );
        let without = verify_timed(
            &public,
            &bytes,
            &bridge,
            &ProjectionOnly,
            FlockProfile::PublicBits,
            None,
        );
        assert!(
            without.accepted(),
            "the projection alone does not bind the bits under fixed matrices"
        );
    });
}

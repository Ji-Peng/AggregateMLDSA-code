//! The batched main-table build on every available SIMD backend equals the record-by-record
//! reference build byte for byte, under every relation, errors included.

use agg_mldsa_simd::{Backend, Engine};
use agg_piop::manifest::{CommitmentManifest, Relation};
use agg_piop::source_tables::MainTables;
use agg_witness::gen::{self, Record};
use agg_witness::instance::PublicInstance;
use agg_witness::params::{C_TILDE_BYTES, L, OMEGA, Z_BITS};

const RELATIONS: [Relation; 2] = Relation::ALL;

fn engines() -> Vec<Engine> {
    [Backend::Scalar, Backend::Avx2, Backend::Avx512]
        .into_iter()
        .filter_map(Engine::new)
        .collect()
}

fn instance(engine: &Engine, records: &[Record]) -> PublicInstance {
    let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
        .iter()
        .map(|r| {
            let mut c = [0u8; C_TILDE_BYTES];
            let n = r.sig.len().min(C_TILDE_BYTES);
            c[..n].copy_from_slice(&r.sig[..n]);
            (r.pk.as_slice(), r.m_prime.as_slice(), c)
        })
        .collect();
    PublicInstance::new_with(engine, &inputs).unwrap()
}

fn assert_builds_agree(engine: &Engine, records: &[Record]) {
    let inst = instance(engine, records);
    let sigs: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
    for relation in RELATIONS {
        let manifest = CommitmentManifest::for_relation(records.len(), relation).unwrap();
        let fast = MainTables::build_with(engine, &manifest, &inst, &sigs).unwrap();
        let reference = MainTables::build_reference(&manifest, &inst, &sigs).unwrap();
        assert!(
            fast == reference,
            "{:?}, {relation:?}, {} records",
            engine.backend(),
            records.len()
        );
    }
}

#[test]
fn main_tables_match_the_reference_build() {
    // 64 records: own keys and three shared keys.
    let mut small = gen::records(0xF1, 40);
    small.extend(gen::records_shared_keys(0xF2, 24, 3));
    // 1,024 records: 200 own keys and 824 records over 37 keys, plus a hint corner case.
    let mut large = gen::records(0xF3, 200);
    large.extend(gen::records_shared_keys(0xF4, 823, 37));
    large.push(gen::hint_corner_record(0xF5));
    for engine in engines() {
        assert_builds_agree(&engine, &small);
        assert_builds_agree(&engine, &large);
        // Odd counts: partial tiles and padded rows.
        assert_builds_agree(&engine, &small[..37]);
        assert_builds_agree(&engine, &large[..1001]);
    }
    // The default entry point is the process-wide engine's build.
    let inst = PublicInstance::from_records(&small).unwrap();
    let sigs: Vec<&[u8]> = small.iter().map(|r| r.sig.as_slice()).collect();
    let manifest = CommitmentManifest::for_relation(small.len(), Relation::HFips).unwrap();
    assert!(
        MainTables::build(&manifest, &inst, &sigs).unwrap()
            == MainTables::build_reference(&manifest, &inst, &sigs).unwrap()
    );
}

/// A named change to one signature.
type Mutation = Box<dyn Fn(&mut Vec<u8>)>;

#[test]
fn a_bad_record_fails_identically_on_both_builds() {
    let records = gen::records_shared_keys(0xF6, 21, 4);
    let other = gen::record(0xF7, 0);
    let hint = C_TILDE_BYTES + L * 32 * Z_BITS;
    let mutations: [(&str, Mutation); 6] = [
        (
            "short",
            Box::new(|s| {
                s.pop();
            }),
        ),
        ("hint", Box::new(move |s| s[hint + OMEGA] = 81)),
        (
            "z norm",
            Box::new(|s| {
                s[C_TILDE_BYTES] = 0;
                s[C_TILDE_BYTES + 1] = 0;
                s[C_TILDE_BYTES + 2] &= !0x03;
            }),
        ),
        ("z byte", Box::new(|s| s[C_TILDE_BYTES + 100] ^= 4)),
        ("c~", Box::new(|s| s[5] ^= 1)),
        ("other key", Box::new(move |s| *s = other.sig.clone())),
    ];
    for engine in engines() {
        for (name, mutate) in &mutations {
            for k in [0usize, 7, 8, 20] {
                for before in [true, false] {
                    let mut bad = records.clone();
                    if before {
                        mutate(&mut bad[k].sig);
                    }
                    let inst = instance(&engine, &bad);
                    if !before {
                        mutate(&mut bad[k].sig);
                    }
                    let sigs: Vec<&[u8]> = bad.iter().map(|r| r.sig.as_slice()).collect();
                    for relation in RELATIONS {
                        let manifest =
                            CommitmentManifest::for_relation(bad.len(), relation).unwrap();
                        let fast =
                            MainTables::build_with(&engine, &manifest, &inst, &sigs).unwrap_err();
                        let reference =
                            MainTables::build_reference(&manifest, &inst, &sigs).unwrap_err();
                        assert_eq!(
                            fast, reference,
                            "{name} at {k}, before {before}, {relation:?}"
                        );
                        assert!(
                            matches!(reference, agg_piop::source_tables::SourceTableError::Derive(i, _) if i == k),
                            "{name} at {k}: {reference:?}"
                        );
                    }
                }
            }
        }
    }
}

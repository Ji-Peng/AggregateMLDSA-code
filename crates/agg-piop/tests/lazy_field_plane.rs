//! A default aggregate must read the narrow columns through their signed
//! mirrors and leave every canonical field plane unexpanded.
//!
//! This is a separate test binary because the counter is process-wide: the
//! other integration tests run concurrently and may legitimately materialize a
//! plane through a fallback reader.

mod common;

use agg_piop::manifest::Relation;
use agg_piop::mode_b::prove_mode_b;
use agg_piop::prover::{ProverInputs, SeededMatrices};
use agg_witness::instance::PublicInstance;
use agg_witness::params::C_TILDE_BYTES;

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("test thread");
}

#[test]
fn default_aggregate_builds_no_lazy_field_plane() {
    on_large_stack(|| {
        let records = common::records(0x1a2f, 60, 4);
        let instance = PublicInstance::from_records(&records).expect("instance");
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|record| {
                let mut c_tilde = [0_u8; C_TILDE_BYTES];
                c_tilde.copy_from_slice(&record.sig[..C_TILDE_BYTES]);
                (record.pk.as_slice(), record.msg.as_slice(), c_tilde)
            })
            .collect();
        let signatures: Vec<&[u8]> = records.iter().map(|record| record.sig.as_slice()).collect();
        let prover_inputs = ProverInputs {
            instance: &instance,
            records: &inputs,
            signatures: &signatures,
            relation: Relation::Hu,
        };

        let before = akita_cpu_backend::diagnostics::lazy_field_plane_builds();
        let _ = prove_mode_b(&prover_inputs, &SeededMatrices).expect("prove");
        let after = akita_cpu_backend::diagnostics::lazy_field_plane_builds();
        assert_eq!(
            after,
            before,
            "the default aggregate expanded {} narrow field plane(s)",
            after - before
        );
    });
}

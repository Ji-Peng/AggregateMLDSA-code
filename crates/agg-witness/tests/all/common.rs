//! Shared helpers of the integration tests.

use agg_witness::derive::{derive, SigWitness};
use agg_witness::gen::{self, Record};
use agg_witness::instance::PublicInstance;
use agg_witness::native::mod_q;
use agg_witness::params::*;
use rayon::prelude::*;

/// Edge record: hedged signature (seed, key index, index) whose w' has a coefficient equal to
/// the Decompose edge residue q - gamma2 (found by `edge_search`, see tests/all/edge.rs).
pub const EDGE_RESIDUE: (u64, u64, u64) = (0xED6E, 0, 783);
/// Edge record whose z has a coefficient with |z| = B_z (found by `edge_search`).
pub const EDGE_ZMAX: (u64, u64, u64) = (0x2A4A, 0, 290);

pub fn witnesses(inst: &PublicInstance, records: &[Record]) -> Vec<SigWitness> {
    inst.records
        .par_iter()
        .zip(records.par_iter())
        .map(|(p, r)| derive(&p.key, &p.mu, &r.sig).expect("honest record derives"))
        .collect()
}

/// Count of positions of a witness whose w' (mod q) lies in [q - gamma2, q - 1].
pub fn edge_positions(w: &SigWitness) -> usize {
    w.w.iter()
        .flatten()
        .filter(|&&x| mod_q(x) >= Q - GAMMA2)
        .count()
}

/// The standard mixed test set: own-key deterministic records plus hedged shared-key records.
pub fn mixed_records(seed: u64, n_own: usize, n_shared: usize) -> Vec<Record> {
    let mut r = gen::records(seed, n_own);
    r.extend(gen::records_shared_keys(seed + 1, n_shared, 3));
    r
}

//! One-off search for edge-case seeds (run with `--ignored`); results are pinned in `common.rs`.

use agg_witness::derive::derive;
use agg_witness::expand::ExpandedKey;
use agg_witness::gen;
use agg_witness::native::{inf_norm, mod_q, mu};
use agg_witness::params::*;
use rayon::prelude::*;

fn find(
    seed: u64,
    limit: u64,
    pred: impl Fn(&agg_witness::derive::SigWitness) -> bool + Sync,
) -> Option<u64> {
    let (_, pk) = gen::keypair(&gen::key_seed(seed, 0));
    let key = ExpandedKey::from_bytes(&pk).unwrap();
    (0..limit).into_par_iter().find_first(|&i| {
        let r = gen::record_hedged(seed, 0, i);
        let m = mu(&key.tr, &r.m_prime);
        let w = derive(&key, &m, &r.sig).unwrap();
        pred(&w)
    })
}

#[test]
#[ignore = "brute-force search for witness edge cases; run with: cargo test --release -p agg-witness --test all edge_search -- --ignored --nocapture"]
fn edge_search() {
    let residue = find(0xED6E, 200_000, |w| {
        w.w.iter().flatten().any(|&x| mod_q(x) == Q - GAMMA2)
    });
    println!("EDGE_RESIDUE index: {residue:?}");
    let zmax = find(0x2A4A, 200_000, |w| inf_norm(&w.z) == B_Z);
    println!("EDGE_ZMAX index: {zmax:?}");
}

//! The tied part of ML-DSA-65 and ML-DSA-87: four bit lanes of `P_pad = 2,048` positions per
//! record, `c~` of 48 and 64 bytes, 7 and 9 permutations. Round trips on random records at 5
//! records (`d = 3`), 300 (`d = 9`, two level-1 blocks of 256 records) and a span of two
//! segments, with the rejections that depend on the layout: a flipped final-vector bit, a
//! wrong `c~`, the chain's copy flipped at a live and at a padded position (ML-DSA-65: rows 6
//! and 7 carry no bit), and a proof checked under the other set's gate. Run with `--release`.

use std::sync::Arc;

use agg_bridge::{seed::derive_matrices_for, shape::Shape};
use agg_keccak::prover::{KeccakError, KeccakProof, Profile};
use agg_keccak::sponge::{sponge_gate_for, SpongeLayout, SpongeRecord};
use agg_keccak::tied::TiedPart;
use agg_witness::params::{ParamSet, Set65, Set87};
use flock_core::test_rng::Rng;

fn random_record<P: ParamSet>(rng: &mut Rng) -> SpongeRecord<P> {
    let mut mu = [0u8; 64];
    let mut w1 = P::zero_w1_encode();
    mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    w1.as_mut().iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    SpongeRecord { mu, w1_encode: w1 }
}

fn round_trip<P: ParamSet, const C: usize>(n: usize, d: u32, segments: usize, seed: u64) {
    let gate = sponge_gate_for(P::SET);
    let layout = SpongeLayout::of(P::SET);
    let shape = Shape::production(P::SET);
    let part = TiedPart::new_span(&gate, n, d, 0, segments, Profile::Fast);
    let mut rng = Rng::new(seed);
    let records: Vec<SpongeRecord<P>> = (0..n).map(|_| random_record(&mut rng)).collect();
    let mu: Vec<[u8; 64]> = records.iter().map(|r| r.mu).collect();
    let c_tilde: Vec<[u8; C]> = records
        .iter()
        .map(|r| r.c_tilde().as_ref().try_into().expect("c~ width"))
        .collect();
    let committed = part.commit(&records);
    assert_eq!(committed.c_tilde::<C>(), &c_tilde[..]);
    let commitment = committed.commitment().clone();
    let sd = [seed as u8; 32];
    let mats = Arc::new(derive_matrices_for(&shape, &sd));
    let fs = [7; 32];
    let out = part.prove(committed, sd, &mats, &fs).expect("prove");
    let verify = |ys: &[Vec<u8>], c: &[[u8; C]], proof: &KeccakProof| {
        part.verify(&mu, c, &commitment, sd, &mats, ys, &fs, proof)
    };
    verify(&out.final_vectors, &c_tilde, &out.proof).expect("honest");
    eprintln!(
        "{} tied part n {n} d {d} segments {segments}: proof {} B, sponge nu {}, chain rounds {}",
        P::NAME,
        out.proof.bytes.len(),
        part.nu(),
        part.cost().chain_rounds
    );
    // A flipped final-vector bit.
    let mut bad = out.final_vectors.clone();
    bad[0][100] ^= 1;
    assert!(verify(&bad, &c_tilde, &out.proof).is_err(), "flipped Y bit");
    // A wrong c~ of the last record changes the public digest the sponge challenger
    // absorbs: Flock's transcript rejects.
    let mut wrong = c_tilde.clone();
    wrong[n - 1][C - 1] ^= 0x10;
    assert!(verify(&out.final_vectors, &wrong, &out.proof).is_err(), "wrong c~");
    // Both copies of a tampered record (the sponge recomputed, Y consistent with it) under
    // the stale statement: the last c~ word the sponge commits is not the published one, and
    // the sponge instance's public claim fails.
    let mut tampered = records.clone();
    tampered[n - 1].w1_encode.as_mut()[200] ^= 0x10;
    let (sponge, bits) = part.honest_parts(&tampered);
    let committed = part.commit_parts(sponge, bits, &records);
    let com_t = committed.commitment().clone();
    let ys = part.final_vectors(&committed, &mats);
    let proof = part
        .prove_with_final_vectors(committed, sd, &mats, ys.clone(), &fs)
        .expect("prove");
    let got = part.verify(&mu, &c_tilde, &com_t, sd, &mats, &ys, &fs, &proof);
    assert!(matches!(got, Err(KeccakError::Tie("public words"))), "{got:?}");
    // The other set's c~ width.
    let other: Vec<[u8; 32]> = c_tilde.iter().map(|c| c[..32].try_into().unwrap()).collect();
    assert!(matches!(
        part.verify(&mu, &other, &commitment, sd, &mats, &out.final_vectors, &fs, &out.proof),
        Err(KeccakError::Statement("c~ width"))
    ));
    // The chain's copy flipped at a live position (lane 2, row 1 of the first record) and, for
    // ML-DSA-65, at a padded position (lane 1, row 6): the sponge commits the honest bits
    // and zero in the padded columns, so the chain's final value is not the sponge's.
    let mut positions = vec![(2, 256 + 17)];
    if layout.positions() < layout.positions_pad {
        positions.push((1, 6 * 256 + 3));
    }
    for (t, p) in positions {
        let (sponge, mut bits) = part.honest_parts(&records);
        let col = part.segment().column(t, p);
        bits[col / 64] ^= 1 << (col % 64);
        let committed = part.commit_parts(sponge, bits, &records);
        assert_eq!(*committed.commitment(), commitment, "the sponge is honest");
        let ys = part.final_vectors(&committed, &mats);
        assert_ne!(ys, out.final_vectors);
        let proof = part
            .prove_with_final_vectors(committed, sd, &mats, ys.clone(), &fs)
            .expect("prove");
        assert!(
            matches!(verify(&ys, &c_tilde, &proof), Err(KeccakError::Tie("parity chain"))),
            "lane {t}, position {p}"
        );
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
fn tied_parts_of_65_round_trip() {
    round_trip::<Set65, 48>(5, 3, 1, 0x65);
    round_trip::<Set65, 48>(300, 9, 1, 0x165);
    round_trip::<Set65, 48>(12, 3, 2, 0x265);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
fn tied_parts_of_87_round_trip() {
    round_trip::<Set87, 64>(5, 3, 1, 0x87);
    round_trip::<Set87, 64>(300, 9, 1, 0x187);
    round_trip::<Set87, 64>(12, 3, 2, 0x287);
}

/// A proof is bound to its gate: the ML-DSA-65 proof does not verify as an ML-DSA-87 part of
/// the same geometry, whose `c~` and layout differ (the statement check rejects it first).
#[test]
#[cfg_attr(debug_assertions, ignore = "Flock proofs; run with --release")]
fn tied_proofs_are_bound_to_the_parameter_set() {
    let (g65, g87) = (sponge_gate_for(Set65::SET), sponge_gate_for(Set87::SET));
    assert_ne!(g65.digest(), g87.digest());
    let part = TiedPart::new_part(&g65, 5, 3, 0, Profile::Fast);
    let mut rng = Rng::new(3);
    let records: Vec<SpongeRecord<Set65>> = (0..5).map(|_| random_record(&mut rng)).collect();
    let committed = part.commit(&records);
    let commitment = committed.commitment().clone();
    let sd = [1; 32];
    let mats = derive_matrices_for(&Shape::production(Set65::SET), &sd);
    let out = part.prove(committed, sd, &mats, &[2; 32]).expect("prove");
    let mu: Vec<[u8; 64]> = records.iter().map(|r| r.mu).collect();
    let ct: Vec<[u8; 64]> = records
        .iter()
        .map(|r| {
            let mut c = [0u8; 64];
            c[..48].copy_from_slice(r.c_tilde().as_ref());
            c
        })
        .collect();
    let other = TiedPart::new_part(&g87, 5, 3, 0, Profile::Fast);
    assert!(other
        .verify(&mu, &ct, &commitment, sd, &mats, &out.final_vectors, &[2; 32], &out.proof)
        .is_err());
}

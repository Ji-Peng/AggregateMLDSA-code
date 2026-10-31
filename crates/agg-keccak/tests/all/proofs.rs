//! End-to-end Flock proofs of the Keccak gate and of the batched SHAKE256
//! sponge: honest round trips, wrong statements, tampered witnesses, tampered
//! proof bytes, and the commit-then-prove order of the PIOP transcript (the
//! commitment is absorbed before the seed is derived, and the verifier checks
//! the proof against exactly that commitment). Run with `--release` (proofs at
//! up to 2^27 bits).

use agg_keccak::{
    gate::{ChainGate, ChainSpec},
    keccak,
    prover::{GateBatch, KeccakCommitment, KeccakError, KeccakParams, Profile, SpongeProver},
    sponge::{self, SpongeRecord, W1_INPUT_BASE},
    witness::pack_slot,
};
use flock_core::test_rng::Rng;
use flock_field::F128;

fn random_record(rng: &mut Rng) -> SpongeRecord {
    let mut mu = [0u8; 64];
    let mut w1 = [0u8; 768];
    mu.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    w1.iter_mut().for_each(|b| *b = rng.next_u32() as u8);
    SpongeRecord { mu, w1_encode: w1 }
}

fn records(n: usize, seed: u64) -> Vec<SpongeRecord> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| random_record(&mut rng)).collect()
}

const FS: [u8; 32] = [7u8; 32];

fn lanes_to_words(s: &keccak::Lanes) -> Vec<F128> {
    let mut v: Vec<F128> = s
        .chunks(2)
        .map(|c| F128::new(c[0], *c.get(1).unwrap_or(&0)))
        .collect();
    v.truncate(13);
    v
}

/// Keccak-f gate: 64 random states, public input and output states.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn permutation_gate_proof_round_trip() {
    let gate = ChainGate::new(ChainSpec::permutation());
    let n = 64;
    let batch = GateBatch::new(&gate, n, Profile::Fast);
    assert_eq!(batch.words_per_instance(), 26);
    let mut rng = Rng::new(1);
    let mut inputs = Vec::new();
    let mut public = Vec::new();
    for _ in 0..n {
        let s0: keccak::Lanes = std::array::from_fn(|_| rng.next_u64());
        let mut s1 = s0;
        keccak::keccak_f(&mut s1);
        public.extend(lanes_to_words(&s0));
        public.extend(lanes_to_words(&s1));
        inputs.push(s0.to_vec());
    }
    let (commitment, proof) = batch.prove(&inputs, &public, &FS);
    batch
        .verify(&public, &commitment, &proof, &FS)
        .expect("honest permutation batch verifies");
    // A wrong output word.
    let mut bad = public.clone();
    bad[26 * 5 + 20] += F128::ONE;
    assert!(batch.verify(&bad, &commitment, &proof, &FS).is_err());
    // A different Fiat-Shamir seed.
    assert!(batch
        .verify(&public, &commitment, &proof, &[8u8; 32])
        .is_err());
}

fn sponge_round_trip(n: usize, seed: u64) {
    let prover = SpongeProver::new(KeccakParams::default());
    let recs = records(n, seed);
    let sub = prover.prove_batch(&recs, &FS);
    let (cm, proof, c_tilde) = (&sub.commitment, &sub.proof, &sub.c_tilde);
    for (r, c) in recs.iter().zip(c_tilde) {
        assert_eq!(*c, r.c_tilde());
    }
    let mu: Vec<[u8; 64]> = recs.iter().map(|r| r.mu).collect();
    prover
        .verify_batch(&mu, c_tilde, cm, proof, &FS)
        .expect("honest sponge batch verifies");
    eprintln!(
        "sponge n={n}: commitment {} bytes, proof {} bytes",
        cm.as_bytes().len(),
        proof.bytes.len()
    );

    // Wrong output: one bit of one c~.
    let mut bad_c = c_tilde.clone();
    bad_c[n / 2][3] ^= 0x10;
    assert!(prover.verify_batch(&mu, &bad_c, cm, proof, &FS).is_err());
    // Wrong public input: one bit of one mu.
    let mut bad_mu = mu.clone();
    bad_mu[n - 1][63] ^= 1;
    assert!(prover
        .verify_batch(&bad_mu, c_tilde, cm, proof, &FS)
        .is_err());
    // Trailing bytes, a truncated proof and a wrong record count are malformed.
    let mut extra = proof.clone();
    extra.bytes.push(0);
    assert!(matches!(
        prover.verify_batch(&mu, c_tilde, cm, &extra, &FS),
        Err(KeccakError::Malformed(_))
    ));
    let mut short = proof.clone();
    short.bytes.truncate(short.bytes.len() / 2);
    assert!(matches!(
        prover.verify_batch(&mu, c_tilde, cm, &short, &FS),
        Err(KeccakError::Malformed(_))
    ));
    assert!(prover
        .verify_batch(&mu[1..], &c_tilde[1..], cm, proof, &FS)
        .is_err());
    // Tampered proof bytes.
    let mut bad_proof = proof.clone();
    let last = bad_proof.bytes.len() - 100;
    bad_proof.bytes[last] ^= 1;
    assert!(prover
        .verify_batch(&mu, c_tilde, cm, &bad_proof, &FS)
        .is_err());
}

#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn sponge_proof_round_trip_64() {
    sponge_round_trip(64, 0x64);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn sponge_proof_round_trip_256() {
    sponge_round_trip(256, 0x256);
}

/// A flipped w1Encode bit with the chi witness of the original input
/// (`a`, `b` recomputed from the tampered `z`, as a cheating prover would):
/// the chi rows no longer hold and the proof is rejected.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn flipped_input_bit_with_stale_witness_is_rejected() {
    let prover = SpongeProver::new(KeccakParams::default());
    let n = 64;
    let recs = records(n, 0xF11);
    let gate = prover.gate();
    let batch = prover.batch(n);
    let c_tilde: Vec<[u8; 32]> = recs.iter().map(SpongeRecord::c_tilde).collect();
    let mu: Vec<[u8; 64]> = recs.iter().map(|r| r.mu).collect();
    let public = SpongeProver::public_words(&mu, &c_tilde);
    let inputs: Vec<Vec<u64>> = recs.iter().map(SpongeRecord::input_bits).collect();
    let target = 17;
    let col = gate.layout().input_col(W1_INPUT_BASE + 4000);
    let buffers = pack_slot(gate, batch.nu(), n, |j, z, a, b| {
        gate.honest_z(&inputs[j], z);
        if j == target {
            z[col / 64] ^= 1 << (col % 64);
        }
        gate.apply_ab(z, a, b);
    });
    let (commitment, proof) = batch.prove_buffers(buffers, &public, &FS);
    assert!(batch.verify(&public, &commitment, &proof, &FS).is_err());
}

/// The pruned last round of P7 (round 23, lanes `(0..3, 0)` only) is still
/// constrained: flip one of its committed AND bits, `t_23[1, 0, 5]`, alone (the
/// output row of `c~` bit 69 then breaks too) and together with that `c~` column
/// and the public `c~` (only the AND row breaks). Both proofs are rejected. The pruned lanes have no
/// column: lane `(4, 0)` of round 23 lies past the useful bits.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn flipped_last_round_chi_bit_is_rejected() {
    let prover = SpongeProver::new(KeccakParams::default());
    let n = 64;
    let recs = records(n, 0xF13);
    let gate = prover.gate();
    let lay = gate.layout();
    let batch = prover.batch(n);
    let c_tilde: Vec<[u8; 32]> = recs.iter().map(SpongeRecord::c_tilde).collect();
    let mu: Vec<[u8; 64]> = recs.iter().map(|r| r.mu).collect();
    let public = SpongeProver::public_words(&mu, &c_tilde);
    let inputs: Vec<Vec<u64>> = recs.iter().map(SpongeRecord::input_bits).collect();
    let target = 23;
    let mut c_flipped = c_tilde.clone();
    c_flipped[target][69 / 8] ^= 1 << (69 % 8);
    let public_flipped = SpongeProver::public_words(&mu, &c_flipped);
    let r_n = gate.spec().n_rounds;
    let t_col = lay.t_col(r_n, sponge::N_PERMS - 1, r_n - 1, 64 + 5);
    assert!(t_col < lay.useful_bits);
    assert_eq!(
        lay.t_col(r_n, sponge::N_PERMS - 1, r_n - 1, 4 * 64),
        lay.useful_bits
    );
    let out_col = lay.output_base + 64 + 5;
    for flip_output in [false, true] {
        let public = if flip_output {
            &public_flipped
        } else {
            &public
        };
        let buffers = pack_slot(gate, batch.nu(), n, |j, z, a, b| {
            gate.honest_z(&inputs[j], z);
            if j == target {
                z[t_col / 64] ^= 1 << (t_col % 64);
                if flip_output {
                    z[out_col / 64] ^= 1 << (out_col % 64);
                }
            }
            gate.apply_ab(z, a, b);
        });
        let (commitment, proof) = batch.prove_buffers(buffers, public, &FS);
        assert!(
            batch.verify(public, &commitment, &proof, &FS).is_err(),
            "flip_output = {flip_output}"
        );
    }
    // The honest witness through the same path verifies.
    let buffers = pack_slot(gate, batch.nu(), n, |j, z, a, b| {
        gate.honest_z(&inputs[j], z);
        gate.apply_ab(z, a, b);
    });
    let (commitment, proof) = batch.prove_buffers(buffers, &public, &FS);
    batch
        .verify(&public, &commitment, &proof, &FS)
        .expect("honest proof");
}

/// A flipped w1Encode bit with a consistent witness: the sponge then outputs a
/// different c~, and claiming the original c~ breaks the public wiring.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn flipped_input_bit_with_consistent_witness_is_rejected() {
    let prover = SpongeProver::new(KeccakParams::default());
    let n = 64;
    let mut recs = records(n, 0xF12);
    let mu: Vec<[u8; 64]> = recs.iter().map(|r| r.mu).collect();
    let c_tilde: Vec<[u8; 32]> = recs.iter().map(SpongeRecord::c_tilde).collect();
    let public = SpongeProver::public_words(&mu, &c_tilde);
    recs[5].w1_encode[100] ^= 4;
    let batch = prover.batch(n);
    let inputs: Vec<Vec<u64>> = recs.iter().map(SpongeRecord::input_bits).collect();
    let (commitment, proof) = batch.prove(&inputs, &public, &FS);
    assert!(batch.verify(&public, &commitment, &proof, &FS).is_err());
}

/// A stand-in for the PIOP transcript: absorb the `S`
/// Flock commitments (the commitment phase), derive `sd`, then one `fs_sigma`
/// per sub-proof (the Flock seed step). Every seed depends on every commitment.
fn transcript_seeds(commitments: &[KeccakCommitment]) -> ([u8; 32], Vec<[u8; 32]>) {
    use sha3::{Digest, Sha3_256};
    let mut h = Sha3_256::new();
    h.update(b"test/event2");
    for c in commitments {
        h.update((c.as_bytes().len() as u64).to_le_bytes());
        h.update(c.as_bytes());
    }
    let sd: [u8; 32] = h.finalize().into();
    let fs = (0..commitments.len() as u64)
        .map(|s| {
            let mut h = Sha3_256::new();
            h.update(b"test/event5");
            h.update(sd);
            h.update(s.to_le_bytes());
            h.finalize().into()
        })
        .collect();
    (sd, fs)
}

/// Regression for the commit-then-prove order (the commitment phase, the bridge-seed step and the Flock seed step): the `S`
/// commitments exist before any seed, the seeds are derived from them, and each
/// proof verifies only against the commitment that was absorbed for its sub-batch.
/// Sub-batching: a target of 500 permutations gives 64 records per proof (the
/// largest power of two with 7 n <= 500); 100 records give 2 proofs.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn commitments_are_absorbed_before_the_seeds_and_bound() {
    let params = KeccakParams {
        profile: Profile::Fast,
        target_permutations: 500,
    };
    assert_eq!(params.records_per_proof(), 64);
    assert_eq!(KeccakParams::default().records_per_proof(), 1 << 13);
    assert_eq!(sponge::N_PERMS * 64, 448);
    let prover = SpongeProver::new(params);
    let recs = records(100, 0x5B);

    // Commit to both sub-batches; no seed exists yet.
    let committed = prover.commit_all(&recs);
    assert_eq!(committed.len(), 2);
    assert_eq!((committed[0].len(), committed[1].len()), (64, 36));
    let commitments: Vec<KeccakCommitment> =
        committed.iter().map(|c| c.commitment().clone()).collect();
    for (c, chunk) in committed.iter().zip(recs.chunks(64)) {
        assert_eq!(
            c.commitment().as_bytes().len(),
            prover.batch(chunk.len()).commitment_len()
        );
        let want: Vec<[u8; 32]> = chunk.iter().map(SpongeRecord::c_tilde).collect();
        assert_eq!(c.c_tilde(), &want[..]);
    }
    eprintln!(
        "commitment bytes per sub-proof: {}",
        commitments[0].as_bytes().len()
    );
    // The commitment is a function of the witness alone.
    assert_eq!(
        prover.commit_batch(&recs[..64]).commitment(),
        &commitments[0]
    );
    let c_tilde: Vec<Vec<[u8; 32]>> = committed.iter().map(|c| c.c_tilde().to_vec()).collect();

    // Seeds from the transcript, then the Flock proofs.
    let (_sd, fs) = transcript_seeds(&commitments);
    let proofs = prover.prove_all(committed, &fs);
    assert_eq!(proofs.len(), 2);
    let mu: Vec<Vec<[u8; 64]>> = recs
        .chunks(64)
        .map(|c| c.iter().map(|r| r.mu).collect())
        .collect();
    for s in 0..2 {
        prover
            .verify_batch(&mu[s], &c_tilde[s], &commitments[s], &proofs[s], &fs[s])
            .expect("sub-batch verifies against its absorbed commitment");
    }

    // A commitment other than the absorbed one is rejected: a flipped cap bit,
    // the commitment of a different witness of the same shape, and the other
    // sub-batch's commitment.
    let mut flipped = commitments[0].as_bytes().to_vec();
    flipped[7] ^= 0x20;
    assert!(matches!(
        prover.verify_batch(
            &mu[0],
            &c_tilde[0],
            &KeccakCommitment::from_bytes(flipped),
            &proofs[0],
            &fs[0]
        ),
        Err(KeccakError::Flock(_))
    ));
    let other = prover.commit_batch(&records(64, 0x5C));
    assert_ne!(other.commitment(), &commitments[0]);
    assert!(prover
        .verify_batch(&mu[0], &c_tilde[0], other.commitment(), &proofs[0], &fs[0])
        .is_err());
    assert!(prover
        .verify_batch(&mu[0], &c_tilde[0], &commitments[1], &proofs[0], &fs[0])
        .is_err());
    // A commitment of the wrong length is malformed, not a panic.
    for len in [
        0,
        31,
        commitments[0].as_bytes().len() - 32,
        commitments[0].as_bytes().len() + 32,
    ] {
        let mut b = commitments[0].as_bytes().to_vec();
        b.resize(len, 0);
        assert!(matches!(
            prover.verify_batch(
                &mu[0],
                &c_tilde[0],
                &KeccakCommitment::from_bytes(b),
                &proofs[0],
                &fs[0]
            ),
            Err(KeccakError::Malformed("commitment length"))
        ));
    }

    // Seeds bound to a different commitment-phase payload do not verify.
    let (_, wrong_fs) = transcript_seeds(&[commitments[1].clone(), commitments[0].clone()]);
    assert!(prover
        .verify_batch(
            &mu[0],
            &c_tilde[0],
            &commitments[0],
            &proofs[0],
            &wrong_fs[0]
        )
        .is_err());
    // The second sub-batch's proof does not verify the first sub-batch's statement.
    assert!(prover
        .verify_batch(&mu[0], &c_tilde[0], &commitments[0], &proofs[1], &fs[0])
        .is_err());
}

/// The Secure profile proves and verifies too.
#[test]
#[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
fn secure_profile_round_trip() {
    let prover = SpongeProver::new(KeccakParams {
        profile: Profile::Secure,
        ..KeccakParams::default()
    });
    let recs = records(16, 0x5EC);
    let sub = prover.prove_batch(&recs, &FS);
    let mu: Vec<[u8; 64]> = recs.iter().map(|r| r.mu).collect();
    prover
        .verify_batch(&mu, &sub.c_tilde, &sub.commitment, &sub.proof, &FS)
        .expect("secure profile verifies");
    let fast = SpongeProver::new(KeccakParams::default());
    assert!(fast
        .verify_batch(&mu, &sub.c_tilde, &sub.commitment, &sub.proof, &FS)
        .is_err());
}

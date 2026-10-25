//! Mode B (spec Section 14, milestone M2): the Keccak-free aggregate. The `w_1'` bits of
//! every live record are published in the clear, so the verifier checks the hash relation
//! (3.3) natively instead of through a Keccak argument, and the PIOP proves (3.1), (3.2)
//! for committed bits tied to the public ones.
//!
//! The public bits take the place of the Keccak side in the transcript (`docs/deviations.md`
//! P9): the event-2 item of segment `sigma` is the concatenation of `w1Encode(w_1'_i)`
//! (768 bytes, FIPS 204 Algorithm 28) of the segment's live records in record order, so
//! the bits are bound before the bridge seed `sd`, as the Flock commitment would be; the
//! event-6 proof items are empty and event 0 carries Flock profile id 3
//! ([`FlockProfile::PublicBits`]). The verifier side ([`PublicBitsVerifier`]) checks, per
//! segment:
//!
//! 1. shape: `768` bytes per live record, empty proof item;
//! 2. range: every 6-bit value of `w1Encode` is below `m_w = 44` (3.2);
//! 3. hash: `c~_i = SHAKE256(mu_i || w1Encode(w_1'_i), 32)` (3.3), `mu_i` from step 2;
//! 4. projection predicate: `Y_sigma = M_3 M_2 M_1 b'` over GF(2) for the public bits `b'`
//!    ([`segment_projection`]), the GF(2) side of the parity bridge (9.1) that the Flock
//!    bridge slot proves in mode A;
//!
//! and after stage 3 ([`KeccakVerifier::check_bits`]):
//!
//! 5. the committed `b_t(r^(3))`, authenticated by the Akita opening, equal the
//!    multilinear extensions of the public bit tables at `r^(3)`
//!    ([`public_bit_evaluations`]).
//!
//! Check 4 with the seed-derived matrices of spec 9.1 (`prover::SeededMatrices`, the
//! derivation of prover and verifier builds) ties the committed bits to the public ones up
//! to `2^-144.6` (spec 9.4); check 5 does so on its own, up to `d_main / |K|`, because the
//! public bits are absorbed before `r^(3)` is sampled, whatever the [`BridgeDerivation`]
//! (the test stand-in `FixedMatrices` included, `docs/deviations.md` P4, P9).

use std::time::Instant;

use aerie_falcon::protocol::algebra::eq_table;
use agg_witness::bridge::{zeta3, BridgeMatrices, J1, M1, M2, M3};
use agg_witness::native::{commitment_hash, w1_decode, w1_encode, SmallK};
use agg_witness::padding::{main_index, Geometry};
use agg_witness::params::{D_POS, K, M_W, N, W1_BITS, W1_ENCODE_BYTES};
use jolt_field::Zero;
use rayon::prelude::*;

use crate::binding::FlockProfile;
use crate::manifest::{CommitmentManifest, Relation};
use crate::prover::{
    main_tables, prove_with_tables, BridgeDerivation, KeccakSide, ProveError, ProveHooks,
    ProverInputs, ProverOutput,
};
use crate::public_data::KeyCache;
use crate::source_tables::MainTables;
use crate::verifier::{verify_timed, BitsContext, KeccakVerifier, SegmentContext, VerifyReport};
use crate::wire::AggregateProof;
use crate::OpeningField as E;

/// Bytes of one record's public bits: `w1Encode(w_1')`.
pub const W1_RECORD_BYTES: usize = W1_ENCODE_BYTES;

/// Records per segment (spec Section 1).
const SEGMENT_RECORDS_LOG2: usize = 13;

/// `w_1'[rho][j] = sum_t 2^t b_t(j, rho, i)` of live record `i` from the committed bit lanes.
#[must_use]
pub fn record_w1(manifest: &CommitmentManifest, main: &MainTables, i: usize) -> SmallK {
    let lanes = main.bit_lanes(manifest);
    let mut w1 = [[0_u8; N]; K];
    for (rho, row) in w1.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            let x = main_index(j, rho, i);
            *value = (0..W1_BITS).fold(0_u8, |acc, t| acc | (((lanes[t][x] & 1) as u8) << t));
        }
    }
    w1
}

/// `w1Encode` of every live record's committed bits (the honest prover's public bits).
#[must_use]
pub fn w1_encodings(
    manifest: &CommitmentManifest,
    main: &MainTables,
) -> Vec<[u8; W1_RECORD_BYTES]> {
    (0..main.geometry().n)
        .into_par_iter()
        .map(|i| w1_encode(&record_w1(manifest, main, i)))
        .collect()
}

/// The prover's Keccak side in mode B: the public bits as the event-2 items.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicBits {
    segments: Vec<Vec<u8>>,
}

impl PublicBits {
    /// Groups the per-record encodings by segment.
    #[must_use]
    pub fn new(geometry: &Geometry, encodings: &[[u8; W1_RECORD_BYTES]]) -> Self {
        assert_eq!(encodings.len(), geometry.n, "one encoding per live record");
        let segments = (0..geometry.segments())
            .map(|sigma| {
                // A trailing segment may hold padded records only (`first >= N`).
                let first = (sigma << SEGMENT_RECORDS_LOG2).min(geometry.n);
                encodings[first..first + geometry.live_in_segment(sigma)].concat()
            })
            .collect();
        Self { segments }
    }

    /// The public bits of the committed bit lanes.
    #[must_use]
    pub fn from_tables(manifest: &CommitmentManifest, main: &MainTables) -> Self {
        Self::new(&main.geometry(), &w1_encodings(manifest, main))
    }

    /// The event-2 items.
    #[must_use]
    pub fn segments(&self) -> &[Vec<u8>] {
        &self.segments
    }
}

impl KeccakSide for PublicBits {
    fn commitments(&self, segments: usize) -> Vec<Vec<u8>> {
        assert_eq!(segments, self.segments.len(), "one item per segment");
        self.segments.clone()
    }

    fn prove(
        &self,
        _bridge_seed: &[u8; 32],
        _matrices: &BridgeMatrices,
        seeds: &[[u8; 32]],
        _final_vectors: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, ProveError> {
        Ok(vec![Vec::new(); seeds.len()])
    }
}

/// The public bits of live record `i` in a mode-B proof, if the proof has that record.
#[must_use]
pub fn public_bits(proof: &AggregateProof, i: usize) -> Option<&[u8]> {
    let item = proof.flock_commitments.get(i >> SEGMENT_RECORDS_LOG2)?;
    let k = i & ((1 << SEGMENT_RECORDS_LOG2) - 1);
    item.get(k * W1_RECORD_BYTES..(k + 1) * W1_RECORD_BYTES)
}

/// `Y_sigma = M_3 M_2 M_1 b'` over GF(2) for the public values `w1[k]` of the segment's
/// live records (index conventions of `agg_witness::bridge`). The six lanes share the
/// matrices, so a lane mask (the 6-bit value `w_1'` itself) is XORed per matrix entry:
/// `14` byte XORs per nonzero position at level 1, per nonzero `y^(1)` row at level 2.
#[must_use]
pub fn segment_projection(
    geometry: &Geometry,
    sigma: usize,
    w1: &[SmallK],
    matrices: &BridgeMatrices,
) -> Vec<u8> {
    assert_eq!(
        w1.len(),
        geometry.live_in_segment(sigma),
        "the segment's live records"
    );
    let first = sigma << SEGMENT_RECORDS_LOG2;
    // The segment's rows of D_1: global index rho_1 + 2^13 beta with beta = i >> 9.
    let rows1 = (1_usize << geometry.d1()).min(1 << 17);
    let mut y1 = vec![0_u8; rows1];
    y1.par_chunks_mut(M1).enumerate().for_each(|(block, rows)| {
        let records = (block << 9)..((block + 1) << 9).min(w1.len());
        for k in records {
            for (rho, row) in w1[k].iter().enumerate() {
                for (j, &mask) in row.iter().enumerate() {
                    if mask != 0 {
                        let zeta = main_index(j, rho, first + k) & (J1 - 1);
                        for &r in matrices.m1.column(zeta) {
                            rows[r as usize] ^= mask;
                        }
                    }
                }
            }
        }
    });
    // Level 2: the in-block index of a D_1 row is its index within the segment.
    let mut y2 = vec![0_u8; M2];
    for (zeta, &mask) in y1.iter().enumerate() {
        if mask != 0 {
            for &r in matrices.m2.column(zeta) {
                y2[r as usize] ^= mask;
            }
        }
    }
    // Level 3: all six lanes into one parity vector.
    let mut y = vec![0_u8; M3];
    for (rho2, &mask) in y2.iter().enumerate() {
        for t in 0..W1_BITS {
            if (mask >> t) & 1 == 1 {
                for &r in matrices.m3.column(zeta3(t, rho2)) {
                    y[r as usize] ^= 1;
                }
            }
        }
    }
    y
}

/// The multilinear extensions `b'_t(r)`, `t < 6`, of the public bit tables on the main
/// domain at `point` (zero on padded rows), from the event-2 items; `None` on a malformed
/// item. Per record the 1,024 equality weights are bucketed by the 6-bit value (64
/// buckets), then each lane sums its 32 buckets.
#[must_use]
pub fn public_bit_evaluations(
    geometry: &Geometry,
    commitments: &[Vec<u8>],
    point: &[E],
) -> Option<[E; W1_BITS]> {
    if point.len() != geometry.d_main() as usize || commitments.len() != geometry.segments() {
        return None;
    }
    let mut items: Vec<(usize, &[u8; W1_RECORD_BYTES])> = Vec::with_capacity(geometry.n);
    for (sigma, item) in commitments.iter().enumerate() {
        if item.len() != geometry.live_in_segment(sigma) * W1_RECORD_BYTES {
            return None;
        }
        let first = sigma << SEGMENT_RECORDS_LOG2;
        let (records, _) = item.as_chunks::<W1_RECORD_BYTES>();
        items.extend(
            records
                .iter()
                .enumerate()
                .map(|(k, bytes)| (first + k, bytes)),
        );
    }
    let eq_pos = eq_table(&point[..D_POS as usize]);
    let eq_sig = eq_table(&point[D_POS as usize..]);
    let zero = [E::zero(); W1_BITS];
    Some(
        items
            .par_iter()
            .map(|&(i, bytes)| {
                let w1 = w1_decode(bytes);
                let mut buckets = [E::zero(); 1 << W1_BITS];
                for (rho, row) in w1.iter().enumerate() {
                    for (j, &value) in row.iter().enumerate() {
                        buckets[usize::from(value)] += eq_pos[main_index(j, rho, 0)];
                    }
                }
                let mut lanes = zero;
                for (value, &bucket) in buckets.iter().enumerate() {
                    for (t, lane) in lanes.iter_mut().enumerate() {
                        if (value >> t) & 1 == 1 {
                            *lane += bucket;
                        }
                    }
                }
                lanes.map(|lane| lane * eq_sig[i])
            })
            .reduce(|| zero, |a, b| std::array::from_fn(|t| a[t] + b[t])),
    )
}

/// The mode-B verifier side (module docs, checks 1 to 5).
pub struct PublicBitsVerifier;

impl KeccakVerifier for PublicBitsVerifier {
    fn profile(&self) -> FlockProfile {
        FlockProfile::PublicBits
    }

    fn verify(&self, segment: &SegmentContext<'_>) -> bool {
        let live = segment.mu.len();
        if !segment.proof.is_empty()
            || segment.c_tilde.len() != live
            || segment.commitment.len() != live * W1_RECORD_BYTES
            || segment.final_vector.len() != M3
        {
            return false;
        }
        let values: Option<Vec<SmallK>> = segment
            .commitment
            .par_chunks_exact(W1_RECORD_BYTES)
            .zip(segment.mu.par_iter().zip(segment.c_tilde))
            .map(|(bytes, (mu, c_tilde))| {
                let bytes: &[u8; W1_RECORD_BYTES] = bytes.try_into().expect("768 bytes");
                let w1 = w1_decode(bytes);
                let in_range = w1.iter().flatten().all(|&v| i64::from(v) < M_W);
                (in_range && commitment_hash(mu, bytes) == *c_tilde).then_some(w1)
            })
            .collect();
        let Some(values) = values else {
            return false;
        };
        segment_projection(&segment.geometry, segment.sigma, &values, segment.matrices)
            == segment.final_vector
    }

    fn check_bits(&self, bits: &BitsContext<'_>) -> bool {
        public_bit_evaluations(&bits.geometry, bits.commitments, bits.point) == Some(bits.claims)
    }
}

/// Prove a mode-B aggregate: derive the witnesses (strict FIPS 204 verification), publish
/// the committed bits, and run the PIOP with the public bits as the Keccak side.
pub fn prove_mode_b(
    inputs: &ProverInputs<'_>,
    bridge: &impl BridgeDerivation,
) -> Result<ProverOutput, ProveError> {
    let start = Instant::now();
    let (manifest, main) = main_tables(inputs)?;
    let side = PublicBits::from_tables(&manifest, &main);
    let tables = start.elapsed();
    let mut output = prove_with_tables(
        inputs,
        &manifest,
        &main,
        bridge,
        &side,
        FlockProfile::PublicBits,
        &ProveHooks::default(),
    )?;
    output.timings.tables = tables;
    output.timings.total += tables;
    Ok(output)
}

/// Verify a mode-B aggregate (the proof bytes, which carry `c~_i` and the public bits) for
/// the records `(pk_i, M_i)` under the caller's `relation`, with the timing of every step.
pub fn verify_mode_b(
    records: &[(&[u8], &[u8])],
    aggregate: &[u8],
    bridge: &impl BridgeDerivation,
    relation: Relation,
    cache: Option<&mut KeyCache>,
) -> VerifyReport {
    verify_timed(
        records,
        aggregate,
        bridge,
        &PublicBitsVerifier,
        FlockProfile::PublicBits,
        relation,
        cache,
    )
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::manifest::Column;
    use crate::test_support::point;
    use agg_witness::bridge::{gf2_composite, M2 as ROWS2, ZETA2_USED};
    use agg_witness::gen::{self, SplitMix64};
    use agg_witness::instance::PublicInstance;
    use agg_witness::params::C_TILDE_BYTES;
    use jolt_field::One;

    struct Fixture {
        instance: PublicInstance,
        manifest: CommitmentManifest,
        main: MainTables,
    }

    fn fixture(seed: u64, own: usize, shared: usize) -> Fixture {
        let mut records = gen::records(seed, own);
        records.extend(gen::records_shared_keys(seed + 1, shared, 2));
        let instance = PublicInstance::from_records(&records).unwrap();
        let manifest = CommitmentManifest::for_batch(records.len()).unwrap();
        let signatures: Vec<&[u8]> = records.iter().map(|r| r.sig.as_slice()).collect();
        let main = MainTables::build(&manifest, &instance, &signatures).unwrap();
        Fixture {
            instance,
            manifest,
            main,
        }
    }

    /// Per-lane GF(2) reference over the live records only (sparse in level 1), with the
    /// global index conventions of `agg_witness::bridge::gf2_composite`.
    fn sparse_reference(geometry: &Geometry, w1: &[SmallK], m: &BridgeMatrices) -> Vec<u8> {
        let mut y1 = vec![[0_u8; W1_BITS]; 1 << geometry.d1()];
        for (i, values) in w1.iter().enumerate() {
            for (rho, row) in values.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    let x = main_index(j, rho, i);
                    for t in 0..W1_BITS {
                        if (v >> t) & 1 == 1 {
                            for &r in m.m1.column(x & (J1 - 1)) {
                                y1[(x >> 19) * M1 + r as usize][t] ^= 1;
                            }
                        }
                    }
                }
            }
        }
        let mut y2 = vec![[0_u8; W1_BITS]; 1 << geometry.d2()];
        for (idx, lanes) in y1.iter().enumerate() {
            for t in 0..W1_BITS {
                if lanes[t] == 1 {
                    for &r in m.m2.column(idx & (ZETA2_USED - 1)) {
                        y2[(idx >> 17) * ROWS2 + r as usize][t] ^= 1;
                    }
                }
            }
        }
        let mut y = vec![0_u8; 1 << geometry.d3()];
        for (idx, lanes) in y2.iter().enumerate() {
            for t in 0..W1_BITS {
                if lanes[t] == 1 {
                    for &r in m.m3.column(zeta3(t, idx & (ROWS2 - 1))) {
                        y[(idx >> 13) * M3 + r as usize] ^= 1;
                    }
                }
            }
        }
        y
    }

    #[test]
    fn segment_projection_matches_the_gf2_side_of_the_bridge() {
        let matrices = BridgeMatrices::random_for_tests(0x51);
        // A real batch: the projection of the committed bits is the witness crate's Y.
        let f = fixture(0x52, 9, 4);
        let geometry = f.main.geometry();
        let values: Vec<SmallK> = (0..geometry.n)
            .map(|i| record_w1(&f.manifest, &f.main, i))
            .collect();
        let expected = gf2_composite(&geometry, &f.main.bit_lanes(&f.manifest), &matrices);
        assert_eq!(
            segment_projection(&geometry, 0, &values, &matrices),
            expected
        );
        // Synthetic sparse values at two and at three segments (d_sig 14 and 15 would need
        // dense tables; the sparse reference covers the segment offsets).
        let mut rng = SplitMix64::new(0x53);
        for n in [600, (1 << 13) + 700, (1 << 14) + 3] {
            let geometry = Geometry::new(n).unwrap();
            let values: Vec<SmallK> = (0..n)
                .map(|_| {
                    let mut w = [[0_u8; N]; K];
                    for _ in 0..3 {
                        let (rho, j) = (rng.below(K as u64) as usize, rng.below(N as u64) as usize);
                        w[rho][j] = rng.below(M_W as u64) as u8;
                    }
                    w
                })
                .collect();
            let reference = sparse_reference(&geometry, &values, &matrices);
            for sigma in 0..geometry.segments() {
                let first = (sigma << SEGMENT_RECORDS_LOG2).min(n);
                let live = &values[first..first + geometry.live_in_segment(sigma)];
                assert_eq!(
                    segment_projection(&geometry, sigma, live, &matrices),
                    reference[sigma * M3..(sigma + 1) * M3],
                    "n {n} segment {sigma}"
                );
            }
        }
    }

    #[test]
    fn public_bit_evaluations_are_the_extensions_of_the_bit_lanes() {
        let f = fixture(0x61, 7, 4);
        let geometry = f.main.geometry();
        let bits = PublicBits::from_tables(&f.manifest, &f.main);
        assert_eq!(bits.segments().len(), 1);
        assert_eq!(bits.segments()[0].len(), 11 * W1_RECORD_BYTES);
        let mut rng = SplitMix64::new(0x62);
        let r = point(&mut rng, geometry.d_main() as usize);
        let expected: [E; W1_BITS] = std::array::from_fn(|t| {
            f.main.column(&f.manifest, Column::W1Bit(t as u8)).evaluate(&r)
        });
        assert_eq!(
            public_bit_evaluations(&geometry, bits.segments(), &r),
            Some(expected)
        );
        // Malformed items and points.
        let mut short = bits.segments().to_vec();
        short[0].pop();
        assert_eq!(public_bit_evaluations(&geometry, &short, &r), None);
        assert_eq!(
            public_bit_evaluations(&geometry, bits.segments(), &r[1..]),
            None
        );
        assert_eq!(public_bit_evaluations(&geometry, &[], &r), None);
        // The encodings are the records' w1Encode: c~ = H(mu || w1Encode).
        for (i, record) in f.instance.records.iter().enumerate() {
            let item = &bits.segments()[0][i * W1_RECORD_BYTES..(i + 1) * W1_RECORD_BYTES];
            assert_eq!(
                commitment_hash(&record.mu, item.try_into().unwrap()),
                record.c_tilde
            );
        }
    }

    #[test]
    fn the_verifier_side_checks_shape_range_hash_and_projection() {
        let f = fixture(0x71, 6, 3);
        let geometry = f.main.geometry();
        let matrices = BridgeMatrices::random_for_tests(0x72);
        let bits = PublicBits::from_tables(&f.manifest, &f.main);
        let mu: Vec<[u8; 64]> = f.instance.records.iter().map(|r| r.mu).collect();
        let c_tilde: Vec<[u8; C_TILDE_BYTES]> =
            f.instance.records.iter().map(|r| r.c_tilde).collect();
        let y = gf2_composite(&geometry, &f.main.bit_lanes(&f.manifest), &matrices);
        let check = |commitment: &[u8],
                     final_vector: &[u8],
                     proof: &[u8],
                     c_tilde: &[[u8; C_TILDE_BYTES]]| {
            PublicBitsVerifier.verify(&SegmentContext {
                sigma: 0,
                geometry,
                commitment,
                seed: &[0; 32],
                bridge_seed: &[0; 32],
                mu: &mu,
                c_tilde,
                final_vector,
                proof,
                matrices: &matrices,
            })
        };
        let item = &bits.segments()[0];
        assert!(check(item, &y, &[], &c_tilde));
        assert!(!check(item, &y, &[1], &c_tilde), "non-empty proof");
        assert!(!check(&item[1..], &y, &[], &c_tilde), "length");
        let mut flipped = item.clone();
        flipped[5 * W1_RECORD_BYTES + 100] ^= 1;
        assert!(!check(&flipped, &y, &[], &c_tilde), "hash");
        let mut wide = item.clone();
        wide[2 * W1_RECORD_BYTES] |= 0x3f;
        assert!(!check(&wide, &y, &[], &c_tilde), "w1' >= 44");
        let mut other_y = y.clone();
        other_y[17] ^= 1;
        assert!(!check(item, &other_y, &[], &c_tilde), "projection");
        let mut other_c = c_tilde.clone();
        other_c[3][0] ^= 1;
        assert!(!check(item, &y, &[], &other_c), "c~");
        // check_bits: the true claims pass, a changed one fails.
        let mut rng = SplitMix64::new(0x73);
        let r = point(&mut rng, geometry.d_main() as usize);
        let claims = public_bit_evaluations(&geometry, bits.segments(), &r).unwrap();
        let context = |claims| BitsContext {
            geometry,
            commitments: bits.segments(),
            point: &r,
            claims,
        };
        assert!(PublicBitsVerifier.check_bits(&context(claims)));
        let mut wrong = claims;
        wrong[4] += E::one();
        assert!(!PublicBitsVerifier.check_bits(&context(wrong)));
    }
}

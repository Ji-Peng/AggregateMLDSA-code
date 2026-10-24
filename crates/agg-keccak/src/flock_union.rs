//! Single-type boolean Flock union proofs with the caller's evaluation claims
//! (docs/keccak.md Section 9; docs/bridge.md Section 5a; deviation K11).
//!
//! Flock 83f963e ties committed words to each other and to public words only through its
//! wiring argument, whose cost is one cell-slot (one gather claim, 16 proof bytes, one
//! `2^nu`-cell column) per schema word of a type: for a one-row type that is one claim per
//! copied word (docs/keccak.md Section 8, finding 1). The tied part ([`crate::tied`]) needs a
//! different tie, `z_hat(point) = value` claims on the committed polynomial at points of the
//! caller's choosing. Flock's merged opening already carries such claims
//! ([`PackedDirectClaim`], the intake of the wiring's gather claims and of the element
//! class), but no prove entry lets a caller add its own. This module is that entry: the
//! boolean-only union pipeline of `flock_prover::prover::prove_fast_ligerito_union`
//! (statement binding, zerocheck, union-column lincheck, merged opening) composed from
//! Flock's public building blocks for a registry of ONE boolean type, plus the caller's
//! packed-direct claims; [`UnionGeometry::verify`] mirrors
//! `flock_core::verifier::verify_ligerito_union` step by step and returns the claimed
//! values for the caller's own checks. The transcript uses Flock's labels in Flock's order
//! (`flock-mixed-v1`, `flock-zerocheck-v0`, `flock-lincheck-v0`, `flock-merged-open-v1`); the
//! extra claims' values are absorbed by the opening exactly as gather claims would be, and
//! their points are derived by the caller from the outer transcript and recomputed by the
//! verifier (never prover messages).
//!
//! A claim point has `M - 7` coordinates over the padded word address space, rows low:
//! word column `c` of row `j` is address `(c << nu) + j`, so the point is `[row coordinates
//! (nu) | column coordinates (k_log - 7)]`, least significant first. A point whose
//! coordinates are exact 0/1 outside a set of free coordinates evaluates the multilinear
//! extension of the words of that aligned region alone ([`eval_packed`]).
//!
//! Proof bytes ([`UnionProof::to_bytes`]): bincode of Flock's `R1csProofMergedLigerito`
//! (zerocheck, lincheck, merged opening) framed by its length, then the claim values, 16
//! bytes each (little-endian `lo`, `hi`).

use bincode::Options as _;
use flock_core::{
    challenger::Challenger,
    lincheck::{self, build_eq_table, LincheckCircuit, QuirkyPoint, SkipPoint, UnionLincheckSlot},
    pcs::{
        commit, commit_lane_major, ligerito::embedded_initial_k_or_default, open_batch_merged,
        rectangular_prefix_columns, ring_switch::s_hat_v_from_z_vec, verify_batch_merged,
        Commitment, DirectEqInd, PackedDirectClaim, PackedDirectClaimRef, PcsParams, LOG_PACKING,
    },
    proof::{R1csProofMergedLigerito, ZClaim},
    schedule::{Registry, TableType},
    union::{SlotWitness, UnionInstance},
    verifier::FlockVerifyError,
    zerocheck,
};
use flock_field::F128;
use rayon::prelude::*;

use crate::{
    prover::{codec, KeccakCommitment, KeccakError, Profile, CAP_NODE_BYTES},
    witness::SlotBuffers,
};

/// The three packed buffers of a slot as the byte strings Flock's zerocheck reads
/// (`a`, `b`, `c = z` for `C = I`), without a copy. The only `unsafe` of this crate: `F128`
/// is `#[repr(C, align(16))] { lo: u64, hi: u64 }`, 16 bytes without padding, so a slice of
/// `n` words is exactly `16 n` initialized bytes; the view borrows the slice.
#[allow(unsafe_code)]
fn as_bytes(v: &[F128]) -> &[u8] {
    // SAFETY: see above; the pointer and the length cover exactly the slice's memory, which
    // stays borrowed (immutably) for the lifetime of the returned slice.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// `x_inner_rest || x_outer` of a quirky point: the multilinear part the opening takes.
fn x_full(point: &QuirkyPoint) -> Vec<F128> {
    let mut v = Vec::with_capacity(point.x_inner_rest.len() + point.x_outer.len());
    v.extend_from_slice(&point.x_inner_rest);
    v.extend_from_slice(&point.x_outer);
    v
}

/// A proof of one single-type union with the caller's claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionProof {
    /// Flock's zerocheck, lincheck and merged opening.
    pub flock: R1csProofMergedLigerito,
    /// The claimed values `z_hat(point_k)`, in the caller's claim order.
    pub values: Vec<F128>,
}

impl UnionProof {
    /// Serialize (module docs).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload = codec()
            .serialize(&self.flock)
            .expect("serializing a proof cannot fail");
        let mut out = Vec::with_capacity(4 + payload.len() + 16 * self.values.len());
        out.extend_from_slice(
            &u32::try_from(payload.len())
                .expect("proof below 4 GiB")
                .to_le_bytes(),
        );
        out.extend_from_slice(&payload);
        for v in &self.values {
            out.extend_from_slice(&v.lo.to_le_bytes());
            out.extend_from_slice(&v.hi.to_le_bytes());
        }
        out
    }

    /// Parse a proof that must carry exactly `claims` values.
    ///
    /// # Errors
    /// On a malformed byte string.
    pub fn from_bytes(bytes: &[u8], claims: usize) -> Result<Self, KeccakError> {
        let malformed = |m: &'static str| KeccakError::Malformed(m);
        if bytes.len() < 4 {
            return Err(malformed("union proof length"));
        }
        let len = u32::from_le_bytes(bytes[..4].try_into().expect("4 bytes")) as usize;
        let rest = bytes.get(4..).ok_or(malformed("union proof length"))?;
        if rest.len() != len + 16 * claims {
            return Err(malformed("union proof length"));
        }
        let flock: R1csProofMergedLigerito = codec()
            .deserialize(&rest[..len])
            .map_err(|_| malformed("union proof payload"))?;
        let values = rest[len..]
            .as_chunks::<16>()
            .0
            .iter()
            .map(|c| {
                F128::new(
                    u64::from_le_bytes(c[..8].try_into().expect("8 bytes")),
                    u64::from_le_bytes(c[8..].try_into().expect("8 bytes")),
                )
            })
            .collect();
        Ok(Self { flock, values })
    }
}

/// `z_hat_packed(point)`: the multilinear extension of the padded word buffer `z`
/// (`2^(point.len())` words, rows in the low `nu` address bits) at `point`. Column weights
/// that are exactly zero (the fixed 0/1 coordinates of a region point) are skipped, so a
/// region point costs its region's columns times `2^nu`.
///
/// # Panics
/// If the lengths disagree.
#[must_use]
pub fn eval_packed(z: &[F128], point: &[F128], nu: usize) -> F128 {
    assert_eq!(z.len(), 1usize << point.len(), "padded buffer length");
    assert!(nu <= point.len(), "row coordinates");
    let eq_rows = build_eq_table(&point[..nu]);
    let eq_cols = build_eq_table(&point[nu..]);
    eq_cols
        .par_iter()
        .enumerate()
        .filter(|(_, e)| **e != F128::ZERO)
        .map(|(col, &e)| {
            let base = col << nu;
            let mut acc = F128::ZERO;
            for (r, &w) in eq_rows.iter().enumerate() {
                acc += w * z[base + r];
            }
            e * acc
        })
        .reduce(|| F128::ZERO, |p, q| p + q)
}

/// The registry, counts and PCS parameters of one single-type boolean union: the geometry
/// prover and verifier derive from the type and the declared count.
pub struct UnionGeometry {
    registry: Registry,
    n: usize,
    pcs_params: PcsParams,
}

impl UnionGeometry {
    /// A registry of the one type `ty` with row capacity `2^nu` and `n` declared rows.
    ///
    /// # Panics
    /// If `n` is zero or exceeds the capacity.
    #[must_use]
    pub fn new(ty: TableType, n: usize, nu: usize, profile: Profile) -> Self {
        assert!(n >= 1 && n <= 1 << nu, "1 <= n <= 2^nu");
        let registry = Registry::new(vec![ty], nu);
        let union = UnionInstance::new(&registry, vec![n]);
        let lp = profile.ligerito();
        let m = union.dense_m();
        let batch = embedded_initial_k_or_default(m, lp);
        let pcs_params = PcsParams {
            m,
            log_inv_rate: lp.log_inv_rate(),
            log_batch_size: batch,
            profile: lp,
            num_lanes: union.commit_lanes(batch),
            merkle_hash: Default::default(),
        };
        Self {
            registry,
            n,
            pcs_params,
        }
    }

    /// The union instance (registry and count).
    #[must_use]
    pub fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, vec![self.n])
    }

    /// Declared rows.
    #[must_use]
    pub fn count(&self) -> usize {
        self.n
    }

    /// Row capacity `log2`.
    #[must_use]
    pub fn nu(&self) -> usize {
        self.registry.nu()
    }

    /// The type's block width `log2`.
    #[must_use]
    pub fn k_log(&self) -> usize {
        self.registry.types()[0].k_log
    }

    /// Address-space variables `M = nu + k_log`.
    #[must_use]
    pub fn m_total(&self) -> usize {
        self.registry.m_total()
    }

    /// Committed polynomial variables (dense stack, bits).
    #[must_use]
    pub fn dense_m(&self) -> usize {
        self.pcs_params.m
    }

    /// Coordinates of a claim point (`M - 7`, the padded word variables).
    #[must_use]
    pub fn word_vars(&self) -> usize {
        self.m_total() - 7
    }

    /// Bytes of the commitment (the Ligerito level-0 cap).
    #[must_use]
    pub fn commitment_len(&self) -> usize {
        CAP_NODE_BYTES << self.pcs_params.level_zero_cap_depth()
    }

    /// Flock's dense stack of the padded slot buffer `z` (declared rows of the used
    /// columns), or `None` when the compaction is the identity.
    fn compact(&self, union: &UnionInstance<'_>, z: &[F128]) -> Option<Vec<F128>> {
        assert_eq!(z.len(), union.packed_len(), "slot buffer length");
        (!union.compaction_is_identity()).then(|| union.compact_witness(z))
    }

    fn commit_stack(&self, q: &[F128]) -> (Commitment, flock_core::pcs::ProverData) {
        if self.pcs_params.num_lanes.is_some() {
            commit_lane_major(q, &self.pcs_params)
        } else {
            commit(q, &self.pcs_params)
        }
    }

    /// Flock's commitment to the padded slot buffer `z` (event 2, before any seed): the
    /// same map [`Self::prove`] applies, so the prover can check the caps agree (K7).
    #[must_use]
    pub fn commit(&self, z: &[F128]) -> KeccakCommitment {
        let union = self.union();
        let q = self.compact(&union, z);
        let (commitment, _) = self.commit_stack(q.as_deref().unwrap_or(z));
        KeccakCommitment::from_flock(&commitment)
    }

    /// Prove the slot's rows under `circuit` and the claims `(point, value)` with the
    /// challenger `ch` (already seeded and bound to the caller's statement). Returns the
    /// proof and the commitment Flock bound.
    ///
    /// # Panics
    /// If a buffer or a point has the wrong length.
    pub fn prove<Ch: Challenger>(
        &self,
        buffers: SlotBuffers,
        circuit: &dyn LincheckCircuit,
        claims: &[(Vec<F128>, F128)],
        ch: &mut Ch,
    ) -> (UnionProof, KeccakCommitment) {
        let union = self.union();
        assert_eq!(
            self.pcs_params.m,
            union.dense_m(),
            "PCS parameters of this union"
        );
        let lig_config = self
            .pcs_params
            .ligerito_prover_config()
            .expect("Ligerito prover config for dense_m >= 22");
        let SlotBuffers { z, a, b, stripe } = buffers;
        // Single slot spanning the union: a passthrough, no copy.
        let (z, a, b) = union.assemble_witness(vec![SlotWitness {
            z_packed: z,
            a_packed: a,
            b_packed: b,
        }]);
        let q_owned = self.compact(&union, &z);
        let (commitment, prover_data) = self.commit_stack(q_owned.as_deref().unwrap_or(&z));
        union.bind_statement(ch, &commitment);

        // Zerocheck over the padded address space (`M_bool = M`: one boolean type), run-list
        // gated on the declared rows; `c` aliases `z` (`C = I`).
        let m = union.m_bool();
        let bool_padding = union.boolean_padding_spec();
        let (zc_proof, zc_claim, s_hat_v_c) =
            zerocheck::prove_packed_padded_capture_s_hat_v_c_with_grinding(
                as_bytes(&a),
                as_bytes(&b),
                as_bytes(&z),
                m,
                &bool_padding,
                self.pcs_params.zerocheck_grinding(),
                ch,
            );
        drop(a);
        drop(b);
        let x_ab = union
            .x_ab_from_multilinear_values(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);

        // Union-column lincheck over the one slot (byte-identical to the single-table one).
        let slots = [UnionLincheckSlot {
            z_lincheck: &stripe,
            circuit,
        }];
        let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_union_capture_z_vec_with_grinding(
            &union,
            &slots,
            &x_ab,
            self.pcs_params.lincheck_grinding(),
            ch,
        );
        let ab = ZClaim {
            point: union.ab_claim_point(
                lc_claim.r_inner_skip,
                &lc_claim.r_inner_rest,
                &x_ab.x_outer,
            ),
            value: lc_claim.w,
        };
        let c = ZClaim {
            point: union.c_claim_point(SkipPoint::Phi8(zc_claim.z), &zc_claim.r_rest),
            value: zc_claim.c_eval,
        };
        let nu = union.n_log();
        let s_hat_v_ab = (m - nu >= LOG_PACKING)
            .then(|| s_hat_v_from_z_vec(&z_vec_pre, &lc_claim.r_inner_rest[1..]));
        let heights = union.jagged_heights();
        // Block-first transport inputs (a full-height stack): the column bit-bank at the C
        // claim's row point, as Flock's union prover computes it.
        let bank_c = rectangular_prefix_columns(&heights, nu).map(|_| {
            assert!(zc_claim.r_rest.len() > nu, "C claim point too short");
            lincheck::union_bitbank_fold(&union, &slots, &zc_claim.r_rest[1..1 + nu])
        });
        drop(stripe);
        let pre: [Option<&[F128]>; 2] = [s_hat_v_ab.as_deref(), Some(s_hat_v_c.as_slice())];
        let banks: [Option<&[F128]>; 2] = match &bank_c {
            Some(bc) => [Some(z_vec_pre.as_slice()), Some(bc.as_slice())],
            None => [None, None],
        };
        let word_vars = self.word_vars();
        let packed_direct: Vec<PackedDirectClaim> = claims
            .iter()
            .map(|(point, value)| {
                assert_eq!(point.len(), word_vars, "claim point length");
                PackedDirectClaim {
                    point: point.clone(),
                    value: *value,
                    eq_ind: DirectEqInd::EqPoint(point.clone()),
                }
            })
            .collect();
        let values: Vec<F128> = claims.iter().map(|(_, v)| *v).collect();
        let x_fulls = [x_full(&ab.point), x_full(&c.point)];
        let x_refs: [&[F128]; 2] = [&x_fulls[0], &x_fulls[1]];
        let padding = union.padding_spec();
        let grinding = self.pcs_params.opening_grinding();
        let pcs_open = match q_owned {
            Some(q) => open_batch_merged(
                q,
                Some(&z),
                &prover_data,
                &commitment,
                &x_refs,
                &pre,
                &banks,
                &packed_direct,
                &padding,
                &heights,
                nu,
                &lig_config,
                grinding,
                ch,
            ),
            None => open_batch_merged(
                z,
                None,
                &prover_data,
                &commitment,
                &x_refs,
                &pre,
                &banks,
                &packed_direct,
                &padding,
                &heights,
                nu,
                &lig_config,
                grinding,
                ch,
            ),
        };
        (
            UnionProof {
                flock: R1csProofMergedLigerito {
                    zerocheck: zc_proof,
                    lincheck: lc_proof,
                    pcs_open,
                },
                values,
            },
            KeccakCommitment::from_flock(&commitment),
        )
    }

    /// Verify a proof against the commitment of event 2, the circuit and the claim
    /// `points` (recomputed by the caller), with a challenger in the prover's state.
    /// Returns the claimed values, which the caller checks against its own expectations.
    ///
    /// # Errors
    /// On a malformed commitment or claim list, or a Flock rejection.
    pub fn verify<Ch: Challenger>(
        &self,
        circuit: &dyn LincheckCircuit,
        commitment: &KeccakCommitment,
        proof: &UnionProof,
        points: &[Vec<F128>],
        ch: &mut Ch,
    ) -> Result<Vec<F128>, KeccakError> {
        let cap = commitment.as_bytes();
        if cap.len() != self.commitment_len() {
            return Err(KeccakError::Malformed("commitment length"));
        }
        if proof.values.len() != points.len() {
            return Err(KeccakError::Malformed("claim count"));
        }
        let word_vars = self.word_vars();
        if points.iter().any(|p| p.len() != word_vars) {
            return Err(KeccakError::Statement("claim point length"));
        }
        let commitment = Commitment {
            cap: cap.as_chunks::<CAP_NODE_BYTES>().0.to_vec(),
            params: self.pcs_params.clone(),
        };
        let union = self.union();
        union.bind_statement(ch, &commitment);
        let zc = zerocheck::verify_with_grinding(
            union.m_bool(),
            &proof.flock.zerocheck,
            self.pcs_params.zerocheck_grinding(),
            ch,
        )
        .map_err(|e| KeccakError::Flock(FlockVerifyError::Zerocheck(e)))?;
        let x_ab = union.x_ab_from_multilinear_values(SkipPoint::Phi8(zc.z), &zc.mlv_challenges);
        let circuits: [&dyn LincheckCircuit; 1] = [circuit];
        // Replays the rounds and discharges the matrix work (the circuit's fold at the
        // verifier's eq table) on the caller's thread pool.
        let lc = lincheck::verify_union_with_grinding(
            &union,
            &circuits,
            &x_ab,
            zc.a_eval,
            zc.b_eval,
            &proof.flock.lincheck,
            self.pcs_params.lincheck_grinding(),
            ch,
        )
        .map_err(|e| KeccakError::Flock(FlockVerifyError::Lincheck(e)))?;
        let ab = ZClaim {
            point: union.ab_claim_point(lc.r_inner_skip, &lc.r_inner_rest, &x_ab.x_outer),
            value: lc.w,
        };
        let c = ZClaim {
            point: union.c_claim_point(SkipPoint::Phi8(zc.z), &zc.r_rest),
            value: zc.c_eval,
        };
        let values = [ab.value, c.value];
        let z_skips = [ab.point.z_skip, c.point.z_skip];
        let x_fulls = [x_full(&ab.point), x_full(&c.point)];
        let x_refs: [&[F128]; 2] = [&x_fulls[0], &x_fulls[1]];
        let pd: Vec<PackedDirectClaimRef<'_>> = points
            .iter()
            .zip(&proof.values)
            .map(|(point, &value)| PackedDirectClaimRef { point, value })
            .collect();
        let lig_v = self
            .pcs_params
            .ligerito_verifier_config()
            .expect("Ligerito verifier config for dense_m >= 22");
        verify_batch_merged(
            &commitment,
            &values,
            &z_skips,
            &x_refs,
            &pd,
            &union.jagged_heights(),
            union.n_log(),
            &proof.flock.pcs_open,
            &lig_v,
            self.pcs_params.opening_grinding(),
            ch,
        )
        .map_err(|e| KeccakError::Flock(FlockVerifyError::PcsOpen(e)))?;
        Ok(proof.values.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::test_rng::Rng;

    /// `build_eq_table` indexes least significant coordinate first, and [`eval_packed`] is
    /// the plain inner product with that table.
    #[test]
    fn eval_packed_is_the_eq_inner_product() {
        let mut rng = Rng::new(11);
        let rand = |rng: &mut Rng| F128::new(rng.next_u64(), rng.next_u64());
        let (nu, k) = (3usize, 4usize);
        let point: Vec<F128> = (0..nu + k).map(|_| rand(&mut rng)).collect();
        let z: Vec<F128> = (0..1 << (nu + k)).map(|_| rand(&mut rng)).collect();
        let mut want = F128::ZERO;
        for (addr, &w) in z.iter().enumerate() {
            let mut e = F128::ONE;
            for (bit, &p) in point.iter().enumerate() {
                e *= if (addr >> bit) & 1 == 1 {
                    p
                } else {
                    F128::ONE + p
                };
            }
            want += e * w;
        }
        assert_eq!(eval_packed(&z, &point, nu), want);
        // A region point: the second half of the columns, all rows.
        let mut region = point.clone();
        region[nu + k - 1] = F128::ONE;
        let mut want = F128::ZERO;
        for (addr, &w) in z.iter().enumerate() {
            if (addr >> (nu + k - 1)) & 1 == 0 {
                continue;
            }
            let mut e = F128::ONE;
            for (bit, &p) in region.iter().enumerate().take(nu + k - 1) {
                e *= if (addr >> bit) & 1 == 1 {
                    p
                } else {
                    F128::ONE + p
                };
            }
            want += e * w;
        }
        assert_eq!(eval_packed(&z, &region, nu), want);
    }

    #[test]
    fn union_proof_bytes_round_trip_and_reject_bad_lengths() {
        let bytes = [7u8, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7];
        assert!(matches!(
            UnionProof::from_bytes(&bytes, 0),
            Err(KeccakError::Malformed("union proof payload"))
        ));
        assert!(matches!(
            UnionProof::from_bytes(&bytes, 1),
            Err(KeccakError::Malformed("union proof length"))
        ));
        assert!(matches!(
            UnionProof::from_bytes(&bytes[..2], 0),
            Err(KeccakError::Malformed("union proof length"))
        ));
    }
}

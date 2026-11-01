//! Boolean Flock union proofs with the caller's evaluation claims (docs/keccak.md,
//! docs/bridge.md).
//!
//! Flock ties committed words to each other and to public words only through its
//! wiring argument, whose cost is one cell-slot (one gather claim, 16 proof bytes, one
//! `2^nu`-cell column) per schema word of a type: for a one-row type that is one claim per
//! copied word (docs/keccak.md). The tied part ([`crate::tied`]) needs a
//! different tie, `z_hat(point) = value` claims on the committed polynomial at points of the
//! caller's choosing. Flock's merged opening already carries such claims
//! ([`PackedDirectClaim`], the intake of the wiring's gather claims and of the element
//! class), but no prove entry lets a caller add its own. This module is that entry: the
//! boolean-only union pipeline of `flock_prover::prover::prove_fast_ligerito_union`
//! (statement binding, zerocheck, union-column lincheck, merged opening) composed from
//! Flock's public building blocks for a boolean registry (the sponge instance of
//! [`crate::tied`], one type), plus the caller's packed-direct claims; [`UnionGeometry::verify`] mirrors
//! `flock_core::verifier::verify_ligerito_union` step by step and returns the claimed
//! values for the caller's own checks. The transcript uses Flock's labels in Flock's order
//! (`flock-mixed-v1`, `flock-zerocheck-v0`, `flock-lincheck-v0`, `flock-merged-open-v1`); the
//! extra claims' values are absorbed by the opening exactly as gather claims would be, and
//! their points are derived by the caller from the outer transcript and recomputed by the
//! verifier (never prover messages).
//!
//! A claim point has `M - 7` coordinates over the padded word address space, rows low:
//! word column `c` of row `j` of a slot is address `(prefix << (nu + k_log - 7)) | (c <<
//! nu) | j`, so the point is `[row coordinates (nu) | column coordinates (k_log - 7) |
//! slot prefix (M - nu - k_log)]`, least significant first
//! ([`UnionGeometry::region_point`]). A point whose coordinates are exact 0/1 outside a set
//! of free coordinates evaluates the multilinear extension of the words of that aligned
//! region alone ([`eval_packed`]).
//!
//! Commit, then prove: [`UnionGeometry::commit_data`] computes the
//! Ligerito commitment and keeps Flock's prover data (codeword and Merkle tree), which
//! [`UnionGeometry::prove`] then opens without committing a second time.
//!
//! Proof bytes ([`UnionProof::to_bytes`]): bincode of Flock's `R1csProofMergedLigerito`
//! (zerocheck, lincheck, merged opening) framed by its length, then the claim values, 16
//! bytes each (little-endian `lo`, `hi`). The Ligerito proof's level-0 cap is not sent: it
//! is the commitment the verifier already holds (committed at the commitment phase), and Flock's verifier rejects any
//! other cap, so [`UnionGeometry::verify`] puts the commitment's cap in its place.

use bincode::Options as _;
use flock_core::{
    challenger::Challenger,
    lincheck::{self, build_eq_table, LincheckCircuit, QuirkyPoint, SkipPoint},
    pcs::{
        commit, commit_lane_major, ligerito::embedded_initial_k_or_default, open_batch_merged,
        rectangular_prefix_columns,
        ring_switch::{
            build_claim_weights_from_skip, claim_check, fold_1b_rows_naive, s_hat_v_from_z_vec,
        },
        verify_batch_merged,
        Commitment, DirectEqInd, PackedDirectClaim, PackedDirectClaimRef, PcsParams, ProverData,
        LOG_PACKING,
    },
    proof::{R1csProofMergedLigerito, ZClaim},
    schedule::{Registry, Slot, TableType},
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

/// A ring-switched claim of the caller on the committed BIT polynomial
/// (`docs/parity-chain.md`): `z_bit(point) = value` where the point's in-word bits
/// 0..5 are the multilinear skip coordinates `skip` (Flock's `SkipPoint::Eq`) and `x_full`
/// is in-word bit 6 followed by the `M - 7` word-address coordinates (rows low, as a
/// packed-direct point). Unlike a packed-direct claim it goes through Flock's ring switch
/// (`s_hat_v`, 128 values) and the twisted multipoint assist (128 dual values).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitClaim {
    /// In-word bits 0..5.
    pub skip: [F128; 6],
    /// In-word bit 6, then the word address (`M - 6` coordinates in all).
    pub x_full: Vec<F128>,
}

impl BitClaim {
    /// Flock's skip point of the claim.
    #[must_use]
    pub fn skip_point(&self) -> SkipPoint {
        SkipPoint::Eq(self.skip)
    }
}

/// The partial column bit-bank of a bit claim on the padded buffer `z` with `2^nu` rows per
/// word column: `bank[128 c + b] = sum_j eq(x_row, j) bit_b(z[(c << nu) | j])` for every word
/// column `c` whose column weight `eq(x_col, c)` is nonzero, zero elsewhere. Enough for
/// `s_hat_v` ([`bit_s_hat_v`], where the other columns are multiplied by their zero weight),
/// NOT for the block-first transport of `pcs::open_batch_merged`, whose block rounds fold the
/// bank's multilinear extension and need every column ([`UnionGeometry::prove`] uses Flock's
/// `union_bitbank_fold` there).
///
/// # Panics
/// If the claim's length does not fit `z` and `nu`.
#[must_use]
pub fn bit_bank(z: &[F128], claim: &BitClaim, nu: usize) -> Vec<F128> {
    let vars = claim.x_full.len() - 1;
    assert_eq!(z.len(), 1usize << vars, "padded buffer length");
    assert!(nu <= vars, "row coordinates");
    let eq_row = build_eq_table(&claim.x_full[1..=nu]);
    let eq_col = build_eq_table(&claim.x_full[1 + nu..]);
    let rows = 1usize << nu;
    let mut bank = vec![F128::ZERO; eq_col.len() << LOG_PACKING];
    for (c, chunk) in bank
        .as_chunks_mut::<{ 1 << LOG_PACKING }>()
        .0
        .iter_mut()
        .enumerate()
    {
        if eq_col[c] != F128::ZERO {
            chunk.copy_from_slice(&fold_1b_rows_naive(&z[c * rows..(c + 1) * rows], &eq_row));
        }
    }
    bank
}

/// `s_hat_v` of a bit claim from its bank: `sum_c eq(x_col, c) bank[c]`.
#[must_use]
pub fn bit_s_hat_v(bank: &[F128], claim: &BitClaim, nu: usize) -> Vec<F128> {
    let eq_col = build_eq_table(&claim.x_full[1 + nu..]);
    let n = 1usize << LOG_PACKING;
    assert_eq!(bank.len(), eq_col.len() * n, "one bank row per word column");
    let mut out = vec![F128::ZERO; n];
    for (c, &e) in eq_col.iter().enumerate() {
        if e != F128::ZERO {
            for (o, &b) in out.iter_mut().zip(&bank[c * n..(c + 1) * n]) {
                *o += e * b;
            }
        }
    }
    out
}

/// `s_hat_v` of a bit claim on the padded buffer `z` (rows `2^nu`) in one pass over the
/// smallest aligned prefix of word columns that holds every column of nonzero weight (the
/// chain's claim has zero column coordinates above its 64 columns): the four-Russians fold of
/// Flock's ring switch over that region with the tensor `eq(x_col, c) eq(x_row, j)`.
#[must_use]
pub fn bit_region_s_hat_v(z: &[F128], claim: &BitClaim, nu: usize) -> Vec<F128> {
    let cols = &claim.x_full[1 + nu..];
    let live = cols
        .iter()
        .rposition(|&x| x != F128::ZERO)
        .map_or(0, |k| k + 1);
    let eq_col = build_eq_table(&cols[..live]);
    let eq_row = build_eq_table(&claim.x_full[1..=nu]);
    let rows = 1usize << nu;
    let tensor: Vec<F128> = (0..eq_col.len() * rows)
        .into_par_iter()
        .with_min_len(1 << 12)
        .map(|i| eq_col[i >> nu] * eq_row[i & (rows - 1)])
        .collect();
    fold_1b_rows_naive(&z[..tensor.len()], &tensor)
}

/// The value `z_bit(point)` of a bit claim on the padded buffer `z` (rows `2^nu`).
#[must_use]
pub fn bit_value(z: &[F128], claim: &BitClaim, nu: usize) -> F128 {
    let s_hat_v = bit_region_s_hat_v(z, claim, nu);
    let weights = build_claim_weights_from_skip(&claim.skip_point().weights(6), claim.x_full[0]);
    claim_check(&weights, &s_hat_v)
}

/// A proof of one union with the caller's claims.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnionProof {
    /// Flock's zerocheck, lincheck and merged opening.
    pub flock: R1csProofMergedLigerito,
    /// The claimed values `z_hat(point_k)`, in the caller's claim order.
    pub values: Vec<F128>,
}

impl UnionProof {
    /// Serialize (module docs): the level-0 cap of the Ligerito proof is left out.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload = codec()
            .serialize(&Self::without_cap(&self.flock))
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

    /// The proof with the Ligerito proof's level-0 cap removed (the serialized form).
    fn without_cap(
        flock: &R1csProofMergedLigerito,
    ) -> std::borrow::Cow<'_, R1csProofMergedLigerito> {
        if flock.pcs_open.inner.ligerito.initial_cap.is_empty() {
            return std::borrow::Cow::Borrowed(flock);
        }
        let mut out = flock.clone();
        out.pcs_open.inner.ligerito.initial_cap = Vec::new();
        std::borrow::Cow::Owned(out)
    }

    /// Serialized bytes of the zerocheck, the lincheck, the merged opening (without the
    /// level-0 cap) and the claim values (the framing adds 4 bytes).
    #[must_use]
    pub fn size_breakdown(&self) -> [usize; 4] {
        let size = "serializable proof component";
        let flock = Self::without_cap(&self.flock);
        [
            codec().serialized_size(&flock.zerocheck).expect(size) as usize,
            codec().serialized_size(&flock.lincheck).expect(size) as usize,
            codec().serialized_size(&flock.pcs_open).expect(size) as usize,
            16 * self.values.len(),
        ]
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
        if !flock.pcs_open.inner.ligerito.initial_cap.is_empty() {
            return Err(malformed("union proof carries its level-0 cap"));
        }
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

/// A commitment computed ahead of the proof together with Flock's prover data (the
/// Reed-Solomon codeword and Merkle tree of the dense stack), so that
/// [`UnionGeometry::prove`] opens it without committing a second time (the
/// Ligerito commit is deterministic, so the data is exactly what the prover would
/// recompute). The caller keeps it from the commitment phase to the final-vector step.
pub struct CommitData {
    commitment: Commitment,
    prover_data: ProverData,
}

impl CommitData {
    /// The commitment-phase bytes (the Ligerito level-0 cap).
    #[must_use]
    pub fn commitment(&self) -> KeccakCommitment {
        KeccakCommitment::from_flock(&self.commitment)
    }

    /// Heap bytes held (codeword and Merkle tree).
    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        std::mem::size_of_val(self.prover_data.codeword.as_slice())
            + std::mem::size_of_val(self.prover_data.merkle_tree.as_slice())
    }
}

/// The registry, counts and PCS parameters of a boolean union ([`Self::new`] builds one of a
/// single type): the geometry prover and verifier derive from the types and the declared
/// counts.
///
/// Types and their per-type inputs (slot buffers, circuits, counts) are given in the
/// CALLER's order; Flock's registry sorts its slots by `k_log` descending (stable), and
/// this struct maps between the two ([`Self::slot`]), so callers never assume a slot order.
pub struct UnionGeometry {
    registry: Registry,
    /// Registry slot of each caller type.
    slot_of: Vec<usize>,
    /// Declared rows per registry slot (slot order).
    counts: Vec<usize>,
    pcs_params: PcsParams,
}

impl UnionGeometry {
    /// A registry of the one boolean type `ty` with row capacity `2^nu` and `n` declared
    /// rows.
    ///
    /// # Panics
    /// If `n` is zero or exceeds the capacity, or `ty` is not boolean.
    #[must_use]
    pub fn new(ty: TableType, n: usize, nu: usize, profile: Profile) -> Self {
        assert!(n >= 1 && n <= 1 << nu, "1 <= n <= 2^nu");
        assert!(!ty.is_element(), "boolean types only");
        let registry = Registry::new(vec![ty], nu);
        let (slot_of, counts) = (vec![0], vec![n]);
        let union = UnionInstance::new(&registry, counts.clone());
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
            slot_of,
            counts,
            pcs_params,
        }
    }

    /// The union instance (registry and counts).
    #[must_use]
    pub fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, self.counts.clone())
    }

    /// Number of types.
    #[must_use]
    pub fn types(&self) -> usize {
        self.slot_of.len()
    }

    /// The registry slot of caller type `i` (offset, prefix and prefix bits).
    #[must_use]
    pub fn slot(&self, i: usize) -> Slot {
        self.registry.slots()[self.slot_of[i]]
    }

    /// Declared rows of caller type `i`.
    #[must_use]
    pub fn count(&self, i: usize) -> usize {
        self.counts[self.slot_of[i]]
    }

    /// Row capacity `log2`.
    #[must_use]
    pub fn nu(&self) -> usize {
        self.registry.nu()
    }

    /// Block width `log2` of caller type `i`.
    #[must_use]
    pub fn k_log(&self, i: usize) -> usize {
        self.registry.types()[self.slot_of[i]].k_log
    }

    /// Address-space variables `M` (`nu + k_log` for one type).
    #[must_use]
    pub fn m_total(&self) -> usize {
        self.registry.m_total()
    }

    /// Committed polynomial variables (dense stack, bits).
    #[must_use]
    pub fn dense_m(&self) -> usize {
        self.pcs_params.m
    }

    /// Committed 128-bit words (the dense stack before power-of-two padding).
    #[must_use]
    pub fn dense_words(&self) -> usize {
        self.union().dense_words()
    }

    /// Coordinates of a claim point (`M - 7`, the padded word variables).
    #[must_use]
    pub fn word_vars(&self) -> usize {
        self.m_total() - 7
    }

    /// The claim point on caller type `i`'s region: `rows` on the `nu` row coordinates and
    /// `cols` on the type's `k_log - 7` word-column coordinates (both zero-extended), then
    /// the slot prefix as 0/1 constants. Word column `c` of row `j` of the slot is the
    /// padded word `(prefix << (nu + k_log - 7)) | (c << nu) | j`.
    ///
    /// # Panics
    /// If `rows` or `cols` is too long.
    #[must_use]
    pub fn region_point(&self, i: usize, rows: &[F128], cols: &[F128]) -> Vec<F128> {
        let (nu, k) = (self.nu(), self.k_log(i) - 7);
        assert!(
            rows.len() <= nu && cols.len() <= k,
            "region point coordinates"
        );
        let slot = self.slot(i);
        let mut p = Vec::with_capacity(self.word_vars());
        p.extend_from_slice(rows);
        p.resize(nu, F128::ZERO);
        p.extend_from_slice(cols);
        p.resize(nu + k, F128::ZERO);
        for b in 0..slot.prefix_bits {
            p.push(if (slot.prefix >> b) & 1 == 1 {
                F128::ONE
            } else {
                F128::ZERO
            });
        }
        debug_assert_eq!(p.len(), self.word_vars());
        p
    }

    /// Bytes of the commitment (the Ligerito level-0 cap).
    #[must_use]
    pub fn commitment_len(&self) -> usize {
        CAP_NODE_BYTES << self.pcs_params.level_zero_cap_depth()
    }

    /// Flock's dense stack of the padded union buffer `z` (declared rows of the used
    /// columns), or `None` when the compaction is the identity.
    fn compact(&self, union: &UnionInstance<'_>, z: &[F128]) -> Option<Vec<F128>> {
        assert_eq!(z.len(), union.packed_len(), "union buffer length");
        (!union.compaction_is_identity()).then(|| union.compact_witness(z))
    }

    fn commit_stack(&self, q: &[F128]) -> (Commitment, ProverData) {
        if self.pcs_params.num_lanes.is_some() {
            commit_lane_major(q, &self.pcs_params)
        } else {
            commit(q, &self.pcs_params)
        }
    }

    /// Whether the registry is one slot spanning the address space (Flock's zero-copy
    /// passthrough: the slot buffer IS the union buffer).
    fn single(&self) -> bool {
        self.registry.num_types() == 1 && self.registry.slots()[0].m_slot == self.m_total()
    }

    /// The padded union `z` of per-type slot buffers (caller order): each slot's words at
    /// its aligned offset, zero elsewhere (single spanning slot: `None`, use it as is).
    fn scatter_z(&self, slot_zs: &[&[F128]]) -> Option<Vec<F128>> {
        assert_eq!(slot_zs.len(), self.types(), "one slot buffer per type");
        for (i, z) in slot_zs.iter().enumerate() {
            let words = 1usize << (self.slot(i).m_slot - 7);
            assert_eq!(z.len(), words, "slot buffer length of type {i}");
        }
        if self.single() {
            return None;
        }
        let mut out = vec![F128::ZERO; self.union().packed_len()];
        for (i, z) in slot_zs.iter().enumerate() {
            let start = self.slot(i).offset >> 7;
            out[start..start + z.len()]
                .par_chunks_mut(1 << 12)
                .zip(z.par_chunks(1 << 12))
                .for_each(|(o, s)| o.copy_from_slice(s));
        }
        Some(out)
    }

    /// Flock's commitment to the per-type slot buffers `z` (caller order) with its prover
    /// data (the commitment phase, before any seed): the map [`Self::prove`] applies, so a proof may
    /// open it without committing again.
    #[must_use]
    pub fn commit_data(&self, slot_zs: &[&[F128]]) -> CommitData {
        let scattered = self.scatter_z(slot_zs);
        let z = scattered.as_deref().unwrap_or(slot_zs[0]);
        let union = self.union();
        let q = self.compact(&union, z);
        let (commitment, prover_data) = self.commit_stack(q.as_deref().unwrap_or(z));
        CommitData {
            commitment,
            prover_data,
        }
    }

    /// The commitment alone (see [`Self::commit_data`]).
    #[must_use]
    pub fn commit(&self, slot_zs: &[&[F128]]) -> KeccakCommitment {
        self.commit_data(slot_zs).commitment()
    }

    /// Prove the slots' rows under `circuits` (caller order, one per type) with the
    /// packed-direct claims at `points`, whose values the prover computes from the
    /// assembled union buffer, with the challenger `ch` (already seeded and bound to the
    /// caller's statement). With `pre` the Flock commitment is opened as is;
    /// otherwise Flock commits here. Returns the proof and the commitment Flock bound.
    ///
    /// # Panics
    /// If a buffer, a circuit or a point has the wrong shape, or `pre` was made for other
    /// PCS parameters.
    pub fn prove<Ch: Challenger>(
        &self,
        buffers: Vec<SlotBuffers>,
        circuits: &[&dyn LincheckCircuit],
        points: &[Vec<F128>],
        bit_claims: &[BitClaim],
        ch: &mut Ch,
        pre: Option<CommitData>,
    ) -> (UnionProof, KeccakCommitment) {
        let mut stamp_clock = std::time::Instant::now();
        let union = self.union();
        assert_eq!(
            self.pcs_params.m,
            union.dense_m(),
            "PCS parameters of this union"
        );
        assert_eq!(buffers.len(), self.types(), "one buffer set per type");
        assert_eq!(circuits.len(), self.types(), "one circuit per type");
        let lig_config = self
            .pcs_params
            .ligerito_prover_config()
            .expect("Ligerito prover config for dense_m >= 22");
        // Caller order to slot order.
        let mut by_slot: Vec<Option<(SlotBuffers, &dyn LincheckCircuit)>> =
            (0..self.types()).map(|_| None).collect();
        for (i, (buf, &circuit)) in buffers.into_iter().zip(circuits).enumerate() {
            by_slot[self.slot_of[i]] = Some((buf, circuit));
        }
        let mut witnesses = Vec::with_capacity(self.types());
        let mut slot_circuits = Vec::with_capacity(self.types());
        for entry in by_slot {
            let (buf, circuit) = entry.expect("every slot");
            let (z, a, b) = buf.into_zab();
            witnesses.push(SlotWitness {
                z_packed: z,
                a_packed: a,
                b_packed: b,
            });
            slot_circuits.push(circuit);
        }
        // Single slot spanning the union: a passthrough, no copy.
        let (z, a, b) = union.assemble_witness(witnesses);
        let nu = union.n_log();
        let word_vars = self.word_vars();
        let packed_direct: Vec<PackedDirectClaim> = points
            .iter()
            .map(|point| {
                assert_eq!(point.len(), word_vars, "claim point length");
                PackedDirectClaim {
                    point: point.clone(),
                    value: eval_packed(&z, point, nu),
                    eq_ind: DirectEqInd::EqPoint(point.clone()),
                }
            })
            .collect();
        let values: Vec<F128> = packed_direct.iter().map(|c| c.value).collect();
        for c in bit_claims {
            assert_eq!(c.x_full.len(), word_vars + 1, "bit claim point length");
        }
        let q_owned = self.compact(&union, &z);
        let (commitment, prover_data) = match pre {
            Some(CommitData {
                commitment,
                prover_data,
            }) => {
                assert!(
                    commitment.params == self.pcs_params,
                    "precomputed commitment of other PCS parameters"
                );
                (commitment, prover_data)
            }
            None => self.commit_stack(q_owned.as_deref().unwrap_or(&z)),
        };
        union.bind_statement(ch, &commitment);
        flock_stamp("prepare", &mut stamp_clock);

        // Zerocheck over the padded address space (`M_bool = M`: boolean types only), run-list
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
        flock_stamp("zerocheck", &mut stamp_clock);
        let x_ab = union
            .x_ab_from_multilinear_values(SkipPoint::Phi8(zc_claim.z), &zc_claim.mlv_challenges);

        let heights = union.jagged_heights();
        // The column bit-banks of the union buffer at every row point known after the
        // zerocheck, in one pass over `z`: the lincheck's `x_outer` (its union fold
        // `z_vec`) and, with the block-first transport (a full-height stack), the C claim's
        // row point and each bit claim's (the block rounds fold the bank's multilinear
        // extension, so every column enters). Without the block-first transport only
        // `s_hat_v` of a bit claim is needed, and the columns of nonzero weight suffice.
        let block_first = rectangular_prefix_columns(&heights, nu).is_some();
        if block_first {
            assert!(zc_claim.r_rest.len() > nu, "C claim point too short");
        }
        let t_banks = std::time::Instant::now();
        // The packed-direct groups' column bit-banks at their row points (first
        // occurrence order, matching `scalar_claim_groups`): the opening's block
        // statistics read the banks instead of passing over `q`.
        let mut pd_rows: Vec<&[F128]> = Vec::new();
        if block_first {
            for c in &packed_direct {
                let r = &c.point[..nu];
                if !pd_rows.contains(&r) {
                    pd_rows.push(r);
                }
            }
        }
        let mut rows: Vec<&[F128]> = vec![x_ab.x_outer.as_slice()];
        if block_first {
            rows.push(&zc_claim.r_rest[1..1 + nu]);
            rows.extend(bit_claims.iter().map(|c| &c.x_full[1..=nu]));
            rows.extend(pd_rows.iter().copied());
        }
        let mut banks_all = lincheck::union_bitbank_fold_multi(&union, &z, &rows).into_iter();
        let z_vec = banks_all.next().expect("lincheck fold");
        let bank_c = block_first.then(|| banks_all.next().expect("C claim bank"));
        let bit_banks: Vec<Vec<F128>> = bit_claims
            .iter()
            .map(|_| banks_all.next().unwrap_or_default())
            .collect();
        let pd_banks: Vec<Vec<F128>> = banks_all.collect();
        if std::env::var_os("LINCHECK_TRACE").is_some() {
            eprintln!(
                "[lc] {:<26} {:>7.2} ms  ({} row points, one pass)",
                "union partial folds",
                t_banks.elapsed().as_secs_f64() * 1e3,
                rows.len()
            );
        }
        drop(rows);
        // Union-column lincheck over the slots in registry order (one slot: byte-identical to
        // the single-table lincheck).
        let (lc_proof, lc_claim, z_vec_pre) = lincheck::prove_union_with_z_vec_with_grinding(
            &union,
            &slot_circuits,
            &x_ab,
            z_vec,
            self.pcs_params.lincheck_grinding(),
            ch,
        );
        flock_stamp("lincheck", &mut stamp_clock);
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
        let s_hat_v_ab = (m - nu >= LOG_PACKING)
            .then(|| s_hat_v_from_z_vec(&z_vec_pre, &lc_claim.r_inner_rest[1..]));
        let t_shat = std::time::Instant::now();
        let bit_s_hat: Vec<Vec<F128>> = bit_claims
            .iter()
            .zip(&bit_banks)
            .map(|(c, bank)| {
                if bank.is_empty() {
                    bit_region_s_hat_v(&z, c, nu)
                } else {
                    bit_s_hat_v(bank, c, nu)
                }
            })
            .collect();
        if !bit_claims.is_empty() && std::env::var_os("PCS_TRACE").is_some() {
            eprintln!(
                "  [flock_union] bit claims' s_hat_v ({}, banks in the one-pass fold): {:6.2} ms",
                if bank_c.is_some() { "full, block-first" } else { "region" },
                t_shat.elapsed().as_secs_f64() * 1e3
            );
        }
        flock_stamp("opening-inputs", &mut stamp_clock);
        let mut pre: Vec<Option<&[F128]>> = vec![s_hat_v_ab.as_deref(), Some(s_hat_v_c.as_slice())];
        pre.extend(bit_s_hat.iter().map(|v| Some(v.as_slice())));
        let banks: Vec<Option<&[F128]>> = match &bank_c {
            Some(bc) => [Some(z_vec_pre.as_slice()), Some(bc.as_slice())]
                .into_iter()
                .chain(bit_banks.iter().map(|b| Some(b.as_slice())))
                .chain(pd_banks.iter().map(|b| Some(b.as_slice())))
                .collect(),
            None => vec![None; 2 + bit_claims.len() + pd_banks.len()],
        };
        let mut x_fulls = vec![x_full(&ab.point), x_full(&c.point)];
        x_fulls.extend(bit_claims.iter().map(|c| c.x_full.clone()));
        let x_refs: Vec<&[F128]> = x_fulls.iter().map(Vec::as_slice).collect();
        let padding = union.padding_spec();
        let grinding = self.pcs_params.opening_grinding();
        // The opening (ring switch and Ligerito) on the calling Flock worker.
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
        flock_stamp("opening", &mut stamp_clock);
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

    /// Verify a proof against the Flock commitment, the circuits (caller order) and
    /// the claim `points` (recomputed by the caller), with a challenger in the prover's
    /// state. Returns the claimed values, which the caller checks against its own
    /// expectations.
    ///
    /// # Errors
    /// On a malformed commitment or claim list, or a Flock rejection.
    pub fn verify<Ch: Challenger>(
        &self,
        circuits: &[&dyn LincheckCircuit],
        commitment: &KeccakCommitment,
        proof: &UnionProof,
        points: &[Vec<F128>],
        bit_claims: &[(BitClaim, F128)],
        ch: &mut Ch,
    ) -> Result<Vec<F128>, KeccakError> {
        assert_eq!(circuits.len(), self.types(), "one circuit per type");
        let cap = commitment.as_bytes();
        if cap.len() != self.commitment_len() {
            return Err(KeccakError::Malformed("commitment length"));
        }
        if proof.values.len() != points.len() {
            return Err(KeccakError::Malformed("claim count"));
        }
        let word_vars = self.word_vars();
        if points.iter().any(|p| p.len() != word_vars)
            || bit_claims.iter().any(|(c, _)| c.x_full.len() != word_vars + 1)
        {
            return Err(KeccakError::Statement("claim point length"));
        }
        let commitment = Commitment {
            cap: cap.as_chunks::<CAP_NODE_BYTES>().0.to_vec(),
            params: self.pcs_params.clone(),
        };
        // The serialized proof leaves out the Ligerito level-0 cap (module docs): Flock's
        // verifier compares it with the commitment's, so the commitment's cap is put in its
        // place (a proof whose cap differed would be rejected there).
        let mut opening = proof.flock.pcs_open.clone();
        if opening.inner.ligerito.initial_cap.is_empty() {
            opening
                .inner
                .ligerito
                .initial_cap
                .clone_from(&commitment.cap);
        }
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
        let mut slot_circuits: Vec<&dyn LincheckCircuit> = circuits.to_vec();
        for (i, &circuit) in circuits.iter().enumerate() {
            slot_circuits[self.slot_of[i]] = circuit;
        }
        // Replays the rounds and discharges the matrix work (the circuits' folds at the
        // verifier's eq tables) on the caller's thread pool.
        let lc = lincheck::verify_union_with_grinding(
            &union,
            &slot_circuits,
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
        let mut values = vec![ab.value, c.value];
        values.extend(bit_claims.iter().map(|(_, v)| *v));
        let mut z_skips = vec![ab.point.z_skip, c.point.z_skip];
        z_skips.extend(bit_claims.iter().map(|(c, _)| c.skip_point()));
        let mut x_fulls = vec![x_full(&ab.point), x_full(&c.point)];
        x_fulls.extend(bit_claims.iter().map(|(c, _)| c.x_full.clone()));
        let x_refs: Vec<&[F128]> = x_fulls.iter().map(Vec::as_slice).collect();
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
            &opening,
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
        // Malformed framing (the round trip of a real proof, without its level-0 cap, is in
        // the tied-part round trips of `tests/all/tied.rs`).
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

/// With `AGG_FLOCK_STAMPS` set, print the wall time since the previous stamp of a Flock
/// sub-step on stderr (`flock-stamp <step> <ms>`) and restart the clock.
fn flock_stamp(step: &str, last: &mut std::time::Instant) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("AGG_FLOCK_STAMPS").is_some()) {
        eprintln!("flock-stamp {step} {:.1}", last.elapsed().as_secs_f64() * 1e3);
    }
    *last = std::time::Instant::now();
}

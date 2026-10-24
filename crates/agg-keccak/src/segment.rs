//! Segment driver: one Flock proof per segment covering the segment's SHAKE256 sponges and
//! its bridge slot (spec Sections 1, 8 and 10; docs/bridge.md Section 5).
//!
//! The sub-batch is a two-type union: `n` instances of the sponge gate (one per live record,
//! [`crate::sponge`]) and one instance of the bridge type ([`crate::bridge_slot`]). The
//! sponge type's IO schema is `mu` (4 words, `In`), the 48 w1Encode words (`In`, lane `t`
//! at words `4 + 8t .. 4 + 8t + 8`) and `c~` (2 words, `Out`); the bridge type's schema lists
//! the copies of every live input word. Wire classes: `{sponge mu/c~ word of record i,
//! public word}` and `{sponge word 4 + 8t + w of record i, bridge word of (t, i, w)}`, so
//! the slot's bits are word-for-word the sponge inputs (no bit permutation, docs/bridge.md
//! Section 2). Public words: `[mu_i, c~_i]` per record.
//!
//! Order (spec Section 10; deviation K7): [`BridgedSegment::commit`] fixes the union witness,
//! the sponge rows and the bridge copies, none of which depends on the bridge seed `sd`, and
//! returns the event-2 commitment. Then the PIOP draws `sd` (event 3), both sides derive the
//! matrices, the prover computes `Y_sigma`, the PIOP draws `fs_sigma` (event 5), and
//! [`BridgedSegment::prove`] builds the bridge rows from `sd` and `Y_sigma`, seeds the Flock
//! challenger with `fs_sigma`, the sponge gate digest and the bridge digest (which covers
//! `sd` and `Y_sigma`), and proves. [`BridgedSegment::verify`] rebuilds the same rows from
//! its own `sd`, matrices and `Y_sigma`.

use std::sync::Arc;

use bincode::Options as _;
use flock_core::{
    challenger::{Challenger, FsChallenger},
    circuit::{Cell, Circuit},
    lincheck::LincheckCircuit,
    pcs::{
        commit, commit_lane_major, ligerito::embedded_initial_k_or_default, Commitment, PcsParams,
    },
    proof::R1csProofCircuitMerged,
    schedule::{IoWord, Registry, TableType},
    union::UnionInstance,
    verifier::verify_ligerito_union_circuit,
};
use flock_field::F128;
use flock_prover::prover::{prove_fast_ligerito_union_circuit, UnionSlotProverInput};

use agg_witness::bridge::BridgeMatrices;

use crate::{
    bridge_slot::{BridgeSlot, SlotLayout, RECORD_WORDS},
    gate::ChainGate,
    prover::{
        capacity_log, codec, sponge_public_words, KeccakCommitment, KeccakError, KeccakProof,
        Profile, CAP_NODE_BYTES, DOMAIN,
    },
    sponge::{SpongeRecord, CTILDE_BYTES, MU_BYTES, W1_INPUT_BASE},
    witness::{pack_rows, pack_slot, SlotBuffers},
};

/// Magic prefix of a bridged segment proof.
const MAGIC: &[u8; 6] = b"AGGKCB";
/// Serialization version of bridged proofs.
const VERSION: u8 = 1;
/// Header bytes: magic, version, profile, kind, `n` (LE32), `d_seg`.
const HEADER: usize = 6 + 1 + 1 + 1 + 4 + 1;

/// One Flock union proof over several table types: registry, counts (slot order), the
/// circuit (wiring and public segment) and the PCS parameters.
struct UnionContext {
    registry: Registry,
    counts: Vec<usize>,
    circuit: Circuit,
    pcs_params: PcsParams,
}

impl UnionContext {
    fn new(
        registry: Registry,
        counts: Vec<usize>,
        num_public: usize,
        wires: Vec<Vec<Cell>>,
        profile: Profile,
    ) -> Self {
        let circuit = Circuit::new(&registry, counts.clone(), num_public, wires)
            .expect("the segment wiring is a valid circuit");
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
            counts,
            circuit,
            pcs_params,
        }
    }

    fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, self.counts.clone())
    }

    /// Flock's commitment to the per-slot `z` buffers (slot order): the map
    /// `prove_fast_ligerito_union_circuit` applies before it binds the statement (each slot
    /// at its aligned block of the padded union buffer, the dense-stack compaction unless it
    /// is the identity, the lane-major commit when whole lanes are empty).
    fn commit(&self, slots: &[&[F128]]) -> Commitment {
        let union = self.union();
        assert_eq!(
            slots.len(),
            self.registry.num_types(),
            "one buffer per slot"
        );
        let mut z = vec![F128::ZERO; union.packed_len()];
        for (t, src) in slots.iter().enumerate() {
            let range = union.slot_word_range(t);
            assert_eq!(src.len(), range.len(), "slot buffer length");
            z[range].copy_from_slice(src);
        }
        let q = if union.compaction_is_identity() {
            z
        } else {
            union.compact_witness(&z)
        };
        let (commitment, _prover_data) = if self.pcs_params.num_lanes.is_some() {
            commit_lane_major(&q, &self.pcs_params)
        } else {
            commit(&q, &self.pcs_params)
        };
        commitment
    }

    fn commitment_len(&self) -> usize {
        CAP_NODE_BYTES << self.pcs_params.level_zero_cap_depth()
    }

    /// Prove with the slots' buffers and circuits in slot order; returns the proof and the
    /// cap Flock committed to.
    fn prove(
        &self,
        slots: Vec<(SlotBuffers, &dyn LincheckCircuit)>,
        public: &[F128],
        ch: &mut FsChallenger,
    ) -> (R1csProofCircuitMerged, Commitment) {
        let union = self.union();
        let inputs = slots
            .into_iter()
            .map(|(buffers, circuit)| UnionSlotProverInput::new(buffers.into_parts(), circuit))
            .collect();
        let (proof, commitment, _claims) = prove_fast_ligerito_union_circuit(
            &union,
            &self.circuit,
            public,
            &self.pcs_params,
            inputs,
            Vec::new(),
            ch,
        );
        (proof, commitment)
    }

    fn received_commitment(&self, c: &KeccakCommitment) -> Result<Commitment, KeccakError> {
        let b = c.as_bytes();
        if b.len() != self.commitment_len() {
            return Err(KeccakError::Malformed("commitment length"));
        }
        Ok(Commitment {
            cap: b.as_chunks::<CAP_NODE_BYTES>().0.to_vec(),
            params: self.pcs_params.clone(),
        })
    }

    fn verify(
        &self,
        public: &[F128],
        circuits: &[&dyn LincheckCircuit],
        commitment: &KeccakCommitment,
        proof: &R1csProofCircuitMerged,
        ch: &mut FsChallenger,
    ) -> Result<(), KeccakError> {
        let commitment = self.received_commitment(commitment)?;
        verify_ligerito_union_circuit(
            &self.union(),
            &self.circuit,
            public,
            circuits,
            &commitment,
            proof,
            &self.pcs_params,
            ch,
        )
        .map(|_| ())
        .map_err(KeccakError::Flock)
    }
}

/// Serialize a union proof with the bridged header.
fn encode_proof(
    profile: Profile,
    kind: u8,
    n: usize,
    d: u8,
    proof: &R1csProofCircuitMerged,
) -> KeccakProof {
    let payload = codec()
        .serialize(proof)
        .expect("serializing a proof cannot fail");
    let mut bytes = Vec::with_capacity(payload.len() + HEADER);
    bytes.extend_from_slice(MAGIC);
    bytes.push(VERSION);
    bytes.push(profile.id());
    bytes.push(kind);
    bytes.extend_from_slice(&(n as u32).to_le_bytes());
    bytes.push(d);
    bytes.extend_from_slice(&payload);
    KeccakProof { bytes }
}

/// Parse a bridged proof whose header must be `(profile, kind, n, d)`.
fn decode_proof(
    proof: &KeccakProof,
    profile: Profile,
    kind: u8,
    n: usize,
    d: u8,
) -> Result<R1csProofCircuitMerged, KeccakError> {
    let b = &proof.bytes;
    if b.len() < HEADER || &b[..6] != MAGIC {
        return Err(KeccakError::Malformed("magic"));
    }
    if b[6] != VERSION {
        return Err(KeccakError::Malformed("version"));
    }
    if b[7] != profile.id() {
        return Err(KeccakError::Malformed("profile"));
    }
    if b[8] != kind {
        return Err(KeccakError::Malformed("proof kind"));
    }
    if u32::from_le_bytes(b[9..13].try_into().expect("4 bytes")) as usize != n || b[13] != d {
        return Err(KeccakError::Malformed("instance count"));
    }
    codec()
        .deserialize(&b[HEADER..])
        .map_err(|_| KeccakError::Malformed("payload"))
}

/// Check a final vector: `m_3` entries, each 0 or 1.
fn check_y(layout: &SlotLayout, y: &[u8]) -> Result<(), KeccakError> {
    if y.len() != layout.segment.output_bits() || y.iter().any(|&b| b > 1) {
        return Err(KeccakError::Statement("final vector"));
    }
    Ok(())
}

/// The bridge slot's packed buffers for a committed block `z` under a relation (`None`:
/// only `z` and the stripe, for the commitment).
fn bridge_buffers(
    layout: &SlotLayout,
    nu: usize,
    z: &[u64],
    slot: Option<&BridgeSlot>,
) -> SlotBuffers {
    pack_rows(
        layout.k_log(),
        layout.useful_bits(),
        nu,
        1,
        |_, zb, ab, bb| {
            zb.copy_from_slice(z);
            if let Some(slot) = slot {
                slot.apply_ab(z, ab, bb);
            }
        },
    )
}

/// Proof kind byte of a standalone bridge slot.
const KIND_SLOT: u8 = 1;
/// Proof kind byte of a bridged segment (sponges and bridge slot).
const KIND_SEGMENT: u8 = 2;

/// A Flock challenger seeded with `fs` and then the given digests (deviation K4).
fn challenger(fs: &[u8; 32], digests: &[[u8; 32]]) -> FsChallenger {
    let mut ch = FsChallenger::new(DOMAIN);
    ch.observe_bytes(fs);
    for d in digests {
        ch.observe_bytes(d);
    }
    ch
}

/// A committed bridge block (sd-independent) and its commitment.
pub struct BridgeCommitted {
    z: Vec<u64>,
    commitment: KeccakCommitment,
}

impl BridgeCommitted {
    /// The commitment (event 2).
    #[must_use]
    pub fn commitment(&self) -> &KeccakCommitment {
        &self.commitment
    }

    /// The committed block.
    #[must_use]
    pub fn block(&self) -> &[u64] {
        &self.z
    }
}

/// One bridge slot proven on its own (one instance of the bridge type, no copy
/// constraints): Flock proves that the committed bits satisfy `Y = M_3 M_2 M_1 b'` for the
/// public `Y`. The relation of the bridged segment without the sponges; used for the toy
/// shape (`L = 2^12`) and for slot-level tests.
pub struct BridgeBatch {
    layout: SlotLayout,
    profile: Profile,
    nu: usize,
    ctx: UnionContext,
}

impl BridgeBatch {
    /// The proving context of a slot layout.
    #[must_use]
    pub fn new(layout: SlotLayout, profile: Profile) -> Self {
        let nu = capacity_log(1, layout.k_log());
        let registry = Registry::new(vec![layout.table_type()], nu);
        let ctx = UnionContext::new(registry, vec![1], 0, Vec::new(), profile);
        Self {
            layout,
            profile,
            nu,
            ctx,
        }
    }

    #[must_use]
    pub fn layout(&self) -> &SlotLayout {
        &self.layout
    }

    /// Commit to an arbitrary block (tests commit tampered blocks).
    #[must_use]
    pub fn commit_block(&self, z: Vec<u64>) -> BridgeCommitted {
        assert_eq!(z.len(), self.layout.useful_words(), "block length");
        let packed = bridge_buffers(&self.layout, self.nu, &z, None);
        let commitment = KeccakCommitment::from_flock(&self.ctx.commit(&[&packed.z]));
        BridgeCommitted { z, commitment }
    }

    /// Commit to the honest block of lane-major packed input bits.
    #[must_use]
    pub fn commit_bits(&self, bits: &[u64]) -> BridgeCommitted {
        self.commit_block(self.layout.honest_z(bits))
    }

    /// The honest final vector of a committed block.
    #[must_use]
    pub fn final_vector(&self, committed: &BridgeCommitted, matrices: &BridgeMatrices) -> Vec<u8> {
        BridgeSlot::parities(&self.layout, matrices, &committed.z)
    }

    /// Prove the committed block against `(sd, Y)` with the Flock seed `fs`.
    ///
    /// # Panics
    /// If Flock commits to another cap than [`Self::commit_block`] (a bug).
    #[must_use]
    pub fn prove(
        &self,
        committed: BridgeCommitted,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        y: Vec<u8>,
        fs: &[u8; 32],
    ) -> KeccakProof {
        let slot = BridgeSlot::new(self.layout, sd, matrices, y);
        let buffers = bridge_buffers(&self.layout, self.nu, &committed.z, Some(&slot));
        let mut ch = challenger(fs, &[slot.digest()]);
        let (proof, commitment) = self.ctx.prove(vec![(buffers, &slot)], &[], &mut ch);
        assert!(
            commitment.cap.as_flattened() == committed.commitment.as_bytes(),
            "Flock's prover committed to a different cap than BridgeBatch::commit_block"
        );
        encode_proof(self.profile, KIND_SLOT, 1, 0, &proof)
    }

    /// Verify a slot proof against the commitment, `(sd, matrices, Y)` and `fs`.
    ///
    /// # Errors
    /// On a malformed proof, commitment or final vector, or a Flock rejection.
    pub fn verify(
        &self,
        commitment: &KeccakCommitment,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        y: &[u8],
        fs: &[u8; 32],
        proof: &KeccakProof,
    ) -> Result<(), KeccakError> {
        check_y(&self.layout, y)?;
        if !self.layout.shape().matches(&matrices) {
            return Err(KeccakError::Statement("bridge matrices"));
        }
        let flock_proof = decode_proof(proof, self.profile, KIND_SLOT, 1, 0)?;
        let slot = BridgeSlot::new(self.layout, sd, matrices, y.to_vec());
        let mut ch = challenger(fs, &[slot.digest()]);
        self.ctx
            .verify(&[], &[&slot], commitment, &flock_proof, &mut ch)
    }
}

/// The sponge type of a bridged segment: the sponge gate with the 48 w1Encode words added
/// to its IO schema as inputs after `mu` (deviation K6).
fn bridged_sponge_type(gate: &ChainGate) -> TableType {
    let spec = gate.spec();
    assert!(
        spec.input_fields.len() == 2
            && spec.input_fields[0].public
            && !spec.input_fields[1].public
            && spec.input_fields[1].bits == 6 * 1024,
        "the bridged sponge gate has fields mu (public) and w1Encode (private, 6,144 bits)"
    );
    let lay = gate.layout();
    let mut schema: Vec<IoWord> = (0..spec.input_fields[0].bits.div_ceil(128))
        .map(|w| IoWord::input(lay.field_base[0] / 128 + w))
        .collect();
    schema.extend((0..48).map(|w| IoWord::input(lay.field_base[1] / 128 + w)));
    schema.extend((0..gate.output_words()).map(|w| IoWord::output(lay.output_base / 128 + w)));
    TableType {
        io_schema: schema,
        ..gate.table_type()
    }
}

/// A committed bridged segment: the sponge slot buffers and the bridge block (both
/// independent of `sd`), the statement words and the event-2 commitment.
pub struct BridgedCommitted {
    n: usize,
    sponge: SlotBuffers,
    bridge_z: Vec<u64>,
    public: Vec<F128>,
    c_tilde: Vec<[u8; CTILDE_BYTES]>,
    commitment: KeccakCommitment,
}

impl BridgedCommitted {
    /// The commitment to absorb at event 2.
    #[must_use]
    pub fn commitment(&self) -> &KeccakCommitment {
        &self.commitment
    }

    /// The published `c~_i`.
    #[must_use]
    pub fn c_tilde(&self) -> &[[u8; CTILDE_BYTES]] {
        &self.c_tilde
    }

    /// The committed bridge block.
    #[must_use]
    pub fn bridge_block(&self) -> &[u64] {
        &self.bridge_z
    }

    /// Records in the segment.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false (a segment has at least one record).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// The prover's output for one segment after event 5: `Y_sigma` and the Flock proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgedProof {
    /// `Y_sigma`, one bit per byte (`m_3` entries).
    pub final_vector: Vec<u8>,
    /// The event-6 proof bytes.
    pub proof: KeccakProof,
}

/// Size and nonzero counts of a bridged segment's relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentCost {
    /// Records.
    pub records: usize,
    /// Structural nonzeros of the `n` sponge instances (`n * ChainGate::cost`).
    pub sponge_nonzeros: u64,
    /// Nonzeros of the factored bridge rows that the walker touches
    /// (`k_1 J + k_2 lanes zeta_2 + k_3 J_3` over the slot).
    pub bridge_factored_nonzeros: u64,
    /// The bridge slot's other nonzeros (identity rows, constant, `B` and `C` of the
    /// composite rows, the `Y` coefficients).
    pub bridge_identity_nonzeros: u64,
    /// Copy constraints between the sponge rows and the bridge slot (`48 n` wire classes).
    pub copy_wires: usize,
    /// `k_log` of the bridge type.
    pub bridge_k_log: usize,
    /// Committed 128-bit words of the bridge slot (one live row).
    pub bridge_committed_words: usize,
    /// Committed 128-bit words of the sponge rows.
    pub sponge_committed_words: usize,
}

/// Proving and verifying context of one bridged segment: `n` live records in a segment of
/// `2^{d_seg}` record positions.
pub struct BridgedSegment<'g> {
    gate: &'g ChainGate,
    n: usize,
    d_seg: u32,
    layout: SlotLayout,
    profile: Profile,
    nu: usize,
    /// Registry slot index of the bridge type (the sponge type is the other one).
    bridge_slot: usize,
    ctx: UnionContext,
}

impl<'g> BridgedSegment<'g> {
    /// `d_seg` of a single-segment batch of `n <= 2^13` records: `log2 max(next_pow2(n), 8)`
    /// (spec Section 1).
    #[must_use]
    pub fn d_seg_for(n: usize) -> u32 {
        assert!(
            (1..=1 << 13).contains(&n),
            "one segment holds 1 to 2^13 records"
        );
        n.next_power_of_two().max(8).trailing_zeros()
    }

    /// The context of `n` live records (the first `n` record positions) of a segment of
    /// `2^{d_seg}` positions, for the production bridge shape and the given sponge gate.
    ///
    /// # Panics
    /// If `n == 0`, `n > 2^{d_seg}` or `d_seg > 13`.
    #[must_use]
    pub fn new(gate: &'g ChainGate, n: usize, d_seg: u32, profile: Profile) -> Self {
        assert!(n >= 1 && n <= 1 << d_seg, "1 <= n <= 2^d_seg");
        let layout = SlotLayout::production(d_seg, n);
        let sponge_type = bridged_sponge_type(gate);
        let bridge_type = layout.table_type();
        let nu = capacity_log(n, gate.layout().k_log);
        let (bridge_words, sponge_words) =
            (bridge_type.io_schema.len(), sponge_type.io_schema.len());
        let bridge_useful = bridge_type.useful_bits;
        let registry = Registry::new(vec![sponge_type, bridge_type], nu);
        let bridge_slot = usize::from(registry.types()[0].useful_bits != bridge_useful);
        assert_eq!(registry.types()[bridge_slot].useful_bits, bridge_useful);
        let mut counts = vec![n, n];
        counts[bridge_slot] = 1;
        // Cell slots: registry order, schema order; then the public slots.
        let (bridge_base, sponge_base) = if bridge_slot == 0 {
            (0, bridge_words)
        } else {
            (sponge_words, 0)
        };
        let pub_base = bridge_words + sponge_words;
        let pub_cell = |p: usize| Cell::new(pub_base + (p >> nu), p & ((1 << nu) - 1));
        let words = sponge_public_words(&[0; MU_BYTES], &[0; CTILDE_BYTES]).len();
        let (mu_words, w1_words) = (MU_BYTES / 16, 48);
        let mut wires = Vec::with_capacity(n * (words + w1_words));
        for i in 0..n {
            for w in 0..mu_words {
                wires.push(vec![Cell::new(sponge_base + w, i), pub_cell(words * i + w)]);
            }
            for w in 0..words - mu_words {
                let s = sponge_base + mu_words + w1_words + w;
                wires.push(vec![Cell::new(s, i), pub_cell(words * i + mu_words + w)]);
            }
            for t in 0..6 {
                for w in 0..RECORD_WORDS {
                    wires.push(vec![
                        Cell::new(sponge_base + mu_words + RECORD_WORDS * t + w, i),
                        Cell::new(bridge_base + layout.schema_index(t, i, w), 0),
                    ]);
                }
            }
        }
        let ctx = UnionContext::new(registry, counts, words * n, wires, profile);
        Self {
            gate,
            n,
            d_seg,
            layout,
            profile,
            nu,
            bridge_slot,
            ctx,
        }
    }

    /// Records in the segment.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false (a segment has at least one record).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The bridge slot layout.
    #[must_use]
    pub fn layout(&self) -> &SlotLayout {
        &self.layout
    }

    /// Row capacity `log2`.
    #[must_use]
    pub fn nu(&self) -> usize {
        self.nu
    }

    /// Union address-space variables.
    #[must_use]
    pub fn m_total(&self) -> usize {
        self.ctx.registry.m_total()
    }

    /// Committed polynomial variables (dense stack, bits).
    #[must_use]
    pub fn dense_m(&self) -> usize {
        self.ctx.pcs_params.m
    }

    /// Size and nonzeros of the relation; `y_weight` is the number of ones of `Y_sigma`.
    #[must_use]
    pub fn cost(&self, y_weight: usize) -> SegmentCost {
        SegmentCost {
            records: self.n,
            sponge_nonzeros: self.n as u64 * self.gate.cost().structural_nonzeros as u64,
            bridge_factored_nonzeros: self.layout.segment.factored_nonzeros(),
            bridge_identity_nonzeros: self.layout.identity_nonzeros(y_weight),
            copy_wires: 48 * self.n,
            bridge_k_log: self.layout.k_log(),
            bridge_committed_words: self.layout.useful_bits().div_ceil(128),
            sponge_committed_words: self.n * self.gate.layout().useful_bits.div_ceil(128),
        }
    }

    /// The honest sponge slot buffers and bridge block of `records` (`n` of them).
    #[must_use]
    pub fn honest_parts(&self, records: &[SpongeRecord]) -> (SlotBuffers, Vec<u64>) {
        assert_eq!(records.len(), self.n, "one record per live position");
        let inputs: Vec<Vec<u64>> = records.iter().map(SpongeRecord::input_bits).collect();
        let sponge = pack_slot(self.gate, self.nu, self.n, |j, z, a, b| {
            self.gate.instance_rows(&inputs[j], z, a, b);
        });
        let mut bits = vec![0_u64; self.layout.input_bits() / 64];
        let lane = self.layout.segment.lane_len;
        for (i, input) in inputs.iter().enumerate() {
            // Lane t of a record is gate input bits W1_INPUT_BASE + 1024 t + (256 rho + j),
            // 16 aligned u64 words (`SpongeRecord::input_bits`).
            for t in 0..6 {
                let src = (W1_INPUT_BASE + 1024 * t) / 64;
                let dst = (lane * t + 1024 * i) / 64;
                bits[dst..dst + 16].copy_from_slice(&input[src..src + 16]);
            }
        }
        (sponge, self.layout.honest_z(&bits))
    }

    /// Commit to given parts (tests commit tampered witnesses); `records` give the
    /// statement words `[mu_i, c~_i]` (with `c~_i` computed natively).
    #[must_use]
    pub fn commit_parts(
        &self,
        sponge: SlotBuffers,
        bridge_z: Vec<u64>,
        records: &[SpongeRecord],
    ) -> BridgedCommitted {
        assert_eq!(records.len(), self.n, "one record per live position");
        assert_eq!(
            bridge_z.len(),
            self.layout.useful_words(),
            "bridge block length"
        );
        let packed = bridge_buffers(&self.layout, self.nu, &bridge_z, None);
        let slots: [&[F128]; 2] = if self.bridge_slot == 0 {
            [&packed.z, &sponge.z]
        } else {
            [&sponge.z, &packed.z]
        };
        let commitment = KeccakCommitment::from_flock(&self.ctx.commit(&slots));
        let c_tilde: Vec<[u8; CTILDE_BYTES]> = records.iter().map(SpongeRecord::c_tilde).collect();
        let public = records
            .iter()
            .zip(&c_tilde)
            .flat_map(|(r, c)| sponge_public_words(&r.mu, c))
            .collect();
        BridgedCommitted {
            n: self.n,
            sponge,
            bridge_z,
            public,
            c_tilde,
            commitment,
        }
    }

    /// Commit to the honest witness of `records` (event 2: before `sd` and `fs_sigma`).
    #[must_use]
    pub fn commit(&self, records: &[SpongeRecord]) -> BridgedCommitted {
        let (sponge, bridge_z) = self.honest_parts(records);
        self.commit_parts(sponge, bridge_z, records)
    }

    /// The honest `Y_sigma = M_3 M_2 M_1 b'` of a committed segment under the matrices of `sd`.
    #[must_use]
    pub fn final_vector(&self, committed: &BridgedCommitted, matrices: &BridgeMatrices) -> Vec<u8> {
        BridgeSlot::parities(&self.layout, matrices, &committed.bridge_z)
    }

    /// Prove a committed segment with the bridge rows of `sd` and the given `Y_sigma`
    /// (tests pass a wrong one), seeded by `fs` (`fs_sigma`).
    ///
    /// # Panics
    /// If `committed` belongs to another segment shape, or Flock commits to another cap.
    #[must_use]
    pub fn prove_with_final_vector(
        &self,
        committed: BridgedCommitted,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        y: Vec<u8>,
        fs: &[u8; 32],
    ) -> KeccakProof {
        assert_eq!(committed.n, self.n, "committed segment of another shape");
        let BridgedCommitted {
            sponge,
            bridge_z,
            public,
            commitment: absorbed,
            ..
        } = committed;
        let slot = BridgeSlot::new(self.layout, sd, matrices, y);
        let bridge = bridge_buffers(&self.layout, self.nu, &bridge_z, Some(&slot));
        let mut ch = challenger(fs, &[self.gate.digest(), slot.digest()]);
        let gate: &dyn LincheckCircuit = self.gate;
        let slots = if self.bridge_slot == 0 {
            vec![(bridge, &slot as &dyn LincheckCircuit), (sponge, gate)]
        } else {
            vec![(sponge, gate), (bridge, &slot as &dyn LincheckCircuit)]
        };
        let (proof, commitment) = self.ctx.prove(slots, &public, &mut ch);
        assert!(
            commitment.cap.as_flattened() == absorbed.as_bytes(),
            "Flock's prover committed to a different cap than BridgedSegment::commit"
        );
        encode_proof(self.profile, KIND_SEGMENT, self.n, self.d_seg as u8, &proof)
    }

    /// Compute `Y_sigma` from the committed bridge block and prove (the honest prover).
    #[must_use]
    pub fn prove(
        &self,
        committed: BridgedCommitted,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        fs: &[u8; 32],
    ) -> BridgedProof {
        let y = self.final_vector(&committed, &matrices);
        let proof = self.prove_with_final_vector(committed, sd, matrices, y.clone(), fs);
        BridgedProof {
            final_vector: y,
            proof,
        }
    }

    /// Verify a segment proof against `(mu_i, c~_i)`, the event-2 commitment, the bridge
    /// seed and its matrices, `Y_sigma` and `fs_sigma`.
    ///
    /// # Errors
    /// On a malformed statement, proof or commitment, or a Flock rejection.
    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        mu: &[[u8; MU_BYTES]],
        c_tilde: &[[u8; CTILDE_BYTES]],
        commitment: &KeccakCommitment,
        sd: [u8; 32],
        matrices: Arc<BridgeMatrices>,
        y: &[u8],
        fs: &[u8; 32],
        proof: &KeccakProof,
    ) -> Result<(), KeccakError> {
        if mu.len() != self.n || c_tilde.len() != self.n {
            return Err(KeccakError::Statement("record count"));
        }
        check_y(&self.layout, y)?;
        if !self.layout.shape().matches(&matrices) {
            return Err(KeccakError::Statement("bridge matrices"));
        }
        let flock_proof =
            decode_proof(proof, self.profile, KIND_SEGMENT, self.n, self.d_seg as u8)?;
        let public: Vec<F128> = mu
            .iter()
            .zip(c_tilde)
            .flat_map(|(m, c)| sponge_public_words(m, c))
            .collect();
        let slot = BridgeSlot::new(self.layout, sd, matrices, y.to_vec());
        let mut ch = challenger(fs, &[self.gate.digest(), slot.digest()]);
        let gate: &dyn LincheckCircuit = self.gate;
        let circuits: [&dyn LincheckCircuit; 2] = if self.bridge_slot == 0 {
            [&slot, gate]
        } else {
            [gate, &slot]
        };
        self.ctx
            .verify(&public, &circuits, commitment, &flock_proof, &mut ch)
    }
}

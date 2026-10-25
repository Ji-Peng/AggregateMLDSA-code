//! Prove/verify API: `n` instances of a chain gate in one Flock union-circuit
//! proof (Ligerito PCS), plus the batched SHAKE256 sponge front end.
//!
//! One proof covers one sub-batch. The registry has one Keccak gate type
//! (`A = I (x) A_0`, one row per instance, capacity `2^nu`); the circuit wires
//! every public input word and every output word of every live instance to a
//! public word of the statement (`mu_i` and `c~_i` for the sponge). Dummy rows
//! `[n, 2^nu)` are identically zero, which satisfies the relation with the
//! constant pinned to 0 (Flock's count-derived pin target).
//!
//! Commit, then prove (spec Section 8, driver; Section 10, events 2 to 6):
//! [`GateBatch::commit`] fixes the witness and returns its [`KeccakCommitment`]
//! before any PIOP challenge that depends on it (the PIOP absorbs it at event
//! 2 and then draws `sd` and `fs_sigma`); [`GateBatch::prove_committed`] runs
//! the Flock prover with `fs_sigma` and checks that Flock commits to the same
//! cap; [`GateBatch::verify`] takes the commitment the PIOP absorbed and hands
//! exactly that to Flock's verifier. The proof bytes do not repeat the
//! commitment, so there is no second copy that could differ from event 2.
//!
//! Fiat-Shamir: the Flock challenger is created with the domain
//! `agg-mldsa/v1/keccak`, then absorbs the 32-byte seed `fs` (spec Section 10,
//! event 5) and the gate digest (the registry type carries empty matrix stubs,
//! so the relation is bound here), then Flock binds the registry, counts,
//! commitment, circuit digest and public words.

use std::fmt;

use bincode::Options as _;

use flock_core::{
    challenger::{Challenger, FsChallenger},
    circuit::{Cell, Circuit},
    lincheck::LincheckCircuit,
    pcs::{
        commit, commit_lane_major,
        ligerito::{embedded_initial_k_or_default, LigeritoProfile},
        Commitment, PcsParams,
    },
    proof::R1csProofCircuitMerged,
    schedule::Registry,
    union::{UnionInstance, MIN_DENSE_M},
    verifier::{verify_ligerito_union_circuit, FlockVerifyError},
};
use flock_field::F128;
use flock_prover::prover::{prove_fast_ligerito_union_circuit, UnionSlotProverInput};

use crate::{
    gate::ChainGate,
    sponge::{self, SpongeRecord, CTILDE_BYTES, MU_BYTES, N_PERMS},
    witness::{pack_slot, SlotBuffers},
};

/// Transcript domain of every Keccak sub-proof.
pub const DOMAIN: &[u8] = b"agg-mldsa/v1/keccak";
/// Magic prefix of serialized proofs.
pub(crate) const MAGIC: &[u8; 6] = b"AGGKCK";
/// Serialization version (2: the commitment travels separately, see [`KeccakCommitment`]).
const VERSION: u8 = 2;
/// Bytes of one Merkle cap node (`flock_merkle::Hash`).
pub(crate) const CAP_NODE_BYTES: usize = 32;
/// Upper bound on a decoded proof payload (Flock's own bundle limit).
const MAX_PROOF_BYTES: u64 = 64 * 1024 * 1024;

/// Fixed-width little-endian bincode, bounded, no trailing bytes.
pub(crate) fn codec() -> impl bincode::Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_PROOF_BYTES)
        .reject_trailing_bytes()
}

/// Ligerito profile of the Flock sub-proofs (Flock's own accounting,
/// `flock_core::pcs::ligerito::LigeritoProfile`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Profile {
    /// Flock's `Fast`: rate 1/2, Johnson list-decoding regime with out-of-domain binding,
    /// 16-bit query proof of work, 128 bits work-normalized. The engineering baseline.
    #[default]
    Fast,
    /// Flock's `Secure`: rate 1/2, unique-decoding regime, 120-bit soundness.
    Secure,
    /// Flock's `Slim`: rate 1/4 with the accounting of `Fast` (128 bits work-normalized);
    /// about half the opening bytes for about twice the level-0 encoding work.
    Slim,
    /// `Slim8` of the vendored Flock core (deviation K18): rate 1/8 at level 0 with `Slim`'s
    /// accounting, derived and validated by Flock's own `derive_profile`/`validate`; about
    /// two thirds of `Slim`'s level-0 queries for twice its level-0 encoding work.
    Slim8,
}

impl Profile {
    pub(crate) fn ligerito(self) -> LigeritoProfile {
        match self {
            Self::Fast => LigeritoProfile::Fast,
            Self::Secure => LigeritoProfile::Secure,
            Self::Slim => LigeritoProfile::Slim,
            Self::Slim8 => LigeritoProfile::Slim8,
        }
    }

    /// `floor(L_max)`, the Johnson-list size of a level-0 Ligerito commitment
    /// (`docs/soundness.md` Section 3.6; Flock Eq. (10) at `eta = 0.02`): 35, 50, 70 for
    /// `Fast`, `Slim`, `Slim8` (rates 1/2, 1/4, 1/8); 1 for the unique-decoding `Secure`.
    #[must_use]
    pub fn list_size(self) -> u64 {
        match self {
            Self::Secure => 1,
            _ => agg_bridge::pow::list_size(self.ligerito().log_inv_rate() as u32),
        }
    }

    /// The profile byte of the proof headers: 0 `Fast`, 1 `Secure`, 3 `Slim`, 4 `Slim8` (2 is
    /// reserved for LaBinius).
    #[must_use]
    pub fn id(self) -> u8 {
        match self {
            Self::Fast => 0,
            Self::Secure => 1,
            Self::Slim => 3,
            Self::Slim8 => 4,
        }
    }

    /// The profile of a header byte.
    #[must_use]
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Fast),
            1 => Some(Self::Secure),
            3 => Some(Self::Slim),
            4 => Some(Self::Slim8),
            _ => None,
        }
    }
}

/// Errors of the verifier and of proof parsing.
#[derive(Debug)]
pub enum KeccakError {
    /// Bytes are not a proof of this format, profile or instance count.
    Malformed(&'static str),
    /// The statement does not have the shape the proof claims.
    Statement(&'static str),
    /// Flock rejected the proof.
    Flock(FlockVerifyError),
    /// A tie between two instances failed (`crate::tied`): the named claim's value does
    /// not match the other instance's or the verifier's table.
    Tie(&'static str),
}

impl fmt::Display for KeccakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(m) => write!(f, "malformed proof: {m}"),
            Self::Statement(m) => write!(f, "bad statement: {m}"),
            Self::Flock(e) => write!(f, "flock verifier rejected: {e:?}"),
            Self::Tie(m) => write!(f, "tie claim mismatch: {m}"),
        }
    }
}

impl std::error::Error for KeccakError {}

/// A serialized Flock union-circuit proof (header, then the proof body). It
/// does not contain the commitment: the verifier takes that from the PIOP
/// transcript (event 2) and passes it to [`GateBatch::verify`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeccakProof {
    pub bytes: Vec<u8>,
}

/// The Flock commitment of one sub-proof: the Ligerito level-0 Merkle cap,
/// `2^c` nodes of 32 bytes (`c = PcsParams::level_zero_cap_depth`), which is
/// exactly what Flock's `bind_statement` absorbs (the PCS parameters are not
/// transmitted; both sides derive them from `n` and the profile).
/// [`Self::as_bytes`] is the event-2 payload of spec Section 10.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeccakCommitment {
    bytes: Vec<u8>,
}

impl KeccakCommitment {
    /// Wrap received bytes (the event-2 payload). The length is checked
    /// against the sub-proof's parameters by [`GateBatch::verify`].
    #[must_use]
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// The canonical encoding: the cap nodes in order, 32 bytes each.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn from_flock(c: &Commitment) -> Self {
        Self {
            bytes: c.cap.as_flattened().to_vec(),
        }
    }
}

/// A sub-batch witness committed before the PIOP draws `sd` and `fs_sigma`
/// (spec Section 8, driver). Holds the slot buffers until
/// [`GateBatch::prove_committed`], so nothing about the witness can change
/// after its commitment entered the transcript.
pub struct CommittedBatch {
    n: usize,
    profile: Profile,
    gate_digest: [u8; 32],
    buffers: SlotBuffers,
    commitment: KeccakCommitment,
}

impl CommittedBatch {
    /// The commitment to absorb at event 2.
    #[must_use]
    pub fn commitment(&self) -> &KeccakCommitment {
        &self.commitment
    }

    /// Instances in the committed sub-batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false (a batch has at least one instance).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Smallest full-height count for which the opening's block-first transport is kept.
const RECT_MIN_ROWS_LOG: usize = 11;

/// Row-capacity `log2` for `n` instances: `ceil(log2 n)`, at least 3 (Flock's
/// lincheck stripe) and at least `MIN_DENSE_M - k_log` (smallest Ligerito
/// config). A full-height stack (`n = 2^nu`) makes Flock's merged opening take
/// its block-first transport, whose cost is about `128^2` field operations per
/// used chunk-column (2,155 for the sponge gate) regardless of `n`; below
/// `2^11` rows that fixed cost exceeds the dense path's, so `nu` is raised by
/// one. A function of `n` only, so prover and verifier agree.
#[must_use]
pub fn capacity_log(n: usize, k_log: usize) -> usize {
    let mut nu = (n.next_power_of_two().trailing_zeros() as usize)
        .max(3)
        .max(MIN_DENSE_M.saturating_sub(k_log));
    if n == 1 << nu && nu < RECT_MIN_ROWS_LOG {
        nu += 1;
    }
    nu
}

/// Proving and verifying context for `n` instances of one gate in one proof.
pub struct GateBatch<'g> {
    gate: &'g ChainGate,
    n: usize,
    profile: Profile,
    registry: Registry,
    circuit: Circuit,
    pcs_params: PcsParams,
}

impl<'g> GateBatch<'g> {
    /// Build the registry (capacity `2^nu`, `nu >= max(3, 22 - k_log)`), the
    /// circuit (public wiring) and the PCS parameters.
    ///
    /// # Panics
    /// If `n == 0`.
    #[must_use]
    pub fn new(gate: &'g ChainGate, n: usize, profile: Profile) -> Self {
        assert!(n >= 1, "at least one instance");
        let nu = capacity_log(n, gate.layout().k_log);
        let registry = Registry::new(vec![gate.table_type()], nu);
        let n_schema = gate.io_schema().len();
        let pub_cell = |p: usize| Cell::new(n_schema + (p >> nu), p & ((1 << nu) - 1));
        let mut wires = Vec::with_capacity(n * n_schema);
        for j in 0..n {
            for slot in 0..n_schema {
                wires.push(vec![Cell::new(slot, j), pub_cell(j * n_schema + slot)]);
            }
        }
        let circuit = Circuit::new(&registry, vec![n], n * n_schema, wires)
            .expect("the public wiring is a valid circuit");
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
            gate,
            n,
            profile,
            registry,
            circuit,
            pcs_params,
        }
    }

    /// Instances in the proof.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Always false (a batch has at least one instance).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Row capacity `log2`.
    #[must_use]
    pub fn nu(&self) -> usize {
        self.registry.nu()
    }

    /// Committed polynomial variables (dense stack, bits).
    #[must_use]
    pub fn dense_m(&self) -> usize {
        self.pcs_params.m
    }

    /// Union address-space variables `nu + k_log`.
    #[must_use]
    pub fn m_total(&self) -> usize {
        self.registry.m_total()
    }

    /// Public words per instance (public input words, then output words).
    #[must_use]
    pub fn words_per_instance(&self) -> usize {
        self.gate.io_schema().len()
    }

    fn challenger(&self, fs: &[u8; 32]) -> FsChallenger {
        let mut ch = FsChallenger::new(DOMAIN);
        ch.observe_bytes(fs);
        ch.observe_bytes(&self.gate.digest());
        ch
    }

    /// Pack the honest witness of the given packed inputs.
    #[must_use]
    pub fn honest_buffers(&self, inputs: &[Vec<u64>]) -> SlotBuffers {
        assert_eq!(inputs.len(), self.n);
        pack_slot(self.gate, self.nu(), self.n, |j, z, a, b| {
            self.gate.instance_rows(&inputs[j], z, a, b);
        })
    }

    /// Bytes of this sub-proof's commitment (`32 * 2^c`).
    #[must_use]
    pub fn commitment_len(&self) -> usize {
        CAP_NODE_BYTES << self.pcs_params.level_zero_cap_depth()
    }

    /// Flock's commitment to the slot buffer `z`, computed by the map
    /// `prove_fast_ligerito_union_circuit` applies before it binds the
    /// statement: a single slot spanning the union is passed through, the
    /// dense stack is `z` itself under identity compaction and
    /// `compact_witness(z)` otherwise, and the lane-major commit is used when
    /// the stack leaves whole lanes empty. [`Self::prove_committed`] checks
    /// that Flock's prover arrives at the same cap.
    fn flock_commit(&self, z: &[F128]) -> Commitment {
        let union = UnionInstance::new(&self.registry, vec![self.n]);
        assert_eq!(z.len(), union.packed_len(), "slot buffer length");
        let compacted;
        let q: &[F128] = if union.compaction_is_identity() {
            z
        } else {
            compacted = union.compact_witness(z);
            &compacted
        };
        let (commitment, _prover_data) = if self.pcs_params.num_lanes.is_some() {
            commit_lane_major(q, &self.pcs_params)
        } else {
            commit(q, &self.pcs_params)
        };
        commitment
    }

    /// Commit to prepared slot buffers (spec Section 8, driver: before `sd`
    /// and `fs_sigma` exist). Honest provers use [`Self::commit_inputs`];
    /// tests use this to commit tampered witnesses.
    #[must_use]
    pub fn commit(&self, buffers: SlotBuffers) -> CommittedBatch {
        let commitment = KeccakCommitment::from_flock(&self.flock_commit(&buffers.z));
        CommittedBatch {
            n: self.n,
            profile: self.profile,
            gate_digest: self.gate.digest(),
            buffers,
            commitment,
        }
    }

    /// Commit to the honest witness of the given packed inputs.
    #[must_use]
    pub fn commit_inputs(&self, inputs: &[Vec<u64>]) -> CommittedBatch {
        self.commit(self.honest_buffers(inputs))
    }

    /// Prove a committed sub-batch against the public words `public` with the
    /// Flock challenger seeded by `fs` (`fs_sigma`, drawn after the commitment
    /// was absorbed). Flock 83f963e has no entry that takes a precomputed
    /// commitment, so its prover commits again (deterministically); the
    /// result must equal [`CommittedBatch::commitment`].
    ///
    /// # Panics
    /// If `committed` belongs to a different batch shape (instance count,
    /// profile or gate), or if Flock's commitment differs from the one
    /// computed by [`Self::commit`] (a bug, never a property of the witness).
    #[must_use]
    pub fn prove_committed(
        &self,
        committed: CommittedBatch,
        public: &[F128],
        fs: &[u8; 32],
    ) -> KeccakProof {
        let CommittedBatch {
            n,
            profile,
            gate_digest,
            buffers,
            commitment: absorbed,
        } = committed;
        assert!(
            n == self.n && profile == self.profile && gate_digest == self.gate.digest(),
            "committed batch belongs to a different sub-proof shape"
        );
        let union = UnionInstance::new(&self.registry, vec![self.n]);
        let slot =
            UnionSlotProverInput::new(buffers.into_parts(), self.gate as &dyn LincheckCircuit);
        let mut ch = self.challenger(fs);
        let (proof, commitment, _claims) = prove_fast_ligerito_union_circuit(
            &union,
            &self.circuit,
            public,
            &self.pcs_params,
            vec![slot],
            Vec::new(),
            &mut ch,
        );
        assert!(
            commitment.cap.as_flattened() == absorbed.as_bytes(),
            "Flock's prover committed to a different cap than GateBatch::commit"
        );
        let payload = codec()
            .serialize(&proof)
            .expect("serializing a proof cannot fail");
        let mut bytes = Vec::with_capacity(payload.len() + 12);
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.push(self.profile.id());
        bytes.extend_from_slice(&(self.n as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        KeccakProof { bytes }
    }

    /// Commit and prove in one call, for callers whose `fs` does not depend
    /// on the commitment (tests, benches); the aggregation driver uses
    /// [`Self::commit`] and [`Self::prove_committed`] around the PIOP's
    /// events 2 to 5.
    #[must_use]
    pub fn prove_buffers(
        &self,
        buffers: SlotBuffers,
        public: &[F128],
        fs: &[u8; 32],
    ) -> (KeccakCommitment, KeccakProof) {
        let committed = self.commit(buffers);
        let commitment = committed.commitment().clone();
        (commitment, self.prove_committed(committed, public, fs))
    }

    /// [`Self::prove_buffers`] on the honest witness of the packed inputs;
    /// `public` lists, per instance, its public input words then its output words.
    #[must_use]
    pub fn prove(
        &self,
        inputs: &[Vec<u64>],
        public: &[F128],
        fs: &[u8; 32],
    ) -> (KeccakCommitment, KeccakProof) {
        self.prove_buffers(self.honest_buffers(inputs), public, fs)
    }

    /// Flock's commitment for the verifier: the received cap with the
    /// verifier's own PCS parameters.
    fn received_commitment(&self, c: &KeccakCommitment) -> Result<Commitment, KeccakError> {
        let b = c.as_bytes();
        if b.len() != self.commitment_len() {
            return Err(KeccakError::Malformed("commitment length"));
        }
        // The length check above leaves no remainder.
        let cap = b.as_chunks::<CAP_NODE_BYTES>().0.to_vec();
        Ok(Commitment {
            cap,
            params: self.pcs_params.clone(),
        })
    }

    /// Verify a proof against the public words and the commitment the PIOP
    /// absorbed at event 2 (Flock's verifier binds and opens exactly this cap).
    ///
    /// # Errors
    /// On a malformed proof or commitment, a statement of the wrong length,
    /// or a Flock rejection (including a commitment other than the proof's).
    pub fn verify(
        &self,
        public: &[F128],
        commitment: &KeccakCommitment,
        proof: &KeccakProof,
        fs: &[u8; 32],
    ) -> Result<(), KeccakError> {
        if public.len() != self.n * self.words_per_instance() {
            return Err(KeccakError::Statement("public word count"));
        }
        let commitment = self.received_commitment(commitment)?;
        let b = &proof.bytes;
        if b.len() < 12 || &b[..6] != MAGIC {
            return Err(KeccakError::Malformed("magic"));
        }
        if b[6] != VERSION {
            return Err(KeccakError::Malformed("version"));
        }
        if b[7] != self.profile.id() {
            return Err(KeccakError::Malformed("profile"));
        }
        let n = u32::from_le_bytes(b[8..12].try_into().expect("4 bytes")) as usize;
        if n != self.n {
            return Err(KeccakError::Malformed("instance count"));
        }
        let flock_proof: R1csProofCircuitMerged = codec()
            .deserialize(&b[12..])
            .map_err(|_| KeccakError::Malformed("payload"))?;
        let union = UnionInstance::new(&self.registry, vec![self.n]);
        let circuits: [&dyn LincheckCircuit; 1] = [self.gate];
        let mut ch = self.challenger(fs);
        verify_ligerito_union_circuit(
            &union,
            &self.circuit,
            public,
            &circuits,
            &commitment,
            &flock_proof,
            &self.pcs_params,
            &mut ch,
        )
        .map(|_| ())
        .map_err(KeccakError::Flock)
    }
}

/// Parameters of the sponge prover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeccakParams {
    pub profile: Profile,
    /// Target permutations per Flock proof (sub-batch size, see
    /// [`Self::records_per_proof`]); the default `2^16` gives `2^13` records.
    pub target_permutations: usize,
}

impl Default for KeccakParams {
    fn default() -> Self {
        Self {
            profile: Profile::Fast,
            target_permutations: 1 << 16,
        }
    }
}

impl KeccakParams {
    /// Records per Flock proof: the largest power of two `n` with `7 n <= target`
    /// (at least 1), so sub-batches align with the PIOP's 2^13-record segments
    /// (57,344 permutations at the default target 2^16) and fill their row capacity.
    #[must_use]
    pub fn records_per_proof(&self) -> usize {
        let n = (self.target_permutations / N_PERMS).max(1);
        1 << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Public words of one sponge record: `mu` (4 words) then `c~` (2 words).
#[must_use]
pub fn sponge_public_words(mu: &[u8; MU_BYTES], c_tilde: &[u8; CTILDE_BYTES]) -> Vec<F128> {
    let mut w = sponge::bytes_to_words(mu);
    w.extend(sponge::bytes_to_words(c_tilde));
    w
}

/// The batched SHAKE256 prover: `c~_i = SHAKE256(mu_i || w1Encode_i, 32)`.
pub struct SpongeProver {
    gate: ChainGate,
    params: KeccakParams,
}

impl SpongeProver {
    #[must_use]
    pub fn new(params: KeccakParams) -> Self {
        Self {
            gate: sponge::sponge_gate(),
            params,
        }
    }

    #[must_use]
    pub fn gate(&self) -> &ChainGate {
        &self.gate
    }

    #[must_use]
    pub fn params(&self) -> &KeccakParams {
        &self.params
    }

    /// The proving context for one sub-batch of `n` records.
    #[must_use]
    pub fn batch(&self, n: usize) -> GateBatch<'_> {
        assert!(
            n <= self.params.records_per_proof(),
            "sub-batch larger than the target"
        );
        GateBatch::new(&self.gate, n, self.params.profile)
    }

    /// Statement words of a sub-batch.
    #[must_use]
    pub fn public_words(mu: &[[u8; MU_BYTES]], c_tilde: &[[u8; CTILDE_BYTES]]) -> Vec<F128> {
        mu.iter()
            .zip(c_tilde)
            .flat_map(|(m, c)| sponge_public_words(m, c))
            .collect()
    }

    /// Commit to one sub-batch (`1 <= records.len() <= records_per_proof`):
    /// the witness, its statement words and the published `c~_i` are fixed
    /// here, before the PIOP draws `sd` and `fs_sigma`.
    #[must_use]
    pub fn commit_batch(&self, records: &[SpongeRecord]) -> SpongeCommitted {
        let batch = self.batch(records.len());
        let c_tilde: Vec<[u8; CTILDE_BYTES]> = records.iter().map(SpongeRecord::c_tilde).collect();
        let mu: Vec<[u8; MU_BYTES]> = records.iter().map(|r| r.mu).collect();
        let public = Self::public_words(&mu, &c_tilde);
        let inputs: Vec<Vec<u64>> = records.iter().map(SpongeRecord::input_bits).collect();
        SpongeCommitted {
            inner: batch.commit_inputs(&inputs),
            public,
            c_tilde,
        }
    }

    /// Prove a committed sub-batch with the Flock seed `fs` (`fs_sigma`).
    #[must_use]
    pub fn prove_committed(&self, committed: SpongeCommitted, fs: &[u8; 32]) -> KeccakProof {
        let batch = self.batch(committed.inner.len());
        batch.prove_committed(committed.inner, &committed.public, fs)
    }

    /// Commit and prove one sub-batch in one call (for callers whose `fs`
    /// does not depend on the commitment; see [`GateBatch::prove_buffers`]).
    #[must_use]
    pub fn prove_batch(&self, records: &[SpongeRecord], fs: &[u8; 32]) -> SpongeSubProof {
        let committed = self.commit_batch(records);
        let commitment = committed.commitment().clone();
        let c_tilde = committed.c_tilde.clone();
        SpongeSubProof {
            commitment,
            proof: self.prove_committed(committed, fs),
            c_tilde,
        }
    }

    /// Verify one sub-batch proof against `(mu_i, c~_i)` and the commitment
    /// absorbed at event 2.
    ///
    /// # Errors
    /// See [`GateBatch::verify`].
    pub fn verify_batch(
        &self,
        mu: &[[u8; MU_BYTES]],
        c_tilde: &[[u8; CTILDE_BYTES]],
        commitment: &KeccakCommitment,
        proof: &KeccakProof,
        fs: &[u8; 32],
    ) -> Result<(), KeccakError> {
        if mu.len() != c_tilde.len() || mu.is_empty() || mu.len() > self.params.records_per_proof()
        {
            return Err(KeccakError::Statement("record count"));
        }
        let batch = self.batch(mu.len());
        batch.verify(&Self::public_words(mu, c_tilde), commitment, proof, fs)
    }

    /// Split `records` into sub-batches of `records_per_proof` and commit to
    /// each (the `S` commitments of event 2, in sub-batch order).
    #[must_use]
    pub fn commit_all(&self, records: &[SpongeRecord]) -> Vec<SpongeCommitted> {
        records
            .chunks(self.params.records_per_proof())
            .map(|c| self.commit_batch(c))
            .collect()
    }

    /// Prove the committed sub-batches of [`Self::commit_all`]; sub-batch `s`
    /// uses the seed `fs[s]` (event 5).
    ///
    /// # Panics
    /// If `fs` does not hold exactly one seed per sub-batch.
    #[must_use]
    pub fn prove_all(&self, committed: Vec<SpongeCommitted>, fs: &[[u8; 32]]) -> Vec<KeccakProof> {
        assert_eq!(
            fs.len(),
            committed.len(),
            "one Fiat-Shamir seed per sub-batch"
        );
        committed
            .into_iter()
            .zip(fs)
            .map(|(c, seed)| self.prove_committed(c, seed))
            .collect()
    }
}

/// A committed sponge sub-batch: the committed gate witness, its statement
/// words `[mu_i, c~_i]` and the published `c~_i`.
pub struct SpongeCommitted {
    inner: CommittedBatch,
    public: Vec<F128>,
    c_tilde: Vec<[u8; CTILDE_BYTES]>,
}

impl SpongeCommitted {
    /// The commitment to absorb at event 2.
    #[must_use]
    pub fn commitment(&self) -> &KeccakCommitment {
        self.inner.commitment()
    }

    /// The published `c~_i` of the sub-batch, in record order.
    #[must_use]
    pub fn c_tilde(&self) -> &[[u8; CTILDE_BYTES]] {
        &self.c_tilde
    }

    /// Records in the sub-batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Always false (a sub-batch has at least one record).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// The result of [`SpongeProver::prove_batch`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpongeSubProof {
    /// The event-2 commitment.
    pub commitment: KeccakCommitment,
    /// The event-6 proof bytes.
    pub proof: KeccakProof,
    /// The published `c~_i`.
    pub c_tilde: Vec<[u8; CTILDE_BYTES]>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gate::ChainSpec, keccak};
    use flock_core::test_rng::Rng;

    /// Packed inputs and public words (input then output state) of `n`
    /// random Keccak-f instances.
    fn permutation_statement(n: usize, seed: u64) -> (Vec<Vec<u64>>, Vec<F128>) {
        let words = |s: &keccak::Lanes| -> Vec<F128> {
            s.chunks(2)
                .map(|c| F128::new(c[0], *c.get(1).unwrap_or(&0)))
                .take(13)
                .collect()
        };
        let mut rng = Rng::new(seed);
        let mut inputs = Vec::with_capacity(n);
        let mut public = Vec::with_capacity(26 * n);
        for _ in 0..n {
            let s0: keccak::Lanes = std::array::from_fn(|_| rng.next_u64());
            let mut s1 = s0;
            keccak::keccak_f(&mut s1);
            public.extend(words(&s0));
            public.extend(words(&s1));
            inputs.push(s0.to_vec());
        }
        (inputs, public)
    }

    /// `GateBatch::commit` reproduces the commitment Flock's own prover binds,
    /// under both compaction arms (identity at a full-height 2^11 stack,
    /// gathered below it) and both commit arms (lane-major when the dense
    /// stack leaves whole lanes empty; plain at the first `n` whose `332 n`
    /// words fill every lane of the committed stack).
    #[test]
    #[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
    fn commit_matches_flock_prover_commitment() {
        let gate = ChainGate::new(ChainSpec::permutation());
        let (mut identity, mut gathered) = (false, false);
        let (mut lane_major, mut plain) = (false, false);
        let full_lanes = (2..1024)
            .find(|&n| {
                GateBatch::new(&gate, n, Profile::Fast)
                    .pcs_params
                    .num_lanes
                    .is_none()
            })
            .expect("some count fills every lane");
        for n in [5usize, 64, full_lanes, 2048] {
            let batch = GateBatch::new(&gate, n, Profile::Fast);
            let union = UnionInstance::new(&batch.registry, vec![n]);
            if union.compaction_is_identity() {
                identity = true;
            } else {
                gathered = true;
            }
            if batch.pcs_params.num_lanes.is_some() {
                lane_major = true;
            } else {
                plain = true;
            }
            let (inputs, public) = permutation_statement(n, n as u64);
            let committed = batch.commit_inputs(&inputs);
            assert_eq!(
                committed.commitment().as_bytes().len(),
                batch.commitment_len()
            );
            let slot = UnionSlotProverInput::new(
                batch.honest_buffers(&inputs).into_parts(),
                &gate as &dyn LincheckCircuit,
            );
            let mut ch = batch.challenger(&[1; 32]);
            let (_, flock_commitment, _) = prove_fast_ligerito_union_circuit(
                &union,
                &batch.circuit,
                &public,
                &batch.pcs_params,
                vec![slot],
                Vec::new(),
                &mut ch,
            );
            assert_eq!(
                &KeccakCommitment::from_flock(&flock_commitment),
                committed.commitment(),
                "n = {n}"
            );
        }
        assert!(identity && gathered, "both compaction arms covered");
        assert!(lane_major && plain, "both commit arms covered");
    }

    /// A committed batch whose recorded commitment was altered after the
    /// fact (as if a different cap had been absorbed) cannot be proven.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "heavy Flock proof; run with --release")]
    #[should_panic(expected = "different cap")]
    fn prove_committed_rejects_a_foreign_commitment() {
        let gate = ChainGate::new(ChainSpec::permutation());
        let batch = GateBatch::new(&gate, 8, Profile::Fast);
        let (inputs, public) = permutation_statement(8, 3);
        let mut committed = batch.commit_inputs(&inputs);
        committed.commitment.bytes[0] ^= 1;
        let _ = batch.prove_committed(committed, &public, &[2; 32]);
    }

    /// A committed batch can only be proven by a context of its own shape.
    #[test]
    #[should_panic(expected = "different sub-proof shape")]
    fn prove_committed_rejects_another_shape() {
        let gate = ChainGate::new(ChainSpec::permutation());
        let small = GateBatch::new(&gate, 8, Profile::Fast);
        let large = GateBatch::new(&gate, 16, Profile::Fast);
        let (inputs, _) = permutation_statement(8, 4);
        let (_, public) = permutation_statement(16, 4);
        let committed = small.commit_inputs(&inputs);
        let _ = large.prove_committed(committed, &public, &[2; 32]);
    }
}

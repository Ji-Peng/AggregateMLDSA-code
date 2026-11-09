//! Stage-3 reduction of the Keccak claims of stages 1 and 2 to point openings at `r^(3)`.
//!
//! Stages 1 and 2 leave claims at the Keccak suffixes `R = (r_z, r_s, r_i)` of `r^(1)` and
//! `r^(2)` on the committed Keccak polynomials and on tables re-indexed from them: a rotation
//! in `z` (`Rot`), a rotation plus a one-slot shift that crosses to the previous slot group at
//! `s = 0` (`Prev`), `Prev` of lane `(0, 0)` XORed with a public round constant (`PrevIota`),
//! and the absorbed blocks (`Blk`: wired `b_t` cells plus public padding constants). A
//! re-indexed table is not a permutation of the domain variables (a rotation carries, the
//! slot shift mixes slot groups), so the verifier cannot evaluate it from a point opening.
//! Each such claim is instead a linear functional of committed cells,
//!
//! ```text
//! claim_k - pub_k = sum_{z', s', i} e_k(z') f_k(s') eq(r_i, i) X(z', s', i)   (one term per polynomial read),
//! ```
//!
//! with a 64-entry `z` weight, a 16-entry slot weight and the shared record factor. Every
//! claim gets its own stage-3 coefficient `c_k`; per committed polynomial `X` and point, the
//! claims that read `X` are summed into one 1,024-entry table
//! `P(z', s') = sum_k c_k e_k(z') f_k(s')` (cellwise times `1 - 2 RC` for the XOR with a round
//! constant), so `X` gets one rank-one weight `P(z', s') eq(r_i, i)` per point. The claims on
//! the absorbed blocks reduce, on the main domain, to the committed bits themselves:
//! `sum_k c_k (Blk_k(R) - pad_k(R)) = sum_t sum_{(j, rho, i)} u_t(j, rho) eq(r_i, i) b_t(j, rho, i)`
//! with `u_t(j, rho) = sum_{k wired to (t, j, rho)} c_k eq(r_z, z_k) eq(r_s, s_k)`, so no second
//! copy of the bits exists. The coefficients are folded into the weights, so the stage-3
//! parts that read them are unbatched and their gamma indices are declared folded
//! ([`StageSpec::folded`](crate::stage::StageSpec::folded)).

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use aerie_falcon::protocol::algebra::{eq_evaluation, eq_table};
use agg_fpkeccak::{rot, Absorbed, KeccakLayout, LANE_BITS, RC, ROUNDS_PER_PERM, SLOTS};
use jolt_field::Zero;
use rayon::prelude::*;

use crate::keccak_fp::masks::KeccakPoint;
use crate::keccak_fp::virtuals::{pad_blk_mle_at, rc_out_mle_at, rc_prev_mle_at};
use crate::kernel::Term;
use crate::manifest::{Column, CommitmentManifest, GroupRole};
use crate::split_weight::RankOneWeight;
use crate::stage::{suffix, EqPoint, GroupSpec, InstanceSpec, LaneId, WeightId};
use crate::OpeningField as E;

/// Cells `(z, s)` of one record of a Keccak polynomial.
pub const CELLS: usize = LANE_BITS * SLOTS;

/// The position of a stage point in the per-point tables (`r^(1)` first).
const fn point_index(point: KeccakPoint) -> usize {
    match point {
        KeccakPoint::R1 => 0,
        KeccakPoint::R2 => 1,
    }
}

/// Which slot `s'` of the read polynomial a claimed table reads at its slot `s`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotRead {
    /// `s' = s`: slot weight `f(s') = eq(r_s, s')`.
    Same,
    /// `s' = s - 1` inside the polynomial: `f(s') = eq(r_s, s' + 1)` for `s' <= 14`, zero at 15.
    Next,
    /// Slot 15 of the previous slot group read at `s = 0`: `f(s') = [s' = 15] eq(r_s, 0)`.
    Wrap,
}

impl SlotRead {
    /// The slot `s` of the claimed table that reads source slot `s'`, if any.
    #[must_use]
    pub const fn dest(self, source: usize) -> Option<usize> {
        match self {
            Self::Same => Some(source),
            Self::Next if source + 1 < SLOTS => Some(source + 1),
            Self::Wrap if source + 1 == SLOTS => Some(0),
            Self::Next | Self::Wrap => None,
        }
    }
}

/// A public round constant XORed into the claimed table (`a xor c = (1 - 2 c) a + c`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flip {
    /// No constant.
    None,
    /// The constant of the previous round, `RC_{(t - 1) mod 24}` at round `t = 16 q + s`
    /// (`1 <= t < 24 P`), of the claimed table in slot group `q` (`PrevIota(q)`).
    Iota(u8),
    /// `RC_23` at the output slot (`Rot` of lane `(0, 0)` in the last slot group).
    Output,
}

impl Flip {
    /// True when the constant has a one at cell `(z, s)` of the claimed table.
    #[must_use]
    pub fn bit(self, layout: &KeccakLayout, z: usize, s: usize) -> bool {
        match self {
            Self::None => false,
            Self::Iota(q) => {
                let t = layout.time(usize::from(q), s);
                t >= 1 && t < layout.rounds && (RC[(t - 1) % ROUNDS_PER_PERM] >> z) & 1 == 1
            }
            Self::Output => s == (layout.rounds - 1) % SLOTS && (RC[ROUNDS_PER_PERM - 1] >> z) & 1 == 1,
        }
    }
}

/// One read of a committed polynomial by a claim: source cell `(z', s')` lands on cell
/// `((z' + rot) mod 64, slot.dest(s'))` of the claimed table, negated where `flip` has a one.
/// Its weight is `e(z') = eq(r_z, (z' + rot) mod 64)` times the slot weight of `slot`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Read {
    /// Index of the claim in the reduction order ([`KeccakReductions::claims`]).
    pub claim: usize,
    /// The rotation in `z`.
    pub rot: u8,
    /// The slot map.
    pub slot: SlotRead,
    /// The XORed constant.
    pub flip: Flip,
}

/// The committed reads of a claimed Keccak lane as `(polynomial, rot, slot, flip)`; empty for a
/// block lane (reduced to the bits) and for the public lanes.
///
/// # Panics
/// On a lane that is not a Keccak lane, or a rotated `A'` lane outside the last slot group.
#[must_use]
pub fn reads_of(layout: &KeccakLayout, id: LaneId) -> Vec<(Column, u8, SlotRead, Flip)> {
    let a = |x: u8, y: u8, q: u8| Column::KeccakA { x, y, q };
    match id {
        LaneId::Column(c @ (Column::KeccakA { .. } | Column::KeccakC { .. } | Column::KeccakCPrime { .. })) => {
            vec![(c, 0, SlotRead::Same, Flip::None)]
        }
        // theta_c reads C[x + 1] rotated by one bit.
        LaneId::KeccakRot(c @ Column::KeccakC { .. }) => vec![(c, 1, SlotRead::Same, Flip::None)],
        LaneId::KeccakRot(c @ Column::KeccakA { x, y, q }) => {
            assert_eq!(usize::from(q) + 1, layout.slot_groups, "a rotated A' lane is read only by the output");
            if (x, y) == (0, 0) {
                vec![(c, 0, SlotRead::Same, Flip::Output)]
            } else {
                vec![(c, rot(usize::from(x), usize::from(y)) as u8, SlotRead::Same, Flip::None)]
            }
        }
        LaneId::KeccakPrev(Column::KeccakA { x, y, q }) => {
            let r = rot(usize::from(x), usize::from(y)) as u8;
            let mut reads = vec![(a(x, y, q), r, SlotRead::Next, Flip::None)];
            if q >= 1 {
                reads.push((a(x, y, q - 1), r, SlotRead::Wrap, Flip::None));
            }
            reads
        }
        LaneId::KeccakPrevIota(q) => {
            let mut reads = vec![(a(0, 0, q), 0, SlotRead::Next, Flip::Iota(q))];
            if q >= 1 {
                reads.push((a(0, 0, q - 1), 0, SlotRead::Wrap, Flip::Iota(q)));
            }
            reads
        }
        LaneId::KeccakBlk { .. } | LaneId::KeccakMu(_) | LaneId::KeccakCTilde(_) => Vec::new(),
        other => panic!("{other:?} is not a Keccak claim lane"),
    }
}

/// The public part of a claimed Keccak lane at the Keccak-domain point `r_k`: the
/// multilinear extension of its round constant or padding constants (zero otherwise).
#[must_use]
pub fn public_part(layout: &KeccakLayout, id: LaneId, r_k: &[E]) -> E {
    match id {
        LaneId::KeccakPrevIota(q) => rc_prev_mle_at(layout, usize::from(q), r_k),
        LaneId::KeccakRot(Column::KeccakA { x: 0, y: 0, .. }) => rc_out_mle_at(layout, r_k),
        LaneId::KeccakBlk { x, y, q } => pad_blk_mle_at(layout, usize::from(x), usize::from(y), usize::from(q), r_k),
        _ => E::zero(),
    }
}

/// One wired cell of an absorbed block: claim `claim` reads `b_t` at main-domain cell `cell`
/// (inside a record) at its cell `(z, s)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbsorbCell {
    /// Index of the block claim.
    pub claim: usize,
    /// Bit `z` of the lane.
    pub z: u8,
    /// Slot `s`.
    pub s: u8,
    /// `j + 256 (rho mod lanes)`, the cell of `b_t` in its polynomial lane inside one record.
    pub cell: u32,
}

/// The reduction weight of one committed polynomial at one point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReduceWeight {
    /// The committed polynomial.
    pub poly: Column,
    /// The point.
    pub point: KeccakPoint,
    /// Every claim at `point` that reads `poly`.
    pub reads: Vec<Read>,
}

/// The weight `u_t` of the bits `b_t` of one polynomial lane at one point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbsorbWeight {
    /// Bit index.
    pub t: u8,
    /// Polynomial lane.
    pub lane: u8,
    /// The point.
    pub point: KeccakPoint,
    /// The wired cells of the block claims at `point`.
    pub cells: Vec<AbsorbCell>,
}

/// The claim-source table of the stage-3 Keccak reductions of one parameter set.
#[derive(Clone, Debug)]
pub struct KeccakReductions {
    layout: KeccakLayout,
    gamma0: usize,
    d_keccak: usize,
    d_main: usize,
    d_sig: usize,
    lane_cells: usize,
    claims: Vec<(KeccakPoint, LaneId)>,
    weights: Vec<ReduceWeight>,
    absorbs: Vec<AbsorbWeight>,
}

/// The `eq` tables of one Keccak-domain point `R = (r_z, r_s, r_i)`.
struct PointEq {
    z: Vec<E>,
    s: Vec<E>,
    record: Vec<E>,
}

impl PointEq {
    fn new(r_k: &[E]) -> Self {
        Self { z: eq_table(&r_k[..6]), s: eq_table(&r_k[6..10]), record: r_k[10..].to_vec() }
    }
}

impl KeccakReductions {
    /// The table for the claims `r1_claims` of stage 1 and `r2_claims` of stage 2 (each in
    /// the order of the stage's Keccak claims, `stage1::keccak_claim_lanes`). Claim `k` of the
    /// concatenation gets gamma index `gamma0 + k`.
    #[must_use]
    pub fn new(manifest: &CommitmentManifest, gamma0: usize, r1_claims: &[LaneId], r2_claims: &[LaneId]) -> Self {
        let layout = KeccakLayout::for_set(manifest.set());
        let shape = manifest.shape();
        let claims: Vec<(KeccakPoint, LaneId)> = r1_claims
            .iter()
            .map(|&id| (KeccakPoint::R1, id))
            .chain(r2_claims.iter().map(|&id| (KeccakPoint::R2, id)))
            .collect();
        let poly_index = |c: Column| match c {
            Column::KeccakA { x, y, q } => layout.a_prime_poly(usize::from(x), usize::from(y), usize::from(q)),
            Column::KeccakC { x, q } => layout.c_poly(usize::from(x), usize::from(q)),
            Column::KeccakCPrime { x, q } => layout.c_prime_poly(usize::from(x), usize::from(q)),
            other => panic!("{other:?} is not a Keccak column"),
        };
        let mut weights: BTreeMap<(usize, usize), ReduceWeight> = BTreeMap::new();
        let mut absorbs: BTreeMap<(usize, u8, u8), AbsorbWeight> = BTreeMap::new();
        for (k, &(point, id)) in claims.iter().enumerate() {
            for (poly, rot, slot, flip) in reads_of(&layout, id) {
                weights
                    .entry((point_index(point), poly_index(poly)))
                    .or_insert_with(|| ReduceWeight { poly, point, reads: Vec::new() })
                    .reads
                    .push(Read { claim: k, rot, slot, flip });
            }
            let LaneId::KeccakBlk { x, y, q } = id else { continue };
            let (x, y, q) = (usize::from(x), usize::from(y), usize::from(q));
            for s in 0..SLOTS {
                let t = layout.time(q, s);
                if !((q == 0 && s == 0) || layout.is_boundary(t)) {
                    continue;
                }
                let m = t / ROUNDS_PER_PERM;
                for z in 0..LANE_BITS {
                    let Absorbed::W1Bit { t: bit, j, rho } = layout.wire(m, x, y, z) else { continue };
                    let (lane, base) = shape.row_lane(rho);
                    let (bit, lane) = (bit as u8, lane as u8);
                    absorbs
                        .entry((point_index(point), bit, lane))
                        .or_insert_with(|| AbsorbWeight { t: bit, lane, point, cells: Vec::new() })
                        .cells
                        .push(AbsorbCell { claim: k, z: z as u8, s: s as u8, cell: (base + j) as u32 });
                }
            }
        }
        Self {
            layout,
            gamma0,
            d_keccak: manifest.d_keccak(),
            d_main: manifest.d_main(),
            d_sig: manifest.d_sig(),
            lane_cells: manifest.lane_cells(),
            claims,
            weights: weights.into_values().collect(),
            absorbs: absorbs.into_values().collect(),
        }
    }

    /// [`Self::new`] for the Keccak claims of `manifest`'s stage-1 and stage-2 specs.
    #[must_use]
    pub fn for_manifest(manifest: &CommitmentManifest, gamma0: usize) -> Self {
        let r1 = crate::stage1::keccak_claim_lanes(&crate::stage1::spec(manifest));
        let r2 = crate::stage1::keccak_claim_lanes(&crate::stage2::spec(manifest));
        Self::new(manifest, gamma0, &r1, &r2)
    }

    /// The claims in coefficient order.
    #[must_use]
    pub fn claims(&self) -> &[(KeccakPoint, LaneId)] {
        &self.claims
    }

    /// The reduction weights, by point and then polynomial index.
    #[must_use]
    pub fn weights(&self) -> &[ReduceWeight] {
        &self.weights
    }

    /// The bit weights, by point, bit and lane.
    #[must_use]
    pub fn absorbs(&self) -> &[AbsorbWeight] {
        &self.absorbs
    }

    /// The gamma indices of the claims (folded into the weights).
    #[must_use]
    pub fn gammas(&self) -> Range<usize> {
        self.gamma0..self.gamma0 + self.claims.len()
    }

    /// The stage-3 groups: per reduction weight, `sum_x W(x) X(x)` on the Keccak domain; per bit
    /// weight, `sum_x W(x) b_t(x)` on the main domain. Each is one unbatched part with the
    /// weight in lane 0 and no equality factor.
    #[must_use]
    pub fn groups(&self) -> Vec<GroupSpec> {
        let part = || {
            vec![InstanceSpec::unbatched(vec![Term::Affine { weight: Some(0), lanes: vec![(1, 1)], constant: 0 }])]
        };
        let reduce = self.weights.iter().map(|w| GroupSpec {
            domain: GroupRole::KeccakA(0),
            eq: EqPoint::None,
            lanes: vec![
                LaneId::Weight(WeightId::KeccakReduce { poly: w.poly, point: w.point }),
                LaneId::Column(w.poly),
            ],
            instances: part(),
        });
        let absorb = self.absorbs.iter().map(|w| GroupSpec {
            domain: GroupRole::Main,
            eq: EqPoint::None,
            lanes: vec![
                LaneId::Weight(WeightId::KeccakAbsorb { t: w.t, lane: w.lane, point: w.point }),
                LaneId::Column(Column::W1Bit(w.t, w.lane)),
            ],
            instances: part(),
        });
        reduce.chain(absorb).collect()
    }

    fn point_eq(&self, r1: &[E], r2: &[E]) -> [PointEq; 2] {
        [PointEq::new(&suffix(r1, self.d_keccak)), PointEq::new(&suffix(r2, self.d_keccak))]
    }

    /// The claimed sums of the reductions in coefficient order: each claim minus its public
    /// part. `r1_claims`, `r2_claims` are the stage Keccak claims in the orders given to
    /// [`Self::new`]; `r1`, `r2` the stage points (their Keccak suffixes are used).
    #[must_use]
    pub fn claimed_sums(&self, r1_claims: &[E], r2_claims: &[E], r1: &[E], r2: &[E]) -> Vec<E> {
        assert_eq!(r1_claims.len() + r2_claims.len(), self.claims.len(), "one value per Keccak claim");
        let points = [suffix(r1, self.d_keccak), suffix(r2, self.d_keccak)];
        self.claims
            .par_iter()
            .zip(r1_claims.par_iter().chain(r2_claims.par_iter()))
            .map(|(&(point, id), &value)| value - public_part(&self.layout, id, &points[point_index(point)]))
            .collect()
    }

    /// The prover's weight tables for the coefficients `coeffs` (one per claim, the gamma
    /// entries [`Self::gammas`]) and the stage points `r1`, `r2`.
    #[must_use]
    pub fn tables(&self, coeffs: &[E], r1: &[E], r2: &[E]) -> KeccakReduceTables {
        assert_eq!(coeffs.len(), self.claims.len(), "one coefficient per Keccak claim");
        let eqs = self.point_eq(r1, r2);
        let layout = &self.layout;
        let reduce = self
            .weights
            .par_iter()
            .map(|w| {
                let eq = &eqs[point_index(w.point)];
                let mut table = vec![E::zero(); CELLS];
                for read in &w.reads {
                    let c = coeffs[read.claim];
                    for source_s in 0..SLOTS {
                        let Some(s) = read.slot.dest(source_s) else { continue };
                        let cs = c * eq.s[s];
                        let row = &mut table[source_s * LANE_BITS..(source_s + 1) * LANE_BITS];
                        for (source_z, cell) in row.iter_mut().enumerate() {
                            let z = (source_z + usize::from(read.rot)) % LANE_BITS;
                            let value = cs * eq.z[z];
                            if read.flip.bit(layout, z, s) {
                                *cell -= value;
                            } else {
                                *cell += value;
                            }
                        }
                    }
                }
                ((w.poly, w.point), table)
            })
            .collect();
        let absorb = self
            .absorbs
            .par_iter()
            .map(|w| {
                let eq = &eqs[point_index(w.point)];
                let mut table = vec![E::zero(); self.lane_cells];
                for cell in &w.cells {
                    table[cell.cell as usize] += coeffs[cell.claim] * eq.z[usize::from(cell.z)] * eq.s[usize::from(cell.s)];
                }
                ((w.t, w.lane, w.point), table)
            })
            .collect();
        KeccakReduceTables { reduce, absorb, record: eqs.map(|eq| eq_table(&eq.record)) }
    }

    /// The verifier's evaluation of every weight at the stage-3 point `r3`: a reduction weight is
    /// `P(r3_z, r3_s) eq(r_i, r3_i)` at the Keccak suffix of `r3`, a bit weight
    /// `u_t(r3_pos) eq(r_i, r3_i)` at the main suffix, both computed from `eq` tables of the
    /// claims' points (`O(64 x 16)` per point, `O(1)` per unflipped read and `O(1,024)` per
    /// read with a round constant, one term per wired cell), never over records.
    #[must_use]
    pub fn evaluate(&self, coeffs: &[E], r1: &[E], r2: &[E], r3: &[E]) -> KeccakReduceValues {
        assert_eq!(coeffs.len(), self.claims.len(), "one coefficient per Keccak claim");
        let eqs = self.point_eq(r1, r2);
        let r3_k = suffix(r3, self.d_keccak);
        let r3_main = suffix(r3, self.d_main);
        let (z3, s3) = (eq_table(&r3_k[..6]), eq_table(&r3_k[6..10]));
        let local3 = eq_table(&r3_main[..self.d_main - self.d_sig]);
        let record3 = &r3_k[10..];
        debug_assert_eq!(record3, &r3_main[self.d_main - self.d_sig..], "both domains end in the record variables");
        // Per point: the record factor, the z part of every rotation and the slot part of
        // every slot map.
        let parts: Vec<(E, Vec<E>, [E; 3])> = eqs
            .iter()
            .map(|eq| {
                let z_part = (0..LANE_BITS)
                    .map(|rot| (0..LANE_BITS).fold(E::zero(), |acc, z| acc + eq.z[(z + rot) % LANE_BITS] * z3[z]))
                    .collect();
                let slot = |read: SlotRead| {
                    (0..SLOTS).fold(E::zero(), |acc, s| match read.dest(s) {
                        Some(d) => acc + eq.s[d] * s3[s],
                        None => acc,
                    })
                };
                let s_part = [slot(SlotRead::Same), slot(SlotRead::Next), slot(SlotRead::Wrap)];
                (eq_evaluation(&eq.record, record3), z_part, s_part)
            })
            .collect();
        let slot_index = |read: SlotRead| match read {
            SlotRead::Same => 0,
            SlotRead::Next => 1,
            SlotRead::Wrap => 2,
        };
        let mut values = HashMap::with_capacity(self.weights.len() + self.absorbs.len());
        for w in &self.weights {
            let (eq, (record, z_part, s_part)) = (&eqs[point_index(w.point)], &parts[point_index(w.point)]);
            let low = w.reads.iter().fold(E::zero(), |acc, read| {
                let mut value = z_part[usize::from(read.rot)] * s_part[slot_index(read.slot)];
                if read.flip != Flip::None {
                    // (1 - 2 c) = 1 - 2 [c = 1]: subtract twice the flipped cells.
                    let mut flipped = E::zero();
                    for source_s in 0..SLOTS {
                        let Some(s) = read.slot.dest(source_s) else { continue };
                        for source_z in 0..LANE_BITS {
                            let z = (source_z + usize::from(read.rot)) % LANE_BITS;
                            if read.flip.bit(&self.layout, z, s) {
                                flipped += eq.z[z] * eq.s[s] * z3[source_z] * s3[source_s];
                            }
                        }
                    }
                    value -= flipped + flipped;
                }
                acc + coeffs[read.claim] * value
            });
            values.insert(WeightId::KeccakReduce { poly: w.poly, point: w.point }, low * *record);
        }
        for w in &self.absorbs {
            let (eq, record) = (&eqs[point_index(w.point)], parts[point_index(w.point)].0);
            let low = w.cells.iter().fold(E::zero(), |acc, cell| {
                acc + coeffs[cell.claim] * eq.z[usize::from(cell.z)] * eq.s[usize::from(cell.s)] * local3[cell.cell as usize]
            });
            values.insert(WeightId::KeccakAbsorb { t: w.t, lane: w.lane, point: w.point }, low * record);
        }
        KeccakReduceValues(values)
    }
}

/// The verifier's values of the stage-3 Keccak weights at `r^(3)`.
#[derive(Clone, Debug, Default)]
pub struct KeccakReduceValues(HashMap<WeightId, E>);

impl KeccakReduceValues {
    /// The value of a Keccak weight.
    ///
    /// # Panics
    /// When the weight is not part of the reduction.
    #[must_use]
    pub fn get(&self, id: WeightId) -> E {
        *self.0.get(&id).unwrap_or_else(|| panic!("{id:?} is not a stage-3 Keccak weight"))
    }
}

/// The prover's stage-3 Keccak weights, as their factors: the 1,024-entry tables per
/// (polynomial, point), the `u_t` tables per (bit, lane, point), and `eq(r_i, .)` per point.
#[derive(Clone, Debug)]
pub struct KeccakReduceTables {
    reduce: HashMap<(Column, KeccakPoint), Vec<E>>,
    absorb: HashMap<(u8, u8, KeccakPoint), Vec<E>>,
    record: [Vec<E>; 2],
}

impl KeccakReduceTables {
    /// The factors of a Keccak weight (`None` for another weight or one outside the reduction).
    #[must_use]
    pub fn rank_one(&self, id: WeightId) -> Option<RankOneWeight<'_>> {
        let (low, point) = match id {
            WeightId::KeccakReduce { poly, point } => (self.reduce.get(&(poly, point))?, point),
            WeightId::KeccakAbsorb { t, lane, point } => (self.absorb.get(&(t, lane, point))?, point),
            _ => return None,
        };
        Some(RankOneWeight { low, high: &self.record[point_index(point)] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keccak_fp::virtuals::KeccakProverTables;
    use crate::test_support::point;
    use agg_fpkeccak::{KeccakPublics, KeccakTables};
    use agg_witness::gen::SplitMix64;
    use agg_witness::params::TR_BYTES;
    use jolt_field::{One, Ring};

    fn manifest() -> CommitmentManifest {
        let manifest = CommitmentManifest::for_batch_fp(8).unwrap();
        assert_eq!(manifest.d_sig(), 3);
        manifest
    }

    fn mle(table: &[E], point: &[E]) -> E {
        let eq = eq_table(point);
        assert_eq!(eq.len(), table.len());
        table.iter().zip(&eq).fold(E::zero(), |acc, (&a, &b)| acc + a * b)
    }

    fn dot(a: &[E], b: &[E]) -> E {
        a.iter().zip(b).fold(E::zero(), |acc, (&x, &y)| acc + x * y)
    }

    /// The table of a claimed lane, from its definition: the value at every cell `(z, s, i)`
    /// read from the committed tables (not from the weights' source cells).
    fn reindexed(layout: &KeccakLayout, id: LaneId, n_star: usize, x: &dyn Fn(Column, usize, usize, usize) -> E) -> Vec<E> {
        let a = |x: u8, y: u8, q: u8| Column::KeccakA { x, y, q };
        let out_slot = (layout.rounds - 1) % SLOTS;
        let prev = |col: Column, z: usize, s: usize, i: usize| {
            let Column::KeccakA { x: cx, y: cy, q } = col else { unreachable!() };
            let r = rot(usize::from(cx), usize::from(cy)) as usize;
            let src_z = (z + LANE_BITS - r) % LANE_BITS;
            if s >= 1 {
                x(col, src_z, s - 1, i)
            } else if q >= 1 {
                x(a(cx, cy, q - 1), src_z, SLOTS - 1, i)
            } else {
                E::zero()
            }
        };
        let xor_bit = |v: E, bit: bool| if bit { E::one() - v } else { v };
        let mut table = vec![E::zero(); CELLS * n_star];
        for i in 0..n_star {
            for s in 0..SLOTS {
                for z in 0..LANE_BITS {
                    table[z + LANE_BITS * s + CELLS * i] = match id {
                        LaneId::Column(c) => x(c, z, s, i),
                        LaneId::KeccakRot(c @ Column::KeccakC { .. }) => x(c, (z + LANE_BITS - 1) % LANE_BITS, s, i),
                        LaneId::KeccakRot(c @ Column::KeccakA { x: 0, y: 0, .. }) => {
                            xor_bit(x(c, z, s, i), s == out_slot && (RC[ROUNDS_PER_PERM - 1] >> z) & 1 == 1)
                        }
                        LaneId::KeccakRot(c @ Column::KeccakA { x: cx, y: cy, .. }) => {
                            let r = rot(usize::from(cx), usize::from(cy)) as usize;
                            x(c, (z + LANE_BITS - r) % LANE_BITS, s, i)
                        }
                        LaneId::KeccakPrev(c) => prev(c, z, s, i),
                        LaneId::KeccakPrevIota(q) => {
                            let t = layout.time(usize::from(q), s);
                            let rc = t >= 1 && t < layout.rounds && (RC[(t - 1) % ROUNDS_PER_PERM] >> z) & 1 == 1;
                            xor_bit(prev(a(0, 0, q), z, s, i), rc)
                        }
                        other => panic!("{other:?} has no committed re-indexing"),
                    };
                }
            }
        }
        table
    }

    fn committed_columns(layout: &KeccakLayout) -> Vec<Column> {
        let mut columns = crate::keccak_fp::masks::a_prime_columns(layout);
        columns.extend(crate::keccak_fp::masks::c_columns(layout));
        columns.extend(crate::keccak_fp::masks::c_prime_columns(layout));
        columns
    }

    #[test]
    fn weights_cover_every_claim_once_per_read() {
        let manifest = manifest();
        let plan = KeccakReductions::for_manifest(&manifest, 7);
        let layout = KeccakLayout::for_set(manifest.set());
        assert_eq!(plan.claims().len(), 740 + 516);
        assert_eq!(plan.gammas(), 7..7 + 1256);
        let at = |p: KeccakPoint| plan.weights().iter().filter(|w| w.point == p).count();
        // Stage 1 reads every committed polynomial; stage 2 the 35 of each of the six boundary
        // slot groups and the A' lanes of the slot groups before them (ten distinct q).
        assert_eq!((at(KeccakPoint::R1), at(KeccakPoint::R2)), (385, 310));
        // Every claim that reads committed cells has as many reads as `reads_of` lists; every
        // block claim of a lane with a w1 bit has wired cells.
        let mut reads = vec![0_usize; plan.claims().len()];
        for w in plan.weights() {
            for read in &w.reads {
                reads[read.claim] += 1;
                assert_eq!(plan.claims()[read.claim].0, w.point);
            }
        }
        let mut wired = vec![0_usize; plan.claims().len()];
        for w in plan.absorbs() {
            for cell in &w.cells {
                wired[cell.claim] += 1;
            }
        }
        for (k, &(_, id)) in plan.claims().iter().enumerate() {
            assert_eq!(reads[k], reads_of(&layout, id).len(), "{id:?}");
            if let LaneId::KeccakBlk { .. } = id {
                assert_eq!(reads[k], 0);
            } else {
                assert!(reads[k] >= 1 && wired[k] == 0, "{id:?}");
            }
        }
        // Every bit of every record's w1Encode is wired exactly once.
        let total: usize = plan.absorbs().iter().map(|w| w.cells.len()).sum();
        assert_eq!(total, 6 * 256 * 4);
        assert_eq!(plan.absorbs().len(), 2 * 6);
        // The specs validate with the folded coefficients and fail without them.
        let mut spec = crate::stage::StageSpec {
            stage: 3,
            degree: 3,
            gamma_count: 7 + plan.claims().len(),
            groups: plan.groups(),
            folded: plan.gammas(),
        };
        spec.groups.push(GroupSpec {
            domain: GroupRole::Main,
            eq: EqPoint::R2,
            lanes: vec![LaneId::Column(Column::W1Bit(0, 0))],
            instances: (0..7).map(|g| InstanceSpec::new(g, vec![Term::Product(vec![0])])).collect(),
        });
        spec.validate().unwrap();
        spec.folded = 7..7;
        assert_eq!(spec.validate().unwrap_err(), "gamma index 7 unused");
        spec.folded = 7..7 + plan.claims().len() + 1;
        assert!(spec.validate().is_err());
        // The sets with polynomial lanes wire every bit once, inside a lane's record cells.
        for set in [agg_witness::params::Set::S65, agg_witness::params::Set::S87] {
            let manifest = CommitmentManifest::for_set_fp(set, 8, crate::manifest::Relation::HExact).unwrap();
            let plan = KeccakReductions::for_manifest(&manifest, 0);
            let shape = manifest.shape();
            let cells: Vec<&AbsorbCell> = plan.absorbs().iter().flat_map(|w| &w.cells).collect();
            assert_eq!(cells.len(), shape.k * 256 * shape.w1_bits);
            assert!(cells.iter().all(|c| (c.cell as usize) < manifest.lane_cells()));
            // The first block holds only the first 144 coefficients (row 0); the later blocks
            // reach every bit of every lane.
            let at = |p: KeccakPoint| plan.absorbs().iter().filter(|w| w.point == p).count();
            assert_eq!((at(KeccakPoint::R1), at(KeccakPoint::R2)), (shape.w1_bits, shape.w1_bits * shape.lanes_k), "{set:?}");
        }
    }

    #[test]
    fn weighted_sums_equal_the_combined_reindexed_claims() {
        let manifest = manifest();
        let layout = KeccakLayout::for_set(manifest.set());
        let n_star = manifest.padded_records();
        let plan = KeccakReductions::for_manifest(&manifest, 0);
        let mut rng = SplitMix64::new(0x5EED_0003);
        let tables: HashMap<Column, Vec<E>> =
            committed_columns(&layout).into_iter().map(|c| (c, point(&mut rng, CELLS * n_star))).collect();
        let cell = |c: Column, z: usize, s: usize, i: usize| tables[&c][z + LANE_BITS * s + CELLS * i];
        let d_stage = manifest.d_stage();
        let (r1, r2) = (point(&mut rng, d_stage), point(&mut rng, d_stage));
        let coeffs = point(&mut rng, plan.claims().len());
        let points = [suffix(&r1, manifest.d_keccak()), suffix(&r2, manifest.d_keccak())];
        // Direct claims of every lane that reads committed cells (blocks are covered with the
        // honest tables, which hold the bits).
        let direct: Vec<E> = plan
            .claims()
            .par_iter()
            .map(|&(p, id)| match id {
                LaneId::KeccakBlk { .. } => E::zero(),
                _ => mle(&reindexed(&layout, id, n_star, &cell), &points[point_index(p)]),
            })
            .collect();
        let r1_len = plan.claims().iter().filter(|(p, _)| *p == KeccakPoint::R1).count();
        let adjusted = plan.claimed_sums(&direct[..r1_len], &direct[r1_len..], &r1, &r2);
        let expected = plan
            .claims()
            .iter()
            .zip(coeffs.iter().zip(&adjusted))
            .filter(|((_, id), _)| !matches!(id, LaneId::KeccakBlk { .. }))
            .fold(E::zero(), |acc, (_, (&c, &v))| acc + c * v);
        let weights = plan.tables(&coeffs, &r1, &r2);
        let got = plan
            .weights()
            .par_iter()
            .map(|w| {
                let weight = weights.rank_one(WeightId::KeccakReduce { poly: w.poly, point: w.point }).unwrap();
                dot(&weight.materialize(), &tables[&w.poly])
            })
            .reduce(E::zero, |a, b| a + b);
        assert_eq!(got, expected);
    }

    #[test]
    fn verifier_values_match_the_materialized_weights() {
        let manifest = manifest();
        let plan = KeccakReductions::for_manifest(&manifest, 0);
        let mut rng = SplitMix64::new(0x5EED_0004);
        let d_stage = manifest.d_stage();
        let (r1, r2, r3) = (point(&mut rng, d_stage), point(&mut rng, d_stage), point(&mut rng, d_stage));
        let coeffs = point(&mut rng, plan.claims().len());
        let tables = plan.tables(&coeffs, &r1, &r2);
        let values = plan.evaluate(&coeffs, &r1, &r2, &r3);
        let r3_k = suffix(&r3, manifest.d_keccak());
        let r3_main = suffix(&r3, manifest.d_main());
        for w in plan.weights() {
            let id = WeightId::KeccakReduce { poly: w.poly, point: w.point };
            assert_eq!(values.get(id), mle(&tables.rank_one(id).unwrap().materialize(), &r3_k), "{id:?}");
        }
        for w in plan.absorbs() {
            let id = WeightId::KeccakAbsorb { t: w.t, lane: w.lane, point: w.point };
            let weight = tables.rank_one(id).unwrap();
            assert_eq!(weight.low.len(), manifest.lane_cells());
            assert_eq!(values.get(id), mle(&weight.materialize(), &r3_main), "{id:?}");
        }
        assert!(tables.rank_one(WeightId::Proj(0)).is_none());
    }

    #[test]
    fn honest_tables_satisfy_the_reductions_and_a_changed_bit_breaks_the_absorption() {
        let manifest = manifest();
        let layout = KeccakLayout::for_set(manifest.set());
        let n_star = manifest.padded_records();
        let shape = manifest.shape();
        let mut rng = SplitMix64::new(0x5EED_0005);
        let records: Vec<([u8; TR_BYTES], Vec<u8>)> = (0..n_star)
            .map(|_| {
                let mu: [u8; TR_BYTES] = std::array::from_fn(|_| rng.next_u64() as u8);
                (mu, (0..layout.w1_encode_bytes).map(|_| rng.next_u64() as u8).collect())
            })
            .collect();
        let witness = KeccakTables::build(&layout, &records, n_star);
        let publics = KeccakPublics::from_records(&layout, &records, n_star);
        let keccak = KeccakProverTables::build(&witness, &publics);
        let plan = KeccakReductions::for_manifest(&manifest, 0);
        let d_stage = manifest.d_stage();
        let (r1, r2) = (point(&mut rng, d_stage), point(&mut rng, d_stage));
        let coeffs = point(&mut rng, plan.claims().len());
        let points = [suffix(&r1, manifest.d_keccak()), suffix(&r2, manifest.d_keccak())];
        let lane_table = |id: LaneId| -> Vec<E> {
            let lane = keccak.lane(id).expect("a Keccak lane");
            (0..CELLS * n_star).map(|x| lane.value(x)).collect()
        };
        let direct: Vec<E> =
            plan.claims().par_iter().map(|&(p, id)| mle(&lane_table(id), &points[point_index(p)])).collect();
        let r1_len = plan.claims().iter().filter(|(p, _)| *p == KeccakPoint::R1).count();
        let adjusted = plan.claimed_sums(&direct[..r1_len], &direct[r1_len..], &r1, &r2);
        let is_blk = |id: LaneId| matches!(id, LaneId::KeccakBlk { .. });
        let combined = |blocks: bool| {
            plan.claims()
                .iter()
                .zip(coeffs.iter().zip(&adjusted))
                .filter(|((_, id), _)| is_blk(*id) == blocks)
                .fold(E::zero(), |acc, (_, (&c, &v))| acc + c * v)
        };
        let weights = plan.tables(&coeffs, &r1, &r2);
        // The committed reads, cross-checked against the prover's own virtual lanes.
        let reduce = plan.weights().iter().fold(E::zero(), |acc, w| {
            let weight = weights.rank_one(WeightId::KeccakReduce { poly: w.poly, point: w.point }).unwrap();
            let column = keccak.column_slice(w.poly).iter().map(|&b| E::from_i64(i64::from(b))).collect::<Vec<_>>();
            acc + dot(&weight.materialize(), &column)
        });
        assert_eq!(reduce, combined(false));
        // The block claims against the committed bits on the main domain:
        // b_t(j + 256 (rho mod lanes) + lane_cells i) in polynomial lane rho / lanes.
        let lane_cells = manifest.lane_cells();
        let mut bits = vec![vec![vec![E::zero(); lane_cells * n_star]; shape.lanes_k]; shape.w1_bits];
        for (i, record) in publics.records().iter().enumerate() {
            for rho in 0..shape.k {
                let (poly_lane, base) = shape.row_lane(rho);
                for j in 0..256 {
                    for (t, bit) in bits.iter_mut().enumerate() {
                        let index = (rho * 256 + j) * shape.w1_bits + t;
                        if (record.w1_bytes[index / 8] >> (index % 8)) & 1 == 1 {
                            bit[poly_lane][base + j + lane_cells * i] = E::one();
                        }
                    }
                }
            }
        }
        let absorbed = |bits: &[Vec<Vec<E>>]| {
            plan.absorbs().iter().fold(E::zero(), |acc, w| {
                let weight = weights.rank_one(WeightId::KeccakAbsorb { t: w.t, lane: w.lane, point: w.point }).unwrap();
                acc + dot(&weight.materialize(), &bits[usize::from(w.t)][usize::from(w.lane)])
            })
        };
        assert_eq!(absorbed(&bits), combined(true));
        // A changed bit in the last block (read at r^(2)) and one in the first (read at r^(1)).
        for (t, j, rho, record) in [(5, 255, 3, 6), (0, 0, 0, 1)] {
            let mut changed = bits.clone();
            let (poly_lane, base) = shape.row_lane(rho);
            let cell = &mut changed[t][poly_lane][base + j + lane_cells * record];
            *cell = E::one() - *cell;
            assert_ne!(absorbed(&changed), combined(true));
        }
        assert!(plan.absorbs().iter().all(|w| w.cells.iter().all(|c| c.cell < lane_cells as u32)));
    }
}

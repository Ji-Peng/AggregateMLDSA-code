//! The level-1 weight `w_1(x) = c_1[x mod 2^z] eq_blocks[x >> z]` kept as its two factors.
//!
//! Stage 2's first group sums `w_1 * sum_t gamma_t b_t` over the main domain. The sum
//! factors as `sum_lo c_1[lo] D[lo]` with `D[lo] = sum_hi eq_blocks[hi] sum_t gamma_t b_t[lo,
//! hi]`, and folding the low `z` variables of the weight and of `D` commutes with the
//! block sum, so the first `z` rounds run on two `2^z` tables instead of on the `2^n`
//! weight and the bit lanes. The block variables follow on `2^(n - z)` cells: the bit lanes
//! folded by the low challenges (one `eq`-weighted pass) and the weight, a scalar times
//! `eq_blocks`. Round polynomials are unique, so the messages equal the generic kernel's.
//!
//! Exactness: the bit lanes are 0/1 cells, so every cell contributes one table entry (a
//! field element) to `D` and at most one field element per lane to the folded lanes; all
//! sums are reduced field additions, with no unreduced accumulator.

use rayon::prelude::*;

use crate::kernel::{Acc, Group, GroupInstance, Lane, Source, Term, IDENTITY_MAP};
use crate::OpeningField as E;
use jolt_field::{One, Ring, Unreduced, Zero};

/// The factors of the level-1 weight table.
#[derive(Clone, Debug)]
pub struct SplitWeight {
    /// `c_1` over the used in-block positions (at most `2^z` entries; zero beyond).
    pub c: Vec<E>,
    /// In-block variables `z`.
    pub zeta_vars: usize,
    /// The block weights `eq(p1_beta, .)`.
    pub blocks: Vec<E>,
    /// Cells of the main domain.
    pub len: usize,
}

impl SplitWeight {
    /// The materialized table (the generic kernel's lane).
    #[must_use]
    pub fn materialize(&self) -> Vec<E> {
        agg_bridge::weights::spread(&self.c, self.zeta_vars, &self.blocks, self.len)
    }

    /// True when the split path applies: whole blocks, a power-of-two block count and more
    /// than one block.
    #[must_use]
    pub fn usable(&self) -> bool {
        self.zeta_vars > 0
            && self.len == self.blocks.len() << self.zeta_vars
            && self.blocks.len() > 1
            && self.blocks.len().is_power_of_two()
            && self.c.len() <= 1 << self.zeta_vars
    }
}

/// Cells per task of the partial pass over the bit lanes (in-block positions). Each task
/// keeps the 384 stream positions of its window live in L2 instead of one line per stream.
const CHUNK: usize = 1 << 11;

/// Cells per subset-table tile of the partial pass. The 64 tables of one tile are 256 KiB
/// and stay in L2 while every stream reads its 512 contiguous bytes.
const TILE: usize = 512;

/// Cells per tile of the `D` pass, where the 64 block terms of one `lo` accumulate
/// unreduced in a 16 KiB array (the 64 `Acc` values are `u128` quadruples).
const D_TILE: usize = 256;

/// The `eq(r, .)` table over `2^r.len()` indices (bit `j` of the index selects `r[j]`).
///
/// The table is built as the product of a table over the low twelve variables and one over
/// the remaining variables: only the two small tables are filled level by level, and the
/// full table (range: all `2^r.len()` entries) is written once, in parallel, with one
/// multiplication per entry.
fn eq_table(r: &[E]) -> Vec<E> {
    const LOW_VARS: usize = 12;

    fn levels(challenges: &[E]) -> Vec<E> {
        let mut table = vec![E::one()];
        for &challenge in challenges {
            let len = table.len();
            let complement = E::one() - challenge;
            let mut next = vec![E::zero(); 2 * len];
            for (i, &value) in table.iter().enumerate() {
                next[i] = value * complement;
                next[len + i] = value * challenge;
            }
            table = next;
        }
        table
    }

    let split = r.len().min(LOW_VARS);
    let low = levels(&r[..split]);
    let high = levels(&r[split..]);
    let mask = low.len() - 1;
    (0..1_usize << r.len())
        .into_par_iter()
        .map(|index| low[index & mask] * high[index >> split])
        .collect()
}

/// The bit pattern of at most eight 0/1 cells (bit `j` from cell `j`). With the
/// identity map the pattern comes from one multiply instead of eight table loads.
#[inline(always)]
fn bit_pattern(cells: &[i8], map: &[i64; 256], identity: bool) -> usize {
    if identity {
        let mut packed = 0_u64;
        for (j, &cell) in cells.iter().enumerate() {
            packed |= u64::from(cell as u8 & 1) << (8 * j);
        }
        (packed.wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize
    } else {
        let mut pattern = 0_usize;
        for (j, &cell) in cells.iter().enumerate() {
            pattern |= ((map[cell as u8 as usize] as usize) & 1) << j;
        }
        pattern
    }
}

/// The pattern of a full group of eight 0/1 cells read as one little-endian word. The
/// multiply gathers bit `j` of byte `j` into bit `j` of the result; the lower terms sum to
/// less than `2^56`, so none carries into the top byte.
#[inline(always)]
fn word_pattern(word: u64) -> usize {
    ((word & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize
}

/// The eight cells as a word. The cells are 0/1 (checked when the group was built).
#[inline(always)]
fn cell_word(cells: &[i8]) -> u64 {
    debug_assert_eq!(cells.len(), 8);
    let mut bytes = [0_u8; 8];
    for (byte, &cell) in bytes.iter_mut().zip(cells) {
        *byte = cell as u8;
    }
    u64::from_le_bytes(bytes)
}

/// `acc += value` for a canonical extension-field element: each coordinate is below `p`, so
/// it is added to the accumulator's low slot and no carry can reach the high slot.
#[inline(always)]
fn lift_canonical(acc: &mut Acc, value: E) {
    acc.0[0] += u128::from(value.c0().to_canonical_u64());
    acc.0[2] += u128::from(value.c1().to_canonical_u64());
}

/// The bit lanes folded by the low challenges: column `t` holds `sum_lo eq[lo] b_t[lo, hi]`
/// for every block `hi`, where `eq = eq(r_low, .)` over the `2^z` in-block positions.
fn fold_low_bits(bits: &[(&[i8], &[i64; 256])], eq: &[E], z: usize, blocks: usize) -> Vec<Vec<E>> {
    let lanes_n = bits.len();
    let rows = 1_usize << z;
    let chunks = rows.div_ceil(CHUNK);
    let groups_per_tile = rows.min(TILE).div_ceil(8);
    let identity = bits.iter().all(|(_, map)| std::ptr::eq(*map, &IDENTITY_MAP));
    // partial[chunk][hi * lanes_n + t] = sum over the chunk's lo of eq[lo] b_t[lo, hi].
    // Eight `lo` share one 256-entry subset-sum table of their eq values, and every
    // block and lane reads that table with its own 8-bit pattern: one field addition per
    // eight cells instead of one per cell, with no branch on the witness bits. A task
    // covers `CHUNK` cells, processed as `TILE`-cell tiles: the tile's 64 tables are
    // built once, then each of the 384 streams reads its 512 contiguous bytes line by
    // line, so no stream line is evicted before its eight patterns are used.
    let partial: Vec<Vec<E>> = (0..chunks)
        .into_par_iter()
        .map_init(
            // One 256 KiB table buffer per worker (64 groups of 256 entries), kept
            // across the chunks the worker takes.
            || vec![E::zero(); groups_per_tile * 256],
            |subset, chunk| {
                let lo0 = chunk * CHUNK;
                let lo1 = (lo0 + CHUNK).min(rows);
                let mut acc = vec![E::zero(); blocks * lanes_n];
                // The subset tables of one tile: `groups_per_tile` groups of eight `eq`
                // values, 256 entries each. Every group rewrites the entries below its
                // pattern count on each tile, and reads stay below that count, so only
                // entry 0 needs an explicit zero.
                let mut tile = lo0;
                while tile < lo1 {
                    let tile_n = (lo1 - tile).min(TILE);
                    let groups = tile_n.div_ceil(8);
                    for g in 0..groups {
                        let lo = tile + 8 * g;
                        let n = (tile_n - 8 * g).min(8);
                        let patterns = 1_usize << n;
                        let base = g * 256;
                        subset[base] = E::zero();
                        for p in 1..patterns {
                            let low = p & (p - 1);
                            subset[base + p] =
                                subset[base + low] + eq[lo + (p ^ low).trailing_zeros() as usize];
                        }
                    }
                    for hi in 0..blocks {
                        let base = (hi << z) + tile;
                        for (t, (cells, map)) in bits.iter().enumerate() {
                            let source = &cells[base..base + tile_n];
                            let slot = &mut acc[hi * lanes_n + t];
                            let full = tile_n / 8 * 8;
                            for (g, cells) in source[..full].chunks_exact(8).enumerate() {
                                let p = if identity {
                                    word_pattern(cell_word(cells))
                                } else {
                                    bit_pattern(cells, map, false)
                                };
                                *slot += subset[g * 256 + p];
                            }
                            let tail = &source[full..];
                            if !tail.is_empty() {
                                let p = bit_pattern(tail, map, identity);
                                let g = full / 8;
                                *slot += subset[g * 256 + p];
                            }
                        }
                    }
                    tile += tile_n;
                }
                acc
            },
        )
        .collect();
    (0..lanes_n)
        .map(|t| (0..blocks).map(|hi| partial.iter().fold(E::zero(), |acc, p| acc + p[hi * lanes_n + t])).collect())
        .collect()
}

/// `w_1 * sum_t gamma_t b_t` of stage 2, as a [`Group`] pair (prover only).
pub struct SplitGroup<'a> {
    /// The rounds over the in-block variables.
    low: Group<'a>,
    /// The rounds over the block variables, built after the low rounds.
    high: Option<Group<'a>>,
    bits: Vec<(&'a [i8], &'a [i64; 256])>,
    instances: Vec<GroupInstance>,
    blocks: Vec<E>,
    zeta_vars: usize,
    degree: usize,
    challenges: Vec<E>,
    arena: Option<crate::kernel::SharedArena>,
}

impl<'a> SplitGroup<'a> {
    /// The group for the weight `split` and the bit lanes `lanes` (lane `l` of the group is
    /// `lanes[l - 1]`); `instances` must be products of the weight (lane 0) and one bit
    /// lane. `None` when the shape is not that, a lane is not a 0/1 byte table of the main
    /// domain, or the split does not apply.
    #[must_use]
    pub fn new(
        split: &SplitWeight,
        lanes: &[Lane<'a>],
        instances: &[GroupInstance],
        degree: usize,
    ) -> Option<Self> {
        if !split.usable() || lanes.is_empty() || lanes.len() > 8 {
            return None;
        }
        let z = split.zeta_vars;
        let mut coeff = vec![E::zero(); lanes.len()];
        for instance in instances {
            match instance.terms.as_slice() {
                [Term::Product(read)] if read.len() == 2 && read[0] == 0 && read[1] >= 1 && read[1] <= lanes.len() => {
                    coeff[read[1] - 1] += instance.gamma;
                }
                _ => return None,
            }
        }
        let bits: Vec<(&'a [i8], &'a [i64; 256])> = lanes
            .iter()
            .map(|lane| match lane {
                Lane::Source(Source::Bytes { cells, map }) if cells.len() == split.len => Some((*cells, *map)),
                _ => None,
            })
            .collect::<Option<_>>()?;
        // D[lo] = sum_hi eq_blocks[hi] * (sum over the set lanes of gamma), one table entry
        // per cell: entry[hi][p] = eq_blocks[hi] * sum_{t in p} coeff[t].
        let patterns = 1_usize << bits.len();
        let mut subset = vec![E::zero(); patterns];
        for p in 1..patterns {
            let low = p & (p - 1);
            subset[p] = subset[low] + coeff[(p ^ low).trailing_zeros() as usize];
        }
        let tables: Vec<Vec<E>> = split
            .blocks
            .iter()
            .map(|&b| subset.iter().map(|&s| b * s).collect())
            .collect();
        let rows = 1_usize << z;
        let invalid = std::sync::atomic::AtomicBool::new(false);
        let identity = bits.iter().all(|(_, map)| std::ptr::eq(*map, &IDENTITY_MAP));
        // Each cell contributes one canonical table entry per block, so a coordinate of the
        // unreduced sum is below `blocks * p < 2^64 * blocks`; workloads have
        // `blocks <= 2^6`, and the low `u128` slot stays exact for any `blocks < 2^64`.
        debug_assert!(tables.len() < 1 << 32);
        // Range: all `rows` slots (8 MiB at z = 19). Every slot is overwritten by the walk
        // below; the initial fill only makes the buffer safe to write in place, and is one
        // pass.
        let mut d = vec![E::zero(); rows];
        d.par_chunks_mut(D_TILE).enumerate().for_each(|(chunk, out)| {
            let lo0 = chunk * D_TILE;
            let n = out.len();
            let mut acc = vec![Acc::zero(); n];
            let mut bad = 0_u64;
            // One 1 KiB slice of `D` stays in L1 while the 64 block terms land in `acc`;
            // the reduced pass is one add per cell instead of 64.
            for (hi, table) in tables.iter().enumerate() {
                let base = (hi << z) + lo0;
                let mut pattern = [0_u8; D_TILE];
                for (t, (cells, map)) in bits.iter().enumerate() {
                    let source = &cells[base..base + n];
                    if identity {
                        for (slot, &cell) in pattern[..n].iter_mut().zip(source) {
                            bad |= u64::from(cell as u8 > 1);
                            *slot |= (cell as u8 & 1) << t;
                        }
                    } else {
                        for (slot, &cell) in pattern[..n].iter_mut().zip(source) {
                            let v = map[cell as u8 as usize];
                            bad |= (v as u64 > 1) as u64;
                            *slot |= ((v as u8) & 1) << t;
                        }
                    }
                }
                for (slot, &p) in acc.iter_mut().zip(&pattern[..n]) {
                    lift_canonical(slot, table[p as usize]);
                }
            }
            for (slot, value) in out.iter_mut().zip(acc) {
                *slot = E::reduce_product(value);
            }
            if bad != 0 {
                invalid.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        });
        if invalid.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let mut c = split.c.clone();
        c.resize(rows, E::zero());
        let one = GroupInstance {
            gamma: E::one(),
            terms: vec![Term::Product(vec![0, 1])],
        };
        let low = Group::new(vec![Lane::Folded(c), Lane::Folded(d)], vec![one], None, z, degree).ok()?;
        Some(Self {
            low,
            high: None,
            bits,
            instances: instances.to_vec(),
            blocks: split.blocks.clone(),
            zeta_vars: z,
            degree,
            challenges: Vec::new(),
            arena: None,
        })
    }

    pub(crate) fn set_arena(&mut self, arena: &crate::kernel::SharedArena) {
        self.low.set_arena(arena);
        self.arena = Some(arena.clone());
    }

    /// The current claim (the high group's, once the low rounds are bound).
    #[must_use]
    pub fn claim(&self) -> E {
        self.high.as_ref().map_or_else(|| self.low.claim(), Group::claim)
    }

    /// The compressed message of `round`.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        if round < self.zeta_vars {
            self.low.round_message(round)
        } else {
            self.high.as_mut().expect("low rounds bound").round_message(round - self.zeta_vars)
        }
    }

    /// Bind the round's variable to `challenge`.
    pub fn bind(&mut self, round: usize, challenge: E) {
        self.challenges.push(challenge);
        if round < self.zeta_vars {
            self.low.bind(round, challenge);
            if round + 1 == self.zeta_vars {
                self.build_high();
            }
        } else {
            self.high.as_mut().expect("low rounds bound").bind(round - self.zeta_vars, challenge);
        }
    }

    /// The lane values at the bound point: the weight, then the bit lanes.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        self.high.as_ref().expect("all rounds bound").terminal_values()
    }

    /// The block rounds: the bit lanes folded by the low challenges and the weight
    /// `c_1(r_low) eq_blocks`.
    fn build_high(&mut self) {
        let z = self.zeta_vars;
        let c_at = self.low.terminal_values()[0];
        // eq(r_low, lo), coordinate j on index bit j (the low variable is bound first).
        let eq = eq_table(&self.challenges[..z]);
        let lanes_n = self.bits.len();
        let blocks = self.blocks.len();
        let columns = fold_low_bits(&self.bits, &eq, z, blocks);
        let mut lanes: Vec<Lane<'a>> = Vec::with_capacity(lanes_n + 1);
        lanes.push(Lane::Folded(self.blocks.iter().map(|&b| c_at * b).collect()));
        lanes.extend(columns.into_iter().map(Lane::Folded));
        let mut high = Group::new(lanes, self.instances.clone(), None, blocks.trailing_zeros() as usize, self.degree)
            .expect("block rounds of a valid split");
        if let Some(arena) = &self.arena {
            high.set_arena(arena);
        }
        self.high = Some(high);
    }
}

/// The level-1 weights of the sets with polynomial lanes, `w_1^(rho)(x) = c_rho[x mod 2^z]
/// eq_blocks[x >> z]`, kept as their factors: all lanes share the block factor, and the
/// in-block factor `c_rho` is `c_1` read at the bridge input positions of lane `rho`
/// (`weights::level1_lane_split`).
#[derive(Clone, Debug)]
pub struct LaneSplitWeight {
    /// `c_rho` per lane over the in-block positions of a lane (`2^z` entries or fewer).
    pub c: Vec<Vec<E>>,
    /// In-block variables `z` of a lane.
    pub zeta_vars: usize,
    /// The block weights `eq(p1_beta, .)`.
    pub blocks: Vec<E>,
    /// Cells of the main domain.
    pub len: usize,
}

impl LaneSplitWeight {
    /// The materialized table of lane `lane` (the generic kernel's lane).
    #[must_use]
    pub fn materialize(&self, lane: usize) -> Vec<E> {
        agg_bridge::weights::spread(&self.c[lane], self.zeta_vars, &self.blocks, self.len)
    }

    /// True when the split path applies: whole blocks, a power-of-two block count and more
    /// than one block.
    #[must_use]
    pub fn usable(&self) -> bool {
        self.zeta_vars > 0
            && self.len == self.blocks.len() << self.zeta_vars
            && self.blocks.len() > 1
            && self.blocks.len().is_power_of_two()
            && self.c.iter().all(|c| c.len() <= 1 << self.zeta_vars)
    }
}

/// The level-1 weight of a stage-2 run as its factors: one table (ML-DSA-44) or one per
/// polynomial lane of the bits.
#[derive(Clone, Copy, Debug)]
pub enum Level1Split<'s> {
    /// `w_1 = c_1 eq_blocks` ([`SplitGroup`]).
    Table(&'s SplitWeight),
    /// `w_1^(rho) = c_rho eq_blocks` ([`LaneSplitGroup`]).
    Lanes(&'s LaneSplitWeight),
}

/// Bit lanes of one weight lane at most (one pattern byte per cell).
const LANE_BITS_MAX: usize = 8;

/// `sum_t gamma_t sum_rho w_1^(rho) b_t^(rho)` of stage 2 for the sets with polynomial lanes,
/// as a [`Group`] pair (prover only). With `B_rho = sum_t gamma_t b_t^(rho)` the sum factors as
/// `sum_lo sum_rho c_rho[lo] D_rho[lo]` with `D_rho[lo] = sum_hi eq_blocks[hi] B_rho[lo, hi]`,
/// so the first `z` rounds run on the `2 k` tables `c_rho, D_rho` of `2^z` entries; the block
/// rounds run on the weights `c_rho(r_low) eq_blocks` and the bit lanes folded by the low
/// challenges, with the original instances. Round polynomials are unique, so the messages
/// equal the generic kernel's on the materialized weights.
pub struct LaneSplitGroup<'a> {
    /// The rounds over the in-block variables.
    low: Group<'a>,
    /// The rounds over the block variables, built after the low rounds.
    high: Option<Group<'a>>,
    /// Weight lanes `k` (group lanes `0..k`).
    weights: usize,
    /// The bit lanes (group lanes `k..`).
    bits: Vec<(&'a [i8], &'a [i64; 256])>,
    instances: Vec<GroupInstance>,
    blocks: Vec<E>,
    zeta_vars: usize,
    degree: usize,
    challenges: Vec<E>,
    arena: Option<crate::kernel::SharedArena>,
}

impl<'a> LaneSplitGroup<'a> {
    /// The group for the weights `split` and the bit lanes `lanes` (group lane `k + l` is
    /// `lanes[l]`, the weights are group lanes `0..k`); `instances` must be sums of products
    /// of one weight and one bit lane, each bit lane read with one weight only, at most
    /// eight bit lanes per weight. `None` when the shape is not that, a lane is not a 0/1
    /// byte table of the main domain, or the split does not apply.
    #[must_use]
    pub fn new(
        split: &LaneSplitWeight,
        lanes: &[Lane<'a>],
        instances: &[GroupInstance],
        degree: usize,
    ) -> Option<Self> {
        let k = split.c.len();
        if !split.usable() || k == 0 || lanes.is_empty() {
            return None;
        }
        let z = split.zeta_vars;
        // The weight of every bit lane and its coefficient `sum gamma` over the instances.
        let mut owner: Vec<Option<usize>> = vec![None; lanes.len()];
        let mut coeff = vec![E::zero(); lanes.len()];
        for instance in instances {
            for term in &instance.terms {
                let Term::Product(read) = term else { return None };
                let &[w, b] = read.as_slice() else { return None };
                if w >= k || b < k || b - k >= lanes.len() {
                    return None;
                }
                let bit = b - k;
                if *owner[bit].get_or_insert(w) != w {
                    return None;
                }
                coeff[bit] += instance.gamma;
            }
        }
        let bits: Vec<(&'a [i8], &'a [i64; 256])> = lanes
            .iter()
            .map(|lane| match lane {
                Lane::Source(Source::Bytes { cells, map }) if cells.len() == split.len => Some((*cells, *map)),
                _ => None,
            })
            .collect::<Option<_>>()?;
        // The bit lanes of each weight, in lane order.
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (bit, w) in owner.iter().enumerate() {
            if let Some(w) = w {
                members[*w].push(bit);
            }
        }
        if members.iter().any(|m| m.len() > LANE_BITS_MAX) {
            return None;
        }
        // tables[w][hi][p] = eq_blocks[hi] * sum_{bits in p} coeff, the pattern bit `j` from
        // the `j`-th bit lane of weight `w`.
        let tables: Vec<Vec<Vec<E>>> = members
            .iter()
            .map(|m| {
                let patterns = 1_usize << m.len();
                let mut subset = vec![E::zero(); patterns];
                for p in 1..patterns {
                    let low = p & (p - 1);
                    subset[p] = subset[low] + coeff[m[(p ^ low).trailing_zeros() as usize]];
                }
                split.blocks.iter().map(|&b| subset.iter().map(|&x| b * x).collect()).collect()
            })
            .collect();
        let rows = 1_usize << z;
        let invalid = std::sync::atomic::AtomicBool::new(false);
        // Each cell contributes one canonical table entry per block to `D_rho`, so a
        // coordinate of the unreduced sum is below `blocks * p`; the low `u128` slot stays
        // exact for any `blocks < 2^64`.
        let d: Vec<Vec<E>> = (0..k)
            .map(|w| {
                let lanes_w: Vec<(&[i8], &[i64; 256])> = members[w].iter().map(|&b| bits[b]).collect();
                let tables_w = &tables[w];
                // Range: `2^z` slots (1 MiB at z = 16), each overwritten below.
                let mut out = vec![E::zero(); rows];
                out.par_chunks_mut(D_TILE).enumerate().for_each(|(chunk, out)| {
                    let lo0 = chunk * D_TILE;
                    let n = out.len();
                    let mut acc = [Acc::zero(); D_TILE];
                    let mut bad = 0_u64;
                    for (hi, table) in tables_w.iter().enumerate() {
                        let base = (hi << z) + lo0;
                        let mut pattern = [0_u8; D_TILE];
                        for (t, (cells, map)) in lanes_w.iter().enumerate() {
                            for (slot, &cell) in pattern[..n].iter_mut().zip(&cells[base..base + n]) {
                                let v = map[cell as u8 as usize];
                                bad |= u64::from(v as u64 > 1);
                                *slot |= ((v as u8) & 1) << t;
                            }
                        }
                        for (slot, &p) in acc[..n].iter_mut().zip(&pattern[..n]) {
                            lift_canonical(slot, table[p as usize]);
                        }
                    }
                    for (slot, value) in out.iter_mut().zip(acc) {
                        *slot = E::reduce_product(value);
                    }
                    if bad != 0 {
                        invalid.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                out
            })
            .collect();
        if invalid.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let mut low_lanes = Vec::with_capacity(2 * k);
        for (c, d) in split.c.iter().zip(d) {
            let mut c = c.clone();
            c.resize(rows, E::zero());
            low_lanes.push(Lane::Folded(c));
            low_lanes.push(Lane::Folded(d));
        }
        let one = GroupInstance {
            gamma: E::one(),
            terms: (0..k).map(|w| Term::Product(vec![2 * w, 2 * w + 1])).collect(),
        };
        let low = Group::new(low_lanes, vec![one], None, z, degree).ok()?;
        Some(Self {
            low,
            high: None,
            weights: k,
            bits,
            instances: instances.to_vec(),
            blocks: split.blocks.clone(),
            zeta_vars: z,
            degree,
            challenges: Vec::new(),
            arena: None,
        })
    }

    pub(crate) fn set_arena(&mut self, arena: &crate::kernel::SharedArena) {
        self.low.set_arena(arena);
        self.arena = Some(arena.clone());
    }

    /// The current claim (the high group's, once the low rounds are bound).
    #[must_use]
    pub fn claim(&self) -> E {
        self.high.as_ref().map_or_else(|| self.low.claim(), Group::claim)
    }

    /// The compressed message of `round`.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        if round < self.zeta_vars {
            self.low.round_message(round)
        } else {
            self.high.as_mut().expect("low rounds bound").round_message(round - self.zeta_vars)
        }
    }

    /// Bind the round's variable to `challenge`.
    pub fn bind(&mut self, round: usize, challenge: E) {
        self.challenges.push(challenge);
        if round < self.zeta_vars {
            self.low.bind(round, challenge);
            if round + 1 == self.zeta_vars {
                self.build_high();
            }
        } else {
            self.high.as_mut().expect("low rounds bound").bind(round - self.zeta_vars, challenge);
        }
    }

    /// The lane values at the bound point: the weights, then the bit lanes.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        self.high.as_ref().expect("all rounds bound").terminal_values()
    }

    /// The block rounds: the weights `c_rho(r_low) eq_blocks` and the bit lanes folded by
    /// the low challenges.
    fn build_high(&mut self) {
        let z = self.zeta_vars;
        let low = self.low.terminal_values();
        let eq = eq_table(&self.challenges[..z]);
        let blocks = self.blocks.len();
        let columns = fold_low_bits(&self.bits, &eq, z, blocks);
        let mut lanes: Vec<Lane<'a>> = Vec::with_capacity(self.weights + self.bits.len());
        for w in 0..self.weights {
            let c_at = low[2 * w];
            lanes.push(Lane::Folded(self.blocks.iter().map(|&b| c_at * b).collect()));
        }
        lanes.extend(columns.into_iter().map(Lane::Folded));
        let mut high = Group::new(lanes, self.instances.clone(), None, blocks.trailing_zeros() as usize, self.degree)
            .expect("block rounds of a valid split");
        if let Some(arena) = &self.arena {
            high.set_arena(arena);
        }
        self.high = Some(high);
    }
}

/// A main-domain weight kept as its two factors, `w(x) = low[x mod 2^z] high[x >> z]` (the
/// projection weights `W^(rho)(j, i) = P_rho(j) eq(r'_sig, i)` of the sets with polynomial
/// lanes, `z = 8`).
#[derive(Clone, Copy, Debug)]
pub struct RankOneWeight<'s> {
    /// The factor over the low `z` variables.
    pub low: &'s [E],
    /// The factor over the remaining variables.
    pub high: &'s [E],
}

impl RankOneWeight<'_> {
    /// The materialized table (the generic kernel's lane).
    #[must_use]
    pub fn materialize(&self) -> Vec<E> {
        let z = self.low.len().trailing_zeros() as usize;
        agg_bridge::weights::spread(self.low, z, self.high, self.low.len() * self.high.len())
    }
}

/// Rows of the high factor per task of the two passes of [`RankOneGroup`].
const RANK_ONE_CHUNK: usize = 1 << 9;

/// The integer value `sum_l c_l f_l(x) + c` of the affine lanes at cell `x`.
#[inline]
fn affine_at(lanes: &[(Source<'_>, i64)], constant: i64, x: usize) -> i64 {
    lanes.iter().fold(constant, |acc, (lane, c)| {
        acc + c * match lane {
            Source::Ints(cells) => i64::from(cells[x]),
            Source::Bytes { cells, map } => map[cells[x] as u8 as usize],
            _ => unreachable!("integer lanes only"),
        }
    })
}

/// `gamma sum_x w(x) (sum_l c_l f_l(x) + c)` for a weight `w` kept as its factors
/// ([`RankOneWeight`]) and integer or byte lanes `f_l`, without an equality factor (prover
/// only). With `g = sum_l c_l f_l + c` the sum is `sum_lo low[lo] D[lo]`,
/// `D[lo] = sum_hi high[hi] g(lo, hi)`, so the first `z` rounds run on the two `2^z` tables
/// `low, D`; the remaining rounds run on the weight `low(r_low) high` and the lanes folded by
/// the low challenges, with the original instance. Round polynomials are unique, so the
/// messages equal the generic kernel's on the materialized weight.
pub struct RankOneGroup<'a> {
    /// The rounds over the low variables.
    low: Group<'a>,
    /// The rounds over the high variables, built after the low rounds.
    high: Option<Group<'a>>,
    /// The group lane of the weight.
    weight_lane: usize,
    /// The other lanes in group order.
    lanes: Vec<Source<'a>>,
    instance: GroupInstance,
    high_factor: Vec<E>,
    zeta_vars: usize,
    degree: usize,
    challenges: Vec<E>,
    arena: Option<crate::kernel::SharedArena>,
}

impl<'a> RankOneGroup<'a> {
    /// The group for the weight `weight` at group lane `weight_lane`, the group's `lanes`
    /// (the weight's entry is ignored) and one instance with one affine term weighted by
    /// that lane. `None` when the shape is not that, a lane is not an integer or byte table
    /// of `2^rounds` cells, a factor is not a power of two of at least two entries, or the
    /// integer combination may leave `i64` (coefficients above `2^20`, a constant above
    /// `2^40`).
    #[must_use]
    pub fn new(
        weight: RankOneWeight<'_>,
        weight_lane: usize,
        lanes: &[Lane<'a>],
        instances: &[GroupInstance],
        rounds: usize,
        degree: usize,
    ) -> Option<Self> {
        let [instance] = instances else { return None };
        let [Term::Affine { weight: Some(w), lanes: affine, constant }] = instance.terms.as_slice() else {
            return None;
        };
        let (z, h) = (weight.low.len(), weight.high.len());
        if *w != weight_lane
            || z < 2
            || h < 2
            || !z.is_power_of_two()
            || !h.is_power_of_two()
            || z * h != 1 << rounds
            || affine.len() > 1 << 10
            || constant.unsigned_abs() > 1 << 40
        {
            return None;
        }
        let rows = 1_usize << rounds;
        let mut others = Vec::with_capacity(lanes.len());
        for (l, lane) in lanes.iter().enumerate() {
            if l == weight_lane {
                continue;
            }
            match lane {
                Lane::Source(source @ (Source::Ints(_) | Source::Bytes { .. })) if source.len() == rows => {
                    others.push(*source);
                }
                _ => return None,
            }
        }
        // Coefficients by lane; the weight lane carries none.
        let mut combination: Vec<(Source<'a>, i64)> = Vec::new();
        for &(l, c) in affine {
            if l == weight_lane || l >= lanes.len() || c.unsigned_abs() > 1 << 20 {
                return None;
            }
            let Lane::Source(source) = &lanes[l] else { return None };
            if matches!(source, Source::Bytes { map, .. } if map.iter().any(|v| v.unsigned_abs() > 1 << 31)) {
                return None;
            }
            combination.push((*source, c));
        }
        // D[lo] = sum_hi high[hi] g(lo, hi): with at most 2^10 terms, coefficients of at most
        // 2^20, cells of at most 2^31 (an `i32`, or a byte-map entry checked above) and a
        // constant of at most 2^40, |g| <= 2^61 + 2^40 < 2^62; one reduced product per cell.
        let d = (0..h)
            .into_par_iter()
            .step_by(RANK_ONE_CHUNK)
            .map(|first| {
                let mut part = vec![E::zero(); z];
                for hi in first..(first + RANK_ONE_CHUNK).min(h) {
                    let factor = weight.high[hi];
                    for (lo, slot) in part.iter_mut().enumerate() {
                        let g = affine_at(&combination, *constant, (hi * z) | lo);
                        if g != 0 {
                            *slot += factor.mul_i64(g);
                        }
                    }
                }
                part
            })
            .reduce(
                || vec![E::zero(); z],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    a
                },
            );
        let low = Group::new(
            vec![Lane::Folded(weight.low.to_vec()), Lane::Folded(d)],
            vec![GroupInstance { gamma: instance.gamma, terms: vec![Term::Product(vec![0, 1])] }],
            None,
            z.trailing_zeros() as usize,
            degree,
        )
        .ok()?;
        Some(Self {
            low,
            high: None,
            weight_lane,
            lanes: others,
            instance: instance.clone(),
            high_factor: weight.high.to_vec(),
            zeta_vars: z.trailing_zeros() as usize,
            degree,
            challenges: Vec::new(),
            arena: None,
        })
    }

    pub(crate) fn set_arena(&mut self, arena: &crate::kernel::SharedArena) {
        self.low.set_arena(arena);
        self.arena = Some(arena.clone());
    }

    /// The current claim (the high group's, once the low rounds are bound).
    #[must_use]
    pub fn claim(&self) -> E {
        self.high.as_ref().map_or_else(|| self.low.claim(), Group::claim)
    }

    /// The compressed message of `round`.
    pub fn round_message(&mut self, round: usize) -> Vec<E> {
        if round < self.zeta_vars {
            self.low.round_message(round)
        } else {
            self.high.as_mut().expect("low rounds bound").round_message(round - self.zeta_vars)
        }
    }

    /// Bind the round's variable to `challenge`.
    pub fn bind(&mut self, round: usize, challenge: E) {
        self.challenges.push(challenge);
        if round < self.zeta_vars {
            self.low.bind(round, challenge);
            if round + 1 == self.zeta_vars {
                self.build_high();
            }
        } else {
            self.high.as_mut().expect("low rounds bound").bind(round - self.zeta_vars, challenge);
        }
    }

    /// The lane values at the bound point, in group lane order.
    #[must_use]
    pub fn terminal_values(&self) -> Vec<E> {
        self.high.as_ref().expect("all rounds bound").terminal_values()
    }

    /// The high rounds: the weight `low(r_low) high` and every other lane folded by the low
    /// challenges, `F_l[hi] = sum_lo eq(r_low, lo) f_l(lo, hi)`.
    fn build_high(&mut self) {
        let z = 1_usize << self.zeta_vars;
        let c_at = self.low.terminal_values()[0];
        let eq = eq_table(&self.challenges[..self.zeta_vars]);
        let h = self.high_factor.len();
        let folded: Vec<Vec<E>> = self
            .lanes
            .iter()
            .map(|lane| {
                (0..h)
                    .into_par_iter()
                    .map(|hi| {
                        let single = [(*lane, 1_i64)];
                        eq.iter().enumerate().fold(E::zero(), |acc, (lo, &e)| {
                            let f = affine_at(&single, 0, (hi * z) | lo);
                            if f == 0 { acc } else { acc + e.mul_i64(f) }
                        })
                    })
                    .collect()
            })
            .collect();
        let mut folded = folded.into_iter();
        let lanes: Vec<Lane<'a>> = (0..=self.lanes.len())
            .map(|l| {
                if l == self.weight_lane {
                    Lane::Folded(self.high_factor.iter().map(|&b| c_at * b).collect())
                } else {
                    Lane::Folded(folded.next().expect("one folded table per lane"))
                }
            })
            .collect();
        let mut high = Group::new(lanes, vec![self.instance.clone()], None, h.trailing_zeros() as usize, self.degree)
            .expect("high rounds of a valid rank-one group");
        if let Some(arena) = &self.arena {
            high.set_arena(arena);
        }
        self.high = Some(high);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::IDENTITY_MAP;
    use crate::test_support::field;
    use agg_witness::gen::SplitMix64;

    /// The split rounds give the generic kernel's messages, claim and terminal values.
    #[test]
    fn split_rounds_match_the_generic_group() {
        let mut rng = SplitMix64::new(77);
        for (z, h, used) in [
            (4_usize, 3_usize, 16_usize),
            (5, 1, 11),
            (3, 4, 8),
            (2, 3, 4),
            // The production shape crosses many chunks and tiles of both split passes.
            (13, 6, 64),
        ] {
            let len = 1_usize << (z + h);
            let split = SplitWeight {
                c: (0..used).map(|_| field(&mut rng)).collect(),
                zeta_vars: z,
                blocks: (0..1_usize << h).map(|_| field(&mut rng)).collect(),
                len,
            };
            let cells: Vec<Vec<i8>> =
                (0..6).map(|_| (0..len).map(|_| (rng.next_u64() & 1) as i8).collect()).collect();
            let lanes: Vec<Lane<'_>> = cells.iter().map(|c| Lane::column(c)).collect();
            let instances: Vec<GroupInstance> = (0..6)
                .map(|t| GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![0, 1 + t])] })
                .collect();
            let degree = 4;
            let mut want_lanes = vec![Lane::Folded(split.materialize())];
            want_lanes.extend(lanes.iter().cloned());
            let mut want = Group::new(want_lanes, instances.clone(), None, z + h, degree).unwrap();
            let mut got = SplitGroup::new(&split, &lanes, &instances, degree).expect("applies");
            assert_eq!(got.claim(), want.claim());
            for round in 0..z + h {
                let (a, b) = (got.round_message(round), want.round_message(round));
                assert_eq!(a, b, "message z={z} h={h} round {round}");
                let r = field(&mut rng);
                got.bind(round, r);
                want.bind(round, r);
            }
            assert_eq!(got.claim(), want.claim());
            assert_eq!(got.terminal_values(), want.terminal_values());
        }
    }

    /// The lane split rounds give the generic kernel's messages, claim and terminal values
    /// on the materialized weights (bit lanes read bit-major, as stage 2 lays them out).
    #[test]
    fn lane_split_rounds_match_the_generic_group() {
        let mut rng = SplitMix64::new(87);
        for (z, h, used, k) in [(4_usize, 3_usize, 16_usize, 3_usize), (5, 1, 11, 2), (3, 2, 8, 6), (9, 3, 300, 8)] {
            let len = 1_usize << (z + h);
            let split = LaneSplitWeight {
                c: (0..k).map(|_| (0..used).map(|_| field(&mut rng)).collect()).collect(),
                zeta_vars: z,
                blocks: (0..1_usize << h).map(|_| field(&mut rng)).collect(),
                len,
            };
            let cells: Vec<Vec<i8>> =
                (0..4 * k).map(|_| (0..len).map(|_| (rng.next_u64() & 1) as i8).collect()).collect();
            let lanes: Vec<Lane<'_>> = cells.iter().map(|c| Lane::column(c)).collect();
            let instances: Vec<GroupInstance> = (0..4)
                .map(|t| GroupInstance {
                    gamma: field(&mut rng),
                    terms: (0..k).map(|rho| Term::Product(vec![rho, k + t * k + rho])).collect(),
                })
                .collect();
            let degree = 5;
            let mut want_lanes: Vec<Lane<'_>> = (0..k).map(|rho| Lane::Folded(split.materialize(rho))).collect();
            want_lanes.extend(lanes.iter().cloned());
            let mut want = Group::new(want_lanes, instances.clone(), None, z + h, degree).unwrap();
            let mut got = LaneSplitGroup::new(&split, &lanes, &instances, degree).expect("applies");
            assert_eq!(got.claim(), want.claim());
            for round in 0..z + h {
                let (a, b) = (got.round_message(round), want.round_message(round));
                assert_eq!(a, b, "message z={z} h={h} k={k} round {round}");
                let r = field(&mut rng);
                got.bind(round, r);
                want.bind(round, r);
            }
            assert_eq!(got.claim(), want.claim());
            assert_eq!(got.terminal_values(), want.terminal_values());
        }
    }

    /// The rank-one rounds give the generic kernel's messages, claim and terminal values on
    /// the materialized weight.
    #[test]
    fn rank_one_rounds_match_the_generic_group() {
        let mut rng = SplitMix64::new(97);
        for (z, h) in [(3_usize, 2_usize), (8, 3), (2, 5)] {
            let len = 1_usize << (z + h);
            let low: Vec<E> = (0..1_usize << z).map(|_| field(&mut rng)).collect();
            let high: Vec<E> = (0..1_usize << h).map(|_| field(&mut rng)).collect();
            let weight = RankOneWeight { low: &low, high: &high };
            let v_lo: Vec<i32> = (0..len).map(|_| (rng.next_u64() % (1 << 16)) as i32 - (1 << 15)).collect();
            let v_hi: Vec<i8> = (0..len).map(|_| (rng.next_u64() % 9) as i8 - 4).collect();
            let lanes = vec![
                Lane::Folded(Vec::new()),
                Lane::Source(Source::Ints(&v_lo)),
                Lane::Source(Source::Bytes { cells: &v_hi, map: &IDENTITY_MAP }),
            ];
            let instances = vec![GroupInstance {
                gamma: field(&mut rng),
                terms: vec![Term::Affine { weight: Some(0), lanes: vec![(1, 1), (2, 1 << 16)], constant: 1 << 15 }],
            }];
            let degree = 3;
            let mut want_lanes = lanes.clone();
            want_lanes[0] = Lane::Folded(weight.materialize());
            let mut want = Group::new(want_lanes, instances.clone(), None, z + h, degree).unwrap();
            let mut got = RankOneGroup::new(weight, 0, &lanes, &instances, z + h, degree).expect("applies");
            assert_eq!(got.claim(), want.claim());
            for round in 0..z + h {
                assert_eq!(got.round_message(round), want.round_message(round), "z={z} h={h} round {round}");
                let r = field(&mut rng);
                got.bind(round, r);
                want.bind(round, r);
            }
            assert_eq!(got.claim(), want.claim());
            assert_eq!(got.terminal_values(), want.terminal_values());
        }
    }

    /// A non-0/1 cell sends the lane set to the generic kernel.
    #[test]
    fn non_boolean_cells_are_refused() {
        let mut rng = SplitMix64::new(78);
        let (z, h) = (3_usize, 2_usize);
        let len = 1_usize << (z + h);
        let split = SplitWeight {
            c: (0..8).map(|_| field(&mut rng)).collect(),
            zeta_vars: z,
            blocks: (0..4).map(|_| field(&mut rng)).collect(),
            len,
        };
        let mut cells = vec![0_i8; len];
        cells[5] = 2;
        let lanes = [Lane::Source(Source::Bytes { cells: &cells, map: &IDENTITY_MAP })];
        let instances =
            [GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![0, 1])] }];
        assert!(SplitGroup::new(&split, &lanes, &instances, 4).is_none());
    }

    /// The packed identity pattern and the generic map path agree.
    #[test]
    fn bit_pattern_matches_the_map() {
        let mut rng = SplitMix64::new(79);
        let mut square = IDENTITY_MAP;
        for (v, slot) in square.iter_mut().enumerate() {
            *slot = ((v * v) & 1) as i64;
        }
        for n in 1..=8 {
            let cells: Vec<i8> = (0..n).map(|_| (rng.next_u64() & 1) as i8).collect();
            let identity = bit_pattern(&cells, &IDENTITY_MAP, true);
            let generic = bit_pattern(&cells, &IDENTITY_MAP, false);
            let want = cells.iter().enumerate().fold(0, |p, (j, &c)| p | ((c as usize) << j));
            assert_eq!(identity, want);
            assert_eq!(generic, want);
            assert_eq!(bit_pattern(&cells, &square, false), want);
        }
    }

    /// Single-thread micro-benchmark of the two hot split-weight passes at a 32K-like
    /// shape: `z = 19`, 64 blocks, six bit lanes and `2^25` cells. Run with
    /// `RAYON_NUM_THREADS=1 cargo test -p agg-piop --release --lib split_weight_bench --
    /// --ignored --nocapture` under `taskset`.
    #[test]
    #[ignore = "micro-benchmark"]
    fn split_weight_bench() {
        let mut rng = SplitMix64::new(2026);
        let (z, h, lanes_n) = (19_usize, 6_usize, 6_usize);
        let blocks = 1_usize << h;
        let len = blocks << z;
        let split = SplitWeight {
            c: (0..1_usize << z).map(|_| field(&mut rng)).collect(),
            zeta_vars: z,
            blocks: (0..blocks).map(|_| field(&mut rng)).collect(),
            len,
        };
        let cells: Vec<Vec<i8>> =
            (0..lanes_n).map(|_| (0..len).map(|_| (rng.next_u64() & 1) as i8).collect()).collect();
        let lanes: Vec<Lane<'_>> = cells.iter().map(|c| Lane::column(c)).collect();
        let instances: Vec<GroupInstance> = (0..lanes_n)
            .map(|t| GroupInstance { gamma: field(&mut rng), terms: vec![Term::Product(vec![0, 1 + t])] })
            .collect();

        let start = std::time::Instant::now();
        let mut group = SplitGroup::new(&split, &lanes, &instances, 4).expect("the shape applies");
        let new_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = std::time::Instant::now();
        for round in 0..z - 1 {
            let _ = group.round_message(round);
            group.bind(round, field(&mut rng));
        }
        let low_ms = start.elapsed().as_secs_f64() * 1e3;

        let start = std::time::Instant::now();
        let _ = group.round_message(z - 1);
        group.bind(z - 1, field(&mut rng));
        let build_ms = start.elapsed().as_secs_f64() * 1e3;

        println!(
            "split-weight bench z={z} blocks={blocks} lanes={lanes_n} cells={len} \
             new_ms={new_ms:.3} low_ms={low_ms:.3} last_message_plus_build_ms={build_ms:.3}"
        );
        std::hint::black_box(group.claim());
    }
}

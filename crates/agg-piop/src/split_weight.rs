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

use crate::kernel::{Group, GroupInstance, Lane, Source, Term};
use crate::OpeningField as E;
use jolt_field::{One, Zero};

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

    /// True when the split path applies: whole blocks and more than one block.
    #[must_use]
    pub fn usable(&self) -> bool {
        self.zeta_vars > 0
            && self.len == self.blocks.len() << self.zeta_vars
            && self.blocks.len() > 1
            && self.c.len() <= 1 << self.zeta_vars
    }
}

/// Cells per task of the passes over the bit lanes (in-block positions).
const CHUNK: usize = 1 << 13;

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
        let mut d = vec![E::zero(); rows];
        d.par_chunks_mut(CHUNK).enumerate().for_each(|(chunk, out)| {
            let lo0 = chunk * CHUNK;
            let mut bad = 0_u64;
            for (hi, table) in tables.iter().enumerate() {
                let base = (hi << z) + lo0;
                for (i, slot) in out.iter_mut().enumerate() {
                    let mut p = 0_usize;
                    for (t, (cells, map)) in bits.iter().enumerate() {
                        let v = map[cells[base + i] as u8 as usize];
                        bad |= (v as u64 > 1) as u64;
                        p |= ((v as usize) & 1) << t;
                    }
                    *slot += table[p];
                }
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
        let mut eq = vec![E::one()];
        for &r in &self.challenges[..z] {
            let len = eq.len();
            let mut next = vec![E::zero(); 2 * len];
            let (lo, hi) = next.split_at_mut(len);
            for ((l, h), &e) in lo.iter_mut().zip(hi.iter_mut()).zip(&eq) {
                let scaled = e * r;
                *h = scaled;
                *l = e - scaled;
            }
            eq = next;
        }
        let lanes_n = self.bits.len();
        let blocks = self.blocks.len();
        let rows = 1_usize << z;
        let chunks = rows.div_ceil(CHUNK);
        // partial[hi * chunks + chunk][t] = sum over the chunk's lo of eq[lo] b_t[lo, hi].
        let partial: Vec<Vec<E>> = (0..blocks * chunks)
            .into_par_iter()
            .map(|job| {
                let (hi, chunk) = (job / chunks, job % chunks);
                let lo0 = chunk * CHUNK;
                let lo1 = (lo0 + CHUNK).min(rows);
                let mut acc = vec![E::zero(); lanes_n];
                for lo in lo0..lo1 {
                    for (t, (cells, map)) in self.bits.iter().enumerate() {
                        // The cells are 0/1 (checked when the group was built).
                        if map[cells[(hi << z) + lo] as u8 as usize] == 1 {
                            acc[t] += eq[lo];
                        }
                    }
                }
                acc
            })
            .collect();
        let mut lanes: Vec<Lane<'a>> = Vec::with_capacity(lanes_n + 1);
        lanes.push(Lane::Folded(self.blocks.iter().map(|&b| c_at * b).collect()));
        for t in 0..lanes_n {
            let column: Vec<E> = (0..blocks)
                .map(|hi| {
                    partial[hi * chunks..(hi + 1) * chunks]
                        .iter()
                        .fold(E::zero(), |acc, p| acc + p[t])
                })
                .collect();
            lanes.push(Lane::Folded(column));
        }
        let mut high = Group::new(lanes, self.instances.clone(), None, blocks.trailing_zeros() as usize, self.degree)
            .expect("block rounds of a valid split");
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
        for (z, h, used) in [(4_usize, 3_usize, 16_usize), (5, 1, 11), (3, 4, 8)] {
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
}

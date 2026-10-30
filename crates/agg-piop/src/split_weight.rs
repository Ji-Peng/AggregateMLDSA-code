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
use jolt_field::{One, Unreduced, Zero};

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
    #[allow(unsafe_code)]
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
        // pass, not the serial concat of the old port.
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
        let rows = 1_usize << z;
        let chunks = rows.div_ceil(CHUNK);
        let groups_per_tile = rows.min(TILE).div_ceil(8);
        let identity = self.bits.iter().all(|(_, map)| std::ptr::eq(*map, &IDENTITY_MAP));
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
                            for (t, (cells, map)) in self.bits.iter().enumerate() {
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
        let mut lanes: Vec<Lane<'a>> = Vec::with_capacity(lanes_n + 1);
        lanes.push(Lane::Folded(self.blocks.iter().map(|&b| c_at * b).collect()));
        for t in 0..lanes_n {
            let column: Vec<E> = (0..blocks)
                .map(|hi| partial.iter().fold(E::zero(), |acc, p| acc + p[hi * lanes_n + t]))
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
        for (z, h, used) in [(4_usize, 3_usize, 16_usize), (5, 1, 11), (3, 4, 8), (2, 3, 4)] {
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

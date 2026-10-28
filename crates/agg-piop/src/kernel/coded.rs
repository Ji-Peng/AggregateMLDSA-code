//! Coded columns: an `i8` column whose cells fall into at most 16 classes under the maps
//! of the lanes reading it, the pair codes of its cells (one byte per pair), the pattern
//! spaces of pairs and quartets, and the pattern-mass accumulation of the compact rounds
//! (Aerie's `CodedSource`, `chunked_bucket_sums`; `docs/piop.md`).

use jolt_field::Zero;
use rayon::prelude::*;

use super::field::fm;
use crate::OpeningField as E;

/// Largest class count of a coded column (one nibble per cell, one byte per pair).
pub(super) const MAX_CLASSES: usize = 16;

/// A coded column.
#[derive(Clone, Debug)]
pub(super) struct Column {
    /// Number of classes `n`.
    pub classes: usize,
    /// One representative cell per class.
    pub reps: Vec<i8>,
    /// `class(cell(2k)) * n + class(cell(2k + 1))` for every pair `k`.
    pub pairs: Vec<u8>,
}

impl Column {
    /// Pair patterns `n^2`.
    pub fn pair_space(&self) -> usize {
        self.classes * self.classes
    }

    /// The two classes of a pair pattern.
    pub fn split(&self, pattern: usize) -> (usize, usize) {
        (pattern / self.classes, pattern % self.classes)
    }
}

/// The set of byte values present in `cells`.
fn present_bytes(cells: &[i8]) -> [bool; 256] {
    let scan = |chunk: &[i8]| {
        let mut seen = [false; 256];
        for &c in chunk {
            seen[c as u8 as usize] = true;
        }
        seen
    };
    let merge = |mut a: [bool; 256], b: [bool; 256]| {
        for (x, y) in a.iter_mut().zip(b) {
            *x |= y;
        }
        a
    };
    if cells.len() >= 1 << 16 {
        cells.par_chunks(1 << 16).map(scan).reduce(|| [false; 256], merge)
    } else {
        scan(cells)
    }
}

/// Code a column under the given maps: classes are the distinct tuples of map values of
/// the present cells (in increasing cell order). `None` when more than 16 classes.
pub(super) fn code_column(cells: &[i8], maps: &[&[i64; 256]]) -> Option<Column> {
    if cells.len() < 2 {
        return None;
    }
    let seen = present_bytes(cells);
    let mut keys: Vec<Vec<i64>> = Vec::new();
    let mut reps: Vec<i8> = Vec::new();
    let mut class_of = [0_u8; 256];
    for value in -128_i16..=127 {
        let byte = value as i8 as u8 as usize;
        if !seen[byte] {
            continue;
        }
        let key: Vec<i64> = maps.iter().map(|map| map[byte]).collect();
        let class = match keys.iter().position(|k| *k == key) {
            Some(class) => class,
            None => {
                if keys.len() == MAX_CLASSES {
                    return None;
                }
                keys.push(key);
                reps.push(value as i8);
                keys.len() - 1
            }
        };
        class_of[byte] = class as u8;
    }
    let classes = keys.len();
    let encode = |pair: &[i8]| {
        (class_of[pair[0] as u8 as usize] as usize * classes + class_of[pair[1] as u8 as usize] as usize) as u8
    };
    let pairs = if cells.len() >= 1 << 16 {
        cells.par_chunks_exact(2).map(encode).collect()
    } else {
        cells.chunks_exact(2).map(encode).collect()
    };
    Some(Column {
        classes,
        reps,
        pairs,
    })
}

/// The columns of one bucketed segment (one column, or several for a joint term) and
/// the mixed-radix strides of their pair patterns.
#[derive(Clone, Debug)]
pub(super) struct SegmentShape {
    pub columns: Vec<usize>,
    /// `stride[i] = prod_{j > i} pair_space(columns[j])`.
    pub strides: Vec<usize>,
    /// Joint pair space.
    pub pair_space: usize,
}

impl SegmentShape {
    pub fn new(columns: Vec<usize>, all: &[Column]) -> Self {
        let mut strides = vec![1; columns.len()];
        let mut space = 1;
        for (i, &c) in columns.iter().enumerate().rev() {
            strides[i] = space;
            space *= all[c].pair_space();
        }
        Self {
            columns,
            strides,
            pair_space: space,
        }
    }

    /// Joint pair pattern of level-0 pair `k`.
    #[cfg(test)]
    pub fn pair_pattern(&self, all: &[Column], k: usize) -> usize {
        let mut pattern = 0;
        for (&c, &stride) in self.columns.iter().zip(&self.strides) {
            pattern += all[c].pairs[k] as usize * stride;
        }
        pattern
    }

    /// The pair pattern of `column` (its position in the segment) within a joint pattern.
    #[inline(always)]
    pub fn component(&self, all: &[Column], pattern: usize, position: usize) -> usize {
        (pattern / self.strides[position]) % all[self.columns[position]].pair_space()
    }
}

/// How the masses of the compact rounds are weighted per item.
pub(super) enum ItemWeights<'w> {
    /// `first[i & mask] second[i >> bits]` (a split equality table).
    Split(&'w [E], &'w [E]),
    /// Weight one (groups without an equality factor).
    One,
}

impl ItemWeights<'_> {
    #[inline(always)]
    fn at(&self, item: usize) -> E {
        match self {
            Self::Split(first, second) => {
                let bits = first.len().trailing_zeros();
                fm(first[item & (first.len() - 1)], second[item >> bits])
            }
            Self::One => jolt_field::One::one(),
        }
    }
}

/// Above this many table entries per chunk, the masses are accumulated per segment
/// subset (each task owns whole segments over all items) instead of per item range.
const NARROW_ENTRIES: usize = 1 << 16;

/// Pattern masses of the segments: per segment, the sum of the item weights per pattern.
/// `quartets` selects quartet patterns (item `q` = level-0 pairs `2q, 2q + 1`, pattern
/// `left * pair_space + right`) over pair patterns (item = pair). The result is one table
/// per segment. Field addition is exact, so every chunking gives the serial sums.
pub(super) fn masses(
    columns: &[Column],
    segments: &[SegmentShape],
    items: usize,
    quartets: bool,
    weights: &ItemWeights<'_>,
) -> Vec<Vec<E>> {
    let widths: Vec<usize> = segments
        .iter()
        .map(|s| if quartets { s.pair_space * s.pair_space } else { s.pair_space })
        .collect();
    // Per segment, its columns' pair codes and strides (no indirection per item).
    let codes: Vec<Vec<(&[u8], usize)>> = segments
        .iter()
        .map(|s| {
            s.columns
                .iter()
                .zip(&s.strides)
                .map(|(&c, &stride)| (&columns[c].pairs[..], stride))
                .collect()
        })
        .collect();
    #[inline(always)]
    fn joint(codes: &[(&[u8], usize)], k: usize) -> usize {
        match codes {
            [(pairs, _)] => pairs[k] as usize,
            _ => codes.iter().map(|&(pairs, stride)| pairs[k] as usize * stride).sum(),
        }
    }
    let pattern = |s: usize, item: usize| -> usize {
        let codes = &codes[s];
        if quartets {
            joint(codes, 2 * item) * segments[s].pair_space + joint(codes, 2 * item + 1)
        } else {
            joint(codes, item)
        }
    };
    let total: usize = widths.iter().sum();
    let threads = rayon::current_num_threads().max(1);
    if total <= NARROW_ENTRIES || segments.len() < 2 || items < 1 << 14 {
        // One table per chunk of items; at most one chunk per thread, each at least a
        // table's width of items (Aerie's `chunked_bucket_sums`).
        let offsets: Vec<usize> = widths
            .iter()
            .scan(0, |acc, &w| {
                let start = *acc;
                *acc += w;
                Some(start)
            })
            .collect();
        let chunks = threads.min(items / total.max(1)).max(1);
        let per_chunk = items.div_ceil(chunks);
        let accumulate = |chunk: usize| {
            let mut sums = vec![E::zero(); total];
            for item in chunk * per_chunk..((chunk + 1) * per_chunk).min(items) {
                let w = weights.at(item);
                for (s, &offset) in offsets.iter().enumerate() {
                    sums[offset + pattern(s, item)] += w;
                }
            }
            sums
        };
        let mut partials: Vec<Vec<E>> = if chunks > 1 {
            (0..chunks).into_par_iter().map(accumulate).collect()
        } else {
            vec![accumulate(0)]
        };
        let mut sums = partials.swap_remove(0);
        if !partials.is_empty() {
            sums.par_chunks_mut(1 << 12).enumerate().for_each(|(block, target)| {
                let start = block << 12;
                for partial in &partials {
                    for (sum, value) in target.iter_mut().zip(&partial[start..]) {
                        *sum += *value;
                    }
                }
            });
        }
        return offsets
            .iter()
            .zip(&widths)
            .map(|(&o, &w)| sums[o..o + w].to_vec())
            .collect();
    }
    // Wide tables (16-class columns at the quartet level): each task owns a subset of the
    // segments over all items, so no table is replicated or reduced; the item weights are
    // computed once for all tasks.
    let tasks = threads.min(segments.len());
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by_key(|&s| std::cmp::Reverse(widths[s]));
    let mut assignment: Vec<Vec<usize>> = vec![Vec::new(); tasks];
    for (i, &s) in order.iter().enumerate() {
        // Round robin over the size order balances the item work (one update per item
        // and segment) and the table sizes.
        assignment[i % tasks].push(s);
    }
    let item_weights: Vec<E> = match weights {
        ItemWeights::Split(..) => (0..items).into_par_iter().map(|item| weights.at(item)).collect(),
        ItemWeights::One => Vec::new(),
    };
    let results: Vec<Vec<(usize, Vec<E>)>> = assignment
        .into_par_iter()
        .map(|owned| {
            let mut tables: Vec<Vec<E>> = owned.iter().map(|&s| vec![E::zero(); widths[s]]).collect();
            for (table, &s) in tables.iter_mut().zip(&owned) {
                if item_weights.is_empty() {
                    let one = jolt_field::One::one();
                    for item in 0..items {
                        table[pattern(s, item)] += one;
                    }
                } else {
                    for (item, &w) in item_weights.iter().enumerate() {
                        table[pattern(s, item)] += w;
                    }
                }
            }
            owned.into_iter().zip(tables).collect()
        })
        .collect();
    let mut out: Vec<Vec<E>> = vec![Vec::new(); segments.len()];
    for (s, table) in results.into_iter().flatten() {
        out[s] = table;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{digit_map, IDENTITY_MAP};
    use crate::range::{hi_map, lo_map, square_map};
    use crate::test_support::field;
    use agg_witness::gen::SplitMix64;

    #[test]
    fn digit_columns_code_into_eight_classes_under_the_square_maps_and_sixteen_under_the_identity() {
        let cells: Vec<i8> = (0..1 << 12).map(|i| (i % 16) as i8 - 8).collect();
        let square = code_column(&cells, &[square_map()]).unwrap();
        assert_eq!(square.classes, 8);
        let leaves = code_column(&cells, &[lo_map(), hi_map()]).unwrap();
        assert_eq!(leaves.classes, 8);
        assert_eq!(leaves.pair_space(), 64);
        let identity = code_column(&cells, &[&IDENTITY_MAP]).unwrap();
        assert_eq!(identity.classes, 16);
        let mixed = code_column(&cells, &[square_map(), &IDENTITY_MAP]).unwrap();
        assert_eq!(mixed.classes, 16);
        // Classes of the square map identify z with -1 - z.
        for (k, pair) in cells.chunks_exact(2).enumerate() {
            let (a, b) = square.split(square.pairs[k] as usize);
            let p = |z: i8| i64::from(z) * (i64::from(z) + 1);
            assert_eq!(p(square.reps[a]), p(pair[0]));
            assert_eq!(p(square.reps[b]), p(pair[1]));
        }
        let bits: Vec<i8> = (0..64).map(|i| (i % 3 == 0) as i8).collect();
        assert_eq!(code_column(&bits, &[&IDENTITY_MAP]).unwrap().classes, 2);
        // A tampered digit outside [-8, 7] makes 17 classes under the identity: not coded.
        let mut wide = cells.clone();
        wide[5] = 9;
        assert!(code_column(&wide, &[&IDENTITY_MAP]).is_none());
        // ... but 9 (9 + 1) = 90 is a ninth class of the square map.
        assert_eq!(code_column(&wide, &[square_map()]).unwrap().classes, 9);
        let shifted = digit_map(|z| z + 100);
        assert_eq!(code_column(&cells, &[&shifted]).unwrap().classes, 16);
    }

    #[test]
    fn masses_match_the_serial_accumulation_in_both_layouts() {
        let mut rng = SplitMix64::new(91);
        let len = 1 << 16;
        let digit_cells: Vec<Vec<i8>> = (0..5).map(|_| (0..len).map(|_| rng.below(16) as i8 - 8).collect()).collect();
        let bit_cells: Vec<Vec<i8>> = (0..2).map(|_| (0..len).map(|_| rng.below(2) as i8).collect()).collect();
        let mut columns: Vec<Column> = digit_cells.iter().map(|c| code_column(c, &[&IDENTITY_MAP]).unwrap()).collect();
        columns.extend(bit_cells.iter().map(|c| code_column(c, &[&IDENTITY_MAP]).unwrap()));
        let first: Vec<E> = (0..64).map(|_| field(&mut rng)).collect();
        let second: Vec<E> = (0..len / 4 / 64).map(|_| field(&mut rng)).collect();
        for pick in [vec![vec![5], vec![6], vec![5, 6]], (0..7).map(|c| vec![c]).collect()] {
            let segments: Vec<SegmentShape> = pick.iter().map(|cols| SegmentShape::new(cols.clone(), &columns)).collect();
            for quartets in [false, true] {
                let items = if quartets { len / 4 } else { len / 2 };
                let weights = if quartets {
                    ItemWeights::Split(&first, &second)
                } else {
                    ItemWeights::One
                };
                let got = masses(&columns, &segments, items, quartets, &weights);
                for (segment, table) in segments.iter().zip(&got) {
                    let mut expected = vec![E::zero(); table.len()];
                    for item in 0..items {
                        let w = weights.at(item);
                        let p = if quartets {
                            segment.pair_pattern(&columns, 2 * item) * segment.pair_space
                                + segment.pair_pattern(&columns, 2 * item + 1)
                        } else {
                            segment.pair_pattern(&columns, item)
                        };
                        expected[p] += w;
                    }
                    assert_eq!(*table, expected);
                }
            }
        }
    }
}

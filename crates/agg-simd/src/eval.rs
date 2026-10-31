//! Multilinear evaluation of a committed integer column at a point (the opening claims of
//! lanes that are never materialised as `K` tables): `sum_y eq(r, y) f(y)` with the split
//! equality tables `first` (the low variables) and `second`, `f(i + |first| b) = cells[..]`,
//! as inner integer dots over `first` per block times `second[block]`.
use crate::{dot, K};
use jolt_field::{Ring, Zero};
use rayon::prelude::*;

/// Cells of one lane as integers.
pub trait IntColumn: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The cell values `start..start + out.len()` as `i64`.
    fn read(&self, start: usize, out: &mut [i64]);
}

impl IntColumn for &[i32] {
    fn len(&self) -> usize {
        <[i32]>::len(self)
    }
    fn read(&self, start: usize, out: &mut [i64]) {
        for (o, &c) in out.iter_mut().zip(&self[start..]) {
            *o = i64::from(c);
        }
    }
}

/// `i8` cells seen through a per-cell map.
pub struct Mapped<'a> {
    pub cells: &'a [i8],
    pub map: &'a [i64; 256],
}

impl IntColumn for Mapped<'_> {
    fn len(&self) -> usize {
        self.cells.len()
    }
    fn read(&self, start: usize, out: &mut [i64]) {
        for (o, &c) in out.iter_mut().zip(&self.cells[start..]) {
            *o = self.map[c as u8 as usize];
        }
    }
}

/// `sum_b second[b] sum_i first[i] f(i + |first| b)`; `|first| |second|` cells, `|f| < 2^51`.
pub fn column_at<C: IntColumn>(cells: &C, first: &[K], second: &[K]) -> K {
    let f = first.len();
    assert_eq!(cells.len(), f * second.len(), "the tables cover the column");
    let planes = crate::buf::KPlanes::from_fn(f, |i| first[i]);
    let block = |b: usize| -> K {
        let mut ints = vec![0_i64; f];
        cells.read(b * f, &mut ints);
        second[b] * dot::k_int_dot(planes.planes(), &ints)
    };
    if second.len() >= 2 && cells.len() >= 1 << 14 {
        (0..second.len()).into_par_iter().map(block).reduce(K::zero, |a, b| a + b)
    } else {
        (0..second.len()).map(block).fold(K::zero(), |a, b| a + b)
    }
}

/// The split equality tables of `point` (low variables in `first`): `first[i] second[b] =
/// eq(point, i + |first| b)`.
#[must_use]
pub fn split_eq_tables(point: &[K]) -> (Vec<K>, Vec<K>) {
    let split = point.len() / 2;
    let table = |vars: &[K]| -> Vec<K> {
        let mut out = vec![K::from_u64(1)];
        for &r in vars {
            let mut next = Vec::with_capacity(2 * out.len());
            for &w in &out {
                next.push(w * (K::from_u64(1) - r));
            }
            for &w in &out {
                next.push(w * r);
            }
            out = next;
        }
        out
    };
    (table(&point[..split]), table(&point[split..]))
}

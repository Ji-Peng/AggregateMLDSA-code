//! Padding of batches and positions to powers of two.
//!
//! Main-domain index x = j + 256 rho + 1024 i (canonical order (j, rho or s, i), least
//! significant first). For the bridge domains D_1, D_2, D_3, their block/in-block indices and
//! the level maps see `crate::bridge`.

use crate::params::{
    ParamSet, Set44, D_COEF, D_POS, D_SEG_MAX, N_MAX, N_STAR_MIN, PCS_MIN_CELLS_LOG2,
};
use core::marker::PhantomData;

/// Batch geometry for N live records, generic over parameter set P.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry<P: ParamSet = Set44> {
    /// Live records N.
    pub n: usize,
    /// N* = max(next_power_of_two(N), 8).
    pub n_star: usize,
    /// d_sig = log2 N*.
    pub d_sig: u32,
    _marker: PhantomData<P>,
}

/// Geometry errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryError {
    /// N = 0.
    Empty,
    /// N > 2^15 (the verifier MUST reject).
    TooMany(usize),
}

impl Geometry<Set44> {
    /// Geometry for `n` live records using ML-DSA-44 default parameters.
    pub fn new(n: usize) -> Result<Self, GeometryError> {
        Self::new_for(n)
    }
}

impl<P: ParamSet> Geometry<P> {
    /// Geometry for `n` live records under parameter set P, 1 <= n <= 2^15.
    pub fn new_for(n: usize) -> Result<Self, GeometryError> {
        if n == 0 {
            return Err(GeometryError::Empty);
        }
        if n > N_MAX {
            return Err(GeometryError::TooMany(n));
        }
        let n_star = n.next_power_of_two().max(N_STAR_MIN);
        Ok(Self {
            n,
            n_star,
            d_sig: n_star.trailing_zeros(),
            _marker: PhantomData,
        })
    }

    /// live(i) = [i < N].
    pub fn live(&self, i: usize) -> bool {
        i < self.n
    }

    /// d_main = d_pos + d_sig.
    pub fn d_main(&self) -> u32 {
        P::D_POS + self.d_sig
    }

    /// Main-domain size 2^{d_main}.
    pub fn main_len(&self) -> usize {
        1 << self.d_main()
    }

    /// d_seg = min(d_sig, 13).
    pub fn d_seg(&self) -> u32 {
        self.d_sig.min(D_SEG_MAX)
    }

    /// Number of segments S = 2^{max(0, d_sig - 13)}.
    pub fn segments(&self) -> usize {
        1 << self.d_sig.saturating_sub(D_SEG_MAX)
    }

    /// Segment of record i.
    pub fn segment_of(i: usize) -> usize {
        i >> D_SEG_MAX
    }

    /// Live records of segment sigma.
    pub fn live_in_segment(&self, sigma: usize) -> usize {
        self.n
            .saturating_sub(sigma << D_SEG_MAX)
            .min(1 << D_SEG_MAX)
    }

    /// Keccak permutations of segment sigma.
    pub fn keccak_perms_in_segment(&self, sigma: usize) -> usize {
        P::N_PERMS * self.live_in_segment(sigma)
    }

    /// Variables of D_1: 13 + max(0, d_sig - log2(B_1)).
    pub fn d1(&self) -> u32 {
        13 + self.d_sig.saturating_sub(P::BLOCK_RECORDS_LOG2)
    }

    /// Variables of D_2: 13 + max(0, d_sig - 13).
    pub fn d2(&self) -> u32 {
        13 + self.d_sig.saturating_sub(13)
    }

    /// Variables of D_3: 12 + max(0, d_sig - 13).
    pub fn d3(&self) -> u32 {
        12 + self.d_sig.saturating_sub(13)
    }

    /// Number of level-1 blocks (i_mid, sigma): 2^{max(0, d_sig - log2(B_1))}.
    pub fn blocks1(&self) -> usize {
        1 << self.d_sig.saturating_sub(P::BLOCK_RECORDS_LOG2)
    }
}

/// x = j + 256 rho + 1024 i for Set44.
#[inline]
pub fn main_index(j: usize, rho: usize, i: usize) -> usize {
    j | (rho << D_COEF) | (i << D_POS)
}

/// Inverse of `main_index`: (j, rho, i) for Set44.
#[inline]
pub fn split_main(x: usize) -> (usize, usize, usize) {
    (x & 0xFF, (x >> D_COEF) & 3, x >> D_POS)
}

/// Canonical cell index for parameter set P: j | (rho_hi << 8) | (i << (8 + D_VEC)).
#[inline]
pub fn main_index_for<P: ParamSet>(j: usize, rho: usize, i: usize) -> usize {
    j | (rho << D_COEF) | (i << P::D_POS)
}

/// Inverse of `main_index_for`: (j, rho, i) for parameter set P.
#[inline]
pub fn split_main_for<P: ParamSet>(x: usize) -> (usize, usize, usize) {
    let mask_rho = if P::D_VEC == 0 { 0 } else { (1 << P::D_VEC) - 1 };
    (x & 0xFF, (x >> D_COEF) & mask_rho, x >> P::D_POS)
}

/// Physical table size of a table with `log2` variables: zero-extended to at least 2^13 cells.
pub fn physical_len(log2: u32) -> usize {
    1 << log2.max(PCS_MIN_CELLS_LOG2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{Set65, Set87};

    #[test]
    fn geometry_values() {
        assert_eq!(Geometry::new(0), Err(GeometryError::Empty));
        assert_eq!(
            Geometry::new(N_MAX + 1),
            Err(GeometryError::TooMany(N_MAX + 1))
        );
        let g = Geometry::new(1).unwrap();
        assert_eq!((g.n_star, g.d_sig, g.d_main()), (8, 3, 13));
        assert_eq!(
            (g.d1(), g.d2(), g.d3(), g.blocks1(), g.segments()),
            (13, 13, 12, 1, 1)
        );
        let g = Geometry::new(1000).unwrap();
        assert_eq!((g.n_star, g.d_sig, g.d_main()), (1024, 10, 20));
        assert_eq!((g.d1(), g.blocks1()), (14, 2));
        let g = Geometry::new(N_MAX).unwrap();
        assert_eq!(
            (g.d_main(), g.d1(), g.d2(), g.d3(), g.segments()),
            (25, 19, 15, 14, 4)
        );
        assert_eq!(g.keccak_perms_in_segment(3), 57_344);
        let g = Geometry::new(8193).unwrap();
        assert_eq!(
            (g.segments(), g.live_in_segment(0), g.live_in_segment(1)),
            (2, 8192, 1)
        );
        assert_eq!(physical_len(12), 1 << 13);
        assert_eq!(physical_len(20), 1 << 20);
    }

    #[test]
    fn geometry_values_65_and_87() {
        let g65 = Geometry::<Set65>::new_for(1000).unwrap();
        assert_eq!((g65.n_star, g65.d_sig), (1024, 10));
        // d_main = D_POS (8) + d_sig (10) = 18.
        assert_eq!(g65.d_main(), 18);
        // BLOCK_RECORDS_LOG2 = 8, so d1 = 13 + max(0, 10 - 8) = 15.
        assert_eq!(g65.d1(), 15);
        assert_eq!(g65.blocks1(), 4);
        assert_eq!(g65.keccak_perms_in_segment(0), 7 * 1000);

        let g87 = Geometry::<Set87>::new_for(1000).unwrap();
        assert_eq!((g87.n_star, g87.d_sig), (1024, 10));
        assert_eq!(g87.d_main(), 18);
        assert_eq!(g87.d1(), 15);
        assert_eq!(g87.blocks1(), 4);
        // Set87 has 9 permutations.
        assert_eq!(g87.keccak_perms_in_segment(0), 9 * 1000);
    }

    #[test]
    fn main_index_roundtrip() {
        for &(j, r, i) in &[(0, 0, 0), (255, 3, 7), (17, 2, 1023)] {
            assert_eq!(split_main(main_index(j, r, i)), (j, r, i));
            assert_eq!(
                split_main_for::<Set44>(main_index_for::<Set44>(j, r, i)),
                (j, r, i)
            );
        }
        assert_eq!(main_index(1, 1, 1), 1 + 256 + 1024);
    }
}

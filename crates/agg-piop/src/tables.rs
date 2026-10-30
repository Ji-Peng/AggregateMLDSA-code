//! The prover's narrow tables and the lanes the stages read from them.
//!
//! [`ProverTables`] borrows the committed columns of both rounds (`i8` width-4 columns and
//! the `i32` lookup columns, read as `Ints` lanes), materializes the
//! level-1 parities `y^(1)_t` (`i8` when every value fits, as for an honest witness whose
//! parities are bits; `i32` otherwise) and holds the public weight tables once they are
//! known (`w_2, w_{3,t}` after `tau`; `w_1` after `r^(1)`; `w_A` after `r^(2)`). Every
//! [`LaneId`] resolves to a [`Lane`] over the narrow source; the stage drivers lift to `K`
//! only inside the kernel's compact rounds.

use agg_witness::bridge::BridgeMatrices;
use agg_witness::params::W1_BITS;

use crate::kernel::{Lane, Source};
use crate::manifest::Column;
use crate::public_data::PublicData;
use crate::range::square_map;
use crate::source_tables::SourceTables;
use crate::stage::{LaneId, WeightId};
use crate::virtual_polys::parity1_tables;
use crate::split_weight::SplitWeight;
use crate::weights::{level1_weight_split, level1_weight_table, level2_weight_table, level3_weight_tables, ring_weight_table};
use crate::OpeningField as E;

/// The level-1 parities `y^(1)_t` on `D_1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parity1 {
    /// Every value fits an `i8` (the honest parities are bits): coded byte lanes.
    Narrow([Vec<i8>; W1_BITS]),
    /// Some value does not fit an `i8` (a tampered witness): `i32` lanes.
    Wide([Vec<i32>; W1_BITS]),
}

impl Parity1 {
    /// The narrow tables when every value of `tables` fits an `i8`, else the wide ones.
    #[must_use]
    pub fn from_values(tables: [Vec<i64>; W1_BITS]) -> Self {
        if tables
            .iter()
            .all(|table| table.iter().all(|&y| i8::try_from(y).is_ok()))
        {
            Self::Narrow(tables.map(|table| table.into_iter().map(|y| y as i8).collect()))
        } else {
            Self::Wide(tables.map(|table| {
                table
                    .into_iter()
                    .map(|y| i32::try_from(y).expect("parity fits i32"))
                    .collect()
            }))
        }
    }

    /// The lane of `y^(1)_t`.
    #[must_use]
    pub fn lane(&self, t: usize) -> Lane<'_> {
        match self {
            Self::Narrow(tables) => Lane::column(&tables[t]),
            Self::Wide(tables) => Lane::Source(Source::Ints(&tables[t])),
        }
    }

    /// The value of `y^(1)_t` at `index`.
    #[must_use]
    pub fn value(&self, t: usize, index: usize) -> i64 {
        match self {
            Self::Narrow(tables) => i64::from(tables[t][index]),
            Self::Wide(tables) => i64::from(tables[t][index]),
        }
    }
}

/// The prover's tables.
pub struct ProverTables<'a> {
    /// Both rounds' committed columns.
    pub tables: SourceTables<'a>,
    /// `y^(1)_t` on `D_1`.
    pub parity1: Parity1,
    /// `w_1` on the main domain (after `r^(1)`).
    pub w1: Option<Vec<E>>,
    /// `w_1` as its two factors instead of the table (`AGG_W1_SPLIT=0` builds the table).
    pub w1_split: Option<SplitWeight>,
    /// `w_2` on `D_1` (after `tau`).
    pub w2: Option<Vec<E>>,
    /// `w_{3,t}` on `D_2` (after `tau`).
    pub w3: Option<[Vec<E>; W1_BITS]>,
    /// `w_A` on the main domain (after `r^(2)`).
    pub wa: Option<Vec<E>>,
    /// The projection weight `W` on the main domain (after `r^(2)`).
    pub wp: Option<Vec<E>>,
}

impl<'a> ProverTables<'a> {
    /// Reuse the parities that `BridgeGroupTables::build` derived from these main tables and
    /// matrices (moved out of the bridge tables by the honest driver). `None` (a carry
    /// mutation or a tamper hook dropped the cache) takes the general path.
    pub(crate) fn from_parity1_cache(
        tables: SourceTables<'a>,
        cache: Option<[Vec<i8>; W1_BITS]>,
        matrices: &BridgeMatrices,
    ) -> Self {
        match cache {
            Some(parities) => Self::with_parity1(tables, Parity1::Narrow(parities)),
            None => Self::new(tables, matrices),
        }
    }

    fn with_parity1(tables: SourceTables<'a>, parity1: Parity1) -> Self {
        Self { tables, parity1, w1: None, w1_split: None, w2: None, w3: None, wa: None, wp: None }
    }

    /// Materialize the parities of the committed bits under the bridge matrices.
    #[must_use]
    pub fn new(tables: SourceTables<'a>, matrices: &BridgeMatrices) -> Self {
        let parity1 = Parity1::from_values(parity1_tables(tables, &matrices.m1));
        Self {
            tables,
            parity1,
            w1: None,
            w1_split: None,
            w2: None,
            w3: None,
            wa: None,
            wp: None,
        }
    }

    /// Build `w_2` and `w_{3,t}` from `tau`.
    pub fn set_tau(&mut self, matrices: &BridgeMatrices, tau: &[E]) {
        let geom = self.tables.main.geometry();
        let tau2 = &tau[tau.len() - geom.d2() as usize..];
        let tau3 = &tau[tau.len() - geom.d3() as usize..];
        self.w2 = Some(level2_weight_table(&geom, &matrices.m2, tau2));
        self.w3 = Some(level3_weight_tables(&geom, &matrices.m3, tau3));
    }

    /// Build `w_1` from `r^(1)` and drop the stage-1 weights.
    pub fn set_r1(&mut self, matrices: &BridgeMatrices, r1: &[E]) {
        let geom = self.tables.main.geometry();
        let r1_1 = &r1[r1.len() - geom.d1() as usize..];
        self.w2 = None;
        self.w3 = None;
        let split_on = std::env::var_os("AGG_W1_SPLIT").is_none_or(|v| v != "0");
        let split = split_on.then(|| level1_weight_split(&geom, &matrices.m1, r1_1)).filter(SplitWeight::usable);
        if split.is_some() {
            self.w1_split = split;
        } else {
            self.w1 = Some(level1_weight_table(&geom, &matrices.m1, r1_1));
        }
    }

    /// Build `w_A` from `r^(2)` and drop `w_1`.
    pub fn set_r2(&mut self, public: &PublicData, r2: &[E]) {
        self.w1 = None;
        self.w1_split = None;
        self.wa = Some(ring_weight_table(public, r2));
    }

    /// Build the projection weight `W` from `Psi` and `r^(2)`.
    pub fn set_projection(&mut self, psi: &[u8], r2: &[E]) {
        self.wp = Some(crate::projection::weight_table(self.tables.manifest, psi, r2));
    }

    /// The lane of a table.
    #[must_use]
    pub fn lane(&self, id: LaneId) -> Lane<'_> {
        let column = |c: Column| self.tables.column(c);
        match id {
            LaneId::Column(c) => column(c).lane(),
            LaneId::DigitSquare(c) => Lane::mapped(column(c).narrow(), square_map()),
            LaneId::Parity1(t) => self.parity1.lane(usize::from(t)),
            LaneId::Weight(WeightId::Level1) => {
                match (&self.w1, &self.w1_split) {
                    (Some(table), _) => Lane::Source(Source::Field(table)),
                    // The stage takes the factors; the table only backs the fallback.
                    (None, Some(_)) => Lane::Folded(Vec::new()),
                    (None, None) => panic!("w_1 after r^(1)"),
                }
            }
            LaneId::Weight(WeightId::Level2) => {
                Lane::Source(Source::Field(self.w2.as_deref().expect("w_2 after tau")))
            }
            LaneId::Weight(WeightId::Level3(t)) => Lane::Source(Source::Field(
                &self.w3.as_ref().expect("w_3 after tau")[usize::from(t)],
            )),
            LaneId::Weight(WeightId::Ring) => {
                Lane::Source(Source::Field(self.wa.as_deref().expect("w_A after r^(2)")))
            }
            LaneId::Weight(WeightId::Proj) => {
                Lane::Source(Source::Field(self.wp.as_deref().expect("W after r^(2)")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Ring;

    #[test]
    fn parities_are_narrow_when_every_value_fits_and_wide_otherwise() {
        let bits: [Vec<i64>; W1_BITS] =
            std::array::from_fn(|t| (0..64).map(|i| ((i + t) % 2) as i64).collect());
        let narrow = Parity1::from_values(bits.clone());
        assert!(matches!(narrow, Parity1::Narrow(_)));
        assert!(matches!(narrow.lane(3), Lane::Source(Source::Bytes { .. })));
        let mut tampered = bits.clone();
        tampered[2][7] = 300;
        let wide = Parity1::from_values(tampered);
        assert!(matches!(wide, Parity1::Wide(_)));
        assert!(matches!(wide.lane(2), Lane::Source(Source::Ints(_))));
        assert_eq!(wide.value(2, 7), 300);
        for (t, table) in bits.iter().enumerate() {
            for (i, &y) in table.iter().enumerate() {
                assert_eq!(narrow.value(t, i), y);
                assert_eq!(narrow.lane(t).value(i), E::from_i64(y));
                if (t, i) != (2, 7) {
                    assert_eq!(wide.lane(t).value(i), E::from_i64(y));
                }
            }
        }
    }
    #[test]
    fn cached_parity_matches_recomputation_and_invalidates_on_carry_mutation() {
        use crate::manifest::CommitmentManifest;
        use crate::source_tables::{BridgeGroupTables, MainTables, RecordValues};
        let manifest = CommitmentManifest::for_batch(3).unwrap();
        let main = MainTables::from_values(&manifest, &vec![RecordValues::zero(); 3]).unwrap();
        let matrices = BridgeMatrices::random_for_tests(939);
        let mut bridge = BridgeGroupTables::build(&manifest, &main, &matrices, &crate::projection::test_psi(1)).unwrap();
        {
            let tables = SourceTables { manifest: &manifest, main: &main, bridge: &bridge };
            let cache = bridge.parity1_cached().cloned();
            assert!(cache.is_some());
            assert_eq!(ProverTables::from_parity1_cache(tables, cache, &matrices).parity1,
                ProverTables::new(tables, &matrices).parity1);
        }
        // Handing the cache over leaves the committed content (and equality) unchanged.
        let before = bridge.clone();
        assert!(bridge.take_parity1().is_some());
        assert!(bridge.parity1_cached().is_none());
        assert_eq!(bridge, before);
        let mut bridge = before;
        bridge.carries_mut(1)[0] -= 1;
        let tables = SourceTables { manifest: &manifest, main: &main, bridge: &bridge };
        let cache = bridge.parity1_cached().cloned();
        assert!(cache.is_none());
        let cached = ProverTables::from_parity1_cache(tables, cache, &matrices);
        assert_eq!(cached.parity1, ProverTables::new(tables, &matrices).parity1);
        assert_eq!(cached.parity1.value(0, 0), 2);
    }

}

//! The prover's narrow tables and the lanes the stages read from them.
//!
//! [`ProverTables`] borrows the committed `i8` columns of both rounds, materializes the
//! level-1 parities `y^(1)_t` (small integers on `D_1`) and holds the public weight tables
//! once they are known (`w_2, w_{3,t}` after `tau`; `w_1` after `r^(1)`; `w_A` after
//! `r^(2)`). Every [`LaneId`] resolves to a [`Lane`] over the narrow source; the stage
//! drivers lift to `K` only inside the first fold.

use agg_witness::bridge::BridgeMatrices;
use agg_witness::params::W1_BITS;

use crate::kernel::{Lane, Source};
use crate::manifest::Column;
use crate::public_data::PublicData;
use crate::range::{hi_map, lo_map, square_map};
use crate::source_tables::SourceTables;
use crate::stage::{LaneId, WeightId};
use crate::virtual_polys::parity1_tables;
use crate::weights::{level1_weight_table, level2_weight_table, level3_weight_tables, ring_weight_table};
use crate::OpeningField as E;

/// The prover's tables.
pub struct ProverTables<'a> {
    /// Both rounds' committed columns.
    pub tables: SourceTables<'a>,
    /// `y^(1)_t` on `D_1`.
    pub parity1: [Vec<i32>; W1_BITS],
    /// `w_1` on the main domain (after `r^(1)`).
    pub w1: Option<Vec<E>>,
    /// `w_2` on `D_1` (after `tau`).
    pub w2: Option<Vec<E>>,
    /// `w_{3,t}` on `D_2` (after `tau`).
    pub w3: Option<[Vec<E>; W1_BITS]>,
    /// `w_A` on the main domain (after `r^(2)`).
    pub wa: Option<Vec<E>>,
}

impl<'a> ProverTables<'a> {
    /// Materialize the parities of the committed bits under the bridge matrices.
    #[must_use]
    pub fn new(tables: SourceTables<'a>, matrices: &BridgeMatrices) -> Self {
        let parities = parity1_tables(tables, &matrices.m1);
        let parity1 = parities.map(|table| {
            table
                .into_iter()
                .map(|y| i32::try_from(y).expect("parity fits i32"))
                .collect()
        });
        Self {
            tables,
            parity1,
            w1: None,
            w2: None,
            w3: None,
            wa: None,
        }
    }

    /// Build `w_2` and `w_{3,t}` from `tau` (spec (7.2), (7.3)).
    pub fn set_tau(&mut self, matrices: &BridgeMatrices, tau: &[E]) {
        let geom = self.tables.main.geometry();
        let tau2 = &tau[tau.len() - geom.d2() as usize..];
        let tau3 = &tau[tau.len() - geom.d3() as usize..];
        self.w2 = Some(level2_weight_table(&geom, &matrices.m2, tau2));
        self.w3 = Some(level3_weight_tables(&geom, &matrices.m3, tau3));
    }

    /// Build `w_1` from `r^(1)` (spec (7.5)) and drop the stage-1 weights.
    pub fn set_r1(&mut self, matrices: &BridgeMatrices, r1: &[E]) {
        let geom = self.tables.main.geometry();
        let r1_1 = &r1[r1.len() - geom.d1() as usize..];
        self.w2 = None;
        self.w3 = None;
        self.w1 = Some(level1_weight_table(&geom, &matrices.m1, r1_1));
    }

    /// Build `w_A` from `r^(2)` (spec (7.9)) and drop `w_1`.
    pub fn set_r2(&mut self, public: &PublicData, r2: &[E]) {
        self.w1 = None;
        self.wa = Some(ring_weight_table(public, r2));
    }

    /// The lane of a table.
    #[must_use]
    pub fn lane(&self, id: LaneId) -> Lane<'_> {
        let column = |c: Column| self.tables.column(c);
        match id {
            LaneId::Column(c) => Lane::column(column(c)),
            LaneId::DigitSquare(c) => Lane::mapped(column(c), square_map()),
            LaneId::DigitLo(c) => Lane::mapped(column(c), lo_map()),
            LaneId::DigitHi(c) => Lane::mapped(column(c), hi_map()),
            LaneId::Parity1(t) => Lane::Source(Source::Ints(&self.parity1[usize::from(t)])),
            LaneId::Weight(WeightId::Level1) => {
                Lane::Source(Source::Field(self.w1.as_deref().expect("w_1 after r^(1)")))
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
        }
    }
}

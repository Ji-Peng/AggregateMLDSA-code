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

use crate::keccak_fp::virtuals::KeccakProverTables;
use crate::keccak_fp::weights::KeccakReduceTables;
use crate::kernel::{Lane, Source};
use crate::manifest::{Column, HashField};
use crate::public_data::PublicData;
use crate::range::square_map;
use crate::source_tables::SourceTables;
use crate::stage::{LaneId, WeightId};
use crate::virtual_polys::parity1_tables_for;
use crate::split_weight::{LaneSplitWeight, Level1Split, RankOneWeight, SplitWeight};
use crate::weights::{
    level1_lane_split, level1_lane_tables, level1_weight_split, level1_weight_table, level2_weight_table, level2_weight_table_for,
    level3_weight_tables, level3_weight_tables_for, ring_weight_planes, ring_weight_table,
};
use crate::OpeningField as E;

/// The level-1 parities `y^(1)_t` on `D_1`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Parity1 {
    /// Every value fits an `i8` (the honest parities are bits): coded byte lanes.
    Narrow(Vec<Vec<i8>>),
    /// Some value does not fit an `i8` (a tampered witness): `i32` lanes.
    Wide(Vec<Vec<i32>>),
}

impl Parity1 {
    /// The narrow tables when every value of `tables` fits an `i8`, else the wide ones.
    #[must_use]
    pub fn from_values(tables: Vec<Vec<i64>>) -> Self {
        if tables
            .iter()
            .all(|table| table.iter().all(|&y| i8::try_from(y).is_ok()))
        {
            Self::Narrow(tables.into_iter().map(|table| table.into_iter().map(|y| y as i8).collect()).collect())
        } else {
            Self::Wide(
                tables
                    .into_iter()
                    .map(|table| {
                        table
                            .into_iter()
                            .map(|y| i32::try_from(y).expect("parity fits i32"))
                            .collect()
                    })
                    .collect(),
            )
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
    /// `w_1` on the main domain (after `r^(1)`; ML-DSA-44).
    pub w1: Option<Vec<E>>,
    /// `w_1^(rho)` per polynomial lane of the bits (after `r^(1)`; the sets with lanes).
    pub w1_lanes: Vec<Vec<E>>,
    /// `w_1` as its two factors instead of the table (`AGG_W1_SPLIT=0` builds the table).
    pub w1_split: Option<SplitWeight>,
    /// `w_1^(rho)` as their factors instead of `w1_lanes` (the sets with lanes;
    /// `AGG_W1_SPLIT=0` builds the tables).
    pub w1_lane_split: Option<LaneSplitWeight>,
    /// `w_2` on `D_1` (after `tau`).
    pub w2: Option<Vec<E>>,
    /// `w_{3,t}` on `D_2` (after `tau`).
    pub w3: Option<Vec<Vec<E>>>,
    /// `w_A` on the main domain (after `r^(2)`).
    pub wa: Option<Vec<E>>,
    /// `w_A` as canonical planes when the plane layout is selected.
    pub wa_planes: Option<agg_simd::buf::KPlanes>,
    /// The projection weight `W` on the main domain (after `r^(2)`).
    pub wp: Option<Vec<E>>,
    /// `W` as coordinate planes (plane layout), in place of `wp`.
    pub wp_planes: Option<agg_simd::buf::KPlanes>,
    /// `W^(rho)` per polynomial lane of `v` (after `r^(2)`; the sets with lanes), as their
    /// factors: the in-lane factor per lane and the signature factor they share.
    pub wp_lanes: (Vec<Vec<E>>, Vec<E>),
    /// The committed and virtual Keccak tables under `HashField::Fp`.
    pub keccak: Option<&'a KeccakProverTables>,
    /// The stage-3 Keccak reduction weights (after `gamma^(3)`, under `HashField::Fp`).
    pub keccak_reduce: Option<KeccakReduceTables>,
}

impl<'a> ProverTables<'a> {
    /// Reuse the parities that `BridgeGroupTables::build` derived from these main tables and
    /// matrices (moved out of the bridge tables by the honest driver). `None` (a carry
    /// mutation or a tamper hook dropped the cache, or a set with polynomial lanes, whose
    /// bridge keeps no cache) takes the general path.
    pub(crate) fn from_parity1_cache(
        tables: SourceTables<'a>,
        cache: Option<Vec<Vec<i8>>>,
        matrices: &BridgeMatrices,
    ) -> Self {
        match cache {
            Some(parities) => Self::with_parity1(tables, Parity1::Narrow(parities)),
            None => Self::new(tables, matrices),
        }
    }

    fn with_parity1(tables: SourceTables<'a>, parity1: Parity1) -> Self {
        Self {
            tables,
            parity1,
            w1: None,
            w1_lanes: Vec::new(),
            w1_split: None,
            w1_lane_split: None,
            w2: None,
            w3: None,
            wa: None,
            wa_planes: None,
            wp: None,
            wp_planes: None,
            wp_lanes: (Vec::new(), Vec::new()),
            keccak: None,
            keccak_reduce: None,
        }
    }

    /// Attach the prime-field Keccak tables for resolving Keccak lanes.
    pub fn set_keccak(&mut self, keccak: &'a KeccakProverTables) {
        self.keccak = Some(keccak);
    }

    /// Attach the stage-3 Keccak reduction weights, built once `gamma^(3)` is drawn.
    pub fn set_keccak_reduce(&mut self, weights: KeccakReduceTables) {
        self.keccak_reduce = Some(weights);
    }

    fn keccak_reduce(&self, id: WeightId) -> &KeccakReduceTables {
        self.keccak_reduce
            .as_ref()
            .unwrap_or_else(|| panic!("the Keccak reduction weights are not set for {id:?}"))
    }

    /// Materialize the parities of the committed bits under the bridge matrices (`Gf2`), or an
    /// empty parity set under `Fp`.
    #[must_use]
    pub fn new(tables: SourceTables<'a>, matrices: &BridgeMatrices) -> Self {
        let parity1 = match tables.manifest.hash_field() {
            HashField::Gf2 => Parity1::from_values(parity1_tables_for(tables, &matrices.m1)),
            HashField::Fp => Parity1::Narrow(Vec::new()),
        };
        Self::with_parity1(tables, parity1)
    }

    /// True for a parameter set whose vector index is a polynomial lane.
    fn has_lanes(&self) -> bool {
        self.tables.manifest.has_lanes()
    }

    /// Build `w_2` and `w_{3,t}` from `tau`.
    pub fn set_tau(&mut self, matrices: &BridgeMatrices, tau: &[E]) {
        let manifest = self.tables.manifest;
        if self.has_lanes() {
            let tau2 = &tau[tau.len() - manifest.domain_vars(crate::manifest::GroupRole::Level2)..];
            let tau3 = &tau[tau.len() - manifest.domain_vars(crate::manifest::GroupRole::Level3)..];
            self.w2 = Some(level2_weight_table_for(manifest, &matrices.m2, tau2));
            self.w3 = Some(level3_weight_tables_for(manifest, &matrices.m3, tau3));
            return;
        }
        let geom = self.tables.main.geometry();
        let tau2 = &tau[tau.len() - geom.d2() as usize..];
        let tau3 = &tau[tau.len() - geom.d3() as usize..];
        self.w2 = Some(level2_weight_table(&geom, &matrices.m2, tau2));
        self.w3 = Some(level3_weight_tables(&geom, &matrices.m3, tau3).into());
    }

    /// Build `w_1` from `r^(1)` and drop the stage-1 weights.
    pub fn set_r1(&mut self, matrices: &BridgeMatrices, r1: &[E]) {
        self.w2 = None;
        self.w3 = None;
        let split_on = std::env::var_os("AGG_W1_SPLIT").is_none_or(|v| v != "0");
        if self.has_lanes() {
            let manifest = self.tables.manifest;
            let r1_1 = &r1[r1.len() - manifest.domain_vars(crate::manifest::GroupRole::Level1)..];
            self.w1_lane_split = split_on
                .then(|| level1_lane_split(manifest, &matrices.m1, r1_1))
                .flatten()
                .filter(LaneSplitWeight::usable);
            self.w1_lanes = if self.w1_lane_split.is_some() {
                Vec::new()
            } else {
                level1_lane_tables(manifest, &matrices.m1, r1_1)
            };
            return;
        }
        let geom = self.tables.main.geometry();
        let r1_1 = &r1[r1.len() - geom.d1() as usize..];
        let split = split_on.then(|| level1_weight_split(&geom, &matrices.m1, r1_1)).filter(SplitWeight::usable);
        if split.is_some() {
            self.w1 = None;
            self.w1_split = split;
        } else {
            self.w1_split = None;
            self.w1 = Some(level1_weight_table(&geom, &matrices.m1, r1_1));
        }
    }

    /// Build `w_A` from `r^(2)` and drop `w_1`.
    pub fn set_r2(&mut self, public: &PublicData, r2: &[E]) {
        self.w1 = None;
        self.w1_split = None;
        self.w1_lane_split = None;
        self.w1_lanes = Vec::new();
        if crate::kernel::layout() == crate::kernel::Layout::Plane {
            self.wa_planes = Some(ring_weight_planes(public, r2));
            self.wa = None;
        } else {
            self.wa = Some(ring_weight_table(public, r2));
            self.wa_planes = None;
        }
    }

    /// Drop `w_1^(rho)` after `r^(2)` (the sets with lanes, whose ring weights depend on the
    /// stage-3 coefficients and are built in stage 3).
    pub fn drop_level1(&mut self) {
        self.w1 = None;
        self.w1_split = None;
        self.w1_lane_split = None;
        self.w1_lanes = Vec::new();
    }

    /// Build the projection weight `W` from `Psi` and `r^(2)` (one weight per polynomial lane
    /// of `v` for the sets with lanes).
    pub fn set_projection(&mut self, psi: &[u8], r2: &[E]) {
        if self.has_lanes() {
            self.wp_lanes = crate::projection::lane_weight_factors(self.tables.manifest, psi, r2);
            return;
        }
        if crate::kernel::layout() == crate::kernel::Layout::Plane {
            self.wp_planes = Some(crate::projection::weight_planes(self.tables.manifest, psi, r2));
            self.wp = None;
        } else {
            self.wp = Some(crate::projection::weight_table(self.tables.manifest, psi, r2));
            self.wp_planes = None;
        }
    }

    /// A weight lane kept as two factors (the projection weights of the sets with lanes, the
    /// stage-3 Keccak reduction weights).
    ///
    /// # Panics
    /// On a Keccak reduction weight before [`Self::set_keccak_reduce`] or outside its tables.
    #[must_use]
    pub fn rank_one(&self, id: LaneId) -> Option<RankOneWeight<'_>> {
        match id {
            LaneId::Weight(w @ (WeightId::KeccakReduce { .. } | WeightId::KeccakAbsorb { .. })) => Some(
                self.keccak_reduce(w)
                    .rank_one(w)
                    .unwrap_or_else(|| panic!("{w:?} is not in the Keccak reduction weights")),
            ),
            LaneId::Weight(WeightId::Proj(lane)) if self.has_lanes() => Some(RankOneWeight {
                low: self.wp_lanes.0.get(usize::from(lane))?,
                high: &self.wp_lanes.1,
            }),
            _ => None,
        }
    }

    /// The level-1 weights as their factors, when they are kept that way.
    #[must_use]
    pub fn level1_split(&self) -> Option<Level1Split<'_>> {
        match (&self.w1_split, &self.w1_lane_split) {
            (Some(split), _) => Some(Level1Split::Table(split)),
            (None, Some(split)) => Some(Level1Split::Lanes(split)),
            (None, None) => None,
        }
    }

    /// The lane of a table.
    #[must_use]
    pub fn lane(&self, id: LaneId) -> Lane<'_> {
        let column = |c: Column| self.tables.column(c);
        match id {
            LaneId::Column(
                Column::KeccakA { .. } | Column::KeccakC { .. } | Column::KeccakCPrime { .. },
            )
            | LaneId::KeccakRot(_)
            | LaneId::KeccakPrev(_)
            | LaneId::KeccakPrevIota(_)
            | LaneId::KeccakBlk { .. }
            | LaneId::KeccakMu(_)
            | LaneId::KeccakCTilde(_) => self
                .keccak
                .unwrap_or_else(|| panic!("KeccakProverTables is not set on ProverTables for {id:?}"))
                .lane(id)
                .expect("Keccak lane"),
            LaneId::Column(c) => column(c).lane(),
            LaneId::DigitSquare(c) => Lane::mapped(column(c).narrow(), square_map()),
            LaneId::Parity1(t) => self.parity1.lane(usize::from(t)),
            LaneId::Weight(WeightId::Level1(lane)) if self.has_lanes() => match self.w1_lanes.get(usize::from(lane)) {
                Some(table) => Lane::Source(Source::Field(table)),
                // The stage takes the factors; the tables only back the fallback.
                None if self.w1_lane_split.is_some() => Lane::Folded(Vec::new()),
                None => panic!("w_1^(rho) after r^(1)"),
            },
            LaneId::Weight(WeightId::Level1(_)) => {
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
            LaneId::Weight(WeightId::Ring(_)) => {
                assert!(!self.has_lanes(), "the lane ring weights are built in stage 3");
                match &self.wa_planes {
                    Some(planes) => Lane::Source(Source::Planes(planes)),
                    None => Lane::Source(Source::Field(self.wa.as_deref().expect("w_A after r^(2)"))),
                }
            }
            // The stage takes the factors ([`Self::rank_one`]); a group that cannot use them
            // materializes the table from them.
            LaneId::Weight(WeightId::Proj(_)) if self.has_lanes() => Lane::Folded(Vec::new()),
            // The stage takes the factors ([`Self::rank_one`]) or materializes them; the lane is
            // only a placeholder, resolved once the weights are set.
            LaneId::Weight(w @ (WeightId::KeccakReduce { .. } | WeightId::KeccakAbsorb { .. })) => {
                let _ = self.keccak_reduce(w);
                Lane::Folded(Vec::new())
            }
            LaneId::Weight(WeightId::Proj(_)) => match &self.wp_planes {
                Some(planes) => Lane::Source(Source::Planes(planes)),
                None => Lane::Source(Source::Field(self.wp.as_deref().expect("W after r^(2)"))),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Ring;

    #[test]
    fn parities_are_narrow_when_every_value_fits_and_wide_otherwise() {
        let bits: Vec<Vec<i64>> = (0..6).map(|t| (0..64).map(|i| ((i + t) % 2) as i64).collect()).collect();
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

    #[test]
    fn keccak_reduction_weights_resolve_to_their_factors_once_set() {
        use crate::keccak_fp::weights::KeccakReductions;
        use crate::manifest::CommitmentManifest;
        use crate::source_tables::{BridgeGroupTables, MainTables, RecordValues};
        let manifest = CommitmentManifest::for_batch(3).unwrap();
        let main = MainTables::from_values(&manifest, &vec![RecordValues::zero(); 3]).unwrap();
        let matrices = BridgeMatrices::random_for_tests(941);
        let bridge = BridgeGroupTables::build(&manifest, &main, &matrices, &crate::projection::test_psi(1)).unwrap();
        let tables = SourceTables { manifest: &manifest, main: &main, bridge: &bridge };
        let mut prover = ProverTables::new(tables, &matrices);
        let fp = CommitmentManifest::for_batch_fp(8).unwrap();
        let plan = KeccakReductions::for_manifest(&fp, 0);
        let mut rng = agg_witness::gen::SplitMix64::new(942);
        let mut point = |len: usize| -> Vec<E> { (0..len).map(|_| E::from_u64(rng.next_u64())).collect() };
        let (r1, r2) = (point(fp.d_stage()), point(fp.d_stage()));
        let weights = plan.tables(&point(plan.claims().len()), &r1, &r2);
        let (w, a) = (&plan.weights()[3], &plan.absorbs()[1]);
        let reduce = WeightId::KeccakReduce { poly: w.poly, point: w.point };
        let absorb = WeightId::KeccakAbsorb { t: a.t, lane: a.lane, point: a.point };
        let expected = (weights.rank_one(reduce).unwrap().materialize(), weights.rank_one(absorb).unwrap().materialize());
        prover.set_keccak_reduce(weights);
        assert_eq!(prover.rank_one(LaneId::Weight(reduce)).unwrap().materialize(), expected.0);
        assert_eq!(prover.rank_one(LaneId::Weight(absorb)).unwrap().materialize(), expected.1);
        assert!(matches!(prover.lane(LaneId::Weight(reduce)), Lane::Folded(ref cells) if cells.is_empty()));
    }

    #[test]
    #[should_panic(expected = "KeccakProverTables is not set on ProverTables")]
    fn keccak_lane_panics_when_keccak_prover_tables_not_set() {
        use crate::manifest::CommitmentManifest;
        use crate::source_tables::{BridgeGroupTables, MainTables, RecordValues};
        let manifest = CommitmentManifest::for_batch(3).unwrap();
        let main = MainTables::from_values(&manifest, &vec![RecordValues::zero(); 3]).unwrap();
        let matrices = BridgeMatrices::random_for_tests(940);
        let bridge = BridgeGroupTables::build(&manifest, &main, &matrices, &crate::projection::test_psi(1)).unwrap();
        let tables = SourceTables { manifest: &manifest, main: &main, bridge: &bridge };
        let prover = ProverTables::new(tables, &matrices);
        let _ = prover.lane(LaneId::KeccakPrevIota(0));
    }
}

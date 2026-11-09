//! Group layouts (16-bit lookup digits) and the
//! generated fold-schedule catalogs.
//!
//! The seven witness groups, in wire and catalog order, are `G_main16` and `G_main4`
//! (round 1) and `G_1`, `G_2`, `G_3`, `G_k`, `G_m` (round 2); `G_m` (the multiplicity
//! digits of the lookup table `T_16`) is Akita's final group and the others are precommitted
//! groups. `G_k` holds the digits of the projection outputs of `v`.
//! Two presets share one batch: the width-16 family (`G_main16`, `G_1`, `G_2`, `G_3`, `G_k`) and
//! the width-4 family (`G_main4`, `G_m`). Akita freezes the source bound into each group's
//! own profile, so a precommitted group of either family enters the final family's grouped
//! row by its scalar producer profile (Akita's mixed-bound batches; the fp128 test
//! `bounded_dense_precommit_with_onehot_final_group` exercises the same path).
//!
//! A catalog row is keyed by the complete ordered layout, so every batch size
//! `N* = 2^d_sig` needs one grouped row in the width-4 catalog plus one scalar row per
//! precommitted layout in its own family's catalog, whose root profile is the producer
//! profile of that group.
//!
//! The relation changes the polynomial count of `G_main4`
//! (9 for H-exact, whose extra Boolean column `a_z` is the slack of the exact `z`
//! form, 12 for H-FIPS, which adds the sign bit, `delta` and `e`) and of `G_2` (13 for
//! H-FIPS: the hint slack `u_h` on `D_2`), so every layout,
//! row name and catalog key is a function of `(d_sig, relation)`. [`CATALOG_D_SIG_H_EXACT`]
//! lists the H-exact and [`CATALOG_D_SIG_H_FIPS`] the H-FIPS batch sizes whose rows are
//! checked in ([`catalog_batches`] lists all); `tests/all/schedule_catalog_regen.rs`
//! regenerates them.

use akita_params::PolynomialGroupLayout;

pub use agg_witness::Set;

/// Log2 of the PCS floor: every physical table has at least `2^13` cells
/// (Aerie `PCS_MIN_CELLS_LOG2`); smaller logical tables are zero-extended
/// with trailing pad variables.
pub const PCS_MIN_CELLS_LOG2: usize = 13;
/// Position variables `(j, rho or s)` of the main domain.
pub const D_POS: usize = 10;
/// Smallest `d_sig` (`N* = max(next_power_of_two(N), 8)`).
pub const MIN_D_SIG: usize = 3;
/// Largest `d_sig` of this wire version (`N <= 2^15`).
pub const MAX_D_SIG: usize = 15;
/// Signatures per segment, log2.
pub const D_SEG_MAX: usize = 13;
/// Variables of the lookup table `T_16` and of its multiplicity column.
pub const TABLE_VARS: usize = 16;
/// Balanced base-16 digits of one multiplicity (`m < 7 (16^7 - 1) / 15`).
pub const MULTIPLICITY_DIGITS: usize = 7;
/// Groups of one batch under `HashField::Gf2`.
pub const GROUP_COUNT: usize = 7;
/// Groups of one batch under `HashField::Fp` (`G_main16`, `G_main4`, the seven Keccak
/// groups `KeccakA(0..4)`, `KeccakC`, `KeccakCPrime`, `G_k`, `G_m`).
pub const GROUP_COUNT_FP: usize = 11;
/// Number of Keccak commitment groups under `HashField::Fp`.
pub const KECCAK_GROUP_COUNT: usize = 7;
/// Rows of the `{0,1}` projection of `v`, log2: `128 = 2^7`.
pub const PROJ_ROW_VARS: usize = 7;

/// Batch sizes (`d_sig`) whose relation-H-exact rows are generated and checked in: every
/// `N* = 2^3 .. 2^15` of the wire version.
pub const CATALOG_D_SIG_H_EXACT: &[usize] = &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Batch sizes (`d_sig`) whose relation-H-FIPS rows are generated and checked in.
pub const CATALOG_D_SIG_H_FIPS: &[usize] = &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Batch sizes (`d_sig`) whose `HashField::Fp` catalog rows are generated and checked in
/// for every parameter set and relation.
pub const CATALOG_D_SIG_FP: &[usize] = &[3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/// Relation byte of H-exact: the proof header's flags byte and byte 3 of the instance-record
/// bytes. Byte 0 belonged to a relaxed relation that this version no longer accepts.
pub const RELATION_H_EXACT: u8 = 1;
/// Relation byte of H-FIPS.
pub const RELATION_H_FIPS: u8 = 2;

/// The relation a proof certifies. Its id is bound twice: as
/// the proof header's flags byte and in the instance-record manifest bytes; it selects the layout of
/// `G_main4` and hence the catalog rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Relation {
    /// H-exact: the FIPS 204 bounds `|z| <= 130,993` and `|u| <= 190,465`, both exact (a
    /// reference relation without the hint).
    HExact,
    /// H-FIPS (the default): H-exact's bounds and the hint: the extracted `(c~, z, h)` is a
    /// FIPS 204 signature.
    #[default]
    HFips,
}

impl Relation {
    /// Every relation, in id order.
    pub const ALL: [Self; 2] = [Self::HExact, Self::HFips];

    /// The relation byte.
    #[must_use]
    pub const fn id(self) -> u8 {
        match self {
            Self::HExact => RELATION_H_EXACT,
            Self::HFips => RELATION_H_FIPS,
        }
    }

    /// Inverse of [`Relation::id`] over the relations this version accepts.
    #[must_use]
    pub const fn from_id(id: u8) -> Option<Self> {
        match id {
            RELATION_H_EXACT => Some(Self::HExact),
            RELATION_H_FIPS => Some(Self::HFips),
            _ => None,
        }
    }

    /// Display name (`H-exact`, `H-FIPS`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::HExact => "H-exact",
            Self::HFips => "H-FIPS",
        }
    }

    /// Command-line spelling (`h-exact`, `h-fips`); [`Relation::parse`] inverts it.
    #[must_use]
    pub const fn flag(self) -> &'static str {
        match self {
            Self::HExact => "h-exact",
            Self::HFips => "h-fips",
        }
    }

    /// Parse a command-line spelling (`h-exact`, `hexact`, `exact`, `h-fips`, `hfips`, or the
    /// relation byte).
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "h-exact" | "hexact" | "exact" | "1" => Some(Self::HExact),
            "h-fips" | "hfips" | "2" => Some(Self::HFips),
            _ => None,
        }
    }

    /// True when `set` defines this relation (every set defines every relation).
    #[must_use]
    pub const fn is_available(self, set: Set) -> bool {
        matches!((set, self), (Set::S44 | Set::S65 | Set::S87, _))
    }

    /// The batch sizes whose rows of this relation are checked in under `HashField::Gf2`.
    #[must_use]
    pub const fn catalog_d_sigs(self) -> &'static [usize] {
        match self {
            Self::HExact => CATALOG_D_SIG_H_EXACT,
            Self::HFips => CATALOG_D_SIG_H_FIPS,
        }
    }
}

impl std::fmt::Display for Relation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The field in which the commitment-hash relation `c~ = SHAKE256(mu || w1Encode(w1'))`
/// is proven.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum HashField {
    /// Mixed-field baseline: Flock over `GF(2)` tied to the `F_p` side by the parity bridge.
    #[default]
    Gf2,
    /// All-prime-field comparator: Keccak-f[1600] round states committed and zero-checked
    /// directly in `F_p`.
    Fp,
}

impl HashField {
    /// Both hash fields, in default-first order.
    pub const ALL: [Self; 2] = [Self::Gf2, Self::Fp];

    /// Display name (`GF(2)`, `F_p`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gf2 => "GF(2)",
            Self::Fp => "F_p",
        }
    }

    /// Command-line spelling (`gf2`, `fp`); [`HashField::parse`] inverts it.
    #[must_use]
    pub const fn flag(self) -> &'static str {
        match self {
            Self::Gf2 => "gf2",
            Self::Fp => "fp",
        }
    }

    /// Parse a command-line spelling (`gf2`, `fp`).
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "gf2" | "gf(2)" => Some(Self::Gf2),
            "fp" | "f_p" => Some(Self::Fp),
            _ => None,
        }
    }

    /// Number of committed groups in a batch under this hash field.
    #[must_use]
    pub const fn group_count(self) -> usize {
        match self {
            Self::Gf2 => GROUP_COUNT,
            Self::Fp => GROUP_COUNT_FP,
        }
    }

    /// The committed groups of a batch in wire order.
    #[must_use]
    pub const fn groups(self) -> &'static [Group] {
        match self {
            Self::Gf2 => &Group::ALL,
            Self::Fp => &Group::ALL_FP,
        }
    }
}

impl std::fmt::Display for HashField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The commitment preset (schedule family) of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Family {
    /// `preset::AggFp64SignedWidth4`: values in `[-8, 7]`.
    Width4,
    /// `preset::AggFp64SignedWidth16`: values in `[-2^15, 2^15 - 1]`.
    Width16,
}

impl Family {
    /// Both families.
    pub const ALL: [Self; 2] = [Self::Width4, Self::Width16];

    /// Canonical transcript byte (the instance record).
    #[must_use]
    pub const fn transcript_id(self) -> u8 {
        match self {
            Self::Width4 => 0,
            Self::Width16 => 1,
        }
    }

    /// Smallest admitted value.
    #[must_use]
    pub const fn min_value(self) -> i64 {
        match self {
            Self::Width4 => -8,
            Self::Width16 => -(1 << 15),
        }
    }

    /// Largest admitted value.
    #[must_use]
    pub const fn max_value(self) -> i64 {
        match self {
            Self::Width4 => 7,
            Self::Width16 => (1 << 15) - 1,
        }
    }
}

/// Committed witness groups across `HashField::Gf2` (seven groups, [`Group::ALL`]) and
/// `HashField::Fp` (eleven groups, [`Group::ALL_FP`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// `Z_lo, U_lo, V_lo, V_hi` on the main domain (round 1, width 16): the 16-bit parts of
    /// `z, u, v`; `Z_lo, U_lo` are looked up, `V_lo, V_hi` are not (the projection of `v`
    /// bounds `v`).
    Main16,
    /// `Z_hi, U_hi, b_0..b_5` on the main domain (round 1, width 4); H-exact appends `a_z`.
    Main,
    /// Post-theta Keccak state lanes `A'[x, y]` (`x in 0..5`, slot groups `q in 0..Q`) for
    /// row `y in 0..5` on the Keccak domain `d_K = 10 + d_sig` (round 1, width 4, `HashField::Fp` only).
    KeccakA(u8),
    /// Pre-theta Keccak column parities `C[x]` (`x in 0..5`, slot groups `q in 0..Q`) on `d_K`
    /// (round 1, width 4, `HashField::Fp` only).
    KeccakC,
    /// Post-theta Keccak column parities `C'[x]` (`x in 0..5`, slot groups `q in 0..Q`) on `d_K`
    /// (round 1, width 4, `HashField::Fp` only).
    KeccakCPrime,
    /// Level-1 carries `kappa^(1)_t` on `D_1` (round 2, width 16, `HashField::Gf2` only).
    Level1,
    /// Level-2 parities `y^(2)_t` and carries `kappa^(2)_t` on `D_2` (round 2, width 16, `HashField::Gf2` only).
    Level2,
    /// The level-3 carry `kappa^(3)` on `D_3` and zero polynomials, in `G_2`'s layout (round 2,
    /// width 16, `HashField::Gf2` only).
    Level3,
    /// The digits `K_0, K_1, K_2, K_3` of the projection outputs `k = <Psi_o, v_i>` on the
    /// projection domain `(o, i)` (round 2, width 16); under `HashField::Fp`, relation H-FIPS
    /// appends the hint slack `u_h` ([`Group::num_polynomials_in`]).
    Proj,
    /// The seven balanced base-16 digits of the multiplicity column `m` of `T_16`
    /// (round 2, width 4, Akita final group).
    Mult,
}

impl Group {
    /// All groups of `HashField::Gf2` in wire and catalog order.
    pub const ALL: [Self; GROUP_COUNT] = [
        Self::Main16,
        Self::Main,
        Self::Level1,
        Self::Level2,
        Self::Level3,
        Self::Proj,
        Self::Mult,
    ];

    /// All groups of `HashField::Fp` in wire and catalog order.
    pub const ALL_FP: [Self; GROUP_COUNT_FP] = [
        Self::Main16,
        Self::Main,
        Self::KeccakA(0),
        Self::KeccakA(1),
        Self::KeccakA(2),
        Self::KeccakA(3),
        Self::KeccakA(4),
        Self::KeccakC,
        Self::KeccakCPrime,
        Self::Proj,
        Self::Mult,
    ];

    /// The seven `HashField::Fp` Keccak groups in wire order.
    pub const KECCAK: [Self; KECCAK_GROUP_COUNT] = [
        Self::KeccakA(0),
        Self::KeccakA(1),
        Self::KeccakA(2),
        Self::KeccakA(3),
        Self::KeccakA(4),
        Self::KeccakC,
        Self::KeccakCPrime,
    ];

    /// Number of 16-round slot groups `Q = ceil(24 * perms / 16)` per Keccak lane for `set`
    /// (`11` for ML-DSA-44 and ML-DSA-65, `14` for ML-DSA-87).
    #[must_use]
    pub const fn keccak_slot_groups(set: Set) -> usize {
        match set {
            Set::S44 | Set::S65 => 11,
            Set::S87 => 14,
        }
    }

    /// True if this group is one of the seven `HashField::Fp` Keccak groups.
    #[must_use]
    pub const fn is_keccak(self) -> bool {
        matches!(self, Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime)
    }

    /// Canonical role index (matches the wire position in `HashField::Gf2` for the seven
    /// `Gf2` groups, and assigns distinct tags `7..=13` to the seven `Fp` Keccak groups).
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Main16 => 0,
            Self::Main => 1,
            Self::Level1 => 2,
            Self::Level2 => 3,
            Self::Level3 => 4,
            Self::Proj => 5,
            Self::Mult => 6,
            Self::KeccakA(y) => {
                assert!(y < 5, "KeccakA row y must be in 0..5");
                7 + y as usize
            }
            Self::KeccakC => 12,
            Self::KeccakCPrime => 13,
        }
    }

    /// Position in the wire order of `hash_field` (`0..7` under `Gf2`, `0..11` under `Fp`).
    #[must_use]
    pub const fn index_in(self, hash_field: HashField) -> usize {
        match hash_field {
            HashField::Gf2 => match self {
                Self::Main16 => 0,
                Self::Main => 1,
                Self::Level1 => 2,
                Self::Level2 => 3,
                Self::Level3 => 4,
                Self::Proj => 5,
                Self::Mult => 6,
                Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime => {
                    panic!("Keccak groups belong to HashField::Fp")
                }
            },
            HashField::Fp => match self {
                Self::Main16 => 0,
                Self::Main => 1,
                Self::KeccakA(y) => {
                    assert!(y < 5, "KeccakA row y must be in 0..5");
                    2 + y as usize
                }
                Self::KeccakC => 7,
                Self::KeccakCPrime => 8,
                Self::Proj => 9,
                Self::Mult => 10,
                Self::Level1 | Self::Level2 | Self::Level3 => {
                    panic!("bridge groups belong to HashField::Gf2")
                }
            },
        }
    }

    /// Commitment round: 1 for the main and Keccak groups, 2 for the bridge, projection
    /// and multiplicity groups.
    #[must_use]
    pub const fn round(self) -> u8 {
        match self {
            Self::Main16
            | Self::Main
            | Self::KeccakA(_)
            | Self::KeccakC
            | Self::KeccakCPrime => 1,
            Self::Level1 | Self::Level2 | Self::Level3 | Self::Proj | Self::Mult => 2,
        }
    }

    /// Whether this is Akita's final group (the last group of the batch).
    #[must_use]
    pub const fn is_final(self) -> bool {
        matches!(self, Self::Mult)
    }

    /// The preset of the group.
    #[must_use]
    pub const fn family(self) -> Family {
        match self {
            Self::Main
            | Self::KeccakA(_)
            | Self::KeccakC
            | Self::KeccakCPrime
            | Self::Mult => Family::Width4,
            Self::Main16 | Self::Level1 | Self::Level2 | Self::Level3 | Self::Proj => {
                Family::Width16
            }
        }
    }

    /// Number of committed polynomials for a parameter set and relation.
    ///
    /// For ML-DSA-44: 4 on the main domain (round 1, width 16); 9 or 12
    /// under relations H-exact and H-FIPS (round 1, width 4); 55 per Keccak group;
    /// 6 carries on Level 1; 12 parities and carries on Level 2 (13 under H-FIPS); Level 3
    /// matches Level 2; 4 projection digits and 7 multiplicity digits.
    ///
    /// For ML-DSA-65 (relation H-exact): 23 on width-16 main domain (5 for z_lo, 6 each for
    /// u_lo, v_lo, v_hi); 46 on width-4 main domain (5 each for z_mid and a_z, 6 each for
    /// u_mid and a_u, 24 for 4 bit planes across 6 lanes); 55 per Keccak group; 4 carries on Level 1;
    /// 8 parities and carries on Level 2; Level 3 matches Level 2; 5 projection digits;
    /// 7 multiplicity digits.
    ///
    /// For ML-DSA-87 (relation H-exact): 31 on width-16 main domain (7 for z_lo, 8 each for
    /// u_lo, v_lo, v_hi); 62 on width-4 main domain (7 each for z_mid and a_z, 8 each for
    /// u_mid and a_u, 32 for 4 bit planes across 8 lanes); 70 per Keccak group; 4 carries on Level 1;
    /// 8 parities and carries on Level 2; Level 3 matches Level 2; 5 projection digits;
    /// 7 multiplicity digits.
    #[must_use]
    pub const fn num_polynomials(self, set: Set, relation: Relation) -> usize {
        if self.is_keccak() {
            return 5 * Self::keccak_slot_groups(set);
        }
        match set {
            Set::S44 => match self {
                Self::Main16 => 4,
                Self::Main => match relation {
                    Relation::HExact => 2 + 6 + 1,
                    Relation::HFips => 2 + 6 + 1 + 3,
                },
                Self::Level1 => 6,
                Self::Level2 => match relation {
                    Relation::HExact => 6 + 6,
                    Relation::HFips => 6 + 6 + 1,
                },
                Self::Level3 => Self::Level2.num_polynomials(set, relation),
                Self::Proj => 4,
                Self::Mult => MULTIPLICITY_DIGITS,
                Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime => unreachable!(),
            },
            Set::S65 => match self {
                Self::Main16 => 5 + 3 * 6,
                Self::Main => match relation {
                    Relation::HExact => 5 + 5 + 6 + 6 + 4 * 6,
                    Relation::HFips => 5 + 5 + 6 + 6 + 4 * 6 + 3 * 6,
                },
                Self::Level1 => 4,
                Self::Level2 => match relation {
                    Relation::HExact => 4 + 4,
                    Relation::HFips => 4 + 4 + 1,
                },
                Self::Level3 => Self::Level2.num_polynomials(set, relation),
                Self::Proj => 5,
                Self::Mult => MULTIPLICITY_DIGITS,
                Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime => unreachable!(),
            },
            Set::S87 => match self {
                Self::Main16 => 7 + 3 * 8,
                Self::Main => match relation {
                    Relation::HExact => 7 + 7 + 8 + 8 + 4 * 8,
                    Relation::HFips => 7 + 7 + 8 + 8 + 4 * 8 + 3 * 8,
                },
                Self::Level1 => 4,
                Self::Level2 => match relation {
                    Relation::HExact => 4 + 4,
                    Relation::HFips => 4 + 4 + 1,
                },
                Self::Level3 => Self::Level2.num_polynomials(set, relation),
                Self::Proj => 5,
                Self::Mult => MULTIPLICITY_DIGITS,
                Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime => unreachable!(),
            },
        }
    }

    /// Number of committed polynomials under `hash_field`: [`Self::num_polynomials`], except
    /// that relation H-FIPS under `HashField::Fp` (no `G_2`) puts the per-signature hint slack
    /// `u_h` into `G_k` after the projection digits, so `G_k` has one polynomial more.
    #[must_use]
    pub const fn num_polynomials_in(self, set: Set, relation: Relation, hash_field: HashField) -> usize {
        let base = self.num_polynomials(set, relation);
        match (self, relation, hash_field) {
            (Self::Proj, Relation::HFips, HashField::Fp) => base + 1,
            _ => base,
        }
    }

    /// Logical variable count of the group's domain for a given parameter set and `d_sig`.
    ///
    /// Every Keccak group has `d_K = 10 + d_sig` variables for all parameter sets.
    ///
    /// For ML-DSA-44: `d_main = 10 + d_sig`, `d_1 = 13 + max(0, d_sig - 9)`,
    /// `d_2 = 13 + max(0, d_sig - 13)`, `d_3 = 12 + max(0, d_sig - 13)`,
    /// `d_k = 7 + d_sig`; 16 for `G_m`.
    ///
    /// For ML-DSA-65: `d_main = 8 + d_sig`, `d_1 = 13 + max(0, d_sig - 8)`,
    /// `d_2 = 13 + max(0, d_sig - 13)`, `d_3 = 12 + max(0, d_sig - 13)`,
    /// `d_k = 7 + d_sig`; 16 for `G_m`.
    ///
    /// For ML-DSA-87: `d_main = 8 + d_sig`, `d_1 = 13 + max(0, d_sig - 8)`,
    /// `d_2 = 13 + max(0, d_sig - 13)`, `d_3 = 12 + max(0, d_sig - 13)`,
    /// `d_k = 8 + d_sig` (two projection row blocks); 16 for `G_m`.
    #[must_use]
    pub const fn logical_vars(self, set: Set, d_sig: usize) -> usize {
        if self.is_keccak() {
            return D_POS + d_sig;
        }
        match set {
            Set::S44 => match self {
                Self::Main16 | Self::Main => D_POS + d_sig,
                Self::Level1 => 13 + d_sig.saturating_sub(9),
                Self::Level2 => 13 + d_sig.saturating_sub(D_SEG_MAX),
                Self::Level3 => 12 + d_sig.saturating_sub(D_SEG_MAX),
                Self::Proj => PROJ_ROW_VARS + d_sig,
                Self::Mult => TABLE_VARS,
                Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime => unreachable!(),
            },
            Set::S65 => match self {
                Self::Main16 | Self::Main => 8 + d_sig,
                Self::Level1 => 13 + d_sig.saturating_sub(8),
                Self::Level2 => 13 + d_sig.saturating_sub(D_SEG_MAX),
                Self::Level3 => 12 + d_sig.saturating_sub(D_SEG_MAX),
                Self::Proj => PROJ_ROW_VARS + d_sig,
                Self::Mult => TABLE_VARS,
                Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime => unreachable!(),
            },
            Set::S87 => match self {
                Self::Main16 | Self::Main => 8 + d_sig,
                Self::Level1 => 13 + d_sig.saturating_sub(8),
                Self::Level2 => 13 + d_sig.saturating_sub(D_SEG_MAX),
                Self::Level3 => 12 + d_sig.saturating_sub(D_SEG_MAX),
                Self::Proj => PROJ_ROW_VARS + 1 + d_sig,
                Self::Mult => TABLE_VARS,
                Self::KeccakA(_) | Self::KeccakC | Self::KeccakCPrime => unreachable!(),
            },
        }
    }

    /// Physical (committed) variable count for a given parameter set and `d_sig`.
    #[must_use]
    pub const fn physical_vars(self, set: Set, d_sig: usize) -> usize {
        if matches!(self, Self::Level3) {
            return Self::Level2.physical_vars(set, d_sig);
        }
        let logical = self.logical_vars(set, d_sig);
        if logical < PCS_MIN_CELLS_LOG2 {
            PCS_MIN_CELLS_LOG2
        } else {
            logical
        }
    }

    /// Akita group layout for a given parameter set at `d_sig` under `relation`.
    #[must_use]
    pub const fn layout(self, set: Set, d_sig: usize, relation: Relation) -> PolynomialGroupLayout {
        PolynomialGroupLayout::new(self.physical_vars(set, d_sig), self.num_polynomials(set, relation))
    }

    /// Akita group layout under `hash_field` ([`Self::num_polynomials_in`]).
    #[must_use]
    pub const fn layout_in(self, set: Set, d_sig: usize, relation: Relation, hash_field: HashField) -> PolynomialGroupLayout {
        PolynomialGroupLayout::new(
            self.physical_vars(set, d_sig),
            self.num_polynomials_in(set, relation, hash_field),
        )
    }

    /// Short label used in catalog row names and logs.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Main16 => "main16",
            Self::Main => "main",
            Self::KeccakA(0) => "keccak_a0",
            Self::KeccakA(1) => "keccak_a1",
            Self::KeccakA(2) => "keccak_a2",
            Self::KeccakA(3) => "keccak_a3",
            Self::KeccakA(4) => "keccak_a4",
            Self::KeccakA(_) => "keccak_a",
            Self::KeccakC => "keccak_c",
            Self::KeccakCPrime => "keccak_c_prime",
            Self::Level1 => "level1",
            Self::Level2 => "level2",
            Self::Level3 => "level3",
            Self::Proj => "proj",
            Self::Mult => "mult",
        }
    }
}

/// The ordered group layout of one batch size `N* = 2^d_sig` under one relation,
/// parameter set and hash field (`GROUP_COUNT` groups under `Gf2`, `GROUP_COUNT_FP`
/// under `Fp`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchLayout {
    /// The parameter set (FIPS 204 ML-DSA-44, 65, or 87).
    pub set: Set,
    /// `log2 N*`.
    pub d_sig: usize,
    /// The relation (it fixes the polynomial counts of `G_main4` and `G_2`).
    pub relation: Relation,
    /// The field in which the commitment hash is proven (`Gf2` or `Fp`).
    pub hash_field: HashField,
    /// Physical layouts of the batch's groups in wire order.
    pub groups: Vec<PolynomialGroupLayout>,
}

impl BatchLayout {
    /// The `HashField::Gf2` layout for `set` at `d_sig` under `relation`; `None` outside
    /// `[MIN_D_SIG, MAX_D_SIG]` or if `set` does not define the relation ([`Relation::is_available`]).
    #[must_use]
    pub fn new(set: Set, d_sig: usize, relation: Relation) -> Option<Self> {
        Self::for_hash_field(set, d_sig, relation, HashField::Gf2)
    }

    /// The `HashField::Fp` layout for `set` at `d_sig` under `relation` (`GROUP_COUNT_FP`
    /// groups in wire order `G_main16, G_main4, KeccakA[0..4], KeccakC, KeccakCPrime, G_k, G_m`).
    #[must_use]
    pub fn new_fp(set: Set, d_sig: usize, relation: Relation) -> Option<Self> {
        Self::for_hash_field(set, d_sig, relation, HashField::Fp)
    }

    /// The layout for `set` at `d_sig` under `relation` and `hash_field`.
    #[must_use]
    pub fn for_hash_field(
        set: Set,
        d_sig: usize,
        relation: Relation,
        hash_field: HashField,
    ) -> Option<Self> {
        if !relation.is_available(set) {
            return None;
        }
        (MIN_D_SIG..=MAX_D_SIG).contains(&d_sig).then(|| Self {
            set,
            d_sig,
            relation,
            hash_field,
            groups: hash_field
                .groups()
                .iter()
                .map(|&group| group.layout_in(set, d_sig, relation, hash_field))
                .collect(),
        })
    }

    /// The precommitted groups (every group but `G_m`), in order.
    #[must_use]
    pub fn precommitted(&self) -> &[PolynomialGroupLayout] {
        &self.groups[..self.groups.len() - 1]
    }

    /// The precommitted groups with their roles, in order.
    #[must_use]
    pub fn precommitted_groups(&self) -> Vec<(Group, PolynomialGroupLayout)> {
        let roles = self.hash_field.groups();
        roles[..roles.len() - 1]
            .iter()
            .copied()
            .zip(self.precommitted().iter().copied())
            .collect()
    }

    /// The final group `G_m`.
    #[must_use]
    pub fn final_group(&self) -> PolynomialGroupLayout {
        *self.groups.last().expect("non-empty groups")
    }

    /// Total committed polynomials (54 for H-exact, 59 for H-FIPS under ML-DSA-44; 101 for
    /// H-exact, 121 for H-FIPS under ML-DSA-65; 125 for H-exact, 151 for H-FIPS under
    /// ML-DSA-87).
    #[must_use]
    pub fn total_polynomials(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.num_polynomials())
            .sum()
    }

    /// Largest physical variable count (the main or Keccak domain).
    #[must_use]
    pub fn max_physical_vars(&self) -> usize {
        self.groups
            .iter()
            .map(|group| group.num_vars())
            .max()
            .expect("non-empty groups")
    }

    /// Catalog row name of the grouped row (`p16x` for H-exact, `p16f` for H-FIPS: the lookup
    /// into `T_16` with the projected `v`; `-fp` before `-batch-` under `HashField::Fp`). The
    /// name is bound in the instance record.
    #[must_use]
    pub fn row_name(&self) -> String {
        let layout = match self.relation {
            Relation::HExact => "p16x",
            Relation::HFips => "p16f",
        };
        let field = match self.hash_field {
            HashField::Gf2 => "",
            HashField::Fp => "-fp",
        };
        format!(
            "mldsa-{}-{layout}{field}-batch-{}-mult-final",
            self.set.name(),
            1_usize << self.d_sig
        )
    }

    /// Whether this batch size has checked-in catalog rows for its set, relation and hash field.
    #[must_use]
    pub fn in_catalog(&self) -> bool {
        self.relation.is_available(self.set)
            && match self.hash_field {
                HashField::Gf2 => self.relation.catalog_d_sigs().contains(&self.d_sig),
                HashField::Fp => CATALOG_D_SIG_FP.contains(&self.d_sig),
            }
    }
}

/// Every `HashField::Gf2` batch layout for a given parameter set whose catalog rows can be resolved.
#[must_use]
pub fn catalog_batches_for_set(set: Set) -> Vec<BatchLayout> {
    Relation::ALL
        .iter()
        .filter(|relation| relation.is_available(set))
        .flat_map(|&relation| {
            relation
                .catalog_d_sigs()
                .iter()
                .filter_map(move |&d_sig| BatchLayout::new(set, d_sig, relation))
        })
        .collect()
}

/// Every `HashField::Fp` batch layout for a given parameter set whose catalog rows can be resolved.
#[must_use]
pub fn catalog_batches_for_set_fp(set: Set) -> Vec<BatchLayout> {
    Relation::ALL
        .iter()
        .filter(|relation| relation.is_available(set))
        .flat_map(|&relation| {
            CATALOG_D_SIG_FP
                .iter()
                .filter_map(move |&d_sig| BatchLayout::new_fp(set, d_sig, relation))
        })
        .collect()
}

/// Every checked-in `HashField::Gf2` batch layout: 26 per parameter set (13 batch sizes
/// under each relation).
#[must_use]
pub fn catalog_batches() -> Vec<BatchLayout> {
    let mut batches = catalog_batches_for_set(Set::S44);
    batches.extend(catalog_batches_for_set(Set::S65));
    batches.extend(catalog_batches_for_set(Set::S87));
    batches
}

/// Every checked-in `HashField::Fp` batch layout across all parameter sets and relations.
#[must_use]
pub fn catalog_batches_fp() -> Vec<BatchLayout> {
    let mut batches = catalog_batches_for_set_fp(Set::S44);
    batches.extend(catalog_batches_for_set_fp(Set::S65));
    batches.extend(catalog_batches_for_set_fp(Set::S87));
    batches
}

/// Every checked-in batch layout across both `HashField::Gf2` and `HashField::Fp`.
#[must_use]
pub fn all_catalog_batches() -> Vec<BatchLayout> {
    let mut batches = catalog_batches();
    batches.extend(catalog_batches_fp());
    batches
}

/// Every distinct scalar (precommitted-producer) layout of `family` needed by `batches`,
/// sorted by `(num_vars, num_polynomials)`.
#[must_use]
pub fn scalar_layouts(batches: &[BatchLayout], family: Family) -> Vec<PolynomialGroupLayout> {
    let mut layouts: Vec<PolynomialGroupLayout> = batches
        .iter()
        .flat_map(|layout| {
            layout
                .precommitted_groups()
                .into_iter()
                .filter(|(group, _)| group.family() == family)
                .map(|(_, layout)| layout)
                .collect::<Vec<_>>()
        })
        .collect();
    layouts.sort_by_key(|layout| (layout.num_vars(), layout.num_polynomials()));
    layouts.dedup();
    layouts
}

/// The checked-in width-4 catalog artifact (scalar `G_main4` rows and the grouped rows).
#[cfg(feature = "schedules")]
pub(crate) const WIDTH4_ARTIFACT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/schedules/agg_mldsa_fp64_signed_width4.aks"
));

/// The checked-in width-16 catalog artifact (scalar rows of `G_main16, G_1, G_2, G_3`).
#[cfg(feature = "schedules")]
pub(crate) const WIDTH16_ARTIFACT: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/src/schedules/agg_mldsa_fp64_signed_width16.aks"
));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_shapes_match_the_committed_layout() {
        let counts = |set, relation| Group::ALL.map(|group| group.num_polynomials(set, relation));
        assert_eq!(counts(Set::S44, Relation::HExact), [4, 9, 6, 12, 12, 4, 7]);
        assert_eq!(counts(Set::S44, Relation::HFips), [4, 12, 6, 13, 13, 4, 7]);
        assert_eq!(counts(Set::S65, Relation::HExact), [23, 46, 4, 8, 8, 5, 7]);
        assert_eq!(counts(Set::S87, Relation::HExact), [31, 62, 4, 8, 8, 5, 7]);
        let total = |set, relation| counts(set, relation).iter().sum::<usize>();
        for (set, exact, fips) in [(Set::S44, 54, 59), (Set::S65, 101, 121), (Set::S87, 125, 151)] {
            assert_eq!((total(set, Relation::HExact), total(set, Relation::HFips)), (exact, fips));
        }
        for relation in Relation::ALL {
            assert_eq!(Relation::from_id(relation.id()), Some(relation));
            assert_eq!(Relation::parse(relation.flag()), Some(relation));
        }
        assert_eq!(Relation::default(), Relation::HFips);
        assert_eq!((RELATION_H_EXACT, RELATION_H_FIPS), (1, 2));
        assert_eq!(Relation::from_id(2), Some(Relation::HFips));
        assert_eq!(Relation::from_id(0), None);
        assert_eq!(Relation::from_id(3), None);
        assert_eq!(Relation::parse("fips"), None);
        assert_eq!(Relation::parse("h-u"), None);
        assert_eq!(Relation::parse("0"), None);
        // The domain sizes at N* = 2^15 for 44.
        assert_eq!(Group::Main16.logical_vars(Set::S44, 15), 25);
        assert_eq!(Group::Main.logical_vars(Set::S44, 15), 25);
        assert_eq!(Group::Level1.logical_vars(Set::S44, 15), 19);
        assert_eq!(Group::Level2.logical_vars(Set::S44, 15), 15);
        assert_eq!(Group::Level3.logical_vars(Set::S44, 15), 14);
        assert_eq!(Group::Proj.logical_vars(Set::S44, 15), 22);
        assert_eq!(Group::Mult.logical_vars(Set::S44, 3), 16);
        // The domain sizes at N* = 2^15 for 65 and 87.
        assert_eq!(Group::Main16.logical_vars(Set::S65, 15), 23);
        assert_eq!(Group::Main.logical_vars(Set::S65, 15), 23);
        assert_eq!(Group::Level1.logical_vars(Set::S65, 15), 20);
        assert_eq!(Group::Level2.logical_vars(Set::S65, 15), 15);
        assert_eq!(Group::Level3.logical_vars(Set::S65, 15), 14);
        assert_eq!(Group::Proj.logical_vars(Set::S65, 15), 22);

        assert_eq!(Group::Main16.logical_vars(Set::S87, 15), 23);
        assert_eq!(Group::Main.logical_vars(Set::S87, 15), 23);
        assert_eq!(Group::Level1.logical_vars(Set::S87, 15), 20);
        assert_eq!(Group::Level2.logical_vars(Set::S87, 15), 15);
        assert_eq!(Group::Level3.logical_vars(Set::S87, 15), 14);
        assert_eq!(Group::Proj.logical_vars(Set::S87, 15), 23);

        // Level 3 takes G_2's physical layout: one trailing pad variable at every size.
        assert_eq!(Group::Level3.logical_vars(Set::S44, 13), 12);
        assert_eq!(Group::Level3.physical_vars(Set::S44, 13), 13);
        assert_eq!(Group::Level3.physical_vars(Set::S44, 15), 15);
        for set in Set::ALL {
            for d_sig in MIN_D_SIG..=MAX_D_SIG {
                for relation in Relation::ALL.into_iter().filter(|r| r.is_available(set)) {
                    assert_eq!(Group::Level3.layout(set, d_sig, relation), Group::Level2.layout(set, d_sig, relation));
                }
            }
        }
        assert_eq!(Group::Main.physical_vars(Set::S44, 3), 13);
        assert_eq!(Group::Main.physical_vars(Set::S65, 3), 13);
        assert_eq!(Group::Main.physical_vars(Set::S87, 3), 13);
        assert_eq!(Group::Main.physical_vars(Set::S65, 4), 13);
        assert_eq!(Group::Main.physical_vars(Set::S65, 5), 13);
        assert_eq!(Group::Main.physical_vars(Set::S65, 6), 14);
        for group in Group::ALL {
            assert_eq!(Group::ALL[group.index()], group);
            assert_eq!(group.is_final(), group == Group::Mult);
        }
        assert_eq!(
            Group::ALL.map(Group::family),
            [
                Family::Width16,
                Family::Width4,
                Family::Width16,
                Family::Width16,
                Family::Width16,
                Family::Width16,
                Family::Width4
            ]
        );
    }

    #[test]
    fn batch_layouts_for_the_catalog_sizes() {
        let shape = |set, d_sig| -> Vec<(usize, usize)> {
            BatchLayout::new(set, d_sig, Relation::HExact)
                .unwrap()
                .groups
                .iter()
                .map(|group| (group.num_vars(), group.num_polynomials()))
                .collect()
        };
        assert_eq!(
            shape(Set::S44, 8),
            [(18, 4), (18, 9), (13, 6), (13, 12), (13, 12), (15, 4), (16, 7)]
        );
        assert_eq!(
            shape(Set::S65, 8),
            [(16, 23), (16, 46), (13, 4), (13, 8), (13, 8), (15, 5), (16, 7)]
        );
        assert_eq!(
            shape(Set::S87, 8),
            [(16, 31), (16, 62), (13, 4), (13, 8), (13, 8), (16, 5), (16, 7)]
        );
        assert!(BatchLayout::new(Set::S44, 2, Relation::HFips).is_none());
        assert!(BatchLayout::new(Set::S44, 16, Relation::HExact).is_none());
        // H-FIPS of ML-DSA-65 and ML-DSA-87: G_main4 gains three columns per row lane, G_2 and
        // G_3 the hint slack.
        assert_eq!(
            BatchLayout::new(Set::S65, 8, Relation::HFips)
                .unwrap()
                .groups
                .iter()
                .map(|g| (g.num_vars(), g.num_polynomials()))
                .collect::<Vec<_>>(),
            [(16, 23), (16, 64), (13, 4), (13, 9), (13, 9), (15, 5), (16, 7)]
        );
        assert_eq!(BatchLayout::new(Set::S87, 15, Relation::HFips).unwrap().total_polynomials(), 151);
        for set in Set::ALL {
            for relation in Relation::ALL {
                for d_sig in MIN_D_SIG..=MAX_D_SIG {
                    assert!(BatchLayout::new(set, d_sig, relation).unwrap().in_catalog());
                }
            }
        }
        assert_eq!(
            BatchLayout::new(Set::S65, 10, Relation::HExact).unwrap().row_name(),
            "mldsa-65-p16x-batch-1024-mult-final"
        );
        let layout = BatchLayout::new(Set::S44, 10, Relation::HFips).unwrap();
        assert_eq!(layout.total_polynomials(), 59);
        assert_eq!(layout.max_physical_vars(), 20);
        assert_eq!(layout.row_name(), "mldsa-44-p16f-batch-1024-mult-final");
        assert_eq!(layout.precommitted().len(), 6);
        assert_eq!(layout.final_group(), PolynomialGroupLayout::new(16, 7));
        let layout_65 = BatchLayout::new(Set::S65, 10, Relation::HExact).unwrap();
        assert_eq!(layout_65.total_polynomials(), 101);
        let layout_87 = BatchLayout::new(Set::S87, 10, Relation::HExact).unwrap();
        assert_eq!(layout_87.total_polynomials(), 125);
        assert_eq!(layout_87.row_name(), "mldsa-87-p16x-batch-1024-mult-final");
        // H-exact: G_main4 at 9 polynomials, its own row name; H-FIPS adds three G_main4
        // columns and the hint slack in G_2 and G_3, every other group is shared.
        let exact = BatchLayout::new(Set::S44, 15, Relation::HExact).unwrap();
        assert_eq!(
            exact
                .groups
                .iter()
                .map(|group| (group.num_vars(), group.num_polynomials()))
                .collect::<Vec<_>>(),
            [(25, 4), (25, 9), (19, 6), (15, 12), (15, 12), (22, 4), (16, 7)]
        );
        assert_eq!(exact.total_polynomials(), 54);
        assert_eq!(exact.row_name(), "mldsa-44-p16x-batch-32768-mult-final");
        let fips = BatchLayout::new(Set::S44, 15, Relation::HFips).unwrap();
        for group in [Group::Main16, Group::Level1, Group::Proj, Group::Mult] {
            assert_eq!(exact.groups[group.index()], fips.groups[group.index()]);
        }
        assert_eq!(
            catalog_batches().len(),
            CATALOG_D_SIG_H_EXACT.len()
                + CATALOG_D_SIG_H_FIPS.len()
                + catalog_batches_for_set(Set::S65).len()
                + catalog_batches_for_set(Set::S87).len()
        );
        assert_eq!(catalog_batches().len(), 78);
        assert_eq!(catalog_batches_for_set(Set::S65).len(), 26);
        assert_eq!(catalog_batches_for_set(Set::S87).len(), 26);
    }

    #[test]
    fn fp_batch_layouts_and_groups() {
        assert_eq!(HashField::default(), HashField::Gf2);
        for field in HashField::ALL {
            assert_eq!(HashField::parse(field.flag()), Some(field));
            assert_eq!(field.groups().len(), field.group_count());
        }
        assert_eq!(HashField::parse("other"), None);
        for (idx, &group) in Group::ALL_FP.iter().enumerate() {
            assert_eq!(group.index_in(HashField::Fp), idx);
            assert_eq!(group.is_final(), group == Group::Mult);
            if group.is_keccak() {
                assert_eq!(group.round(), 1);
                assert_eq!(group.family(), Family::Width4);
                assert_eq!(group.num_polynomials(Set::S44, Relation::HExact), 55);
                assert_eq!(group.num_polynomials(Set::S65, Relation::HFips), 55);
                assert_eq!(group.num_polynomials(Set::S87, Relation::HExact), 70);
                assert_eq!(group.logical_vars(Set::S44, 10), 20);
                assert_eq!(group.logical_vars(Set::S65, 10), 20);
                assert_eq!(group.logical_vars(Set::S87, 10), 20);
            }
        }
        let fp44 = BatchLayout::new_fp(Set::S44, 10, Relation::HExact).unwrap();
        assert_eq!(fp44.hash_field, HashField::Fp);
        assert_eq!(fp44.groups.len(), GROUP_COUNT_FP);
        assert_eq!(fp44.precommitted().len(), GROUP_COUNT_FP - 1);
        assert_eq!(fp44.row_name(), "mldsa-44-p16x-fp-batch-1024-mult-final");
        // 4 + 9 + 7 * 55 + 4 + 7 = 409 polynomials.
        assert_eq!(fp44.total_polynomials(), 409);
        assert_eq!(fp44.max_physical_vars(), 20);
        assert!(fp44.in_catalog());

        let fp65 = BatchLayout::new_fp(Set::S65, 10, Relation::HExact).unwrap();
        assert_eq!(fp65.row_name(), "mldsa-65-p16x-fp-batch-1024-mult-final");
        assert_eq!(fp65.max_physical_vars(), 20);
        // 23 + 46 + 7 * 55 + 5 + 7 = 466 polynomials.
        assert_eq!(fp65.total_polynomials(), 466);

        // H-FIPS under Fp: u_h is G_k's last polynomial (4 + 12 + 7 * 55 + 5 + 7 = 413).
        let fips44 = BatchLayout::new_fp(Set::S44, 10, Relation::HFips).unwrap();
        assert_eq!(fips44.row_name(), "mldsa-44-p16f-fp-batch-1024-mult-final");
        assert_eq!(fips44.groups[Group::Proj.index_in(HashField::Fp)], PolynomialGroupLayout::new(17, 5));
        assert_eq!(fips44.total_polynomials(), 413);
        assert_eq!(Group::Proj.num_polynomials_in(Set::S44, Relation::HFips, HashField::Gf2), 4);
        assert_eq!(Group::Proj.num_polynomials_in(Set::S44, Relation::HExact, HashField::Fp), 4);

        let fp87 = BatchLayout::new_fp(Set::S87, 10, Relation::HFips).unwrap();
        assert_eq!(fp87.row_name(), "mldsa-87-p16f-fp-batch-1024-mult-final");
        assert_eq!(fp87.max_physical_vars(), 20);
        // 31 + 86 + 7 * 70 + 6 + 7 = 620 polynomials.
        assert_eq!(fp87.total_polynomials(), 620);
    }

    #[test]
    fn scalar_layouts_are_deduplicated_per_family() {
        let shapes = |relations: &[Relation], family| -> Vec<(usize, usize)> {
            let batches: Vec<BatchLayout> = relations
                .iter()
                .flat_map(|&relation| {
                    [8, 10].map(|d_sig| BatchLayout::new(Set::S44, d_sig, relation).unwrap())
                })
                .collect();
            scalar_layouts(&batches, family)
                .iter()
                .map(|layout| (layout.num_vars(), layout.num_polynomials()))
                .collect()
        };
        assert_eq!(shapes(&[Relation::HExact], Family::Width4), vec![(18, 9), (20, 9)]);
        let width16 = vec![(13, 6), (13, 12), (14, 6), (15, 4), (17, 4), (18, 4), (20, 4)];
        assert_eq!(shapes(&[Relation::HExact], Family::Width16), width16);
        // H-FIPS adds the 12-polynomial G_main4 rows and the 13-polynomial G_2 row (u_h).
        assert_eq!(
            shapes(&Relation::ALL, Family::Width4),
            vec![(18, 9), (18, 12), (20, 9), (20, 12)]
        );
        assert_eq!(
            shapes(&Relation::ALL, Family::Width16),
            vec![
                (13, 6),
                (13, 12),
                (13, 13),
                (14, 6),
                (15, 4),
                (17, 4),
                (18, 4),
                (20, 4)
            ]
        );
    }
}

//! Failure reports of the integer checker of relation H-FIPS (`crate::hfips::check_batch`).
//!
//! The checker evaluates every identity of the relation pointwise on the Boolean domain over
//! the integers and records each failing cell here: the identity, the column or group, the
//! index in its domain and an explanation.

use std::collections::BTreeMap;

/// The identities checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Identity {
    /// Record count against the instance.
    Shape,
    /// `bool_b`: b_t (b_t - 1) = 0.
    BoolB,
    /// `range_w1`: b5 (b4 + b3 b2) = 0.
    RangeW1,
    /// `ring`: Delta = 0 at every position.
    Ring,
    /// Commitment hash: c~ = SHAKE256(mu || w1Encode(w1'), 32).
    CommitmentHash,
    /// Booleans B_zhi, a_z and L_hi.
    BoolSlack,
    /// sigma^2 = 1.
    SignSquare,
    /// sum delta^2 + u_h = omega.
    HintWeight,
    /// A looked-up value (L_lo, B_zlo, u_h, the two halves of v) outside its table.
    Range16,
    /// delta (delta - sigma) = 0.
    DeltaSign,
    /// e (e - 1)(e - delta) = 0.
    EdgeE,
    /// e (e + 1) w1' = 0 (e = 1 only at w1' = 0).
    WrapLow,
    /// e (e - 1)(w1' - 43) = 0 (e = -1 only at w1' = 43).
    WrapHigh,
}

/// One failing check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// Identity.
    pub identity: Identity,
    /// Column or group name.
    pub what: String,
    /// Index in the column's domain (main index or record).
    pub index: usize,
    /// Explanation.
    pub detail: String,
}

/// Result of a check: every failing identity with its count and the first failures.
#[derive(Clone, Debug, Default)]
pub struct CheckReport {
    /// First failures (at most `MAX_DETAILS`).
    pub failures: Vec<Failure>,
    /// Failing cells per identity.
    pub counts: BTreeMap<Identity, usize>,
}

const MAX_DETAILS: usize = 64;

impl CheckReport {
    pub(crate) fn fail(&mut self, identity: Identity, what: &str, index: usize, detail: String) {
        *self.counts.entry(identity).or_insert(0) += 1;
        if self.failures.len() < MAX_DETAILS {
            self.failures.push(Failure {
                identity,
                what: what.to_string(),
                index,
                detail,
            });
        }
    }

    /// No failure at all.
    pub fn is_ok(&self) -> bool {
        self.counts.is_empty()
    }

    /// Whether `id` failed somewhere.
    pub fn failed(&self, id: Identity) -> bool {
        self.counts.contains_key(&id)
    }
}

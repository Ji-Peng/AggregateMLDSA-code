//! Public instance: the verifier's per-record precomputation.
//!
//! For live record i: the expanded key (A_hat_i, A_i, tr_i, shared between records with the same
//! key), mu_i = H(tr_i || 0x00 || 0x00 || M_i, 64), c_i = SampleInBall(c~_i) and
//! T_i = 2^d (c_i * t1_i) over Z. The public main-domain table T(j, rho, i) is zero on padded rows.

use crate::expand::{public_t, ExpandedKey};
use crate::native::{mu, sample_in_ball, Challenge, VecK};
use crate::padding::{main_index, Geometry, GeometryError};
use crate::params::{C_TILDE_BYTES, K, N, TR_BYTES};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Public data of one live record.
#[derive(Clone, Debug)]
pub struct PublicRecord {
    /// Expanded public key.
    pub key: Arc<ExpandedKey>,
    /// Message M.
    pub msg: Vec<u8>,
    /// c~ from the aggregate.
    pub c_tilde: [u8; C_TILDE_BYTES],
    /// mu.
    pub mu: [u8; TR_BYTES],
    /// c = SampleInBall(c~).
    pub c: Challenge,
    /// T = 2^d (c * t1) over Z.
    pub t: VecK,
}

/// Errors building the instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstanceError {
    /// Bad batch size.
    Geometry(GeometryError),
    /// Public key of record i has the wrong length.
    PublicKey(usize),
}

/// The public instance of a batch.
#[derive(Clone, Debug)]
pub struct PublicInstance {
    /// Batch geometry.
    pub geom: Geometry,
    /// Live records, in order.
    pub records: Vec<PublicRecord>,
}

impl PublicInstance {
    /// Builds the instance from (pk, M, c~) triples; keys are expanded once per distinct pk.
    pub fn new(inputs: &[(&[u8], &[u8], [u8; C_TILDE_BYTES])]) -> Result<Self, InstanceError> {
        let geom = Geometry::new(inputs.len()).map_err(InstanceError::Geometry)?;
        let mut distinct: Vec<&[u8]> = Vec::new();
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        for (pk, _, _) in inputs {
            index.entry(pk).or_insert_with(|| {
                distinct.push(pk);
                distinct.len() - 1
            });
        }
        let keys: Vec<Option<Arc<ExpandedKey>>> = distinct
            .par_iter()
            .map(|pk| ExpandedKey::from_bytes(pk).map(Arc::new))
            .collect();
        let records = inputs
            .par_iter()
            .enumerate()
            .map(|(i, (pk, msg, c_tilde))| {
                let key = keys[index[pk]].clone().ok_or(InstanceError::PublicKey(i))?;
                let mu = mu(&key.tr, msg, &[]);
                let c = sample_in_ball(c_tilde);
                let t = public_t(&c, &key.t1);
                Ok(PublicRecord {
                    key,
                    msg: msg.to_vec(),
                    c_tilde: *c_tilde,
                    mu,
                    c,
                    t,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { geom, records })
    }

    /// Builds the instance from records with signatures (c~ = the first 32 signature bytes).
    pub fn from_records(records: &[crate::gen::Record]) -> Result<Self, InstanceError> {
        let inputs: Vec<(&[u8], &[u8], [u8; C_TILDE_BYTES])> = records
            .iter()
            .map(|r| {
                let mut ct = [0u8; C_TILDE_BYTES];
                let n = r.sig.len().min(C_TILDE_BYTES);
                ct[..n].copy_from_slice(&r.sig[..n]);
                (r.pk.as_slice(), r.msg.as_slice(), ct)
            })
            .collect();
        Self::new(&inputs)
    }

    /// The public table T(j, rho, i) on the main domain, zero on padded rows.
    pub fn t_table(&self) -> Vec<i64> {
        let mut out = vec![0i64; self.geom.main_len()];
        for (i, r) in self.records.iter().enumerate() {
            for rho in 0..K {
                for j in 0..N {
                    out[main_index(j, rho, i)] = r.t[rho][j];
                }
            }
        }
        out
    }
}

//! Public instance: the verifier's per-record precomputation.
//!
//! For live record i: the expanded key (A_hat_i, A_i, tr_i, shared between records with the same
//! key), mu_i = H(tr_i || 0x00 || 0x00 || M_i, 64), c_i = SampleInBall(c~_i) and
//! T_i = 2^d (c_i * t1_i) over Z. The public main-domain table T(j, rho, i) is zero on padded rows.

use crate::expand::{public_t_into, ExpandedKey};
use crate::native::{mu, sample_in_ball, Challenge};
use crate::padding::{main_index, Geometry, GeometryError};
use crate::params::{C_TILDE_BYTES, K, N, TR_BYTES};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// One record's `T = 2^d (c * t1)` (|T| < 2^29, stored as `i32`) inside the batch's single
/// flat allocation (`k 256` values per record, record order). The records share the
/// allocation; a mutable borrow copies it first (`Arc::make_mut`), so a tampered clone of an
/// instance never changes the original.
#[derive(Clone, Debug)]
pub struct TView {
    data: Arc<Vec<i32>>,
    index: usize,
}

impl TView {
    /// Record `index` of a flat `T` buffer of whole records.
    ///
    /// # Panics
    /// If the buffer holds no record `index`.
    #[must_use]
    pub fn new(data: Arc<Vec<i32>>, index: usize) -> Self {
        assert!(index < data.len() / (K * N), "record inside the T buffer");
        Self { data, index }
    }
}

impl std::ops::Deref for TView {
    type Target = [[i32; N]; K];
    fn deref(&self) -> &Self::Target {
        let record = &self.data[self.index * K * N..(self.index + 1) * K * N];
        let (rows, _) = record.as_chunks::<N>();
        rows.try_into().expect("k rows of 256")
    }
}

impl std::ops::DerefMut for TView {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let index = self.index;
        let data = Arc::make_mut(&mut self.data);
        let record = &mut data[index * K * N..(index + 1) * K * N];
        let (rows, _) = record.as_chunks_mut::<N>();
        rows.try_into().expect("k rows of 256")
    }
}

impl PartialEq for TView {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl Eq for TView {}

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
    pub t: TView,
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
        // The first record whose key does not decode, as the serial order finds it.
        if let Some(i) = inputs.iter().position(|(pk, _, _)| keys[index[pk]].is_none()) {
            return Err(InstanceError::PublicKey(i));
        }
        // Indexed collects write each record in place; the records are small (T lives in
        // one flat buffer below), so no per-thread chunk is grown and concatenated.
        let partial: Vec<(Arc<ExpandedKey>, [u8; TR_BYTES], Challenge)> = inputs
            .par_iter()
            .map(|(pk, msg, c_tilde)| {
                let key = Arc::clone(keys[index[pk]].as_ref().expect("checked above"));
                let mu = mu(&key.tr, msg, &[]);
                (key, mu, sample_in_ball(c_tilde))
            })
            .collect();
        // One zeroed allocation (fresh pages, first touched by the parallel writers).
        let mut t = vec![0_i32; inputs.len() * K * N];
        t.par_chunks_exact_mut(K * N)
            .zip(&partial)
            .for_each(|(out, (key, _, c))| {
                let (rows, _) = out.as_chunks_mut::<N>();
                public_t_into(c, &key.t1, rows.try_into().expect("k rows of 256"));
            });
        let t = Arc::new(t);
        let records = partial
            .into_par_iter()
            .zip(inputs)
            .enumerate()
            .map(|(i, ((key, mu, c), (_, msg, c_tilde)))| PublicRecord {
                key,
                msg: msg.to_vec(),
                c_tilde: *c_tilde,
                mu,
                c,
                t: TView::new(Arc::clone(&t), i),
            })
            .collect();
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
                    out[main_index(j, rho, i)] = i64::from(r.t[rho][j]);
                }
            }
        }
        out
    }
}

//! A persistent cache of expanded public keys for the verifier.
//!
//! The verifier's "cached" timing assumes `A_i = ctr_q(NTT^{-1}(ExpandA(rho_i)))` and `tr_i`
//! are known for every key; this module stores them across runs. The file holds one entry
//! per key, addressed by the public-key hash `SHA3-256(pk)`:
//!
//! `magic "AGGKCACH" || version u8 || count LE32 || (hash[32] || pk[1312] || tr[64] ||
//! A[k][l][256] as LE32)*`
//!
//! On load every entry is checked: the hash must be `SHA3-256(pk)` and `pk` must decode;
//! `t1` is recomputed by `pkDecode` (unpacking only). A corrupted cache can only make the
//! verifier reject (its `A` and `tr` enter the checks as public data), never accept a false
//! statement for the keys given; still, a cache file is trusted local state, like a key store.
//! In memory the entries live in `agg_piop::public_data::KeyCache`.

use std::path::Path;
use std::sync::Arc;

use agg_piop::public_data::{KeyCache, VerifierKey};
use agg_witness::expand::MatrixCoeff;
use agg_witness::native::pk_decode;
use agg_witness::params::{K, L, N, PK_BYTES, TR_BYTES};
use sha3::{Digest, Sha3_256};

/// File magic.
pub const MAGIC: &[u8; 8] = b"AGGKCACH";
/// File version.
pub const VERSION: u8 = 1;
/// Bytes of one entry.
pub const ENTRY_BYTES: usize = 32 + PK_BYTES + TR_BYTES + 4 * K * L * N;

/// `SHA3-256(pk)`, the address of a key in the cache file.
#[must_use]
pub fn key_hash(pk: &[u8]) -> [u8; 32] {
    Sha3_256::digest(pk).into()
}

/// Serialize the entries of `cache`, sorted by key hash.
#[must_use]
pub fn to_bytes(cache: &KeyCache) -> Vec<u8> {
    let mut entries: Vec<(&Arc<VerifierKey>, [u8; 32])> =
        cache.entries().map(|k| (k, key_hash(&k.pk))).collect();
    entries.sort_by_key(|e| e.1);
    let mut out = Vec::with_capacity(13 + entries.len() * ENTRY_BYTES);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, hash) in entries {
        out.extend_from_slice(&hash);
        out.extend_from_slice(&key.pk);
        out.extend_from_slice(&key.tr);
        for row in key.a.iter() {
            for poly in row {
                for &c in poly {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
    }
    out
}

/// A malformed cache file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheError {
    /// Wrong magic or version.
    Header,
    /// Wrong length.
    Length,
    /// Entry `i` has a hash other than `SHA3-256(pk)` or an undecodable key.
    Entry(usize),
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "key cache: {self:?}")
    }
}

impl std::error::Error for CacheError {}

/// Parse a cache file into a [`KeyCache`].
///
/// # Errors
/// On a malformed file or entry.
pub fn from_bytes(bytes: &[u8]) -> Result<KeyCache, CacheError> {
    if bytes.len() < 13 || &bytes[..8] != MAGIC || bytes[8] != VERSION {
        return Err(CacheError::Header);
    }
    let count = u32::from_le_bytes(bytes[9..13].try_into().expect("4 bytes")) as usize;
    let body = &bytes[13..];
    if count.checked_mul(ENTRY_BYTES) != Some(body.len()) {
        return Err(CacheError::Length);
    }
    let mut cache = KeyCache::default();
    for (i, entry) in body.as_chunks::<ENTRY_BYTES>().0.iter().enumerate() {
        let (hash, rest) = entry.split_at(32);
        let (pk, rest) = rest.split_at(PK_BYTES);
        let (tr, a_bytes) = rest.split_at(TR_BYTES);
        if hash != key_hash(pk) {
            return Err(CacheError::Entry(i));
        }
        let decoded = pk_decode(pk).ok_or(CacheError::Entry(i))?;
        let mut a: Box<MatrixCoeff> = Box::new([[[0_i32; N]; L]; K]);
        for (c, word) in a
            .iter_mut()
            .flatten()
            .flatten()
            .zip(a_bytes.as_chunks::<4>().0)
        {
            *c = i32::from_le_bytes(*word);
        }
        cache.insert(Arc::new(VerifierKey {
            pk: pk.to_vec(),
            a,
            t1: decoded.t1,
            tr: tr.try_into().expect("64 bytes"),
        }));
    }
    Ok(cache)
}

/// Load a cache file; a missing file is an empty cache.
///
/// # Errors
/// On an unreadable or malformed file.
pub fn load(path: &Path) -> Result<KeyCache, Box<dyn std::error::Error>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(from_bytes(&bytes)?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(KeyCache::default()),
        Err(e) => Err(e.into()),
    }
}

/// Write a cache file.
///
/// # Errors
/// On a write failure.
pub fn save(path: &Path, cache: &KeyCache) -> std::io::Result<()> {
    std::fs::write(path, to_bytes(cache))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agg_witness::gen;

    #[test]
    fn cache_files_round_trip_and_reject_corruption() {
        let records = gen::records_shared_keys(5, 4, 2);
        let mut cache = KeyCache::default();
        for r in &records {
            cache.insert(Arc::new(VerifierKey::from_bytes(&r.pk).unwrap()));
        }
        assert_eq!(cache.len(), 2);
        let bytes = to_bytes(&cache);
        assert_eq!(bytes.len(), 13 + 2 * ENTRY_BYTES);
        let loaded = from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);
        for r in &records {
            assert_eq!(loaded.get(&r.pk), cache.get(&r.pk));
        }
        assert_eq!(to_bytes(&loaded), bytes, "deterministic order");
        let mut bad_hash = bytes.clone();
        bad_hash[13] ^= 1;
        assert_eq!(from_bytes(&bad_hash).unwrap_err(), CacheError::Entry(0));
        let mut bad_pk = bytes.clone();
        bad_pk[13 + ENTRY_BYTES + 40] ^= 1;
        assert_eq!(from_bytes(&bad_pk).unwrap_err(), CacheError::Entry(1));
        assert_eq!(
            from_bytes(&bytes[..bytes.len() - 1]).unwrap_err(),
            CacheError::Length
        );
        assert_eq!(from_bytes(b"AGGKCACX").unwrap_err(), CacheError::Header);
        let empty = to_bytes(&KeyCache::default());
        assert!(from_bytes(&empty).unwrap().is_empty());
    }
}

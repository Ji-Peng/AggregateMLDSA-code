//! A persistent cache of expanded public keys for the verifier.
//!
//! The verifier's "cached" timing assumes `A_i = ctr_q(NTT^{-1}(ExpandA(rho_i)))` and `tr_i`
//! are known for every key; this module stores them across runs. The file holds the keys of
//! one parameter set, one entry per key, addressed by the public-key hash `SHA3-256(pk)`:
//!
//! `magic "AGGKCACH" || version 2 || set id u8 (0x44, 0x65, 0x87) || count LE32 ||
//! (hash[32] || pk || tr[64] || A[k][l][256] as LE32)*`
//!
//! with the `pk`, `k` and `l` of the set (FIPS 204 Tables 1 and 2). Version 1 files (no set
//! byte, ML-DSA-44 keys) are still read.
//!
//! On load every entry is checked: the hash must be `SHA3-256(pk)`, `pk` must decode, `tr`
//! must equal `H(pk, 64)` and every word of `A` must lie in `[-(q-1)/2, (q-1)/2]`; `t1` is
//! recomputed by `pkDecode` (unpacking only). `A` itself is not re-derived from `rho`, since
//! re-expanding it would defeat the cache: the cache file is trusted local state, like a key
//! store, and a planted `A` breaks soundness for that key.
//! In memory the entries live in `agg_piop::public_data::KeyCache` (ML-DSA-44) or
//! `agg_piop::lane_public::LaneKeyCache` (ML-DSA-65 and ML-DSA-87).

use std::path::Path;
use std::sync::Arc;

use agg_piop::lane_public::{LaneKey, LaneKeyCache};
use agg_piop::public_data::{KeyCache, VerifierKey};
use agg_witness::expand::MatrixCoeff;
use agg_witness::native::{pk_decode, pk_decode_for, shake256};
use agg_witness::params::{dispatch_set, ParamSet, Set, K, L, N, PK_BYTES, Q, TR_BYTES};
use sha3::{Digest, Sha3_256};

use crate::mode_a::KeyCacheMut;

/// File magic.
pub const MAGIC: &[u8; 8] = b"AGGKCACH";
/// File version (with the parameter-set byte).
pub const VERSION: u8 = 2;
/// The version without a set byte (ML-DSA-44 keys only), still read.
pub const VERSION_44: u8 = 1;
/// Bytes of one ML-DSA-44 entry.
pub const ENTRY_BYTES: usize = entry_bytes(Set::S44);

/// Bytes of one entry of `set`: hash, `pk`, `tr` and `A` (`k l 256` words).
#[must_use]
pub const fn entry_bytes(set: Set) -> usize {
    dispatch_set!(set, |P| 32 + P::PK_BYTES + TR_BYTES + 4 * P::K * P::L * N)
}

/// `SHA3-256(pk)`, the address of a key in the cache file.
#[must_use]
pub fn key_hash(pk: &[u8]) -> [u8; 32] {
    Sha3_256::digest(pk).into()
}

/// The expanded keys of one parameter set.
#[derive(Clone, Debug)]
pub enum KeyStore {
    /// ML-DSA-44.
    Rows(KeyCache),
    /// ML-DSA-65 or ML-DSA-87.
    Lanes(LaneKeyCache),
}

impl KeyStore {
    /// An empty store for `set`.
    #[must_use]
    pub fn new(set: Set) -> Self {
        match set {
            Set::S44 => Self::Rows(KeyCache::default()),
            other => Self::Lanes(LaneKeyCache::new(other)),
        }
    }

    /// The parameter set.
    #[must_use]
    pub fn set(&self) -> Set {
        match self {
            Self::Rows(_) => Set::S44,
            Self::Lanes(cache) => cache.set(),
        }
    }

    /// Cached keys.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Rows(cache) => cache.len(),
            Self::Lanes(cache) => cache.len(),
        }
    }

    /// True when no key is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The store as the verifier's cache argument.
    pub fn as_mut(&mut self) -> KeyCacheMut<'_> {
        match self {
            Self::Rows(cache) => KeyCacheMut::Rows(cache),
            Self::Lanes(cache) => KeyCacheMut::Lanes(cache),
        }
    }
}

/// One entry: the key, `tr` and the centered `A` in `(rho, s, j)` order.
struct Entry<'a> {
    pk: &'a [u8],
    tr: &'a [u8; TR_BYTES],
    a: Vec<i32>,
}

fn entries(store: &KeyStore) -> Vec<Entry<'_>> {
    match store {
        KeyStore::Rows(cache) => cache
            .entries()
            .map(|key| Entry {
                pk: &key.pk,
                tr: &key.tr,
                a: key.a.iter().flatten().flatten().copied().collect(),
            })
            .collect(),
        KeyStore::Lanes(cache) => cache
            .entries()
            .map(|key| Entry {
                pk: &key.pk,
                tr: &key.tr,
                a: key.a.clone(),
            })
            .collect(),
    }
}

/// Serialize the entries of `store` (version 2), sorted by key hash.
#[must_use]
pub fn to_bytes(store: &KeyStore) -> Vec<u8> {
    let set = store.set();
    let mut entries: Vec<(Entry<'_>, [u8; 32])> =
        entries(store).into_iter().map(|e| {
            let hash = key_hash(e.pk);
            (e, hash)
        }).collect();
    entries.sort_by_key(|e| e.1);
    let mut out = Vec::with_capacity(14 + entries.len() * entry_bytes(set));
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(set.id());
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (entry, hash) in entries {
        out.extend_from_slice(&hash);
        out.extend_from_slice(entry.pk);
        out.extend_from_slice(entry.tr);
        for c in entry.a {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// A malformed cache file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheError {
    /// Wrong magic, version or set byte.
    Header,
    /// Wrong length.
    Length,
    /// Entry `i` has a hash other than `SHA3-256(pk)`, an undecodable key, a `tr` other
    /// than `H(pk, 64)` or a word of `A` outside the centered range.
    Entry(usize),
    /// The file holds keys of another parameter set than the one asked for.
    Set {
        /// The set asked for.
        expected: Set,
        /// The file's set.
        found: Set,
    },
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Set { expected, found } => write!(
                f,
                "key cache: ML-DSA-{} keys, not ML-DSA-{}",
                found.name(),
                expected.name()
            ),
            other => write!(f, "key cache: {other:?}"),
        }
    }
}

impl std::error::Error for CacheError {}

/// The words of `A` from little-endian bytes.
fn words(bytes: &[u8]) -> impl Iterator<Item = i32> + '_ {
    bytes.as_chunks::<4>().0.iter().map(|w| i32::from_le_bytes(*w))
}

/// Check the parts of an entry that are cheap to recompute: `tr = H(pk, 64)` and the
/// centered range `|w| <= (q-1)/2` of every word of `A`.
fn check_entry(pk: &[u8], tr: &[u8], a_bytes: &[u8], i: usize) -> Result<(), CacheError> {
    let mut expected = [0u8; TR_BYTES];
    shake256(&[pk], &mut expected);
    let half = ((Q - 1) / 2) as i32;
    if tr != expected || words(a_bytes).any(|w| !(-half..=half).contains(&w)) {
        return Err(CacheError::Entry(i));
    }
    Ok(())
}

/// The ML-DSA-44 entries of a body.
fn rows_from(body: &[u8]) -> Result<KeyCache, CacheError> {
    let mut cache = KeyCache::default();
    for (i, entry) in body.as_chunks::<ENTRY_BYTES>().0.iter().enumerate() {
        let (hash, rest) = entry.split_at(32);
        let (pk, rest) = rest.split_at(PK_BYTES);
        let (tr, a_bytes) = rest.split_at(TR_BYTES);
        if hash != key_hash(pk) {
            return Err(CacheError::Entry(i));
        }
        let decoded = pk_decode(pk).ok_or(CacheError::Entry(i))?;
        check_entry(pk, tr, a_bytes, i)?;
        let mut a: Box<MatrixCoeff> = Box::new([[[0_i32; N]; L]; K]);
        for (c, word) in a.iter_mut().flatten().flatten().zip(words(a_bytes)) {
            *c = word;
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

/// The entries of a body of a set with polynomial lanes.
fn lanes_from<P: ParamSet>(body: &[u8]) -> Result<LaneKeyCache, CacheError> {
    let mut cache = LaneKeyCache::new(P::SET);
    for (i, entry) in body.chunks_exact(entry_bytes(P::SET)).enumerate() {
        let (hash, rest) = entry.split_at(32);
        let (pk, rest) = rest.split_at(P::PK_BYTES);
        let (tr, a_bytes) = rest.split_at(TR_BYTES);
        if hash != key_hash(pk) {
            return Err(CacheError::Entry(i));
        }
        let decoded = pk_decode_for::<P>(pk).ok_or(CacheError::Entry(i))?;
        check_entry(pk, tr, a_bytes, i)?;
        cache.insert(Arc::new(LaneKey {
            set: P::SET,
            pk: pk.to_vec(),
            tr: tr.try_into().expect("64 bytes"),
            t1: (0..P::K).map(|rho| decoded.t1[rho]).collect(),
            a: words(a_bytes).collect(),
        }));
    }
    Ok(cache)
}

/// Parse a cache file (version 2, or version 1 of ML-DSA-44).
///
/// # Errors
/// On a malformed file or entry.
pub fn from_bytes(bytes: &[u8]) -> Result<KeyStore, CacheError> {
    if bytes.len() < 9 || &bytes[..8] != MAGIC {
        return Err(CacheError::Header);
    }
    let (set, at) = match bytes[8] {
        VERSION_44 => (Set::S44, 9),
        VERSION => {
            let id = *bytes.get(9).ok_or(CacheError::Header)?;
            (Set::from_id(id).ok_or(CacheError::Header)?, 10)
        }
        _ => return Err(CacheError::Header),
    };
    let count_bytes = bytes.get(at..at + 4).ok_or(CacheError::Header)?;
    let count = u32::from_le_bytes(count_bytes.try_into().expect("4 bytes")) as usize;
    let body = &bytes[at + 4..];
    if count.checked_mul(entry_bytes(set)) != Some(body.len()) {
        return Err(CacheError::Length);
    }
    Ok(match set {
        Set::S44 => KeyStore::Rows(rows_from(body)?),
        Set::S65 => KeyStore::Lanes(lanes_from::<agg_witness::params::Set65>(body)?),
        Set::S87 => KeyStore::Lanes(lanes_from::<agg_witness::params::Set87>(body)?),
    })
}

/// Load the cache file of `set`; a missing file is an empty cache.
///
/// # Errors
/// On an unreadable or malformed file, or a file of another set.
pub fn load(path: &Path, set: Set) -> Result<KeyStore, Box<dyn std::error::Error>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let store = from_bytes(&bytes)?;
            if store.set() != set {
                return Err(CacheError::Set {
                    expected: set,
                    found: store.set(),
                }
                .into());
            }
            Ok(store)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(KeyStore::new(set)),
        Err(e) => Err(e.into()),
    }
}

/// Write a cache file.
///
/// # Errors
/// On a write failure.
pub fn save(path: &Path, store: &KeyStore) -> std::io::Result<()> {
    std::fs::write(path, to_bytes(store))
}

#[cfg(test)]
mod tests {
    use super::*;
    use agg_witness::expand::ExpandedKey;
    use agg_witness::gen;
    use agg_witness::params::Set65;

    #[test]
    fn cache_files_round_trip_and_reject_corruption() {
        let records = gen::records_shared_keys(5, 4, 2);
        let mut cache = KeyCache::default();
        for r in &records {
            cache.insert(Arc::new(VerifierKey::from_bytes(&r.pk).unwrap()));
        }
        assert_eq!(cache.len(), 2);
        let store = KeyStore::Rows(cache.clone());
        let bytes = to_bytes(&store);
        assert_eq!(bytes.len(), 14 + 2 * ENTRY_BYTES);
        assert_eq!(bytes[8..10], [VERSION, 0x44]);
        let KeyStore::Rows(loaded) = from_bytes(&bytes).unwrap() else {
            panic!("ML-DSA-44 keys");
        };
        assert_eq!(loaded.len(), 2);
        for r in &records {
            assert_eq!(loaded.get(&r.pk), cache.get(&r.pk));
        }
        assert_eq!(to_bytes(&KeyStore::Rows(loaded)), bytes, "deterministic order");
        // A version-1 file (no set byte) holds the same ML-DSA-44 entries.
        let mut v1 = bytes[..9].to_vec();
        v1[8] = VERSION_44;
        v1.extend_from_slice(&bytes[10..]);
        assert_eq!(to_bytes(&from_bytes(&v1).unwrap()), bytes);
        let mut bad_hash = bytes.clone();
        bad_hash[14] ^= 1;
        assert_eq!(from_bytes(&bad_hash).unwrap_err(), CacheError::Entry(0));
        let mut bad_pk = bytes.clone();
        bad_pk[14 + ENTRY_BYTES + 40] ^= 1;
        assert_eq!(from_bytes(&bad_pk).unwrap_err(), CacheError::Entry(1));
        // A planted `tr` or an out-of-range word of `A` is rejected too.
        let a_at = 14 + 32 + PK_BYTES + TR_BYTES;
        let mut bad_tr = bytes.clone();
        bad_tr[a_at - 1] ^= 1;
        assert_eq!(from_bytes(&bad_tr).unwrap_err(), CacheError::Entry(0));
        let mut bad_a = bytes.clone();
        bad_a[a_at..a_at + 4].copy_from_slice(&(((Q - 1) / 2 + 1) as i32).to_le_bytes());
        assert_eq!(from_bytes(&bad_a).unwrap_err(), CacheError::Entry(0));
        assert_eq!(
            from_bytes(&bytes[..bytes.len() - 1]).unwrap_err(),
            CacheError::Length
        );
        let mut other_set = bytes.clone();
        other_set[9] = 0x65;
        assert_eq!(from_bytes(&other_set).unwrap_err(), CacheError::Length);
        let mut unknown_set = bytes.clone();
        unknown_set[9] = 0x66;
        assert_eq!(from_bytes(&unknown_set).unwrap_err(), CacheError::Header);
        assert_eq!(from_bytes(b"AGGKCACX").unwrap_err(), CacheError::Header);
        let empty = to_bytes(&KeyStore::new(Set::S44));
        assert!(from_bytes(&empty).unwrap().is_empty());
    }

    #[test]
    fn lane_caches_round_trip_with_their_set() {
        let records = gen::records_shared_keys_for::<Set65>(6, 3, 2);
        let mut cache = LaneKeyCache::new(Set::S65);
        for r in &records {
            let key = ExpandedKey::<Set65>::from_bytes_for(&r.pk).unwrap();
            cache.insert(Arc::new(LaneKey::from_expanded(&key)));
        }
        assert_eq!(cache.len(), 2);
        let bytes = to_bytes(&KeyStore::Lanes(cache.clone()));
        assert_eq!(bytes.len(), 14 + 2 * entry_bytes(Set::S65));
        assert_eq!(bytes[8..10], [VERSION, 0x65]);
        let KeyStore::Lanes(loaded) = from_bytes(&bytes).unwrap() else {
            panic!("ML-DSA-65 keys");
        };
        assert_eq!(loaded.set(), Set::S65);
        for r in &records {
            assert_eq!(loaded.get(&r.pk), cache.get(&r.pk));
        }
        let mut bad_pk = bytes.clone();
        bad_pk[14 + 40] ^= 1;
        assert_eq!(from_bytes(&bad_pk).unwrap_err(), CacheError::Entry(0));
        let a_at = 14 + 32 + Set65::PK_BYTES + TR_BYTES;
        let mut bad_a = bytes.clone();
        bad_a[a_at..a_at + 4].copy_from_slice(&(-((Q - 1) / 2) as i32 - 1).to_le_bytes());
        assert_eq!(from_bytes(&bad_a).unwrap_err(), CacheError::Entry(0));
        let empty = to_bytes(&KeyStore::new(Set::S87));
        assert_eq!(from_bytes(&empty).unwrap().set(), Set::S87);
    }
}

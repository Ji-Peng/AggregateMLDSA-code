//! Batch files of the `agg` tool: key pairs, signed records, aggregates.
//!
//! - Keys (`keygen-batch`): `"AGGKEYS1" || count LE32 || (seed[32] || LE32 len || pk)*`. The seed
//!   is the FIPS 204 `KeyGen` seed `xi`; this is a test and benchmark tool, so the file holds
//!   secret material and must be treated as such.
//! - Records (`sign-batch`, input of `aggregate` and `verify`): `"AGGRECS1" || count LE32 ||
//!   (LE32 len || pk, LE32 len || M, LE32 len || signature)*`. The verifier reads only `(pk, M)`
//!   and never touches the signatures.
//! - Aggregates (`aggregate`, input of `verify`): the wire bytes of `agg_piop::wire`, which
//!   start with the header and the `c~_i` list.

use std::path::Path;

/// Magic of a key file.
pub const KEYS_MAGIC: &[u8; 8] = b"AGGKEYS1";
/// Magic of a record file.
pub const RECORDS_MAGIC: &[u8; 8] = b"AGGRECS1";

/// A malformed batch file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileError {
    /// Wrong magic.
    Magic,
    /// Truncated or trailing bytes, or a field of the wrong length.
    Format(&'static str),
}

impl std::fmt::Display for FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "batch file: {self:?}")
    }
}

impl std::error::Error for FileError {}

/// One key pair of a key file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyEntry {
    /// `KeyGen` seed.
    pub seed: [u8; 32],
    /// Encoded public key.
    pub pk: Vec<u8>,
}

/// One signed record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordEntry {
    /// Encoded public key.
    pub pk: Vec<u8>,
    /// Message.
    pub msg: Vec<u8>,
    /// Signature.
    pub sig: Vec<u8>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], FileError> {
        let end = self
            .at
            .checked_add(len)
            .ok_or(FileError::Format("length"))?;
        let out = self
            .bytes
            .get(self.at..end)
            .ok_or(FileError::Format("truncated"))?;
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<usize, FileError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4 bytes")) as usize)
    }

    fn framed(&mut self) -> Result<Vec<u8>, FileError> {
        let len = self.u32()?;
        Ok(self.take(len)?.to_vec())
    }

    fn header(&mut self, magic: &[u8; 8]) -> Result<usize, FileError> {
        if self.take(8)? != magic {
            return Err(FileError::Magic);
        }
        let count = self.u32()?;
        // Every entry has at least 12 bytes: refuse absurd counts before allocating.
        if count > self.bytes.len() / 12 + 1 {
            return Err(FileError::Format("count"));
        }
        Ok(count)
    }

    fn finish(&self) -> Result<(), FileError> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(FileError::Format("trailing bytes"))
        }
    }
}

fn put_framed(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(
        &u32::try_from(bytes.len())
            .expect("field below 4 GiB")
            .to_le_bytes(),
    );
    out.extend_from_slice(bytes);
}

/// Serialize a key file.
#[must_use]
pub fn keys_to_bytes(keys: &[KeyEntry]) -> Vec<u8> {
    let mut out = KEYS_MAGIC.to_vec();
    out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for k in keys {
        out.extend_from_slice(&k.seed);
        put_framed(&mut out, &k.pk);
    }
    out
}

/// Parse a key file.
///
/// # Errors
/// On a malformed file.
pub fn keys_from_bytes(bytes: &[u8]) -> Result<Vec<KeyEntry>, FileError> {
    let mut r = Reader { bytes, at: 0 };
    let count = r.header(KEYS_MAGIC)?;
    let keys = (0..count)
        .map(|_| {
            Ok(KeyEntry {
                seed: r.take(32)?.try_into().expect("32 bytes"),
                pk: r.framed()?,
            })
        })
        .collect::<Result<Vec<_>, FileError>>()?;
    r.finish()?;
    Ok(keys)
}

/// Serialize a record file.
#[must_use]
pub fn records_to_bytes(records: &[RecordEntry]) -> Vec<u8> {
    let mut out = RECORDS_MAGIC.to_vec();
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    for r in records {
        put_framed(&mut out, &r.pk);
        put_framed(&mut out, &r.msg);
        put_framed(&mut out, &r.sig);
    }
    out
}

/// Parse a record file.
///
/// # Errors
/// On a malformed file.
pub fn records_from_bytes(bytes: &[u8]) -> Result<Vec<RecordEntry>, FileError> {
    let mut r = Reader { bytes, at: 0 };
    let count = r.header(RECORDS_MAGIC)?;
    let records = (0..count)
        .map(|_| {
            Ok(RecordEntry {
                pk: r.framed()?,
                msg: r.framed()?,
                sig: r.framed()?,
            })
        })
        .collect::<Result<Vec<_>, FileError>>()?;
    r.finish()?;
    Ok(records)
}

/// Read and parse a record file.
///
/// # Errors
/// On an unreadable or malformed file.
pub fn read_records(path: &Path) -> Result<Vec<RecordEntry>, Box<dyn std::error::Error>> {
    let bytes = crate::read_input(path, None)?;
    records_from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()).into())
}

/// Read and parse a key file.
///
/// # Errors
/// On an unreadable or malformed file.
pub fn read_keys(path: &Path) -> Result<Vec<KeyEntry>, Box<dyn std::error::Error>> {
    let bytes = crate::read_input(path, None)?;
    keys_from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()).into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_files_round_trip_and_reject_malformed_input() {
        let keys = vec![
            KeyEntry {
                seed: [1; 32],
                pk: vec![2; 5],
            },
            KeyEntry {
                seed: [3; 32],
                pk: vec![],
            },
        ];
        let bytes = keys_to_bytes(&keys);
        assert_eq!(keys_from_bytes(&bytes).unwrap(), keys);
        assert!(keys_from_bytes(&bytes[..bytes.len() - 1]).is_err());
        let records = vec![RecordEntry {
            pk: vec![4; 3],
            msg: vec![],
            sig: vec![5; 7],
        }];
        let bytes = records_to_bytes(&records);
        assert_eq!(records_from_bytes(&bytes).unwrap(), records);
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            records_from_bytes(&trailing),
            Err(FileError::Format("trailing bytes"))
        );
        assert_eq!(keys_from_bytes(&bytes), Err(FileError::Magic));
        let mut huge = bytes;
        huge[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(records_from_bytes(&huge), Err(FileError::Format("count")));
    }
}

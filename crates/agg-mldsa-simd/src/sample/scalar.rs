//! FIPS 204 rejection sampling and little-endian bit packing.
use super::RejState;
use crate::{
    params::{GAMMA1, Q},
    poly::{Block, Poly32},
};
pub(crate) fn rej_uniform(block: &Block, st: &mut RejState) -> bool {
    for b in block.bytes[..168].as_chunks::<3>().0 {
        if st.count == 256 {
            break;
        }
        let x = (b[0] as i32 | (b[1] as i32) << 8 | (b[2] as i32) << 16) & 0x7fffff;
        if x < Q {
            st.coeffs[st.count] = x;
            st.count += 1;
        }
    }
    st.count == 256
}
pub(crate) fn unpack_z(z: &[u8; 2304], out: &mut [Poly32; 4]) -> u32 {
    let mut max = 0;
    for (poly, bytes) in out.iter_mut().zip(z.as_chunks::<576>().0.iter()) {
        for (coeffs, b) in poly
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(bytes.as_chunks::<9>().0.iter())
        {
            let lo = u64::from_le_bytes(b[..8].try_into().unwrap());
            let bits = lo as u128 | (b[8] as u128) << 64;
            for (j, x) in coeffs.iter_mut().enumerate() {
                *x = GAMMA1 - ((bits >> (18 * j)) & 0x3ffff) as i32;
                max = max.max(x.unsigned_abs());
            }
        }
    }
    max
}
pub(crate) fn pack_w1(w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
    for (row, bytes) in w1.iter().zip(out.as_chunks_mut::<192>().0.iter_mut()) {
        for (v, b) in row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(bytes.as_chunks_mut::<3>().0.iter_mut())
        {
            let x = v[0] as u32 | (v[1] as u32) << 6 | (v[2] as u32) << 12 | (v[3] as u32) << 18;
            b.copy_from_slice(&x.to_le_bytes()[..3]);
        }
    }
}

pub(crate) fn unpack_z_20(z: &[u8], out: &mut [Poly32]) -> u32 {
    let mut max = 0u32;
    let gamma1 = 1i32 << 19;
    assert_eq!(z.len(), out.len() * 640);
    for (poly, bytes) in out.iter_mut().zip(z.as_chunks::<640>().0.iter()) {
        for (coeffs, b) in poly
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(bytes.as_chunks::<5>().0.iter())
        {
            let c0 = (b[0] as u32) | ((b[1] as u32) << 8) | (((b[2] & 0x0f) as u32) << 16);
            let c1 = ((b[2] >> 4) as u32) | ((b[3] as u32) << 4) | ((b[4] as u32) << 12);
            let z0 = gamma1 - c0 as i32;
            let z1 = gamma1 - c1 as i32;
            coeffs[0] = z0;
            coeffs[1] = z1;
            max = max.max(z0.unsigned_abs()).max(z1.unsigned_abs());
        }
    }
    max
}

pub(crate) fn pack_w1_4(w1: &[[u8; 256]], out: &mut [u8]) {
    assert_eq!(out.len(), w1.len() * 128);
    for (row, bytes) in w1.iter().zip(out.as_chunks_mut::<128>().0.iter_mut()) {
        for (pair, b) in row.as_chunks::<2>().0.iter().zip(bytes.iter_mut()) {
            *b = (pair[0] & 0x0f) | ((pair[1] & 0x0f) << 4);
        }
    }
}

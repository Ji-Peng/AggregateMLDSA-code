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

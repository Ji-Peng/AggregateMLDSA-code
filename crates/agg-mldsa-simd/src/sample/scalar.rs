//! Scalar implementation of samplers and packing routines.

use super::RejState;
use crate::poly::{Block, Poly32};

pub(crate) fn rej_uniform(block: &Block, st: &mut RejState) -> bool {
    let _ = (block, st);
    todo!("T2")
}

pub(crate) fn unpack_z(z: &[u8; 2304], out: &mut [Poly32; 4]) -> u32 {
    let _ = (z, out);
    todo!("T2")
}

pub(crate) fn pack_w1(w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
    let _ = (w1, out);
    todo!("T2")
}

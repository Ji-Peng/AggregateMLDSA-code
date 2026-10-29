//! AVX-512 implementation of samplers and packing routines.

use super::RejState;
use crate::poly::{Block, Poly32};

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn rej_uniform(block: &Block, st: &mut RejState) -> bool {
    super::scalar::rej_uniform(block, st)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn unpack_z(z: &[u8; 2304], out: &mut [Poly32; 4]) -> u32 {
    super::scalar::unpack_z(z, out)
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt")]
pub(crate) unsafe fn pack_w1(w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
    super::scalar::pack_w1(w1, out)
}

//! AVX2 NTT translated from pq-crystals Dilithium (CC0/Apache-2.0).
//! Adjacent coefficients remain in the reference backend order throughout arithmetic.
//! On x86-64 Linux the forward transform, the q inverse, `dot4` and the order maps
//! run the assembly port; elsewhere the equivalent intrinsics kernels run.
// The assembly is SysV ELF; other x86-64 targets use the intrinsics kernels.
#[cfg(target_os = "linux")]
#[path = "avx2_asm.rs"]
mod assembly;
#[path = "avx2_tables.rs"]
mod tables;
use super::{Form, Scale};

use crate::{
    params::{consts, Prime},
    poly::Poly32,
};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
// Forward inputs start at |x| <= 2^22. Each lazy sum/difference has
// |out| <= |low| + |high*z|/2^32 + p/2; the schedule test proves < 4.52q or 4.25P2.
#[cfg(any(test, not(target_os = "linux")))]
macro_rules! forward {($l:expr,$h:expr,$zl:expr,$zlo:expr,$zh:expr,$zho:expr,$p:expr)=>{{
let (vl,vh,vzl,vzlo,vzh,vzho,p)=($l,$h,$zl,$zlo,$zh,$zho,$p);
let v13 = _mm256_mul_epi32(vh, vzl);
let v12 = _mm256_shuffle_epi32::<0xF5>(vh);
let v14 = _mm256_mul_epi32(v12, vzlo);
let vh = _mm256_mul_epi32(vh, vzh);
let v12 = _mm256_mul_epi32(v12, vzho);
let v13 = _mm256_mul_epi32(v13, p);
let v14 = _mm256_mul_epi32(v14, p);
let vh = _mm256_shuffle_epi32::<0xF5>(vh);
let vh = _mm256_blend_epi32::<0xAA>(vh, v12);
let v12 = _mm256_sub_epi32(vl, vh);
let vl = _mm256_add_epi32(vl, vh);
let v13 = _mm256_shuffle_epi32::<0xF5>(v13);
let v13 = _mm256_blend_epi32::<0xAA>(v13, v14);
let vh = _mm256_add_epi32(v12, v13);
let vl = _mm256_sub_epi32(vl, v13);
(vl,vh)
}};}
// Inverse lazy sums and differences have |out| <= |low|+|high|.
// q stays below 256q; P2 reduces fourth-layer sums before further doubling,
// P3 its third- and fifth-layer sums (see `reduce_sums`).
macro_rules! backward {($l:expr,$h:expr,$zl:expr,$zlo:expr,$zh:expr,$zho:expr,$p:expr)=>{{
let (vl,vh,vzl,vzlo,vzh,vzho,p)=($l,$h,$zl,$zlo,$zh,$zho,$p);
let v12 = _mm256_sub_epi32(vh, vl);
let vl = _mm256_add_epi32(vl, vh);
let v13 = _mm256_mul_epi32(v12, vzl);
let vh = _mm256_shuffle_epi32::<0xF5>(v12);
let v14 = _mm256_mul_epi32(vh, vzlo);
let v12 = _mm256_mul_epi32(v12, vzh);
let vh = _mm256_mul_epi32(vh, vzho);
let v13 = _mm256_mul_epi32(v13, p);
let v14 = _mm256_mul_epi32(v14, p);
let v12 = _mm256_sub_epi32(v12, v13);
let vh = _mm256_sub_epi32(vh, v14);
let v12 = _mm256_shuffle_epi32::<0xF5>(v12);
let vh = _mm256_blend_epi32::<0xAA>(v12, vh);
(vl,vh)
}};}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn load(a: &[i32; 8]) -> __m256i {
    // SAFETY: the array contains exactly eight initialized i32 lanes.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn store(a: &mut [i32; 8], v: __m256i) {
    // SAFETY: the array has room for all eight lanes.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

#[target_feature(enable = "avx2,popcnt")]
#[cfg(any(test, not(target_os = "linux")))]
unsafe fn ntt_intrinsics(p: Prime, a: &mut Poly32) {
    let t = std::hint::black_box(tables::table(p, Scale::Mont));
    let v0 = _mm256_set1_epi32(consts(p).p);
    {
        let v1 = _mm256_set1_epi32(t.0[33]);
        let v2 = _mm256_set1_epi32(t.0[329]);
        let v4 = load(a.0[0..8].try_into().unwrap());
        let v5 = load(a.0[32..40].try_into().unwrap());
        let v6 = load(a.0[64..72].try_into().unwrap());
        let v7 = load(a.0[96..104].try_into().unwrap());
        let v8 = load(a.0[128..136].try_into().unwrap());
        let v9 = load(a.0[160..168].try_into().unwrap());
        let v10 = load(a.0[192..200].try_into().unwrap());
        let v11 = load(a.0[224..232].try_into().unwrap());
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = forward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[0..8]).try_into().unwrap(), v4);
        store((&mut a.0[32..40]).try_into().unwrap(), v5);
        store((&mut a.0[64..72]).try_into().unwrap(), v6);
        store((&mut a.0[96..104]).try_into().unwrap(), v7);
        store((&mut a.0[128..136]).try_into().unwrap(), v8);
        store((&mut a.0[160..168]).try_into().unwrap(), v9);
        store((&mut a.0[192..200]).try_into().unwrap(), v10);
        store((&mut a.0[224..232]).try_into().unwrap(), v11);
    }
    {
        let v1 = _mm256_set1_epi32(t.0[33]);
        let v2 = _mm256_set1_epi32(t.0[329]);
        let v4 = load(a.0[8..16].try_into().unwrap());
        let v5 = load(a.0[40..48].try_into().unwrap());
        let v6 = load(a.0[72..80].try_into().unwrap());
        let v7 = load(a.0[104..112].try_into().unwrap());
        let v8 = load(a.0[136..144].try_into().unwrap());
        let v9 = load(a.0[168..176].try_into().unwrap());
        let v10 = load(a.0[200..208].try_into().unwrap());
        let v11 = load(a.0[232..240].try_into().unwrap());
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = forward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[8..16]).try_into().unwrap(), v4);
        store((&mut a.0[40..48]).try_into().unwrap(), v5);
        store((&mut a.0[72..80]).try_into().unwrap(), v6);
        store((&mut a.0[104..112]).try_into().unwrap(), v7);
        store((&mut a.0[136..144]).try_into().unwrap(), v8);
        store((&mut a.0[168..176]).try_into().unwrap(), v9);
        store((&mut a.0[200..208]).try_into().unwrap(), v10);
        store((&mut a.0[232..240]).try_into().unwrap(), v11);
    }
    {
        let v1 = _mm256_set1_epi32(t.0[33]);
        let v2 = _mm256_set1_epi32(t.0[329]);
        let v4 = load(a.0[16..24].try_into().unwrap());
        let v5 = load(a.0[48..56].try_into().unwrap());
        let v6 = load(a.0[80..88].try_into().unwrap());
        let v7 = load(a.0[112..120].try_into().unwrap());
        let v8 = load(a.0[144..152].try_into().unwrap());
        let v9 = load(a.0[176..184].try_into().unwrap());
        let v10 = load(a.0[208..216].try_into().unwrap());
        let v11 = load(a.0[240..248].try_into().unwrap());
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = forward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[16..24]).try_into().unwrap(), v4);
        store((&mut a.0[48..56]).try_into().unwrap(), v5);
        store((&mut a.0[80..88]).try_into().unwrap(), v6);
        store((&mut a.0[112..120]).try_into().unwrap(), v7);
        store((&mut a.0[144..152]).try_into().unwrap(), v8);
        store((&mut a.0[176..184]).try_into().unwrap(), v9);
        store((&mut a.0[208..216]).try_into().unwrap(), v10);
        store((&mut a.0[240..248]).try_into().unwrap(), v11);
    }
    {
        let v1 = _mm256_set1_epi32(t.0[33]);
        let v2 = _mm256_set1_epi32(t.0[329]);
        let v4 = load(a.0[24..32].try_into().unwrap());
        let v5 = load(a.0[56..64].try_into().unwrap());
        let v6 = load(a.0[88..96].try_into().unwrap());
        let v7 = load(a.0[120..128].try_into().unwrap());
        let v8 = load(a.0[152..160].try_into().unwrap());
        let v9 = load(a.0[184..192].try_into().unwrap());
        let v10 = load(a.0[216..224].try_into().unwrap());
        let v11 = load(a.0[248..256].try_into().unwrap());
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = forward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[24..32]).try_into().unwrap(), v4);
        store((&mut a.0[56..64]).try_into().unwrap(), v5);
        store((&mut a.0[88..96]).try_into().unwrap(), v6);
        store((&mut a.0[120..128]).try_into().unwrap(), v7);
        store((&mut a.0[152..160]).try_into().unwrap(), v8);
        store((&mut a.0[184..192]).try_into().unwrap(), v9);
        store((&mut a.0[216..224]).try_into().unwrap(), v10);
        store((&mut a.0[248..256]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[0..8].try_into().unwrap());
        let v5 = load(a.0[8..16].try_into().unwrap());
        let v6 = load(a.0[16..24].try_into().unwrap());
        let v7 = load(a.0[24..32].try_into().unwrap());
        let v8 = load(a.0[32..40].try_into().unwrap());
        let v9 = load(a.0[40..48].try_into().unwrap());
        let v10 = load(a.0[48..56].try_into().unwrap());
        let v11 = load(a.0[56..64].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[36]);
        let v2 = _mm256_set1_epi32(t.0[332]);
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v3 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v5, v9);
        let v9 = _mm256_permute2x128_si256::<0x31>(v5, v9);
        let v5 = _mm256_permute2x128_si256::<0x20>(v6, v10);
        let v10 = _mm256_permute2x128_si256::<0x31>(v6, v10);
        let v6 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = load(t.0[40..48].try_into().unwrap());
        let v2 = load(t.0[336..344].try_into().unwrap());
        let (v3, v5) = forward!(v3, v5, v1, v1, v2, v2, v0);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        let v7 = _mm256_unpacklo_epi64(v3, v5);
        let v5 = _mm256_unpackhi_epi64(v3, v5);
        let v3 = _mm256_unpacklo_epi64(v8, v10);
        let v10 = _mm256_unpackhi_epi64(v8, v10);
        let v8 = _mm256_unpacklo_epi64(v4, v6);
        let v6 = _mm256_unpackhi_epi64(v4, v6);
        let v4 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = load(t.0[72..80].try_into().unwrap());
        let v2 = load(t.0[368..376].try_into().unwrap());
        let (v7, v8) = forward!(v7, v8, v1, v1, v2, v2, v0);
        let (v5, v6) = forward!(v5, v6, v1, v1, v2, v2, v0);
        let (v3, v4) = forward!(v3, v4, v1, v1, v2, v2, v0);
        let (v10, v11) = forward!(v10, v11, v1, v1, v2, v2, v0);
        let v9 = _mm256_shuffle_epi32::<0xA0>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v7, v9);
        let v7 = _mm256_srli_epi64::<32>(v7);
        let v8 = _mm256_blend_epi32::<0xAA>(v7, v8);
        let v7 = _mm256_shuffle_epi32::<0xA0>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v5, v7);
        let v5 = _mm256_srli_epi64::<32>(v5);
        let v6 = _mm256_blend_epi32::<0xAA>(v5, v6);
        let v5 = _mm256_shuffle_epi32::<0xA0>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v3, v5);
        let v3 = _mm256_srli_epi64::<32>(v3);
        let v4 = _mm256_blend_epi32::<0xAA>(v3, v4);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v3 = _mm256_blend_epi32::<0xAA>(v10, v3);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = load(t.0[104..112].try_into().unwrap());
        let v2 = load(t.0[400..408].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v5) = forward!(v9, v5, v1, v10, v2, v15, v0);
        let (v8, v4) = forward!(v8, v4, v1, v10, v2, v15, v0);
        let (v7, v3) = forward!(v7, v3, v1, v10, v2, v15, v0);
        let (v6, v11) = forward!(v6, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[136..144].try_into().unwrap());
        let v2 = load(t.0[432..440].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v7) = forward!(v9, v7, v1, v10, v2, v15, v0);
        let (v8, v6) = forward!(v8, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[168..176].try_into().unwrap());
        let v2 = load(t.0[464..472].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v3) = forward!(v5, v3, v1, v10, v2, v15, v0);
        let (v4, v11) = forward!(v4, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[200..208].try_into().unwrap());
        let v2 = load(t.0[496..504].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v8) = forward!(v9, v8, v1, v10, v2, v15, v0);
        let v1 = load(t.0[232..240].try_into().unwrap());
        let v2 = load(t.0[528..536].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v7, v6) = forward!(v7, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[264..272].try_into().unwrap());
        let v2 = load(t.0[560..568].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v4) = forward!(v5, v4, v1, v10, v2, v15, v0);
        let v1 = load(t.0[296..304].try_into().unwrap());
        let v2 = load(t.0[592..600].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v3, v11) = forward!(v3, v11, v1, v10, v2, v15, v0);
        store((&mut a.0[0..8]).try_into().unwrap(), v9);
        store((&mut a.0[8..16]).try_into().unwrap(), v8);
        store((&mut a.0[16..24]).try_into().unwrap(), v7);
        store((&mut a.0[24..32]).try_into().unwrap(), v6);
        store((&mut a.0[32..40]).try_into().unwrap(), v5);
        store((&mut a.0[40..48]).try_into().unwrap(), v4);
        store((&mut a.0[48..56]).try_into().unwrap(), v3);
        store((&mut a.0[56..64]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[64..72].try_into().unwrap());
        let v5 = load(a.0[72..80].try_into().unwrap());
        let v6 = load(a.0[80..88].try_into().unwrap());
        let v7 = load(a.0[88..96].try_into().unwrap());
        let v8 = load(a.0[96..104].try_into().unwrap());
        let v9 = load(a.0[104..112].try_into().unwrap());
        let v10 = load(a.0[112..120].try_into().unwrap());
        let v11 = load(a.0[120..128].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[37]);
        let v2 = _mm256_set1_epi32(t.0[333]);
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v3 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v5, v9);
        let v9 = _mm256_permute2x128_si256::<0x31>(v5, v9);
        let v5 = _mm256_permute2x128_si256::<0x20>(v6, v10);
        let v10 = _mm256_permute2x128_si256::<0x31>(v6, v10);
        let v6 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = load(t.0[48..56].try_into().unwrap());
        let v2 = load(t.0[344..352].try_into().unwrap());
        let (v3, v5) = forward!(v3, v5, v1, v1, v2, v2, v0);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        let v7 = _mm256_unpacklo_epi64(v3, v5);
        let v5 = _mm256_unpackhi_epi64(v3, v5);
        let v3 = _mm256_unpacklo_epi64(v8, v10);
        let v10 = _mm256_unpackhi_epi64(v8, v10);
        let v8 = _mm256_unpacklo_epi64(v4, v6);
        let v6 = _mm256_unpackhi_epi64(v4, v6);
        let v4 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = load(t.0[80..88].try_into().unwrap());
        let v2 = load(t.0[376..384].try_into().unwrap());
        let (v7, v8) = forward!(v7, v8, v1, v1, v2, v2, v0);
        let (v5, v6) = forward!(v5, v6, v1, v1, v2, v2, v0);
        let (v3, v4) = forward!(v3, v4, v1, v1, v2, v2, v0);
        let (v10, v11) = forward!(v10, v11, v1, v1, v2, v2, v0);
        let v9 = _mm256_shuffle_epi32::<0xA0>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v7, v9);
        let v7 = _mm256_srli_epi64::<32>(v7);
        let v8 = _mm256_blend_epi32::<0xAA>(v7, v8);
        let v7 = _mm256_shuffle_epi32::<0xA0>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v5, v7);
        let v5 = _mm256_srli_epi64::<32>(v5);
        let v6 = _mm256_blend_epi32::<0xAA>(v5, v6);
        let v5 = _mm256_shuffle_epi32::<0xA0>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v3, v5);
        let v3 = _mm256_srli_epi64::<32>(v3);
        let v4 = _mm256_blend_epi32::<0xAA>(v3, v4);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v3 = _mm256_blend_epi32::<0xAA>(v10, v3);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = load(t.0[112..120].try_into().unwrap());
        let v2 = load(t.0[408..416].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v5) = forward!(v9, v5, v1, v10, v2, v15, v0);
        let (v8, v4) = forward!(v8, v4, v1, v10, v2, v15, v0);
        let (v7, v3) = forward!(v7, v3, v1, v10, v2, v15, v0);
        let (v6, v11) = forward!(v6, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[144..152].try_into().unwrap());
        let v2 = load(t.0[440..448].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v7) = forward!(v9, v7, v1, v10, v2, v15, v0);
        let (v8, v6) = forward!(v8, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[176..184].try_into().unwrap());
        let v2 = load(t.0[472..480].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v3) = forward!(v5, v3, v1, v10, v2, v15, v0);
        let (v4, v11) = forward!(v4, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[208..216].try_into().unwrap());
        let v2 = load(t.0[504..512].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v8) = forward!(v9, v8, v1, v10, v2, v15, v0);
        let v1 = load(t.0[240..248].try_into().unwrap());
        let v2 = load(t.0[536..544].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v7, v6) = forward!(v7, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[272..280].try_into().unwrap());
        let v2 = load(t.0[568..576].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v4) = forward!(v5, v4, v1, v10, v2, v15, v0);
        let v1 = load(t.0[304..312].try_into().unwrap());
        let v2 = load(t.0[600..608].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v3, v11) = forward!(v3, v11, v1, v10, v2, v15, v0);
        store((&mut a.0[64..72]).try_into().unwrap(), v9);
        store((&mut a.0[72..80]).try_into().unwrap(), v8);
        store((&mut a.0[80..88]).try_into().unwrap(), v7);
        store((&mut a.0[88..96]).try_into().unwrap(), v6);
        store((&mut a.0[96..104]).try_into().unwrap(), v5);
        store((&mut a.0[104..112]).try_into().unwrap(), v4);
        store((&mut a.0[112..120]).try_into().unwrap(), v3);
        store((&mut a.0[120..128]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[128..136].try_into().unwrap());
        let v5 = load(a.0[136..144].try_into().unwrap());
        let v6 = load(a.0[144..152].try_into().unwrap());
        let v7 = load(a.0[152..160].try_into().unwrap());
        let v8 = load(a.0[160..168].try_into().unwrap());
        let v9 = load(a.0[168..176].try_into().unwrap());
        let v10 = load(a.0[176..184].try_into().unwrap());
        let v11 = load(a.0[184..192].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[38]);
        let v2 = _mm256_set1_epi32(t.0[334]);
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v3 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v5, v9);
        let v9 = _mm256_permute2x128_si256::<0x31>(v5, v9);
        let v5 = _mm256_permute2x128_si256::<0x20>(v6, v10);
        let v10 = _mm256_permute2x128_si256::<0x31>(v6, v10);
        let v6 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = load(t.0[56..64].try_into().unwrap());
        let v2 = load(t.0[352..360].try_into().unwrap());
        let (v3, v5) = forward!(v3, v5, v1, v1, v2, v2, v0);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        let v7 = _mm256_unpacklo_epi64(v3, v5);
        let v5 = _mm256_unpackhi_epi64(v3, v5);
        let v3 = _mm256_unpacklo_epi64(v8, v10);
        let v10 = _mm256_unpackhi_epi64(v8, v10);
        let v8 = _mm256_unpacklo_epi64(v4, v6);
        let v6 = _mm256_unpackhi_epi64(v4, v6);
        let v4 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = load(t.0[88..96].try_into().unwrap());
        let v2 = load(t.0[384..392].try_into().unwrap());
        let (v7, v8) = forward!(v7, v8, v1, v1, v2, v2, v0);
        let (v5, v6) = forward!(v5, v6, v1, v1, v2, v2, v0);
        let (v3, v4) = forward!(v3, v4, v1, v1, v2, v2, v0);
        let (v10, v11) = forward!(v10, v11, v1, v1, v2, v2, v0);
        let v9 = _mm256_shuffle_epi32::<0xA0>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v7, v9);
        let v7 = _mm256_srli_epi64::<32>(v7);
        let v8 = _mm256_blend_epi32::<0xAA>(v7, v8);
        let v7 = _mm256_shuffle_epi32::<0xA0>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v5, v7);
        let v5 = _mm256_srli_epi64::<32>(v5);
        let v6 = _mm256_blend_epi32::<0xAA>(v5, v6);
        let v5 = _mm256_shuffle_epi32::<0xA0>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v3, v5);
        let v3 = _mm256_srli_epi64::<32>(v3);
        let v4 = _mm256_blend_epi32::<0xAA>(v3, v4);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v3 = _mm256_blend_epi32::<0xAA>(v10, v3);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = load(t.0[120..128].try_into().unwrap());
        let v2 = load(t.0[416..424].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v5) = forward!(v9, v5, v1, v10, v2, v15, v0);
        let (v8, v4) = forward!(v8, v4, v1, v10, v2, v15, v0);
        let (v7, v3) = forward!(v7, v3, v1, v10, v2, v15, v0);
        let (v6, v11) = forward!(v6, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[152..160].try_into().unwrap());
        let v2 = load(t.0[448..456].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v7) = forward!(v9, v7, v1, v10, v2, v15, v0);
        let (v8, v6) = forward!(v8, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[184..192].try_into().unwrap());
        let v2 = load(t.0[480..488].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v3) = forward!(v5, v3, v1, v10, v2, v15, v0);
        let (v4, v11) = forward!(v4, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[216..224].try_into().unwrap());
        let v2 = load(t.0[512..520].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v8) = forward!(v9, v8, v1, v10, v2, v15, v0);
        let v1 = load(t.0[248..256].try_into().unwrap());
        let v2 = load(t.0[544..552].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v7, v6) = forward!(v7, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[280..288].try_into().unwrap());
        let v2 = load(t.0[576..584].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v4) = forward!(v5, v4, v1, v10, v2, v15, v0);
        let v1 = load(t.0[312..320].try_into().unwrap());
        let v2 = load(t.0[608..616].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v3, v11) = forward!(v3, v11, v1, v10, v2, v15, v0);
        store((&mut a.0[128..136]).try_into().unwrap(), v9);
        store((&mut a.0[136..144]).try_into().unwrap(), v8);
        store((&mut a.0[144..152]).try_into().unwrap(), v7);
        store((&mut a.0[152..160]).try_into().unwrap(), v6);
        store((&mut a.0[160..168]).try_into().unwrap(), v5);
        store((&mut a.0[168..176]).try_into().unwrap(), v4);
        store((&mut a.0[176..184]).try_into().unwrap(), v3);
        store((&mut a.0[184..192]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[192..200].try_into().unwrap());
        let v5 = load(a.0[200..208].try_into().unwrap());
        let v6 = load(a.0[208..216].try_into().unwrap());
        let v7 = load(a.0[216..224].try_into().unwrap());
        let v8 = load(a.0[224..232].try_into().unwrap());
        let v9 = load(a.0[232..240].try_into().unwrap());
        let v10 = load(a.0[240..248].try_into().unwrap());
        let v11 = load(a.0[248..256].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[39]);
        let v2 = _mm256_set1_epi32(t.0[335]);
        let (v4, v8) = forward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = forward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = forward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = forward!(v7, v11, v1, v1, v2, v2, v0);
        let v3 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v5, v9);
        let v9 = _mm256_permute2x128_si256::<0x31>(v5, v9);
        let v5 = _mm256_permute2x128_si256::<0x20>(v6, v10);
        let v10 = _mm256_permute2x128_si256::<0x31>(v6, v10);
        let v6 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = load(t.0[64..72].try_into().unwrap());
        let v2 = load(t.0[360..368].try_into().unwrap());
        let (v3, v5) = forward!(v3, v5, v1, v1, v2, v2, v0);
        let (v8, v10) = forward!(v8, v10, v1, v1, v2, v2, v0);
        let (v4, v6) = forward!(v4, v6, v1, v1, v2, v2, v0);
        let (v9, v11) = forward!(v9, v11, v1, v1, v2, v2, v0);
        let v7 = _mm256_unpacklo_epi64(v3, v5);
        let v5 = _mm256_unpackhi_epi64(v3, v5);
        let v3 = _mm256_unpacklo_epi64(v8, v10);
        let v10 = _mm256_unpackhi_epi64(v8, v10);
        let v8 = _mm256_unpacklo_epi64(v4, v6);
        let v6 = _mm256_unpackhi_epi64(v4, v6);
        let v4 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = load(t.0[96..104].try_into().unwrap());
        let v2 = load(t.0[392..400].try_into().unwrap());
        let (v7, v8) = forward!(v7, v8, v1, v1, v2, v2, v0);
        let (v5, v6) = forward!(v5, v6, v1, v1, v2, v2, v0);
        let (v3, v4) = forward!(v3, v4, v1, v1, v2, v2, v0);
        let (v10, v11) = forward!(v10, v11, v1, v1, v2, v2, v0);
        let v9 = _mm256_shuffle_epi32::<0xA0>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v7, v9);
        let v7 = _mm256_srli_epi64::<32>(v7);
        let v8 = _mm256_blend_epi32::<0xAA>(v7, v8);
        let v7 = _mm256_shuffle_epi32::<0xA0>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v5, v7);
        let v5 = _mm256_srli_epi64::<32>(v5);
        let v6 = _mm256_blend_epi32::<0xAA>(v5, v6);
        let v5 = _mm256_shuffle_epi32::<0xA0>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v3, v5);
        let v3 = _mm256_srli_epi64::<32>(v3);
        let v4 = _mm256_blend_epi32::<0xAA>(v3, v4);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v3 = _mm256_blend_epi32::<0xAA>(v10, v3);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = load(t.0[128..136].try_into().unwrap());
        let v2 = load(t.0[424..432].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v5) = forward!(v9, v5, v1, v10, v2, v15, v0);
        let (v8, v4) = forward!(v8, v4, v1, v10, v2, v15, v0);
        let (v7, v3) = forward!(v7, v3, v1, v10, v2, v15, v0);
        let (v6, v11) = forward!(v6, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[160..168].try_into().unwrap());
        let v2 = load(t.0[456..464].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v7) = forward!(v9, v7, v1, v10, v2, v15, v0);
        let (v8, v6) = forward!(v8, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[192..200].try_into().unwrap());
        let v2 = load(t.0[488..496].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v3) = forward!(v5, v3, v1, v10, v2, v15, v0);
        let (v4, v11) = forward!(v4, v11, v1, v10, v2, v15, v0);
        let v1 = load(t.0[224..232].try_into().unwrap());
        let v2 = load(t.0[520..528].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v9, v8) = forward!(v9, v8, v1, v10, v2, v15, v0);
        let v1 = load(t.0[256..264].try_into().unwrap());
        let v2 = load(t.0[552..560].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v7, v6) = forward!(v7, v6, v1, v10, v2, v15, v0);
        let v1 = load(t.0[288..296].try_into().unwrap());
        let v2 = load(t.0[584..592].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v5, v4) = forward!(v5, v4, v1, v10, v2, v15, v0);
        let v1 = load(t.0[320..328].try_into().unwrap());
        let v2 = load(t.0[616..624].try_into().unwrap());
        let v10 = _mm256_srli_epi64::<32>(v1);
        let v15 = _mm256_shuffle_epi32::<0xF5>(v2);
        let (v3, v11) = forward!(v3, v11, v1, v10, v2, v15, v0);
        store((&mut a.0[192..200]).try_into().unwrap(), v9);
        store((&mut a.0[200..208]).try_into().unwrap(), v8);
        store((&mut a.0[208..216]).try_into().unwrap(), v7);
        store((&mut a.0[216..224]).try_into().unwrap(), v6);
        store((&mut a.0[224..232]).try_into().unwrap(), v5);
        store((&mut a.0[232..240]).try_into().unwrap(), v4);
        store((&mut a.0[240..248]).try_into().unwrap(), v3);
        store((&mut a.0[248..256]).try_into().unwrap(), v11);
    }
}
#[target_feature(enable = "avx2,popcnt")]
fn inverse(p: Prime, t: &tables::Table, a: &mut Poly32, form: Form) {
    let v0 = _mm256_set1_epi32(consts(p).p);
    {
        let v4 = load(a.0[0..8].try_into().unwrap());
        let v5 = load(a.0[8..16].try_into().unwrap());
        let v6 = load(a.0[16..24].try_into().unwrap());
        let v7 = load(a.0[24..32].try_into().unwrap());
        let v8 = load(a.0[32..40].try_into().unwrap());
        let v9 = load(a.0[40..48].try_into().unwrap());
        let v10 = load(a.0[48..56].try_into().unwrap());
        let v11 = load(a.0[56..64].try_into().unwrap());
        let v3 = inv_twiddle(t, 320, false);
        let v15 = inv_twiddle(t, 616, false);
        let v1 = inv_twiddle(t, 320, true);
        let v2 = inv_twiddle(t, 616, true);
        let (v4, v5) = backward!(v4, v5, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 288, false);
        let v15 = inv_twiddle(t, 584, false);
        let v1 = inv_twiddle(t, 288, true);
        let v2 = inv_twiddle(t, 584, true);
        let (v6, v7) = backward!(v6, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 256, false);
        let v15 = inv_twiddle(t, 552, false);
        let v1 = inv_twiddle(t, 256, true);
        let v2 = inv_twiddle(t, 552, true);
        let (v8, v9) = backward!(v8, v9, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 224, false);
        let v15 = inv_twiddle(t, 520, false);
        let v1 = inv_twiddle(t, 224, true);
        let v2 = inv_twiddle(t, 520, true);
        let (v10, v11) = backward!(v10, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 192, false);
        let v15 = inv_twiddle(t, 488, false);
        let v1 = inv_twiddle(t, 192, true);
        let v2 = inv_twiddle(t, 488, true);
        let (v4, v6) = backward!(v4, v6, v1, v3, v2, v15, v0);
        let (v5, v7) = backward!(v5, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 160, false);
        let v15 = inv_twiddle(t, 456, false);
        let v1 = inv_twiddle(t, 160, true);
        let v2 = inv_twiddle(t, 456, true);
        let (v8, v10) = backward!(v8, v10, v1, v3, v2, v15, v0);
        let (v9, v11) = backward!(v9, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 128, false);
        let v15 = inv_twiddle(t, 424, false);
        let v1 = inv_twiddle(t, 128, true);
        let v2 = inv_twiddle(t, 424, true);
        let (v4, v8) = backward!(v4, v8, v1, v3, v2, v15, v0);
        let (v5, v9) = backward!(v5, v9, v1, v3, v2, v15, v0);
        let (v6, v10) = backward!(v6, v10, v1, v3, v2, v15, v0);
        let (v7, v11) = backward!(v7, v11, v1, v3, v2, v15, v0);
        // P3 third-layer sums (<= 4p) are reduced; see `reduce_sums`.
        let (v4, v5, v6, v7) = reduce_sums(p == Prime::P3, p, v0, v4, v5, v6, v7);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v5);
        let v3 = _mm256_blend_epi32::<0xAA>(v4, v3);
        let v4 = _mm256_srli_epi64::<32>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v4, v5);
        let v4 = _mm256_shuffle_epi32::<0xA0>(v7);
        let v4 = _mm256_blend_epi32::<0xAA>(v6, v4);
        let v6 = _mm256_srli_epi64::<32>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v6, v7);
        let v6 = _mm256_shuffle_epi32::<0xA0>(v9);
        let v6 = _mm256_blend_epi32::<0xAA>(v8, v6);
        let v8 = _mm256_srli_epi64::<32>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v8, v9);
        let v8 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v8 = _mm256_blend_epi32::<0xAA>(v10, v8);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = inv_twiddle(t, 96, false);
        let v2 = inv_twiddle(t, 392, false);
        let (v3, v5) = backward!(v3, v5, v1, v1, v2, v2, v0);
        let (v4, v7) = backward!(v4, v7, v1, v1, v2, v2, v0);
        let (v6, v9) = backward!(v6, v9, v1, v1, v2, v2, v0);
        let (v8, v11) = backward!(v8, v11, v1, v1, v2, v2, v0);
        // P2 fourth-layer sums: at most 8p before, below 0.63p after.
        let (v3, v4, v6, v8) = reduce_sums(p == Prime::P2, p, v0, v3, v4, v6, v8);
        let v10 = _mm256_unpacklo_epi64(v3, v4);
        let v4 = _mm256_unpackhi_epi64(v3, v4);
        let v3 = _mm256_unpacklo_epi64(v6, v8);
        let v8 = _mm256_unpackhi_epi64(v6, v8);
        let v6 = _mm256_unpacklo_epi64(v5, v7);
        let v7 = _mm256_unpackhi_epi64(v5, v7);
        let v5 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = inv_twiddle(t, 64, false);
        let v2 = inv_twiddle(t, 360, false);
        let (v10, v4) = backward!(v10, v4, v1, v1, v2, v2, v0);
        let (v3, v8) = backward!(v3, v8, v1, v1, v2, v2, v0);
        let (v6, v7) = backward!(v6, v7, v1, v1, v2, v2, v0);
        let (v5, v11) = backward!(v5, v11, v1, v1, v2, v2, v0);
        let (v10, v3, v6, v5) = reduce_sums(p == Prime::P3, p, v0, v10, v3, v6, v5);
        let v9 = _mm256_permute2x128_si256::<0x20>(v10, v3);
        let v3 = _mm256_permute2x128_si256::<0x31>(v10, v3);
        let v10 = _mm256_permute2x128_si256::<0x20>(v6, v5);
        let v5 = _mm256_permute2x128_si256::<0x31>(v6, v5);
        let v6 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = _mm256_set1_epi32(t.0[39]);
        let v2 = _mm256_set1_epi32(t.0[335]);
        let (v9, v3) = backward!(v9, v3, v1, v1, v2, v2, v0);
        let (v10, v5) = backward!(v10, v5, v1, v1, v2, v2, v0);
        let (v6, v8) = backward!(v6, v8, v1, v1, v2, v2, v0);
        let (v4, v11) = backward!(v4, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[0..8]).try_into().unwrap(), v9);
        store((&mut a.0[8..16]).try_into().unwrap(), v10);
        store((&mut a.0[16..24]).try_into().unwrap(), v6);
        store((&mut a.0[24..32]).try_into().unwrap(), v4);
        store((&mut a.0[32..40]).try_into().unwrap(), v3);
        store((&mut a.0[40..48]).try_into().unwrap(), v5);
        store((&mut a.0[48..56]).try_into().unwrap(), v8);
        store((&mut a.0[56..64]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[64..72].try_into().unwrap());
        let v5 = load(a.0[72..80].try_into().unwrap());
        let v6 = load(a.0[80..88].try_into().unwrap());
        let v7 = load(a.0[88..96].try_into().unwrap());
        let v8 = load(a.0[96..104].try_into().unwrap());
        let v9 = load(a.0[104..112].try_into().unwrap());
        let v10 = load(a.0[112..120].try_into().unwrap());
        let v11 = load(a.0[120..128].try_into().unwrap());
        let v3 = inv_twiddle(t, 312, false);
        let v15 = inv_twiddle(t, 608, false);
        let v1 = inv_twiddle(t, 312, true);
        let v2 = inv_twiddle(t, 608, true);
        let (v4, v5) = backward!(v4, v5, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 280, false);
        let v15 = inv_twiddle(t, 576, false);
        let v1 = inv_twiddle(t, 280, true);
        let v2 = inv_twiddle(t, 576, true);
        let (v6, v7) = backward!(v6, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 248, false);
        let v15 = inv_twiddle(t, 544, false);
        let v1 = inv_twiddle(t, 248, true);
        let v2 = inv_twiddle(t, 544, true);
        let (v8, v9) = backward!(v8, v9, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 216, false);
        let v15 = inv_twiddle(t, 512, false);
        let v1 = inv_twiddle(t, 216, true);
        let v2 = inv_twiddle(t, 512, true);
        let (v10, v11) = backward!(v10, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 184, false);
        let v15 = inv_twiddle(t, 480, false);
        let v1 = inv_twiddle(t, 184, true);
        let v2 = inv_twiddle(t, 480, true);
        let (v4, v6) = backward!(v4, v6, v1, v3, v2, v15, v0);
        let (v5, v7) = backward!(v5, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 152, false);
        let v15 = inv_twiddle(t, 448, false);
        let v1 = inv_twiddle(t, 152, true);
        let v2 = inv_twiddle(t, 448, true);
        let (v8, v10) = backward!(v8, v10, v1, v3, v2, v15, v0);
        let (v9, v11) = backward!(v9, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 120, false);
        let v15 = inv_twiddle(t, 416, false);
        let v1 = inv_twiddle(t, 120, true);
        let v2 = inv_twiddle(t, 416, true);
        let (v4, v8) = backward!(v4, v8, v1, v3, v2, v15, v0);
        let (v5, v9) = backward!(v5, v9, v1, v3, v2, v15, v0);
        let (v6, v10) = backward!(v6, v10, v1, v3, v2, v15, v0);
        let (v7, v11) = backward!(v7, v11, v1, v3, v2, v15, v0);
        // P3 third-layer sums (<= 4p) are reduced; see `reduce_sums`.
        let (v4, v5, v6, v7) = reduce_sums(p == Prime::P3, p, v0, v4, v5, v6, v7);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v5);
        let v3 = _mm256_blend_epi32::<0xAA>(v4, v3);
        let v4 = _mm256_srli_epi64::<32>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v4, v5);
        let v4 = _mm256_shuffle_epi32::<0xA0>(v7);
        let v4 = _mm256_blend_epi32::<0xAA>(v6, v4);
        let v6 = _mm256_srli_epi64::<32>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v6, v7);
        let v6 = _mm256_shuffle_epi32::<0xA0>(v9);
        let v6 = _mm256_blend_epi32::<0xAA>(v8, v6);
        let v8 = _mm256_srli_epi64::<32>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v8, v9);
        let v8 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v8 = _mm256_blend_epi32::<0xAA>(v10, v8);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = inv_twiddle(t, 88, false);
        let v2 = inv_twiddle(t, 384, false);
        let (v3, v5) = backward!(v3, v5, v1, v1, v2, v2, v0);
        let (v4, v7) = backward!(v4, v7, v1, v1, v2, v2, v0);
        let (v6, v9) = backward!(v6, v9, v1, v1, v2, v2, v0);
        let (v8, v11) = backward!(v8, v11, v1, v1, v2, v2, v0);
        // P2 fourth-layer sums: at most 8p before, below 0.63p after.
        let (v3, v4, v6, v8) = reduce_sums(p == Prime::P2, p, v0, v3, v4, v6, v8);
        let v10 = _mm256_unpacklo_epi64(v3, v4);
        let v4 = _mm256_unpackhi_epi64(v3, v4);
        let v3 = _mm256_unpacklo_epi64(v6, v8);
        let v8 = _mm256_unpackhi_epi64(v6, v8);
        let v6 = _mm256_unpacklo_epi64(v5, v7);
        let v7 = _mm256_unpackhi_epi64(v5, v7);
        let v5 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = inv_twiddle(t, 56, false);
        let v2 = inv_twiddle(t, 352, false);
        let (v10, v4) = backward!(v10, v4, v1, v1, v2, v2, v0);
        let (v3, v8) = backward!(v3, v8, v1, v1, v2, v2, v0);
        let (v6, v7) = backward!(v6, v7, v1, v1, v2, v2, v0);
        let (v5, v11) = backward!(v5, v11, v1, v1, v2, v2, v0);
        let (v10, v3, v6, v5) = reduce_sums(p == Prime::P3, p, v0, v10, v3, v6, v5);
        let v9 = _mm256_permute2x128_si256::<0x20>(v10, v3);
        let v3 = _mm256_permute2x128_si256::<0x31>(v10, v3);
        let v10 = _mm256_permute2x128_si256::<0x20>(v6, v5);
        let v5 = _mm256_permute2x128_si256::<0x31>(v6, v5);
        let v6 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = _mm256_set1_epi32(t.0[38]);
        let v2 = _mm256_set1_epi32(t.0[334]);
        let (v9, v3) = backward!(v9, v3, v1, v1, v2, v2, v0);
        let (v10, v5) = backward!(v10, v5, v1, v1, v2, v2, v0);
        let (v6, v8) = backward!(v6, v8, v1, v1, v2, v2, v0);
        let (v4, v11) = backward!(v4, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[64..72]).try_into().unwrap(), v9);
        store((&mut a.0[72..80]).try_into().unwrap(), v10);
        store((&mut a.0[80..88]).try_into().unwrap(), v6);
        store((&mut a.0[88..96]).try_into().unwrap(), v4);
        store((&mut a.0[96..104]).try_into().unwrap(), v3);
        store((&mut a.0[104..112]).try_into().unwrap(), v5);
        store((&mut a.0[112..120]).try_into().unwrap(), v8);
        store((&mut a.0[120..128]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[128..136].try_into().unwrap());
        let v5 = load(a.0[136..144].try_into().unwrap());
        let v6 = load(a.0[144..152].try_into().unwrap());
        let v7 = load(a.0[152..160].try_into().unwrap());
        let v8 = load(a.0[160..168].try_into().unwrap());
        let v9 = load(a.0[168..176].try_into().unwrap());
        let v10 = load(a.0[176..184].try_into().unwrap());
        let v11 = load(a.0[184..192].try_into().unwrap());
        let v3 = inv_twiddle(t, 304, false);
        let v15 = inv_twiddle(t, 600, false);
        let v1 = inv_twiddle(t, 304, true);
        let v2 = inv_twiddle(t, 600, true);
        let (v4, v5) = backward!(v4, v5, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 272, false);
        let v15 = inv_twiddle(t, 568, false);
        let v1 = inv_twiddle(t, 272, true);
        let v2 = inv_twiddle(t, 568, true);
        let (v6, v7) = backward!(v6, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 240, false);
        let v15 = inv_twiddle(t, 536, false);
        let v1 = inv_twiddle(t, 240, true);
        let v2 = inv_twiddle(t, 536, true);
        let (v8, v9) = backward!(v8, v9, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 208, false);
        let v15 = inv_twiddle(t, 504, false);
        let v1 = inv_twiddle(t, 208, true);
        let v2 = inv_twiddle(t, 504, true);
        let (v10, v11) = backward!(v10, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 176, false);
        let v15 = inv_twiddle(t, 472, false);
        let v1 = inv_twiddle(t, 176, true);
        let v2 = inv_twiddle(t, 472, true);
        let (v4, v6) = backward!(v4, v6, v1, v3, v2, v15, v0);
        let (v5, v7) = backward!(v5, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 144, false);
        let v15 = inv_twiddle(t, 440, false);
        let v1 = inv_twiddle(t, 144, true);
        let v2 = inv_twiddle(t, 440, true);
        let (v8, v10) = backward!(v8, v10, v1, v3, v2, v15, v0);
        let (v9, v11) = backward!(v9, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 112, false);
        let v15 = inv_twiddle(t, 408, false);
        let v1 = inv_twiddle(t, 112, true);
        let v2 = inv_twiddle(t, 408, true);
        let (v4, v8) = backward!(v4, v8, v1, v3, v2, v15, v0);
        let (v5, v9) = backward!(v5, v9, v1, v3, v2, v15, v0);
        let (v6, v10) = backward!(v6, v10, v1, v3, v2, v15, v0);
        let (v7, v11) = backward!(v7, v11, v1, v3, v2, v15, v0);
        // P3 third-layer sums (<= 4p) are reduced; see `reduce_sums`.
        let (v4, v5, v6, v7) = reduce_sums(p == Prime::P3, p, v0, v4, v5, v6, v7);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v5);
        let v3 = _mm256_blend_epi32::<0xAA>(v4, v3);
        let v4 = _mm256_srli_epi64::<32>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v4, v5);
        let v4 = _mm256_shuffle_epi32::<0xA0>(v7);
        let v4 = _mm256_blend_epi32::<0xAA>(v6, v4);
        let v6 = _mm256_srli_epi64::<32>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v6, v7);
        let v6 = _mm256_shuffle_epi32::<0xA0>(v9);
        let v6 = _mm256_blend_epi32::<0xAA>(v8, v6);
        let v8 = _mm256_srli_epi64::<32>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v8, v9);
        let v8 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v8 = _mm256_blend_epi32::<0xAA>(v10, v8);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = inv_twiddle(t, 80, false);
        let v2 = inv_twiddle(t, 376, false);
        let (v3, v5) = backward!(v3, v5, v1, v1, v2, v2, v0);
        let (v4, v7) = backward!(v4, v7, v1, v1, v2, v2, v0);
        let (v6, v9) = backward!(v6, v9, v1, v1, v2, v2, v0);
        let (v8, v11) = backward!(v8, v11, v1, v1, v2, v2, v0);
        // P2 fourth-layer sums: at most 8p before, below 0.63p after.
        let (v3, v4, v6, v8) = reduce_sums(p == Prime::P2, p, v0, v3, v4, v6, v8);
        let v10 = _mm256_unpacklo_epi64(v3, v4);
        let v4 = _mm256_unpackhi_epi64(v3, v4);
        let v3 = _mm256_unpacklo_epi64(v6, v8);
        let v8 = _mm256_unpackhi_epi64(v6, v8);
        let v6 = _mm256_unpacklo_epi64(v5, v7);
        let v7 = _mm256_unpackhi_epi64(v5, v7);
        let v5 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = inv_twiddle(t, 48, false);
        let v2 = inv_twiddle(t, 344, false);
        let (v10, v4) = backward!(v10, v4, v1, v1, v2, v2, v0);
        let (v3, v8) = backward!(v3, v8, v1, v1, v2, v2, v0);
        let (v6, v7) = backward!(v6, v7, v1, v1, v2, v2, v0);
        let (v5, v11) = backward!(v5, v11, v1, v1, v2, v2, v0);
        let (v10, v3, v6, v5) = reduce_sums(p == Prime::P3, p, v0, v10, v3, v6, v5);
        let v9 = _mm256_permute2x128_si256::<0x20>(v10, v3);
        let v3 = _mm256_permute2x128_si256::<0x31>(v10, v3);
        let v10 = _mm256_permute2x128_si256::<0x20>(v6, v5);
        let v5 = _mm256_permute2x128_si256::<0x31>(v6, v5);
        let v6 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = _mm256_set1_epi32(t.0[37]);
        let v2 = _mm256_set1_epi32(t.0[333]);
        let (v9, v3) = backward!(v9, v3, v1, v1, v2, v2, v0);
        let (v10, v5) = backward!(v10, v5, v1, v1, v2, v2, v0);
        let (v6, v8) = backward!(v6, v8, v1, v1, v2, v2, v0);
        let (v4, v11) = backward!(v4, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[128..136]).try_into().unwrap(), v9);
        store((&mut a.0[136..144]).try_into().unwrap(), v10);
        store((&mut a.0[144..152]).try_into().unwrap(), v6);
        store((&mut a.0[152..160]).try_into().unwrap(), v4);
        store((&mut a.0[160..168]).try_into().unwrap(), v3);
        store((&mut a.0[168..176]).try_into().unwrap(), v5);
        store((&mut a.0[176..184]).try_into().unwrap(), v8);
        store((&mut a.0[184..192]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[192..200].try_into().unwrap());
        let v5 = load(a.0[200..208].try_into().unwrap());
        let v6 = load(a.0[208..216].try_into().unwrap());
        let v7 = load(a.0[216..224].try_into().unwrap());
        let v8 = load(a.0[224..232].try_into().unwrap());
        let v9 = load(a.0[232..240].try_into().unwrap());
        let v10 = load(a.0[240..248].try_into().unwrap());
        let v11 = load(a.0[248..256].try_into().unwrap());
        let v3 = inv_twiddle(t, 296, false);
        let v15 = inv_twiddle(t, 592, false);
        let v1 = inv_twiddle(t, 296, true);
        let v2 = inv_twiddle(t, 592, true);
        let (v4, v5) = backward!(v4, v5, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 264, false);
        let v15 = inv_twiddle(t, 560, false);
        let v1 = inv_twiddle(t, 264, true);
        let v2 = inv_twiddle(t, 560, true);
        let (v6, v7) = backward!(v6, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 232, false);
        let v15 = inv_twiddle(t, 528, false);
        let v1 = inv_twiddle(t, 232, true);
        let v2 = inv_twiddle(t, 528, true);
        let (v8, v9) = backward!(v8, v9, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 200, false);
        let v15 = inv_twiddle(t, 496, false);
        let v1 = inv_twiddle(t, 200, true);
        let v2 = inv_twiddle(t, 496, true);
        let (v10, v11) = backward!(v10, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 168, false);
        let v15 = inv_twiddle(t, 464, false);
        let v1 = inv_twiddle(t, 168, true);
        let v2 = inv_twiddle(t, 464, true);
        let (v4, v6) = backward!(v4, v6, v1, v3, v2, v15, v0);
        let (v5, v7) = backward!(v5, v7, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 136, false);
        let v15 = inv_twiddle(t, 432, false);
        let v1 = inv_twiddle(t, 136, true);
        let v2 = inv_twiddle(t, 432, true);
        let (v8, v10) = backward!(v8, v10, v1, v3, v2, v15, v0);
        let (v9, v11) = backward!(v9, v11, v1, v3, v2, v15, v0);
        let v3 = inv_twiddle(t, 104, false);
        let v15 = inv_twiddle(t, 400, false);
        let v1 = inv_twiddle(t, 104, true);
        let v2 = inv_twiddle(t, 400, true);
        let (v4, v8) = backward!(v4, v8, v1, v3, v2, v15, v0);
        let (v5, v9) = backward!(v5, v9, v1, v3, v2, v15, v0);
        let (v6, v10) = backward!(v6, v10, v1, v3, v2, v15, v0);
        let (v7, v11) = backward!(v7, v11, v1, v3, v2, v15, v0);
        // P3 third-layer sums (<= 4p) are reduced; see `reduce_sums`.
        let (v4, v5, v6, v7) = reduce_sums(p == Prime::P3, p, v0, v4, v5, v6, v7);
        let v3 = _mm256_shuffle_epi32::<0xA0>(v5);
        let v3 = _mm256_blend_epi32::<0xAA>(v4, v3);
        let v4 = _mm256_srli_epi64::<32>(v4);
        let v5 = _mm256_blend_epi32::<0xAA>(v4, v5);
        let v4 = _mm256_shuffle_epi32::<0xA0>(v7);
        let v4 = _mm256_blend_epi32::<0xAA>(v6, v4);
        let v6 = _mm256_srli_epi64::<32>(v6);
        let v7 = _mm256_blend_epi32::<0xAA>(v6, v7);
        let v6 = _mm256_shuffle_epi32::<0xA0>(v9);
        let v6 = _mm256_blend_epi32::<0xAA>(v8, v6);
        let v8 = _mm256_srli_epi64::<32>(v8);
        let v9 = _mm256_blend_epi32::<0xAA>(v8, v9);
        let v8 = _mm256_shuffle_epi32::<0xA0>(v11);
        let v8 = _mm256_blend_epi32::<0xAA>(v10, v8);
        let v10 = _mm256_srli_epi64::<32>(v10);
        let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
        let v1 = inv_twiddle(t, 72, false);
        let v2 = inv_twiddle(t, 368, false);
        let (v3, v5) = backward!(v3, v5, v1, v1, v2, v2, v0);
        let (v4, v7) = backward!(v4, v7, v1, v1, v2, v2, v0);
        let (v6, v9) = backward!(v6, v9, v1, v1, v2, v2, v0);
        let (v8, v11) = backward!(v8, v11, v1, v1, v2, v2, v0);
        // P2 fourth-layer sums: at most 8p before, below 0.63p after.
        let (v3, v4, v6, v8) = reduce_sums(p == Prime::P2, p, v0, v3, v4, v6, v8);
        let v10 = _mm256_unpacklo_epi64(v3, v4);
        let v4 = _mm256_unpackhi_epi64(v3, v4);
        let v3 = _mm256_unpacklo_epi64(v6, v8);
        let v8 = _mm256_unpackhi_epi64(v6, v8);
        let v6 = _mm256_unpacklo_epi64(v5, v7);
        let v7 = _mm256_unpackhi_epi64(v5, v7);
        let v5 = _mm256_unpacklo_epi64(v9, v11);
        let v11 = _mm256_unpackhi_epi64(v9, v11);
        let v1 = inv_twiddle(t, 40, false);
        let v2 = inv_twiddle(t, 336, false);
        let (v10, v4) = backward!(v10, v4, v1, v1, v2, v2, v0);
        let (v3, v8) = backward!(v3, v8, v1, v1, v2, v2, v0);
        let (v6, v7) = backward!(v6, v7, v1, v1, v2, v2, v0);
        let (v5, v11) = backward!(v5, v11, v1, v1, v2, v2, v0);
        let (v10, v3, v6, v5) = reduce_sums(p == Prime::P3, p, v0, v10, v3, v6, v5);
        let v9 = _mm256_permute2x128_si256::<0x20>(v10, v3);
        let v3 = _mm256_permute2x128_si256::<0x31>(v10, v3);
        let v10 = _mm256_permute2x128_si256::<0x20>(v6, v5);
        let v5 = _mm256_permute2x128_si256::<0x31>(v6, v5);
        let v6 = _mm256_permute2x128_si256::<0x20>(v4, v8);
        let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
        let v4 = _mm256_permute2x128_si256::<0x20>(v7, v11);
        let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
        let v1 = _mm256_set1_epi32(t.0[36]);
        let v2 = _mm256_set1_epi32(t.0[332]);
        let (v9, v3) = backward!(v9, v3, v1, v1, v2, v2, v0);
        let (v10, v5) = backward!(v10, v5, v1, v1, v2, v2, v0);
        let (v6, v8) = backward!(v6, v8, v1, v1, v2, v2, v0);
        let (v4, v11) = backward!(v4, v11, v1, v1, v2, v2, v0);
        store((&mut a.0[192..200]).try_into().unwrap(), v9);
        store((&mut a.0[200..208]).try_into().unwrap(), v10);
        store((&mut a.0[208..216]).try_into().unwrap(), v6);
        store((&mut a.0[216..224]).try_into().unwrap(), v4);
        store((&mut a.0[224..232]).try_into().unwrap(), v3);
        store((&mut a.0[232..240]).try_into().unwrap(), v5);
        store((&mut a.0[240..248]).try_into().unwrap(), v8);
        store((&mut a.0[248..256]).try_into().unwrap(), v11);
    }
    {
        let v4 = load(a.0[0..8].try_into().unwrap());
        let v5 = load(a.0[32..40].try_into().unwrap());
        let v6 = load(a.0[64..72].try_into().unwrap());
        let v7 = load(a.0[96..104].try_into().unwrap());
        let v8 = load(a.0[128..136].try_into().unwrap());
        let v9 = load(a.0[160..168].try_into().unwrap());
        let v10 = load(a.0[192..200].try_into().unwrap());
        let v11 = load(a.0[224..232].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v4, v6) = backward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = backward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v8, v10) = backward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = backward!(v9, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[32]);
        let v2 = _mm256_set1_epi32(t.0[328]);
        let (v4, v8) = backward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = backward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = backward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = backward!(v7, v11, v1, v1, v2, v2, v0);
        store(
            (&mut a.0[128..136]).try_into().unwrap(),
            finish(v8, v0, form),
        );
        store(
            (&mut a.0[160..168]).try_into().unwrap(),
            finish(v9, v0, form),
        );
        store(
            (&mut a.0[192..200]).try_into().unwrap(),
            finish(v10, v0, form),
        );
        store(
            (&mut a.0[224..232]).try_into().unwrap(),
            finish(v11, v0, form),
        );
        let v1 = load(t.0[16..24].try_into().unwrap());
        let v2 = load(t.0[24..32].try_into().unwrap());
        let v12 = _mm256_mul_epi32(v4, v1);
        let v13 = _mm256_mul_epi32(v5, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v4 = _mm256_mul_epi32(v4, v2);
        let v5 = _mm256_mul_epi32(v5, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v4 = _mm256_sub_epi32(v4, v12);
        let v5 = _mm256_sub_epi32(v5, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v4 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v5 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v4 = _mm256_blend_epi32::<0xAA>(v4, v8);
        let v5 = _mm256_blend_epi32::<0xAA>(v5, v9);
        let v12 = _mm256_mul_epi32(v6, v1);
        let v13 = _mm256_mul_epi32(v7, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v6 = _mm256_mul_epi32(v6, v2);
        let v7 = _mm256_mul_epi32(v7, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v6 = _mm256_sub_epi32(v6, v12);
        let v7 = _mm256_sub_epi32(v7, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v6 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v7 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v6 = _mm256_blend_epi32::<0xAA>(v6, v8);
        let v7 = _mm256_blend_epi32::<0xAA>(v7, v9);
        store((&mut a.0[0..8]).try_into().unwrap(), finish(v4, v0, form));
        store((&mut a.0[32..40]).try_into().unwrap(), finish(v5, v0, form));
        store((&mut a.0[64..72]).try_into().unwrap(), finish(v6, v0, form));
        store(
            (&mut a.0[96..104]).try_into().unwrap(),
            finish(v7, v0, form),
        );
    }
    {
        let v4 = load(a.0[8..16].try_into().unwrap());
        let v5 = load(a.0[40..48].try_into().unwrap());
        let v6 = load(a.0[72..80].try_into().unwrap());
        let v7 = load(a.0[104..112].try_into().unwrap());
        let v8 = load(a.0[136..144].try_into().unwrap());
        let v9 = load(a.0[168..176].try_into().unwrap());
        let v10 = load(a.0[200..208].try_into().unwrap());
        let v11 = load(a.0[232..240].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v4, v6) = backward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = backward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v8, v10) = backward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = backward!(v9, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[32]);
        let v2 = _mm256_set1_epi32(t.0[328]);
        let (v4, v8) = backward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = backward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = backward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = backward!(v7, v11, v1, v1, v2, v2, v0);
        store(
            (&mut a.0[136..144]).try_into().unwrap(),
            finish(v8, v0, form),
        );
        store(
            (&mut a.0[168..176]).try_into().unwrap(),
            finish(v9, v0, form),
        );
        store(
            (&mut a.0[200..208]).try_into().unwrap(),
            finish(v10, v0, form),
        );
        store(
            (&mut a.0[232..240]).try_into().unwrap(),
            finish(v11, v0, form),
        );
        let v1 = load(t.0[16..24].try_into().unwrap());
        let v2 = load(t.0[24..32].try_into().unwrap());
        let v12 = _mm256_mul_epi32(v4, v1);
        let v13 = _mm256_mul_epi32(v5, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v4 = _mm256_mul_epi32(v4, v2);
        let v5 = _mm256_mul_epi32(v5, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v4 = _mm256_sub_epi32(v4, v12);
        let v5 = _mm256_sub_epi32(v5, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v4 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v5 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v4 = _mm256_blend_epi32::<0xAA>(v4, v8);
        let v5 = _mm256_blend_epi32::<0xAA>(v5, v9);
        let v12 = _mm256_mul_epi32(v6, v1);
        let v13 = _mm256_mul_epi32(v7, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v6 = _mm256_mul_epi32(v6, v2);
        let v7 = _mm256_mul_epi32(v7, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v6 = _mm256_sub_epi32(v6, v12);
        let v7 = _mm256_sub_epi32(v7, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v6 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v7 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v6 = _mm256_blend_epi32::<0xAA>(v6, v8);
        let v7 = _mm256_blend_epi32::<0xAA>(v7, v9);
        store((&mut a.0[8..16]).try_into().unwrap(), finish(v4, v0, form));
        store((&mut a.0[40..48]).try_into().unwrap(), finish(v5, v0, form));
        store((&mut a.0[72..80]).try_into().unwrap(), finish(v6, v0, form));
        store(
            (&mut a.0[104..112]).try_into().unwrap(),
            finish(v7, v0, form),
        );
    }
    {
        let v4 = load(a.0[16..24].try_into().unwrap());
        let v5 = load(a.0[48..56].try_into().unwrap());
        let v6 = load(a.0[80..88].try_into().unwrap());
        let v7 = load(a.0[112..120].try_into().unwrap());
        let v8 = load(a.0[144..152].try_into().unwrap());
        let v9 = load(a.0[176..184].try_into().unwrap());
        let v10 = load(a.0[208..216].try_into().unwrap());
        let v11 = load(a.0[240..248].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v4, v6) = backward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = backward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v8, v10) = backward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = backward!(v9, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[32]);
        let v2 = _mm256_set1_epi32(t.0[328]);
        let (v4, v8) = backward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = backward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = backward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = backward!(v7, v11, v1, v1, v2, v2, v0);
        store(
            (&mut a.0[144..152]).try_into().unwrap(),
            finish(v8, v0, form),
        );
        store(
            (&mut a.0[176..184]).try_into().unwrap(),
            finish(v9, v0, form),
        );
        store(
            (&mut a.0[208..216]).try_into().unwrap(),
            finish(v10, v0, form),
        );
        store(
            (&mut a.0[240..248]).try_into().unwrap(),
            finish(v11, v0, form),
        );
        let v1 = load(t.0[16..24].try_into().unwrap());
        let v2 = load(t.0[24..32].try_into().unwrap());
        let v12 = _mm256_mul_epi32(v4, v1);
        let v13 = _mm256_mul_epi32(v5, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v4 = _mm256_mul_epi32(v4, v2);
        let v5 = _mm256_mul_epi32(v5, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v4 = _mm256_sub_epi32(v4, v12);
        let v5 = _mm256_sub_epi32(v5, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v4 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v5 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v4 = _mm256_blend_epi32::<0xAA>(v4, v8);
        let v5 = _mm256_blend_epi32::<0xAA>(v5, v9);
        let v12 = _mm256_mul_epi32(v6, v1);
        let v13 = _mm256_mul_epi32(v7, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v6 = _mm256_mul_epi32(v6, v2);
        let v7 = _mm256_mul_epi32(v7, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v6 = _mm256_sub_epi32(v6, v12);
        let v7 = _mm256_sub_epi32(v7, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v6 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v7 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v6 = _mm256_blend_epi32::<0xAA>(v6, v8);
        let v7 = _mm256_blend_epi32::<0xAA>(v7, v9);
        store((&mut a.0[16..24]).try_into().unwrap(), finish(v4, v0, form));
        store((&mut a.0[48..56]).try_into().unwrap(), finish(v5, v0, form));
        store((&mut a.0[80..88]).try_into().unwrap(), finish(v6, v0, form));
        store(
            (&mut a.0[112..120]).try_into().unwrap(),
            finish(v7, v0, form),
        );
    }
    {
        let v4 = load(a.0[24..32].try_into().unwrap());
        let v5 = load(a.0[56..64].try_into().unwrap());
        let v6 = load(a.0[88..96].try_into().unwrap());
        let v7 = load(a.0[120..128].try_into().unwrap());
        let v8 = load(a.0[152..160].try_into().unwrap());
        let v9 = load(a.0[184..192].try_into().unwrap());
        let v10 = load(a.0[216..224].try_into().unwrap());
        let v11 = load(a.0[248..256].try_into().unwrap());
        let v1 = _mm256_set1_epi32(t.0[35]);
        let v2 = _mm256_set1_epi32(t.0[331]);
        let (v4, v6) = backward!(v4, v6, v1, v1, v2, v2, v0);
        let (v5, v7) = backward!(v5, v7, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[34]);
        let v2 = _mm256_set1_epi32(t.0[330]);
        let (v8, v10) = backward!(v8, v10, v1, v1, v2, v2, v0);
        let (v9, v11) = backward!(v9, v11, v1, v1, v2, v2, v0);
        let v1 = _mm256_set1_epi32(t.0[32]);
        let v2 = _mm256_set1_epi32(t.0[328]);
        let (v4, v8) = backward!(v4, v8, v1, v1, v2, v2, v0);
        let (v5, v9) = backward!(v5, v9, v1, v1, v2, v2, v0);
        let (v6, v10) = backward!(v6, v10, v1, v1, v2, v2, v0);
        let (v7, v11) = backward!(v7, v11, v1, v1, v2, v2, v0);
        store(
            (&mut a.0[152..160]).try_into().unwrap(),
            finish(v8, v0, form),
        );
        store(
            (&mut a.0[184..192]).try_into().unwrap(),
            finish(v9, v0, form),
        );
        store(
            (&mut a.0[216..224]).try_into().unwrap(),
            finish(v10, v0, form),
        );
        store(
            (&mut a.0[248..256]).try_into().unwrap(),
            finish(v11, v0, form),
        );
        let v1 = load(t.0[16..24].try_into().unwrap());
        let v2 = load(t.0[24..32].try_into().unwrap());
        let v12 = _mm256_mul_epi32(v4, v1);
        let v13 = _mm256_mul_epi32(v5, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v4 = _mm256_mul_epi32(v4, v2);
        let v5 = _mm256_mul_epi32(v5, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v4 = _mm256_sub_epi32(v4, v12);
        let v5 = _mm256_sub_epi32(v5, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v4 = _mm256_shuffle_epi32::<0xF5>(v4);
        let v5 = _mm256_shuffle_epi32::<0xF5>(v5);
        let v4 = _mm256_blend_epi32::<0xAA>(v4, v8);
        let v5 = _mm256_blend_epi32::<0xAA>(v5, v9);
        let v12 = _mm256_mul_epi32(v6, v1);
        let v13 = _mm256_mul_epi32(v7, v1);
        let v8 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v9 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v14 = _mm256_mul_epi32(v8, v1);
        let v15 = _mm256_mul_epi32(v9, v1);
        let v6 = _mm256_mul_epi32(v6, v2);
        let v7 = _mm256_mul_epi32(v7, v2);
        let v8 = _mm256_mul_epi32(v8, v2);
        let v9 = _mm256_mul_epi32(v9, v2);
        let v12 = _mm256_mul_epi32(v12, v0);
        let v13 = _mm256_mul_epi32(v13, v0);
        let v14 = _mm256_mul_epi32(v14, v0);
        let v15 = _mm256_mul_epi32(v15, v0);
        let v6 = _mm256_sub_epi32(v6, v12);
        let v7 = _mm256_sub_epi32(v7, v13);
        let v8 = _mm256_sub_epi32(v8, v14);
        let v9 = _mm256_sub_epi32(v9, v15);
        let v6 = _mm256_shuffle_epi32::<0xF5>(v6);
        let v7 = _mm256_shuffle_epi32::<0xF5>(v7);
        let v6 = _mm256_blend_epi32::<0xAA>(v6, v8);
        let v7 = _mm256_blend_epi32::<0xAA>(v7, v9);
        store((&mut a.0[24..32]).try_into().unwrap(), finish(v4, v0, form));
        store((&mut a.0[56..64]).try_into().unwrap(), finish(v5, v0, form));
        store((&mut a.0[88..96]).try_into().unwrap(), finish(v6, v0, form));
        store(
            (&mut a.0[120..128]).try_into().unwrap(),
            finish(v7, v0, form),
        );
    }
}
/// Montgomery-multiplies four registers of inverse sums by R mod p when `on`,
/// which keeps each residue and leaves |r| < |x| (R mod p)/2^32 + p/2.
///
/// P2 reduces its fourth-layer sums (|x| <= 8p, then below 0.63p). For P3,
/// 8p exceeds 2^31, so it reduces its third-layer sums (|x| <= 4p) and its
/// fifth-layer sums (|x| < 2.56p), each to below 0.67p; every other P3 sum
/// stays below 4.68p < 2^31 (5.69p), as `schedule_bounds` checks.
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn reduce_sums(
    on: bool,
    p: Prime,
    v0: __m256i,
    a: __m256i,
    b: __m256i,
    c: __m256i,
    d: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    if !on {
        return (a, b, c, d);
    }
    let z = _mm256_set1_epi32(consts(p).mont);
    let qi = _mm256_set1_epi32(consts(p).pinv);
    (mont(a, z, v0, qi), mont(b, z, v0, qi), mont(c, z, v0, qi), mont(d, z, v0, qi))
}
/// Inverse twiddles of the eight-word group at reference word `w`, read
/// pre-shuffled: the qword-reversed group, or (`odd`) its odd words duplicated.
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn inv_twiddle(t: &tables::Table, w: usize, odd: bool) -> __m256i {
    let (i, z) = if w >= 328 { (w - 328, 1) } else { (w - 32, 0) };
    let v = 8 * (tables::FWD_VECS + 4 * (i / 8 - 1) + 2 * z + usize::from(odd));
    load(t.1[v..v + 8].try_into().unwrap())
}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn mont(a: __m256i, z: __m256i, p: __m256i, qi: __m256i) -> __m256i {
    let lo = _mm256_mullo_epi32(a, _mm256_mullo_epi32(z, qi));
    let e = _mm256_sub_epi64(_mm256_mul_epi32(a, z), _mm256_mul_epi32(lo, p));
    let o = _mm256_sub_epi64(
        _mm256_mul_epi32(_mm256_srli_epi64::<32>(a), _mm256_srli_epi64::<32>(z)),
        _mm256_mul_epi32(_mm256_srli_epi64::<32>(lo), p),
    );
    _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(e), o)
}
#[inline]
#[target_feature(enable = "avx2,popcnt")]
fn finish(x: __m256i, p: __m256i, f: Form) -> __m256i {
    // Input in (-p,p): the inverse final multiply gives below 0.74q, 0.61P2 resp.
    // 0.71P3, the rounded-reciprocal reduction below 0.751p. A negative x is a huge
    // unsigned value while x + p is in [0,p), so the unsigned minimum is canonical.
    let x = _mm256_min_epu32(x, _mm256_add_epi32(x, p));
    if f == Form::Canonical {
        x
    } else {
        let half = _mm256_srli_epi32::<1>(p);
        _mm256_sub_epi32(x, _mm256_and_si256(_mm256_cmpgt_epi32(x, half), p))
    }
}
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn intt(p: Prime, a: &mut Poly32, s: Scale, f: Form) {
    #[cfg(target_os = "linux")]
    if p == Prime::Q {
        // SAFETY: dispatch proved AVX2; the polynomial and the table are aligned
        // and complete, as the SysV assembly reads them.
        unsafe {
            assembly::agg_mldsa_inverse_avx2(
                a,
                tables::table(p, s),
                u32::from(f == Form::Centered),
            );
        }
        return;
    }
    inverse(p, tables::table(p, s), a, f);
}
#[target_feature(enable = "avx2,popcnt")]
/// `CENTER` conditional +-p passes center the Montgomery output: none for q,
/// one for P2 (|x| < 1.08p for L <= 7), two for P3 (|x| < 1.9p for L <= 7,
/// beyond the 1.5p one pass can center).
fn dot_inner_l<const L: usize, const CENTER: usize>(p: Prime, a: &[Poly32; L], b: &[Poly32; L], out: &mut Poly32) {
    let c = consts(p);
    let pv = _mm256_set1_epi32(c.p);
    let qi = _mm256_set1_epi32(c.pinv);
    for block in (0..256).step_by(16) {
        for lane in 0..2 {
            let j = block + 8 * lane;
            let mut e = _mm256_setzero_si256();
            let mut o = e;
            // Centered P2/P3 (canonical q) matrix times lazy NTT(z), L <= 7:
            // the 64-bit sums stay below 2^61.
            for k in 0..L {
                let av = load(a[k].0[j..j + 8].try_into().unwrap());
                let bv = load(b[k].0[j..j + 8].try_into().unwrap());
                e = _mm256_add_epi64(e, _mm256_mul_epi32(av, bv));
                o = _mm256_add_epi64(
                    o,
                    _mm256_mul_epi32(_mm256_srli_epi64::<32>(av), _mm256_srli_epi64::<32>(bv)),
                );
            }
            let e = _mm256_sub_epi64(e, _mm256_mul_epi32(_mm256_mul_epi32(e, qi), pv));
            let o = _mm256_sub_epi64(o, _mm256_mul_epi32(_mm256_mul_epi32(o, qi), pv));
            let mut x = _mm256_blend_epi32::<0xAA>(_mm256_srli_epi64::<32>(e), o);
            for _ in 0..CENTER {
                x = _mm256_add_epi32(
                    _mm256_sub_epi32(
                        x,
                        _mm256_and_si256(_mm256_cmpgt_epi32(x, _mm256_set1_epi32(c.p / 2)), pv),
                    ),
                    _mm256_and_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(-c.p / 2), x), pv),
                );
            }
            store((&mut out.0[j..j + 8]).try_into().unwrap(), x);
        }
    }
}
#[target_feature(enable = "avx2,popcnt")]
#[cfg(any(test, not(target_os = "linux")))]
unsafe fn dot4_intrinsics(p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    match p {
        Prime::Q => dot_inner_l::<4, 0>(p, a, b, out),
        Prime::P2 => dot_inner_l::<4, 1>(p, a, b, out),
        Prime::P3 => dot_inner_l::<4, 2>(p, a, b, out),
    }
}
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn dot_l<const L: usize>(p: Prime, a: &[Poly32; L], b: &[Poly32; L], out: &mut Poly32) {
    #[cfg(target_os = "linux")]
    if L == 4 {
        // SAFETY: with L == 4, `[Poly32; L]` and `[Poly32; 4]` are the same type
        // layout, and the references keep the borrows of `a` and `b`.
        let a4: &[Poly32; 4] = unsafe { &*(a.as_ptr() as *const [Poly32; 4]) };
        // SAFETY: as for `a4`.
        let b4: &[Poly32; 4] = unsafe { &*(b.as_ptr() as *const [Poly32; 4]) };
        // SAFETY: the caller proved AVX2/POPCNT; the typed arrays are aligned and complete.
        unsafe { dot4(p, a4, b4, out) };
        return;
    }
    match p {
        Prime::Q => dot_inner_l::<L, 0>(p, a, b, out),
        Prime::P2 => dot_inner_l::<L, 1>(p, a, b, out),
        Prime::P3 => dot_inner_l::<L, 2>(p, a, b, out),
    }
}
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn reduce(p: Prime, a: &mut Poly32, f: Form) {
    let c = consts(p);
    let pv = _mm256_set1_epi32(c.p);
    // Rounded reciprocal quotient: the residual of any i32 is less than 0.751p.
    let recip = match p {
        Prime::Q => 513,
        Prime::P2 => 26,
        Prime::P3 => 11,
    };
    let rv = _mm256_set1_epi32(recip);
    for chunk in a.0.chunks_exact_mut(8) {
        let chunk: &mut [i32; 8] = chunk.try_into().unwrap();
        let x = load(chunk);
        let e = _mm256_srli_epi64::<32>(_mm256_add_epi64(
            _mm256_mul_epi32(x, rv),
            _mm256_set1_epi64x(1 << 31),
        ));
        let o = _mm256_add_epi64(
            _mm256_mul_epi32(_mm256_srli_epi64::<32>(x), rv),
            _mm256_set1_epi64x(1 << 31),
        );
        let quot = _mm256_blend_epi32::<0xAA>(e, o);
        let r = _mm256_sub_epi32(x, _mm256_mullo_epi32(quot, pv));
        store(chunk, finish(r, pv, f));
    }
}
#[target_feature(enable = "avx2,popcnt")]
#[cfg(any(test, not(target_os = "linux")))]
unsafe fn order_from_fips_intrinsics(a: &mut Poly32) {
    permute(a, false);
}
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn order_to_fips(a: &mut Poly32) {
    // SAFETY: the feature and typed-buffer contracts match the forward order map.
    unsafe {
        order_from_fips(a);
    }
}
#[target_feature(enable = "avx2,popcnt")]
#[cfg(any(test, not(target_os = "linux")))]
fn permute(a: &mut Poly32, _inverse: bool) {
    for chunk in a.0.chunks_exact_mut(64) {
        transpose(chunk.try_into().unwrap());
    }
}
#[inline(never)]
#[target_feature(enable = "avx2,popcnt")]
#[cfg(any(test, not(target_os = "linux")))]
fn transpose(a: &mut [i32; 64]) {
    let v4 = load(a[0..8].try_into().unwrap());
    let v5 = load(a[8..16].try_into().unwrap());
    let v6 = load(a[16..24].try_into().unwrap());
    let v7 = load(a[24..32].try_into().unwrap());
    let v8 = load(a[32..40].try_into().unwrap());
    let v9 = load(a[40..48].try_into().unwrap());
    let v10 = load(a[48..56].try_into().unwrap());
    let v11 = load(a[56..64].try_into().unwrap());
    let v3 = _mm256_permute2x128_si256::<0x20>(v4, v8);
    let v8 = _mm256_permute2x128_si256::<0x31>(v4, v8);
    let v4 = _mm256_permute2x128_si256::<0x20>(v5, v9);
    let v9 = _mm256_permute2x128_si256::<0x31>(v5, v9);
    let v5 = _mm256_permute2x128_si256::<0x20>(v6, v10);
    let v10 = _mm256_permute2x128_si256::<0x31>(v6, v10);
    let v6 = _mm256_permute2x128_si256::<0x20>(v7, v11);
    let v11 = _mm256_permute2x128_si256::<0x31>(v7, v11);
    let v7 = _mm256_unpacklo_epi64(v3, v5);
    let v5 = _mm256_unpackhi_epi64(v3, v5);
    let v3 = _mm256_unpacklo_epi64(v8, v10);
    let v10 = _mm256_unpackhi_epi64(v8, v10);
    let v8 = _mm256_unpacklo_epi64(v4, v6);
    let v6 = _mm256_unpackhi_epi64(v4, v6);
    let v4 = _mm256_unpacklo_epi64(v9, v11);
    let v11 = _mm256_unpackhi_epi64(v9, v11);
    let v9 = _mm256_shuffle_epi32::<0xA0>(v8);
    let v9 = _mm256_blend_epi32::<0xAA>(v7, v9);
    let v7 = _mm256_srli_epi64::<32>(v7);
    let v8 = _mm256_blend_epi32::<0xAA>(v7, v8);
    let v7 = _mm256_shuffle_epi32::<0xA0>(v6);
    let v7 = _mm256_blend_epi32::<0xAA>(v5, v7);
    let v5 = _mm256_srli_epi64::<32>(v5);
    let v6 = _mm256_blend_epi32::<0xAA>(v5, v6);
    let v5 = _mm256_shuffle_epi32::<0xA0>(v4);
    let v5 = _mm256_blend_epi32::<0xAA>(v3, v5);
    let v3 = _mm256_srli_epi64::<32>(v3);
    let v4 = _mm256_blend_epi32::<0xAA>(v3, v4);
    let v3 = _mm256_shuffle_epi32::<0xA0>(v11);
    let v3 = _mm256_blend_epi32::<0xAA>(v10, v3);
    let v10 = _mm256_srli_epi64::<32>(v10);
    let v11 = _mm256_blend_epi32::<0xAA>(v10, v11);
    store((&mut a[0..8]).try_into().unwrap(), v9);
    store((&mut a[8..16]).try_into().unwrap(), v8);
    store((&mut a[16..24]).try_into().unwrap(), v7);
    store((&mut a[24..32]).try_into().unwrap(), v6);
    store((&mut a[32..40]).try_into().unwrap(), v5);
    store((&mut a[40..48]).try_into().unwrap(), v4);
    store((&mut a[48..56]).try_into().unwrap(), v3);
    store((&mut a[56..64]).try_into().unwrap(), v11);
}

#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn ntt(p: Prime, a: &mut Poly32) {
    // SAFETY: dispatch proved AVX2/POPCNT; the polynomial and the table are
    // aligned and complete, as the SysV assembly reads them.
    #[cfg(target_os = "linux")]
    unsafe {
        assembly::agg_mldsa_ntt_avx2(a, tables::table(p, Scale::Mont));
    }
    // SAFETY: dispatch proved AVX2/POPCNT.
    #[cfg(not(target_os = "linux"))]
    unsafe {
        ntt_intrinsics(p, a);
    }
}
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn dot4(p: Prime, a: &[Poly32; 4], b: &[Poly32; 4], out: &mut Poly32) {
    // The assembly centers its output once; P3 needs two passes.
    #[cfg(target_os = "linux")]
    if p == Prime::P3 {
        dot_inner_l::<4, 2>(p, a, b, out);
        return;
    }
    // SAFETY: dispatch proved AVX2/POPCNT; all typed arrays are aligned and
    // complete, and the SysV assembly reads only inside them.
    #[cfg(target_os = "linux")]
    unsafe {
        assembly::agg_mldsa_dot_avx2(
            out,
            a,
            b,
            tables::table(p, Scale::Mont),
            u32::from(p == Prime::P2),
        );
    }
    // SAFETY: dispatch proved AVX2/POPCNT.
    #[cfg(not(target_os = "linux"))]
    unsafe {
        dot4_intrinsics(p, a, b, out);
    }
}
#[target_feature(enable = "avx2,popcnt")]
pub(crate) unsafe fn order_from_fips(a: &mut Poly32) {
    // SAFETY: dispatch proved AVX2; all accesses remain within this aligned polynomial.
    #[cfg(target_os = "linux")]
    unsafe {
        assembly::agg_mldsa_order_avx2(a);
    }
    // SAFETY: dispatch proved AVX2/POPCNT.
    #[cfg(not(target_os = "linux"))]
    unsafe {
        order_from_fips_intrinsics(a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intrinsics_match_selected_kernels() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt")) {
            eprintln!("Skipping: AVX2/POPCNT unavailable");
            return;
        }
        for p in [Prime::Q, Prime::P2, Prime::P3] {
            for case in 0..12 {
                let input = Poly32(std::array::from_fn(|i| {
                    ((i * 773 + case * 5137) % 8388609) as i32 - 4194304
                }));
                let mut a = input;
                let mut b = input;
                // SAFETY: the features were checked above; all buffers are typed and aligned.
                unsafe {
                    ntt(p, &mut a);
                    ntt_intrinsics(p, &mut b);
                }
                assert_eq!(a, b);
                let mut ord = input;
                let mut intr = input;
                // SAFETY: the features were checked above.
                unsafe {
                    order_from_fips(&mut ord);
                    order_from_fips_intrinsics(&mut intr);
                }
                assert_eq!(ord, intr);
                // SAFETY: the features were checked above.
                unsafe {
                    reduce(p, &mut a, Form::Centered);
                }
                let aa = [a; 4];
                let bb = [b; 4];
                let mut dot = Poly32::zero();
                let mut di = dot;
                // SAFETY: the features were checked above, and dot input bounds hold.
                unsafe {
                    dot4(p, &aa, &bb, &mut dot);
                    dot4_intrinsics(p, &aa, &bb, &mut di);
                }
                assert_eq!(dot, di);
                for scale in [Scale::Plain, Scale::Mont] {
                    for form in [Form::Canonical, Form::Centered] {
                        let mut out = dot;
                        let mut oi = dot;
                        // SAFETY: the features were checked above; inverse inputs are bounded by p.
                        unsafe {
                            intt(p, &mut out, scale, form);
                            inverse(p, tables::table(p, scale), &mut oi, form);
                        }
                        assert_eq!(out, oi);
                    }
                }
            }
        }
    }
}

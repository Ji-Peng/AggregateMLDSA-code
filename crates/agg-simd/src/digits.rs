//! 16-lane AVX-512 digit decomposition of `(z, u, v)` and bit extraction of `w_1'` directly
//! into the round-1 `G_main16` (`i32`) and `G_main4` (`i8`) column writers.
//!
//! Supported forms:
//! - `z` (relation H-u): `z + 2^17 = z_lo + 2^16 z_hi` for `z in [-2^17, 2^17)`, stored as
//!   `Z_lo = z_lo - 2^15` (`i32`) and `Z_hi = z_hi - 2` (`i8`).
//! - `z` (relation H-exact): `z + 130,993 = B_zlo + 2^16 B_zhi + 130,915 a_z` for
//!   `z in [-130,993, 130,993]`, with `a_z = [z + 130,993 >= 2^17]`, stored as
//!   `bz_lo = B_zlo - 2^15` (`i32`), `bz_hi = B_zhi` (`i8`), `a_z` (`i8`).
//! - `u` (relations H-u and H-exact): step form `u + 190,465 = u_lo + 63,079 u_hi` for
//!   `u in [-190,465, 190,465]` with `u_hi = min(floor((u + 190,465) / 63,079), 5)` computed by
//!   multiply-high `floor((y * 4,357,677) / 2^38)` (`4,357,677 * 63,079 = 2^38 + 789`, exact on
//!   `[0, 380,930]`), stored as `U_lo = u_lo - 2^15` (`i32`), `U_hi = u_hi - 3` (`i8`).
//! - `v`: `v + 2^31 = v_lo + 2^16 v_hi` for `v in [-2^31, 2^31)`, stored as
//!   `V_lo = v_lo - 2^15 = (v & 0xFFFF) - 2^15` (`i32`) and `V_hi = v_hi - 2^15 = v >> 16` (`i32`).
//! - `w_1'`: six bits `b_t = (w_1' >> t) & 1` (`i8`) for `w_1' in [0, 64)`.

use crate::buf::ColumnWriter;
use crate::detect::{backend, Backend};

const N: usize = 256;
const K: usize = 4;
const POSITIONS: usize = K * N;

const Z_HU_MIN: i64 = -(1 << 17);
const Z_HU_MAX: i64 = (1 << 17) - 1;
const Z_HEXACT_BOUND: i64 = 130_993;
const SLACK_Z: i32 = 130_915;
const B_U: i64 = 190_465;
const STEP_U: i32 = 63_079;
/// Magic multiplier `ceil(2^38 / 63,079) = 4,357,677`: `4,357,677 * 63,079 = 2^38 + 789`,
/// so `floor((y * 4,357,677) >> 38) == floor(y / 63,079)` for every `y in [0, 380,930]`.
const U_MAGIC: u32 = 4_357_677;
const V_MIN: i64 = -(1 << 31);
const V_MAX: i64 = (1 << 31) - 1;

/// Decompose 16 `u` values in `[-190_465, 190_465]` into `(U_lo: [i32; 16], U_hi: [i8; 16])`
/// using the multiply-high quotient `min(floor((y * 4,357,677) >> 38), 5)`.
pub fn decompose_u_step_16(u: &[i32; 16]) -> ([i32; 16], [i8; 16]) {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        // SAFETY: the call is guarded by the backend check and the length assertions above.
        return unsafe { avx512::decompose_u_step_16_avx512(u) };
    }
    let mut u_lo = [0i32; 16];
    let mut u_hi = [0i8; 16];
    for i in 0..16 {
        let y = (u[i] + B_U as i32) as u32;
        let q = (((u64::from(y) * u64::from(U_MAGIC)) >> 38) as u32).min(5) as i32;
        u_lo[i] = (y as i32 - STEP_U * q) - (1 << 15);
        u_hi[i] = (q - 3) as i8;
    }
    (u_lo, u_hi)
}

#[inline]
fn check_cell_hu(z: i64, u: i64, v: i64, w1: u8) -> Result<(), &'static str> {
    if !(Z_HU_MIN..=Z_HU_MAX).contains(&z) {
        return Err("z outside [-2^17, 2^17)");
    }
    if !(-B_U..=B_U).contains(&u) {
        return Err("u outside [-190,465, 190,465]");
    }
    if !(V_MIN..=V_MAX).contains(&v) {
        return Err("v outside [-2^31, 2^31)");
    }
    if w1 >= 64 {
        return Err("w1' above six bits");
    }
    Ok(())
}

#[inline]
fn check_cell_hexact(z: i64, u: i64, v: i64, w1: u8) -> Result<(), &'static str> {
    if !(-Z_HEXACT_BOUND..=Z_HEXACT_BOUND).contains(&z) {
        return Err("z outside [-130,993, 130,993]");
    }
    if !(-B_U..=B_U).contains(&u) {
        return Err("u outside [-190,465, 190,465]");
    }
    if !(V_MIN..=V_MAX).contains(&v) {
        return Err("v outside [-2^31, 2^31)");
    }
    if w1 >= 64 {
        return Err("w1' above six bits");
    }
    Ok(())
}

/// Encode one record (`1024` coefficients) under relation H-u directly into `wide` and `narrow`.
///
/// `wide_slots` holds `[z_lo, u_lo, v_lo, v_hi]` (typically `[0, 1, 2, 3]`);
/// `narrow_slots` holds `[z_hi, u_hi, b_0, b_1, b_2, b_3, b_4, b_5]` (typically `[0..8]`).
#[allow(clippy::too_many_arguments)]
pub fn encode_record_hu(
    z: &[[i64; N]; K],
    u: &[[i64; N]; K],
    v: &[[i64; N]; K],
    w1: &[[u8; N]; K],
    wide_slots: [usize; 4],
    narrow_slots: [usize; 8],
    wide: &mut ColumnWriter<'_, i32>,
    narrow: &mut ColumnWriter<'_, i8>,
) -> Result<(), &'static str> {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        // SAFETY: the call is guarded by the backend check and the length assertions above.
        return unsafe {
            avx512::encode_record_hu_avx512(
                z,
                u,
                v,
                w1,
                wide_slots,
                narrow_slots,
                wide,
                narrow,
            )
        };
    }
    encode_record_hu_scalar(z, u, v, w1, wide_slots, narrow_slots, wide, narrow)
}

/// Encode one record (`1024` coefficients) under relation H-exact directly into `wide` and `narrow`.
///
/// `wide_slots` holds `[bz_lo, u_lo, v_lo, v_hi]`;
/// `narrow_slots` holds `[bz_hi, u_hi, b_0, b_1, b_2, b_3, b_4, b_5, a_z]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_record_hexact(
    z: &[[i64; N]; K],
    u: &[[i64; N]; K],
    v: &[[i64; N]; K],
    w1: &[[u8; N]; K],
    wide_slots: [usize; 4],
    narrow_slots: [usize; 9],
    wide: &mut ColumnWriter<'_, i32>,
    narrow: &mut ColumnWriter<'_, i8>,
) -> Result<(), &'static str> {
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx512 {
        // SAFETY: the call is guarded by the backend check and the length assertions above.
        return unsafe {
            avx512::encode_record_hexact_avx512(
                z,
                u,
                v,
                w1,
                wide_slots,
                narrow_slots,
                wide,
                narrow,
            )
        };
    }
    encode_record_hexact_scalar(z, u, v, w1, wide_slots, narrow_slots, wide, narrow)
}

#[allow(clippy::too_many_arguments)]
fn encode_record_hu_scalar(
    z: &[[i64; N]; K],
    u: &[[i64; N]; K],
    v: &[[i64; N]; K],
    w1: &[[u8; N]; K],
    wide_slots: [usize; 4],
    narrow_slots: [usize; 8],
    wide: &mut ColumnWriter<'_, i32>,
    narrow: &mut ColumnWriter<'_, i8>,
) -> Result<(), &'static str> {
    for row in 0..K {
        for j in 0..N {
            let pos = row * N + j;
            let (zv, uv, vv, w1v) = (z[row][j], u[row][j], v[row][j], w1[row][j]);
            check_cell_hu(zv, uv, vv, w1v)?;

            let yz = (zv as i32) + (1 << 17);
            let z_lo = (yz & 0xFFFF) - (1 << 15);
            let z_hi = ((yz >> 16) - 2) as i8;

            let yu = (uv as i32 + B_U as i32) as u32;
            let u_hi_u = (((u64::from(yu) * u64::from(U_MAGIC)) >> 38) as u32).min(5) as i32;
            let u_lo = (yu as i32 - STEP_U * u_hi_u) - (1 << 15);
            let u_hi = (u_hi_u - 3) as i8;

            let vi = vv as i32;
            let v_lo = (vi & 0xFFFF) - (1 << 15);
            let v_hi = vi >> 16;

            wide.write(wide_slots[0], pos, z_lo);
            wide.write(wide_slots[1], pos, u_lo);
            wide.write(wide_slots[2], pos, v_lo);
            wide.write(wide_slots[3], pos, v_hi);

            narrow.write(narrow_slots[0], pos, z_hi);
            narrow.write(narrow_slots[1], pos, u_hi);
            for t in 0..6 {
                narrow.write(narrow_slots[2 + t], pos, ((w1v >> t) & 1) as i8);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_record_hexact_scalar(
    z: &[[i64; N]; K],
    u: &[[i64; N]; K],
    v: &[[i64; N]; K],
    w1: &[[u8; N]; K],
    wide_slots: [usize; 4],
    narrow_slots: [usize; 9],
    wide: &mut ColumnWriter<'_, i32>,
    narrow: &mut ColumnWriter<'_, i8>,
) -> Result<(), &'static str> {
    for row in 0..K {
        for j in 0..N {
            let pos = row * N + j;
            let (zv, uv, vv, w1v) = (z[row][j], u[row][j], v[row][j], w1[row][j]);
            check_cell_hexact(zv, uv, vv, w1v)?;

            let yz = (zv as i32) + (Z_HEXACT_BOUND as i32);
            let a_z = i32::from(yz >= (1 << 17));
            let rest = yz - SLACK_Z * a_z;
            let bz_lo = (rest & 0xFFFF) - (1 << 15);
            let bz_hi = (rest >> 16) as i8;

            let yu = (uv as i32 + B_U as i32) as u32;
            let u_hi_u = (((u64::from(yu) * u64::from(U_MAGIC)) >> 38) as u32).min(5) as i32;
            let u_lo = (yu as i32 - STEP_U * u_hi_u) - (1 << 15);
            let u_hi = (u_hi_u - 3) as i8;

            let vi = vv as i32;
            let v_lo = (vi & 0xFFFF) - (1 << 15);
            let v_hi = vi >> 16;

            wide.write(wide_slots[0], pos, bz_lo);
            wide.write(wide_slots[1], pos, u_lo);
            wide.write(wide_slots[2], pos, v_lo);
            wide.write(wide_slots[3], pos, v_hi);

            narrow.write(narrow_slots[0], pos, bz_hi);
            narrow.write(narrow_slots[1], pos, u_hi);
            for t in 0..6 {
                narrow.write(narrow_slots[2 + t], pos, ((w1v >> t) & 1) as i8);
            }
            narrow.write(narrow_slots[8], pos, a_z as i8);
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// Load 16 `i64` values from `ptr`, verify each is in `[min, max]`, and pack into 16 `i32` lanes.
    /// Returns `(packed_i32, out_of_range_mask)`.
    #[inline(always)]
    unsafe fn load_and_check_i64_16(
        ptr: *const i64,
        min_v: __m512i,
        max_v: __m512i,
    ) -> (__m512i, u16) {
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            let v0 = _mm512_loadu_si512(ptr.cast());
            let v1 = _mm512_loadu_si512(ptr.add(8).cast());
            let bad0 = _mm512_cmplt_epi64_mask(v0, min_v) | _mm512_cmpgt_epi64_mask(v0, max_v);
            let bad1 = _mm512_cmplt_epi64_mask(v1, min_v) | _mm512_cmpgt_epi64_mask(v1, max_v);
            let lo = _mm512_cvtepi64_epi32(v0);
            let hi = _mm512_cvtepi64_epi32(v1);
            let packed = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi);
            (packed, u16::from(bad0) | (u16::from(bad1) << 8))
        }
    }

    /// 16-lane H-u step decomposition of `u in [-190_465, 190_465]`:
    /// `y = u + 190,465 in [0, 380,930]`, `h = min(floor((y * 4,357,677) >> 38), 5)`,
    /// `U_lo = (y - 63,079 * h) - 32,768`, `U_hi = h - 3`.
    #[inline(always)]
    unsafe fn decompose_u_step_vec(u: __m512i) -> (__m512i, __m512i) {
        // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
        unsafe {
            let y = _mm512_add_epi32(u, _mm512_set1_epi32(B_U as i32));
            let magic = _mm512_set1_epi32(U_MAGIC as i32);
            // Even 32-bit lanes: (y_even * U_MAGIC) >> 38 lands in the low 32 bits of each 64-bit lane.
            let even = _mm512_srli_epi64::<38>(_mm512_mul_epu32(y, magic));
            // Odd 32-bit lanes: shifting right by 38 - 32 = 6 places the quotient in the high 32-bit slot.
            let odd = _mm512_srli_epi64::<6>(_mm512_mul_epu32(_mm512_srli_epi64::<32>(y), magic));
            let q = _mm512_mask_blend_epi32(0xAAAA, even, odd);
            let h = _mm512_min_epu32(q, _mm512_set1_epi32(5));
            let u_lo = _mm512_sub_epi32(
                _mm512_sub_epi32(y, _mm512_mullo_epi32(h, _mm512_set1_epi32(STEP_U))),
                _mm512_set1_epi32(1 << 15),
            );
            let u_hi = _mm512_sub_epi32(h, _mm512_set1_epi32(3));
            (u_lo, u_hi)
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    pub unsafe fn decompose_u_step_16_avx512(u: &[i32; 16]) -> ([i32; 16], [i8; 16]) {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            let uv = _mm512_loadu_si512(u.as_ptr().cast());
            let (u_lo_v, u_hi_v) = decompose_u_step_vec(uv);
            let mut u_lo = [0i32; 16];
            let mut u_hi = [0i8; 16];
            _mm512_storeu_si512(u_lo.as_mut_ptr().cast(), u_lo_v);
            _mm_storeu_si128(u_hi.as_mut_ptr().cast(), _mm512_cvtepi32_epi8(u_hi_v));
            (u_lo, u_hi)
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn encode_record_hu_avx512(
        z: &[[i64; N]; K],
        u: &[[i64; N]; K],
        v: &[[i64; N]; K],
        w1: &[[u8; N]; K],
        wide_slots: [usize; 4],
        narrow_slots: [usize; 8],
        wide: &mut ColumnWriter<'_, i32>,
        narrow: &mut ColumnWriter<'_, i8>,
    ) -> Result<(), &'static str> {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            let z_min = _mm512_set1_epi64(Z_HU_MIN);
            let z_max = _mm512_set1_epi64(Z_HU_MAX);
            let u_min = _mm512_set1_epi64(-B_U);
            let u_max = _mm512_set1_epi64(B_U);
            let v_min = _mm512_set1_epi64(V_MIN);
            let v_max = _mm512_set1_epi64(V_MAX);
            let mask16 = _mm512_set1_epi32(0xFFFF);
            let shift15 = _mm512_set1_epi32(1 << 15);
            let shift17 = _mm512_set1_epi32(1 << 17);
            let two = _mm512_set1_epi32(2);
            let one = _mm512_set1_epi32(1);
            let max_w1 = _mm512_set1_epi32(63);

            for row in 0..K {
                for j in (0..N).step_by(16) {
                    let pos = row * N + j;
                    let (zv, bad_z) = load_and_check_i64_16(z[row].as_ptr().add(j), z_min, z_max);
                    let (uv, bad_u) = load_and_check_i64_16(u[row].as_ptr().add(j), u_min, u_max);
                    let (vv, bad_v) = load_and_check_i64_16(v[row].as_ptr().add(j), v_min, v_max);
                    let w1_128 = _mm_loadu_si128(w1[row].as_ptr().add(j).cast());
                    let w1v = _mm512_cvtepu8_epi32(w1_128);
                    let bad_w1 = _mm512_cmpgt_epu32_mask(w1v, max_w1);

                    if (bad_z | bad_u | bad_v | bad_w1) != 0 {
                        for k in 0..16 {
                            check_cell_hu(
                                z[row][j + k],
                                u[row][j + k],
                                v[row][j + k],
                                w1[row][j + k],
                            )?;
                        }
                    }

                    // z + 2^17 = z_lo + 2^16 z_hi
                    let yz = _mm512_add_epi32(zv, shift17);
                    let z_lo = _mm512_sub_epi32(_mm512_and_si512(yz, mask16), shift15);
                    let z_hi = _mm512_sub_epi32(_mm512_srli_epi32::<16>(yz), two);

                    // u + 190,465 = u_lo + 63,079 u_hi
                    let (u_lo, u_hi) = decompose_u_step_vec(uv);

                    // v + 2^31 = v_lo + 2^16 v_hi
                    let v_lo = _mm512_sub_epi32(_mm512_and_si512(vv, mask16), shift15);
                    let v_hi = _mm512_srai_epi32::<16>(vv);

                    wide.write_16_i32(wide_slots[0], pos, z_lo);
                    wide.write_16_i32(wide_slots[1], pos, u_lo);
                    wide.write_16_i32(wide_slots[2], pos, v_lo);
                    wide.write_16_i32(wide_slots[3], pos, v_hi);

                    narrow.write_16_i8(narrow_slots[0], pos, _mm512_cvtepi32_epi8(z_hi));
                    narrow.write_16_i8(narrow_slots[1], pos, _mm512_cvtepi32_epi8(u_hi));
                    narrow.write_16_i8(
                        narrow_slots[2],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(w1v, one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[3],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<1>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[4],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<2>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[5],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<3>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[6],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<4>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[7],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<5>(w1v), one)),
                    );
                }
            }
            let _ = POSITIONS;
            Ok(())
        }
    }

    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn encode_record_hexact_avx512(
        z: &[[i64; N]; K],
        u: &[[i64; N]; K],
        v: &[[i64; N]; K],
        w1: &[[u8; N]; K],
        wide_slots: [usize; 4],
        narrow_slots: [usize; 9],
        wide: &mut ColumnWriter<'_, i32>,
        narrow: &mut ColumnWriter<'_, i8>,
    ) -> Result<(), &'static str> {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            let z_min = _mm512_set1_epi64(-Z_HEXACT_BOUND);
            let z_max = _mm512_set1_epi64(Z_HEXACT_BOUND);
            let u_min = _mm512_set1_epi64(-B_U);
            let u_max = _mm512_set1_epi64(B_U);
            let v_min = _mm512_set1_epi64(V_MIN);
            let v_max = _mm512_set1_epi64(V_MAX);
            let mask16 = _mm512_set1_epi32(0xFFFF);
            let shift15 = _mm512_set1_epi32(1 << 15);
            let shift17 = _mm512_set1_epi32(1 << 17);
            let z_offset = _mm512_set1_epi32(Z_HEXACT_BOUND as i32);
            let slack_z = _mm512_set1_epi32(SLACK_Z);
            let zero = _mm512_setzero_si512();
            let one = _mm512_set1_epi32(1);
            let max_w1 = _mm512_set1_epi32(63);

            for row in 0..K {
                for j in (0..N).step_by(16) {
                    let pos = row * N + j;
                    let (zv, bad_z) = load_and_check_i64_16(z[row].as_ptr().add(j), z_min, z_max);
                    let (uv, bad_u) = load_and_check_i64_16(u[row].as_ptr().add(j), u_min, u_max);
                    let (vv, bad_v) = load_and_check_i64_16(v[row].as_ptr().add(j), v_min, v_max);
                    let w1_128 = _mm_loadu_si128(w1[row].as_ptr().add(j).cast());
                    let w1v = _mm512_cvtepu8_epi32(w1_128);
                    let bad_w1 = _mm512_cmpgt_epu32_mask(w1v, max_w1);

                    if (bad_z | bad_u | bad_v | bad_w1) != 0 {
                        for k in 0..16 {
                            check_cell_hexact(
                                z[row][j + k],
                                u[row][j + k],
                                v[row][j + k],
                                w1[row][j + k],
                            )?;
                        }
                    }

                    // H-exact z: yz = z + 130,993; a_z = [yz >= 2^17]; rest = yz - 130,915 * a_z
                    let yz = _mm512_add_epi32(zv, z_offset);
                    let az_mask = _mm512_cmpge_epi32_mask(yz, shift17);
                    let a_z = _mm512_mask_mov_epi32(zero, az_mask, one);
                    let rest = _mm512_mask_sub_epi32(yz, az_mask, yz, slack_z);
                    let bz_lo = _mm512_sub_epi32(_mm512_and_si512(rest, mask16), shift15);
                    let bz_hi = _mm512_srli_epi32::<16>(rest);

                    // u + 190,465 = u_lo + 63,079 u_hi
                    let (u_lo, u_hi) = decompose_u_step_vec(uv);

                    // v + 2^31 = v_lo + 2^16 v_hi
                    let v_lo = _mm512_sub_epi32(_mm512_and_si512(vv, mask16), shift15);
                    let v_hi = _mm512_srai_epi32::<16>(vv);

                    wide.write_16_i32(wide_slots[0], pos, bz_lo);
                    wide.write_16_i32(wide_slots[1], pos, u_lo);
                    wide.write_16_i32(wide_slots[2], pos, v_lo);
                    wide.write_16_i32(wide_slots[3], pos, v_hi);

                    narrow.write_16_i8(narrow_slots[0], pos, _mm512_cvtepi32_epi8(bz_hi));
                    narrow.write_16_i8(narrow_slots[1], pos, _mm512_cvtepi32_epi8(u_hi));
                    narrow.write_16_i8(
                        narrow_slots[2],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(w1v, one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[3],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<1>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[4],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<2>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[5],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<3>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[6],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<4>(w1v), one)),
                    );
                    narrow.write_16_i8(
                        narrow_slots[7],
                        pos,
                        _mm512_cvtepi32_epi8(_mm512_and_si512(_mm512_srli_epi32::<5>(w1v), one)),
                    );
                    narrow.write_16_i8(narrow_slots[8], pos, _mm512_cvtepi32_epi8(a_z));
                }
            }
            Ok(())
        }
    }

    /// Pack 6 Boolean `i8` slices of length `len` into one byte per cell (`sum_{t=0..6} b_t << t`)
    /// using 64-lane AVX-512BW operations. Returns `false` if any input byte is outside `{0, 1}`.
    #[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl")]
    pub unsafe fn pack_six_bits_chunk_avx512(
        lanes: &[&[i8]; 6],
        start: usize,
        out: &mut [u8],
    ) -> bool {
        // SAFETY: the enclosing function enables the CPU features its intrinsics need; callers checked those features at run time.
        unsafe {
            let one = _mm512_set1_epi8(1);
            let len = out.len();
            let vec_len = len / 64 * 64;
            let mut bad = 0u64;
            for offset in (0..vec_len).step_by(64) {
                let idx = start + offset;
                let b0 = _mm512_loadu_si512(lanes[0].as_ptr().add(idx).cast());
                let b1 = _mm512_loadu_si512(lanes[1].as_ptr().add(idx).cast());
                let b2 = _mm512_loadu_si512(lanes[2].as_ptr().add(idx).cast());
                let b3 = _mm512_loadu_si512(lanes[3].as_ptr().add(idx).cast());
                let b4 = _mm512_loadu_si512(lanes[4].as_ptr().add(idx).cast());
                let b5 = _mm512_loadu_si512(lanes[5].as_ptr().add(idx).cast());

                bad |= _mm512_cmpgt_epu8_mask(b0, one)
                    | _mm512_cmpgt_epu8_mask(b1, one)
                    | _mm512_cmpgt_epu8_mask(b2, one)
                    | _mm512_cmpgt_epu8_mask(b3, one)
                    | _mm512_cmpgt_epu8_mask(b4, one)
                    | _mm512_cmpgt_epu8_mask(b5, one);

                // Since each b_t is in {0, 1}, 16-bit shifts never cross byte boundaries.
                let p01 = _mm512_or_si512(b0, _mm512_slli_epi16::<1>(b1));
                let p23 = _mm512_or_si512(
                    _mm512_slli_epi16::<2>(b2),
                    _mm512_slli_epi16::<3>(b3),
                );
                let p45 = _mm512_or_si512(
                    _mm512_slli_epi16::<4>(b4),
                    _mm512_slli_epi16::<5>(b5),
                );
                let packed = _mm512_or_si512(p01, _mm512_or_si512(p23, p45));
                _mm512_storeu_si512(out.as_mut_ptr().add(offset).cast(), packed);
            }
            if bad != 0 {
                return false;
            }
            for offset in vec_len..len {
                let idx = start + offset;
                let mut byte = 0u8;
                for t in 0..6 {
                    let b = lanes[t][idx] as u8;
                    if b > 1 {
                        return false;
                    }
                    byte |= b << t;
                }
                out[offset] = byte;
            }
            true
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// [`super::avx512::pack_six_bits_chunk_avx512`] with 32-lane AVX2 operations.
    #[target_feature(enable = "avx2")]
    pub unsafe fn pack_six_bits_chunk_avx2(lanes: &[&[i8]; 6], start: usize, out: &mut [u8]) -> bool {
        // SAFETY: the enclosing function enables AVX2; the caller checked it at run time and
        // every lane holds at least `start + out.len()` bytes.
        unsafe {
            let len = out.len();
            let vec_len = len / 32 * 32;
            // A byte outside {0, 1} has a bit set in 0xFE.
            let high = _mm256_set1_epi8(0xFE_u8 as i8);
            let mut bad = _mm256_setzero_si256();
            for offset in (0..vec_len).step_by(32) {
                let idx = start + offset;
                let load = |t: usize| _mm256_loadu_si256(lanes[t].as_ptr().add(idx).cast());
                let (b0, b1, b2, b3, b4, b5) = (load(0), load(1), load(2), load(3), load(4), load(5));
                let any = _mm256_or_si256(
                    _mm256_or_si256(_mm256_or_si256(b0, b1), _mm256_or_si256(b2, b3)),
                    _mm256_or_si256(b4, b5),
                );
                bad = _mm256_or_si256(bad, _mm256_and_si256(any, high));
                // Each b_t is in {0, 1} (checked above), so 16-bit shifts never cross a byte.
                let p01 = _mm256_or_si256(b0, _mm256_slli_epi16::<1>(b1));
                let p23 = _mm256_or_si256(_mm256_slli_epi16::<2>(b2), _mm256_slli_epi16::<3>(b3));
                let p45 = _mm256_or_si256(_mm256_slli_epi16::<4>(b4), _mm256_slli_epi16::<5>(b5));
                let packed = _mm256_or_si256(p01, _mm256_or_si256(p23, p45));
                _mm256_storeu_si256(out.as_mut_ptr().add(offset).cast(), packed);
            }
            if _mm256_testz_si256(bad, bad) == 0 {
                return false;
            }
            for offset in vec_len..len {
                let idx = start + offset;
                let mut byte = 0u8;
                for t in 0..6 {
                    let b = lanes[t][idx] as u8;
                    if b > 1 {
                        return false;
                    }
                    byte |= b << t;
                }
                out[offset] = byte;
            }
            true
        }
    }
}

/// Pack 6 Boolean `i8` columns of equal length into a single `Vec<u8>` (`six_bits[i] = sum_t b_t[i] << t`).
/// Returns `None` if any entry is outside `{0, 1}`.
pub fn pack_six_bits(lanes: &[&[i8]; 6]) -> Option<Vec<u8>> {
    use std::sync::atomic::{AtomicBool, Ordering};

    let len = lanes[0].len();
    assert!(lanes.iter().all(|l| l.len() == len));
    let ok = AtomicBool::new(true);
    // The kernels use byte compares and shifts only: AVX-512BW without IFMA, or AVX2.
    let avx512 = crate::detect::avx512_int();
    let avx2 = crate::detect::avx2();
    let out = crate::buf::fill_vec_par_chunks(len, 1 << 16, |chunk_idx, dst| {
        if !ok.load(Ordering::Relaxed) {
            dst.fill(0);
            return;
        }
        let start = chunk_idx * (1 << 16);
        #[cfg(target_arch = "x86_64")]
        if avx512 || avx2 {
            // SAFETY: the detector checked the CPU features of the kernel, and every lane
            // holds `len >= start + dst.len()` bytes (asserted above).
            let chunk_ok = unsafe {
                if avx512 {
                    avx512::pack_six_bits_chunk_avx512(lanes, start, dst)
                } else {
                    avx2::pack_six_bits_chunk_avx2(lanes, start, dst)
                }
            };
            if !chunk_ok {
                ok.store(false, Ordering::Relaxed);
            }
            return;
        }
        for (offset, slot) in dst.iter_mut().enumerate() {
            let idx = start + offset;
            let mut byte = 0u8;
            for t in 0..6 {
                let b = lanes[t][idx] as u8;
                if b > 1 {
                    ok.store(false, Ordering::Relaxed);
                    dst.fill(0);
                    return;
                }
                byte |= b << t;
            }
            *slot = byte;
        }
    });
    ok.into_inner().then_some(out)
}

/// Lookup table mapping a 5-bit mask `b & 0x1F` to five 12-bit lane increments at bit offsets
/// `0, 12, 24, 36, 48` inside a `u64` accumulator (each field supports sums up to `4,095`).
const EXPAND_5X12: [u64; 32] = {
    let mut table = [0u64; 32];
    let mut b = 0usize;
    while b < 32 {
        table[b] = (b as u64 & 1)
            | (((b as u64 >> 1) & 1) << 12)
            | (((b as u64 >> 2) & 1) << 24)
            | (((b as u64 >> 3) & 1) << 36)
            | (((b as u64 >> 4) & 1) << 48);
        b += 1;
    }
    table
};

/// Accumulate packed 6-bit inputs `six_bits[zeta]` over the column range `col_start..col_end`
/// of a sparse matrix (`entries` has `k` row indices per column) into `acc0` (lanes 0 through 4 in
/// 12-bit fields) and `acc5` (lane 5 in `u16`).
///
/// Lazy range: each 12-bit field in `acc0` holds `[0, 4095]`; the production bridge row weight
/// cap is `W_1 = W_2 = 2,047 < 4,095`, so no carry ever crosses a 12-bit boundary.
#[inline]
pub fn accumulate_six_lanes_strip(
    six_bits: &[u8],
    entries: &[u32],
    k: usize,
    col_start: usize,
    col_end: usize,
    acc0: &mut [u64],
    acc5: &mut [u16],
) {
    for (zeta, &b) in six_bits[col_start..col_end].iter().enumerate() {
        if b != 0 {
            let w0 = EXPAND_5X12[(b & 0x1F) as usize];
            let w5 = u16::from(b >> 5);
            let col_off = (col_start + zeta) * k;
            for &row in &entries[col_off..col_off + k] {
                let r = row as usize;
                acc0[r] += w0;
                acc5[r] += w5;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_pack(lanes: &[&[i8]; 6]) -> Option<Vec<u8>> {
        (0..lanes[0].len())
            .map(|i| {
                (0..6).try_fold(0_u8, |byte, t| {
                    let b = lanes[t][i] as u8;
                    (b <= 1).then_some(byte | b << t)
                })
            })
            .collect()
    }

    #[test]
    fn six_bit_packing_matches_the_scalar_definition_on_every_kernel() {
        // A length that is neither a multiple of 32 nor of 64 exercises the tails.
        let len = 3 * 64 + 37;
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let columns: Vec<Vec<i8>> = (0..6)
            .map(|_| {
                (0..len)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state & 1) as i8
                    })
                    .collect()
            })
            .collect();
        for bad in [None, Some((2, 5)), Some((4, len - 1)), Some((0, 100))] {
            let mut columns = columns.clone();
            if let Some((t, i)) = bad {
                columns[t][i] = if i % 2 == 0 { 2 } else { -1 };
            }
            let lanes: [&[i8]; 6] = std::array::from_fn(|t| columns[t].as_slice());
            let want = scalar_pack(&lanes);
            assert_eq!(want.is_none(), bad.is_some());
            assert_eq!(pack_six_bits(&lanes), want, "dispatched, {bad:?}");
            #[cfg(target_arch = "x86_64")]
            {
                let mut out = vec![0_u8; len];
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU has AVX2 and every lane has `len` bytes.
                    let ok = unsafe { avx2::pack_six_bits_chunk_avx2(&lanes, 0, &mut out) };
                    assert_eq!(ok, want.is_some(), "AVX2, {bad:?}");
                    if let Some(want) = &want {
                        assert_eq!(&out, want, "AVX2");
                    }
                }
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512dq")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl")
                {
                    // SAFETY: the CPU has AVX-512F/DQ/BW/VL and every lane has `len` bytes.
                    let ok = unsafe { avx512::pack_six_bits_chunk_avx512(&lanes, 0, &mut out) };
                    assert_eq!(ok, want.is_some(), "AVX-512, {bad:?}");
                    if let Some(want) = &want {
                        assert_eq!(&out, want, "AVX-512");
                    }
                }
            }
        }
    }
}

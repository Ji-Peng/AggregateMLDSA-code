#[cfg(not(target_arch = "aarch64"))]
use crate::{field::F256Unreduced, zerocheck::univariate_skip_optimized::F128};
use crate::{
    field::gf2_8::gf8_reduce,
    zerocheck::univariate_skip_optimized::{F8, InvNttTableByteSingleGf8, N_CHUNKS},
};

/// Scalar bit transpose for C.
///
/// Input bytes are indexed by `(x_small * 8 + b_chunk)` with bit `t`
/// selecting a lane. Output bytes are indexed by `(b_chunk * 8 + t)` with
/// bit `x_small` selecting the inner polynomial coefficient.
#[allow(dead_code)]
pub(in super::super) fn bit_transpose_64bytes_scalar(input: &[u8; 64], output: &mut [u8; 64]) {
    output.fill(0);
    for (byte_idx, &input_byte) in input.iter().enumerate() {
        let x_small = byte_idx / 8;
        let b_chunk = byte_idx % 8;
        for t in 0..8 {
            if (input_byte >> t) & 1 != 0 {
                output[b_chunk * 8 + t] |= 1u8 << x_small;
            }
        }
    }
}

/// Scalar shift-reduce kernel and oracle for the architecture backends.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn shift_reduce_inner_ab_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    let mut acc = [0u16; 64];
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    for k in 0..8 {
        let chunk_off = byte_base_b + k * N_CHUNKS;
        inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
        inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
        for lane in 0..64 {
            let y = (a_col[lane] * b_col[lane]).0 as u16;
            acc[lane] ^= y << k;
        }
    }
    for lane in 0..64 {
        out[lane] = gf8_reduce(acc[lane]);
    }
}

/// GF(2^8) exponent table for the generator 3 (AES polynomial
/// x^8 + x^4 + x^3 + x + 1): `EXP[i] = 3^(i mod 255)` for `i < 510`, and zero
/// for `510 <= i < 1024` so that the `LOG` sentinel of zero sends any product
/// with a zero factor to zero.
#[allow(dead_code)]
const GF8_EXP: [u8; 1024] = gf8_exp_log().0;
/// GF(2^8) discrete logarithm base 3; `LOG[0] = 510` is the zero sentinel
/// (`LOG[a] + LOG[b] >= 510` exactly when a factor is zero, and the sum stays
/// below 1024).
#[allow(dead_code)]
const GF8_LOG: [u16; 256] = gf8_exp_log().1;

#[allow(dead_code)]
const fn gf8_exp_log() -> ([u8; 1024], [u16; 256]) {
    let mut exp = [0u8; 1024];
    let mut log = [510u16; 256];
    let mut x: u8 = 1;
    let mut i = 0;
    while i < 255 {
        exp[i] = x;
        exp[i + 255] = x;
        log[x as usize] = i as u16;
        // x * 3 = x * (x_poly + 1): xtime(x) ^ x, reduced by 0x1B.
        let xt = (x << 1) ^ if x & 0x80 != 0 { 0x1B } else { 0 };
        x ^= xt;
        i += 1;
    }
    (exp, log)
}

/// Shift-reduce kernel with GF(2^8) products from L1-resident log/exp tables
/// (1.25 KiB) instead of a software carry-less multiply: one product is two
/// `LOG` loads, an add and an `EXP` load. Output byte-identical to
/// [`shift_reduce_inner_ab_scalar`]: every product is the exact field
/// product, and the per-lane accumulation and final reduction are the same.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn shift_reduce_inner_ab_logexp(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    assert!(a_col.len() == 64 && b_col.len() == 64);
    // acc[lane] holds XOR_k (a_k b_k) x^k with a_k b_k reduced (8 bits) and
    // k < 8: at most 15 bits, so u16 holds it without loss.
    let mut acc = [0u16; 64];
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    for k in 0..8 {
        let chunk_off = byte_base_b + k * N_CHUNKS;
        inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
        inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
        for ((acc, a), b) in acc.iter_mut().zip(a_col.iter()).zip(b_col.iter()) {
            let e = GF8_LOG[a.0 as usize] + GF8_LOG[b.0 as usize];
            *acc ^= (GF8_EXP[e as usize] as u16) << k;
        }
    }
    for (o, a) in out.iter_mut().zip(acc.iter()) {
        *o = gf8_reduce(*a);
    }
}

/// Bit-sliced shift-reduce kernel: the 64 lanes of every GF(2^8) row are held
/// as eight `u64` bit planes, so one AND/XOR plane operation acts on all 64
/// lanes. Per `k` the rows of `a` and `b` are built directly in plane form
/// from the table's bit-plane image (one cache line per input byte, no byte
/// transpose), multiplied by schoolbook on planes (64 AND, 49 XOR for 64
/// products), and the `x^k` shift is an index offset into the 22 accumulator
/// planes. One reduction modulo `x^8 + x^4 + x^3 + x + 1` per window, then
/// one 64-lane plane-to-byte transpose. Output byte-identical to
/// [`shift_reduce_inner_ab_scalar`]: both compute the canonical residue of
/// `XOR_k a_k b_k x^k` per lane. Requires `ell == 64`.
#[allow(dead_code)]
pub(in super::super) fn shift_reduce_inner_ab_bitsliced(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    let planes = inv_table.planes();
    assert!(inv_table.ell == 64 && N_CHUNKS == 8 && planes.len() == 8 * 256);
    #[inline(always)]
    fn row_planes(bytes: &[u8], planes: &[crate::ntt::inv_table::BytePlanes]) -> [u64; 8] {
        let mut p = planes[bytes[0] as usize].0;
        for (b, &byte) in bytes.iter().enumerate().skip(1) {
            let q = &planes[b * 256 + byte as usize].0;
            for j in 0..8 {
                p[j] ^= q[j];
            }
        }
        p
    }
    // acc[t]: plane of the coefficient of x^t of XOR_k a_k b_k x^k, t <= 21
    // (product degree <= 14, shift k <= 7). XOR accumulation, exact.
    let mut acc = [0u64; 22];
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    for k in 0..8 {
        let off = byte_base_b + k * N_CHUNKS;
        let a = row_planes(&a_packed[off..off + N_CHUNKS], planes);
        let b = row_planes(&b_packed[off..off + N_CHUNKS], planes);
        for i in 0..8 {
            for j in 0..8 {
                acc[i + j + k] ^= a[i] & b[j];
            }
        }
    }
    // x^t = x^(t-4) + x^(t-5) + x^(t-7) + x^(t-8) for t >= 8, from the top
    // down so folded terms that land at t' >= 8 fold again.
    for t in (8..22).rev() {
        let h = acc[t];
        acc[t - 4] ^= h;
        acc[t - 5] ^= h;
        acc[t - 7] ^= h;
        acc[t - 8] ^= h;
    }
    // Plane-to-byte transpose: lanes 8 g .. 8 g + 8 are byte g of every plane.
    for g in 0..8 {
        let mut m = 0u64;
        for (j, plane) in acc[..8].iter().enumerate() {
            m |= ((plane >> (8 * g)) & 0xFF) << (8 * j);
        }
        // Byte j = bit plane j (bit t = lane 8 g + t); transpose so byte t is
        // lane 8 g + t's value.
        let t = (m ^ (m >> 7)) & 0x00AA_00AA_00AA_00AA;
        m ^= t ^ (t << 7);
        let t = (m ^ (m >> 14)) & 0x0000_CCCC_0000_CCCC;
        m ^= t ^ (t << 14);
        let t = (m ^ (m >> 28)) & 0x0000_0000_F0F0_F0F0;
        m ^= t ^ (t << 28);
        out[8 * g..8 * g + 8].copy_from_slice(&m.to_le_bytes());
    }
}

/// Word-level bit transpose for C, same contract as
/// [`bit_transpose_64bytes_scalar`]: for each `b_chunk` the eight input bytes
/// `input[x_small * 8 + b_chunk]` form an 8 x 8 bit matrix (row `x_small`,
/// column `t`), transposed in one `u64` with three masked delta swaps and
/// stored as the eight output bytes `output[b_chunk * 8 + t]`.
#[allow(dead_code)]
pub(in super::super) fn bit_transpose_64bytes_words(input: &[u8; 64], output: &mut [u8; 64]) {
    for b_chunk in 0..8 {
        // Row x_small in byte x_small of g (bit t = column t).
        let mut g = 0u64;
        for x_small in 0..8 {
            g |= (input[x_small * 8 + b_chunk] as u64) << (8 * x_small);
        }
        // Transpose: bit 8 r + c moves to bit 8 c + r.
        let t = (g ^ (g >> 7)) & 0x00AA_00AA_00AA_00AA;
        g ^= t ^ (t << 7);
        let t = (g ^ (g >> 14)) & 0x0000_CCCC_0000_CCCC;
        g ^= t ^ (t << 14);
        let t = (g ^ (g >> 28)) & 0x0000_0000_F0F0_F0F0;
        g ^= t ^ (t << 28);
        output[b_chunk * 8..b_chunk * 8 + 8].copy_from_slice(&g.to_le_bytes());
    }
}

/// Window accumulation for one `x_outer_lo`: per lane, the convert-table sum
/// of the 16 `b_med` bytes times `eq_lo_val`, XORed as a 256-bit unreduced
/// product into `acc_*` (exact in `F_2[x]`, no term bound; the caller reduces
/// each lane once per `x_hi` window). Three CLMUL-pair products instead of
/// reduced multiplies, and no reduction per window.
#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    acc_ab: &mut [F256Unreduced; 64],
    acc_c: &mut [F256Unreduced; 64],
) {
    for lane in 0..64 {
        let mut converted_ab = F128::ZERO;
        let mut converted_c = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            converted_ab += convert[table_base + chunk_ab_bytes[b_med][lane] as usize];
            converted_c += convert[table_base + chunk_c_bytes[b_med][lane] as usize];
        }
        acc_ab[lane] ^= converted_ab.mul_unreduced(eq_lo_val);
        acc_c[lane] ^= converted_c.mul_unreduced(eq_lo_val);
    }
}

/// Two-bank C variant of [`accumulate_convert`]; same unreduced accumulation.
#[allow(clippy::too_many_arguments)]
#[cfg(not(target_arch = "aarch64"))]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    acc_ab: &mut [F256Unreduced; 64],
    acc_c_0: &mut [F256Unreduced; 64],
    acc_c_1: &mut [F256Unreduced; 64],
) {
    for lane in 0..64 {
        let mut converted_ab = F128::ZERO;
        let mut converted_c_0 = F128::ZERO;
        let mut converted_c_1 = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            let c = chunk_c_bytes[b_med][lane] as usize;
            converted_ab += convert[table_base + chunk_ab_bytes[b_med][lane] as usize];
            converted_c_0 += convert[table_base + (c & 0x55)];
            converted_c_1 += convert[table_base + (c & 0xaa)];
        }
        acc_ab[lane] ^= converted_ab.mul_unreduced(eq_lo_val);
        acc_c_0[lane] ^= converted_c_0.mul_unreduced(eq_lo_val);
        acc_c_1[lane] ^= converted_c_1.mul_unreduced(eq_lo_val);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GF8_EXP, GF8_LOG, bit_transpose_64bytes_scalar, bit_transpose_64bytes_words,
        shift_reduce_inner_ab_bitsliced, shift_reduce_inner_ab_logexp,
        shift_reduce_inner_ab_scalar,
    };
    use crate::{
        ntt::AdditiveNttGf8,
        test_rng::Rng,
        zerocheck::univariate_skip_optimized::{F8, InvNttTableByteSingleGf8, N_CHUNKS},
    };

    #[test]
    fn logexp_products_are_exact() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let e = GF8_LOG[a as usize] + GF8_LOG[b as usize];
                assert_eq!(F8(GF8_EXP[e as usize]), F8(a) * F8(b), "{a} * {b}");
            }
        }
    }

    #[test]
    fn word_transpose_matches_scalar() {
        let mut rng = Rng::new(0x7A5E);
        for _ in 0..256 {
            let mut input = [0u8; 64];
            rng.fill_bytes(&mut input);
            let (mut want, mut got) = ([0u8; 64], [0u8; 64]);
            bit_transpose_64bytes_scalar(&input, &mut want);
            bit_transpose_64bytes_words(&input, &mut got);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn logexp_inner_matches_scalar_inner() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(6, F8(1u8 << 6));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let mut rng = Rng::new(0x10_6E8);
        let n = 1 << 14;
        let mut a = vec![0u8; n];
        let mut b = vec![0u8; n];
        rng.fill_bytes(&mut a);
        rng.fill_bytes(&mut b);
        // Sparse rows too: zero factors exercise the LOG sentinel.
        for i in (0..n).step_by(3) {
            a[i] = 0;
        }
        let (mut a_col, mut b_col) = ([F8::ZERO; 64], [F8::ZERO; 64]);
        for base in [0usize, 64, 1024, 4096, n - 16 * N_CHUNKS * 8] {
            for b_med in 0..16 {
                if base + (b_med + 1) * N_CHUNKS * 8 > n {
                    continue;
                }
                let (mut want, mut got) = ([0u8; 64], [0u8; 64]);
                shift_reduce_inner_ab_scalar(
                    &a, &b, &table, base, b_med, &mut want, &mut a_col, &mut b_col,
                );
                shift_reduce_inner_ab_logexp(
                    &a, &b, &table, base, b_med, &mut got, &mut a_col, &mut b_col,
                );
                assert_eq!(got, want, "base {base} b_med {b_med}");
                let mut sliced = [0u8; 64];
                shift_reduce_inner_ab_bitsliced(&a, &b, &table, base, b_med, &mut sliced);
                assert_eq!(sliced, want, "bit-sliced, base {base} b_med {b_med}");
            }
        }
    }
}

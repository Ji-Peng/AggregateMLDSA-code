//! Lazy radix-51 contractions; coordinate planes are never materialized as limbs.
use crate::{
    detect::{backend, Backend},
    k::Planes,
    scalar, F, K,
};
use jolt_field::Ring;
fn check(a: Planes<'_>, b: Planes<'_>) {
    let n = a.0.len();
    assert_eq!(a.1.len(), n);
    assert_eq!(b.0.len(), n);
    assert_eq!(b.1.len(), n);
}
pub fn fp_dot(a: &[u64], b: &[u64]) -> u64 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active (checked here), and the lengths asserted above bound
        // every read and write.
        unsafe {
            return fp_vector(a, b);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 64 {
        if let Some(x) = crate::wide::dispatch!(fp_w8(a, b), fp_w4(a, b)) {
            return x;
        }
    }
    scalar::fp_dot(a, b)
}
pub fn k_dot(a: Planes<'_>, b: Planes<'_>) -> K {
    check(a, b);
    #[cfg(target_arch = "x86_64")]
    if a.0.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active (checked here), and the lengths asserted above bound
        // every read and write.
        unsafe {
            return k_vector(a, b);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if a.0.len() >= 64 {
        if let Some(x) = crate::wide::dispatch!(k_w8(a, b), k_w4(a, b)) {
            return x;
        }
    }
    scalar::k_dot(a, b)
}
pub fn k_base_dot(a: Planes<'_>, b: &[u64]) -> K {
    check(a, (b, b));
    #[cfg(target_arch = "x86_64")]
    if b.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active (checked here), and the lengths asserted above bound
        // every read and write.
        unsafe {
            return base_vector(a, b);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if b.len() >= 64 {
        if let Some(x) = crate::wide::dispatch!(base_w8(a, b), base_w4(a, b)) {
            return x;
        }
    }
    scalar::k_base_dot(a, b)
}
/// Signed one-limb dot. The API returns the shifted dot and correction separately.
pub fn k_int_dot_parts(a: Planes<'_>, b: &[i64]) -> (K, K) {
    check(a, (a.0, a.1));
    assert_eq!(a.0.len(), b.len());
    assert!(b.iter().all(|&x| x > -(1i64 << 51) && x < (1i64 << 51)));
    #[cfg(target_arch = "x86_64")]
    if b.len() >= 64 && backend() == Backend::Avx512 {
        // SAFETY: the IFMA backend is active (checked here), and the lengths asserted above bound
        // every read and write.
        unsafe {
            return int_vector(a, b.as_ptr().cast(), b.len(), 1 << 51);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if b.len() >= 64 {
        if let Some((dot, sum)) = crate::wide::dispatch!(int_w8(a, b), int_w4(a, b)) {
            // The shifted form `sum w (k + 2^51)` and its correction `-2^51 sum w`.
            let shift = K::from_u64(1 << 51);
            return (dot + sum * shift, -sum * shift);
        }
    }
    let mut shifted = K::from_u64(0);
    let mut sum = K::from_u64(0);
    for (i, &k) in b.iter().enumerate() {
        let w = scalar::get(a, i);
        shifted += w * K::from_u64((k + (1i64 << 51)) as u64);
        sum += w;
    }
    (shifted, -sum * K::from_u64(1 << 51))
}

pub fn k_int_dot(a: Planes<'_>, b: &[i64]) -> K {
    let (x, c) = k_int_dot_parts(a, b);
    x + c
}
pub fn k_int2_dot(a: Planes<'_>, lo: &[u64], hi: &[i64]) -> K {
    check(a, (a.0, a.1));
    assert_eq!(a.0.len(), lo.len());
    assert_eq!(lo.len(), hi.len());
    assert!(lo.iter().all(|&x| x < (1 << 52)));
    #[cfg(target_arch = "x86_64")]
    if lo.len() >= 64
        && backend() == Backend::Avx512
        && hi.iter().all(|&x| x > -(1i64 << 51) && x < (1i64 << 51))
    {
        // SAFETY: the IFMA backend is active (checked here), and the lengths asserted above bound
        // every read and write.
        unsafe {
            let (l, _) = int_vector(a, lo.as_ptr(), lo.len(), 0);
            let (h, c) = int_vector(a, hi.as_ptr().cast(), hi.len(), 1 << 51);
            return l + (h + c) * K::from_u64(1 << 52);
        }
    }
    #[cfg(target_arch = "x86_64")]
    if lo.len() >= 64 {
        // `lo < 2^52` is canonical, so both limbs are signed-integer dots.
        // SAFETY: `u64` and `i64` share size and alignment; every `lo < 2^52` reads unchanged.
        let lo: &[i64] = unsafe { std::slice::from_raw_parts(lo.as_ptr().cast(), lo.len()) };
        if let Some(((l, _), (h, _))) =
            crate::wide::dispatch!((int_w8(a, lo), int_w8(a, hi)), (int_w4(a, lo), int_w4(a, hi)))
        {
            return l + h * K::from_u64(1 << 52);
        }
    }
    scalar::k_int2_dot(a, lo, hi)
}
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn fp_vector(a: &[u64], b: &[u64]) -> u64 {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // `fp_dot` asserts `b.len() == a.len()`, and every vector load ends at or below
    // `end <= n <= a.len()` (the second chain loads only when `i + 8 < end`).
    unsafe {
        use crate::x86::*;
        let n = a.len() / 8 * 8;
        let mut sum = z();
        // Each digit gains <2^52 per lane term. Merge two chains after <=1024 total
        // terms per lane; exact signed-reduction bound is 2047, not 2048.
        for start in (0..n).step_by(1024 * 8) {
            let end = (start + 1024 * 8).min(n);
            let mut c0 = a5();
            let mut c1 = a5();
            for i in (start..end).step_by(16) {
                mac5(
                    &mut c0,
                    split51(load(a.as_ptr().add(i))),
                    split51(load(b.as_ptr().add(i))),
                );
                if i + 8 < end {
                    mac5(
                        &mut c1,
                        split51(load(a.as_ptr().add(i + 8))),
                        split51(load(b.as_ptr().add(i + 8))),
                    );
                }
            }
            sum = add_mod(sum, red5(add5(c0, c1)));
        }
        let tail = scalar::fp_dot(&a[n..], &b[n..]);
        {
            let (r, c) = hsum(sum).overflowing_add(tail);
            let r = r.wrapping_add(u64::from(c) * 59);
            r.min(r.wrapping_sub(crate::P))
        }
    }
}
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn k_vector(a: Planes<'_>, b: Planes<'_>) -> K {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // `check` asserts that all four planes hold `a.0.len()` words, and every vector load
    // ends at or below `n <= a.0.len()`.
    unsafe {
        use crate::x86::*;
        let n = a.0.len() / 8 * 8;
        let mut re = z();
        let mut im = z();
        // Three acc5 chains; each digit gains <2^52 (q <2^28). The real
        // combination triples digit bounds. Flush after 512 terms/lane; exact bound 682.
        for start in (0..n).step_by(512 * 8) {
            let end = (start + 512 * 8).min(n);
            let mut ac = a5();
            let mut bd = a5();
            let mut abcd = a5();
            for i in (start..end).step_by(8) {
                let aa = split51(load(a.0.as_ptr().add(i)));
                let bb = split51(load(a.1.as_ptr().add(i)));
                let cc = split51(load(b.0.as_ptr().add(i)));
                let dd = split51(load(b.1.as_ptr().add(i)));
                mac5(&mut ac, aa, cc);
                mac5(&mut bd, bb, dd);
                mac5(&mut abcd, add51(aa, bb), add51(cc, dd));
            }
            re = add_mod(re, red5(add5(ac, add5(bd, bd))));
            im = add_mod(im, red5(sub5(sub5(abcd, ac), bd)));
        }
        K::new(F::from_u64(hsum(re)), F::from_u64(hsum(im)))
            + scalar::k_dot((&a.0[n..], &a.1[n..]), (&b.0[n..], &b.1[n..]))
    }
}
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
#[cfg(target_arch = "x86_64")]
unsafe fn base_vector(a: Planes<'_>, b: &[u64]) -> K {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // `k_base_dot` asserts that both planes of `a` hold `b.len()` words, and every vector
    // load ends at or below `n <= b.len()`.
    unsafe {
        use crate::x86::*;
        let n = b.len() / 8 * 8;
        let mut re = z();
        let mut im = z();
        // Two independent acc5 chains, each <1024*2^52 per digit at flush.
        for start in (0..n).step_by(1024 * 8) {
            let end = (start + 1024 * 8).min(n);
            let mut c0 = a5();
            let mut c1 = a5();
            for i in (start..end).step_by(8) {
                let bb = split51(load(b.as_ptr().add(i)));
                mac5(&mut c0, split51(load(a.0.as_ptr().add(i))), bb);
                mac5(&mut c1, split51(load(a.1.as_ptr().add(i))), bb);
            }
            re = add_mod(re, red5(c0));
            im = add_mod(im, red5(c1));
        }
        K::new(F::from_u64(hsum(re)), F::from_u64(hsum(im)))
            + scalar::k_base_dot((&a.0[n..], &a.1[n..]), &b[n..])
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn int_vector(a: Planes<'_>, b: *const u64, len: usize, shift: u64) -> (K, K) {
    // SAFETY: the target features are enabled here and the caller checked them at run time.
    // The callers assert that both planes of `a` hold `len` words and pass `b` from a slice
    // of `len` words; every vector load ends at or below `n <= len`.
    unsafe {
        use crate::x86::*;
        let n = len / 8 * 8;
        let mut re = z();
        let mut im = z();
        let mut sr = z();
        let mut si = z();
        // Per term the product middle digit gains <2*2^52, the low digit <2^52.
        // Weight sums gain <2^52 in their low digit. Flush at 512 terms/lane,
        // so all unsigned acc3 digits stay <2^63 and the top digit <2^34.
        for start in (0..n).step_by(512 * 8) {
            let end = (start + 512 * 8).min(n);
            let mut c0 = a3();
            let mut c1 = a3();
            let mut w0 = a3();
            let mut w1 = a3();
            for i in (start..end).step_by(8) {
                let k = add(load(b.add(i)), bc(shift));
                let x = split52(load(a.0.as_ptr().add(i)));
                let y = split52(load(a.1.as_ptr().add(i)));
                c0.a = lo(c0.a, x.l, k);
                c0.b = hi(c0.b, x.l, k);
                c0.b = lo(c0.b, x.h, k);
                c0.c = hi(c0.c, x.h, k);
                c1.a = lo(c1.a, y.l, k);
                c1.b = hi(c1.b, y.l, k);
                c1.b = lo(c1.b, y.h, k);
                c1.c = hi(c1.c, y.h, k);
                w0.a = add(w0.a, x.l);
                w0.b = add(w0.b, x.h);
                w1.a = add(w1.a, y.l);
                w1.b = add(w1.b, y.h);
            }
            re = add_mod(re, red3(c0));
            im = add_mod(im, red3(c1));
            sr = add_mod(sr, red3(w0));
            si = add_mod(si, red3(w1));
        }
        let mut shifted = K::new(F::from_u64(hsum(re)), F::from_u64(hsum(im)));
        let mut sum = K::new(F::from_u64(hsum(sr)), F::from_u64(hsum(si)));
        for i in n..len {
            let w = scalar::get(a, i);
            shifted += w * K::from_u64((*b.add(i)).wrapping_add(shift));
            sum += w;
        }
        (shifted, -sum * K::from_u64(shift))
    }
}

/// First-round coefficients for sum_i w_i g_i, paired along the low variable.
/// The integer combination is produced in its final contiguous stream by the caller.
pub fn weight_round(w:Planes<'_>,g:&[i64])->[K;3] {
 check(w,(w.0,w.1));assert_eq!(w.0.len(),g.len());assert_eq!(g.len()%2,0);
 #[cfg(target_arch="x86_64")]
 // SAFETY: the IFMA backend is active, and the planes and `g` have equal even lengths
 // (asserted above).
 if g.len()>=64 && backend()==Backend::Avx512 {unsafe {return weight_vector(w,g);}}
 #[cfg(target_arch="x86_64")]
 if g.len()>=64 {if let Some(x)=crate::wide::dispatch!(weight_w8(w,g),weight_w4(w,g)) {return x;}}
 let mut out=[K::from_u64(0);3];
 for i in (0..g.len()).step_by(2) {let a=scalar::get(w,i);let b=scalar::get(w,i+1);let ga=K::from_i64(g[i]);let gb=K::from_i64(g[i+1]);out[0]+=a*ga;out[1]+=b*gb;out[2]+=(b-a)*(gb-ga);}
 out
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
/// # Safety
/// Requires the IFMA features, and both planes of `w` as long as `g`.
unsafe fn weight_vector(w:Planes<'_>,g:&[i64])->[K;3] {unsafe {
 use crate::x86::*;use std::arch::x86_64::*;
 let idx=_mm512_set_epi64(14,12,10,8,6,4,2,0);let n=g.len()/16*16;let mut out=[K::from_u64(0);3];
 for part in 0..3 {let mut re=z();let mut im=z();
  // Each acc5 digit gains <2^52 per term. Flush at 1024 products/lane,
  // below the exact 2047 bound, even on the canonical residues of signed g.
  for start in (0..n).step_by(1024*16) {let end=(start+1024*16).min(n);let mut c0=a5();let mut c1=a5();
   for i in (start..end).step_by(16) {
    let offset=if part==1 {1} else {0};
    let mut a=_mm512_i64gather_epi64::<8>(idx,w.0.as_ptr().add(i+offset).cast());
    let mut b=_mm512_i64gather_epi64::<8>(idx,w.1.as_ptr().add(i+offset).cast());
    let mut k=_mm512_i64gather_epi64::<8>(idx,g.as_ptr().add(i+offset).cast());
    k=_mm512_mask_add_epi64(k,_mm512_cmplt_epi64_mask(k,z()),k,bc(crate::P));
    if part==2 {
     a=sub_mod(_mm512_i64gather_epi64::<8>(idx,w.0.as_ptr().add(i+1).cast()),a);
     b=sub_mod(_mm512_i64gather_epi64::<8>(idx,w.1.as_ptr().add(i+1).cast()),b);
     let other=_mm512_i64gather_epi64::<8>(idx,g.as_ptr().add(i+1).cast());
     let other=_mm512_mask_add_epi64(other,_mm512_cmplt_epi64_mask(other,z()),other,bc(crate::P));k=sub_mod(other,k);
    }
    let k=split51(k);mac5(&mut c0,split51(a),k);mac5(&mut c1,split51(b),k);
   }
   re=add_mod(re,red5(c0));im=add_mod(im,red5(c1));
  }
  out[part]=K::new(F::from_u64(hsum(re)),F::from_u64(hsum(im)));
 }
 if n<g.len() {let tail=weight_round((&w.0[n..],&w.1[n..]),&g[n..]);for i in 0..3 {out[i]+=tail[i];}}
 out
}}

/// Products per accumulator between reductions in the dots below (two products per
/// coordinate accumulator and step at most, so `2 * CHUNK <= MAC_LIMIT`).
#[cfg(target_arch = "x86_64")]
const CHUNK: usize = crate::wide::MAC_LIMIT / 2;

/// Steps per chunk of [`k_wide`]: its real part `ac + 2 bd` is reduced as one accumulator
/// that carries three products per step, so `3 * K_CHUNK <= MAC_LIMIT` keeps the `d2` limb
/// below the `2^56` that [`red`](crate::wide::red) assumes.
#[cfg(target_arch = "x86_64")]
const K_CHUNK: usize = crate::wide::MAC_LIMIT / 3;

/// [`CHUNK`] for the tests that cross it.
#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) const fn wide_chunk() -> usize {
    CHUNK
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn fp_wide<B: crate::wide::Simd>(a: &[u64], b: &[u64]) -> u64 {
    use crate::wide::*;
    let l = B::LANES;
    let n = a.len() / l * l;
    // SAFETY: the caller guarantees the backend's features; every index is below `n`.
    unsafe {
        let mut sum = zero::<B>();
        for start in (0..n).step_by(CHUNK * l) {
            let end = (start + CHUNK * l).min(n);
            let mut c = acc0::<B>();
            for i in (start..end).step_by(l) {
                mac::<B>(&mut c, split::<B>(B::load(a.as_ptr().add(i))), split::<B>(B::load(b.as_ptr().add(i))));
            }
            sum = B::add_mod(sum, red::<B>(c));
        }
        (F::from_u64(hsum::<B>(sum)) + F::from_u64(scalar::fp_dot(&a[n..], &b[n..]))).to_canonical_u64()
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(fp_wide => fp_w8, fp_w4; (a: &[u64], b: &[u64]) -> u64);

/// `ac`, `bd` and `ad + bc` in three accumulators; the real part `ac + 2 bd` is formed by
/// doubling the `bd` limbs at the reduction (bounds per chunk of [`K_CHUNK`] steps: the real
/// part carries three products per step and `ad + bc` two, at most
/// [`MAC_LIMIT`](crate::wide::MAC_LIMIT) each).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn k_wide<B: crate::wide::Simd>(a: Planes<'_>, b: Planes<'_>) -> K {
    use crate::wide::*;
    let l = B::LANES;
    let n = a.0.len() / l * l;
    // SAFETY: the caller guarantees the backend's features; every index is below `n`.
    unsafe {
        let mut re = zero::<B>();
        let mut im = zero::<B>();
        for start in (0..n).step_by(K_CHUNK * l) {
            let end = (start + K_CHUNK * l).min(n);
            let (mut ac, mut bd, mut ab) = (acc0::<B>(), acc0::<B>(), acc0::<B>());
            for i in (start..end).step_by(l) {
                let (x0, x1) = (split::<B>(B::load(a.0.as_ptr().add(i))), split::<B>(B::load(a.1.as_ptr().add(i))));
                let (y0, y1) = (split::<B>(B::load(b.0.as_ptr().add(i))), split::<B>(B::load(b.1.as_ptr().add(i))));
                mac::<B>(&mut ac, x0, y0);
                mac::<B>(&mut bd, x1, y1);
                mac::<B>(&mut ab, x0, y1);
                mac::<B>(&mut ab, x1, y0);
            }
            // ac takes K_CHUNK products and 2 bd the gain of 2 K_CHUNK: 3 K_CHUNK <= MAC_LIMIT.
            re = B::add_mod(re, red::<B>(add_acc::<B>(ac, dbl_acc::<B>(bd))));
            im = B::add_mod(im, red::<B>(ab));
        }
        hsum_k::<B>((re, im)) + scalar::k_dot((&a.0[n..], &a.1[n..]), (&b.0[n..], &b.1[n..]))
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(k_wide => k_w8, k_w4; (a: Planes<'_>, b: Planes<'_>) -> K);

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn base_wide<B: crate::wide::Simd>(a: Planes<'_>, b: &[u64]) -> K {
    use crate::wide::*;
    let l = B::LANES;
    let n = b.len() / l * l;
    // SAFETY: the caller guarantees the backend's features; every index is below `n`.
    unsafe {
        let mut sum = (zero::<B>(), zero::<B>());
        for start in (0..n).step_by(CHUNK * l) {
            let end = (start + CHUNK * l).min(n);
            let mut c = acck::<B>();
            for i in (start..end).step_by(l) {
                let x = (split::<B>(B::load(a.0.as_ptr().add(i))), split::<B>(B::load(a.1.as_ptr().add(i))));
                kmac_base::<B>(&mut c, x, split::<B>(B::load(b.as_ptr().add(i))));
            }
            sum = addk::<B>(sum, redk::<B>(c));
        }
        hsum_k::<B>(sum) + scalar::k_base_dot((&a.0[n..], &a.1[n..]), &b[n..])
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(base_wide => base_w8, base_w4; (a: Planes<'_>, b: &[u64]) -> K);

/// `(sum w k, sum w)` over signed integers `k` (any `i64`), each read as its canonical residue.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn int_wide<B: crate::wide::Simd>(a: Planes<'_>, b: &[i64]) -> (K, K) {
    use crate::wide::*;
    let l = B::LANES;
    let n = b.len() / l * l;
    // SAFETY: the caller guarantees the backend's features; every index is below `n`.
    unsafe {
        let mut dot = (zero::<B>(), zero::<B>());
        let mut sum = (zero::<B>(), zero::<B>());
        for start in (0..n).step_by(CHUNK * l) {
            let end = (start + CHUNK * l).min(n);
            let mut c = acck::<B>();
            let mut w = acck::<B>();
            for i in (start..end).step_by(l) {
                let (x0, x1) = (B::load(a.0.as_ptr().add(i)), B::load(a.1.as_ptr().add(i)));
                let k = B::residue(B::load(b.as_ptr().add(i).cast()));
                kmac_base::<B>(&mut c, (split::<B>(x0), split::<B>(x1)), split::<B>(k));
                lift::<B>(&mut w.0, x0);
                lift::<B>(&mut w.1, x1);
            }
            dot = addk::<B>(dot, redk::<B>(c));
            sum = addk::<B>(sum, redk::<B>(w));
        }
        let mut dot = hsum_k::<B>(dot);
        let mut sum = hsum_k::<B>(sum);
        for i in n..b.len() {
            let w = scalar::get(a, i);
            dot += w * K::from_i64(b[i]);
            sum += w;
        }
        (dot, sum)
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(int_wide => int_w8, int_w4; (a: Planes<'_>, b: &[i64]) -> (K, K));

/// The three round coefficients of [`weight_round`]: the even and odd rows deinterleaved from
/// `2 LANES` consecutive rows, the signed integers read as canonical residues.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn weight_wide<B: crate::wide::Simd>(w: Planes<'_>, g: &[i64]) -> [K; 3] {
    use crate::wide::*;
    let l = B::LANES;
    let n = g.len() / (2 * l) * (2 * l);
    // SAFETY: the caller guarantees the backend's features; every index is below `n`.
    unsafe {
        let mut sums = [(zero::<B>(), zero::<B>()); 3];
        for start in (0..n).step_by(CHUNK * 2 * l) {
            let end = (start + CHUNK * 2 * l).min(n);
            let mut c = [acck::<B>(); 3];
            for i in (start..end).step_by(2 * l) {
                // Every stream is split in the same item order; the sums do not depend on it.
                let (e0, o0) = B::deinterleave_any(B::load(w.0.as_ptr().add(i)), B::load(w.0.as_ptr().add(i + l)));
                let (e1, o1) = B::deinterleave_any(B::load(w.1.as_ptr().add(i)), B::load(w.1.as_ptr().add(i + l)));
                let (ge, go) = B::deinterleave_any(B::load(g.as_ptr().add(i).cast()), B::load(g.as_ptr().add(i + l).cast()));
                let (ge, go) = (B::residue(ge), B::residue(go));
                kmac_base::<B>(&mut c[0], (split::<B>(e0), split::<B>(e1)), split::<B>(ge));
                kmac_base::<B>(&mut c[1], (split::<B>(o0), split::<B>(o1)), split::<B>(go));
                let d = (split::<B>(B::sub_mod(o0, e0)), split::<B>(B::sub_mod(o1, e1)));
                kmac_base::<B>(&mut c[2], d, split::<B>(B::sub_mod(go, ge)));
            }
            for t in 0..3 {
                sums[t] = addk::<B>(sums[t], redk::<B>(c[t]));
            }
        }
        // A loop, not a closure: closures do not inherit the entry point's target features.
        let mut out = [K::from_u64(0); 3];
        for t in 0..3 {
            out[t] = hsum_k::<B>(sums[t]);
        }
        if n < g.len() {
            let tail = weight_round((&w.0[n..], &w.1[n..]), &g[n..]);
            for t in 0..3 {
                out[t] += tail[t];
            }
        }
        out
    }
}
#[cfg(target_arch = "x86_64")]
crate::wide::entry_points!(weight_wide => weight_w8, weight_w4; (w: Planes<'_>, g: &[i64]) -> [K; 3]);

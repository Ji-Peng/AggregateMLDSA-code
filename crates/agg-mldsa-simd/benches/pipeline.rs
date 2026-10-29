mod common;

use agg_mldsa_simd::{
    dispatch::{Backend, Engine},
    finish::{FinishOut, FinishRow},
    ntt::{Form, Scale},
    params::Prime,
    poly::{ExpandedA, Poly32},
};
use std::hint::black_box;

#[inline(never)]
fn rust_expand_a(e: &Engine, rho: &[u8; 32], out: &mut ExpandedA) {
    e.expand_a_many(std::slice::from_ref(rho), std::slice::from_mut(out));
    black_box(out);
}

#[cfg(cref)]
#[inline(never)]
fn c_expand_a_avx2(rho: &[u8; 32], mat: &mut [[Poly32; 4]; 4]) {
    // SAFETY: Poly32 is repr(C), contains only 256 i32 values, and adds alignment.
    let raw = unsafe { &mut *(mat as *mut _ as *mut [[[i32; 256]; 4]; 4]) };
    agg_mldsa_cref::polyvec_matrix_expand(raw, rho);
    black_box(mat);
}

#[cfg(not(cref))]
#[inline(never)]
fn c_expand_a_avx2(_rho: &[u8; 32], _mat: &mut [[Poly32; 4]; 4]) {}

#[cfg(cref)]
#[no_mangle]
pub unsafe extern "C" fn XURQ_AVX2_shake128x4_squeezeblocks(
    out0: *mut u8,
    out1: *mut u8,
    out2: *mut u8,
    out3: *mut u8,
    nblocks: usize,
    state: *mut u64,
) {
    // SAFETY: Forward to pq-crystals implementation with identical state layout.
    unsafe {
        agg_mldsa_cref::ffi::pqcrystals_dilithium_fips202x4_avx2_shake128x4_squeezeblocks(
            out0, out1, out2, out3, nblocks, state,
        );
    }
}

#[cfg(cref)]
#[inline(never)]
fn c_expand_a_avx512(rho: &[u8; 32], mat: &mut [[Poly32; 4]; 4]) {
    // SAFETY: Poly32 has the same size and coefficient layout, with 64-byte alignment.
    let raw = unsafe { &mut *(mat as *mut _ as *mut [[[i32; 256]; 4]; 4]) };
    agg_mldsa_cref::expand_a_shuffled(raw, rho);
    black_box(mat);
}

#[cfg(not(cref))]
#[inline(never)]
fn c_expand_a_avx512(_rho: &[u8; 32], _mat: &mut [[Poly32; 4]; 4]) {}

#[inline(never)]
fn run_step2_chain_8(
    e: &Engine,
    z_bytes: &[&[u8; 2304]; 8],
    prepared_q: &[&[[Poly32; 4]; 4]; 8],
    prepared_p2: &[&[[Poly32; 4]; 4]; 8],
    t: &[&[[i32; 256]; 4]; 8],
    h: &[&[[u8; 256]; 4]; 8],
    mu: &[&[u8; 64]; 8],
    w1_bytes: &mut [[u8; 768]; 8],
    digests: &mut [[u8; 32]; 8],
) {
    for i in 0..8 {
        let mut z_poly = [Poly32::zero(); 4];
        let _max = e.unpack_z(z_bytes[i], &mut z_poly);
        let mut z_q = z_poly;
        let mut z_p = z_poly;
        for col in 0..4 {
            e.ntt(Prime::Q, &mut z_q[col]);
            e.ntt(Prime::P2, &mut z_p[col]);
        }
        let mut w1 = [[0u8; 256]; 4];
        let mut x_q = Poly32::zero();
        let mut x_p = Poly32::zero();
        let mut finish_out = [FinishOut::zero(); 4];
        for r in 0..4 {
            e.dot4(Prime::Q, &prepared_q[i][r], &z_q, &mut x_q);
            e.intt(Prime::Q, &mut x_q, Scale::Mont, Form::Canonical);
            e.dot4(Prime::P2, &prepared_p2[i][r], &z_p, &mut x_p);
            e.intt(Prime::P2, &mut x_p, Scale::Mont, Form::Canonical);
            let row = FinishRow {
                xq: &x_q,
                xp: &x_p,
                t: &t[i][r],
                h: &h[i][r],
            };
            e.finish_row(row, &mut finish_out[r]).unwrap();
            w1[r] = finish_out[r].w1;
        }
        e.pack_w1(&w1, &mut w1_bytes[i]);
    }
    let inputs: [[&[u8]; 4]; 8] =
        std::array::from_fn(|i| [&mu[i][..], &w1_bytes[i][..], &[][..], &[][..]]);
    e.shake256_many(&inputs, 32, digests.as_flattened_mut());
    black_box(digests);
}

fn main() {
    common::print_system_info();

    let rho = [0x42u8; 32];
    let mut out = ExpandedA::default();
    let mut mat_c = [[Poly32::zero(); 4]; 4];

    // ExpandA includes conversion to the backend NTT order on both sides.
    if let Some(e) = Engine::new(Backend::Avx2) {
        if agg_mldsa_cref::available() {
            rust_expand_a(&e, &rho, &mut out);
            c_expand_a_avx2(&rho, &mut mat_c);
            assert_eq!(out.a_hat, mat_c, "AVX2 ExpandA reference mismatch");
        }
        common::compare(
            "expand_a",
            "avx2",
            || rust_expand_a(&e, &rho, &mut out),
            || c_expand_a_avx2(&rho, &mut mat_c),
            agg_mldsa_cref::available(),
        );
    }

    if let Some(e) = Engine::new(Backend::Avx512) {
        if agg_mldsa_cref::available() {
            rust_expand_a(&e, &rho, &mut out);
            c_expand_a_avx512(&rho, &mut mat_c);
            assert_eq!(out.a_hat, mat_c, "AVX-512 ExpandA reference mismatch");
        }
        common::compare(
            "expand_a",
            "avx512",
            || rust_expand_a(&e, &rho, &mut out),
            || c_expand_a_avx512(&rho, &mut mat_c),
            agg_mldsa_cref::available(),
        );
    }

    // 2. Step-2 kernel chain per backend (Rust-only per-record timings)
    let test_records = agg_witness::gen::records(12345, 8);
    let backends = [
        ("scalar", Backend::Scalar),
        ("avx2", Backend::Avx2),
        ("avx512", Backend::Avx512),
    ];

    for (name, b) in backends {
        let Some(e) = Engine::new(b) else { continue };

        let inputs: Vec<(&[u8], &[u8], [u8; 32])> = test_records
            .iter()
            .map(|r| {
                let mut c_tilde = [0u8; 32];
                c_tilde.copy_from_slice(&r.sig[..32]);
                (r.pk.as_slice(), r.msg.as_slice(), c_tilde)
            })
            .collect();
        let inst = agg_witness::instance::PublicInstance::new_with(&e, &inputs).unwrap();

        let mut z_bytes: [&[u8; 2304]; 8] = [&[0u8; 2304]; 8];
        let mut hints: [[[u8; 256]; 4]; 8] = [[[0u8; 256]; 4]; 8];
        let mut prepared_q = [&*inst.records[0].key.prepared.a_hat_q; 8];
        let mut prepared_p2 = [&*inst.records[0].key.prepared.a_hat_p2; 8];
        let mut t = [&*inst.records[0].t; 8];
        let mut mu = [&inst.records[0].mu; 8];

        for i in 0..8 {
            let sig = &test_records[i].sig;
            let (_c_tilde, rest) = sig.split_at(32);
            let (z_slice, hint_slice) = rest.split_at(2304);
            z_bytes[i] = z_slice.try_into().unwrap();
            hints[i] = agg_witness::native::hint_bit_unpack(hint_slice).unwrap();
            prepared_q[i] = &*inst.records[i].key.prepared.a_hat_q;
            prepared_p2[i] = &*inst.records[i].key.prepared.a_hat_p2;
            t[i] = &inst.records[i].t;
            mu[i] = &inst.records[i].mu;
        }

        let h_refs: [&[[u8; 256]; 4]; 8] = std::array::from_fn(|i| &hints[i]);

        // Measure individual components per record
        // (a) unpack_z
        let mut z_poly = [Poly32::zero(); 4];
        let unpack_stats = common::measure_kernel(|| {
            black_box(e.unpack_z(z_bytes[0], &mut z_poly));
        });
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            "unpack_z", name, unpack_stats.median, "-", "-", "RUST_ONLY"
        );

        // (b) 16 transforms (4 fwd Q, 4 fwd P2, 4 inv Q, 4 inv P2)
        let ntt16_stats = common::measure_kernel(|| {
            let mut z_q_bench = z_poly;
            let mut z_p_bench = z_poly;
            let mut x_q_bench = Poly32::zero();
            let mut x_p_bench = Poly32::zero();
            for col in 0..4 {
                e.ntt(Prime::Q, &mut z_q_bench[col]);
                e.ntt(Prime::P2, &mut z_p_bench[col]);
            }
            for _r in 0..4 {
                e.intt(Prime::Q, &mut x_q_bench, Scale::Mont, Form::Canonical);
                e.intt(Prime::P2, &mut x_p_bench, Scale::Mont, Form::Canonical);
                black_box((&x_q_bench, &x_p_bench));
            }
            black_box((&z_q_bench, &z_p_bench, &x_q_bench, &x_p_bench));
        });
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            "16_transforms", name, ntt16_stats.median, "-", "-", "RUST_ONLY"
        );

        // Precompute honest forward NTT of z for record 0
        let mut honest_zq = z_poly;
        let mut honest_zp = z_poly;
        for col in 0..4 {
            e.ntt(Prime::Q, &mut honest_zq[col]);
            e.ntt(Prime::P2, &mut honest_zp[col]);
        }

        // (c) 8 dots (4 Q dot4 + 4 P2 dot4)
        let dots_stats = common::measure_kernel(|| {
            let mut xq = Poly32::zero();
            let mut xp = Poly32::zero();
            for r in 0..4 {
                e.dot4(Prime::Q, &prepared_q[0][r], &honest_zq, &mut xq);
                e.dot4(Prime::P2, &prepared_p2[0][r], &honest_zp, &mut xp);
                black_box((&xq, &xp));
            }
            black_box((&xq, &xp));
        });
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            "8_dots", name, dots_stats.median, "-", "-", "RUST_ONLY"
        );

        // (d) 4 finish_row
        // Precompute honest xq and xp from dot4 and intt for record 0
        let mut honest_xq = [Poly32::zero(); 4];
        let mut honest_xp = [Poly32::zero(); 4];
        for r in 0..4 {
            e.dot4(Prime::Q, &prepared_q[0][r], &honest_zq, &mut honest_xq[r]);
            e.intt(Prime::Q, &mut honest_xq[r], Scale::Mont, Form::Canonical);
            e.dot4(Prime::P2, &prepared_p2[0][r], &honest_zp, &mut honest_xp[r]);
            e.intt(Prime::P2, &mut honest_xp[r], Scale::Mont, Form::Canonical);
        }

        let mut finish_out = [FinishOut::zero(); 4];
        let finish_stats = common::measure_kernel(|| {
            for r in 0..4 {
                let row = FinishRow {
                    xq: &honest_xq[r],
                    xp: &honest_xp[r],
                    t: &t[0][r],
                    h: &hints[0][r],
                };
                e.finish_row(row, &mut finish_out[r]).unwrap();
            }
            black_box(&finish_out);
        });
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            "4_finish_row", name, finish_stats.median, "-", "-", "RUST_ONLY"
        );

        // (e) pack_w1
        let w1 = finish_out.map(|f| f.w1);
        let mut w1_out = [0u8; 768];
        let pack_stats = common::measure_kernel(|| {
            e.pack_w1(&w1, &mut w1_out);
            black_box(&w1_out);
        });
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            "pack_w1", name, pack_stats.median, "-", "-", "RUST_ONLY"
        );

        // (f) 1/8 commitment batch
        let dummy_w1 = [[0u8; 768]; 8];
        let batch_inputs: [[&[u8]; 4]; 8] =
            std::array::from_fn(|i| [&mu[i][..], &dummy_w1[i][..], &[][..], &[][..]]);
        let mut batch_digests = [[0u8; 32]; 8];
        let commit_stats = common::measure_kernel(|| {
            e.shake256_many(&batch_inputs, 32, batch_digests.as_flattened_mut());
            black_box(&batch_digests);
        });
        let commit_per_rec = commit_stats.median / 8.0;
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            "1/8_commit_batch", name, commit_per_rec, "-", "-", "RUST_ONLY"
        );

        // (g) Composite Step-2 kernel chain (per record)
        let mut chain_w1 = [[0u8; 768]; 8];
        let mut chain_digests = [[0u8; 32]; 8];
        let chain_stats = common::measure_kernel(|| {
            run_step2_chain_8(
                &e,
                &z_bytes,
                &prepared_q,
                &prepared_p2,
                &t,
                &h_refs,
                &mu,
                &mut chain_w1,
                &mut chain_digests,
            );
        });
        let chain_per_rec = chain_stats.median / 8.0;
        println!(
            "{:<20} {:<10} {:>10.2} {:>10} {:>8} {:>12}",
            "step2_chain", name, chain_per_rec, "-", "-", "RUST_ONLY"
        );
    }
    common::finish_benchmarks();
}

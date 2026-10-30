//! Performance gate and microbenchmarks for samplers and packing routines.
//!
//! Evaluates AVX2 and AVX-512 implementations against the C reference implementations
//! (pq-crystals Dilithium2 AVX2 and Hybrid-SHA-3 AVX-512) and baseline implementations.

#![allow(unexpected_cfgs)]

mod common;

use agg_mldsa_simd::dispatch::{Backend, Engine};
use agg_mldsa_simd::params::Q;
use agg_mldsa_simd::poly::{self, Block, Poly32, RejState};
use agg_mldsa_simd::{ExpandedA, Sink, XofJob};
use std::hint::black_box;
#[allow(dead_code)]
#[path = "../src/keccak/avx2.rs"]
mod keccak_avx2;
#[allow(dead_code)]
#[path = "../src/keccak/lanes.rs"]
mod lanes;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            let len = chunk.len();
            chunk.copy_from_slice(&val[..len]);
        }
    }
}

#[inline(never)]
fn run_rust_rej(e: &Engine, blocks: &[Block; 5], st: &mut RejState) {
    st.count = 0;
    for b in blocks {
        if e.rej_uniform(b, st) {
            break;
        }
    }
    black_box(st);
}

#[cfg(cref)]
#[inline(never)]
fn run_c_rej_avx2(buf: &[u8; 848], r: &mut [i32; 256]) {
    black_box(agg_mldsa_cref::rej_uniform_avx(r, buf));
}

#[cfg(not(cref))]
#[inline(never)]
fn run_c_rej_avx2(_buf: &[u8; 848], _r: &mut [i32; 256]) {}

#[cfg(cref)]
#[inline(never)]
fn run_c_rej_avx512(buf: &[u8; 840], r: &mut [i32; 256]) {
    black_box(agg_mldsa_cref::rej_uniform_avx512(r, buf));
}

#[cfg(not(cref))]
#[inline(never)]
fn run_c_rej_avx512(_buf: &[u8; 840], _r: &mut [i32; 256]) {}

#[inline(never)]
fn run_rust_unpack_z(e: &Engine, z: &[u8; 2304], out: &mut [Poly32; 4]) {
    let max = e.unpack_z(z, out);
    black_box(max);
    black_box(out);
}

#[cfg(cref)]
#[inline(never)]
fn run_c_unpack_z(z_padded: &[u8; 2304 + 32], out: &mut [Poly32; 4]) {
    for i in 0..4 {
        let chunk: &[u8; 576] = (&z_padded[i * 576..i * 576 + 576]).try_into().unwrap();
        agg_mldsa_cref::polyz_unpack(&mut out[i].0, chunk);
    }
    black_box(out);
}

#[cfg(not(cref))]
#[inline(never)]
fn run_c_unpack_z(_z_padded: &[u8; 2304 + 32], _out: &mut [Poly32; 4]) {}

#[inline(never)]
fn run_rust_pack_w1(e: &Engine, w1: &[[u8; 256]; 4], out: &mut [u8; 768]) {
    e.pack_w1(w1, out);
    black_box(out);
}

#[cfg(cref)]
#[inline(never)]
fn run_c_pack_w1(w1: &[Poly32; 4], out_padded: &mut [u8; 768 + 32]) {
    for i in 0..4 {
        let chunk: &mut [u8; 192] = (&mut out_padded[i * 192..i * 192 + 192]).try_into().unwrap();
        agg_mldsa_cref::polyw1_pack(chunk, &w1[i].0);
    }
    black_box(out_padded);
}

#[cfg(not(cref))]
#[inline(never)]
fn run_c_pack_w1(_w1: &[Poly32; 4], _out_padded: &mut [u8; 768 + 32]) {}

// VBMI micro-benchmarks for comparing store strategies
#[cfg(target_arch = "x86_64")]
const IDX_U: [u8; 64] = [
    0, 1, 2, 0x80, 3, 4, 5, 0x80, 6, 7, 8, 0x80, 9, 10, 11, 0x80,
    12, 13, 14, 0x80, 15, 16, 17, 0x80, 18, 19, 20, 0x80, 21, 22, 23, 0x80,
    24, 25, 26, 0x80, 27, 28, 29, 0x80, 30, 31, 32, 0x80, 33, 34, 35, 0x80,
    36, 37, 38, 0x80, 39, 40, 41, 0x80, 42, 43, 44, 0x80, 45, 46, 47, 0x80,
];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,popcnt,avx512vbmi")]
unsafe fn bench_vbmi_step<const COMPRESS_STORE: bool>(block: &Block, st: &mut RejState) -> bool {
    let mut ctr = st.count;
    if ctr >= 256 {
        st.count = 256;
        return true;
    }

    let bound = _mm512_set1_epi32(Q);
    let mask = _mm512_set1_epi32(0x7FFFFF);
    // SAFETY: IDX_U is 64 bytes.
    let idx_u = unsafe { _mm512_loadu_si512(IDX_U.as_ptr() as *const __m512i) };

    for chunk in 0..3 {
        let offset = chunk * 48;
        // SAFETY: Block has 256 bytes; offset + 64 <= 160 <= 256.
        let mut d = unsafe {
            _mm512_loadu_si512(block.bytes.as_ptr().add(offset) as *const __m512i)
        };
        d = _mm512_permutexvar_epi8(idx_u, d);
        d = _mm512_and_si512(d, mask);
        let g = _mm512_cmplt_epi32_mask(d, bound);

        if COMPRESS_STORE {
            // SAFETY: RejState has 272 entries. Storing <= 16 entries writes <= 271.
            unsafe {
                _mm512_mask_compressstoreu_epi32(
                    st.coeffs.as_mut_ptr().add(ctr),
                    g,
                    d,
                );
            }
        } else {
            let compressed = _mm512_maskz_compress_epi32(g, d);
            // SAFETY: Storing full 64-byte vector writes 16 entries into slack area.
            unsafe {
                _mm512_storeu_si512(
                    st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i,
                    compressed,
                );
            }
        }
        ctr += _popcnt32(g as i32) as usize;
        if ctr >= 256 {
            st.count = 256;
            return true;
        }
    }

    // SAFETY: Offset 144 + 64 = 208 <= 256 bytes in Block.
    let mut d = unsafe {
        _mm512_loadu_si512(block.bytes.as_ptr().add(144) as *const __m512i)
    };
    d = _mm512_permutexvar_epi8(idx_u, d);
    d = _mm512_and_si512(d, mask);
    let g = _mm512_cmplt_epi32_mask(d, bound) & 0x00FF;

    if COMPRESS_STORE {
        // SAFETY: Storing <= 8 entries into slack area.
        unsafe {
            _mm512_mask_compressstoreu_epi32(
                st.coeffs.as_mut_ptr().add(ctr),
                g,
                d,
            );
        }
    } else {
        let compressed = _mm512_maskz_compress_epi32(g, d);
        // SAFETY: Storing full 64-byte vector writes 16 entries into slack area.
        unsafe {
            _mm512_storeu_si512(
                st.coeffs.as_mut_ptr().add(ctr) as *mut __m512i,
                compressed,
            );
        }
    }
    ctr += _popcnt32(g as i32) as usize;
    if ctr >= 256 {
        st.count = 256;
        return true;
    }

    st.count = ctr;
    ctr >= 256
}

/// Counts down a fixed number of blocks per job.
struct CountSink([u8; 16]);
impl Sink for CountSink {
    fn block(&mut self, job: usize, _: &Block, _: usize) -> bool {
        self.0[job] -= 1;
        self.0[job] == 0
    }
}

/// Per-step TSC ticks of one key's AVX2 ExpandA (16 polynomials, 20 four-way
/// permutations), each step next to its pq-crystals counterpart.
#[cfg(cref)]
fn expand_steps() {
    let Some(e) = Engine::new(Backend::Avx2) else {
        return;
    };
    let row = |name: &str, r: f64, c: f64| {
        println!("{name:<28} {r:>10.1} {c:>10.1} {:>8.3}", r / c);
    };
    // Rust and C trials alternate, as in the gate harness.
    let pair = |name: &str, mut rf: &mut dyn FnMut(), mut cf: &mut dyn FnMut()| {
        for _ in 0..300 {
            rf();
            cf();
        }
        let time = |f: &mut dyn FnMut()| {
            let start = common::start_tsc();
            for _ in 0..common::CALLS_PER_TRIAL {
                f();
            }
            (common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64
        };
        let (mut rs, mut cs) = (Vec::new(), Vec::new());
        for i in 0..common::NUM_TRIALS {
            if i % 2 == 0 {
                rs.push(time(&mut rf));
                cs.push(time(&mut cf));
            } else {
                cs.push(time(&mut cf));
                rs.push(time(&mut rf));
            }
        }
        rs.sort_by(f64::total_cmp);
        cs.sort_by(f64::total_cmp);
        row(name, rs[rs.len() / 2], cs[cs.len() / 2]);
    };
    println!(
        "{:<28} {:>10} {:>10} {:>8}",
        "AVX2 ExpandA step (per key)", "rust", "c", "ratio"
    );
    let rho = [0x42u8; 32];
    let mut st = lanes::State([[0u64; 4]; 25]);
    let mut cs = [0u64; 100];
    pair(
        "20 permutations",
        &mut || {
            for _ in 0..20 {
                // SAFETY: AVX2 was detected above.
                unsafe { keccak_avx2::permute(black_box(&mut st)) };
            }
        },
        &mut || {
            for _ in 0..20 {
                agg_mldsa_cref::f1600x4(black_box(&mut cs));
            }
        },
    );
    let mut st = lanes::State([[0u64; 4]; 25]);
    let mut cs = [0u64; 100];
    let mut lane_bufs = [[0u8; 848]; 4];
    let mut buf = vec![[0u8; 840]; 4];
    pair(
        "20 squeezes",
        &mut || {
            for k in 0..20 {
                // SAFETY: AVX2 was detected above.
                unsafe {
                    keccak_avx2::permute(&mut st);
                    keccak_avx2::extract_at(&st, black_box(&mut lane_bufs), 168 * (k % 5));
                }
            }
        },
        &mut || {
            for _ in 0..4 {
                let [b0, b1, b2, b3] = &mut buf[..] else {
                    unreachable!()
                };
                agg_mldsa_cref::shake128x4_squeezeblocks(b0, b1, b2, b3, 5, &mut cs);
                black_box(&buf);
            }
        },
    );
    let nonces: [[u8; 2]; 16] = core::array::from_fn(|i| [(i % 4) as u8, (i / 4) as u8]);
    let jobs: Vec<XofJob> = nonces
        .iter()
        .map(|n| XofJob {
            xof: agg_mldsa_simd::Xof::Shake128,
            input: [&rho, n, &[], &[]],
        })
        .collect();
    let inputs: [[u8; 34]; 4] = core::array::from_fn(|s| {
        let mut x = [0u8; 34];
        x[..32].copy_from_slice(&rho);
        x[32] = s as u8;
        x
    });
    pair(
        "scheduler 16 x 5 / absorb+sq",
        &mut || {
            let mut sink = CountSink([5; 16]);
            e.run_xof(black_box(&jobs), &mut sink);
        },
        &mut || {
            for _ in 0..4 {
                agg_mldsa_cref::shake128x4_absorb_once(
                    &mut cs, &inputs[0], &inputs[1], &inputs[2], &inputs[3],
                );
                let [b0, b1, b2, b3] = &mut buf[..] else {
                    unreachable!()
                };
                agg_mldsa_cref::shake128x4_squeezeblocks(b0, b1, b2, b3, 5, &mut cs);
                black_box(&buf);
            }
        },
    );
    let mut rng = SimpleRng::new(7);
    let mut five = [Block::zero(); 5];
    let mut c840 = [0u8; 848];
    for (i, b) in five.iter_mut().enumerate() {
        rng.fill_bytes(&mut b.bytes[..168]);
        c840[168 * i..168 * (i + 1)].copy_from_slice(&b.bytes[..168]);
    }
    let mut rs = RejState::new();
    let mut r = [0i32; 256];
    pair(
        "16 rejection samplings",
        &mut || {
            for _ in 0..16 {
                run_rust_rej(&e, &five, &mut rs);
            }
        },
        &mut || {
            for _ in 0..16 {
                agg_mldsa_cref::rej_uniform_avx(black_box(&mut r), &c840);
            }
        },
    );
    let mut p = Poly32::zero();
    let mut pc = Poly32::zero();
    pair(
        "16 order maps",
        &mut || {
            for _ in 0..16 {
                e.order_from_fips(black_box(&mut p));
            }
        },
        &mut || {
            for _ in 0..16 {
                agg_mldsa_cref::nttunpack_avx(black_box(&mut pc.0));
            }
        },
    );
    let mut out = ExpandedA::zero();
    let mut mat = [[[0i32; 256]; 4]; 4];
    pair(
        "complete ExpandA",
        &mut || {
            e.expand_a_many(
                black_box(std::slice::from_ref(&rho)),
                std::slice::from_mut(&mut out),
            );
            black_box(&out);
        },
        &mut || {
            agg_mldsa_cref::polyvec_matrix_expand(black_box(&mut mat), &rho);
            black_box(&mat);
        },
    );
}

fn main() {
    #[cfg(cref)]
    if std::env::var_os("EXPANDA_STEPS").is_some() && !std::env::args().any(|a| a == "--gate") {
        expand_steps();
        return;
    }
    common::print_system_info();
    let c_avail = cfg!(cref) && agg_mldsa_cref::available();
    if !c_avail {
        println!("Note: C references not available. Set AGG_CREF_SRC and RUSTFLAGS='--cfg cref' to enable C comparisons.");
    }

    let mut rng = SimpleRng::new(0x1337_C0DE_5A4B_3C2D);

    // Prepare 840-byte buffer for rejection sampling (5 rate blocks of 168 bytes)
    let mut blocks = [Block::zero(); 5];
    let mut c_buf_848 = [0u8; 848];
    for (i, b) in blocks.iter_mut().enumerate() {
        rng.fill_bytes(&mut b.bytes);
        c_buf_848[i * 168..(i + 1) * 168].copy_from_slice(&b.bytes[..168]);
    }
    let c_buf_840: [u8; 840] = c_buf_848[..840].try_into().unwrap();

    // Prepare response vector z (2,304 bytes + 32 bytes padding for C over-read)
    let mut z_bytes = [0u8; 2304];
    rng.fill_bytes(&mut z_bytes);
    let mut z_c_padded = [0u8; 2304 + 32];
    z_c_padded[..2304].copy_from_slice(&z_bytes);

    // Prepare high bits w1 (4 polynomials of 256 coefficients in [0, 43])
    let mut w1_u8 = [[0u8; 256]; 4];
    let mut w1_aligned = [Poly32::zero(); 4];
    for p in 0..4 {
        for (i, val_out) in w1_u8[p].iter_mut().enumerate() {
            let val = (rng.next_u64() % 44) as u8;
            *val_out = val;
            w1_aligned[p].0[i] = val as i32;
        }
    }

    // Output buffers (64-byte aligned Poly32)
    let mut st_rej = RejState::zero();
    let mut c_rej_out = [0i32; 256];
    let mut z_poly_out = [Poly32::zero(); 4];
    let mut c_z_out = [Poly32::zero(); 4];
    let mut w1_out = [0u8; 768];
    let mut c_w1_padded = [0u8; 768 + 32];

    // 1. AVX2 Gate
    if let Some(e_avx2) = Engine::new(Backend::Avx2) {
        common::compare(
            "rej_uniform/840B",
            "avx2",
            || run_rust_rej(&e_avx2, black_box(&blocks), black_box(&mut st_rej)),
            || run_c_rej_avx2(black_box(&c_buf_848), black_box(&mut c_rej_out)),
            c_avail,
        );

        common::compare(
            "unpack_z/4x",
            "avx2",
            || run_rust_unpack_z(&e_avx2, black_box(&z_bytes), black_box(&mut z_poly_out)),
            || run_c_unpack_z(black_box(&z_c_padded), black_box(&mut c_z_out)),
            c_avail,
        );

        common::compare(
            "pack_w1/4x",
            "avx2",
            || run_rust_pack_w1(&e_avx2, black_box(&w1_u8), black_box(&mut w1_out)),
            || run_c_pack_w1(black_box(&w1_aligned), black_box(&mut c_w1_padded)),
            c_avail,
        );
    } else {
        println!("AVX2 backend not supported on this host.");
    }

    // 2. AVX-512 Gate
    if let Some(e_avx512) = Engine::new(Backend::Avx512) {
        // rej_uniform vs C Hybrid AVX-512
        common::compare(
            "rej_uniform/840B",
            "avx512",
            || run_rust_rej(&e_avx512, black_box(&blocks), black_box(&mut st_rej)),
            || run_c_rej_avx512(black_box(&c_buf_840), black_box(&mut c_rej_out)),
            c_avail,
        );

        // Measure both VBMI store strategies
        #[cfg(target_arch = "x86_64")]
        if e_avx512.has_vbmi() {
            let stats_reg = common::measure_kernel(|| {
                let mut st = RejState::zero();
                for b in &blocks {
                    // SAFETY: Host has VBMI and required AVX-512 features.
                    if unsafe { bench_vbmi_step::<false>(b, &mut st) } {
                        break;
                    }
                }
                black_box(st);
            });

            let stats_store = common::measure_kernel(|| {
                let mut st = RejState::zero();
                for b in &blocks {
                    // SAFETY: Host has VBMI and required AVX-512 features.
                    if unsafe { bench_vbmi_step::<true>(b, &mut st) } {
                        break;
                    }
                }
                black_box(st);
            });

            println!("--------------------------------------------------------------------------------");
            println!(
                "VBMI store comparison (median TSC ticks): compress+store = {:.2}, compressstoreu = {:.2}",
                stats_reg.median, stats_store.median
            );
            println!("--------------------------------------------------------------------------------");
        }

        // unpack_z and pack_w1 vs Rust AVX2 versions
        if let Some(e_avx2) = Engine::new(Backend::Avx2) {
            let mut z_avx2_out = [Poly32::zero(); 4];
            let mut w1_avx2_out = [0u8; 768];

            common::compare(
                "unpack_z/4x",
                "avx512",
                || run_rust_unpack_z(&e_avx512, black_box(&z_bytes), black_box(&mut z_poly_out)),
                || run_rust_unpack_z(&e_avx2, black_box(&z_bytes), black_box(&mut z_avx2_out)),
                true,
            );

            common::compare(
                "pack_w1/4x",
                "avx512",
                || run_rust_pack_w1(&e_avx512, black_box(&w1_u8), black_box(&mut w1_out)),
                || run_rust_pack_w1(&e_avx2, black_box(&w1_u8), black_box(&mut w1_avx2_out)),
                true,
            );
        }
    } else {
        println!("AVX-512 backend not supported on this host.");
    }

    common::finish_benchmarks();
}

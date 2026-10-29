//! Run C comparisons with AGG_CREF_SRC set and RUSTFLAGS='--cfg cref'.
#![allow(unexpected_cfgs)]
mod common;
use agg_mldsa_simd::{poly, Backend, Block, Engine, Sink, Xof, XofJob};
use std::hint::black_box;
#[allow(dead_code)]
#[path = "../src/keccak/avx2.rs"]
mod avx2;
#[allow(dead_code)]
#[path = "../src/keccak/avx512.rs"]
mod avx512;
#[allow(dead_code)]
#[path = "../src/keccak/lanes.rs"]
mod lanes;

#[inline(never)]
fn rust4(state: &mut lanes::State<4>) {
    // SAFETY: Only called when an AVX2 engine exists.
    unsafe { avx2::permute(black_box(state)) }
}
#[inline(never)]
fn rust8(state: &mut lanes::State<8>) {
    // SAFETY: Only called when an AVX-512 engine exists.
    unsafe { avx512::permute(black_box(state)) }
}
struct CountSink<'a> {
    counts: &'a mut [usize],
    limits: &'a [usize],
}
impl Sink for CountSink<'_> {
    #[inline]
    fn block(&mut self, job: usize, block: &Block, _: usize) -> bool {
        black_box(block);
        self.counts[job] += 1;
        self.counts[job] == self.limits[job]
    }
}
#[inline(never)]
fn shake(engine: &Engine, jobs: &[XofJob<'_>], counts: &mut [usize], limits: &[usize]) {
    counts.fill(0);
    engine.run_xof(black_box(jobs), &mut CountSink { counts, limits });
}

// Alternate the scheduler and its bare-permutation baseline at the same frequency.
fn paired(mut rust: impl FnMut(), mut baseline: impl FnMut()) -> (f64, f64) {
    rust();
    baseline();
    let mut r = [0.0f64; common::NUM_TRIALS];
    let mut c = [0.0f64; common::NUM_TRIALS];
    for i in 0..common::NUM_TRIALS {
        let measure = |f: &mut dyn FnMut()| {
            let start = common::start_tsc();
            for _ in 0..common::CALLS_PER_TRIAL {
                f();
            }
            (common::end_tsc() - start) as f64 / common::CALLS_PER_TRIAL as f64
        };
        if i % 2 == 0 {
            r[i] = measure(&mut rust);
            c[i] = measure(&mut baseline);
        } else {
            c[i] = measure(&mut baseline);
            r[i] = measure(&mut rust);
        }
    }
    r.sort_by(f64::total_cmp);
    c.sort_by(f64::total_cmp);
    (r[common::NUM_TRIALS / 2], c[common::NUM_TRIALS / 2])
}

fn scheduler(engine: &Engine, w: usize, _bare_ticks: f64) -> bool {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let mut jobs = Vec::new();
    let mut limits = Vec::new();
    for key in 0..8 {
        jobs.push(XofJob {
            xof: Xof::Shake256,
            input: [&pk, &[], &[], &[]],
        });
        limits.push(1);
        for entry in 0..16 {
            jobs.push(XofJob {
                xof: Xof::Shake128,
                input: [&seed, &[], &[], &[]],
            });
            limits.push(5 + (key + entry) % 2);
        }
    }
    let mut counts = vec![0; jobs.len()];
    // The scheduler's progress is independent of permutation contents.
    let kernel = lanes::Kernel {
        permute: |_: &mut lanes::State<4>| {},
        extract: |_: &lanes::State<4>, _: &mut [Block; 4], _| {},
        clear: |_: &mut lanes::State<4>, _| {},
    };
    let (used, capacity) = if w == 4 {
        lanes::run(
            &jobs,
            &mut CountSink {
                counts: &mut counts,
                limits: &limits,
            },
            kernel,
        )
    } else {
        lanes::run(
            &jobs,
            &mut CountSink {
                counts: &mut counts,
                limits: &limits,
            },
            lanes::Kernel {
                permute: |_: &mut lanes::State<8>| {},
                extract: |_: &lanes::State<8>, _: &mut [Block; 8], _| {},
                clear: |_: &mut lanes::State<8>, _| {},
            },
        )
    };
    let mut s4 = lanes::State([[0; 4]; 25]);
    let mut s8 = lanes::State([[0; 8]; 25]);
    let (time, baseline) = paired(
        || shake(engine, &jobs, &mut counts, &limits),
        || {
            for _ in 0..capacity / w {
                if w == 4 {
                    rust4(&mut s4);
                } else {
                    rust8(&mut s8);
                }
            }
        },
    );
    let utilization = used as f64 / capacity as f64;
    let ratio = time / baseline;
    let blocks: usize = limits.iter().sum();
    println!("mixed {:?}: {used}/{capacity} occupied ({:.2}%), {:.2} ticks/output lane-block, scheduler/bare {:.3} {}", engine.backend(), utilization * 100.0, time / blocks as f64, ratio, if utilization >= 0.95 && ratio <= 1.10 { "PASS" } else { "FAIL" });
    utilization >= 0.95 && ratio <= 1.10
}

fn main() {
    common::print_system_info();
    let mut ok = true;
    for (backend, w) in [(Backend::Avx2, 4), (Backend::Avx512, 8)] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        let mut s4 = lanes::State([[0u64; 4]; 25]);
        let mut s8 = lanes::State([[0u64; 8]; 25]);
        let bare_ticks = if w == 4 {
            common::measure_kernel(|| rust4(&mut s4)).median
        } else {
            common::measure_kernel(|| rust8(&mut s8)).median
        };
        println!(
            "bare {backend:?}: {bare_ticks:.2} ticks/permutation ({:.2} ticks/lane-block)",
            bare_ticks / w as f64
        );
        #[cfg(cref)]
        {
            #[repr(C, align(64))]
            struct Aligned<const N: usize>([u64; N]);
            let mut c4 = Aligned([0u64; 100]);
            let mut c8 = Aligned([0u64; 200]);
            if w == 4 {
                common::compare(
                    "permutation",
                    "avx2",
                    || rust4(&mut s4),
                    || agg_mldsa_cref::f1600x4(black_box(&mut c4.0)),
                    true,
                );
            } else {
                common::compare(
                    "permutation XKCP",
                    "avx512",
                    || rust8(&mut s8),
                    || {
                        agg_mldsa_cref::keccak_p1600_times8_permute_all_24rounds(black_box(
                            &mut c8.0,
                        ))
                    },
                    true,
                );
                common::compare(
                    "permutation Hybrid",
                    "avx512",
                    || rust8(&mut s8),
                    || agg_mldsa_cref::xurq_keccak8x_permute(black_box(&mut c8.0)),
                    true,
                );
            }
            let padded_input = [42u8; 64];
            let input = &padded_input[..34];
            let jobs = vec![
                XofJob {
                    xof: Xof::Shake128,
                    input: [input, &[], &[], &[]]
                };
                w
            ];
            let limits = vec![5; w];
            let mut counts = vec![0; w];
            let mut out = [[0u8; 840]; 8];
            common::compare(
                "SHAKE128 34B/5",
                if w == 4 { "avx2" } else { "avx512" },
                || shake(&engine, &jobs, &mut counts, &limits),
                || {
                    let [o0, o1, o2, o3, o4, o5, o6, o7] = &mut out;
                    if w == 4 {
                        agg_mldsa_cref::shake128x4_absorb_once(
                            &mut c4.0, &input, &input, &input, &input,
                        );
                        agg_mldsa_cref::shake128x4_squeezeblocks(o0, o1, o2, o3, 5, &mut c4.0);
                    } else {
                        agg_mldsa_cref::xurq_avx512_shake128x8_absorb(
                            &mut c8.0, &input, &input, &input, &input, &input, &input, &input,
                            &input,
                        );
                        agg_mldsa_cref::xurq_avx512_shake128x8_squeezeblocks(
                            &mut c8.0, o0, o1, o2, o3, o4, o5, o6, o7, 5,
                        );
                    }
                    black_box(&out);
                },
                true,
            );
        }
        #[cfg(not(cref))]
        println!("C comparisons UNRESOLVED; set AGG_CREF_SRC and RUSTFLAGS='--cfg cref'");
        ok &= scheduler(&engine, w, bare_ticks);
    }
    common::finish_benchmarks();
    if std::env::args().any(|a| a == "--gate") && !ok {
        std::process::exit(1);
    }
}

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

// Median of per-pair ratios over many short alternating trials; frequency drift
// between neighbouring trials cancels.
#[cfg(cref)]
fn pair_ratio(mut a: impl FnMut(), mut b: impl FnMut()) -> f64 {
    let mut ratios = Vec::with_capacity(401);
    let time = |f: &mut dyn FnMut()| {
        let start = common::start_tsc();
        for _ in 0..200 {
            f();
        }
        (common::end_tsc() - start) as f64
    };
    for _ in 0..50 {
        a();
        b();
    }
    for i in 0..401 {
        let (x, y) = if i % 2 == 0 {
            let x = time(&mut a);
            (x, time(&mut b))
        } else {
            let y = time(&mut b);
            (time(&mut a), y)
        };
        ratios.push(x / y);
    }
    ratios.sort_by(f64::total_cmp);
    ratios[200]
}

/// Eight keys: per key one 1312-byte SHAKE256 job squeezing one block and sixteen
/// 34-byte SHAKE128 jobs squeezing five or six blocks.
fn mixed_jobs<'a>(seed: &'a [u8; 34], pk: &'a [u8; 1312]) -> (Vec<XofJob<'a>>, Vec<usize>) {
    let mut jobs = Vec::new();
    let mut limits = Vec::new();
    for key in 0..8 {
        jobs.push(XofJob {
            xof: Xof::Shake256,
            input: [pk, &[], &[], &[]],
        });
        limits.push(1);
        for entry in 0..16 {
            jobs.push(XofJob {
                xof: Xof::Shake128,
                input: [seed, &[], &[], &[]],
            });
            limits.push(5 + (key + entry) % 2);
        }
    }
    (jobs, limits)
}

fn scheduler(engine: &Engine, w: usize, _bare_ticks: f64) -> bool {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let (jobs, limits) = mixed_jobs(&seed, &pk);
    let mut counts = vec![0; jobs.len()];
    // The scheduler's progress is independent of permutation contents.
    let kernel = lanes::Kernel {
        permute: |_: &mut lanes::State<4>| {},
        absorb: |_: &mut lanes::State<4>, _, _: &[lanes::Input<'_>], _, _| {},
        extract: |_: &lanes::State<4>, _: &mut [Block; 4], _| {},
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
                absorb: |_: &mut lanes::State<8>, _, _: &[lanes::Input<'_>], _, _| {},
                extract: |_: &lanes::State<8>, _: &mut [Block; 8], _| {},
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

#[cfg(cref)]
fn check_cref(backend: Backend) {
    use sha3::digest::{ExtendableOutput, Update, XofReader};
    #[repr(C, align(64))]
    struct Aligned<const N: usize>([u64; N]);
    let input = [42u8; 64]; // Pad the pinned C routine's whole-word tail read.
    let mut expected = [0u8; 168];
    let mut hasher = sha3::Shake128::default();
    hasher.update(&input[..34]);
    hasher.finalize_xof().read(&mut expected);
    let mut out = [[0u8; 168]; 8];
    let [a, b, c, d, e, f, g, h] = &mut out;
    if backend == Backend::Avx2 {
        let mut state = Aligned([0; 100]);
        agg_mldsa_cref::shake128x4_absorb_once(
            &mut state.0,
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
        );
        agg_mldsa_cref::shake128x4_squeezeblocks(a, b, c, d, 1, &mut state.0);
        println!(
            "C SHAKE128 oracle (pq-crystals): {}",
            if out[..4].iter().all(|x| x == &expected) {
                "PASS"
            } else {
                "INVALID"
            }
        );
    } else {
        let mut state = Aligned([0; 200]);
        agg_mldsa_cref::xurq_avx512_shake128x8_absorb(
            &mut state.0,
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
            &input[..34],
        );
        agg_mldsa_cref::xurq_avx512_shake128x8_squeezeblocks(
            &mut state.0,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            h,
            1,
        );
        println!(
            "C SHAKE128 oracle (Hybrid): {}",
            if out.iter().all(|x| x == &expected) {
                "PASS"
            } else {
                "INVALID; absorb omits SHAKE padding; timings are diagnostic only"
            }
        );
        let mut expected_state = [0; 25];
        ::keccak::f1600(&mut expected_state);
        state.0.fill(0);
        agg_mldsa_cref::keccak_p1600_times8_permute_all_24rounds(&mut state.0);
        let matches =
            (0..25).all(|word| (0..8).all(|lane| state.0[word * 8 + lane] == expected_state[word]));
        println!(
            "C permutation oracle (modified XKCP): {}",
            if matches {
                "PASS"
            } else {
                "INVALID; zero state differs from Keccak-f[1600]; timings are diagnostic only"
            }
        );
    }
}

// Splits the mixed workload's time into permutations, extraction and the rest.
#[cfg(cref)]
fn diag_breakdown() {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let (jobs, limits) = mixed_jobs(&seed, &pk);
    let mut counts = vec![0; jobs.len()];
    let mut s4 = lanes::State([[0u64; 4]; 25]);
    let mut s8 = lanes::State([[0u64; 8]; 25]);
    let single: Vec<XofJob<'_>> = vec![
        XofJob {
            xof: Xof::Shake128,
            input: [&seed, &[], &[], &[]]
        };
        8
    ];
    let single_limits = [5; 8];
    for (backend, w) in [(Backend::Avx2, 4), (Backend::Avx512, 8)] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        for (name, jobs, limits) in [
            ("mixed", &jobs[..], &limits[..]),
            ("34B/5", &single[..w], &single_limits[..w]),
        ] {
            macro_rules! run_with {
                ($b:ident, $e:expr, $z:expr) => {
                    paired(
                        || {
                            counts.fill(0);
                            if w == 4 {
                                lanes::run(
                                    black_box(jobs),
                                    &mut CountSink {
                                        counts: &mut counts,
                                        limits,
                                    },
                                    lanes::Kernel {
                                        permute: |_: &mut lanes::State<4>| {},
                                        absorb: |s: &mut lanes::State<4>,
                                                 f,
                                                 i: &[lanes::Input<'_>],
                                                 p,
                                                 q| {
                                            if $z {
                                                // SAFETY: the engine exists, so the ISA is available.
                                                unsafe { avx2::absorb(s, f, i, p, q) }
                                            }
                                        },
                                        extract: |s: &lanes::State<4>, b: &mut [Block; 4], r| {
                                            if $e {
                                                // SAFETY: the engine exists, so the ISA is available.
                                                unsafe { avx2::extract(s, b, r) }
                                            }
                                        },
                                    },
                                )
                            } else {
                                lanes::run(
                                    black_box(jobs),
                                    &mut CountSink {
                                        counts: &mut counts,
                                        limits,
                                    },
                                    lanes::Kernel {
                                        permute: |_: &mut lanes::State<8>| {},
                                        absorb: |s: &mut lanes::State<8>,
                                                 f,
                                                 i: &[lanes::Input<'_>],
                                                 p,
                                                 q| {
                                            if $z {
                                                // SAFETY: the engine exists, so the ISA is available.
                                                unsafe { avx512::absorb(s, f, i, p, q) }
                                            }
                                        },
                                        extract: |s: &lanes::State<8>, b: &mut [Block; 8], r| {
                                            if $e {
                                                // SAFETY: the engine exists, so the ISA is available.
                                                unsafe { avx512::extract(s, b, r) }
                                            }
                                        },
                                    },
                                )
                            };
                        },
                        &mut $b,
                    )
                };
            }
            counts.fill(0);
            let (_, cap) = if w == 4 {
                lanes::run(
                    jobs,
                    &mut CountSink {
                        counts: &mut counts,
                        limits,
                    },
                    lanes::Kernel {
                        permute: |_: &mut lanes::State<4>| {},
                        absorb: |_: &mut lanes::State<4>, _, _: &[lanes::Input<'_>], _, _| {},
                        extract: |_: &lanes::State<4>, _: &mut [Block; 4], _| {},
                    },
                )
            } else {
                lanes::run(
                    jobs,
                    &mut CountSink {
                        counts: &mut counts,
                        limits,
                    },
                    lanes::Kernel {
                        permute: |_: &mut lanes::State<8>| {},
                        absorb: |_: &mut lanes::State<8>, _, _: &[lanes::Input<'_>], _, _| {},
                        extract: |_: &lanes::State<8>, _: &mut [Block; 8], _| {},
                    },
                )
            };
            let perms = cap / w;
            let mut bare_loop = || {
                for _ in 0..perms {
                    if w == 4 {
                        rust4(&mut s4)
                    } else {
                        rust8(&mut s8)
                    }
                }
            };
            let (full, bare) = paired(|| shake(&engine, jobs, &mut counts, limits), &mut bare_loop);
            let (no_perm, bare2) = run_with!(bare_loop, true, true);
            let (bookkeeping, bare3) = run_with!(bare_loop, false, false);
            let (extract_only, bare4) = run_with!(bare_loop, true, false);
            println!(
                "diag {backend:?} {name}: extract {:.0}",
                (extract_only * bare / bare4) - bookkeeping * bare / bare3
            );
            let (no_perm, bookkeeping) = (no_perm * bare / bare2, bookkeeping * bare / bare3);
            println!(
                "diag {backend:?} {name}: full {full:.0}, permutations {bare:.0} ({perms}), overhead {:.0} = absorb+extract {:.0} + rest {bookkeeping:.0} (+{:.0} unexplained), ratio {:.3}",
                full - bare,
                no_perm - bookkeeping,
                full - bare - no_perm,
                full / bare
            );
        }
    }
}

// Paired ratios against the C permutations only (set KECCAK_DIAG).
#[cfg(cref)]
fn diag() {
    diag_breakdown();
    #[repr(C, align(64))]
    struct Aligned<const N: usize>([u64; N]);
    let mut c4 = Aligned([0u64; 100]);
    let mut c8 = Aligned([0u64; 200]);
    let mut s4 = lanes::State([[0u64; 4]; 25]);
    let mut s8 = lanes::State([[0u64; 8]; 25]);
    if Engine::new(Backend::Avx2).is_some() {
        let r = pair_ratio(
            || rust4(&mut s4),
            || agg_mldsa_cref::f1600x4(black_box(&mut c4.0)),
        );
        println!("diag avx2 permutation rust/C {r:.3}");
    }
    if Engine::new(Backend::Avx512).is_some() {
        let r = pair_ratio(
            || rust8(&mut s8),
            || agg_mldsa_cref::xurq_keccak8x_permute(black_box(&mut c8.0)),
        );
        println!("diag avx512 permutation rust/Hybrid {r:.3}");
        let r = pair_ratio(
            || rust8(&mut s8),
            || agg_mldsa_cref::keccak_p1600_times8_permute_all_24rounds(black_box(&mut c8.0)),
        );
        println!("diag avx512 permutation rust/XKCP {r:.3}");
    }
}

fn main() {
    #[cfg(cref)]
    if std::env::var_os("KECCAK_DIAG").is_some() {
        diag();
        return;
    }
    common::print_system_info();
    let mut ok = true;
    for (backend, w) in [(Backend::Avx2, 4), (Backend::Avx512, 8)] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        #[cfg(cref)]
        check_cref(backend);
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
                            &mut c4.0, input, input, input, input,
                        );
                        agg_mldsa_cref::shake128x4_squeezeblocks(o0, o1, o2, o3, 5, &mut c4.0);
                    } else {
                        agg_mldsa_cref::xurq_avx512_shake128x8_absorb(
                            &mut c8.0, input, input, input, input, input, input, input, input,
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

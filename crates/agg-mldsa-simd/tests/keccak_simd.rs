#![allow(unexpected_cfgs)]
mod common;
use agg_mldsa_simd::poly;

use agg_mldsa_simd::{Backend, Block, Engine, Sink, Xof, XofJob};
// Include the private kernels so permutation tests need no public testing API.
#[allow(dead_code)]
#[path = "../src/keccak/avx2.rs"]
mod avx2;
#[allow(dead_code)]
#[path = "../src/keccak/avx512.rs"]
mod avx512;
#[allow(dead_code)]
#[path = "../src/keccak/lanes.rs"]
mod lanes;

fn permutations<const W: usize>(mut permute: impl FnMut(&mut lanes::State<W>)) {
    let mut rng = common::SplitMix64::new(0x1600);
    for trial in 0..101 {
        let mut state = lanes::State(std::array::from_fn(|_| {
            std::array::from_fn(|_| if trial == 0 { 0 } else { rng.next_u64() })
        }));
        let mut expected: [[u64; 25]; W] =
            std::array::from_fn(|lane| std::array::from_fn(|word| state.0[word][lane]));
        for lane in &mut expected {
            keccak::f1600(lane);
        }
        permute(&mut state);
        for (lane, expected) in expected.iter().enumerate() {
            for (word, value) in state.0.iter().zip(expected) {
                assert_eq!(word[lane], *value);
            }
        }
    }
}

#[test]
fn permutation_oracles() {
    #[cfg(target_arch = "x86_64")]
    {
        if Engine::new(Backend::Avx2).is_some() {
            // SAFETY: Engine construction checked the required ISA.
            permutations::<4>(|state| unsafe { avx2::permute(state) });
        }
        if Engine::new(Backend::Avx512).is_some() {
            // SAFETY: Engine construction checked the required ISA.
            permutations::<8>(|state| unsafe { avx512::permute(state) });
        }
    }
}

struct CheckSink {
    expected: Vec<Vec<u8>>,
    counts: Vec<usize>,
    limits: Vec<usize>,
}
impl Sink for CheckSink {
    fn block(&mut self, job: usize, block: &Block, rate: usize) -> bool {
        let count = self.counts[job];
        assert!(count < self.limits[job], "callback after completion");
        assert_eq!(
            &block.bytes[..rate],
            &self.expected[job][count * rate..(count + 1) * rate],
            "job {job}, block {count}"
        );
        assert!(block.bytes[rate..].iter().all(|&x| x == 0));
        self.counts[job] += 1;
        self.counts[job] == self.limits[job]
    }
}

fn check(jobs: &[XofJob<'_>], limits: &[usize]) {
    let expected: Vec<_> = jobs
        .iter()
        .zip(limits)
        .map(|(job, &n)| {
            let mut out = vec![0u8; n * job.xof.rate()];
            match job.xof {
                Xof::Shake128 => common::shake128_segments(&job.input, &mut out),
                Xof::Shake256 => common::shake256_segments(&job.input, &mut out),
            }
            out
        })
        .collect();
    for backend in [Backend::Scalar, Backend::Avx2, Backend::Avx512] {
        let Some(engine) = Engine::new(backend) else {
            continue;
        };
        let mut sink = CheckSink {
            expected: expected.clone(),
            counts: vec![0; jobs.len()],
            limits: limits.to_vec(),
        };
        engine.run_xof(jobs, &mut sink);
        assert_eq!(sink.counts, limits);
    }
}

#[test]
fn mixed_segmented_jobs() {
    let mut rng = common::SplitMix64::new(0x5348414b45);
    for n in (0..=17).chain([33, 200, 401]) {
        let inputs: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let len = if n == 401 {
                    i
                } else {
                    let edges = [0, 1, 135, 136, 137, 167, 168, 169, 272, 336, 400];
                    if i < edges.len() {
                        edges[i]
                    } else {
                        rng.next_u64() as usize % 401
                    }
                };
                let mut bytes = vec![0; len];
                rng.fill_bytes(&mut bytes);
                bytes
            })
            .collect();
        for shake in 0..3 {
            let jobs: Vec<_> = inputs
                .iter()
                .enumerate()
                .map(|(i, bytes)| {
                    let a = rng.next_u64() as usize % (bytes.len() + 1);
                    let b = a + rng.next_u64() as usize % (bytes.len() - a + 1);
                    let xof = if (shake == 0 && i % 2 == 0) || shake == 1 {
                        Xof::Shake128
                    } else {
                        Xof::Shake256
                    };
                    XofJob {
                        xof,
                        input: [&bytes[..a], &[], &bytes[a..b], &bytes[b..]],
                    }
                })
                .collect();
            let limits: Vec<_> = (0..n).map(|_| 1 + rng.next_u64() as usize % 7).collect();
            check(&jobs, &limits);
        }
    }
}

#[test]
fn expand_a_mixed_keys() {
    let seed = [42u8; 34];
    let pk = [73u8; 1312];
    let mut jobs = Vec::new();
    let mut limits = Vec::new();
    for key in 0..8 {
        for entry in 0..16 {
            jobs.push(XofJob {
                xof: Xof::Shake128,
                input: [&seed[..32], &seed[32..], &[], &[]],
            });
            limits.push(5 + (key + entry) % 2);
        }
        jobs.push(XofJob {
            xof: Xof::Shake256,
            input: [&pk[..136], &[], &pk[136..], &[]],
        });
        limits.push(1);
    }
    check(&jobs, &limits);
}

#[cfg(cref)]
#[test]
fn c_permutation_oracles() {
    #[repr(C, align(64))]
    struct Aligned<const N: usize>([u64; N]);
    if Engine::new(Backend::Avx2).is_some() {
        permutations::<4>(|state| {
            let mut c = Aligned([0u64; 100]);
            for word in 0..25 {
                for lane in 0..4 {
                    c.0[word * 4 + lane] = state.0[word][lane];
                }
            }
            agg_mldsa_cref::f1600x4(&mut c.0);
            for word in 0..25 {
                for lane in 0..4 {
                    state.0[word][lane] = c.0[word * 4 + lane];
                }
            }
        });
    }
    if Engine::new(Backend::Avx512).is_some() {
        permutations::<8>(|state| {
            let mut c = Aligned([0u64; 200]);
            for word in 0..25 {
                for lane in 0..8 {
                    c.0[word * 8 + lane] = state.0[word][lane];
                }
            }
            agg_mldsa_cref::xurq_keccak8x_permute(&mut c.0);
            for word in 0..25 {
                for lane in 0..8 {
                    state.0[word][lane] = c.0[word * 8 + lane];
                }
            }
        });
    }
}

/// Accounting must describe the same schedule as the production instantiation:
/// each input rate block adds one permutation before the first output block.
#[test]
fn scheduler_accounting_matches_execution() {
    fn run<const W: usize, const STATS: bool>(jobs: &[XofJob<'_>], limits: &[usize]) {
        let expected: Vec<_> = jobs
            .iter()
            .zip(limits)
            .map(|(job, &n)| {
                let mut out = vec![0; n * job.xof.rate()];
                match job.xof {
                    Xof::Shake128 => common::shake128_segments(&job.input, &mut out),
                    Xof::Shake256 => common::shake256_segments(&job.input, &mut out),
                }
                out
            })
            .collect();
        let mut sink = CheckSink {
            expected,
            counts: vec![0; jobs.len()],
            limits: limits.to_vec(),
        };
        let mut calls = 0;
        let (occupied, capacity) = lanes::run::<W, STATS, _, _>(
            jobs,
            &mut sink,
            lanes::Kernel {
                permute: |state: &mut lanes::State<W>| {
                    calls += 1;
                    for i in 0..W {
                        let mut lane = std::array::from_fn(|w| state.0[w][i]);
                        keccak::f1600(&mut lane);
                        for (w, word) in lane.into_iter().enumerate() {
                            state.0[w][i] = word;
                        }
                    }
                },
                extract: |state: &lanes::State<W>, blocks: &mut [Block; W], wide: u8| {
                    for (i, block) in blocks.iter_mut().enumerate() {
                        let words = if wide & (1 << i) != 0 { 21 } else { 17 };
                        block.bytes.fill(0);
                        for w in 0..words {
                            block.bytes[w * 8..w * 8 + 8]
                                .copy_from_slice(&state.0[w][i].to_le_bytes());
                        }
                    }
                },
                clear: |state: &mut lanes::State<W>, mask: u8| {
                    for word in &mut state.0 {
                        for (i, value) in word.iter_mut().enumerate() {
                            if mask & (1 << i) != 0 {
                                *value = 0;
                            }
                        }
                    }
                },
                absorb: |l: &mut [lanes::Lane; W], j: &[XofJob<'_>], s: &mut lanes::State<W>, m| {
                    lanes::absorb_with(l, j, m, |i, w, value| s.0[w][i] ^= value)
                },
            },
        );
        assert_eq!(sink.counts, limits);
        if STATS {
            let expected: usize = jobs
                .iter()
                .zip(limits)
                .map(|(job, limit)| {
                    job.input.iter().map(|s| s.len()).sum::<usize>() / job.xof.rate() + limit
                })
                .sum();
            assert_eq!(occupied, expected);
            assert_eq!(capacity, W * calls);
        } else {
            assert_eq!((occupied, capacity), (0, 0));
        }
    }
    let input = [42; 400];
    let jobs: Vec<_> = (0..33)
        .map(|i| {
            let len = (i * 137) % 401;
            let split = len / 3;
            XofJob {
                xof: if i % 3 == 0 {
                    Xof::Shake256
                } else {
                    Xof::Shake128
                },
                input: [&input[..split], &[], &input[split..len], &[]],
            }
        })
        .collect();
    let limits: Vec<_> = (0..jobs.len()).map(|i| 1 + i % 7).collect();
    run::<4, false>(&jobs, &limits);
    run::<4, true>(&jobs, &limits);
    run::<8, false>(&jobs, &limits);
    run::<8, true>(&jobs, &limits);
}

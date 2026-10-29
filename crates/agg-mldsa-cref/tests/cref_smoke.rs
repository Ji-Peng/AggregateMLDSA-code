#[cfg(cref)]
mod smoke_tests {
    use agg_mldsa_cref::*;

    const Q: i64 = 8_380_417;

    #[repr(align(64))]
    struct Align64<T>(pub T);

    struct SplitMix64(u64);
    impl SplitMix64 {
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
    }

    #[test]
    fn test_c_ntt_invntt_roundtrip() {
        assert!(available());
        let mut rng = SplitMix64::new(0xABCD_1234);
        let mut a = Align64([0i32; 256]);
        let mut orig = [0i32; 256];
        for i in 0..256 {
            orig[i] = (rng.next_u64() % (Q as u64)) as i32;
            a.0[i] = orig[i];
        }

        ntt_avx(&mut a.0, qdata());
        invntt_avx(&mut a.0, qdata());

        // invntt_tomont semantics: result is scaled by 2^32 mod q
        let r = ((1u128 << 32) % (Q as u128)) as i64;
        for i in 0..256 {
            let expected = ((orig[i] as i64 * r) % Q).rem_euclid(Q);
            let got = (a.0[i] as i64).rem_euclid(Q);
            assert_eq!(
                got, expected,
                "Mismatch at index {i}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_c_keccak_lanes_match_scalar() {
        assert!(available());
        let mut rng = SplitMix64::new(0x5678_9ABC);

        // Test f1600x4 (4 lanes)
        let mut s4 = Align64([0u64; 100]);
        for x in s4.0.iter_mut() {
            *x = rng.next_u64();
        }
        let s4_orig = s4.0;

        f1600x4(&mut s4.0);

        for lane in 0..4 {
            let mut expected = [0u64; 25];
            for i in 0..25 {
                expected[i] = s4_orig[i * 4 + lane];
            }
            keccak::f1600(&mut expected);
            for i in 0..25 {
                assert_eq!(
                    s4.0[i * 4 + lane],
                    expected[i],
                    "f1600x4 lane {lane} mismatch at state word {i}"
                );
            }
        }

        // Test XURQ_keccak8x_permute (8 lanes)
        let mut s8 = Align64([0u64; 200]);
        for x in s8.0.iter_mut() {
            *x = rng.next_u64();
        }
        let s8_orig = s8.0;

        xurq_keccak8x_permute(&mut s8.0);

        for lane in 0..8 {
            let mut expected = [0u64; 25];
            for i in 0..25 {
                expected[i] = s8_orig[i * 8 + lane];
            }
            keccak::f1600(&mut expected);
            for i in 0..25 {
                assert_eq!(
                    s8.0[i * 8 + lane],
                    expected[i],
                    "XURQ_keccak8x_permute lane {lane} mismatch at state word {i}"
                );
            }
        }
    }

    #[test]
    fn test_c_polyw1_pack_matches_witness() {
        assert!(available());
        let mut rng = SplitMix64::new(0x9988_7766);
        let mut w1_row = [0u8; 256];
        let mut poly_i32 = Align64([0i32; 256]);

        for i in 0..256 {
            // w1 coefficients are in [0, 44)
            let val = (rng.next_u64() % 44) as u8;
            w1_row[i] = val;
            poly_i32.0[i] = val as i32;
        }

        let mut c_packed = [0u8; 192];
        polyw1_pack(&mut c_packed, &poly_i32.0);

        let small_k = [w1_row, [0u8; 256], [0u8; 256], [0u8; 256]];
        let ref_encoded = agg_witness::native::w1_encode(&small_k);

        assert_eq!(&c_packed[..], &ref_encoded[..192]);
    }

    #[inline(always)]
    fn start_tsc() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_lfence();
            let t = core::arch::x86_64::_rdtsc();
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            t
        }
        #[cfg(not(target_arch = "x86_64"))]
        0
    }

    #[inline(always)]
    fn end_tsc() -> u64 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
            let mut aux = 0u32;
            let t = core::arch::x86_64::__rdtscp(&mut aux);
            core::arch::x86_64::_mm_lfence();
            t
        }
        #[cfg(not(target_arch = "x86_64"))]
        0
    }

    #[inline(never)]
    fn empty_call() {
        std::hint::black_box(());
    }

    #[test]
    fn test_perf_gate_harness_self_check() {
        assert!(available());

        // 1. Empty call median <= 40 TSC ticks
        for _ in 0..500 {
            empty_call();
        }
        let mut trials = [0.0f64; 31];
        for trial in trials.iter_mut() {
            let start = start_tsc();
            for _ in 0..2000 {
                empty_call();
            }
            let end = end_tsc();
            *trial = (end.wrapping_sub(start) as f64) / 2000.0;
        }
        trials.sort_by(f64::total_cmp);
        let empty_median = trials[15];
        println!("Empty call median: {empty_median:.2} ticks");
        assert!(
            empty_median <= 40.0,
            "Empty call median exceeds 40 ticks: {empty_median}"
        );

        // 2. Three repeated runs of C ntt_avx agree within 2%
        let mut a = Align64([0i32; 256]);
        let qdata = qdata();

        // Warmup
        for _ in 0..1000 {
            ntt_avx(&mut a.0, qdata);
        }

        let mut run_medians = [0.0f64; 3];
        for run_idx in 0..3 {
            let mut run_trials = [0.0f64; 31];
            for trial in run_trials.iter_mut() {
                let start = start_tsc();
                for _ in 0..2000 {
                    ntt_avx(&mut a.0, qdata);
                }
                let end = end_tsc();
                *trial = (end.wrapping_sub(start) as f64) / 2000.0;
            }
            run_trials.sort_by(f64::total_cmp);
            run_medians[run_idx] = run_trials[15];
            println!("ntt_avx Run {run_idx} median: {:.2} ticks", run_medians[run_idx]);
        }

        let min_m = run_medians[0].min(run_medians[1]).min(run_medians[2]);
        let max_m = run_medians[0].max(run_medians[1]).max(run_medians[2]);
        let diff = (max_m - min_m) / min_m;
        println!("ntt_avx 3 runs spread: {:.2}%", diff * 100.0);
        assert!(
            diff <= 0.02,
            "ntt_avx runs spread exceeds 2%: {:.2}%",
            diff * 100.0
        );
    }
}

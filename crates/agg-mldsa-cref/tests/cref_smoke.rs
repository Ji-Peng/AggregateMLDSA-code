#[cfg(cref)]
mod smoke_tests {
    use agg_mldsa_cref::*;
    use agg_mldsa_simd::{ParamSet, Poly32, Set65, Set87};

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

        let mut c_packed = [0u8; POLYW1_PACK_WRITE_88];
        polyw1_pack(&mut c_packed, &poly_i32.0);

        let small_k = [w1_row, [0u8; 256], [0u8; 256], [0u8; 256]];
        let ref_encoded = agg_witness::native::w1_encode(&small_k);

        assert_eq!(&c_packed[..192], &ref_encoded[..192]);
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
    #[ignore = "timing check: needs a quiet pinned core; run with -- --ignored"]
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
        for _ in 0..50000 {
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
            println!(
                "ntt_avx Run {run_idx} median: {:.2} ticks",
                run_medians[run_idx]
            );
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

    #[cfg(unix)]
    struct GuardPageBuffer {
        ptr: *mut u8,
        total_len: usize,
    }

    #[cfg(unix)]
    extern "C" {
        fn mmap(
            addr: *mut std::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut std::ffi::c_void;
        fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
        fn mprotect(addr: *mut std::ffi::c_void, len: usize, prot: i32) -> i32;
    }

    #[cfg(unix)]
    impl GuardPageBuffer {
        fn new(len: usize) -> (Self, &'static [u8]) {
            const PROT_READ: i32 = 1;
            const PROT_WRITE: i32 = 2;
            const PROT_NONE: i32 = 0;
            const MAP_PRIVATE: i32 = 2;
            const MAP_ANONYMOUS: i32 = 0x20;
            const PAGE_SIZE: usize = 4096;

            let total_len = 2 * PAGE_SIZE;
            unsafe {
                let mapped = mmap(
                    core::ptr::null_mut(),
                    total_len,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                );
                assert!(!mapped.is_null() && mapped != usize::MAX as *mut std::ffi::c_void);
                let guard_page = (mapped as *mut u8).add(PAGE_SIZE);
                let ret = mprotect(guard_page as *mut std::ffi::c_void, PAGE_SIZE, PROT_NONE);
                assert_eq!(ret, 0);

                let slice_start = guard_page.sub(len);
                for i in 0..len {
                    *slice_start.add(i) = ((i * 37 + 11) & 0xff) as u8;
                }
                let slice = core::slice::from_raw_parts(slice_start, len);
                (
                    GuardPageBuffer {
                        ptr: mapped as *mut u8,
                        total_len,
                    },
                    slice,
                )
            }
        }
    }

    #[cfg(unix)]
    impl Drop for GuardPageBuffer {
        fn drop(&mut self) {
            unsafe {
                munmap(self.ptr as *mut std::ffi::c_void, self.total_len);
            }
        }
    }

    #[test]
    fn test_absorb_exact_allocation_end() {
        assert!(available());
        for len in [0, 1, 7, 8, 9, 34, 135, 136, 137, 167, 168, 169] {
            #[cfg(unix)]
            {
                let (_guard0, s0) = GuardPageBuffer::new(len);
                let (_guard1, s1) = GuardPageBuffer::new(len);
                let (_guard2, s2) = GuardPageBuffer::new(len);
                let (_guard3, s3) = GuardPageBuffer::new(len);
                let (_guard4, s4) = GuardPageBuffer::new(len);
                let (_guard5, s5) = GuardPageBuffer::new(len);
                let (_guard6, s6) = GuardPageBuffer::new(len);
                let (_guard7, s7) = GuardPageBuffer::new(len);

                let mut state4 = Align64([0u64; 100]);
                shake128x4_absorb_once(&mut state4.0, s0, s1, s2, s3);
                let mut out4 = [[0u8; 168]; 4];
                let [o0, o1, o2, o3] = &mut out4;
                shake128x4_squeezeblocks(o0, o1, o2, o3, 1, &mut state4.0);

                let mut state256 = Align64([0u64; 100]);
                shake256x4_absorb_once(&mut state256.0, s0, s1, s2, s3);
                let mut out256 = [[0u8; 136]; 4];
                let [o0_256, o1_256, o2_256, o3_256] = &mut out256;
                shake256x4_squeezeblocks(o0_256, o1_256, o2_256, o3_256, 1, &mut state256.0);

                let mut state8 = Align64([0u64; 200]);
                xurq_avx512_shake128x8_absorb(&mut state8.0, s0, s1, s2, s3, s4, s5, s6, s7);
                let mut out8 = [[0u8; 168]; 8];
                let [a, b, c, d, e, f, g, h] = &mut out8;
                xurq_avx512_shake128x8_squeezeblocks(&mut state8.0, a, b, c, d, e, f, g, h, 1);
                assert_eq!(out4[0], out8[0]);
            }

            // Also test with exact-capacity boxed slices on all platforms.
            let b0 = vec![42u8; len].into_boxed_slice();
            let b1 = vec![42u8; len].into_boxed_slice();
            let b2 = vec![42u8; len].into_boxed_slice();
            let b3 = vec![42u8; len].into_boxed_slice();
            let b4 = vec![42u8; len].into_boxed_slice();
            let b5 = vec![42u8; len].into_boxed_slice();
            let b6 = vec![42u8; len].into_boxed_slice();
            let b7 = vec![42u8; len].into_boxed_slice();

            let mut state4 = Align64([0u64; 100]);
            shake128x4_absorb_once(&mut state4.0, &b0, &b1, &b2, &b3);
            let mut state8 = Align64([0u64; 200]);
            xurq_avx512_shake128x8_absorb(&mut state8.0, &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7);
        }
    }

    #[test]
    fn test_absorb_unpadded_matches_padded() {
        assert!(available());
        for len in [0, 1, 7, 8, 9, 34, 135, 136, 137, 167, 168, 169] {
            // Buffer has at least 8 bytes of slack beyond len
            let mut buf0 = vec![0u8; len + 16];
            let mut buf1 = vec![0u8; len + 16];
            let mut buf2 = vec![0u8; len + 16];
            let mut buf3 = vec![0u8; len + 16];
            for i in 0..len {
                buf0[i] = ((i * 37 + 11) & 0xff) as u8;
                buf1[i] = ((i * 41 + 13) & 0xff) as u8;
                buf2[i] = ((i * 43 + 17) & 0xff) as u8;
                buf3[i] = ((i * 47 + 19) & 0xff) as u8;
            }
            let s0 = &buf0[..len];
            let s1 = &buf1[..len];
            let s2 = &buf2[..len];
            let s3 = &buf3[..len];

            let mut state_padded = Align64([0u64; 100]);
            shake128x4_absorb_once(&mut state_padded.0, s0, s1, s2, s3);
            let mut state_unpadded = Align64([0u64; 100]);
            unsafe {
                shake128x4_absorb_once_unpadded(&mut state_unpadded.0, s0, s1, s2, s3);
            }
            assert_eq!(
                state_padded.0, state_unpadded.0,
                "shake128x4 mismatch at len {len}"
            );

            let mut state_p256 = Align64([0u64; 100]);
            shake256x4_absorb_once(&mut state_p256.0, s0, s1, s2, s3);
            let mut state_u256 = Align64([0u64; 100]);
            unsafe {
                shake256x4_absorb_once_unpadded(&mut state_u256.0, s0, s1, s2, s3);
            }
            assert_eq!(
                state_p256.0, state_u256.0,
                "shake256x4 mismatch at len {len}"
            );
        }
    }

    #[test]
    fn test_c_polyw1_pack_matches_witness_modes_3_and_5() {
        assert!(available());
        let mut rng = SplitMix64::new(0x1122_3344);
        let mut poly_i32 = Align64([0i32; 256]);
        let mut w1_u8 = [0u8; 256];

        for i in 0..256 {
            // w1 coefficients for modes 3 and 5 are in [0, 16)
            let val = (rng.next_u64() % 16) as u8;
            w1_u8[i] = val;
            poly_i32.0[i] = val as i32;
        }

        // The engine encodes all k rows; put the random row first, zero the
        // rest, and compare the first 128 bytes with the C single-row packer.
        fn rust_first_row<P: ParamSet>(row: &[u8; 256]) -> Vec<u8> {
            let mut w1 = vec![[0u8; 256]; P::K];
            w1[0] = *row;
            let mut out = vec![0u8; P::W1_ENCODE_BYTES];
            agg_mldsa_simd::engine().for_set::<P>().pack_w1(&w1, &mut out);
            out.truncate(128);
            out
        }

        // Mode 3
        let mut c_packed3 = [0u8; 128];
        dilithium3_polyw1_pack(&mut c_packed3, &poly_i32.0);
        assert_eq!(&c_packed3[..], &rust_first_row::<Set65>(&w1_u8)[..]);

        // Mode 5
        let mut c_packed5 = [0u8; 128];
        dilithium5_polyw1_pack(&mut c_packed5, &poly_i32.0);
        assert_eq!(&c_packed5[..], &rust_first_row::<Set87>(&w1_u8)[..]);
    }

    #[test]
    fn test_c_polyz_unpack_matches_simd_modes_3_and_5() {
        assert!(available());
        let mut rng = SplitMix64::new(0x5566_7788);
        // The C unpacker reads past the 640 packed bytes; the slack stays zero.
        let mut z_bytes = [0u8; POLYZ_UNPACK_READ_19];
        for b in z_bytes[..640].iter_mut() {
            *b = rng.next_u64() as u8;
        }
        let chunk: &[u8; 640] = z_bytes[..640].try_into().unwrap();

        // The engine decodes all l rows; put the random chunk first, zero the
        // rest, and compare row 0 with the C single-row unpacker.
        fn rust_first_row<P: ParamSet>(chunk: &[u8; 640]) -> Poly32 {
            let mut z = vec![0u8; P::L * 640];
            z[..640].copy_from_slice(chunk);
            let mut out = vec![Poly32::zero(); P::L];
            agg_mldsa_simd::engine().for_set::<P>().unpack_z(&z, &mut out);
            out[0]
        }

        // Mode 3
        let mut c_unpacked3 = Align64([0i32; 256]);
        dilithium3_polyz_unpack(&mut c_unpacked3.0, &z_bytes);
        assert_eq!(&c_unpacked3.0[..], &rust_first_row::<Set65>(chunk).0[..]);

        // Mode 5
        let mut c_unpacked5 = Align64([0i32; 256]);
        dilithium5_polyz_unpack(&mut c_unpacked5.0, &z_bytes);
        assert_eq!(&c_unpacked5.0[..], &rust_first_row::<Set87>(chunk).0[..]);
    }

    #[test]
    fn test_c_poly_use_hint_and_decompose_modes_3_and_5() {
        assert!(available());
        let mut rng = SplitMix64::new(0x99aa_bbcc);
        let mut a = Align64([0i32; 256]);
        let mut h = Align64([0i32; 256]);
        for i in 0..256 {
            a.0[i] = (rng.next_u64() % (Q as u64)) as i32;
            h.0[i] = (rng.next_u64() & 1) as i32;
        }

        let mut b3 = Align64([0i32; 256]);
        dilithium3_poly_use_hint(&mut b3.0, &a.0, &h.0);
        let mut a1_3 = Align64([0i32; 256]);
        let mut a0_3 = Align64([0i32; 256]);
        dilithium3_poly_decompose(&mut a1_3.0, &mut a0_3.0, &a.0);

        let mut b5 = Align64([0i32; 256]);
        dilithium5_poly_use_hint(&mut b5.0, &a.0, &h.0);
        let mut a1_5 = Align64([0i32; 256]);
        let mut a0_5 = Align64([0i32; 256]);
        dilithium5_poly_decompose(&mut a1_5.0, &mut a0_5.0, &a.0);

        // Decompose and UseHint for gamma2 = (q-1)/32 must agree between modes 3 and 5
        assert_eq!(&b3.0[..], &b5.0[..]);
        assert_eq!(&a1_3.0[..], &a1_5.0[..]);
        assert_eq!(&a0_3.0[..], &a0_5.0[..]);

        // High bits must be in [0, 16)
        for &w in &b3.0 {
            assert!(w >= 0 && w < 16);
        }
    }

    #[test]
    fn test_c_ntt_invntt_roundtrip_modes_3_and_5() {
        assert!(available());
        let mut rng = SplitMix64::new(0xddee_ff00);
        let mut a = Align64([0i32; 256]);
        let mut orig = [0i32; 256];
        for i in 0..256 {
            orig[i] = (rng.next_u64() % (Q as u64)) as i32;
            a.0[i] = orig[i];
        }

        dilithium3_ntt_avx(&mut a.0, dilithium3_qdata());
        dilithium3_invntt_avx(&mut a.0, dilithium3_qdata());

        let r = ((1u128 << 32) % (Q as u128)) as i64;
        for i in 0..256 {
            let expected = ((orig[i] as i64 * r) % Q).rem_euclid(Q);
            let got = (a.0[i] as i64).rem_euclid(Q);
            assert_eq!(got, expected, "Mode 3 mismatch at {i}");
        }

        let mut a5 = Align64(orig);
        dilithium5_ntt_avx(&mut a5.0, dilithium5_qdata());
        dilithium5_invntt_avx(&mut a5.0, dilithium5_qdata());
        for i in 0..256 {
            let expected = ((orig[i] as i64 * r) % Q).rem_euclid(Q);
            let got = (a5.0[i] as i64).rem_euclid(Q);
            assert_eq!(got, expected, "Mode 5 mismatch at {i}");
        }
    }
}

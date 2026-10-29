//! C reference implementations (pq-crystals and Hybrid-SHA-3) for verification and benchmarks.

/// Returns whether the compiled C references are available in this build.
pub fn available() -> bool {
    cfg!(cref)
}

#[cfg(cref)]
pub mod ffi {
    extern "C" {
        // --- pq-crystals Dilithium2 AVX2 ---
        pub static pqcrystals_dilithium2_avx2_qdata: [i32; 624];
        pub static pqcrystals_dilithium_fips202_avx2_KeccakF_RoundConstants: [u64; 24];

        pub fn pqcrystals_dilithium2_avx2_ntt_avx(a: *mut i32, qdata: *const i32);
        pub fn pqcrystals_dilithium2_avx2_invntt_avx(a: *mut i32, qdata: *const i32);
        pub fn pqcrystals_dilithium2_avx2_nttunpack_avx(a: *mut i32);
        pub fn pqcrystals_dilithium2_avx2_pointwise_avx(
            c: *mut i32,
            a: *const i32,
            b: *const i32,
            qdata: *const i32,
        );
        pub fn pqcrystals_dilithium2_avx2_pointwise_acc_avx(
            c: *mut i32,
            a: *const i32,
            b: *const i32,
            qdata: *const i32,
        );
        pub fn pqcrystals_dilithium2_avx2_rej_uniform_avx(
            r: *mut i32,
            buf: *const u8,
        ) -> std::ffi::c_uint;
        pub fn pqcrystals_dilithium2_avx2_polyz_unpack(r: *mut i32, a: *const u8);
        pub fn pqcrystals_dilithium2_avx2_polyw1_pack(r: *mut u8, a: *const i32);
        pub fn pqcrystals_dilithium2_avx2_poly_use_hint(b: *mut i32, a: *const i32, h: *const i32);
        pub fn pqcrystals_dilithium2_avx2_poly_decompose(
            a1: *mut i32,
            a0: *mut i32,
            a: *const i32,
        );
        pub fn pqcrystals_dilithium2_avx2_polyvec_matrix_expand(
            mat: *mut i32,
            rho: *const u8,
        );
        pub fn pqcrystals_dilithium2_avx2_poly_challenge(c: *mut i32, seed: *const u8);

        pub fn pqcrystals_dilithium_fips202x4_avx2_f1600x4(s: *mut u64, rc: *const u64);
        pub fn pqcrystals_dilithium_fips202x4_avx2_shake128x4_absorb_once(
            state: *mut u64,
            in0: *const u8,
            in1: *const u8,
            in2: *const u8,
            in3: *const u8,
            inlen: usize,
        );
        pub fn pqcrystals_dilithium_fips202x4_avx2_shake128x4_squeezeblocks(
            out0: *mut u8,
            out1: *mut u8,
            out2: *mut u8,
            out3: *mut u8,
            nblocks: usize,
            state: *mut u64,
        );
        pub fn pqcrystals_dilithium_fips202x4_avx2_shake256x4_absorb_once(
            state: *mut u64,
            in0: *const u8,
            in1: *const u8,
            in2: *const u8,
            in3: *const u8,
            inlen: usize,
        );
        pub fn pqcrystals_dilithium_fips202x4_avx2_shake256x4_squeezeblocks(
            out0: *mut u8,
            out1: *mut u8,
            out2: *mut u8,
            out3: *mut u8,
            nblocks: usize,
            state: *mut u64,
        );
        pub fn pqcrystals_dilithium_fips202_avx2_shake256(
            out: *mut u8,
            outlen: usize,
            in_: *const u8,
            inlen: usize,
        );

        // --- Hybrid-SHA-3 AVX-512 ---
        pub fn hyb512_ntt_so_avx512(a: *mut i32);
        pub fn hyb512_intt_so_avx512(a: *mut i32);
        pub fn hyb512_shuffle(a: *mut i32);
        pub fn hyb512_pointwise_avx512(c: *mut i32, a: *const i32, b: *const i32);
        pub fn hyb512_XURQ_keccak8x_permute(state: *mut u64);
        pub fn hyb512_KeccakP1600times8_PermuteAll_24rounds(state: *mut u64);
        pub fn hyb512_XURQ_AVX512_shake128x8_absorb(
            state: *mut u64,
            in0: *const u8,
            in1: *const u8,
            in2: *const u8,
            in3: *const u8,
            in4: *const u8,
            in5: *const u8,
            in6: *const u8,
            in7: *const u8,
            inlen: std::ffi::c_int,
        ) -> std::ffi::c_int;
        pub fn hyb512_XURQ_AVX512_shake128x8_squeezeblocks(
            state: *mut u64,
            out0: *mut u8,
            out1: *mut u8,
            out2: *mut u8,
            out3: *mut u8,
            out4: *mut u8,
            out5: *mut u8,
            out6: *mut u8,
            out7: *mut u8,
            nblocks: std::ffi::c_int,
        ) -> std::ffi::c_int;
        pub fn hyb512_rej_uniform_avx512(r: *mut i32, buf: *const u8) -> std::ffi::c_int;
        pub fn hyb512_poly_uniform_8x(
            a0: *mut i32,
            a1: *mut i32,
            a2: *mut i32,
            a3: *mut i32,
            a4: *mut i32,
            a5: *mut i32,
            a6: *mut i32,
            a7: *mut i32,
            seed: *const u64,
            n0: u16,
            n1: u16,
            n2: u16,
            n3: u16,
            n4: u16,
            n5: u16,
            n6: u16,
            n7: u16,
        );
        pub fn hyb512_ExpandA_shuffled(mat: *mut i32, rho: *const u8);
        pub fn hyb512_poly_emulate_ct(
            r: *mut i32,
            c: *const i32,
            t: *const i32,
        ) -> std::ffi::c_int;
        pub fn hyb512_decompose_avx512(a1: *mut i32, a0: *mut i32, a: *const i32);

        // --- Hybrid-SHA-3 AVX2 ---
        pub fn hyb2_poly_emulate_ct(
            r: *mut i32,
            c: *const i32,
            t: *const i32,
        ) -> std::ffi::c_int;
    }
}

#[cfg(cref)]
pub use safe_wrappers::*;

#[cfg(cref)]
mod safe_wrappers {
    use super::ffi;

    /// Returns a reference to pq-crystals Dilithium2 AVX2 `qdata` constant table.
    pub fn qdata() -> &'static [i32; 624] {
        unsafe { &ffi::pqcrystals_dilithium2_avx2_qdata }
    }

    /// Copies `base` qdata and patches entries at offsets 0, 8, 16, 24, 32, 328
    /// to configure custom modulus, scaling factors, or zetas.
    pub fn patch_qdata(
        base: &[i32; 624],
        p: i32,
        pinv: i32,
        div: i32,
        div_qinv: i32,
        zetas: Option<&[i32; 296]>,
        zetas_qinv: Option<&[i32; 296]>,
    ) -> [i32; 624] {
        let mut out = *base;
        // _8XQ (offsets 0..8)
        for i in 0..8 {
            out[i] = p;
        }
        // _8XQINV (offsets 8..16)
        for i in 8..16 {
            out[i] = pinv;
        }
        // _8XDIV_QINV (offsets 16..24)
        for i in 16..24 {
            out[i] = div_qinv;
        }
        // _8XDIV (offsets 24..32)
        for i in 24..32 {
            out[i] = div;
        }
        if let Some(z_qinv) = zetas_qinv {
            out[32..328].copy_from_slice(z_qinv);
        }
        if let Some(z) = zetas {
            out[328..624].copy_from_slice(z);
        }
        out
    }

    /// Patches `qdata` with Plain-scale factor (16,382) for prime Q.
    pub fn patch_qdata_plain(base: &[i32; 624]) -> [i32; 624] {
        let q = 8_380_417;
        let qinv = 58_728_449;
        let inv_plain = 16_382;
        let div_qinv = (inv_plain as i64 * qinv as i64) as i32;
        patch_qdata(base, q, qinv, inv_plain, div_qinv, None, None)
    }

    /// Forward NTT in AVX2 backend order.
    pub fn ntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_ntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Inverse NTT (tomont semantics: result is scaled by 2^32 mod q).
    pub fn invntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_invntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Permutes coefficients from FIPS order to AVX2 order.
    pub fn nttunpack_avx(a: &mut [i32; 256]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_nttunpack_avx(a.as_mut_ptr());
        }
    }

    /// Pointwise Montgomery multiplication: c = a o b / 2^32 mod q.
    pub fn pointwise_avx(
        c: &mut [i32; 256],
        a: &[i32; 256],
        b: &[i32; 256],
        qdata: &[i32; 624],
    ) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_pointwise_avx(
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                qdata.as_ptr(),
            );
        }
    }

    /// Pointwise accumulate Montgomery multiplication: c += a o b / 2^32 mod q.
    pub fn pointwise_acc_avx(
        c: &mut [i32; 256],
        a: &[i32; 256],
        b: &[i32; 256],
        qdata: &[i32; 624],
    ) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_pointwise_acc_avx(
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                qdata.as_ptr(),
            );
        }
    }

    /// Uniform rejection sampling in AVX2.
    pub fn rej_uniform_avx(r: &mut [i32; 256], buf: &[u8; 848]) -> u32 {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_rej_uniform_avx(r.as_mut_ptr(), buf.as_ptr())
        }
    }

    /// Unpacks 576 bytes into 256 signed coefficients z.
    pub fn polyz_unpack(r: &mut [i32; 256], a: &[u8; 576]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_polyz_unpack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// Packs high bits w1 into 192 bytes.
    pub fn polyw1_pack(r: &mut [u8; 192], a: &[i32; 256]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_polyw1_pack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// UseHint on one polynomial: b = UseHint(h, a).
    pub fn poly_use_hint(b: &mut [i32; 256], a: &[i32; 256], h: &[i32; 256]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_poly_use_hint(
                b.as_mut_ptr(),
                a.as_ptr(),
                h.as_ptr(),
            );
        }
    }

    /// Decomposes polynomial a into high bits a1 and low bits a0.
    pub fn poly_decompose(a1: &mut [i32; 256], a0: &mut [i32; 256], a: &[i32; 256]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_poly_decompose(
                a1.as_mut_ptr(),
                a0.as_mut_ptr(),
                a.as_ptr(),
            );
        }
    }

    /// Keccak-f1600 4-lane permutation.
    pub fn f1600x4(s: &mut [u64; 100]) {
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_f1600x4(
                s.as_mut_ptr(),
                ffi::pqcrystals_dilithium_fips202_avx2_KeccakF_RoundConstants.as_ptr(),
            );
        }
    }

    /// 4-lane SHAKE128 single-shot absorb.
    pub fn shake128x4_absorb_once(
        state: &mut [u64; 100],
        in0: &[u8],
        in1: &[u8],
        in2: &[u8],
        in3: &[u8],
    ) {
        assert_eq!(in0.len(), in1.len());
        assert_eq!(in0.len(), in2.len());
        assert_eq!(in0.len(), in3.len());
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_shake128x4_absorb_once(
                state.as_mut_ptr(),
                in0.as_ptr(),
                in1.as_ptr(),
                in2.as_ptr(),
                in3.as_ptr(),
                in0.len(),
            );
        }
    }

    /// 4-lane SHAKE128 squeezeblocks.
    pub fn shake128x4_squeezeblocks(
        out0: &mut [u8],
        out1: &mut [u8],
        out2: &mut [u8],
        out3: &mut [u8],
        nblocks: usize,
        state: &mut [u64; 100],
    ) {
        assert_eq!(out0.len(), nblocks * 168);
        assert_eq!(out1.len(), nblocks * 168);
        assert_eq!(out2.len(), nblocks * 168);
        assert_eq!(out3.len(), nblocks * 168);
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_shake128x4_squeezeblocks(
                out0.as_mut_ptr(),
                out1.as_mut_ptr(),
                out2.as_mut_ptr(),
                out3.as_mut_ptr(),
                nblocks,
                state.as_mut_ptr(),
            );
        }
    }

    /// 4-lane SHAKE256 single-shot absorb.
    pub fn shake256x4_absorb_once(
        state: &mut [u64; 100],
        in0: &[u8],
        in1: &[u8],
        in2: &[u8],
        in3: &[u8],
    ) {
        assert_eq!(in0.len(), in1.len());
        assert_eq!(in0.len(), in2.len());
        assert_eq!(in0.len(), in3.len());
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_shake256x4_absorb_once(
                state.as_mut_ptr(),
                in0.as_ptr(),
                in1.as_ptr(),
                in2.as_ptr(),
                in3.as_ptr(),
                in0.len(),
            );
        }
    }

    /// 4-lane SHAKE256 squeezeblocks.
    pub fn shake256x4_squeezeblocks(
        out0: &mut [u8],
        out1: &mut [u8],
        out2: &mut [u8],
        out3: &mut [u8],
        nblocks: usize,
        state: &mut [u64; 100],
    ) {
        assert_eq!(out0.len(), nblocks * 136);
        assert_eq!(out1.len(), nblocks * 136);
        assert_eq!(out2.len(), nblocks * 136);
        assert_eq!(out3.len(), nblocks * 136);
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_shake256x4_squeezeblocks(
                out0.as_mut_ptr(),
                out1.as_mut_ptr(),
                out2.as_mut_ptr(),
                out3.as_mut_ptr(),
                nblocks,
                state.as_mut_ptr(),
            );
        }
    }

    /// ExpandA in Dilithium2 AVX2: expands 4x4 matrix of polynomials.
    pub fn polyvec_matrix_expand(mat: &mut [[[i32; 256]; 4]; 4], rho: &[u8; 32]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_polyvec_matrix_expand(
                mat.as_mut_ptr() as *mut i32,
                rho.as_ptr(),
            );
        }
    }

    /// SampleInBall / PolyChallenge in AVX2.
    pub fn poly_challenge(c: &mut [i32; 256], seed: &[u8; 32]) {
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_poly_challenge(c.as_mut_ptr(), seed.as_ptr());
        }
    }

    /// Scalar SHAKE256 from pq-crystals reference implementation.
    pub fn shake256(out: &mut [u8], in_: &[u8]) {
        unsafe {
            ffi::pqcrystals_dilithium_fips202_avx2_shake256(
                out.as_mut_ptr(),
                out.len(),
                in_.as_ptr(),
                in_.len(),
            );
        }
    }

    // --- Hybrid-SHA-3 AVX-512 ---

    /// Forward NTT in AVX-512 backend order.
    pub fn ntt_so_avx512(a: &mut [i32; 256]) {
        unsafe {
            ffi::hyb512_ntt_so_avx512(a.as_mut_ptr());
        }
    }

    /// Inverse NTT in AVX-512 backend order.
    pub fn intt_so_avx512(a: &mut [i32; 256]) {
        unsafe {
            ffi::hyb512_intt_so_avx512(a.as_mut_ptr());
        }
    }

    /// Coefficient shuffle from natural/FIPS order to AVX-512 NTT input order.
    pub fn shuffle(a: &mut [i32; 256]) {
        unsafe {
            ffi::hyb512_shuffle(a.as_mut_ptr());
        }
    }

    /// Pointwise Montgomery multiplication in AVX-512.
    pub fn pointwise_avx512(c: &mut [i32; 256], a: &[i32; 256], b: &[i32; 256]) {
        unsafe {
            ffi::hyb512_pointwise_avx512(c.as_mut_ptr(), a.as_ptr(), b.as_ptr());
        }
    }

    /// Keccak-f1600 8-lane permutation (XURQ AVX-512).
    pub fn xurq_keccak8x_permute(state: &mut [u64; 200]) {
        unsafe {
            ffi::hyb512_XURQ_keccak8x_permute(state.as_mut_ptr());
        }
    }

    /// Keccak-f1600 8-lane permutation 24 rounds (XKCP AVX-512).
    pub fn keccak_p1600_times8_permute_all_24rounds(state: &mut [u64; 200]) {
        unsafe {
            ffi::hyb512_KeccakP1600times8_PermuteAll_24rounds(state.as_mut_ptr());
        }
    }

    /// 8-lane SHAKE128 absorb in AVX-512.
    #[allow(clippy::too_many_arguments)]
    pub fn xurq_avx512_shake128x8_absorb(
        state: &mut [u64; 200],
        in0: &[u8],
        in1: &[u8],
        in2: &[u8],
        in3: &[u8],
        in4: &[u8],
        in5: &[u8],
        in6: &[u8],
        in7: &[u8],
    ) {
        assert_eq!(in0.len(), in1.len());
        assert_eq!(in0.len(), in2.len());
        assert_eq!(in0.len(), in3.len());
        assert_eq!(in0.len(), in4.len());
        assert_eq!(in0.len(), in5.len());
        assert_eq!(in0.len(), in6.len());
        assert_eq!(in0.len(), in7.len());
        unsafe {
            ffi::hyb512_XURQ_AVX512_shake128x8_absorb(
                state.as_mut_ptr(),
                in0.as_ptr(),
                in1.as_ptr(),
                in2.as_ptr(),
                in3.as_ptr(),
                in4.as_ptr(),
                in5.as_ptr(),
                in6.as_ptr(),
                in7.as_ptr(),
                in0.len() as std::ffi::c_int,
            );
        }
    }

    /// 8-lane SHAKE128 squeezeblocks in AVX-512.
    #[allow(clippy::too_many_arguments)]
    pub fn xurq_avx512_shake128x8_squeezeblocks(
        state: &mut [u64; 200],
        out0: &mut [u8],
        out1: &mut [u8],
        out2: &mut [u8],
        out3: &mut [u8],
        out4: &mut [u8],
        out5: &mut [u8],
        out6: &mut [u8],
        out7: &mut [u8],
        nblocks: usize,
    ) {
        assert_eq!(out0.len(), nblocks * 168);
        assert_eq!(out1.len(), nblocks * 168);
        assert_eq!(out2.len(), nblocks * 168);
        assert_eq!(out3.len(), nblocks * 168);
        assert_eq!(out4.len(), nblocks * 168);
        assert_eq!(out5.len(), nblocks * 168);
        assert_eq!(out6.len(), nblocks * 168);
        assert_eq!(out7.len(), nblocks * 168);
        unsafe {
            ffi::hyb512_XURQ_AVX512_shake128x8_squeezeblocks(
                state.as_mut_ptr(),
                out0.as_mut_ptr(),
                out1.as_mut_ptr(),
                out2.as_mut_ptr(),
                out3.as_mut_ptr(),
                out4.as_mut_ptr(),
                out5.as_mut_ptr(),
                out6.as_mut_ptr(),
                out7.as_mut_ptr(),
                nblocks as std::ffi::c_int,
            );
        }
    }

    /// Uniform rejection sampling in AVX-512.
    pub fn rej_uniform_avx512(r: &mut [i32; 256], buf: &[u8; 840]) -> i32 {
        unsafe { ffi::hyb512_rej_uniform_avx512(r.as_mut_ptr(), buf.as_ptr()) }
    }

    /// Expands 8 polynomials in parallel using AVX-512 SHAKE128x8.
    #[allow(clippy::too_many_arguments)]
    pub fn poly_uniform_8x(
        a0: &mut [i32; 256],
        a1: &mut [i32; 256],
        a2: &mut [i32; 256],
        a3: &mut [i32; 256],
        a4: &mut [i32; 256],
        a5: &mut [i32; 256],
        a6: &mut [i32; 256],
        a7: &mut [i32; 256],
        seed: &[u8; 32],
        nonces: [u16; 8],
    ) {
        unsafe {
            ffi::hyb512_poly_uniform_8x(
                a0.as_mut_ptr(),
                a1.as_mut_ptr(),
                a2.as_mut_ptr(),
                a3.as_mut_ptr(),
                a4.as_mut_ptr(),
                a5.as_mut_ptr(),
                a6.as_mut_ptr(),
                a7.as_mut_ptr(),
                seed.as_ptr() as *const u64,
                nonces[0],
                nonces[1],
                nonces[2],
                nonces[3],
                nonces[4],
                nonces[5],
                nonces[6],
                nonces[7],
            );
        }
    }

    /// ExpandA with shuffled output order in AVX-512.
    pub fn expand_a_shuffled(mat: &mut [[[i32; 256]; 4]; 4], rho: &[u8; 32]) {
        unsafe {
            ffi::hyb512_ExpandA_shuffled(mat.as_mut_ptr() as *mut i32, rho.as_ptr());
        }
    }

    /// Computes c * t in AVX-512 (Hybrid pspm.c).
    pub fn poly_emulate_ct_avx512(r: &mut [i32; 256], c: &[i32; 256], t: &[i32; 256]) -> i32 {
        unsafe {
            ffi::hyb512_poly_emulate_ct(r.as_mut_ptr(), c.as_ptr(), t.as_ptr())
        }
    }

    /// Computes c * t in AVX2 (Hybrid pspm.c).
    pub fn poly_emulate_ct_avx2(r: &mut [i32; 256], c: &[i32; 256], t: &[i32; 256]) -> i32 {
        unsafe {
            ffi::hyb2_poly_emulate_ct(r.as_mut_ptr(), c.as_ptr(), t.as_ptr())
        }
    }

    /// Decomposes polynomial in AVX-512 into a1 and a0.
    pub fn decompose_avx512(a1: &mut [i32; 256], a0: &mut [i32; 256], a: &[i32; 256]) {
        unsafe {
            ffi::hyb512_decompose_avx512(a1.as_mut_ptr(), a0.as_mut_ptr(), a.as_ptr());
        }
    }
}

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
        pub fn pqcrystals_dilithium2_avx2_poly_chknorm(a: *const i32, b: i32) -> i32;
        pub fn pqcrystals_dilithium2_avx2_polyw1_pack(r: *mut u8, a: *const i32);
        pub fn pqcrystals_dilithium2_avx2_poly_use_hint(b: *mut i32, a: *const i32, h: *const i32);
        pub fn pqcrystals_dilithium2_avx2_poly_decompose(a1: *mut i32, a0: *mut i32, a: *const i32);
        pub fn pqcrystals_dilithium2_avx2_polyvec_matrix_expand(mat: *mut i32, rho: *const u8);

        // --- pq-crystals Dilithium3 AVX2 ---
        pub static pqcrystals_dilithium3_avx2_qdata: [i32; 624];
        pub fn pqcrystals_dilithium3_avx2_ntt_avx(a: *mut i32, qdata: *const i32);
        pub fn pqcrystals_dilithium3_avx2_invntt_avx(a: *mut i32, qdata: *const i32);
        pub fn pqcrystals_dilithium3_avx2_polyz_unpack(r: *mut i32, a: *const u8);
        pub fn pqcrystals_dilithium3_avx2_polyw1_pack(r: *mut u8, a: *const i32);
        pub fn pqcrystals_dilithium3_avx2_poly_use_hint(b: *mut i32, a: *const i32, h: *const i32);
        pub fn pqcrystals_dilithium3_avx2_poly_decompose(a1: *mut i32, a0: *mut i32, a: *const i32);

        // --- pq-crystals Dilithium5 AVX2 ---
        pub static pqcrystals_dilithium5_avx2_qdata: [i32; 624];
        pub fn pqcrystals_dilithium5_avx2_ntt_avx(a: *mut i32, qdata: *const i32);
        pub fn pqcrystals_dilithium5_avx2_invntt_avx(a: *mut i32, qdata: *const i32);
        pub fn pqcrystals_dilithium5_avx2_polyz_unpack(r: *mut i32, a: *const u8);
        pub fn pqcrystals_dilithium5_avx2_polyw1_pack(r: *mut u8, a: *const i32);
        pub fn pqcrystals_dilithium5_avx2_poly_use_hint(b: *mut i32, a: *const i32, h: *const i32);
        pub fn pqcrystals_dilithium5_avx2_poly_decompose(a1: *mut i32, a0: *mut i32, a: *const i32);

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

        // --- Hybrid-SHA-3 AVX-512 ---
        pub fn hyb512_ntt_so_avx512(a: *mut i32);
        pub fn hyb512_intt_so_avx512(a: *mut i32);
        pub fn hyb512_shuffle(a: *mut i32);
        pub fn hyb512_pointwise_avx512(c: *mut i32, a: *const i32, b: *const i32);
        pub fn hyb512_XURQ_keccak8x_permute(state: *mut u64);
        pub fn hyb512_KeccakP1600times8_PermuteAll_24rounds(state: *mut u64);
        pub fn agg_cref_shake128x8_absorb(
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
        );
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
        pub fn hyb512_ExpandA_shuffled(mat: *mut i32, rho: *const u8);
        pub fn hyb512_poly_emulate_ct(r: *mut i32, c: *const i32, t: *const i32)
            -> std::ffi::c_int;

        // --- Hybrid-SHA-3 AVX2 ---
        pub fn hyb2_poly_emulate_ct(r: *mut i32, c: *const i32, t: *const i32) -> std::ffi::c_int;
    }
}

#[cfg(cref)]
pub use safe_wrappers::*;

#[cfg(cref)]
mod safe_wrappers {
    use super::ffi;

    /// Vector alignment of the pq-crystals AVX2 polynomial, table and state types.
    const AVX2: usize = 32;
    /// Vector alignment of the Hybrid-SHA-3 AVX-512 polynomial and state types.
    const AVX512: usize = 64;

    /// The C kernels read and write these arguments with aligned vector loads and
    /// stores; a misaligned array would fault (or be undefined behaviour), so the
    /// safe wrappers check the alignment first.
    #[track_caller]
    fn aligned<T>(p: *const T, align: usize) {
        assert!(
            (p as usize).is_multiple_of(align),
            "C reference arguments must be {align}-byte aligned"
        );
    }

    /// Bytes the AVX2 z unpacker reads for gamma1 = 2^17: one 32-byte load per 8
    /// coefficients at a stride of 18 bytes, so 14 bytes past the 576 packed ones.
    pub const POLYZ_UNPACK_READ_17: usize = 18 * 31 + 32;
    /// Bytes the AVX2 z unpacker reads for gamma1 = 2^19: 32-byte loads at a stride
    /// of 20 bytes, so 12 bytes past the 640 packed ones.
    pub const POLYZ_UNPACK_READ_19: usize = 20 * 31 + 32;
    /// Bytes the AVX2 w1 packer writes for gamma2 = (q-1)/88: one 32-byte store per
    /// 32 coefficients at a stride of 24 bytes, so 8 bytes past the 192 packed ones.
    pub const POLYW1_PACK_WRITE_88: usize = 24 * 7 + 32;

    /// Returns a reference to pq-crystals Dilithium2 AVX2 `qdata` constant table.
    pub fn qdata() -> &'static [i32; 624] {
        unsafe { &ffi::pqcrystals_dilithium2_avx2_qdata }
    }

    /// Forward NTT in AVX2 backend order.
    pub fn ntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        aligned(a.as_ptr(), AVX2);
        aligned(qdata.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_ntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Inverse NTT (tomont semantics: result is scaled by 2^32 mod q).
    pub fn invntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        aligned(a.as_ptr(), AVX2);
        aligned(qdata.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_invntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Permutes coefficients from FIPS order to AVX2 order.
    pub fn nttunpack_avx(a: &mut [i32; 256]) {
        aligned(a.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_nttunpack_avx(a.as_mut_ptr());
        }
    }

    /// Uniform rejection sampling in AVX2.
    pub fn rej_uniform_avx(r: &mut [i32; 256], buf: &[u8; 848]) -> u32 {
        unsafe { ffi::pqcrystals_dilithium2_avx2_rej_uniform_avx(r.as_mut_ptr(), buf.as_ptr()) }
    }

    /// Unpacks the first 576 bytes of `a` into 256 signed coefficients z. The C
    /// kernel reads [`POLYZ_UNPACK_READ_17`] bytes, so `a` must be that long.
    pub fn polyz_unpack(r: &mut [i32; 256], a: &[u8]) {
        aligned(r.as_ptr(), AVX2);
        assert!(a.len() >= POLYZ_UNPACK_READ_17, "z input needs the unpacker's read slack");
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_polyz_unpack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// Returns 1 if some coefficient has absolute value at least `b`, else 0.
    pub fn poly_chknorm(a: &[i32; 256], b: i32) -> i32 {
        aligned(a.as_ptr(), AVX2);
        unsafe { ffi::pqcrystals_dilithium2_avx2_poly_chknorm(a.as_ptr(), b) }
    }

    /// Packs high bits w1 into the first 192 bytes of `r`. The C kernel writes
    /// [`POLYW1_PACK_WRITE_88`] bytes (the 8 past the packed ones are clobbered), so
    /// `r` must be that long.
    pub fn polyw1_pack(r: &mut [u8], a: &[i32; 256]) {
        aligned(a.as_ptr(), AVX2);
        assert!(r.len() >= POLYW1_PACK_WRITE_88, "w1 output needs the packer's write slack");
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_polyw1_pack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// UseHint on one polynomial: b = UseHint(h, a).
    pub fn poly_use_hint(b: &mut [i32; 256], a: &[i32; 256], h: &[i32; 256]) {
        aligned(b.as_ptr(), AVX2);
        aligned(a.as_ptr(), AVX2);
        aligned(h.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_poly_use_hint(b.as_mut_ptr(), a.as_ptr(), h.as_ptr());
        }
    }

    /// Dilithium3 (ML-DSA-65) qdata constant table.
    pub fn dilithium3_qdata() -> &'static [i32; 624] {
        unsafe { &ffi::pqcrystals_dilithium3_avx2_qdata }
    }

    /// Dilithium3 (ML-DSA-65) forward NTT in AVX2 backend order.
    pub fn dilithium3_ntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        aligned(a.as_ptr(), AVX2);
        aligned(qdata.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium3_avx2_ntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Dilithium3 (ML-DSA-65) inverse NTT.
    pub fn dilithium3_invntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        aligned(a.as_ptr(), AVX2);
        aligned(qdata.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium3_avx2_invntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Dilithium3 (ML-DSA-65) unpacks the first 640 bytes of `a` into 256 signed
    /// coefficients z. The C kernel reads [`POLYZ_UNPACK_READ_19`] bytes, so `a` must be
    /// that long.
    pub fn dilithium3_polyz_unpack(r: &mut [i32; 256], a: &[u8]) {
        aligned(r.as_ptr(), AVX2);
        assert!(a.len() >= POLYZ_UNPACK_READ_19, "z input needs the unpacker's read slack");
        unsafe {
            ffi::pqcrystals_dilithium3_avx2_polyz_unpack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// Dilithium3 (ML-DSA-65) packs high bits w1 into 128 bytes.
    pub fn dilithium3_polyw1_pack(r: &mut [u8; 128], a: &[i32; 256]) {
        aligned(a.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium3_avx2_polyw1_pack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// Dilithium3 (ML-DSA-65) UseHint on one polynomial: b = UseHint(h, a).
    pub fn dilithium3_poly_use_hint(b: &mut [i32; 256], a: &[i32; 256], h: &[i32; 256]) {
        aligned(b.as_ptr(), AVX2);
        aligned(a.as_ptr(), AVX2);
        aligned(h.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium3_avx2_poly_use_hint(b.as_mut_ptr(), a.as_ptr(), h.as_ptr());
        }
    }

    /// Dilithium3 (ML-DSA-65) decomposes polynomial a into high bits a1 and low bits a0.
    pub fn dilithium3_poly_decompose(a1: &mut [i32; 256], a0: &mut [i32; 256], a: &[i32; 256]) {
        aligned(a1.as_ptr(), AVX2);
        aligned(a0.as_ptr(), AVX2);
        aligned(a.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium3_avx2_poly_decompose(
                a1.as_mut_ptr(),
                a0.as_mut_ptr(),
                a.as_ptr(),
            );
        }
    }

    /// Dilithium5 (ML-DSA-87) qdata constant table.
    pub fn dilithium5_qdata() -> &'static [i32; 624] {
        unsafe { &ffi::pqcrystals_dilithium5_avx2_qdata }
    }

    /// Dilithium5 (ML-DSA-87) forward NTT in AVX2 backend order.
    pub fn dilithium5_ntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        aligned(a.as_ptr(), AVX2);
        aligned(qdata.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium5_avx2_ntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Dilithium5 (ML-DSA-87) inverse NTT.
    pub fn dilithium5_invntt_avx(a: &mut [i32; 256], qdata: &[i32; 624]) {
        aligned(a.as_ptr(), AVX2);
        aligned(qdata.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium5_avx2_invntt_avx(a.as_mut_ptr(), qdata.as_ptr());
        }
    }

    /// Dilithium5 (ML-DSA-87) unpacks the first 640 bytes of `a` into 256 signed
    /// coefficients z. The C kernel reads [`POLYZ_UNPACK_READ_19`] bytes, so `a` must be
    /// that long.
    pub fn dilithium5_polyz_unpack(r: &mut [i32; 256], a: &[u8]) {
        aligned(r.as_ptr(), AVX2);
        assert!(a.len() >= POLYZ_UNPACK_READ_19, "z input needs the unpacker's read slack");
        unsafe {
            ffi::pqcrystals_dilithium5_avx2_polyz_unpack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// Dilithium5 (ML-DSA-87) packs high bits w1 into 128 bytes.
    pub fn dilithium5_polyw1_pack(r: &mut [u8; 128], a: &[i32; 256]) {
        aligned(a.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium5_avx2_polyw1_pack(r.as_mut_ptr(), a.as_ptr());
        }
    }

    /// Dilithium5 (ML-DSA-87) UseHint on one polynomial: b = UseHint(h, a).
    pub fn dilithium5_poly_use_hint(b: &mut [i32; 256], a: &[i32; 256], h: &[i32; 256]) {
        aligned(b.as_ptr(), AVX2);
        aligned(a.as_ptr(), AVX2);
        aligned(h.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium5_avx2_poly_use_hint(b.as_mut_ptr(), a.as_ptr(), h.as_ptr());
        }
    }

    /// Dilithium5 (ML-DSA-87) decomposes polynomial a into high bits a1 and low bits a0.
    pub fn dilithium5_poly_decompose(a1: &mut [i32; 256], a0: &mut [i32; 256], a: &[i32; 256]) {
        aligned(a1.as_ptr(), AVX2);
        aligned(a0.as_ptr(), AVX2);
        aligned(a.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium5_avx2_poly_decompose(
                a1.as_mut_ptr(),
                a0.as_mut_ptr(),
                a.as_ptr(),
            );
        }
    }

    /// Keccak-f1600 4-lane permutation.
    pub fn f1600x4(s: &mut [u64; 100]) {
        aligned(s.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_f1600x4(
                s.as_mut_ptr(),
                ffi::pqcrystals_dilithium_fips202_avx2_KeccakF_RoundConstants.as_ptr(),
            );
        }
    }

    /// Padded input buffer providing at least 8 bytes of zero slack past the
    /// input slice length, rounding up to a multiple of 8 plus 8.
    /// This prevents whole-word SIMD gathers in C implementations from reading past
    /// the allocation boundary.
    enum PaddedInput {
        Stack([u8; 1024]),
        Heap(Vec<u8>),
    }

    impl PaddedInput {
        #[inline]
        fn new(input: &[u8]) -> Self {
            let len = input.len();
            let padded_len = ((len + 7) & !7) + 8;
            if padded_len <= 1024 {
                let mut buf = [0u8; 1024];
                buf[..len].copy_from_slice(input);
                Self::Stack(buf)
            } else {
                let mut buf = vec![0u8; padded_len];
                buf[..len].copy_from_slice(input);
                Self::Heap(buf)
            }
        }

        #[inline(always)]
        fn as_ptr(&self) -> *const u8 {
            match self {
                Self::Stack(buf) => buf.as_ptr(),
                Self::Heap(buf) => buf.as_ptr(),
            }
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
        aligned(state.as_ptr(), AVX2);
        assert_eq!(in0.len(), in1.len());
        assert_eq!(in0.len(), in2.len());
        assert_eq!(in0.len(), in3.len());
        let b0 = PaddedInput::new(in0);
        let b1 = PaddedInput::new(in1);
        let b2 = PaddedInput::new(in2);
        let b3 = PaddedInput::new(in3);
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_shake128x4_absorb_once(
                state.as_mut_ptr(),
                b0.as_ptr(),
                b1.as_ptr(),
                b2.as_ptr(),
                b3.as_ptr(),
                in0.len(),
            );
        }
    }

    /// 4-lane SHAKE128 single-shot absorb without defensive buffer copy.
    ///
    /// # Safety
    ///
    /// The caller must ensure that at least 8 readable bytes follow the end of each
    /// input slice (or that the slice length is an exact multiple of 8), because
    /// the pq-crystals AVX2 implementation performs whole-word 64-bit SIMD gathers.
    pub unsafe fn shake128x4_absorb_once_unpadded(
        state: &mut [u64; 100],
        in0: &[u8],
        in1: &[u8],
        in2: &[u8],
        in3: &[u8],
    ) {
        aligned(state.as_ptr(), AVX2);
        debug_assert_eq!(in0.len(), in1.len());
        debug_assert_eq!(in0.len(), in2.len());
        debug_assert_eq!(in0.len(), in3.len());
        ffi::pqcrystals_dilithium_fips202x4_avx2_shake128x4_absorb_once(
            state.as_mut_ptr(),
            in0.as_ptr(),
            in1.as_ptr(),
            in2.as_ptr(),
            in3.as_ptr(),
            in0.len(),
        );
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
        aligned(state.as_ptr(), AVX2);
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
        aligned(state.as_ptr(), AVX2);
        assert_eq!(in0.len(), in1.len());
        assert_eq!(in0.len(), in2.len());
        assert_eq!(in0.len(), in3.len());
        let b0 = PaddedInput::new(in0);
        let b1 = PaddedInput::new(in1);
        let b2 = PaddedInput::new(in2);
        let b3 = PaddedInput::new(in3);
        unsafe {
            ffi::pqcrystals_dilithium_fips202x4_avx2_shake256x4_absorb_once(
                state.as_mut_ptr(),
                b0.as_ptr(),
                b1.as_ptr(),
                b2.as_ptr(),
                b3.as_ptr(),
                in0.len(),
            );
        }
    }

    /// 4-lane SHAKE256 single-shot absorb without defensive buffer copy.
    ///
    /// # Safety
    ///
    /// The caller must ensure that at least 8 readable bytes follow the end of each
    /// input slice (or that the slice length is an exact multiple of 8), because
    /// the pq-crystals AVX2 implementation performs whole-word 64-bit SIMD gathers.
    pub unsafe fn shake256x4_absorb_once_unpadded(
        state: &mut [u64; 100],
        in0: &[u8],
        in1: &[u8],
        in2: &[u8],
        in3: &[u8],
    ) {
        aligned(state.as_ptr(), AVX2);
        debug_assert_eq!(in0.len(), in1.len());
        debug_assert_eq!(in0.len(), in2.len());
        debug_assert_eq!(in0.len(), in3.len());
        ffi::pqcrystals_dilithium_fips202x4_avx2_shake256x4_absorb_once(
            state.as_mut_ptr(),
            in0.as_ptr(),
            in1.as_ptr(),
            in2.as_ptr(),
            in3.as_ptr(),
            in0.len(),
        );
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
        aligned(state.as_ptr(), AVX2);
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
        aligned(mat.as_ptr(), AVX2);
        unsafe {
            ffi::pqcrystals_dilithium2_avx2_polyvec_matrix_expand(
                mat.as_mut_ptr() as *mut i32,
                rho.as_ptr(),
            );
        }
    }

    // --- Hybrid-SHA-3 AVX-512 ---

    /// Coefficient shuffle from natural/FIPS order to AVX-512 NTT input order.
    pub fn shuffle(a: &mut [i32; 256]) {
        aligned(a.as_ptr(), AVX512);
        unsafe {
            ffi::hyb512_shuffle(a.as_mut_ptr());
        }
    }

    /// Keccak-f1600 8-lane permutation (XURQ AVX-512).
    pub fn xurq_keccak8x_permute(state: &mut [u64; 200]) {
        aligned(state.as_ptr(), AVX512);
        unsafe {
            ffi::hyb512_XURQ_keccak8x_permute(state.as_mut_ptr());
        }
    }

    /// Keccak-f1600 8-lane permutation 24 rounds (XKCP AVX-512).
    pub fn keccak_p1600_times8_permute_all_24rounds(state: &mut [u64; 200]) {
        aligned(state.as_ptr(), AVX512);
        unsafe {
            ffi::hyb512_KeccakP1600times8_PermuteAll_24rounds(state.as_mut_ptr());
        }
    }

    /// 8-lane SHAKE128 absorb in AVX-512.
    ///
    /// The C implementation (`agg_cref_shake128x8_absorb`) uses `memcpy` for tail bytes
    /// and never reads beyond `in*.len()`, so no padding buffer is required.
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
        aligned(state.as_ptr(), AVX512);
        assert_eq!(in0.len(), in1.len());
        assert_eq!(in0.len(), in2.len());
        assert_eq!(in0.len(), in3.len());
        assert_eq!(in0.len(), in4.len());
        assert_eq!(in0.len(), in5.len());
        assert_eq!(in0.len(), in6.len());
        assert_eq!(in0.len(), in7.len());
        unsafe {
            ffi::agg_cref_shake128x8_absorb(
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
        aligned(state.as_ptr(), AVX512);
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

    /// ExpandA with shuffled output order in AVX-512.
    pub fn expand_a_shuffled(mat: &mut [[[i32; 256]; 4]; 4], rho: &[u8; 32]) {
        aligned(mat.as_ptr(), AVX512);
        unsafe {
            ffi::hyb512_ExpandA_shuffled(mat.as_mut_ptr() as *mut i32, rho.as_ptr());
        }
    }
}

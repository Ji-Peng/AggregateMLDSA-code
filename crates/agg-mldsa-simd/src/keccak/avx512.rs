//! 8-way Keccak-f[1600] and block extraction (AVX-512F).
//!
//! The permutation uses ZMM vectors for the 25 words and follows the
//! structure of XKCP's `KeccakP-1600-times8-SIMD512.c` (CC0; implementer
//! Ronny Van Keer, Keccak designed by Guido Bertoni, Joan Daemen, Michael
//! Peeters and Gilles Van Assche): `vprolq` rotations, `vpternlogq` 0x96 for
//! the column parities and 0xD2 for chi, and the in-place lane renaming that
//! repeats every four rounds. Theta folds `D[x] = C[x-1] ^ rol(C[x+1], 1)` into
//! each lane with one three-input XOR, as in Xurong Qi's Hybrid-SHA-3
//! `keccak8x.c`. The project owner authorized direct Rust translation on
//! 2026-10-29, confirming permission from the Hybrid-SHA-3 authors. The scope
//! includes the permutation and squeeze techniques attributed here; the
//! original permission correspondence is not reproduced here. Columns 2 to 4 apply
//! it just before their lanes enter chi, which interleaves the rotations with
//! the logic operations. Extraction uses 8x8 word transposes built from
//! unpack and two-input shuffles, with zero vectors for the last tile's padding.
//!
//! LLVM's own order for these rounds runs about 5-10% slower than gcc's code
//! for the same operations on Rocket Lake, so the production [`permute`]
//! fixes the instruction order with single-instruction `asm!` statements; the
//! intrinsics form remains as [`permute_intrinsics`].

use super::{
    lanes::{self, Kernel, State},
    Sink, XofJob,
};
use crate::poly::Block;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn load(word: &[u64; 8]) -> __m512i {
    // SAFETY: The complete vector is contained in the supplied fixed-size array.
    unsafe { _mm512_loadu_si512(word.as_ptr().cast()) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn store(word: &mut [u64; 8], value: __m512i) {
    // SAFETY: The complete vector is contained in the supplied fixed-size array.
    unsafe { _mm512_storeu_si512(word.as_mut_ptr().cast(), value) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn rotate<const N: i32>(x: __m512i) -> __m512i {
    _mm512_rol_epi64::<N>(x)
}

/// `vpternlogq` as its own `asm!` statement. The statements are not `pure`,
/// so LLVM keeps their order (the schedule below) but still allocates their
/// registers and inserts the copies that the destructive form needs.
#[cfg(target_arch = "x86_64")]
macro_rules! tern {
    ($imm:literal; $a:ident, $b:expr, $c:expr) => {
        // SAFETY: A register-only AVX-512F instruction in an avx512f function.
        unsafe {
            core::arch::asm!(concat!("vpternlogq {a}, {b}, {c}, ", stringify!($imm)),
                a = inout(zmm_reg) $a, b = in(zmm_reg) $b, c = in(zmm_reg) $c,
                options(nomem, nostack, preserves_flags))
        }
    };
}

/// `vprolq` as its own `asm!` statement (see `tern!`).
#[cfg(target_arch = "x86_64")]
macro_rules! rol {
    ($n:literal, $s:expr) => {{
        let d: __m512i;
        // SAFETY: A register-only AVX-512F instruction in an avx512f function.
        unsafe {
            core::arch::asm!(concat!("vprolq {d}, {s}, ", stringify!($n)),
                d = lateout(zmm_reg) d, s = in(zmm_reg) $s,
                options(nomem, nostack, preserves_flags))
        }
        d
    }};
}

#[cfg(target_arch = "x86_64")]
static RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Keccak-f[1600] on eight word-major states. Caller must have detected `avx512f`.
///
/// The same operations as [`permute_intrinsics`] (90 per round: 10 parity and
/// 25 theta three-input XORs, 29 rotations, 25 chi, iota) in a fixed order:
/// a loop of four rounds (it stays in the decoded-instruction cache) whose
/// instructions were list-scheduled for two vector ports, rotations only on
/// port 0, with at most 31 live vectors; each round also computes the column
/// parities of the next. The repository's `scripts/keccak_avx512_schedule.py`
/// generates the
/// loop body. The column parities of the input are computed before the loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn permute(state: &mut State<8>) {
    let [mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7, mut a8, mut a9, mut a10, mut a11, mut a12, mut a13, mut a14, mut a15, mut a16, mut a17, mut a18, mut a19, mut a20, mut a21, mut a22, mut a23, mut a24] =
        core::array::from_fn::<__m512i, 25, _>(|w| load(&state.0[w]));
    let t = |a, b, c| _mm512_ternarylogic_epi64::<0x96>(a, b, c);
    let mut c0 = t(t(a0, a5, a10), a15, a20);
    let mut c1 = t(t(a1, a6, a11), a16, a21);
    let mut c2 = t(t(a2, a7, a12), a17, a22);
    let mut c3 = t(t(a3, a8, a13), a18, a23);
    let mut c4 = t(t(a4, a9, a14), a19, a24);
    // An opaque length keeps the loop rolled.
    let mut rcs: &[u64] = core::hint::black_box(&RC[..]);
    while let [r0, r1, r2, r3, rest @ ..] = rcs {
        let rc = [*r0, *r1, *r2, *r3];
        let v0 = rol!(1, c1);
        let mut v15 = a10;
        tern!(0x96; v15, c4, v0);
        let mut v25 = a20;
        tern!(0x96; v25, c4, v0);
        let v42 = rol!(3, v15);
        let mut v5 = a0;
        tern!(0x96; v5, c4, v0);
        let v54 = rol!(18, v25);
        let mut v20 = a15;
        tern!(0x96; v20, c4, v0);
        let mut v10 = a5;
        tern!(0x96; v10, c4, v0);
        let v1 = rol!(1, c2);
        let v3 = rol!(1, c4);
        let mut v6 = a1;
        tern!(0x96; v6, c0, v1);
        let v50 = rol!(1, v6);
        let mut v8 = a3;
        tern!(0x96; v8, c2, v3);
        let v40 = rol!(28, v8);
        let mut v11 = a6;
        tern!(0x96; v11, c0, v1);
        let v30 = rol!(44, v11);
        let mut v18 = a13;
        tern!(0x96; v18, c2, v3);
        let v52 = rol!(25, v18);
        let mut v21 = a16;
        tern!(0x96; v21, c0, v1);
        let v43 = rol!(45, v21);
        let mut v23 = a18;
        tern!(0x96; v23, c2, v3);
        let v32 = rol!(21, v23);
        let mut v16 = a11;
        tern!(0x96; v16, c0, v1);
        let mut v26 = a21;
        tern!(0x96; v26, c0, v1);
        let mut v28 = a23;
        tern!(0x96; v28, c2, v3);
        let v2 = rol!(1, c3);
        let mut v13 = a8;
        tern!(0x96; v13, c2, v3);
        let v4 = rol!(1, c0);
        let mut v17 = a12;
        tern!(0x96; v17, c1, v2);
        let v31 = rol!(43, v17);
        let mut v24 = a19;
        tern!(0x96; v24, c3, v4);
        let v53 = rol!(8, v24);
        let mut v27 = a22;
        tern!(0x96; v27, c1, v2);
        let v44 = rol!(61, v27);
        let mut v29 = a24;
        tern!(0x96; v29, c3, v4);
        let v33 = rol!(14, v29);
        let mut v7 = a2;
        tern!(0x96; v7, c1, v2);
        let v62 = rol!(10, v16);
        let mut v9 = a4;
        tern!(0x96; v9, c3, v4);
        let v60 = rol!(27, v9);
        let mut v12 = a7;
        tern!(0x96; v12, c1, v2);
        let v51 = rol!(6, v12);
        let mut v14 = a9;
        tern!(0x96; v14, c3, v4);
        let v41 = rol!(20, v14);
        let mut v19 = a14;
        tern!(0x96; v19, c3, v4);
        let v64 = rol!(56, v28);
        let mut v22 = a17;
        tern!(0x96; v22, c1, v2);
        let v63 = rol!(15, v22);
        let mut v34 = v5;
        tern!(0xd2; v34, v30, v31);
        let v70 = rol!(62, v7);
        let mut v37 = v31;
        tern!(0xd2; v37, v32, v33);
        let v72 = rol!(39, v19);
        let mut v38 = v32;
        tern!(0xd2; v38, v33, v5);
        let v73 = rol!(41, v20);
        let mut v47 = v42;
        tern!(0xd2; v47, v43, v44);
        let v74 = rol!(2, v26);
        let mut v48 = v43;
        tern!(0xd2; v48, v44, v40);
        let v61 = rol!(36, v10);
        let mut v57 = v52;
        tern!(0xd2; v57, v53, v54);
        let mut v36 = v31;
        tern!(0xc6; v36, v30, v32);
        let mut v58 = v53;
        tern!(0xd2; v58, v54, v50);
        let mut v39 = v33;
        tern!(0xd2; v39, v5, v30);
        let v35 = _mm512_xor_si512(v34, _mm512_set1_epi64(rc[0] as i64));
        let v71 = rol!(55, v13);
        let mut v45 = v40;
        tern!(0xd2; v45, v41, v42);
        let mut v46 = v42;
        tern!(0xc6; v46, v41, v43);
        let mut v49 = v44;
        tern!(0xd2; v49, v40, v41);
        let mut v55 = v50;
        tern!(0xd2; v55, v51, v52);
        let mut v56 = v52;
        tern!(0xc6; v56, v51, v53);
        let mut v59 = v54;
        tern!(0xd2; v59, v50, v51);
        let mut v67 = v62;
        tern!(0xd2; v67, v63, v64);
        let mut v68 = v63;
        tern!(0xd2; v68, v64, v60);
        let mut v77 = v72;
        tern!(0xd2; v77, v73, v74);
        let mut v78 = v73;
        tern!(0xd2; v78, v74, v70);
        let mut v84 = v37;
        tern!(0x96; v84, v47, v57);
        let mut v86 = v38;
        tern!(0x96; v86, v48, v58);
        let mut v66 = v63;
        tern!(0x9c; v66, v61, v62);
        let mut v65 = v62;
        tern!(0x9c; v65, v60, v61);
        let mut v69 = v64;
        tern!(0xd2; v69, v60, v61);
        let mut v75 = v70;
        tern!(0xd2; v75, v71, v72);
        let mut v76 = v72;
        tern!(0xc6; v76, v71, v73);
        let mut v79 = v74;
        tern!(0xd2; v79, v70, v71);
        let mut v80 = v35;
        tern!(0x96; v80, v45, v55);
        let mut v82 = v36;
        tern!(0x96; v82, v46, v56);
        let mut v85 = v84;
        tern!(0x96; v85, v67, v77);
        let v91 = rol!(1, v85);
        let mut v87 = v86;
        tern!(0x96; v87, v68, v78);
        let v92 = rol!(1, v87);
        let mut v81 = v80;
        tern!(0x96; v81, v65, v75);
        let mut v83 = v82;
        tern!(0x96; v83, v66, v76);
        let mut v101 = v46;
        tern!(0x96; v101, v81, v91);
        let v120 = rol!(44, v101);
        let mut v107 = v57;
        tern!(0x96; v107, v83, v92);
        let v121 = rol!(43, v107);
        let mut v96 = v36;
        tern!(0x96; v96, v81, v91);
        let v140 = rol!(1, v96);
        let mut v102 = v47;
        tern!(0x96; v102, v83, v92);
        let v141 = rol!(6, v102);
        let mut v111 = v66;
        tern!(0x96; v111, v81, v91);
        let v133 = rol!(45, v111);
        let mut v117 = v77;
        tern!(0x96; v117, v83, v92);
        let v134 = rol!(61, v117);
        let mut v97 = v37;
        tern!(0x96; v97, v83, v92);
        let mut v112 = v67;
        tern!(0x96; v112, v83, v92);
        let mut v88 = v39;
        tern!(0x96; v88, v49, v59);
        let v90 = rol!(1, v83);
        let mut v89 = v88;
        tern!(0x96; v89, v69, v79);
        let v153 = rol!(15, v112);
        let mut v95 = v35;
        tern!(0x96; v95, v89, v90);
        let v160 = rol!(62, v97);
        let mut v105 = v55;
        tern!(0x96; v105, v89, v90);
        let v132 = rol!(3, v105);
        let mut v115 = v75;
        tern!(0x96; v115, v89, v90);
        let v144 = rol!(18, v115);
        let mut v100 = v45;
        tern!(0x96; v100, v89, v90);
        let mut v110 = v65;
        tern!(0x96; v110, v89, v90);
        let mut v106 = v56;
        tern!(0x96; v106, v81, v91);
        let v93 = rol!(1, v89);
        let mut v116 = v76;
        tern!(0x96; v116, v81, v91);
        let v94 = rol!(1, v81);
        let mut v98 = v38;
        tern!(0x96; v98, v85, v93);
        let v130 = rol!(28, v98);
        let mut v104 = v49;
        tern!(0x96; v104, v87, v94);
        let v131 = rol!(20, v104);
        let mut v108 = v58;
        tern!(0x96; v108, v85, v93);
        let v142 = rol!(25, v108);
        let mut v113 = v68;
        tern!(0x96; v113, v85, v93);
        let v122 = rol!(21, v113);
        let mut v114 = v69;
        tern!(0x96; v114, v87, v94);
        let v143 = rol!(8, v114);
        let mut v119 = v79;
        tern!(0x96; v119, v87, v94);
        let v123 = rol!(14, v119);
        let mut v99 = v39;
        tern!(0x96; v99, v87, v94);
        let v150 = rol!(27, v99);
        let mut v103 = v48;
        tern!(0x96; v103, v85, v93);
        let v151 = rol!(36, v100);
        let mut v109 = v59;
        tern!(0x96; v109, v87, v94);
        let v152 = rol!(10, v106);
        let mut v118 = v78;
        tern!(0x96; v118, v85, v93);
        let v154 = rol!(56, v118);
        let mut v124 = v95;
        tern!(0xd2; v124, v120, v121);
        let v161 = rol!(55, v103);
        let v125 = _mm512_xor_si512(v124, _mm512_set1_epi64(rc[1] as i64));
        let v162 = rol!(39, v109);
        let mut v126 = v120;
        tern!(0xd2; v126, v121, v122);
        let v163 = rol!(41, v110);
        let mut v127 = v121;
        tern!(0xd2; v127, v122, v123);
        let v164 = rol!(2, v116);
        let mut v128 = v122;
        tern!(0xd2; v128, v123, v95);
        let mut v129 = v123;
        tern!(0xd2; v129, v95, v120);
        let mut v135 = v130;
        tern!(0xd2; v135, v131, v132);
        let mut v136 = v131;
        tern!(0xd2; v136, v132, v133);
        let mut v137 = v132;
        tern!(0xd2; v137, v133, v134);
        let mut v138 = v133;
        tern!(0xd2; v138, v134, v130);
        let mut v139 = v134;
        tern!(0xd2; v139, v130, v131);
        let mut v145 = v140;
        tern!(0xd2; v145, v141, v142);
        let mut v146 = v141;
        tern!(0xd2; v146, v142, v143);
        let mut v147 = v142;
        tern!(0xd2; v147, v143, v144);
        let mut v148 = v143;
        tern!(0xd2; v148, v144, v140);
        let mut v149 = v144;
        tern!(0xd2; v149, v140, v141);
        let mut v155 = v150;
        tern!(0xd2; v155, v151, v152);
        let mut v156 = v151;
        tern!(0xd2; v156, v152, v153);
        let mut v157 = v152;
        tern!(0xd2; v157, v153, v154);
        let mut v158 = v153;
        tern!(0xd2; v158, v154, v150);
        let mut v159 = v154;
        tern!(0xd2; v159, v150, v151);
        let mut v165 = v160;
        tern!(0xd2; v165, v161, v162);
        let mut v166 = v161;
        tern!(0xd2; v166, v162, v163);
        let mut v167 = v162;
        tern!(0xd2; v167, v163, v164);
        let mut v168 = v163;
        tern!(0xd2; v168, v164, v160);
        let mut v169 = v164;
        tern!(0xd2; v169, v160, v161);
        let mut v170 = v125;
        tern!(0x96; v170, v135, v145);
        let mut v172 = v126;
        tern!(0x96; v172, v136, v146);
        let mut v174 = v127;
        tern!(0x96; v174, v137, v147);
        let mut v176 = v128;
        tern!(0x96; v176, v138, v148);
        let mut v178 = v129;
        tern!(0x96; v178, v139, v149);
        let mut v171 = v170;
        tern!(0x96; v171, v155, v165);
        let mut v173 = v172;
        tern!(0x96; v173, v156, v166);
        let v180 = rol!(1, v173);
        let mut v175 = v174;
        tern!(0x96; v175, v157, v167);
        let mut v177 = v176;
        tern!(0x96; v177, v158, v168);
        let mut v179 = v178;
        tern!(0x96; v179, v159, v169);
        let mut v195 = v145;
        tern!(0x96; v195, v179, v180);
        let mut v205 = v165;
        tern!(0x96; v205, v179, v180);
        let v222 = rol!(3, v195);
        let mut v185 = v125;
        tern!(0x96; v185, v179, v180);
        let v234 = rol!(18, v205);
        let mut v200 = v155;
        tern!(0x96; v200, v179, v180);
        let mut v190 = v135;
        tern!(0x96; v190, v179, v180);
        let v181 = rol!(1, v175);
        let v183 = rol!(1, v179);
        let mut v186 = v126;
        tern!(0x96; v186, v171, v181);
        let v230 = rol!(1, v186);
        let mut v188 = v128;
        tern!(0x96; v188, v175, v183);
        let v220 = rol!(28, v188);
        let mut v191 = v136;
        tern!(0x96; v191, v171, v181);
        let v210 = rol!(44, v191);
        let mut v198 = v148;
        tern!(0x96; v198, v175, v183);
        let v232 = rol!(25, v198);
        let mut v201 = v156;
        tern!(0x96; v201, v171, v181);
        let v223 = rol!(45, v201);
        let mut v203 = v158;
        tern!(0x96; v203, v175, v183);
        let v212 = rol!(21, v203);
        let mut v196 = v146;
        tern!(0x96; v196, v171, v181);
        let mut v206 = v166;
        tern!(0x96; v206, v171, v181);
        let mut v208 = v168;
        tern!(0x96; v208, v175, v183);
        let v182 = rol!(1, v177);
        let mut v193 = v138;
        tern!(0x96; v193, v175, v183);
        let v184 = rol!(1, v171);
        let mut v197 = v147;
        tern!(0x96; v197, v173, v182);
        let v211 = rol!(43, v197);
        let mut v204 = v159;
        tern!(0x96; v204, v177, v184);
        let v233 = rol!(8, v204);
        let mut v207 = v167;
        tern!(0x96; v207, v173, v182);
        let v224 = rol!(61, v207);
        let mut v209 = v169;
        tern!(0x96; v209, v177, v184);
        let v213 = rol!(14, v209);
        let mut v187 = v127;
        tern!(0x96; v187, v173, v182);
        let v242 = rol!(10, v196);
        let mut v189 = v129;
        tern!(0x96; v189, v177, v184);
        let v240 = rol!(27, v189);
        let mut v192 = v137;
        tern!(0x96; v192, v173, v182);
        let v231 = rol!(6, v192);
        let mut v194 = v139;
        tern!(0x96; v194, v177, v184);
        let v221 = rol!(20, v194);
        let mut v199 = v149;
        tern!(0x96; v199, v177, v184);
        let v244 = rol!(56, v208);
        let mut v202 = v157;
        tern!(0x96; v202, v173, v182);
        let v243 = rol!(15, v202);
        let mut v214 = v185;
        tern!(0xd2; v214, v210, v211);
        let v250 = rol!(62, v187);
        let mut v217 = v211;
        tern!(0xd2; v217, v212, v213);
        let v252 = rol!(39, v199);
        let mut v218 = v212;
        tern!(0xd2; v218, v213, v185);
        let v253 = rol!(41, v200);
        let mut v227 = v222;
        tern!(0xd2; v227, v223, v224);
        let v254 = rol!(2, v206);
        let mut v228 = v223;
        tern!(0xd2; v228, v224, v220);
        let v241 = rol!(36, v190);
        let mut v237 = v232;
        tern!(0xd2; v237, v233, v234);
        let mut v216 = v211;
        tern!(0xc6; v216, v210, v212);
        let mut v238 = v233;
        tern!(0xd2; v238, v234, v230);
        let mut v219 = v213;
        tern!(0xd2; v219, v185, v210);
        let v215 = _mm512_xor_si512(v214, _mm512_set1_epi64(rc[2] as i64));
        let v251 = rol!(55, v193);
        let mut v225 = v220;
        tern!(0xd2; v225, v221, v222);
        let mut v226 = v222;
        tern!(0xc6; v226, v221, v223);
        let mut v229 = v224;
        tern!(0xd2; v229, v220, v221);
        let mut v235 = v230;
        tern!(0xd2; v235, v231, v232);
        let mut v236 = v232;
        tern!(0xc6; v236, v231, v233);
        let mut v239 = v234;
        tern!(0xd2; v239, v230, v231);
        let mut v247 = v242;
        tern!(0xd2; v247, v243, v244);
        let mut v248 = v243;
        tern!(0xd2; v248, v244, v240);
        let mut v257 = v252;
        tern!(0xd2; v257, v253, v254);
        let mut v258 = v253;
        tern!(0xd2; v258, v254, v250);
        let mut v264 = v217;
        tern!(0x96; v264, v227, v237);
        let mut v266 = v218;
        tern!(0x96; v266, v228, v238);
        let mut v246 = v243;
        tern!(0x9c; v246, v241, v242);
        let mut v245 = v242;
        tern!(0x9c; v245, v240, v241);
        let mut v249 = v244;
        tern!(0xd2; v249, v240, v241);
        let mut v255 = v250;
        tern!(0xd2; v255, v251, v252);
        let mut v256 = v252;
        tern!(0xc6; v256, v251, v253);
        let mut v259 = v254;
        tern!(0xd2; v259, v250, v251);
        let mut v260 = v215;
        tern!(0x96; v260, v225, v235);
        let mut v262 = v216;
        tern!(0x96; v262, v226, v236);
        let mut v265 = v264;
        tern!(0x96; v265, v247, v257);
        let v271 = rol!(1, v265);
        let mut v267 = v266;
        tern!(0x96; v267, v248, v258);
        let v272 = rol!(1, v267);
        let mut v261 = v260;
        tern!(0x96; v261, v245, v255);
        let mut v263 = v262;
        tern!(0x96; v263, v246, v256);
        let mut v281 = v226;
        tern!(0x96; v281, v261, v271);
        let v300 = rol!(44, v281);
        let mut v287 = v237;
        tern!(0x96; v287, v263, v272);
        let v301 = rol!(43, v287);
        let mut v276 = v216;
        tern!(0x96; v276, v261, v271);
        let v320 = rol!(1, v276);
        let mut v282 = v227;
        tern!(0x96; v282, v263, v272);
        let v321 = rol!(6, v282);
        let mut v291 = v246;
        tern!(0x96; v291, v261, v271);
        let v313 = rol!(45, v291);
        let mut v297 = v257;
        tern!(0x96; v297, v263, v272);
        let v314 = rol!(61, v297);
        let mut v277 = v217;
        tern!(0x96; v277, v263, v272);
        let mut v292 = v247;
        tern!(0x96; v292, v263, v272);
        let mut v268 = v219;
        tern!(0x96; v268, v229, v239);
        let v270 = rol!(1, v263);
        let mut v269 = v268;
        tern!(0x96; v269, v249, v259);
        let v333 = rol!(15, v292);
        let mut v275 = v215;
        tern!(0x96; v275, v269, v270);
        let v340 = rol!(62, v277);
        let mut v285 = v235;
        tern!(0x96; v285, v269, v270);
        let v312 = rol!(3, v285);
        let mut v295 = v255;
        tern!(0x96; v295, v269, v270);
        let v324 = rol!(18, v295);
        let mut v280 = v225;
        tern!(0x96; v280, v269, v270);
        let mut v290 = v245;
        tern!(0x96; v290, v269, v270);
        let mut v286 = v236;
        tern!(0x96; v286, v261, v271);
        let v273 = rol!(1, v269);
        let mut v296 = v256;
        tern!(0x96; v296, v261, v271);
        let v274 = rol!(1, v261);
        let mut v278 = v218;
        tern!(0x96; v278, v265, v273);
        let v310 = rol!(28, v278);
        let mut v284 = v229;
        tern!(0x96; v284, v267, v274);
        let v311 = rol!(20, v284);
        let mut v288 = v238;
        tern!(0x96; v288, v265, v273);
        let v322 = rol!(25, v288);
        let mut v293 = v248;
        tern!(0x96; v293, v265, v273);
        let v302 = rol!(21, v293);
        let mut v294 = v249;
        tern!(0x96; v294, v267, v274);
        let v323 = rol!(8, v294);
        let mut v299 = v259;
        tern!(0x96; v299, v267, v274);
        let v303 = rol!(14, v299);
        let mut v279 = v219;
        tern!(0x96; v279, v267, v274);
        let v330 = rol!(27, v279);
        let mut v283 = v228;
        tern!(0x96; v283, v265, v273);
        let v331 = rol!(36, v280);
        let mut v289 = v239;
        tern!(0x96; v289, v267, v274);
        let v332 = rol!(10, v286);
        let mut v298 = v258;
        tern!(0x96; v298, v265, v273);
        let v334 = rol!(56, v298);
        let mut v304 = v275;
        tern!(0xd2; v304, v300, v301);
        let v341 = rol!(55, v283);
        let v305 = _mm512_xor_si512(v304, _mm512_set1_epi64(rc[3] as i64));
        let v342 = rol!(39, v289);
        let mut v306 = v300;
        tern!(0xd2; v306, v301, v302);
        let v343 = rol!(41, v290);
        let mut v307 = v301;
        tern!(0xd2; v307, v302, v303);
        let v344 = rol!(2, v296);
        let mut v308 = v302;
        tern!(0xd2; v308, v303, v275);
        let mut v309 = v303;
        tern!(0xd2; v309, v275, v300);
        let mut v315 = v310;
        tern!(0xd2; v315, v311, v312);
        let mut v316 = v311;
        tern!(0xd2; v316, v312, v313);
        let mut v317 = v312;
        tern!(0xd2; v317, v313, v314);
        let mut v318 = v313;
        tern!(0xd2; v318, v314, v310);
        let mut v319 = v314;
        tern!(0xd2; v319, v310, v311);
        let mut v325 = v320;
        tern!(0xd2; v325, v321, v322);
        let mut v326 = v321;
        tern!(0xd2; v326, v322, v323);
        let mut v327 = v322;
        tern!(0xd2; v327, v323, v324);
        let mut v328 = v323;
        tern!(0xd2; v328, v324, v320);
        let mut v329 = v324;
        tern!(0xd2; v329, v320, v321);
        let mut v335 = v330;
        tern!(0xd2; v335, v331, v332);
        let mut v336 = v331;
        tern!(0xd2; v336, v332, v333);
        let mut v337 = v332;
        tern!(0xd2; v337, v333, v334);
        let mut v338 = v333;
        tern!(0xd2; v338, v334, v330);
        let mut v339 = v334;
        tern!(0xd2; v339, v330, v331);
        let mut v345 = v340;
        tern!(0xd2; v345, v341, v342);
        let mut v346 = v341;
        tern!(0xd2; v346, v342, v343);
        let mut v347 = v342;
        tern!(0xd2; v347, v343, v344);
        let mut v348 = v343;
        tern!(0xd2; v348, v344, v340);
        let mut v349 = v344;
        tern!(0xd2; v349, v340, v341);
        let mut v350 = v305;
        tern!(0x96; v350, v315, v325);
        let mut v352 = v306;
        tern!(0x96; v352, v316, v326);
        let mut v354 = v307;
        tern!(0x96; v354, v317, v327);
        let mut v356 = v308;
        tern!(0x96; v356, v318, v328);
        let mut v358 = v309;
        tern!(0x96; v358, v319, v329);
        let mut v351 = v350;
        tern!(0x96; v351, v335, v345);
        let mut v353 = v352;
        tern!(0x96; v353, v336, v346);
        let mut v355 = v354;
        tern!(0x96; v355, v337, v347);
        let mut v357 = v356;
        tern!(0x96; v357, v338, v348);
        let mut v359 = v358;
        tern!(0x96; v359, v339, v349);
        a0 = v305;
        a1 = v306;
        a2 = v307;
        a3 = v308;
        a4 = v309;
        a5 = v315;
        a6 = v316;
        a7 = v317;
        a8 = v318;
        a9 = v319;
        a10 = v325;
        a11 = v326;
        a12 = v327;
        a13 = v328;
        a14 = v329;
        a15 = v335;
        a16 = v336;
        a17 = v337;
        a18 = v338;
        a19 = v339;
        a20 = v345;
        a21 = v346;
        a22 = v347;
        a23 = v348;
        a24 = v349;
        c0 = v351;
        c1 = v353;
        c2 = v355;
        c3 = v357;
        c4 = v359;
        rcs = rest;
    }
    let _ = (c0, c1, c2, c3, c4);
    for (word, value) in state.0.iter_mut().zip([
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
        a20, a21, a22, a23, a24,
    ]) {
        store(word, value);
    }
}

/// Keccak-f[1600] in intrinsics, in the source order of the XKCP and
/// Hybrid-SHA-3 rounds; kept for review and as a differential test of
/// [`permute`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(dead_code)]
pub(crate) fn permute_intrinsics(state: &mut State<8>) {
    let mut a0 = load(&state.0[0]);
    let mut a1 = load(&state.0[1]);
    let mut a2 = load(&state.0[2]);
    let mut a3 = load(&state.0[3]);
    let mut a4 = load(&state.0[4]);
    let mut a5 = load(&state.0[5]);
    let mut a6 = load(&state.0[6]);
    let mut a7 = load(&state.0[7]);
    let mut a8 = load(&state.0[8]);
    let mut a9 = load(&state.0[9]);
    let mut a10 = load(&state.0[10]);
    let mut a11 = load(&state.0[11]);
    let mut a12 = load(&state.0[12]);
    let mut a13 = load(&state.0[13]);
    let mut a14 = load(&state.0[14]);
    let mut a15 = load(&state.0[15]);
    let mut a16 = load(&state.0[16]);
    let mut a17 = load(&state.0[17]);
    let mut a18 = load(&state.0[18]);
    let mut a19 = load(&state.0[19]);
    let mut a20 = load(&state.0[20]);
    let mut a21 = load(&state.0[21]);
    let mut a22 = load(&state.0[22]);
    let mut a23 = load(&state.0[23]);
    let mut a24 = load(&state.0[24]);
    macro_rules! round {
        ($rc:expr; $a0:ident, $a1:ident, $a2:ident, $a3:ident, $a4:ident, $a5:ident, $a6:ident, $a7:ident, $a8:ident, $a9:ident, $a10:ident, $a11:ident, $a12:ident, $a13:ident, $a14:ident, $a15:ident, $a16:ident, $a17:ident, $a18:ident, $a19:ident, $a20:ident, $a21:ident, $a22:ident, $a23:ident, $a24:ident) => {{
            let c0 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a0, $a5, $a10),
                $a15,
                $a20,
            );
            let c1 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a1, $a6, $a11),
                $a16,
                $a21,
            );
            let c2 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a2, $a7, $a12),
                $a17,
                $a22,
            );
            let c3 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a3, $a8, $a13),
                $a18,
                $a23,
            );
            let c4 = _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>($a4, $a9, $a14),
                $a19,
                $a24,
            );
            // Theta for columns 0 and 1; columns 2 to 4 are deferred to their rows so
            // rotations and three-input operations interleave (D[x] = C[x-1] ^ rol(C[x+1], 1)).
            let r1 = rotate::<1>(c1);
            $a0 = _mm512_ternarylogic_epi64::<0x96>($a0, c4, r1);
            $a5 = _mm512_ternarylogic_epi64::<0x96>($a5, c4, r1);
            $a10 = _mm512_ternarylogic_epi64::<0x96>($a10, c4, r1);
            $a15 = _mm512_ternarylogic_epi64::<0x96>($a15, c4, r1);
            $a20 = _mm512_ternarylogic_epi64::<0x96>($a20, c4, r1);
            let r2 = rotate::<1>(c2);
            $a1 = _mm512_ternarylogic_epi64::<0x96>($a1, c0, r2);
            $a6 = _mm512_ternarylogic_epi64::<0x96>($a6, c0, r2);
            $a11 = _mm512_ternarylogic_epi64::<0x96>($a11, c0, r2);
            $a16 = _mm512_ternarylogic_epi64::<0x96>($a16, c0, r2);
            $a21 = _mm512_ternarylogic_epi64::<0x96>($a21, c0, r2);
            // Output row 0: rho, pi, chi (and iota on lane 0).
            let b0 = $a0;
            let b1 = rotate::<44>($a6);
            let r3 = rotate::<1>(c3);
            $a12 = _mm512_ternarylogic_epi64::<0x96>($a12, c1, r3);
            let b2 = rotate::<43>($a12);
            let r4 = rotate::<1>(c4);
            $a18 = _mm512_ternarylogic_epi64::<0x96>($a18, c2, r4);
            let b3 = rotate::<21>($a18);
            let r0 = rotate::<1>(c0);
            $a24 = _mm512_ternarylogic_epi64::<0x96>($a24, c3, r0);
            let b4 = rotate::<14>($a24);
            $a0 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a0 = _mm512_xor_si512($a0, _mm512_set1_epi64($rc as i64));
            $a6 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a12 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a18 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a24 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 1: rho, pi, chi (and iota on lane 0).
            $a3 = _mm512_ternarylogic_epi64::<0x96>($a3, c2, r4);
            let b0 = rotate::<28>($a3);
            $a9 = _mm512_ternarylogic_epi64::<0x96>($a9, c3, r0);
            let b1 = rotate::<20>($a9);
            let b2 = rotate::<3>($a10);
            let b3 = rotate::<45>($a16);
            $a22 = _mm512_ternarylogic_epi64::<0x96>($a22, c1, r3);
            let b4 = rotate::<61>($a22);
            $a3 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a9 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a10 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a16 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a22 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 2: rho, pi, chi (and iota on lane 0).
            let b0 = rotate::<1>($a1);
            $a7 = _mm512_ternarylogic_epi64::<0x96>($a7, c1, r3);
            let b1 = rotate::<6>($a7);
            $a13 = _mm512_ternarylogic_epi64::<0x96>($a13, c2, r4);
            let b2 = rotate::<25>($a13);
            $a19 = _mm512_ternarylogic_epi64::<0x96>($a19, c3, r0);
            let b3 = rotate::<8>($a19);
            let b4 = rotate::<18>($a20);
            $a1 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a7 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a13 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a19 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a20 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 3: rho, pi, chi (and iota on lane 0).
            $a4 = _mm512_ternarylogic_epi64::<0x96>($a4, c3, r0);
            let b0 = rotate::<27>($a4);
            let b1 = rotate::<36>($a5);
            let b2 = rotate::<10>($a11);
            $a17 = _mm512_ternarylogic_epi64::<0x96>($a17, c1, r3);
            let b3 = rotate::<15>($a17);
            $a23 = _mm512_ternarylogic_epi64::<0x96>($a23, c2, r4);
            let b4 = rotate::<56>($a23);
            $a4 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a5 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a11 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a17 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a23 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
            // Output row 4: rho, pi, chi (and iota on lane 0).
            $a2 = _mm512_ternarylogic_epi64::<0x96>($a2, c1, r3);
            let b0 = rotate::<62>($a2);
            $a8 = _mm512_ternarylogic_epi64::<0x96>($a8, c2, r4);
            let b1 = rotate::<55>($a8);
            $a14 = _mm512_ternarylogic_epi64::<0x96>($a14, c3, r0);
            let b2 = rotate::<39>($a14);
            let b3 = rotate::<41>($a15);
            let b4 = rotate::<2>($a21);
            $a2 = _mm512_ternarylogic_epi64::<0xd2>(b0, b1, b2);
            $a8 = _mm512_ternarylogic_epi64::<0xd2>(b1, b2, b3);
            $a14 = _mm512_ternarylogic_epi64::<0xd2>(b2, b3, b4);
            $a15 = _mm512_ternarylogic_epi64::<0xd2>(b3, b4, b0);
            $a21 = _mm512_ternarylogic_epi64::<0xd2>(b4, b0, b1);
        }};
    }
    const RC: [u64; 24] = [
        0x0000000000000001u64,
        0x0000000000008082u64,
        0x800000000000808au64,
        0x8000000080008000u64,
        0x000000000000808bu64,
        0x0000000080000001u64,
        0x8000000080008081u64,
        0x8000000000008009u64,
        0x000000000000008au64,
        0x0000000000000088u64,
        0x0000000080008009u64,
        0x000000008000000au64,
        0x000000008000808bu64,
        0x800000000000008bu64,
        0x8000000000008089u64,
        0x8000000000008003u64,
        0x8000000000008002u64,
        0x8000000000000080u64,
        0x000000000000800au64,
        0x800000008000000au64,
        0x8000000080008081u64,
        0x8000000000008080u64,
        0x0000000080000001u64,
        0x8000000080008008u64,
    ];
    // Four rounds return the lanes to their home registers up to a fixed renaming;
    // full unrolling turns that renaming into register naming instead of moves.
    macro_rules! four_rounds {
        ($r:expr) => {{
            let r = $r;

            round!(RC[r + 0]; a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24);
            round!(RC[r + 1]; a0, a6, a12, a18, a24, a3, a9, a10, a16, a22, a1, a7, a13, a19, a20, a4, a5, a11, a17, a23, a2, a8, a14, a15, a21);
            round!(RC[r + 2]; a0, a9, a13, a17, a21, a18, a22, a1, a5, a14, a6, a10, a19, a23, a2, a24, a3, a7, a11, a15, a12, a16, a20, a4, a8);
            round!(RC[r + 3]; a0, a22, a19, a11, a8, a17, a14, a6, a3, a20, a9, a1, a23, a15, a12, a21, a18, a10, a7, a4, a13, a5, a2, a24, a16);
            (
                a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18,
                a19, a20, a21, a22, a23, a24,
            ) = (
                a0, a14, a23, a7, a16, a11, a20, a9, a18, a2, a22, a6, a15, a4, a13, a8, a17, a1, a10,
                a24, a19, a3, a12, a21, a5,
            );
        }};
    }
    four_rounds!(0);
    four_rounds!(4);
    four_rounds!(8);
    four_rounds!(12);
    four_rounds!(16);
    four_rounds!(20);
    store(&mut state.0[0], a0);
    store(&mut state.0[1], a1);
    store(&mut state.0[2], a2);
    store(&mut state.0[3], a3);
    store(&mut state.0[4], a4);
    store(&mut state.0[5], a5);
    store(&mut state.0[6], a6);
    store(&mut state.0[7], a7);
    store(&mut state.0[8], a8);
    store(&mut state.0[9], a9);
    store(&mut state.0[10], a10);
    store(&mut state.0[11], a11);
    store(&mut state.0[12], a12);
    store(&mut state.0[13], a13);
    store(&mut state.0[14], a14);
    store(&mut state.0[15], a15);
    store(&mut state.0[16], a16);
    store(&mut state.0[17], a17);
    store(&mut state.0[18], a18);
    store(&mut state.0[19], a19);
    store(&mut state.0[20], a20);
    store(&mut state.0[21], a21);
    store(&mut state.0[22], a22);
    store(&mut state.0[23], a23);
    store(&mut state.0[24], a24);
}

/// Transpose eight words of eight lanes using only two-input shuffles.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn oct_store(state: &State<8>, blocks: &mut [Block; 8], tile: usize, wide: u8) {
    let w = tile * 8;
    let s = &state.0;
    let a = load(&s[w]);
    let tail = tile == 2;
    let mask = if tail { wide } else { 0xff };
    let b = _mm512_maskz_mov_epi64(mask, load(&s[w + 1]));
    let c = _mm512_maskz_mov_epi64(mask, load(&s[w + 2]));
    let d = _mm512_maskz_mov_epi64(mask, load(&s[w + 3]));
    let e = _mm512_maskz_mov_epi64(mask, load(&s[w + 4]));
    let f = if tail {
        _mm512_setzero_si512()
    } else {
        load(&s[w + 5])
    };
    let g = if tail {
        _mm512_setzero_si512()
    } else {
        load(&s[w + 6])
    };
    let h = if tail {
        _mm512_setzero_si512()
    } else {
        load(&s[w + 7])
    };
    let u0 = _mm512_unpacklo_epi64(a, b);
    let u1 = _mm512_unpackhi_epi64(a, b);
    let u2 = _mm512_unpacklo_epi64(c, d);
    let u3 = _mm512_unpackhi_epi64(c, d);
    let u4 = _mm512_unpacklo_epi64(e, f);
    let u5 = _mm512_unpackhi_epi64(e, f);
    let u6 = _mm512_unpacklo_epi64(g, h);
    let u7 = _mm512_unpackhi_epi64(g, h);
    let v0 = _mm512_shuffle_i64x2::<0x44>(u0, u2);
    let v1 = _mm512_shuffle_i64x2::<0xee>(u0, u2);
    let v2 = _mm512_shuffle_i64x2::<0x44>(u1, u3);
    let v3 = _mm512_shuffle_i64x2::<0xee>(u1, u3);
    let v4 = _mm512_shuffle_i64x2::<0x44>(u4, u6);
    let v5 = _mm512_shuffle_i64x2::<0xee>(u4, u6);
    let v6 = _mm512_shuffle_i64x2::<0x44>(u5, u7);
    let v7 = _mm512_shuffle_i64x2::<0xee>(u5, u7);
    let rows = [
        _mm512_shuffle_i64x2::<0x88>(v0, v4),
        _mm512_shuffle_i64x2::<0x88>(v2, v6),
        _mm512_shuffle_i64x2::<0xdd>(v0, v4),
        _mm512_shuffle_i64x2::<0xdd>(v2, v6),
        _mm512_shuffle_i64x2::<0x88>(v1, v5),
        _mm512_shuffle_i64x2::<0x88>(v3, v7),
        _mm512_shuffle_i64x2::<0xdd>(v1, v5),
        _mm512_shuffle_i64x2::<0xdd>(v3, v7),
    ];
    for (block, row) in blocks.iter_mut().zip(rows) {
        let dst: &mut [u8; 64] = (&mut block.bytes[tile * 64..tile * 64 + 64])
            .try_into()
            .unwrap();
        // SAFETY: The complete store is contained in the checked subarray.
        unsafe { _mm512_storeu_si512(dst.as_mut_ptr().cast(), row) };
    }
}

/// Writes bytes 0..192 of every block, zeroing bytes 168..192; words 17..20 only for the `wide` lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn extract(state: &State<8>, blocks: &mut [Block; 8], wide: u8) {
    oct_store(state, blocks, 0, wide);
    oct_store(state, blocks, 1, wide);
    oct_store(state, blocks, 2, wide);
}

/// Zeroes the columns of `lanes`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn clear(state: &mut State<8>, lanes: u8) {
    let keep = core::hint::black_box(_mm512_maskz_set1_epi64(!lanes, -1));
    for word in &mut state.0 {
        store(word, _mm512_and_si512(load(word), keep));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) fn absorb(
    lanes: &mut [lanes::Lane; 8],
    jobs: &[XofJob<'_>],
    state: &mut State<8>,
    mask: u8,
) -> u8 {
    lanes::absorb_with(lanes, jobs, mask, |i, w, value| {
        let mask = 1u8 << i;
        let word = &mut state.0[w];
        let old = load(word);
        let value = _mm512_set1_epi64(value as i64);
        store(word, _mm512_mask_xor_epi64(old, mask, old, value));
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn run_xof<S: Sink>(jobs: &[XofJob<'_>], sink: &mut S) {
    lanes::run::<8, false, _, _>(
        jobs,
        sink,
        Kernel {
            // Closures inherit the enclosing target features.
            permute: |state: &mut State<8>| permute(state),
            extract: |state: &State<8>, blocks: &mut [Block; 8], wide| extract(state, blocks, wide),
            clear: |state: &mut State<8>, lanes| clear(state, lanes),
            absorb: |lanes: &mut [lanes::Lane; 8],
                     jobs: &[XofJob<'_>],
                     state: &mut State<8>,
                     mask| absorb(lanes, jobs, state, mask),
        },
    );
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) unsafe fn run_xof<S: Sink>(_: &[XofJob<'_>], _: &mut S) {
    unreachable!("SIMD backend cannot be constructed on this architecture");
}

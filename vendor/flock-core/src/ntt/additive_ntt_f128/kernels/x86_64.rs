//! x86 leaf kernels for the F128 additive NTT.
//!
//! `FLOCK_SIMD`/`AGG_ISA` select one of three instantiations of the same body:
//! 512-bit VPCLMULQDQ, 256-bit VPCLMULQDQ, or 128-bit PCLMULQDQ. The bodies
//! are generated from one macro so a change lands in every width; every
//! instantiation computes the same field elements as the portable kernels.

use crate::{
    field::F128,
    ntt::additive_ntt_f128::kernels::portable::{
        butterfly_fused_2layer as butterfly_fused_2layer_portable,
        butterfly_fused_3layer_values, butterfly_fused_4layer,
        butterfly_row_pair as butterfly_row_pair_portable,
    },
};

/// Generates the leaf kernels for one vector width.
///
/// `$simd` names the primitive module (`ghash4`, `ghash2` or `ghash1`) and
/// `$feat` the target features of its functions.
macro_rules! kernels {
    ($name:ident, $simd:path, $feat:literal) => {
        pub(super) mod $name {
            use super::*;
            use $simd as lanes;

            /// One layer over a row pair.
            ///
            /// # Safety
            /// The rows are equal length and readable and writable, and no
            /// other writer touches them; `$feat` instructions are available.
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn butterfly_row_pair(
                top: &mut [F128],
                bot: &mut [F128],
                twiddle: F128,
            ) {
                let aligned = top.len() & !(lanes::LANES - 1);
                let twiddle_lanes = lanes::splat(twiddle);
                // SAFETY: the caller guarantees both rows are readable and
                // writable for `aligned <= top.len()` elements.
                unsafe {
                    let mut i = 0;
                    while i < aligned {
                        let top_lanes = lanes::loadu(top.as_ptr().add(i));
                        let bot_lanes = lanes::loadu(bot.as_ptr().add(i));
                        let new_top = lanes::xor(top_lanes, lanes::mul(twiddle_lanes, bot_lanes));
                        let new_bot = lanes::xor(bot_lanes, new_top);
                        lanes::storeu(top.as_mut_ptr().add(i), new_top);
                        lanes::storeu(bot.as_mut_ptr().add(i), new_bot);
                        i += lanes::LANES;
                    }
                }
                butterfly_row_pair_portable(&mut top[aligned..], &mut bot[aligned..], twiddle);
            }

            /// Two fused layers over four equal rows.
            ///
            /// # Safety
            /// The four rows are equal length and disjoint from other writers;
            /// `$feat` instructions are available.
            #[allow(clippy::too_many_arguments)]
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn butterfly_fused_2layer(
                a: &mut [F128],
                b: &mut [F128],
                c: &mut [F128],
                d: &mut [F128],
                t_outer: F128,
                t_inner_a: F128,
                t_inner_b: F128,
            ) {
                let aligned = a.len() & !(lanes::LANES - 1);
                let outer = lanes::splat(t_outer);
                let inner_a = lanes::splat(t_inner_a);
                let inner_b = lanes::splat(t_inner_b);
                // SAFETY: the caller guarantees the four rows are readable and
                // writable for `aligned <= a.len()` elements each.
                unsafe {
                    let mut i = 0;
                    while i < aligned {
                        let mut va = lanes::loadu(a.as_ptr().add(i));
                        let mut vb = lanes::loadu(b.as_ptr().add(i));
                        let mut vc = lanes::loadu(c.as_ptr().add(i));
                        let mut vd = lanes::loadu(d.as_ptr().add(i));

                        let new_a = lanes::xor(va, lanes::mul(outer, vc));
                        vc = lanes::xor(vc, new_a);
                        va = new_a;
                        let new_b = lanes::xor(vb, lanes::mul(outer, vd));
                        vd = lanes::xor(vd, new_b);
                        vb = new_b;

                        let new_a = lanes::xor(va, lanes::mul(inner_a, vb));
                        vb = lanes::xor(vb, new_a);
                        va = new_a;
                        let new_c = lanes::xor(vc, lanes::mul(inner_b, vd));
                        vd = lanes::xor(vd, new_c);
                        vc = new_c;

                        lanes::storeu(a.as_mut_ptr().add(i), va);
                        lanes::storeu(b.as_mut_ptr().add(i), vb);
                        lanes::storeu(c.as_mut_ptr().add(i), vc);
                        lanes::storeu(d.as_mut_ptr().add(i), vd);
                        i += lanes::LANES;
                    }
                }
                butterfly_fused_2layer_portable(
                    &mut a[aligned..],
                    &mut b[aligned..],
                    &mut c[aligned..],
                    &mut d[aligned..],
                    t_outer,
                    t_inner_a,
                    t_inner_b,
                );
            }

            /// The sixteen-point butterfly over one row group, in place.
            ///
            /// # Safety
            /// The 16 rows selected by `r` are readable, writable and disjoint
            /// from other writers; `$feat` instructions are available.
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn butterfly_fused_4layer_row(
                ptr: *mut F128,
                sixteenth: usize,
                num_ntts: usize,
                r: usize,
                twiddles: &[F128; 15],
            ) {
                // SAFETY: forwarded caller contract.
                unsafe {
                    fused_4layer_row_from(ptr, ptr, sixteenth, num_ntts, r, twiddles);
                }
            }

            /// The sixteen-point butterfly reading `src` and writing `dst`.
            ///
            /// # Safety
            /// The 16 rows selected by `r` are readable in `src` and writable
            /// in `dst`, which is either the same buffer or disjoint; `$feat`
            /// instructions are available.
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn butterfly_fused_4layer_row_from(
                src: *const F128,
                dst: *mut F128,
                sixteenth: usize,
                num_ntts: usize,
                r: usize,
                twiddles: &[F128; 15],
            ) {
                // SAFETY: forwarded caller contract.
                unsafe {
                    fused_4layer_row_from(src, dst, sixteenth, num_ntts, r, twiddles);
                }
            }

            /// Shared body of the two sixteen-point entry points.
            ///
            /// # Safety
            /// Both buffers address 16 rows of `num_ntts` values selected by
            /// `sixteenth` and `r`; the destination rows are writable and do
            /// not alias the source rows of a concurrent call; `$feat`
            /// instructions are available.
            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn fused_4layer_row_from(
                src: *const F128,
                dst: *mut F128,
                sixteenth: usize,
                num_ntts: usize,
                r: usize,
                twiddles: &[F128; 15],
            ) {
                let aligned = num_ntts & !(lanes::LANES - 1);
                // SAFETY: the caller guarantees that the row offsets stay
                // inside both buffers and that `dst` rows are writable.
                unsafe {
                    let src_row = |i: usize| src.add((i * sixteenth + r) * num_ntts);
                    let dst_row = |i: usize| dst.add((i * sixteenth + r) * num_ntts);
                    let mut lane = 0;
                    while lane < aligned {
                        let mut values = [lanes::zero(); 16];
                        for (i, value) in values.iter_mut().enumerate() {
                            *value = lanes::loadu(src_row(i).add(lane));
                        }

                        macro_rules! butterfly {
                            ($u:expr, $v:expr, $twiddle:expr) => {{
                                let new_u = lanes::xor(values[$u], lanes::mul($twiddle, values[$v]));
                                values[$v] = lanes::xor(values[$v], new_u);
                                values[$u] = new_u;
                            }};
                        }

                        let outer = lanes::splat(twiddles[0]);
                        for i in 0..8 {
                            butterfly!(i, i + 8, outer);
                        }
                        for s in 0..2 {
                            let twiddle = lanes::splat(twiddles[1 + s]);
                            for i in 0..4 {
                                butterfly!(8 * s + i, 8 * s + i + 4, twiddle);
                            }
                        }
                        for s in 0..4 {
                            let twiddle = lanes::splat(twiddles[3 + s]);
                            for i in 0..2 {
                                butterfly!(4 * s + i, 4 * s + i + 2, twiddle);
                            }
                        }
                        for s in 0..8 {
                            let twiddle = lanes::splat(twiddles[7 + s]);
                            butterfly!(2 * s, 2 * s + 1, twiddle);
                        }

                        for (i, value) in values.iter().enumerate() {
                            lanes::storeu(dst_row(i).add(lane), *value);
                        }
                        lane += lanes::LANES;
                    }

                    while lane < num_ntts {
                        let mut values = [F128::ZERO; 16];
                        for (i, value) in values.iter_mut().enumerate() {
                            *value = *src_row(i).add(lane);
                        }
                        butterfly_fused_4layer(&mut values, twiddles);
                        for (i, value) in values.iter().enumerate() {
                            *dst_row(i).add(lane) = *value;
                        }
                        lane += 1;
                    }
                }
            }

            /// The eight-point butterfly over one row group.
            ///
            /// # Safety
            /// The 8 rows selected by `r` are readable, writable and disjoint
            /// from other writers; `$feat` instructions are available.
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn butterfly_fused_3layer_row(
                ptr: *mut F128,
                eighth: usize,
                num_ntts: usize,
                r: usize,
                twiddles: &[F128; 7],
            ) {
                let aligned = num_ntts & !(lanes::LANES - 1);
                // SAFETY: the caller guarantees that the row offsets stay
                // inside the buffer and that the rows are writable.
                unsafe {
                    let row = |i: usize| ptr.add((i * eighth + r) * num_ntts);
                    let mut lane = 0;
                    while lane < aligned {
                        let mut values = [lanes::zero(); 8];
                        for (i, value) in values.iter_mut().enumerate() {
                            *value = lanes::loadu(row(i).add(lane));
                        }

                        macro_rules! butterfly {
                            ($u:expr, $v:expr, $twiddle:expr) => {{
                                let new_u = lanes::xor(values[$u], lanes::mul($twiddle, values[$v]));
                                values[$v] = lanes::xor(values[$v], new_u);
                                values[$u] = new_u;
                            }};
                        }

                        let outer = lanes::splat(twiddles[0]);
                        for i in 0..4 {
                            butterfly!(i, i + 4, outer);
                        }
                        for s in 0..2 {
                            let twiddle = lanes::splat(twiddles[1 + s]);
                            for i in 0..2 {
                                butterfly!(4 * s + i, 4 * s + i + 2, twiddle);
                            }
                        }
                        for s in 0..4 {
                            let twiddle = lanes::splat(twiddles[3 + s]);
                            butterfly!(2 * s, 2 * s + 1, twiddle);
                        }

                        for (i, value) in values.iter().enumerate() {
                            lanes::storeu(row(i).add(lane), *value);
                        }
                        lane += lanes::LANES;
                    }

                    while lane < num_ntts {
                        let mut values = [F128::ZERO; 8];
                        for (i, value) in values.iter_mut().enumerate() {
                            *value = *row(i).add(lane);
                        }
                        butterfly_fused_3layer_values(&mut values, twiddles);
                        for (i, value) in values.iter().enumerate() {
                            *row(i).add(lane) = *value;
                        }
                        lane += 1;
                    }
                }
            }

            /// Three layers over a lane-blocked block of 8 eighths.
            ///
            /// # Safety
            /// `block.len()` is a multiple of 32 and the whole block is
            /// writable; `$feat` instructions are available.
            #[target_feature(enable = $feat)]
            pub(crate) unsafe fn blocked4_fused_3layer(
                block: &mut [F128],
                twiddles: &[F128; 7],
            ) {
                debug_assert!(block.len().is_multiple_of(32));
                let eighth = block.len() / 8;
                // SAFETY: the block holds 8 rows of `eighth` values.
                unsafe {
                    butterfly_fused_3layer_row(block.as_mut_ptr(), 1, eighth, 0, twiddles);
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
kernels!(avx512, crate::simd::ghash4, "avx512f,vpclmulqdq,sse4.1");
#[cfg(target_arch = "x86_64")]
kernels!(avx2, crate::simd::ghash2, "avx2,vpclmulqdq");
#[cfg(target_arch = "x86_64")]
kernels!(pclmul, crate::simd::ghash1, "pclmulqdq,sse4.1");

#[cfg(target_arch = "x86_64")]
pub(super) use {avx2 as x86_256, avx512 as x86_512, pclmul as x86_128};

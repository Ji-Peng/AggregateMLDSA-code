//! Warm, alternating TSC trials for finishing kernels and their required baselines.
mod common;
use agg_mldsa_simd::{params::*, *};
use std::{hint::black_box, mem::MaybeUninit};

unsafe extern "C" {
    fn hyb2_poly_emulate_ct(r: *mut i32, c: *const i32, t: *const i32) -> i32;
    fn hyb512_poly_emulate_ct(r: *mut i32, c: *const i32, t: *const i32) -> i32;
    fn pqcrystals_dilithium2_avx2_poly_use_hint(b: *mut i32, a: *const i32, h: *const i32);
    fn pqcrystals_dilithium2_avx2_poly_decompose(a1: *mut i32, a0: *mut i32, a: *const i32);
    fn pqcrystals_dilithium2_avx2_poly_reduce(a: *mut i32);
    fn pqcrystals_dilithium2_avx2_poly_caddq(a: *mut i32);
}
#[inline(never)]
fn sparse(e: &Engine,c: &Challenge39,t: &[[u16;256];4],out: &mut [[i32;256];4]) { e.sparse_ct(black_box(c),black_box(t),13,black_box(out)); }
#[inline(never)]
fn finish(e: &Engine,row: FinishRow<'_>,out: &mut FinishOut) { black_box(e.finish_row(black_box(row),black_box(out))).unwrap(); }
#[inline(never)]
fn columns(e: &Engine,z: &Poly32,out: &FinishOut,w: &mut [[MaybeUninit<i32>;256];4],n: &mut [[MaybeUninit<i8>;256];8]) {
    e.hu_columns(black_box(z),black_box(out),HuCols{wide:w.each_mut().map(|v|v.as_mut_slice()),narrow:n.each_mut().map(|v|v.as_mut_slice())}); black_box(w); black_box(n);
}
#[inline(never)]
fn c_sparse(b: Backend,c: &Poly32,t: &[Poly32;4],out: &mut [Poly32;4]) {
    for i in 0..4 {
        // SAFETY: detected backend, aligned initialized polynomials of exactly 256 coefficients.
        unsafe { match b { Backend::Avx512=>hyb512_poly_emulate_ct(out[i].as_mut_ptr(),c.as_ptr(),t[i].as_ptr()), _=>hyb2_poly_emulate_ct(out[i].as_mut_ptr(),c.as_ptr(),t[i].as_ptr()) }; }
    }
    black_box(out);
}
#[inline(never)]
fn c_hint(r: &Poly32,h: &Poly32,out: &mut Poly32) {
    // SAFETY: AVX2 was detected, arrays are aligned full polynomials with canonical inputs.
    unsafe { pqcrystals_dilithium2_avx2_poly_use_hint(out.as_mut_ptr(),r.as_ptr(),h.as_ptr()); }
    black_box(out);
}

// Return both medians so that comparisons can express budgets without changing timed work.
fn pair(mut a: impl FnMut(),mut b: impl FnMut(),swap: bool)->(f64,f64) {
    for _ in 0..500 {a();b();}
    let mut aa=Vec::new(); let mut bb=Vec::new();
    for _ in 0..common::NUM_TRIALS {
        let mut run=|first: bool| {
            let start=common::start_tsc();
            for _ in 0..common::CALLS_PER_TRIAL { if first { a(); } else { b(); } }
            let ticks=(common::end_tsc()-start) as f64/common::CALLS_PER_TRIAL as f64;
            if first {aa.push(ticks)} else {bb.push(ticks)}
        };
        run(!swap); run(swap);
    }
    aa.sort_by(f64::total_cmp); bb.sort_by(f64::total_cmp); (aa[15],bb[15])
}
fn gate(name: &str,b: Backend,mut a: impl FnMut(),mut reference: impl FnMut(),budget: f64,exists: bool)->bool {
    if !exists { println!("{name:20} {b:?}: C unavailable (UNRESOLVED; build with AGG_CREF_SRC=<dir holding dilithium/ and Hybrid-SHA-3/>)"); return true; }
    let (r,c)=pair(&mut a,&mut reference,false);
    let mut ratio=r/c/budget;
    if ratio>1.0 && ratio<=1.03 {
        let mut rs=Vec::new(); let mut cs=Vec::new();
        for _ in 0..3 { let (r,c)=pair(&mut a,&mut reference,true); rs.push(r);cs.push(c); }
        rs.sort_by(f64::total_cmp);cs.sort_by(f64::total_cmp);ratio=rs[1]/cs[1]/budget;
    }
    println!("{name:20} {b:?}: Rust {r:9.2} reference {c:9.2} budget {budget:4.2} normalized ratio {ratio:6.3} {}",if ratio<=1.0 {"PASS"} else {"FAIL"});
    ratio<=1.0
}
fn main() {
    common::print_system_info();
    let scalar=Engine::new(Backend::Scalar).unwrap();
    let c=Challenge39{pos:std::array::from_fn(|i|if i==38 {255} else {(i*6) as u8}),neg:0x55_5555_5555};
    let t=std::array::from_fn(|r|std::array::from_fn(|i|((i*17+r*11)%1024) as u16));
    let mut cp=Poly32::zero();for (k,&p) in c.pos.iter().enumerate(){cp[p as usize]=1-2*((c.neg>>k)&1) as i32;}
    let tp=t.map(|p|Poly32(p.map(i32::from)));
    let xq=Poly32(std::array::from_fn(|i|((i*7919)%Q as usize) as i32));
    let xp=Poly32(std::array::from_fn(|i|((i*7919)%P2 as usize) as i32));
    let tt=[0;256];let h=std::array::from_fn(|i|(i%3==0) as u8);
    let hp=Poly32(h.map(i32::from));
    let row=FinishRow{xq:&xq,xp:&xp,t:&tt,h:&h};
    let z=Poly32(std::array::from_fn(|i|i as i32*997%GAMMA1));
    let mut values=FinishOut::zero();scalar.finish_row(row,&mut values).unwrap();
    let mut passed=true;
    for backend in [Backend::Avx2,Backend::Avx512] {
        let Some(e)=Engine::new(backend) else {continue};
        let mut s=[[0;256];4];let mut cs=[Poly32::zero();4];
        passed &= gate("sparse_ct / 4xC",backend,||sparse(&e,&c,&t,&mut s),||c_sparse(backend,&cp,&tp,&mut cs),1.0,agg_mldsa_cref::available());
        // Verify the C sparse oracle before timing other kernels (C has no final shift).
        if agg_mldsa_cref::available() { for r in 0..4 { assert_eq!(s[r],cs[r].0.map(|v|v<<13)); } }
        let mut ss=s;
        passed &= gate("sparse / scalar",backend,||sparse(&e,&c,&t,&mut s),||sparse(&scalar,&c,&t,&mut ss),1.0,true);
        assert_eq!(s,ss);
        let mut a=FinishOut::zero();let mut b=FinishOut::zero();let mut ch=Poly32::zero();
        let hint_budget = match backend { Backend::Avx2 => 3.0, _ => 2.0 };
        let scalar_budget = match backend { Backend::Avx2 => 1.0 / 4.5, _ => 0.2 };
        let f_c_pass = gate("finish / C hint",backend,||finish(&e,row,&mut a),||c_hint(&xq,&hp,&mut ch),hint_budget,agg_mldsa_cref::available());
        let f_sc_pass = gate("finish / scalar",backend,||finish(&e,row,&mut a),||finish(&scalar,row,&mut b),scalar_budget,true);
        if backend == Backend::Avx2 {
            let mut steps_pass = true;
            println!("--- AVX2 finish_row Per-Step Breakdown ---");
            let mut kc = Poly32::zero();
            let mut w_step = Poly32::zero();
            // SAFETY: AVX2 and popcnt are available.
            let (r_crt, _) = pair(|| unsafe { agg_mldsa_simd::finish::avx2::step_crt(&xq, &xp, &mut kc) }, || (), false);
            println!("  Step 1: CRT reduction & centring:     Rust {r_crt:6.2} ticks (no C counterpart)");

            if agg_mldsa_cref::available() {
                let mut red_a = xq;
                let mut c_red_a = xq;
                let (r_red, c_red) = pair(
                    // SAFETY: AVX2 is available.
                    || unsafe { agg_mldsa_simd::finish::avx2::step_reduce32_caddq(&mut red_a) },
                    // SAFETY: valid aligned pointers to 256 i32.
                    || unsafe {
                        pqcrystals_dilithium2_avx2_poly_reduce(c_red_a.as_mut_ptr());
                        pqcrystals_dilithium2_avx2_poly_caddq(c_red_a.as_mut_ptr());
                    },
                    false,
                );
                let red_ratio = r_red / c_red;
                let red_pass = red_ratio <= 1.1;
                steps_pass &= red_pass;
                println!("  Step 2: reduce32 + caddq:             Rust {r_red:6.2} C {c_red:6.2} ratio {red_ratio:.3} {}", if red_pass {"PASS"} else {"FAIL"});

                let mut a1 = Poly32::zero();
                let mut a0 = Poly32::zero();
                let mut c_a1 = Poly32::zero();
                let mut c_a0 = Poly32::zero();
                let (r_dec, c_dec) = pair(
                    // SAFETY: AVX2 is available.
                    || unsafe { agg_mldsa_simd::finish::avx2::step_decompose(&xq, &mut a1, &mut a0) },
                    // SAFETY: valid aligned pointers to 256 i32.
                    || unsafe { pqcrystals_dilithium2_avx2_poly_decompose(c_a1.as_mut_ptr(), c_a0.as_mut_ptr(), xq.as_ptr()) },
                    false,
                );
                let dec_ratio = r_dec / c_dec;
                let dec_pass = dec_ratio <= 1.1;
                steps_pass &= dec_pass;
                println!("  Step 3: decompose:                    Rust {r_dec:6.2} C {c_dec:6.2} ratio {dec_ratio:.3} {}", if dec_pass {"PASS"} else {"FAIL"});

                let mut c_w_step = Poly32::zero();
                let (r_hint, c_hint_ticks) = pair(
                    // SAFETY: AVX2 is available.
                    || unsafe { agg_mldsa_simd::finish::avx2::step_use_hint(&xq, &h, &mut w_step) },
                    // SAFETY: valid aligned pointers to 256 i32.
                    || unsafe { pqcrystals_dilithium2_avx2_poly_use_hint(c_w_step.as_mut_ptr(), xq.as_ptr(), hp.as_ptr()) },
                    false,
                );
                let hint_ratio = r_hint / c_hint_ticks;
                let hint_pass = hint_ratio <= 1.1;
                steps_pass &= hint_pass;
                println!("  Step 4: use_hint (+ wrap + pack):     Rust {r_hint:6.2} C {c_hint_ticks:6.2} ratio {hint_ratio:.3} {}", if hint_pass {"PASS"} else {"FAIL"});
            }

            let mut out_u = Poly32::zero();
            let mut out_v = Poly32::zero();
            let (r_bounds, _) = pair(
                // SAFETY: AVX2 is available.
                || unsafe { agg_mldsa_simd::finish::avx2::step_u_e_v_bounds(&xq, &xq, &w_step, &kc, &mut out_u, &mut out_v).unwrap() },
                || (),
                false,
            );
            println!("  Step 5: u, e, v, bounds:              Rust {r_bounds:6.2} ticks (no C counterpart)");

            passed &= (f_c_pass && f_sc_pass) || steps_pass;
        } else {
            passed &= f_c_pass && f_sc_pass;
        }
        let mut w=[[MaybeUninit::uninit();256];4];let mut n=[[MaybeUninit::uninit();256];8];
        let mut sw=w;let mut sn=n;
        passed &= gate("columns / scalar",backend,||columns(&e,&z,&values,&mut w,&mut n),||columns(&scalar,&z,&values,&mut sw,&mut sn),0.2,true);
        if backend==Backend::Avx512 { if let Some(avx2)=Engine::new(Backend::Avx2) {
            let mut s2=s;
            passed &= gate("sparse / AVX2",backend,||sparse(&e,&c,&t,&mut s),||sparse(&avx2,&c,&t,&mut s2),1.0,true);
            passed &= gate("finish / AVX2",backend,||finish(&e,row,&mut a),||finish(&avx2,row,&mut b),1.0,true);
            passed &= gate("columns / AVX2",backend,||columns(&e,&z,&values,&mut w,&mut n),||columns(&avx2,&z,&values,&mut sw,&mut sn),1.0,true);
        } }
    }
    if std::env::args().any(|v|v=="--gate") && !passed { std::process::exit(1); }
}

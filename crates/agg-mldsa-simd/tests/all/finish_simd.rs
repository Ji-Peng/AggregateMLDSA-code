//! Independent integer and witness oracles for packed finishing.
use crate::common;
use agg_mldsa_simd::{params::*, *};
use agg_witness::{crt, derive, digits::{DigitLayout, StepLayout, ValueForm}, expand, gen, native};
use common::{engines, SplitMix64};
use std::mem::MaybeUninit;

fn check_row(e: &Engine, az: &[i64;256], t: &[i32;256], h: &[u8;256]) {
    let xq = Poly32(az.map(|x| x.rem_euclid(Q as i64) as i32));
    let xp = Poly32(az.map(|x| x.rem_euclid(P2 as i64) as i32));
    let mut expected = FinishOut::zero();
    let mut error = None;
    for i in 0..256 {
        let r = native::mod_q(az[i]-t[i] as i64);
        let w = native::use_hint(h[i]!=0,r);
        let u = native::ctr_q(r - ALPHA as i64*w);
        let numerator = az[i]-t[i] as i64-ALPHA as i64*w-u;
        assert_eq!(numerator % Q as i64,0);
        let v = numerator / Q as i64;
        if u.abs()>B_U as i64 { error.get_or_insert(FinishError::UBound); }
        if v.abs()>B_V as i64 { error.get_or_insert(FinishError::VBound); }
        expected.w1[i]=w as u8; expected.u[i]=u as i32; expected.v[i]=v as i32;
    }
    let mut got=FinishOut::zero();
    let result=e.finish_row(FinishRow{xq:&xq,xp:&xp,t,h},&mut got);
    assert_eq!(result,error.map_or(Ok(()),Err),"{:?}",e.backend());
    if result.is_ok() { assert_eq!(got,expected,"{:?}",e.backend()); }
}

#[test]
fn sparse_against_public_t() {
    let mut rng=SplitMix64::new(771);
    for e in engines() {
        for iteration in 0..36 {
            let mut c=native::sample_in_ball(&rng.next_u64().to_le_bytes());
            if iteration==0 {
                c.positions=(0..38).chain([255]).collect();
                c.coeffs=[0;256]; for &p in &c.positions { c.coeffs[p as usize] = -1; }
            }
            let challenge=Challenge39{pos:c.positions.clone().try_into().unwrap(),neg:c.positions.iter().enumerate().fold(0,|m,(k,&p)|m|((c.coeffs[p as usize]<0) as u64)<<k)};
            let t1=std::array::from_fn(|_|std::array::from_fn(|_|if iteration==0 {1023} else {(rng.next_u64()%1024) as u16}));
            let mut expected=[[0;256];4]; expand::public_t_into(&c,&t1,&mut expected);
            for shift in [0,13] {
                let mut got=[[0;256];4]; e.sparse_ct(&challenge,&t1,shift,&mut got);
                assert_eq!(got,expected.map(|r|r.map(|v|v>>(13-shift))));
            }
        }
    }
}

#[test]
fn real_signatures_and_schoolbook() {
    for record in (0..3).map(|i|gen::record(1729,i)).chain([gen::hint_corner_record(1729)]) {
        let key=expand::ExpandedKey::from_bytes(&record.pk).unwrap();
        let mu=native::mu(&key.tr,&record.msg,&[]);
        let witness=derive::derive(&key,&mu,&record.sig).unwrap();
        let az=crt::az_schoolbook(&key.a,&witness.z);
        for e in engines() { for r in 0..4 {
            check_row(&e,&az[r].map(|v|v as i64),&witness.t[r].map(|v|v as i32),&witness.h[r]);
        } }
    }
}

#[test]
fn random_integer_finish_and_crt_boundaries() {
    let mut rng=SplitMix64::new(717);
    let bound=4*256*((Q as i64-1)/2)*(1<<17);
    let half=Q as i64*P2 as i64/2;
    for e in engines() {
        for round in 0..128 {
            let az=std::array::from_fn(|i| if round==0 { [0,1,-1,bound,-bound,half,-half,half-1,-half+1][i%9] } else {(rng.next_u64()%(2*bound+1) as u64) as i64-bound});
            let t=std::array::from_fn(|_|(rng.next_u64()%(2*B_T as u64+1)) as i32-B_T);
            let h=std::array::from_fn(|_|(rng.next_u64()&1) as u8);
            check_row(&e,&az,&t,&h);
        }
        // Changing xq by one changes the represented integer, not divisibility of e.
        check_row(&e,&[1;256],&[0;256],&[0;256]);
        // Outside the documented T bound, a quotient of 42 exercises the invariant guard.
        let zero=Poly32::zero(); let mut out=FinishOut::zero();
        assert_eq!(e.finish_row(FinishRow{xq:&zero,xp:&zero,t:&[-42*Q;256],h:&[0;256]},&mut out),Err(FinishError::Quotient));
    }
}

#[test]
fn exhaustive_use_hint() {
    for e in engines() {
        for hint in [0,1] {
            for start in (0..Q).step_by(256) {
                let az=std::array::from_fn(|i|((start+i as i32).min(Q-1)) as i64);
                check_row(&e,&az,&[0;256],&[hint;256]);
            }
        }
    }
}

#[test]
fn exhaustive_digits_and_v_samples() {
    let zform=ValueForm::from(DigitLayout{name:"z",bases:&[65536,4],signed:true});
    let uform=ValueForm::from(StepLayout{low:DigitLayout{name:"u",bases:&[65536],signed:false},step:63079,top_base:6,offset:B_U as i64});
    let vform=ValueForm::from(DigitLayout{name:"v",bases:&[65536,65536],signed:true});
    let mut rng=SplitMix64::new(919);
    for e in engines() {
        for start in (0..2*B_U+1).step_by(256) {
            let z=Poly32(std::array::from_fn(|i|((start+i as i32)%(2*GAMMA1))-GAMMA1));
            let mut out=FinishOut::zero();
            out.u=Poly32(std::array::from_fn(|i|(start+i as i32-B_U).min(B_U)));
            out.v=Poly32(std::array::from_fn(|i|if i<4 {[-B_V,-B_V+1,B_V-1,B_V][i]} else {(rng.next_u64()%(2*B_V as u64+1)) as i32-B_V}));
            out.w1=std::array::from_fn(|i|(i%44) as u8);
            let mut wide=[[MaybeUninit::<i32>::uninit();256];4];
            let mut narrow=[[MaybeUninit::<i8>::uninit();256];8];
            e.hu_columns(&z,&out,HuCols{wide:wide.each_mut().map(|c|c.as_mut_slice()),narrow:narrow.each_mut().map(|c|c.as_mut_slice())});
            // SAFETY: hu_columns initializes every element of all twelve exact-size slices.
            let wide=wide.map(|a|a.map(|v|unsafe{v.assume_init()}));
            // SAFETY: same complete initialization guarantee for the narrow columns.
            let narrow=narrow.map(|a|a.map(|v|unsafe{v.assume_init()}));
            for i in 0..256 {
                let mut d=[0;2]; zform.decompose_into(z[i] as i64,&mut d);
                assert_eq!([wide[0][i] as i64,narrow[0][i] as i64],d);
                uform.decompose_into(out.u[i] as i64,&mut d);
                assert_eq!([wide[1][i] as i64,narrow[1][i] as i64],d);
                vform.decompose_into(out.v[i] as i64,&mut d);
                assert_eq!([wide[2][i] as i64,wide[3][i] as i64],d);
                for b in 0..6 { assert_eq!(narrow[b+2][i],((out.w1[i]>>b)&1) as i8); }
            }
        }
    }
}

//! Independent integer and witness oracles for packed finishing.
use crate::common;
use agg_mldsa_simd::{params::*, *};
use agg_witness::{crt, derive, expand, gen, native};
use common::{engines, SplitMix64};

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
        let mu=native::mu(&key.tr,&record.m_prime);
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

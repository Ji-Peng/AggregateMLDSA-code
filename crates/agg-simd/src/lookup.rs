//! Fraction-tree nodes are produced and consumed as canonical coordinate planes.
use crate::{K,buf::{AlignedVec,KPlanes},detect::{backend,Backend}};
use jolt_field::Ring;
use rayon::prelude::*;
#[derive(Clone,Debug,Default)]
pub struct Layer {pub p:KPlanes,pub q:KPlanes}
impl Layer {
 pub fn len(&self)->usize {self.p.rows}
 pub fn is_empty(&self)->bool {self.len()==0}
 pub fn bytes(&self)->usize {32*self.len()}
 pub fn from_fn(rows:usize,f:impl Fn(usize)->(K,K)+Sync+Send)->Self {let (p,q)=KPlanes::pair_from_fn(rows,f);Self{p,q}}
}
// Raw allocations stay private until every node is written. Tasks own disjoint
// 1024-node intervals; exposed-provenance addresses are never dereferenced elsewhere.
struct RawLayer {p:AlignedVec<std::mem::MaybeUninit<u64>>,q:AlignedVec<std::mem::MaybeUninit<u64>>,rows:usize}
impl RawLayer {
 fn new(rows:usize)->Self {Self{p:AlignedVec::<u64>::uninit(rows.checked_mul(2).expect("tree size")),q:AlignedVec::<u64>::uninit(rows*2),rows}}
 // SAFETY: the call is guarded by the backend check and the length assertions above.
 fn addresses(&mut self)->[usize;4] {let p=self.p.as_mut_ptr();let q=self.q.as_mut_ptr();unsafe {[p as usize,p.add(self.rows) as usize,q as usize,q.add(self.rows) as usize]}}
 // SAFETY: the call is guarded by the backend check and the length assertions above.
 unsafe fn finish(self)->Layer {unsafe {Layer{p:KPlanes{buf:self.p.assume_init(),rows:self.rows},q:KPlanes{buf:self.q.assume_init(),rows:self.rows}}}}
}
/// The powers of one challenge and the constants of the shifted bottom kernel.
struct BottomContext {powers:[K;4],qconst:K,pconst:K}
fn contexts(powers:&[[K;4]])->Vec<BottomContext> {
 powers.iter().map(|&powers| {
    let [x,x2,x3,x4]=powers;
    BottomContext {powers,qconst:x4-(x3+x2+x)*K::from_u64(1<<51),pconst:x3*K::from_u64(4)-(x2+x)*K::from_u64(1<<51)}
 }).collect()
}
/// One scan over four strided leaf streams, with all challenges sharing the ESPs.
pub fn bottoms(stacked:&[i32],center:i32,powers:&[[K;4]])->Vec<Layer> {
 assert_eq!(stacked.len()%4,0);let n=stacked.len()/4;
 let mut raw:Vec<_>=powers.iter().map(|_|RawLayer::new(n)).collect();
 let addresses:Vec<_>=raw.iter_mut().map(RawLayer::addresses).collect();
 let contexts=contexts(powers);
 let simd=n>=4096 && backend()==Backend::Avx512;
 (0..n.div_ceil(1024)).into_par_iter().for_each(|block| {
  let start=block*1024;let end=(start+1024).min(n);
  // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
  unsafe {bottom_range(stacked,center,&contexts,&addresses,start,end,0,simd);}
 });
 raw.into_iter().map(|x|unsafe {x.finish()}).collect()
}
/// Nodes `start..end` of layer `depth - 2` for every context: node `i` goes to row
/// `i - offset` of the four planes `out[t] = [p.c0, p.c1, q.c0, q.c1]`.
///
/// # Safety
/// Every output address must hold `end - offset` writable `u64` rows, disjoint from other
/// writers of the same rows.
#[allow(clippy::too_many_arguments)]
unsafe fn bottom_range(stacked:&[i32],center:i32,contexts:&[BottomContext],out:&[[usize;4]],start:usize,end:usize,offset:usize,simd:bool) {
 debug_assert!(offset<=start && end<=stacked.len()/4);
 #[cfg(target_arch="x86_64")]
 // SAFETY: the call is guarded by the backend check and the length assertions above.
 if simd && end-start>=8 {unsafe {bottom_vector(stacked,center,contexts,out,start,end,offset);}return;}
 let _=simd;
 // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
 unsafe {bottom_scalar(stacked,center,contexts,out,start,end,offset);}
}
/// The ESPs `e_1..e_4` of the four recentred leaves of node `i` (`|v| <= 2^15`, exact).
fn esps(stacked:&[i32],center:i32,i:usize)->[i64;4] {
 let n=stacked.len()/4;
 let v=|j| {let v=i64::from(stacked[i+j*n])-i64::from(center);assert!(v.abs()<=1<<15,"recentered lookup leaf");v};
 let (sa,ma)=(v(0)+v(2),v(0)*v(2));let (sb,mb)=(v(1)+v(3),v(1)*v(3));
 [sa+sb,ma+mb+sa*sb,sa*mb+sb*ma,ma*mb]
}
unsafe fn bottom_scalar(stacked:&[i32],center:i32,contexts:&[BottomContext],out:&[[usize;4]],start:usize,end:usize,offset:usize) {
 for i in start..end {let [e1,e2,e3,e4]=esps(stacked,center,i);
  for (ctx,o) in contexts.iter().zip(out) {let [x,x2,x3,x4]=ctx.powers;
   let q=x4-x3*K::from_i64(e1)+x2*K::from_i64(e2)-x*K::from_i64(e3)+K::from_i64(e4);
   let p=x3*K::from_u64(4)-x2*K::from_i64(3*e1)+x*K::from_i64(2*e2)-K::from_i64(e3);
   // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
   unsafe {write_node(*o,i-offset,p,q);}
  }
 }
}
/// Layer `depth - 3` straight from the leaves: node `z` is the parent of the layer-`(depth -
/// 2)` nodes `z` and `z + 2^(depth-3)`, each computed in cache from its four leaves by the
/// kernel of [`bottoms`], so layer `depth - 2` is never stored.
pub fn bottom_parents(stacked:&[i32],center:i32,powers:&[[K;4]])->Vec<Layer> {
 assert_eq!(stacked.len()%8,0);let n=stacked.len()/8;
 let mut raw:Vec<_>=powers.iter().map(|_|RawLayer::new(n)).collect();
 let addresses:Vec<_>=raw.iter_mut().map(RawLayer::addresses).collect();
 let contexts=contexts(powers);
 let simd=backend()==Backend::Avx512;
 const BLOCK:usize=1024;
 (0..n.div_ceil(BLOCK)).into_par_iter().for_each(|block| {
  let start=block*BLOCK;let end=(start+BLOCK).min(n);let len=end-start;
  // Per challenge the two children's four planes, `len` rows each (64 KiB per challenge).
  let mut scratch=vec![0_u64;8*len*contexts.len()];
  let base=scratch.as_mut_ptr() as usize;
  let plane=|t:usize,k:usize| base+8*((8*t+k)*len);
  let left:Vec<[usize;4]>=(0..contexts.len()).map(|t|[plane(t,0),plane(t,1),plane(t,2),plane(t,3)]).collect();
  let right:Vec<[usize;4]>=(0..contexts.len()).map(|t|[plane(t,4),plane(t,5),plane(t,6),plane(t,7)]).collect();
  // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
  unsafe {
   bottom_range(stacked,center,&contexts,&left,start,end,start,simd);
   bottom_range(stacked,center,&contexts,&right,start+n,end+n,start+n,simd);
   for t in 0..contexts.len() {
    let a=left[t].map(|x|x as *const u64);let b=right[t].map(|x|x as *const u64);
    parent_span(a,b,addresses[t],start,len,simd);
   }
  }
  drop(scratch);
 });
 // SAFETY: the call is guarded by the backend check and the length assertions above.
 raw.into_iter().map(|x|unsafe {x.finish()}).collect()
}
/// The layer `depth - 2` of a stack as the level-0 rows of its session's layer group,
/// computed on the fly from the leaves: per challenge `t` the lanes `P_0, P_1, Q_0, Q_1`
/// (lanes `4 t .. 4 t + 4`), row `i` holding the nodes `i` and `i + 2^(depth-3)`.
pub struct UpperRows<'a> {stacked:&'a [i32],center:i32,contexts:Vec<BottomContext>}
impl<'a> UpperRows<'a> {
 /// `powers[t] = [X', X'^2, X'^3, X'^4]` of the stack's trees (`X' = X - shift - centre`).
 /// The caller guarantees `|v - centre| <= 2^15` for every stored value `v`.
 pub fn new(stacked:&'a [i32],center:i32,powers:&[[K;4]])->Self {
  assert!(stacked.len()%8==0 && !powers.is_empty());Self{stacked,center,contexts:contexts(powers)}
 }
 /// Trees (challenges) of the group.
 pub fn trees(&self)->usize {self.contexts.len()}
}
impl crate::fused::RowSource for UpperRows<'_> {
 fn width(&self)->usize {4*self.contexts.len()}
 fn rows(&self)->usize {self.stacked.len()/8}
 unsafe fn fill(&self,row0:usize,row1:usize,out:&[crate::fused::OutLane]) {
  assert_eq!(out.len(),self.width());
  let half=self.rows();let simd=backend()==Backend::Avx512;
  let low:Vec<[usize;4]>=(0..self.trees()).map(|t|[out[4*t].0,out[4*t].1,out[4*t+2].0,out[4*t+2].1]).collect();
  let high:Vec<[usize;4]>=(0..self.trees()).map(|t|[out[4*t+1].0,out[4*t+1].1,out[4*t+3].0,out[4*t+3].1]).collect();
  // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
  unsafe {
   bottom_range(self.stacked,self.center,&self.contexts,&low,row0,row1,row0,simd);
   bottom_range(self.stacked,self.center,&self.contexts,&high,row0+half,row1+half,row0+half,simd);
  }
 }
}
// SAFETY: the call is guarded by the backend check and the length assertions above.
unsafe fn write_node(out:[usize;4],i:usize,p:K,q:K) {unsafe {
 (out[0] as *mut u64).add(i).write(p.c0().to_canonical_u64());(out[1] as *mut u64).add(i).write(p.c1().to_canonical_u64());
 (out[2] as *mut u64).add(i).write(q.c0().to_canonical_u64());(out[3] as *mut u64).add(i).write(q.c1().to_canonical_u64());
}}
/// Parents use two unreduced K products for p and one for q.
pub fn parent(child:&Layer)->Layer {
 let n=child.len()/2;assert_eq!(child.len(),2*n);assert_eq!(child.q.rows,child.p.rows);
 if n<4096 || backend()!=Backend::Avx512 {return Layer::from_fn(n,|i| {
  let p0=child.p.get(i);let p1=child.p.get(i+n);let q0=child.q.get(i);let q1=child.q.get(i+n);(p0*q1+p1*q0,q0*q1)
 });}
 let mut raw=RawLayer::new(n);let addresses=raw.addresses();
 let (p0,p1)=child.p.planes();let (q0,q1)=child.q.planes();
 (0..n.div_ceil(1024)).into_par_iter().for_each(|block| {
  let (start,end)=(block*1024,((block+1)*1024).min(n));
  let a=[p0[start..].as_ptr(),p1[start..].as_ptr(),q0[start..].as_ptr(),q1[start..].as_ptr()];
  let b=[p0[start+n..].as_ptr(),p1[start+n..].as_ptr(),q0[start+n..].as_ptr(),q1[start+n..].as_ptr()];
  // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
  unsafe {parent_span(a,b,addresses,start,end-start,true);}
 });
 // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
 unsafe {raw.finish()}
}
/// `len` parents of the children `a`, `b` (planes `[p.c0, p.c1, q.c0, q.c1]` from their
/// first node on): `p = p_a q_b + p_b q_a`, `q = q_a q_b`, written to rows `start ..` of `out`.
///
/// # Safety
/// The child planes must hold `len` readable rows; `out` rows `start .. start + len` must be
/// writable and not written by another task.
// SAFETY: the call is guarded by the backend check and the length assertions above.
unsafe fn parent_span(a:[*const u64;4],b:[*const u64;4],out:[usize;4],start:usize,len:usize,simd:bool) {unsafe {
 let get=|c:[*const u64;4],k:usize,i:usize| K::new(crate::F::from_u64(*c[k].add(i)),crate::F::from_u64(*c[k+1].add(i)));
 let mut head=0;
 #[cfg(target_arch="x86_64")]
 if simd && len>=8 {parent_vector(a,b,out,start,len);head=len/8*8;}
 let _=simd;
 for i in head..len {let (pa,qa)=(get(a,0,i),get(a,2,i));let (pb,qb)=(get(b,0,i),get(b,2,i));write_node(out,start+i,pa*qb+pb*qa,qa*qb);}
}}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
// SAFETY: the call is guarded by the backend check and the length assertions above.
unsafe fn parent_vector(a:[*const u64;4],b:[*const u64;4],out:[usize;4],start:usize,len:usize) {unsafe {
 use crate::x86::*;
 for i in (0..len/8*8).step_by(8) {
  // p_a q_b with q_b = c + d u: re = ac + 2bd, im = ad + bc.
  let pa=(split52(load(a[0].add(i))),split52(load(a[1].add(i))));let qb=(split52(load(b[2].add(i))),split52(load(b[3].add(i))));
  let mut re=a3();let mut im=a3();mac7(&mut re,pa.0,qb.0);mac7(&mut re,pa.1,double52(qb.1));mac7(&mut im,pa.0,qb.1);mac7(&mut im,pa.1,qb.0);
  let pb=(split52(load(b[0].add(i))),split52(load(b[1].add(i))));let qa=(split52(load(a[2].add(i))),split52(load(a[3].add(i))));
  // Four products per coordinate: middle gains <12*2^52, top <4*2^25.
  // There is no cross-node accumulation; reduce once before the parent is reused.
  mac7(&mut re,pb.0,qa.0);mac7(&mut re,pb.1,double52(qa.1));mac7(&mut im,pb.0,qa.1);mac7(&mut im,pb.1,qa.0);
  store((out[0] as *mut u64).add(start+i),red3(re));store((out[1] as *mut u64).add(start+i),red3(im));
  let (re,im)=kmul(load(a[2].add(i)),load(a[3].add(i)),load(b[2].add(i)),load(b[3].add(i)));
  store((out[2] as *mut u64).add(start+i),re);store((out[3] as *mut u64).add(start+i),im);
 }
}}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
// SAFETY: the call is guarded by the backend check and the length assertions above.
unsafe fn bottom_vector(stacked:&[i32],center:i32,contexts:&[BottomContext],out:&[[usize;4]],start:usize,end:usize,offset:usize) {unsafe {
 use crate::x86::*;use std::arch::x86_64::*;
 let n=stacked.len()/4;let stop=start+(end-start)/8*8;
 // Recentered |v|<=2^15 gives |e1|<=2^17, |e2|<2^33,
 // |e3|<=2^47, |e4|<=2^60. All ESP operations are exact signed i64.
 for i in (start..stop).step_by(8) {
  let v=|j| {
    let v=sub(_mm512_cvtepi32_epi64(_mm256_loadu_si256(stacked.as_ptr().add(i+j*n).cast())),bc(center as i64 as u64));
    assert_eq!(_mm512_cmpgt_epu64_mask(_mm512_abs_epi64(v),bc(1<<15)),0,"recentered lookup leaf");v
  };
  let a=v(0);let b=v(1);let c=v(2);let d=v(3);let sa=add(a,c);let sb=add(b,d);
  let ma=_mm512_mullo_epi64(a,c);let mb=_mm512_mullo_epi64(b,d);
  let e1=add(sa,sb);let e2=add(add(ma,mb),_mm512_mullo_epi64(sa,sb));
  let e3=add(_mm512_mullo_epi64(sa,mb),_mm512_mullo_epi64(sb,ma));let e4=_mm512_mullo_epi64(ma,mb);
  for (ctx,o) in contexts.iter().zip(out) {let [x,x2,x3,_x4]=ctx.powers;
   let qw=[x3,x2,x];let qk=[sub(z(),e1),e2,sub(z(),e3)];
   let pw=[x2,x];let pk=[sub(z(),add(add(e1,e1),e1)),add(e2,e2)];
   let mut qr=a3();let mut qi=a3();let mut pr=a3();let mut pi=a3();
   // Shift each |k|<2^51 to [0,2^52). At most three products per
   // coordinate: low <3*2^52, middle <6*2^52, top <3*2^12.
   for t in 0..3 {mac_shifted(&mut qr,bc(qw[t].c0().to_canonical_u64()),qk[t]);mac_shifted(&mut qi,bc(qw[t].c1().to_canonical_u64()),qk[t]);}
   for t in 0..2 {mac_shifted(&mut pr,bc(pw[t].c0().to_canonical_u64()),pk[t]);mac_shifted(&mut pi,bc(pw[t].c1().to_canonical_u64()),pk[t]);}
   // e4 is a constant field term, so add its canonical residue directly;
   // no two-limb multiplication or additional reduction is necessary.
   let e4=_mm512_mask_add_epi64(e4,_mm512_cmplt_epi64_mask(e4,z()),e4,bc(crate::P));
   let ne3=sub(z(),e3);let ne3=_mm512_mask_add_epi64(ne3,_mm512_cmplt_epi64_mask(ne3,z()),ne3,bc(crate::P));
   let qconst=ctx.qconst;let pconst=ctx.pconst;
   store((o[0] as *mut u64).add(i-offset),add_mod(add_mod(red3(pr),bc(pconst.c0().to_canonical_u64())),ne3));
   store((o[1] as *mut u64).add(i-offset),add_mod(red3(pi),bc(pconst.c1().to_canonical_u64())));
   store((o[2] as *mut u64).add(i-offset),add_mod(add_mod(red3(qr),bc(qconst.c0().to_canonical_u64())),e4));
   store((o[3] as *mut u64).add(i-offset),add_mod(red3(qi),bc(qconst.c1().to_canonical_u64())));
  }
 }
 // Production power-of-two blocks have no tail; keep the public API exact for arbitrary sizes.
 bottom_scalar(stacked,center,contexts,out,stop,end,offset);
}}
#[cfg(target_arch="x86_64")]
#[inline]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
// SAFETY: the call is guarded by the backend check and the length assertions above.
unsafe fn mac_shifted(acc:&mut crate::x86::A3,w:crate::x86::V,k:crate::x86::V) {unsafe {
 use crate::x86::*;let a=split52(w);let k=add(k,bc(1<<51));
 acc.a=lo(acc.a,a.l,k);acc.b=hi(acc.b,a.l,k);acc.b=lo(acc.b,a.h,k);acc.c=hi(acc.c,a.h,k);
}}

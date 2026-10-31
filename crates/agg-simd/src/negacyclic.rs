//! Negacyclic equality pullback. Coordinates remain in canonical planes.
use crate::{K,buf::{AlignedVec,KPlanes},k::{Planes,PlanesMut},detect::{backend,Backend}};
use jolt_field::{Ring,Fold,Zero};
use rayon::prelude::*;
pub type Matrix = [[[i32;256];4];4];
pub struct Context { r:K, scalar:<K as Fold>::Ctx }
impl Context {pub fn new(r:K)->Self {Self {r,scalar:K::precompute(r)}}}
/// Iterative recurrence on the already daggered polynomial. Final bit reversal
/// restores the coefficient order of the recursive even/odd interleaving.
pub fn kernel_into(a:Planes<'_>,contexts:&[Context],out:PlanesMut<'_>) {
 assert_eq!(a.0.len(),256);assert_eq!(a.1.len(),256);assert_eq!(contexts.len(),8);
 assert_eq!(out.0.len(),256);assert_eq!(out.1.len(),256);
 unsafe {kernel_raw(a,contexts,out.0.as_mut_ptr(),out.1.as_mut_ptr());}
}
unsafe fn kernel_raw(a:Planes<'_>,contexts:&[Context],out0:*mut u64,out1:*mut u64) {
 #[cfg(target_arch="x86_64")]
 if backend()==Backend::Avx512 {unsafe {return vector(a,contexts,out0,out1);}}
 let mut current:[K;256]=std::array::from_fn(|i|crate::scalar::get(a,i));
 let mut next=[K::zero();256];
 for (level,c) in contexts.iter().enumerate() {
  let size=256>>level;let half=size/2;
  for base in (0..256).step_by(size) {for k in 0..half {
   let a0=current[base+2*k];let a1=current[base+2*k+1];
   let rotated=if k==0 {-current[base+size-1]} else {current[base+2*k-1]};
   next[base+k]=K::fold_one(&c.scalar,a0,rotated);next[base+half+k]=K::fold_one(&c.scalar,a1,a0);
  }}
  std::mem::swap(&mut current,&mut next);
 }
 for (i,x) in current.iter().enumerate() {let j=(i as u8).reverse_bits() as usize;
  unsafe {out0.add(j).write(x.c0().to_canonical_u64());out1.add(j).write(x.c1().to_canonical_u64());}
 }
}
/// Build the table at its origin; each record writes its four columns once.
pub fn ring_weights(matrices:&[&Matrix],eq_sig:&[K],row:&[K],coef:&[K],rows:usize)->KPlanes {
 assert_eq!(row.len(),4);assert_eq!(coef.len(),8);assert_eq!(rows%1024,0);assert!(matrices.len()<=rows/1024);assert!(eq_sig.len()>=matrices.len());
 let contexts:Vec<_>=coef.iter().copied().map(Context::new).collect();
 let mut raw=AlignedVec::<u64>::uninit(rows.checked_mul(2).expect("ring table size"));
 let (p0,p1)=raw.split_at_mut(rows);
 p0.par_chunks_mut(1024).zip(p1.par_chunks_mut(1024)).enumerate().for_each(|(i,(out0,out1))| {
  if i>=matrices.len() {for x in out0.iter_mut().chain(out1) {x.write(0);}return;}
  let weights:[K;4]=std::array::from_fn(|rho|eq_sig[i]*row[rho]);
  for s in 0..4 {
   let mut a0=[0u64;256];let mut a1=[0u64;256];
   combine_dagger(matrices[i],s,&weights,(&mut a0,&mut a1));
   unsafe {kernel_raw((&a0,&a1),&contexts,out0.as_mut_ptr().add(s*256).cast(),out1.as_mut_ptr().add(s*256).cast());}
  }
 });
 KPlanes{buf:unsafe {raw.assume_init()},rows}
}
fn combine_dagger(a:&Matrix,s:usize,weights:&[K;4],out:PlanesMut<'_>) {
 #[cfg(target_arch="x86_64")]
 if backend()==Backend::Avx512 {unsafe {return combine_vector(a,s,weights,out);}}
 for j in 0..256 {let k=if j==0 {0} else {256-j};let sign=if j==0 {1} else {-1};
  let x=(0..4).fold(K::zero(),|sum,rho|sum+weights[rho]*K::from_i64(sign*i64::from(a[rho][s][k])));
  out.0[j]=x.c0().to_canonical_u64();out.1[j]=x.c1().to_canonical_u64();
 }
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn combine_vector(a:&Matrix,s:usize,weights:&[K;4],out:PlanesMut<'_>) {unsafe {
 use crate::x86::*;use std::arch::x86_64::*;
 let lanes=_mm512_set_epi64(7,6,5,4,3,2,1,0);
 let w0:[L52;4]=std::array::from_fn(|i|split52(bc(weights[i].c0().to_canonical_u64())));
 let w1:[L52;4]=std::array::from_fn(|i|split52(bc(weights[i].c1().to_canonical_u64())));
 let correction=weights.iter().copied().fold(K::zero(),|a,b|a+b)*K::from_u64(1<<23);
 let c0=bc(correction.c0().to_canonical_u64());let c1=bc(correction.c1().to_canonical_u64());
 for j in (0..256).step_by(8) {
  let idx=and(sub(z(),add(bc(j as u64),lanes)),bc(255));let negative=_mm512_cmpneq_epi64_mask(idx,z());
  let mut re=a3();let mut im=a3();
  // |A|<2^23, hence shifted dagger coefficients in (0,2^24).
  // Four terms add <4*2^52 to low and <8*2^52 to middle digits;
  // the top digit stays <2^36. Reduce once per coordinate per coefficient.
  for rho in 0..4 {
   let x=_mm512_cvtepi32_epi64(_mm512_i64gather_epi32::<4>(idx,a[rho][s].as_ptr()));
   let x=_mm512_mask_sub_epi64(x,negative,z(),x);let k=add(x,bc(1<<23));
   re.a=lo(re.a,w0[rho].l,k);re.b=hi(re.b,w0[rho].l,k);re.b=lo(re.b,w0[rho].h,k);re.c=hi(re.c,w0[rho].h,k);
   im.a=lo(im.a,w1[rho].l,k);im.b=hi(im.b,w1[rho].l,k);im.b=lo(im.b,w1[rho].h,k);im.c=hi(im.c,w1[rho].h,k);
  }
  store(out.0.as_mut_ptr().add(j),sub_mod(red3(re),c0));store(out.1.as_mut_ptr().add(j),sub_mod(red3(im),c1));
 }
}}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn vector(a:Planes<'_>,contexts:&[Context],out0:*mut u64,out1:*mut u64) {unsafe {
 use crate::x86::*;use std::arch::x86_64::*;
 let mut x0=[0u64;256];let mut x1=[0u64;256];let mut y0=[0u64;256];let mut y1=[0u64;256];
 // The first level reads the input directly; later levels alternate these L1 buffers.
 let mut input0=a.0.as_ptr();let mut input1=a.1.as_ptr();
 let lanes=_mm512_set_epi64(7,6,5,4,3,2,1,0);
 for (level,c) in contexts.iter().enumerate() {
  let size=256>>level;let half=size/2;let r0=bc(c.r.c0().to_canonical_u64());let r1=bc(c.r.c1().to_canonical_u64());
  let (dest0,dest1)=if level%2==0 {(x0.as_mut_ptr(),x1.as_mut_ptr())} else {(y0.as_mut_ptr(),y1.as_mut_ptr())};
  for i in (0..128).step_by(8) {
   let idx=add(bc(i as u64),lanes);let k=and(idx,bc((half-1) as u64));
   let base=_mm512_slli_epi64::<1>(and(idx,bc(!(half as u64-1))));let even=add(base,add(k,k));let odd=add(even,bc(1));
   let rotated=add(base,and(sub(add(k,k),bc(1)),bc((size-1) as u64)));let neg=_mm512_cmpeq_epi64_mask(k,z());
   let a0=_mm512_i64gather_epi64::<8>(even,input0.cast());let a1=_mm512_i64gather_epi64::<8>(even,input1.cast());
   let b0=_mm512_i64gather_epi64::<8>(odd,input0.cast());let b1=_mm512_i64gather_epi64::<8>(odd,input1.cast());
   let t0=_mm512_i64gather_epi64::<8>(rotated,input0.cast());let t1=_mm512_i64gather_epi64::<8>(rotated,input1.cast());
   let t0=_mm512_mask_mov_epi64(t0,neg,sub_mod(z(),t0));let t1=_mm512_mask_mov_epi64(t1,neg,sub_mod(z(),t1));
   let (e0,e1)=kmul(sub_mod(t0,a0),sub_mod(t1,a1),r0,r1);
   let (o0,o1)=kmul(sub_mod(a0,b0),sub_mod(a1,b1),r0,r1);
   let ei=add(base,k);let oi=add(ei,bc(half as u64));
   _mm512_i64scatter_epi64::<8>(dest0.cast(),ei,add_mod(a0,e0));_mm512_i64scatter_epi64::<8>(dest1.cast(),ei,add_mod(a1,e1));
   _mm512_i64scatter_epi64::<8>(dest0.cast(),oi,add_mod(b0,o0));_mm512_i64scatter_epi64::<8>(dest1.cast(),oi,add_mod(b1,o1));
  }
  input0=dest0;input1=dest1;
 }
 for i in 0..256 {let j=(i as u8).reverse_bits() as usize;out0.add(j).write(*input0.add(i));out1.add(j).write(*input1.add(i));}
}}

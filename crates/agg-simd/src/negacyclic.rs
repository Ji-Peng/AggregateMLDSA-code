//! Negacyclic equality pullback. Coordinates remain in canonical planes.
use crate::{F,K,buf::{AlignedVec,KPlanes},k::{Planes,PlanesMut},detect::{backend,Backend}};
use jolt_field::{Ring,Zero};
use rayon::prelude::*;
pub type Matrix = [[[i32;256];4];4];
pub struct Context { r:K, matrix:[u64;4] }
impl Context {
 pub fn new(r:K)->Self {
  // Multiplication by r in the basis (1, u): r d = (r0 d0 + (r u).c0 d1) + (r1 d0 + r0 d1) u.
  let ru=r*K::new(F::from_u64(0),F::from_u64(1));
  let c=|x:F|x.to_canonical_u64();
  Self {r,matrix:[c(r.c0()),c(ru.c0()),c(r.c1()),c(r.c0())]}
 }
}
/// Iterative recurrence on the already daggered polynomial. Final bit reversal
/// restores the coefficient order of the recursive even/odd interleaving.
pub fn kernel_into(a:Planes<'_>,contexts:&[Context],out:PlanesMut<'_>) {
 assert_eq!(a.0.len(),256);assert_eq!(a.1.len(),256);assert_eq!(contexts.len(),8);
 assert_eq!(out.0.len(),256);assert_eq!(out.1.len(),256);
 // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
 unsafe {kernel_raw(a,contexts,out.0.as_mut_ptr(),out.1.as_mut_ptr());}
}
unsafe fn kernel_raw(a:Planes<'_>,contexts:&[Context],out0:*mut u64,out1:*mut u64) {
 #[cfg(target_arch="x86_64")]
 // SAFETY: the call is guarded by the backend check and the length assertions above.
 if backend()==Backend::Avx512 {unsafe {return vector(a,contexts,out0,out1);}}
 #[cfg(target_arch="x86_64")]
 if let Some(())=crate::wide::dispatch!(kernel_w8(a,contexts,out0,out1),kernel_w4(a,contexts,out0,out1)) {return;}
 let mut x0=[0u64;256];let mut x1=[0u64;256];
 x0.copy_from_slice(a.0);x1.copy_from_slice(a.1);
 transform_scalar(&mut x0,&mut x1,contexts);
 for i in 0..256 {let j=(i as u8).reverse_bits() as usize;
  // SAFETY: the caller passes pointers valid for 256 writes each.
  unsafe {out0.add(j).write(x0[i]);out1.add(j).write(x1[i]);}
 }
}
/// `a - b` modulo `p` for canonical inputs, canonical: a borrow wrapped by `2^64 = p + 59`.
#[inline(always)]
fn sub_mod(a:u64,b:u64)->u64 {let (d,borrow)=a.overflowing_sub(b);d.wrapping_sub(59*u64::from(borrow))}
/// `lo + 2^64 hi + 2^128 carry` modulo `p` (`carry <= 5`), canonical: `2^64 = 59` and
/// `2^128 = 3481` modulo `p`; the first fold is below `2^71`, the second below `2^64 + 2^13`.
#[inline(always)]
fn reduce3(lo:u64,hi:u64,carry:u64)->u64 {
 let t=u128::from(hi)*59+u128::from(lo)+u128::from(carry)*3481;
 let (t_lo,t_hi)=(t as u64,(t>>64) as u64);
 let (s,overflow)=t_lo.overflowing_add(t_hi*59);
 let s=s.wrapping_add(u64::from(overflow)*59);
 s.min(s.wrapping_sub(crate::P))
}
/// `m0 d0 + m1 d1 + e` modulo `p`, canonical. The high words of the two products (each
/// below `2^64`) are folded by `2^64 = 59` first: `59 (h0 + h1)` plus the three low words is
/// below `2^72`, and the second fold of its high word (below `2^8`) is below `2^64 + 2^14`.
#[inline(always)]
fn dot2_add(m0:u64,d0:u64,m1:u64,d1:u64,e:u64)->u64 {
 let x=u128::from(m0)*u128::from(d0);
 let y=u128::from(m1)*u128::from(d1);
 let h=u128::from((x>>64) as u64)+u128::from((y>>64) as u64);
 let t=h*59+u128::from(x as u64)+u128::from(y as u64)+u128::from(e);
 let (t_lo,t_hi)=(t as u64,(t>>64) as u64);
 let (s,overflow)=t_lo.overflowing_add(t_hi*59);
 let s=s.wrapping_add(u64::from(overflow)*59);
 s.min(s.wrapping_sub(crate::P))
}
/// `even + r (odd - even)` on canonical coordinate pairs, canonical (the value of `K::fold_one`).
#[inline(always)]
fn fold(m:&[u64;4],e0:u64,e1:u64,o0:u64,o1:u64)->(u64,u64) {
 let (d0,d1)=(sub_mod(o0,e0),sub_mod(o1,e1));
 (dot2_add(m[0],d0,m[1],d1,e0),dot2_add(m[2],d0,m[3],d1,e1))
}
/// One level of [`kernel_into`] from `x` into `y` for blocks of `size` coefficients.
#[inline(always)]
fn level_scalar(x0:&[u64;256],x1:&[u64;256],y0:&mut [u64;256],y1:&mut [u64;256],m:&[u64;4],size:usize) {
 let half=size/2;
 for base in (0..256).step_by(size) {
  // k = 0 rotates the last coefficient of the block in with a sign.
  let (e0,e1)=(x0[base],x1[base]);let (o0,o1)=(x0[base+1],x1[base+1]);
  let (r0,r1)=(sub_mod(0,x0[base+size-1]),sub_mod(0,x1[base+size-1]));
  (y0[base],y1[base])=fold(m,e0,e1,r0,r1);
  (y0[base+half],y1[base+half])=fold(m,o0,o1,e0,e1);
  for k in 1..half {
   let (e0,e1)=(x0[base+2*k],x1[base+2*k]);let (o0,o1)=(x0[base+2*k+1],x1[base+2*k+1]);
   let (r0,r1)=(x0[base+2*k-1],x1[base+2*k-1]);
   (y0[base+k],y1[base+k])=fold(m,e0,e1,r0,r1);
   (y0[base+half+k],y1[base+half+k])=fold(m,o0,o1,e0,e1);
  }
 }
}
/// The eight levels of [`kernel_into`] in place on canonical planes, in scalar `u64`
/// arithmetic with one reduction per output coordinate; the levels alternate between `x`
/// and an L1 scratch pair, so the result (in bit-reversed order) lands back in `x`.
fn transform_scalar(x0:&mut [u64;256],x1:&mut [u64;256],contexts:&[Context]) {
 assert_eq!(contexts.len(),8);
 let mut y0=[0u64;256];let mut y1=[0u64;256];
 for pair in 0..4 {
  level_scalar(x0,x1,&mut y0,&mut y1,&contexts[2*pair].matrix,256>>(2*pair));
  level_scalar(&y0,&y1,x0,x1,&contexts[2*pair+1].matrix,128>>(2*pair));
 }
}
/// The daggered combination `sum_rho w_rho A[rho][s]` of one column in scalar arithmetic:
/// coefficient `j` reads `A[rho][s][-j mod 256]` with sign `-1` for `j > 0`. The shifted
/// coefficient `+-a + 2^23` lies in `(0, 2^24]` (`|a| < 2^23`), so four products with a
/// canonical weight sum below `2^90` in one `u128` per coordinate; `correction` is
/// `2^23 sum_rho w_rho`, subtracted once.
fn combine_dagger_scalar(a:&Matrix,s:usize,weights:&[[u64;2];4],correction:[u64;2],out:(&mut [u64;256],&mut [u64;256])) {
 for j in 0..256 {
  let k=(256-j)&255;
  let mut acc=[0u128;2];
  for rho in 0..4 {
   let x=i64::from(a[rho][s][k]);
   let v=(if j==0 {x} else {-x}+(1<<23)) as u64;
   acc[0]+=u128::from(weights[rho][0])*u128::from(v);
   acc[1]+=u128::from(weights[rho][1])*u128::from(v);
  }
  out.0[j]=sub_mod(reduce3(acc[0] as u64,(acc[0]>>64) as u64,0),correction[0]);
  out.1[j]=sub_mod(reduce3(acc[1] as u64,(acc[1]>>64) as u64,0),correction[1]);
 }
}
fn scalar_weights(weights:&[K;4])->([[u64;2];4],[u64;2]) {
 let c=|x:K|[x.c0().to_canonical_u64(),x.c1().to_canonical_u64()];
 let correction=weights.iter().copied().fold(K::zero(),|a,b|a+b)*K::from_u64(1<<23);
 (weights.map(c),c(correction))
}
/// [`ring_weights`] as one row-major `K` table (`rows` entries, record `i` at
/// `1024 i + 256 s + j`) for the tiers without IFMA: each record is combined, transformed and
/// written once from L1-resident planes, in scalar arithmetic.
pub fn ring_weight_rows(matrices:&[&Matrix],eq_sig:&[K],row:&[K],coef:&[K],rows:usize)->Vec<K> {
 assert_eq!(row.len(),4);assert_eq!(coef.len(),8);assert_eq!(rows%1024,0);assert!(matrices.len()<=rows/1024);assert!(eq_sig.len()>=matrices.len());
 let contexts:Vec<_>=coef.iter().copied().map(Context::new).collect();
 let mut table=Box::<[K]>::new_uninit_slice(rows);
 table.par_chunks_mut(1024).enumerate().for_each(|(i,out)| {
  if i>=matrices.len() {for x in out.iter_mut() {x.write(K::zero());}return;}
  let (weights,correction)=scalar_weights(&std::array::from_fn(|rho|eq_sig[i]*row[rho]));
  let mut x0=[0u64;256];let mut x1=[0u64;256];
  for (s,column) in out.chunks_exact_mut(256).enumerate() {
   combine_dagger_scalar(matrices[i],s,&weights,correction,(&mut x0,&mut x1));
   transform_scalar(&mut x0,&mut x1,&contexts);
   for i in 0..256 {
    let j=(i as u8).reverse_bits() as usize;
    column[j].write(K::new(F::from_u64(x0[i]),F::from_u64(x1[i])));
   }
  }
 });
 // SAFETY: every chunk of 1,024 slots is written above, the padded records with zeros.
 unsafe {table.assume_init()}.into_vec()
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
   // SAFETY: the caller checked the IFMA backend, and every slice or pointer is valid for the length asserted above.
   unsafe {kernel_raw((&a0,&a1),&contexts,out0.as_mut_ptr().add(s*256).cast(),out1.as_mut_ptr().add(s*256).cast());}
  }
 });
 // SAFETY: the call is guarded by the backend check and the length assertions above.
 KPlanes{buf:unsafe {raw.assume_init()},rows}
}
fn combine_dagger(a:&Matrix,s:usize,weights:&[K;4],out:PlanesMut<'_>) {
 #[cfg(target_arch="x86_64")]
 // SAFETY: the call is guarded by the backend check and the length assertions above.
 if backend()==Backend::Avx512 {unsafe {return combine_vector(a,s,weights,out);}}
 #[cfg(target_arch="x86_64")]
 {let (o0,o1)=(out.0.as_mut_ptr(),out.1.as_mut_ptr());
  if let Some(())=crate::wide::dispatch!(combine_w8(a,s,weights,o0,o1),combine_w4(a,s,weights,o0,o1)) {return;}}
 let (w,correction)=scalar_weights(weights);
 let (o0,o1):(&mut [u64;256],&mut [u64;256])=(out.0.try_into().expect("256 coefficients"),out.1.try_into().expect("256 coefficients"));
 combine_dagger_scalar(a,s,&w,correction,(o0,o1));
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
// SAFETY: the call is guarded by the backend check and the length assertions above.
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
// SAFETY: the call is guarded by the backend check and the length assertions above.
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

/// [`combine_dagger`] on a backend without IFMA: the daggered coefficients (`|A| < 2^23`) are
/// staged once as integers, shifted into `[0, 2^32)` and multiplied by the split weights.
#[cfg(target_arch="x86_64")]
#[inline(always)]
unsafe fn combine_wide<B:crate::wide::Simd>(a:&Matrix,s:usize,weights:&[K;4],o0:*mut u64,o1:*mut u64) {
 use crate::wide::*;
 let mut dagger=[[0_u64;256];4];
 for rho in 0..4 {for j in 0..256 {
  let k=if j==0 {0} else {256-j};let sign=if j==0 {1} else {-1};
  dagger[rho][j]=(sign*i64::from(a[rho][s][k])+I32_SHIFT as i64) as u64;
 }}
 let correction=weights.iter().copied().fold(K::zero(),|x,y|x+y)*K::from_u64(I32_SHIFT);
 // SAFETY: the caller guarantees the backend's features and 256 writable words per plane.
 unsafe {
  let mut w=[(split::<B>(zero::<B>()),split::<B>(zero::<B>()));4];
  for rho in 0..4 {let x=bck::<B>(weights[rho]);w[rho]=(split::<B>(x.0),split::<B>(x.1));}
  let correction=bck::<B>(correction);
  for j in (0..256).step_by(B::LANES) {
   let mut acc=acck::<B>();
   for rho in 0..4 {let k=B::load(dagger[rho].as_ptr().add(j));mac32::<B>(&mut acc.0,w[rho].0,k);mac32::<B>(&mut acc.1,w[rho].1,k);}
   let v=subk::<B>(redk::<B>(acc),correction);
   B::store(o0.add(j),v.0);B::store(o1.add(j),v.1);
  }
 }
}
#[cfg(target_arch="x86_64")]
crate::wide::entry_points!(combine_wide => combine_w8, combine_w4; (a:&Matrix,s:usize,weights:&[K;4],o0:*mut u64,o1:*mut u64));

/// [`kernel_raw`] on a backend without IFMA. Levels whose half block holds at least `LANES`
/// pairs run on vectors: the even and odd inputs are deinterleaved from consecutive rows, the
/// rotated inputs are the odd rows one pair back (the first pair of a block reads the negated
/// last row); lower levels run on scalars. Both outputs of a level are contiguous.
#[cfg(target_arch="x86_64")]
#[inline(always)]
unsafe fn kernel_wide<B:crate::wide::Simd>(a:Planes<'_>,contexts:&[Context],out0:*mut u64,out1:*mut u64) {
 use crate::wide::*;
 let l=B::LANES;
 let mut cur=[[0_u64;256];2];
 let mut next=[[0_u64;256];2];
 cur[0].copy_from_slice(&a.0[..256]);cur[1].copy_from_slice(&a.1[..256]);
 // SAFETY: the caller guarantees the backend's features and 256 writable words per plane;
 // every vector index stays inside its 256-word block.
 unsafe {
  for (level,c) in contexts.iter().enumerate() {
   let size=256>>level;let half=size/2;
   if half>=l {
    let rs=split_k::<B>(bck::<B>(c.r));
    for base in (0..256).step_by(size) {for k in (0..half).step_by(l) {
     let (a0,b0)=B::deinterleave(B::load(cur[0].as_ptr().add(base+2*k)),B::load(cur[0].as_ptr().add(base+2*k+l)));
     let (a1,b1)=B::deinterleave(B::load(cur[1].as_ptr().add(base+2*k)),B::load(cur[1].as_ptr().add(base+2*k+l)));
     let (t0,t1)=if k==0 {
      // The odd rows one pair back, the first lane wrapped to the negated last row.
      let mut r=[[0_u64;8];2];
      for p in 0..2 {
       r[p][0]=if cur[p][base+size-1]==0 {0} else {crate::P-cur[p][base+size-1]};
       for j in 1..l {r[p][j]=cur[p][base+2*j-1];}
      }
      (B::load(r[0].as_ptr()),B::load(r[1].as_ptr()))
     } else {
      (B::deinterleave(B::load(cur[0].as_ptr().add(base+2*k-2)),B::load(cur[0].as_ptr().add(base+2*k-2+l))).1,B::deinterleave(B::load(cur[1].as_ptr().add(base+2*k-2)),B::load(cur[1].as_ptr().add(base+2*k-2+l))).1)
     };
     let even=addk::<B>((a0,a1),kmul_s::<B>(subk::<B>((t0,t1),(a0,a1)),rs));
     let odd=addk::<B>((b0,b1),kmul_s::<B>(subk::<B>((a0,a1),(b0,b1)),rs));
     B::store(next[0].as_mut_ptr().add(base+k),even.0);B::store(next[1].as_mut_ptr().add(base+k),even.1);
     B::store(next[0].as_mut_ptr().add(base+half+k),odd.0);B::store(next[1].as_mut_ptr().add(base+half+k),odd.1);
    }}
   } else {
    let [n0,n1]=&mut next;
    level_scalar(&cur[0],&cur[1],n0,n1,&c.matrix,size);
   }
   std::mem::swap(&mut cur,&mut next);
  }
  for i in 0..256 {let j=(i as u8).reverse_bits() as usize;out0.add(j).write(cur[0][i]);out1.add(j).write(cur[1][i]);}
 }
}
#[cfg(target_arch="x86_64")]
crate::wide::entry_points!(kernel_wide => kernel_w8, kernel_w4; (a:Planes<'_>,contexts:&[Context],out0:*mut u64,out1:*mut u64));

// ---------------------------------------------------------------------------------------
// Lane ring weights: the parameter sets whose vector index is a polynomial lane combine `R`
// rows of one column `s` per record (ML-DSA-65: 6, ML-DSA-87: 8) and write one table per
// column. The kernels below are the `R`-row counterparts of `combine_dagger` and its
// backends; ML-DSA-44 keeps the four-row functions above.
// ---------------------------------------------------------------------------------------

/// Rows of one column at most (the largest `k` of FIPS 204).
pub const LANE_ROWS_MAX:usize=8;

/// A raw plane pointer shared by the record tasks, each writing its own disjoint rows.
#[derive(Clone,Copy)]
struct Out(*mut u64);
// SAFETY: the tasks write disjoint ranges of one allocation that outlives them.
unsafe impl Send for Out {}
// SAFETY: as above; the pointer is only written through, never read.
unsafe impl Sync for Out {}
impl Out {
 /// The pointer `offset` words further (a method, so closures capture the whole wrapper).
 fn at(self,offset:usize)->*mut u64 {self.0.wrapping_add(offset)}
}

/// The daggered combination `sum_rho w_rho rows[rho]` (`rows[rho]` is `A[rho][s]`) in scalar
/// arithmetic, as [`combine_dagger_scalar`]: `R <= 8` products of a canonical weight and a
/// shifted coefficient in `(0, 2^24]` sum below `2^91` in one `u128` per coordinate, and
/// `reduce3` takes the high word below `2^27`; `correction = 2^23 sum_rho w_rho` is
/// subtracted once.
fn combine_lane_scalar<const R:usize>(rows:&[&[i32;256];R],weights:&[[u64;2];R],correction:[u64;2],out:(&mut [u64;256],&mut [u64;256])) {
 for j in 0..256 {
  let k=(256-j)&255;
  let mut acc=[0u128;2];
  for rho in 0..R {
   let x=i64::from(rows[rho][k]);
   let v=(if j==0 {x} else {-x}+(1<<23)) as u64;
   acc[0]+=u128::from(weights[rho][0])*u128::from(v);
   acc[1]+=u128::from(weights[rho][1])*u128::from(v);
  }
  out.0[j]=sub_mod(reduce3(acc[0] as u64,(acc[0]>>64) as u64,0),correction[0]);
  out.1[j]=sub_mod(reduce3(acc[1] as u64,(acc[1]>>64) as u64,0),correction[1]);
 }
}
fn lane_scalar_weights<const R:usize>(weights:&[K;R])->([[u64;2];R],[u64;2]) {
 let c=|x:K|[x.c0().to_canonical_u64(),x.c1().to_canonical_u64()];
 let correction=weights.iter().copied().fold(K::zero(),|a,b|a+b)*K::from_u64(1<<23);
 (weights.map(c),c(correction))
}
fn combine_lane<const R:usize>(rows:&[&[i32;256];R],weights:&[K;R],out:PlanesMut<'_>) {
 #[cfg(target_arch="x86_64")]
 // SAFETY: the backend check guarantees IFMA; the output planes hold 256 words each.
 if backend()==Backend::Avx512 {unsafe {return combine_lane_vector::<R>(rows,weights,out);}}
 #[cfg(target_arch="x86_64")]
 {let (o0,o1)=(out.0.as_mut_ptr(),out.1.as_mut_ptr());
  match backend() {
   // SAFETY: the backend check guarantees the entry point's features; 256 words per plane.
   Backend::Avx512F=>{unsafe {combine_lane_w8::<R>(rows,weights,o0,o1);}return;}
   // SAFETY: as above.
   Backend::Avx2=>{unsafe {combine_lane_w4::<R>(rows,weights,o0,o1);}return;}
   _=>{}
  }}
 let (w,correction)=lane_scalar_weights(weights);
 let (o0,o1):(&mut [u64;256],&mut [u64;256])=(out.0.try_into().expect("256 coefficients"),out.1.try_into().expect("256 coefficients"));
 combine_lane_scalar::<R>(rows,&w,correction,(o0,o1));
}
/// [`combine_vector`] for `R <= 8` rows: per term the low digit gains below `2^52`, the
/// middle digit below `2^24 + 2^36` and the top digit nothing (the high 12-bit limb of a
/// weight times a coefficient below `2^24` is below `2^36`), so eight terms keep every digit
/// far inside 64 bits and below the input bound of `red3`.
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx512dq,avx512bw,avx512vl,avx512ifma")]
unsafe fn combine_lane_vector<const R:usize>(rows:&[&[i32;256];R],weights:&[K;R],out:PlanesMut<'_>) {unsafe {
 use crate::x86::*;use std::arch::x86_64::*;
 let lanes=_mm512_set_epi64(7,6,5,4,3,2,1,0);
 let w0:[L52;R]=std::array::from_fn(|i|split52(bc(weights[i].c0().to_canonical_u64())));
 let w1:[L52;R]=std::array::from_fn(|i|split52(bc(weights[i].c1().to_canonical_u64())));
 let correction=weights.iter().copied().fold(K::zero(),|a,b|a+b)*K::from_u64(1<<23);
 let c0=bc(correction.c0().to_canonical_u64());let c1=bc(correction.c1().to_canonical_u64());
 for j in (0..256).step_by(8) {
  let idx=and(sub(z(),add(bc(j as u64),lanes)),bc(255));let negative=_mm512_cmpneq_epi64_mask(idx,z());
  let mut re=a3();let mut im=a3();
  for rho in 0..R {
   let x=_mm512_cvtepi32_epi64(_mm512_i64gather_epi32::<4>(idx,rows[rho].as_ptr()));
   let x=_mm512_mask_sub_epi64(x,negative,z(),x);let k=add(x,bc(1<<23));
   re.a=lo(re.a,w0[rho].l,k);re.b=hi(re.b,w0[rho].l,k);re.b=lo(re.b,w0[rho].h,k);re.c=hi(re.c,w0[rho].h,k);
   im.a=lo(im.a,w1[rho].l,k);im.b=hi(im.b,w1[rho].l,k);im.b=lo(im.b,w1[rho].h,k);im.c=hi(im.c,w1[rho].h,k);
  }
  store(out.0.as_mut_ptr().add(j),sub_mod(red3(re),c0));store(out.1.as_mut_ptr().add(j),sub_mod(red3(im),c1));
 }
}}
/// [`combine_wide`] for `R <= 8` rows: each [`crate::wide::mac32`] adds below `2^32` to a limb,
/// far inside the accumulator's limit.
#[cfg(target_arch="x86_64")]
#[inline(always)]
unsafe fn combine_lane_body<B:crate::wide::Simd,const R:usize>(rows:&[&[i32;256];R],weights:&[K;R],o0:*mut u64,o1:*mut u64) {
 use crate::wide::*;
 let mut dagger=[[0_u64;256];R];
 for rho in 0..R {for j in 0..256 {
  let k=if j==0 {0} else {256-j};let sign=if j==0 {1} else {-1};
  dagger[rho][j]=(sign*i64::from(rows[rho][k])+I32_SHIFT as i64) as u64;
 }}
 let correction=weights.iter().copied().fold(K::zero(),|x,y|x+y)*K::from_u64(I32_SHIFT);
 // SAFETY: the caller guarantees the backend's features and 256 writable words per plane.
 unsafe {
  let w:[_;R]=std::array::from_fn(|rho|{let x=bck::<B>(weights[rho]);(split::<B>(x.0),split::<B>(x.1))});
  let correction=bck::<B>(correction);
  for j in (0..256).step_by(B::LANES) {
   let mut acc=acck::<B>();
   for rho in 0..R {let k=B::load(dagger[rho].as_ptr().add(j));mac32::<B>(&mut acc.0,w[rho].0,k);mac32::<B>(&mut acc.1,w[rho].1,k);}
   let v=subk::<B>(redk::<B>(acc),correction);
   B::store(o0.add(j),v.0);B::store(o1.add(j),v.1);
  }
 }
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f")]
unsafe fn combine_lane_w8<const R:usize>(rows:&[&[i32;256];R],weights:&[K;R],o0:*mut u64,o1:*mut u64) {
 // SAFETY: the caller guarantees AVX-512F and 256 writable words per plane.
 unsafe {combine_lane_body::<crate::wide::W8,R>(rows,weights,o0,o1)}
}
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn combine_lane_w4<const R:usize>(rows:&[&[i32;256];R],weights:&[K;R],o0:*mut u64,o1:*mut u64) {
 // SAFETY: the caller guarantees AVX2 and 256 writable words per plane.
 unsafe {combine_lane_body::<crate::wide::W4,R>(rows,weights,o0,o1)}
}

/// The rows `A[rho][s]`, `rho < R`, of column `s` of a matrix stored flat in `(rho, s, j)`
/// order with `l` columns.
fn lane_rows<const R:usize>(a:&[i32],l:usize,s:usize)->[&[i32;256];R] {
 std::array::from_fn(|rho|a[(rho*l+s)*256..(rho*l+s+1)*256].try_into().expect("256 coefficients"))
}

fn lane_planes_into<'a,const R:usize>(records:usize,l:usize,a:&(dyn Fn(usize)->&'a [i32]+Sync),eq_sig:&[K],gammas:&[K],contexts:&[Context],rows:usize,out:Out) {
 (0..rows/256).into_par_iter().for_each(|i| {
  let plane=|s:usize,c:usize|out.at(2*s*rows+c*rows+256*i);
  if i>=records {
   for s in 0..l {for c in 0..2 {
    // SAFETY: record i owns rows 256 i .. 256 (i + 1) of every plane of every lane.
    unsafe {std::ptr::write_bytes(plane(s,c),0,256);}
   }}
   return;
  }
  let matrix=a(i);
  let weights:[K;R]=std::array::from_fn(|rho|eq_sig[i]*gammas[rho]);
  let mut a0=[0u64;256];let mut a1=[0u64;256];
  for s in 0..l {
   combine_lane::<R>(&lane_rows::<R>(matrix,l,s),&weights,(&mut a0,&mut a1));
   // SAFETY: record i owns rows 256 i .. 256 (i + 1) of lane s; kernel_raw dispatches on the
   // backend and writes 256 words per plane.
   unsafe {kernel_raw((&a0,&a1),contexts,plane(s,0),plane(s,1));}
  }
 });
}

/// The lane ring weights `w_A^(s)(j, i) = eq_sig[i] K_{A~_{i,s}, coef}[j]` with
/// `A~_{i,s} = sum_rho gammas[rho] A_i[rho][s]`, for the `l` columns `s`, as lane-major
/// canonical planes: lane `s` holds words `2 s rows .. 2 (s + 1) rows`, its first coordinate
/// plane then its second, and record `i` sits at rows `256 i .. 256 (i + 1)`; the rows of
/// records `i >= records` are zero. `a(i)` is the matrix of record `i` flat in `(rho, s, j)`
/// order (`|A| < 2^23`), with `k = gammas.len()` rows of `l` columns. Each record is
/// combined, transformed and written once per column, as [`ring_weights`] does for the
/// four-row matrices of ML-DSA-44.
///
/// # Panics
/// For `k` other than 4, 6 or 8, a `coef` of other than 8 coordinates, or `rows` not a
/// multiple of 256 or below `256 records`.
pub fn lane_ring_weight_planes<'a>(records:usize,l:usize,a:&(dyn Fn(usize)->&'a [i32]+Sync),eq_sig:&[K],gammas:&[K],coef:&[K],rows:usize)->AlignedVec<u64> {
 assert_eq!(coef.len(),8);assert_eq!(rows%256,0);assert!(records<=rows/256);assert!(eq_sig.len()>=records);
 let contexts:Vec<_>=coef.iter().copied().map(Context::new).collect();
 let mut raw=AlignedVec::<u64>::uninit(rows.checked_mul(2*l).expect("ring table size"));
 let out=Out(raw.as_mut_ptr().cast());
 match gammas.len() {
  4=>lane_planes_into::<4>(records,l,a,eq_sig,gammas,&contexts,rows,out),
  6=>lane_planes_into::<6>(records,l,a,eq_sig,gammas,&contexts,rows,out),
  8=>lane_planes_into::<8>(records,l,a,eq_sig,gammas,&contexts,rows,out),
  k=>panic!("lane matrices of {k} rows"),
 }
 // SAFETY: every record task writes all 256 rows of both planes of every lane.
 unsafe {raw.assume_init()}
}

fn lane_rows_into<'a,const R:usize>(records:usize,l:usize,a:&(dyn Fn(usize)->&'a [i32]+Sync),eq_sig:&[K],gammas:&[K],contexts:&[Context],tables:&mut [Box<[std::mem::MaybeUninit<K>]>]) {
 let len=tables[0].len();
 let ptrs:Vec<_>=tables.iter_mut().map(|t|Out(t.as_mut_ptr().cast())).collect();
 (0..len/256).into_par_iter().for_each(|i| {
  // SAFETY: record i owns entries 256 i .. 256 (i + 1) of every table.
  let column=|s:usize|unsafe {std::slice::from_raw_parts_mut(ptrs[s].at(0).cast::<std::mem::MaybeUninit<K>>().add(256*i),256)};
  if i>=records {for s in 0..l {for x in column(s) {x.write(K::zero());}}return;}
  let matrix=a(i);
  let (weights,correction)=lane_scalar_weights::<R>(&std::array::from_fn(|rho|eq_sig[i]*gammas[rho]));
  let mut x0=[0u64;256];let mut x1=[0u64;256];
  for s in 0..l {
   combine_lane_scalar::<R>(&lane_rows::<R>(matrix,l,s),&weights,correction,(&mut x0,&mut x1));
   transform_scalar(&mut x0,&mut x1,contexts);
   let out=column(s);
   for t in 0..256 {
    let j=(t as u8).reverse_bits() as usize;
    out[j].write(K::new(F::from_u64(x0[t]),F::from_u64(x1[t])));
   }
  }
 });
}

/// [`lane_ring_weight_planes`] as one row-major `K` table per column (`rows` entries, record
/// `i` at `256 i + j`) for the tiers without the plane layout, in scalar arithmetic.
///
/// # Panics
/// As [`lane_ring_weight_planes`].
pub fn lane_ring_weight_rows<'a>(records:usize,l:usize,a:&(dyn Fn(usize)->&'a [i32]+Sync),eq_sig:&[K],gammas:&[K],coef:&[K],rows:usize)->Vec<Vec<K>> {
 assert_eq!(coef.len(),8);assert_eq!(rows%256,0);assert!(records<=rows/256);assert!(eq_sig.len()>=records);
 let contexts:Vec<_>=coef.iter().copied().map(Context::new).collect();
 let mut tables:Vec<_>=(0..l).map(|_|Box::<[K]>::new_uninit_slice(rows)).collect();
 match gammas.len() {
  4=>lane_rows_into::<4>(records,l,a,eq_sig,gammas,&contexts,&mut tables),
  6=>lane_rows_into::<6>(records,l,a,eq_sig,gammas,&contexts,&mut tables),
  8=>lane_rows_into::<8>(records,l,a,eq_sig,gammas,&contexts,&mut tables),
  k=>panic!("lane matrices of {k} rows"),
 }
 // SAFETY: every record task writes all 256 entries of every table.
 tables.into_iter().map(|t|unsafe {t.assume_init()}.into_vec()).collect()
}

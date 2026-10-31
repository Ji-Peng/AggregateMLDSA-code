use agg_simd::{dot, fold, fp, k, scalar, F, K, P};
use jolt_field::Ring;
fn random(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    (z ^ (z >> 31)) % P
}
#[test]
fn kernels_match_jolt_at_flush_boundaries() {
    let mut seed = 7757;
    // Limits count products per vector lane; include both requested lengths and
    // the eight-lane flush boundaries, with a tail and with two full flushes.
    for n in [
        0, 1, 7, 8, 31, 32, 63, 64, 65, 511, 512, 513, 682, 683, 1023, 1024, 1025, 2047, 2048,
        4095, 4096, 4097, 5456, 5464, 8184, 8191, 8192, 8193, 16385,
    ] {
        for worst in [false, true] {
            let mut make = || {
                (0..n)
                    .map(|_| if worst { P - 1 } else { random(&mut seed) })
                    .collect::<Vec<_>>()
            };
            let a = make();
            let b = make();
            let c = make();
            let d = make();
            let ap = (&a[..], &b[..]);
            let bp = (&c[..], &d[..]);
            let mut out = vec![0; n];
            let mut out1 = vec![0; n];
            let mut exp = vec![0; n];
            let mut exp1 = vec![0; n];
            fp::fp_mul(&a, &c, &mut out);
            scalar::fp_mul(&a, &c, &mut exp);
            assert_eq!(out, exp, "fp_mul n={n}");
            assert_eq!(
                dot::fp_dot(&a, &c),
                scalar::fp_dot(&a, &c),
                "fp_dot n={n} worst={worst}"
            );
            k::k_mul(ap, bp, (&mut out, &mut out1));
            scalar::k_mul(ap, bp, (&mut exp, &mut exp1));
            assert_eq!((&out, &out1), (&exp, &exp1), "k_mul n={n}");
            k::k_mul_base(ap, &c, (&mut out, &mut out1));
            scalar::k_mul_base(ap, &c, (&mut exp, &mut exp1));
            assert_eq!((&out, &out1), (&exp, &exp1));
            assert_eq!(
                dot::k_dot(ap, bp),
                scalar::k_dot(ap, bp),
                "k_dot n={n} worst={worst}"
            );
            assert_eq!(
                dot::k_base_dot(ap, &c),
                scalar::k_base_dot(ap, &c),
                "base_dot n={n}"
            );
            let r = K::new(
                F::from_u64(random(&mut seed)),
                F::from_u64(random(&mut seed)),
            );
            fold::k_fold(r, ap, bp, (&mut out, &mut out1));
            scalar::k_fold(r, ap, bp, (&mut exp, &mut exp1));
            assert_eq!((&out, &out1), (&exp, &exp1), "fold n={n}");
            let ints: Vec<_> = (0..n)
                .map(|i| match i % 5 {
                    0 => -(1i64 << 51) + 1,
                    1 => (1i64 << 51) - 1,
                    2 => -1,
                    3 => 0,
                    _ => 1,
                })
                .collect();
            assert_eq!(
                dot::k_int_dot(ap, &ints),
                scalar::k_int_dot(ap, &ints),
                "int_dot n={n}"
            );
            let lo: Vec<_> = a.iter().map(|x| x & ((1 << 52) - 1)).collect();
            assert_eq!(
                dot::k_int2_dot(ap, &lo, &ints),
                scalar::k_int2_dot(ap, &lo, &ints)
            );
            fold::int_fold(
                r,
                &ints,
                &ints.iter().map(|x| -x).collect::<Vec<_>>(),
                (&mut out, &mut out1),
            );
            scalar::int_fold(
                r,
                &ints,
                &ints.iter().map(|x| -x).collect::<Vec<_>>(),
                (&mut exp, &mut exp1),
            );
            assert_eq!((&out, &out1), (&exp, &exp1));
        }
    }
}
#[test]
fn soa_weight_round_matches_scalar_coefficients() {
 let mut seed=149;
 for n in [0,2,62,64,66,1024,2048,16368,16384,16400] {
  for worst in [false,true] {
   let a:Vec<_>=(0..n).map(|_|if worst {P-1} else {random(&mut seed)}).collect();
   let b:Vec<_>=(0..n).map(|_|if worst {P-1} else {random(&mut seed)}).collect();
   let g:Vec<_>=(0..n).map(|i|[i64::MIN,i64::MAX,-1,0,1][i%5]).collect();
   let mut expected=[K::from_u64(0);3];
   for i in (0..n).step_by(2) {let x=scalar::get((&a,&b),i);let y=scalar::get((&a,&b),i+1);let gx=K::from_i64(g[i]);let gy=K::from_i64(g[i+1]);expected[0]+=x*gx;expected[1]+=y*gy;expected[2]+=(y-x)*(gy-gx);}
   assert_eq!(dot::weight_round((&a,&b),&g),expected,"n={n}, worst={worst}");
  }
 }
}
#[test]
fn iterative_negacyclic_matches_recursive_order() {
 use agg_simd::negacyclic::{Context,kernel_into};
 use jolt_field::{Fold,Zero};
 fn recurse(a:&mut [K],r:&[K],scratch:&mut [K]) {
  if a.len()==1 {return;}
  let n=a.len();let h=n/2;let ctx=K::precompute(r[0]);let (even,odd)=scratch[..n].split_at_mut(h);
  for k in 0..h {let (x,y)=(a[2*k],a[2*k+1]);let t=if k==0 {-a[n-1]} else {a[2*k-1]};even[k]=K::fold_one(&ctx,x,t);odd[k]=K::fold_one(&ctx,y,x);}
  let (lo,hi)=a.split_at_mut(h);recurse(even,&r[1..],lo);recurse(odd,&r[1..],hi);
  for k in 0..h {a[2*k]=scratch[k];a[2*k+1]=scratch[h+k];}
 }
 let mut seed=150;
 for worst in [false,true] {
  let a:Vec<_>=(0..256).map(|_|K::new(F::from_u64(if worst {P-1} else {random(&mut seed)}),F::from_u64(if worst {P-1} else {random(&mut seed)}))).collect();
  let r:Vec<_>=(0..8).map(|_|K::new(F::from_u64(random(&mut seed)),F::from_u64(random(&mut seed)))).collect();
  let contexts:Vec<_>=r.iter().copied().map(Context::new).collect();
  let p0:Vec<_>=a.iter().map(|x|x.c0().to_canonical_u64()).collect();let p1:Vec<_>=a.iter().map(|x|x.c1().to_canonical_u64()).collect();
  let mut out0=vec![0;256];let mut out1=vec![0;256];kernel_into((&p0,&p1),&contexts,(&mut out0,&mut out1));
  let mut expected=a;recurse(&mut expected,&r,&mut vec![K::zero();256]);
  for i in 0..256 {assert_eq!(scalar::get((&out0,&out1),i),expected[i],"coefficient {i}");}
 }
}
#[test]
fn fraction_tree_planes_match_field_products_at_extremes() {
 use agg_simd::{buf::KPlanes,lookup::{Layer,bottoms,parent}};
 let mut seed=993;
 for n in [1,7,8,1024,4095,4096,4097] {
  for worst in [false,true] {
   let pv:Vec<_>=(0..2*n).map(|_|K::new(F::from_u64(if worst {P-1} else {random(&mut seed)}),F::from_u64(if worst {P-1} else {random(&mut seed)}))).collect();
   let qv:Vec<_>=(0..2*n).map(|_|K::new(F::from_u64(if worst {P-1} else {random(&mut seed)}),F::from_u64(if worst {P-1} else {random(&mut seed)}))).collect();
   let p=KPlanes::from_fn(2*n,|i|pv[i]);let q=KPlanes::from_fn(2*n,|i|qv[i]);
   let layer=Layer{p,q};let result=parent(&layer);
   for i in 0..n {assert_eq!(result.p.get(i),layer.p.get(i)*layer.q.get(n+i)+layer.p.get(n+i)*layer.q.get(i));assert_eq!(result.q.get(i),layer.q.get(i)*layer.q.get(n+i));}
   let leaves:Vec<_>=(0..4*n).map(|i|if worst {if i%2==0 {32768} else {-32768}} else {((i*313+17)%65537) as i32-32768}).collect();
   let powers:Vec<_>=(0..2).map(|_| {let x=K::new(F::from_u64(random(&mut seed)),F::from_u64(random(&mut seed)));let x2=x*x;[x,x2,x2*x,x2*x2]}).collect();
   let actual=bottoms(&leaves,0,&powers);
   for (l,powers) in actual.iter().zip(&powers) {let x=powers[0];for i in 0..n {
    let q:Vec<_>=(0..4).map(|j|x-K::from_i64(leaves[i+j*n] as i64)).collect();
    let expected_q=q[0]*q[1]*q[2]*q[3];let expected_p=(q[0]+q[2])*q[1]*q[3]+(q[1]+q[3])*q[0]*q[2];
    assert_eq!(l.p.get(i),expected_p,"p n={n}, i={i}");assert_eq!(l.q.get(i),expected_q,"q n={n}, i={i}");
   }}
  }
 }
}
#[test]
fn fused_fold_and_eval_match_field_arithmetic() {
 use agg_simd::{buf::KPlanes,fused::{eval_items,fold_rows,Terms}};
 let mut seed=4242;
 let rk=|s:&mut u64| K::new(F::from_u64(random(s)),F::from_u64(random(s)));
 for rows in [2,16,18,32,1030,2048,4096+18] {
  for worst in [false,true] {
   let width=5;
   let tables:Vec<Vec<K>>=(0..width).map(|_|(0..rows).map(|_|if worst {K::new(F::from_u64(P-1),F::from_u64(P-1))} else {rk(&mut seed)}).collect()).collect();
   let planes:Vec<KPlanes>=tables.iter().map(|t|KPlanes::from_fn(rows,|i|t[i])).collect();
   let lanes:Vec<(&[u64],&[u64])>=planes.iter().map(|p|p.planes()).collect();
   let items=rows/2;
   // Fold every lane, then compare with scalar folds.
   let r=rk(&mut seed);
   let mut out:Vec<KPlanes>=(0..width).map(|_|KPlanes::from_fn(items,|_|K::from_u64(0))).collect();
   let addresses:Vec<(usize,usize)>=out.iter_mut().map(|o| {let (a,b)=o.planes_mut();(a.as_mut_ptr() as usize,b.as_mut_ptr() as usize)}).collect();
   unsafe {fold_rows(r,&lanes,&addresses,0,items);}
   for l in 0..width {for k in 0..items {let a=tables[l][2*k];let b=tables[l][2*k+1];assert_eq!(out[l].get(k),a+r*(b-a),"fold lane {l} row {k} rows {rows}");}}
   // Evaluate with weights and without, full and not.
   let linear=[(0usize,0usize),(4,1)];let products=[(0usize,1usize,0usize),(2,3,1),(1,4,1),(3,3,0)];
   let terms=Terms{linear:&linear,products:&products,slots:2};
   let weights:Vec<K>=(0..items).map(|_|rk(&mut seed)).collect();
   let wp=KPlanes::from_fn(items,|i|weights[i]);
   for weighted in [false,true] {for full in [false,true] {
    let mut got=[[K::from_u64(0);3];2];
    eval_items(&lanes,terms,weighted.then(||wp.planes()),0,items,full,&mut got);
    let mut expected=[[K::from_u64(0);3];2];
    for k in 0..items {
     let w=if weighted {weights[k]} else {K::from_u64(1)};
     let a=|l:usize|tables[l][2*k];let b=|l:usize|tables[l][2*k+1];
     for &(l,s) in &linear {expected[s][0]+=w*a(l);if full {expected[s][1]+=w*b(l);}}
     for &(i,j,s) in &products {expected[s][0]+=w*a(i)*a(j);expected[s][2]+=w*(b(i)-a(i))*(b(j)-a(j));if full {expected[s][1]+=w*b(i)*b(j);}}
    }
    assert_eq!(got,expected,"eval rows {rows} weighted {weighted} full {full} worst {worst}");
   }}
  }
 }
}
#[test]
fn pattern_prefix_matches_direct_sums() {
 use agg_simd::{buf::KPlanes,pattern::{fill_rows,prefix_sums,Bottom,Leaves,Symmetric}};
 let mut seed=99;
 let rk=|s:&mut u64| K::new(F::from_u64(random(s)),F::from_u64(random(s)));
 for (fl,sl) in [(1usize,1usize),(2,2),(8,1),(8,3),(16,2),(1024,3),(4096,2)] {
  let rows=4*fl*sl;
  for worst in [false,true] {
   let first:Vec<K>=(0..fl).map(|_|if worst {K::new(F::from_u64(P-1),F::from_u64(P-1))} else {rk(&mut seed)}).collect();
   let second:Vec<K>=(0..sl).map(|_|rk(&mut seed)).collect();
   let w=|y:usize| {let hi=y/4;first[hi%fl]*second[hi/fl]};
   let extreme=|s:&mut u64,bound:i64| if worst {if random(s)%2==0 {bound as i32} else {-bound as i32}} else {(random(s)%(2*bound as u64+1)) as i64 as i32-bound as i32};
   // Bottom: |v| < 2^20.
   let v0:Vec<i32>=(0..rows).map(|_|extreme(&mut seed,(1<<20)-1)).collect();
   let v1:Vec<i32>=(0..rows).map(|_|extreme(&mut seed,(1<<20)-1)).collect();
   let bottom=Bottom{v0:&v0,v1:&v1};
   // Symmetric: a quarter column of 4 rows each, |v - centre| <= 2^15.
   let center=extreme(&mut seed,1<<14);
   let stacked:Vec<i32>=(0..4*rows).map(|_|center+extreme(&mut seed,1<<15)).collect();
   let sym=Symmetric{stacked:&stacked,center};
   fn check<L:Leaves>(leaves:&L,first:&[K],second:&[K],w:&dyn Fn(usize)->K,name:&str) {
    let sums=prefix_sums(leaves,first,second);
    let rows=leaves.rows();
    let mut total=K::from_u64(0);for y in (0..rows).step_by(4) {total+=w(y);}
    assert_eq!(sums.total,total,"{name} total");
    for (form,lanes) in L::LINEAR.iter().enumerate() {for a in 0..4 {
     let mut e=K::from_u64(0);for y in (a..rows).step_by(4) {let v=leaves.values(y);e+=w(y)*K::from_i64(lanes.iter().map(|&l|v[l]).sum::<i64>());}
     assert_eq!(sums.linear[form][a],e,"{name} linear form {form} pattern {a}");
    }}
    for (form,(pairs,_)) in L::PRODUCTS.iter().enumerate() {for a in 0..4 {for b in 0..4 {
     let mut e=K::from_u64(0);for hi in 0..rows/4 {let va=leaves.values(4*hi+a);let vb=leaves.values(4*hi+b);e+=w(4*hi)*K::from_i64(pairs.iter().map(|&(i,j)|va[i]*vb[j]).sum::<i64>());}
     assert_eq!(sums.products[form][4*a+b],e,"{name} product form {form} patterns {a} {b}");
    }}}
    // Materialisation by four pattern weights.
    let e:[K;4]=std::array::from_fn(|i|K::from_u64(7+3*i as u64)*K::new(F::from_u64(5),F::from_u64(11)));
    let mut out:Vec<KPlanes>=(0..L::LANES).map(|_|KPlanes::from_fn(rows/4,|_|K::from_u64(0))).collect();
    let addresses:Vec<(usize,usize)>=out.iter_mut().map(|o| {let (a,b)=o.planes_mut();(a.as_mut_ptr() as usize,b.as_mut_ptr() as usize)}).collect();
    unsafe {fill_rows(leaves,e,&addresses,0,rows/4);}
    for l in 0..L::LANES {for hi in 0..rows/4 {
     let expected=(0..4).fold(K::from_u64(0),|s,a|s+e[a]*K::from_i64(leaves.values(4*hi+a)[l]));
     assert_eq!(out[l].get(hi),expected,"{name} fill lane {l} row {hi}");
    }}
   }
   check(&bottom,&first,&second,&w,&format!("bottom {fl}x{sl} worst {worst}"));
   check(&sym,&first,&second,&w,&format!("sym {fl}x{sl} worst {worst}"));
  }
 }
}
#[test]
fn column_evaluation_and_mixed_folds_match_direct_sums() {
 use agg_simd::{buf::KPlanes,eval::{column_at,split_eq_tables,Mapped},fused::{fold_mixed,MixedLane},pattern::{fill_combined,ByteColumns,Columns}};
 let mut seed=31337;
 let rk=|s:&mut u64| K::new(F::from_u64(random(s)),F::from_u64(random(s)));
 for vars in [1usize,3,6,10,15] {
  let rows=1<<vars;
  let point:Vec<K>=(0..vars).map(|_|rk(&mut seed)).collect();
  let (first,second)=split_eq_tables(&point);
  // eq(point, y) directly.
  let eq=|y:usize| (0..vars).fold(K::from_u64(1),|w,k| w*if (y>>k)&1==1 {point[k]} else {K::from_u64(1)-point[k]});
  for y in [0,rows-1,rows/3] {assert_eq!(first[y%first.len()]*second[y/first.len()],eq(y),"tables vars {vars} y {y}");}
  let ints:Vec<i32>=(0..rows).map(|_|(random(&mut seed)%(1<<21)) as i32-(1<<20)).collect();
  let expected=(0..rows).fold(K::from_u64(0),|s,y|s+eq(y)*K::from_i64(ints[y] as i64));
  assert_eq!(column_at(&ints.as_slice(),&first,&second),expected,"column_at i32 vars {vars}");
  let bytes:Vec<i8>=(0..rows).map(|_|(random(&mut seed)%256) as u8 as i8).collect();
  let map:[i64;256]=std::array::from_fn(|i|(i as i64)*37-1000);
  let expected=(0..rows).fold(K::from_u64(0),|s,y|s+eq(y)*K::from_i64(map[bytes[y] as u8 as usize]));
  assert_eq!(column_at(&Mapped{cells:&bytes,map:&map},&first,&second),expected,"column_at i8 vars {vars}");
  // Mixed fold: one plane lane and one integer lane.
  if rows>=2 {
   let plane:Vec<K>=(0..rows).map(|_|rk(&mut seed)).collect();let planes=KPlanes::from_fn(rows,|i|plane[i]);
   let g=|i:usize| ints[i] as i64*3-7;
   let r=rk(&mut seed);
   let mut out:Vec<KPlanes>=(0..2).map(|_|KPlanes::from_fn(rows/2,|_|K::from_u64(0))).collect();
   let addresses:Vec<(usize,usize)>=out.iter_mut().map(|o| {let (a,b)=o.planes_mut();(a.as_mut_ptr() as usize,b.as_mut_ptr() as usize)}).collect();
   let lanes=[MixedLane::Planes(planes.planes()),MixedLane::Ints(&g)];
   unsafe {fold_mixed(r,&lanes,&addresses,0,rows/2);}
   for k in 0..rows/2 {
    assert_eq!(out[0].get(k),plane[2*k]+r*(plane[2*k+1]-plane[2*k]),"mixed plane row {k}");
    let (a,b)=(K::from_i64(g(2*k)),K::from_i64(g(2*k+1)));
    assert_eq!(out[1].get(k),a+r*(b-a),"mixed int row {k}");
   }
  }
  // Combined pattern fill of four columns.
  if rows>=4 {
   let cols:Vec<Vec<i32>>=(0..4).map(|_|(0..rows).map(|_|(random(&mut seed)%(1<<17)) as i32-(1<<16)).collect()).collect();
   let columns=Columns{lanes:vec![&cols[0],&cols[1],&cols[2],&cols[3]]};
   let weights:Vec<[K;4]>=(0..4).map(|_|std::array::from_fn(|_|rk(&mut seed))).collect();
   let mut out=KPlanes::from_fn(rows/4,|_|K::from_u64(0));
   let address={let (a,b)=out.planes_mut();(a.as_mut_ptr() as usize,b.as_mut_ptr() as usize)};
   unsafe {fill_combined(&columns,&weights,address,0,rows/4);}
   for row in 0..rows/4 {
    let mut e=K::from_u64(0);for a in 0..4 {for l in 0..4 {e+=weights[l][a]*K::from_i64(cols[l][4*row+a] as i64);}}
    assert_eq!(out.get(row),e,"combined row {row} vars {vars}");
   }
   // The same fill over three `i8` columns (one lane missing), against the direct sum.
   let bytes:Vec<Vec<i8>>=(0..3).map(|_|(0..rows).map(|_|(random(&mut seed)%256) as u8 as i8).collect()).collect();
   let columns=ByteColumns{lanes:vec![&bytes[0],&bytes[1],&bytes[2]]};
   let mut out=KPlanes::from_fn(rows/4,|_|K::from_u64(0));
   let address={let (a,b)=out.planes_mut();(a.as_mut_ptr() as usize,b.as_mut_ptr() as usize)};
   unsafe {fill_combined(&columns,&weights,address,0,rows/4);}
   for row in 0..rows/4 {
    let mut e=K::from_u64(0);for a in 0..4 {for l in 0..3 {e+=weights[l][a]*K::from_i64(bytes[l][4*row+a] as i64);}}
    assert_eq!(out.get(row),e,"byte combined row {row} vars {vars}");
   }
  }
 }
}
/// `eval_items` refuses lanes shorter than its items and terms naming a missing lane (the
/// vector path would otherwise read past the slices).
#[test]
fn eval_items_rejects_short_lanes_and_missing_lanes() {
 use agg_simd::{buf::KPlanes,fused::{eval_items,Terms}};
 let short=KPlanes::from_fn(16,|i|K::from_u64(i as u64));
 let long=KPlanes::from_fn(32,|i|K::from_u64(i as u64));
 let linear=[(0usize,0usize)];
 let terms=Terms{linear:&linear,products:&[],slots:1};
 let short_lanes=[short.planes()];
 let r=std::panic::catch_unwind(|| {let mut out=[[K::from_u64(0);3];1];eval_items(&short_lanes,terms,None,0,16,true,&mut out);});
 assert!(r.is_err(),"16 items over 16 rows");
 let missing=[(1usize,0usize)];
 let terms=Terms{linear:&missing,products:&[],slots:1};
 let long_lanes=[long.planes()];
 let r=std::panic::catch_unwind(|| {let mut out=[[K::from_u64(0);3];1];eval_items(&long_lanes,terms,None,0,16,true,&mut out);});
 assert!(r.is_err(),"term names lane 1 of one");
 let terms=Terms{linear:&linear,products:&[],slots:1};
 let mut out=[[K::from_u64(0);3];1];
 eval_items(&long_lanes,terms,None,0,16,true,&mut out);
 assert_eq!(out[0][0],(0..16).map(|k|K::from_u64(2*k)).fold(K::from_u64(0),|a,b|a+b));
}
/// The vector evaluation groups terms by slot, splits each lane once per step, uses the
/// squaring form when the two lanes coincide and weights with radix-2^51 Karatsuba chains.
/// Every item count, slot count and weight mode must give exactly the scalar sums.
#[test]
fn eval_kernel_matches_scalar_reference() {
    use agg_simd::{buf::KPlanes, fused::{eval_items, Terms, MAX_SLOTS}};
    let mut seed = 20261031;
    let worst = |s: &mut u64| {
        if random(s) % 2 == 0 { K::new(F::from_u64(P - 1), F::from_u64(P - 1)) }
        else { K::new(F::from_u64(random(s)), F::from_u64(random(s))) }
    };
    let width = 12_usize;
    for &items in &[1_usize, 2, 4, 8, 9, 64, 4097] {
        let rows = 2 * items;
        for &slots in &[1_usize, 2, MAX_SLOTS] {
            let mut tables: Vec<Vec<K>> = Vec::new();
            for _ in 0..width {
                let mut lane = Vec::with_capacity(rows);
                for _ in 0..rows {
                    lane.push(worst(&mut seed));
                }
                tables.push(lane);
            }
            let planes: Vec<KPlanes> = tables.iter().map(|t| KPlanes::from_fn(rows, |i| t[i])).collect();
            let lanes: Vec<(&[u64], &[u64])> = planes.iter().map(|p| p.planes()).collect();
            // One linear term and one product per slot, some products on equal lanes and
            // some extra products sharing a slot.
            let mut linear = Vec::new();
            let mut products = Vec::new();
            for s in 0..slots {
                linear.push(((3 * s + 1) % width, s));
                let i = (5 * s + 2) % width;
                let j = if s % 3 == 0 { i } else { (7 * s + 4) % width };
                products.push((i, j, s));
                if s % 4 == 1 {
                    products.push((i, (i + 1) % width, s));
                }
            }
            let terms = Terms { linear: &linear, products: &products, slots };
            let mut weights = Vec::with_capacity(items);
            for _ in 0..items {
                weights.push(worst(&mut seed));
            }
            let wp = KPlanes::from_fn(items, |i| weights[i]);
            for weighted in [false, true] {
                for full in [false, true] {
                    let mut got = vec![[K::from_u64(0); 3]; slots];
                    eval_items(&lanes, terms, weighted.then(|| wp.planes()), 0, items, full, &mut got);
                    let mut expected = vec![[K::from_u64(0); 3]; slots];
                    for k in 0..items {
                        let w = if weighted { weights[k] } else { K::from_u64(1) };
                        let a = |l: usize| tables[l][2 * k];
                        let b = |l: usize| tables[l][2 * k + 1];
                        for &(l, s) in &linear {
                            expected[s][0] += w * a(l);
                            if full {
                                expected[s][1] += w * b(l);
                            }
                        }
                        for &(i, j, s) in &products {
                            expected[s][0] += w * a(i) * a(j);
                            expected[s][2] += w * (b(i) - a(i)) * (b(j) - a(j));
                            if full {
                                expected[s][1] += w * b(i) * b(j);
                            }
                        }
                    }
                    assert_eq!(got, expected,
                        "items={items} slots={slots} weighted={weighted} full={full}");
                }
            }
        }
    }
}
/// The split equality weights `first[k mod |first|] second[k / |first|]` are applied per
/// block when the short table has fewer than 64 entries and per table entry otherwise.
/// Both forms, including the single-entry table, must give the direct weighted sums.
#[test]
fn fused_pass_short_first_tables_match_direct_sums() {
    use agg_simd::{buf::KPlanes, fused::{fused_pass, Eval, Fill, Terms, MAX_SLOTS}, pattern::Columns};
    let mut seed = 20261032;
    let rk = |s: &mut u64| K::new(F::from_u64(random(s)), F::from_u64(random(s)));
    // `8192` items are `16384` rows, so the pass runs in several tasks and the weight plane
    // is built on blocks whose first item is not item zero.
    for &items in &[128_usize, 2048, 8192] {
        let rows = 2 * items;
        let width = 6_usize;
        let mut planes: Vec<KPlanes> = Vec::with_capacity(width);
        for _ in 0..width {
            let mut lane = Vec::with_capacity(rows);
            for _ in 0..rows {
                lane.push(rk(&mut seed));
            }
            planes.push(KPlanes::from_fn(rows, |i| lane[i]));
        }
        let lanes: Vec<(&[u64], &[u64])> = planes.iter().map(|p| p.planes()).collect();
        for &slots in &[1_usize, MAX_SLOTS] {
            let mut linear = Vec::new();
            let mut products = Vec::new();
            for s in 0..slots {
                linear.push(((s + 1) % width, s));
                let i = (2 * s + 1) % width;
                products.push((i, if s % 2 == 0 { i } else { (i + 2) % width }, s));
            }
            let terms = Terms { linear: &linear, products: &products, slots };
            for &first_len in &[1_usize, 2, 4, 8, 16, 32, 64] {
                let mut first: Vec<K> = (0..first_len).map(|_| rk(&mut seed)).collect();
                if first_len == 1 {
                    first[0] = K::from_u64(1);
                }
                let second: Vec<K> = (0..items / first_len).map(|_| rk(&mut seed)).collect();
                let first_planes = KPlanes::from_fn(first_len, |i| first[i]);
                let eval = Eval {
                    terms,
                    first: first_planes.planes(),
                    second: &second,
                    full: true,
                };
                let got = fused_pass(None, &lanes, None::<Fill<'_, Columns>>, Some(eval));
                let mut expected = vec![[K::from_u64(0); 3]; slots];
                for k in 0..items {
                    let w = first[k % first_len] * second[k / first_len];
                    let a = |l: usize| planes[l].get(2 * k);
                    let b = |l: usize| planes[l].get(2 * k + 1);
                    for &(l, s) in &linear {
                        expected[s][0] += w * a(l);
                        expected[s][1] += w * b(l);
                    }
                    for &(i, j, s) in &products {
                        expected[s][0] += w * a(i) * a(j);
                        expected[s][2] += w * (b(i) - a(i)) * (b(j) - a(j));
                        expected[s][1] += w * b(i) * b(j);
                    }
                }
                assert_eq!(got, expected, "items={items} slots={slots} first_len={first_len}");
            }
        }
    }
}

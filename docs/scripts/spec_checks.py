import math
q=8380417; g2=(q-1)//88; g1=2**17; tau=39; beta=78; d=13
# (1) bit identity: v<44 iff b5*(b4+b3*b2)==0 over [0,64)
ok=all(((v<44)==((v>>5&1)*(((v>>4)&1)+((v>>3)&1)*((v>>2)&1))==0)) for v in range(64))
print("bit identity", ok)
# (2) UseHint residual bound: max |ctr_q(r - 2*g2*UseHint(h,r))| over r in Z_q, h in {0,1}
import numpy as np
r=np.arange(q,dtype=np.int64)
rp=r%q
r0=rp%(2*g2); r0=np.where(r0>g2, r0-2*g2, r0)
r1=np.where(rp-r0==q-1, 0, (rp-r0)//(2*g2))
r0=np.where(rp-r0==q-1, r0-1, r0)
m=44
w_h0=r1
w_h1=np.where(r0>0,(r1+1)%m,(r1-1)%m)
def ctr(x):
    x=x%q; return np.where(x>(q-1)//2, x-q, x)
b0=np.abs(ctr(r-2*g2*w_h0)).max(); b1=np.abs(ctr(r-2*g2*w_h1)).max()
print("usehint residual max h0,h1:",b0,b1, "2g2+1=",2*g2+1)
# (3) quotient bound
Az=1024*((q-1)//2)*(g1-beta-1); T=2**d*tau*1023; W=2*g2*43; U=2*g2+1
Dmax=Az+T+W+U; print("Delta max",Dmax, "log2",math.log2(Dmax)); print("v max",Dmax//q, "2^26=",2**26, "<2^26:",Dmax//q<2**26)
# relaxed: Bz=2^18-? , Bu
for Bz,Bu in [(2**17,2**18),(2**18,2**19),(2**19,2**20)]:
    Dm=1024*((q-1)//2)*Bz+T+W+Bu
    print("ext Bz,Bu",Bz,Bu,"log2 Delta",round(math.log2(Dm),2),"zeta'",max(Bz,Bu+2**(d-1)*tau))
print("zeta std",max(g1-beta,2*g2+1+2**(d-1)*tau))
# (4) bridge error
def lc(m,k): return math.log2(math.comb(m,k))
e=[lc(8192,14),lc(8192,14),lc(4096,18)]
tot=sum(2**-x for x in e); print("bridge levels",[round(x,1) for x in e],"total log2",round(math.log2(tot),2))
# (5) projection wrap: composite coefficient variance, y std for Delta rms 2^42.5 at 2^25 entries
var=961*52**3; print("composite var log2",round(math.log2(var),2),"y std log2 (rms 2^42.5, 2^25)",round(0.5*math.log2(var)+12.5+42.5,1),"p/2 log2",63)
# honest Delta rms: A uniform centered (var q^2/12), z uniform in +-2^17 (var 2^34/3), 1024 terms
print("honest Delta rms log2",round(0.5*math.log2(1024*(q**2/12)*(2**34/3)),1))
# (6) sponge blocks: 64+768=832 bytes, rate 136
print("blocks",math.ceil((832+1)/136), "block1 w1 bytes",136-64,"blocks2-6",5*136,"block7 bytes",832-64-72-680)
# (7) bridge geometry at N*=2^15: lane bits, blocks
for ds in [3,8,10,13,15]:
    L=6144*2**ds; lane=1024*2**ds; b1=max(1,lane//2**19); o1=b1*8192
    print("d_sig",ds,"lane bits",lane,"L1 blocks/lane/seg",min(b1,16),"segments",max(1,2**ds//8192),"L1 out/lane",o1)
# (8) digit decompositions of Section 4.3: bases least significant first, balanced alphabets
import itertools
def layout(bases, signed):
    Wt=[1]
    for b in bases[:-1]: Wt.append(Wt[-1]*b)
    B=1
    for b in bases: B*=b
    C=sum(w*b//2 for w,b in zip(Wt,bases)) - (B//2 if signed else 0)
    return Wt,B,C
def check(name,bases,signed,lo,hi,exhaustive=True):
    Wt,B,C=layout(bases,signed)
    if exhaustive:
        vals=sorted(sum(w*z for w,z in zip(Wt,zs))+C for zs in itertools.product(*[range(-b//2,b//2) for b in bases]))
        ok = vals==list(range(lo,hi+1))
    else:
        mn=sum(w*(-b//2) for w,b in zip(Wt,bases))+C; mx=sum(w*(b//2-1) for w,b in zip(Wt,bases))+C
        ok = (mn,mx)==(lo,hi) and B==hi-lo+1
    print("digits",name,bases,"C=",C,"range",lo,hi,ok)
check("z",(16,16,16,16,4),True,-2**17,2**17-1)
check("u",(16,16,16,16,8),True,-2**18,2**18-1)
check("v",(16,)*6+(8,),True,-2**26,2**26-1,exhaustive=False)
check("kappa1/2",(16,16,4),False,0,1023)
check("kappa3",(16,16),False,0,255)
check("b_z H-exact",(16,16,16,16,2),False,0,2**17-1)
check("b_u H-exact",(16,16,16,16,4),False,0,2**18-1)
# (9) leaf predicates Q_b(P), P=Z(Z+1), vanish exactly on the balanced alphabet
def Q(b,P):
    r=1
    for t in {4:[0,2],6:[0,2,6],8:[0,2,6,12],16:[0,2,6,12,20,30,42,56]}[b]: r*=(P-t)
    return r
for b in (4,6,8,16):
    print("Q",b,"exact on balanced alphabet",all((Q(b,Z*(Z+1))==0)==(-b//2<=Z<b//2) for Z in range(-4*b,4*b)))
# (10) lifting bound (6.1) with the digit ranges, and the v ceiling
Az=1024*((q-1)//2)*2**17; D0=Az+T+W+2**18; D=D0+q*2**26
print("Delta lift log2",round(math.log2(D),2),"<63:",D<2**63,"v ceiling log2",round(math.log2((2**63-D0)//q),2))
# (11) committed elements per signature at N=2^15 (main 23 columns; D1 18; D2 24; D3 2)
main=23*1024; d1=18*2**19/2**15; d2=24*2**15/2**15; d3=2*2**14/2**15
print("elements/sig",main+d1+d2+d3,"main",main,"bridge",d1+d2+d3)
# (12) relation H-u (default since wire version 3, docs/h-exact.md 2.1 and step A):
# u = B_ulo + 63,079 H_u - B_u, B_ulo in [0,2^16), H_u in [0,6); stored U_lo = B_ulo - 2^15,
# U_hi = H_u - 3, so u = U_lo + 63,079 U_hi + 31,540. Exact reachable set, constant, padding,
# boundaries, the unique single-digit base c <= 16, and zeta' of (3.5).
Bu=2*g2+1; A=63079
reach=set()
for H in range(6):
    reach.update(range(A*H-Bu, A*H+2**16-Bu))
print("H-u u blocks cover exactly [-B_u,B_u]:", reach==set(range(-Bu,Bu+1)), "5A+2^16=2B_u+1:", 5*A+2**16==2*Bu+1)
Cu=2**15+3*A-Bu
def enc_u(u):
    H=min((u+Bu)//A,5); H=max(H,0); return (u+Bu-A*H-2**15, H-3)
ok=all(lo+A*hi+Cu==u and -2**15<=lo<2**15 and -3<=hi<=2 for u in range(-Bu,Bu+1) for (lo,hi) in [enc_u(u)])
print("C_u",Cu,"(31,540)","canonical round trip",ok,"padding (U_lo,U_hi)",enc_u(0),"edges",enc_u(Bu),enc_u(-Bu))
print("+-190,466 unreachable:", (Bu+1) not in reach and -(Bu+1) not in reach)
sols=[c for c in range(2,17) if (2*Bu+1-2**16)%(c-1)==0 and (2*Bu+1-2**16)//(c-1)<=2**16]
print("single-digit bases c<=16:",sols)
print("zeta' H-u",max(2**17,Bu+2**(d-1)*tau),"zeta",max(g1-beta,2*g2+1+2**(d-1)*tau))

//! pq-crystals Dilithium AVX2 assembly port (CC0/Apache-2.0).
//! Kept for kernels whose LLVM register allocation misses the reference gate.
//! Forward butterflies keep |x| below 4.52q or 4.25P2; q inverse sums
//! stay below 256q, and its final values lie in (-0.74q, 0.74q) before the
//! canonical or centered store. Four dot products accumulate in i64 below 2^58.
//! Twiddle shuffles that the reference performs per call are read pre-shuffled
//! from the table's second field.
//!
//! The code is SysV x86-64 ELF assembly (argument registers rdi, rsi, rdx, rcx,
//! r8; all vector registers caller-saved), so it is built only on x86-64 Linux
//! and declared with the `sysv64` ABI.
use super::{tables, Poly32};
core::arch::global_asm!(
    r#"
.macro agg_avx2_shuffle8 r0,r1,r2,r3
vperm2i128	$0x20,%ymm\r1,%ymm\r0,%ymm\r2
vperm2i128	$0x31,%ymm\r1,%ymm\r0,%ymm\r3
.endm

.macro agg_avx2_shuffle4 r0,r1,r2,r3
vpunpcklqdq	%ymm\r1,%ymm\r0,%ymm\r2
vpunpckhqdq	%ymm\r1,%ymm\r0,%ymm\r3
.endm

.macro agg_avx2_shuffle2 r0,r1,r2,r3
#vpsllq		$32,%ymm\r1,%ymm\r2
vmovsldup	%ymm\r1,%ymm\r2
vpblendd	$0xAA,%ymm\r2,%ymm\r0,%ymm\r2
vpsrlq		$32,%ymm\r0,%ymm\r0
#vmovshdup	%ymm\r0,%ymm\r0
vpblendd	$0xAA,%ymm\r1,%ymm\r0,%ymm\r3
.endm

.macro agg_avx2_shuffle1 r0,r1,r2,r3
vpslld		$16,%ymm\r1,%ymm\r2
vpblendw	$0xAA,%ymm\r2,%ymm\r0,%ymm\r2
vpsrld		$16,%ymm\r0,%ymm\r0
vpblendw	$0xAA,%ymm\r1,%ymm\r0,%ymm\r3
.endm

.set agg_avx2__8XQ,0
.set agg_avx2__8XQINV,8
.set agg_avx2__8XDIV_QINV,16
.set agg_avx2__8XDIV,24
.set agg_avx2__ZETAS_QINV,32
.set agg_avx2__ZETAS,328
.set agg_avx2__FWD,624
.set agg_avx2__INV,1072

.macro agg_avx2_fwdz c,ql,zl
vmovdqa		(agg_avx2__FWD+16*(\c))*4(%rsi),%ymm\ql
vmovdqa		(agg_avx2__FWD+16*(\c)+8)*4(%rsi),%ymm\zl
.endm

.macro agg_avx2_invz n,aq,bq,az,bz
vmovdqa		(agg_avx2__INV+32*(\n))*4(%rsi),%ymm\aq
vmovdqa		(agg_avx2__INV+32*(\n)+8)*4(%rsi),%ymm\bq
vmovdqa		(agg_avx2__INV+32*(\n)+16)*4(%rsi),%ymm\az
vmovdqa		(agg_avx2__INV+32*(\n)+24)*4(%rsi),%ymm\bz
.endm

/* Forward butterfly: r = (h z - t p) / 2^32 with t = lo32(h z pinv), then
   (l + r, l - r). The two 64-bit products share their low words, so a
   32-bit subtraction leaves r in the high word; subtracting before the
   lane merge needs one merge instead of the reference's two, with the
   same 32-bit results. |r| <= |h z|/2^32 + p/2. */
.macro agg_avx2_butterfly l,h,zl0=1,zl1=1,zh0=2,zh1=2
vpmuldq		%ymm\zl0,%ymm\h,%ymm13
vmovshdup	%ymm\h,%ymm12
vpmuldq		%ymm\zl1,%ymm12,%ymm14

vpmuldq		%ymm\zh0,%ymm\h,%ymm\h
vpmuldq		%ymm\zh1,%ymm12,%ymm12

vpmuldq		%ymm0,%ymm13,%ymm13
vpmuldq		%ymm0,%ymm14,%ymm14

vpsubd		%ymm13,%ymm\h,%ymm\h
vpsubd		%ymm14,%ymm12,%ymm12

vmovshdup	%ymm\h,%ymm\h
vpblendd	$0xAA,%ymm12,%ymm\h,%ymm12

vpsubd		%ymm12,%ymm\l,%ymm\h
vpaddd		%ymm12,%ymm\l,%ymm\l
.endm

.macro agg_avx2_levels0t1 off
/* level 0 */
vpbroadcastd	(agg_avx2__ZETAS_QINV+1)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+1)*4(%rsi),%ymm2

vmovdqa		  0+32*\off(%rdi),%ymm4
vmovdqa		128+32*\off(%rdi),%ymm5
vmovdqa		256+32*\off(%rdi),%ymm6
vmovdqa	 	384+32*\off(%rdi),%ymm7
vmovdqa		512+32*\off(%rdi),%ymm8
vmovdqa		640+32*\off(%rdi),%ymm9
vmovdqa		768+32*\off(%rdi),%ymm10
vmovdqa	 	896+32*\off(%rdi),%ymm11

agg_avx2_butterfly	4,8
agg_avx2_butterfly	5,9
agg_avx2_butterfly	6,10
agg_avx2_butterfly	7,11

/* level 1 */
vpbroadcastd	(agg_avx2__ZETAS_QINV+2)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+2)*4(%rsi),%ymm2
agg_avx2_butterfly	4,6
agg_avx2_butterfly	5,7

vpbroadcastd	(agg_avx2__ZETAS_QINV+3)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+3)*4(%rsi),%ymm2
agg_avx2_butterfly	8,10
agg_avx2_butterfly	9,11

vmovdqa		%ymm4,  0+32*\off(%rdi)
vmovdqa		%ymm5,128+32*\off(%rdi)
vmovdqa		%ymm6,256+32*\off(%rdi)
vmovdqa		%ymm7,384+32*\off(%rdi)
vmovdqa		%ymm8,512+32*\off(%rdi)
vmovdqa		%ymm9,640+32*\off(%rdi)
vmovdqa		%ymm10,768+32*\off(%rdi)
vmovdqa		%ymm11,896+32*\off(%rdi)
.endm

.macro agg_avx2_levels2t7 off
/* level 2 */
vmovdqa		256*\off+  0(%rdi),%ymm4
vmovdqa		256*\off+ 32(%rdi),%ymm5
vmovdqa		256*\off+ 64(%rdi),%ymm6
vmovdqa	 	256*\off+ 96(%rdi),%ymm7
vmovdqa		256*\off+128(%rdi),%ymm8
vmovdqa		256*\off+160(%rdi),%ymm9
vmovdqa		256*\off+192(%rdi),%ymm10
vmovdqa	 	256*\off+224(%rdi),%ymm11

vpbroadcastd	(agg_avx2__ZETAS_QINV+4+\off)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+4+\off)*4(%rsi),%ymm2

agg_avx2_butterfly	4,8
agg_avx2_butterfly	5,9
agg_avx2_butterfly	6,10
agg_avx2_butterfly	7,11

agg_avx2_shuffle8	4,8,3,8
agg_avx2_shuffle8	5,9,4,9
agg_avx2_shuffle8	6,10,5,10
agg_avx2_shuffle8	7,11,6,11

/* level 3 */
vmovdqa		(agg_avx2__ZETAS_QINV+8+8*\off)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+8+8*\off)*4(%rsi),%ymm2

agg_avx2_butterfly	3,5
agg_avx2_butterfly	8,10
agg_avx2_butterfly	4,6
agg_avx2_butterfly	9,11

agg_avx2_shuffle4	3,5,7,5
agg_avx2_shuffle4	8,10,3,10
agg_avx2_shuffle4	4,6,8,6
agg_avx2_shuffle4	9,11,4,11

/* level 4 */
vmovdqa		(agg_avx2__ZETAS_QINV+40+8*\off)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+40+8*\off)*4(%rsi),%ymm2

agg_avx2_butterfly	7,8
agg_avx2_butterfly	5,6
agg_avx2_butterfly	3,4
agg_avx2_butterfly	10,11

agg_avx2_shuffle2	7,8,9,8
agg_avx2_shuffle2	5,6,7,6
agg_avx2_shuffle2	3,4,5,4
agg_avx2_shuffle2	10,11,3,11

/* level 5 */
vmovdqa		(agg_avx2__ZETAS_QINV+72+8*\off)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+72+8*\off)*4(%rsi),%ymm2
agg_avx2_fwdz	\off,10,15

agg_avx2_butterfly	9,5,1,10,2,15
agg_avx2_butterfly	8,4,1,10,2,15
agg_avx2_butterfly	7,3,1,10,2,15
agg_avx2_butterfly	6,11,1,10,2,15

/* level 6 */
vmovdqa		(agg_avx2__ZETAS_QINV+104+8*\off)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+104+8*\off)*4(%rsi),%ymm2
agg_avx2_fwdz	\off+4,10,15
agg_avx2_butterfly	9,7,1,10,2,15
agg_avx2_butterfly	8,6,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+104+8*\off+32)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+104+8*\off+32)*4(%rsi),%ymm2
agg_avx2_fwdz	\off+8,10,15
agg_avx2_butterfly	5,3,1,10,2,15
agg_avx2_butterfly	4,11,1,10,2,15

/* level 7 */
vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off)*4(%rsi),%ymm2
agg_avx2_fwdz	\off+12,10,15
agg_avx2_butterfly	9,8,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off+32)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off+32)*4(%rsi),%ymm2
agg_avx2_fwdz	\off+16,10,15
agg_avx2_butterfly	7,6,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off+64)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off+64)*4(%rsi),%ymm2
agg_avx2_fwdz	\off+20,10,15
agg_avx2_butterfly	5,4,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off+96)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off+96)*4(%rsi),%ymm2
agg_avx2_fwdz	\off+24,10,15
agg_avx2_butterfly	3,11,1,10,2,15

vmovdqa		%ymm9,256*\off+  0(%rdi)
vmovdqa		%ymm8,256*\off+ 32(%rdi)
vmovdqa		%ymm7,256*\off+ 64(%rdi)
vmovdqa		%ymm6,256*\off+ 96(%rdi)
vmovdqa		%ymm5,256*\off+128(%rdi)
vmovdqa		%ymm4,256*\off+160(%rdi)
vmovdqa		%ymm3,256*\off+192(%rdi)
vmovdqa		%ymm11,256*\off+224(%rdi)
.endm

.text
.global agg_mldsa_ntt_avx2
.p2align 5
agg_mldsa_ntt_avx2:
vmovdqa		agg_avx2__8XQ*4(%rsi),%ymm0

agg_avx2_levels0t1	0
agg_avx2_levels0t1	1
agg_avx2_levels0t1	2
agg_avx2_levels0t1	3

agg_avx2_levels2t7	0
agg_avx2_levels2t7	1
agg_avx2_levels2t7	2
agg_avx2_levels2t7	3

vzeroupper
ret

.purgem agg_avx2_butterfly
/* Four products of 16 coefficients. a's odd words are shifted down; b's odd
   words are read from b + 4 bytes, since vpmuldq only reads the even word of
   each qword. On the final b block that read would pass the end of b, so
   `last` takes b's odd words with vmovshdup instead. (Reading a's odd words
   at a + 4 as well was slower: the extra cache-line-split loads cost more
   than the shifts they save.) */
.macro agg_avx2_pointwise off,d0,d1,d2,d3,last=0
vmovdqa		\off(%rsi),%ymm\d0
vmovdqa		\off+32(%rsi),%ymm\d2
vpsrlq		$32,%ymm\d0,%ymm\d1
vpsrlq		$32,%ymm\d2,%ymm\d3
vpmuldq		\off(%rdx),%ymm\d0,%ymm\d0
vpmuldq		\off+32(%rdx),%ymm\d2,%ymm\d2
vpmuldq		\off+4(%rdx),%ymm\d1,%ymm\d1
.if \last
vmovshdup	\off+32(%rdx),%ymm13
vpmuldq		%ymm13,%ymm\d3,%ymm\d3
.else
vpmuldq		\off+36(%rdx),%ymm\d3,%ymm\d3
.endif
.endm

.macro agg_avx2_acc
vpaddq		%ymm6,%ymm2,%ymm2
vpaddq		%ymm7,%ymm3,%ymm3
vpaddq		%ymm8,%ymm4,%ymm4
vpaddq		%ymm9,%ymm5,%ymm5
.endm

/* q: signed Montgomery as pq-crystals, |out| < 0.54q for canonical a.
   P2: |sum| < 2^31 P2 (centered a, |b| < 6.4 P2). With the unsigned quotient
   t in [0,2^32), out = (sum - t P2) / 2^32 lies in (sum/2^32 - P2, sum/2^32],
   a subset of [-(P2-1)/2 - P2, (P2-1)/2], so adding P2 below -(P2-1)/2 centers it. */
.macro agg_avx2_dotbody center,last=0
agg_avx2_pointwise	0,2,3,4,5
agg_avx2_pointwise	1024,6,7,8,9
agg_avx2_acc
agg_avx2_pointwise	2048,6,7,8,9
agg_avx2_acc
agg_avx2_pointwise	3072,6,7,8,9,\last
agg_avx2_acc

#reduce
vpmuldq		%ymm0,%ymm2,%ymm6
vpmuldq		%ymm0,%ymm3,%ymm7
vpmuldq		%ymm0,%ymm4,%ymm8
vpmuldq		%ymm0,%ymm5,%ymm9
.if \center
vpmuludq	%ymm1,%ymm6,%ymm6
vpmuludq	%ymm1,%ymm7,%ymm7
vpmuludq	%ymm1,%ymm8,%ymm8
vpmuludq	%ymm1,%ymm9,%ymm9
.else
vpmuldq		%ymm1,%ymm6,%ymm6
vpmuldq		%ymm1,%ymm7,%ymm7
vpmuldq		%ymm1,%ymm8,%ymm8
vpmuldq		%ymm1,%ymm9,%ymm9
.endif
vpsubq		%ymm6,%ymm2,%ymm2
vpsubq		%ymm7,%ymm3,%ymm3
vpsubq		%ymm8,%ymm4,%ymm4
vpsubq		%ymm9,%ymm5,%ymm5
vpsrlq		$32,%ymm2,%ymm2
vmovshdup	%ymm4,%ymm4

#store
vpblendd	$0xAA,%ymm3,%ymm2,%ymm2
vpblendd	$0xAA,%ymm5,%ymm4,%ymm4
.if \center
vpcmpgtd	%ymm2,%ymm14,%ymm6
vpcmpgtd	%ymm4,%ymm14,%ymm7
vpand		%ymm1,%ymm6,%ymm6
vpand		%ymm1,%ymm7,%ymm7
vpaddd		%ymm6,%ymm2,%ymm2
vpaddd		%ymm7,%ymm4,%ymm4
.endif
vmovdqa		%ymm2,(%rdi)
vmovdqa		%ymm4,32(%rdi)

add		$64,%rsi
add		$64,%rdx
add		$64,%rdi
.endm

.global agg_mldsa_dot_avx2
.p2align 5
agg_mldsa_dot_avx2:
vmovdqa		agg_avx2__8XQINV*4(%rcx),%ymm0
vmovdqa		agg_avx2__8XQ*4(%rcx),%ymm1
mov		$5,%eax
test		%r8d,%r8d
jnz		990f
.p2align 5
989:
.rept 3
agg_avx2_dotbody 0
.endr
sub		$1,%eax
jnz		989b
agg_avx2_dotbody 0,1
vzeroupper
ret
990:
/* ymm14 = -(P2-1)/2 */
vpsrld		$1,%ymm1,%ymm14
vpxor		%ymm15,%ymm15,%ymm15
vpsubd		%ymm14,%ymm15,%ymm14
.p2align 5
988:
.rept 3
agg_avx2_dotbody 1
.endr
sub		$1,%eax
jnz		988b
agg_avx2_dotbody 1,1
vzeroupper
ret

.text
.global agg_mldsa_order_avx2
agg_mldsa_order_avx2:
#load
vmovdqa		(%rdi),%ymm4
vmovdqa		32(%rdi),%ymm5
vmovdqa		64(%rdi),%ymm6
vmovdqa		96(%rdi),%ymm7
vmovdqa		128(%rdi),%ymm8
vmovdqa		160(%rdi),%ymm9
vmovdqa		192(%rdi),%ymm10
vmovdqa		224(%rdi),%ymm11

agg_avx2_shuffle8	4,8,3,8
agg_avx2_shuffle8	5,9,4,9
agg_avx2_shuffle8	6,10,5,10
agg_avx2_shuffle8	7,11,6,11

agg_avx2_shuffle4	3,5,7,5
agg_avx2_shuffle4	8,10,3,10
agg_avx2_shuffle4	4,6,8,6
agg_avx2_shuffle4	9,11,4,11

agg_avx2_shuffle2	7,8,9,8
agg_avx2_shuffle2	5,6,7,6
agg_avx2_shuffle2	3,4,5,4
agg_avx2_shuffle2	10,11,3,11

#store
vmovdqa		%ymm9,(%rdi)
vmovdqa		%ymm8,32(%rdi)
vmovdqa		%ymm7,64(%rdi)
vmovdqa		%ymm6,96(%rdi)
vmovdqa		%ymm5,128(%rdi)
vmovdqa		%ymm4,160(%rdi)
vmovdqa		%ymm3,192(%rdi)
vmovdqa		%ymm11,224(%rdi)

add $256,%rdi
#load
vmovdqa		(%rdi),%ymm4
vmovdqa		32(%rdi),%ymm5
vmovdqa		64(%rdi),%ymm6
vmovdqa		96(%rdi),%ymm7
vmovdqa		128(%rdi),%ymm8
vmovdqa		160(%rdi),%ymm9
vmovdqa		192(%rdi),%ymm10
vmovdqa		224(%rdi),%ymm11

agg_avx2_shuffle8	4,8,3,8
agg_avx2_shuffle8	5,9,4,9
agg_avx2_shuffle8	6,10,5,10
agg_avx2_shuffle8	7,11,6,11

agg_avx2_shuffle4	3,5,7,5
agg_avx2_shuffle4	8,10,3,10
agg_avx2_shuffle4	4,6,8,6
agg_avx2_shuffle4	9,11,4,11

agg_avx2_shuffle2	7,8,9,8
agg_avx2_shuffle2	5,6,7,6
agg_avx2_shuffle2	3,4,5,4
agg_avx2_shuffle2	10,11,3,11

#store
vmovdqa		%ymm9,(%rdi)
vmovdqa		%ymm8,32(%rdi)
vmovdqa		%ymm7,64(%rdi)
vmovdqa		%ymm6,96(%rdi)
vmovdqa		%ymm5,128(%rdi)
vmovdqa		%ymm4,160(%rdi)
vmovdqa		%ymm3,192(%rdi)
vmovdqa		%ymm11,224(%rdi)

add $256,%rdi
#load
vmovdqa		(%rdi),%ymm4
vmovdqa		32(%rdi),%ymm5
vmovdqa		64(%rdi),%ymm6
vmovdqa		96(%rdi),%ymm7
vmovdqa		128(%rdi),%ymm8
vmovdqa		160(%rdi),%ymm9
vmovdqa		192(%rdi),%ymm10
vmovdqa		224(%rdi),%ymm11

agg_avx2_shuffle8	4,8,3,8
agg_avx2_shuffle8	5,9,4,9
agg_avx2_shuffle8	6,10,5,10
agg_avx2_shuffle8	7,11,6,11

agg_avx2_shuffle4	3,5,7,5
agg_avx2_shuffle4	8,10,3,10
agg_avx2_shuffle4	4,6,8,6
agg_avx2_shuffle4	9,11,4,11

agg_avx2_shuffle2	7,8,9,8
agg_avx2_shuffle2	5,6,7,6
agg_avx2_shuffle2	3,4,5,4
agg_avx2_shuffle2	10,11,3,11

#store
vmovdqa		%ymm9,(%rdi)
vmovdqa		%ymm8,32(%rdi)
vmovdqa		%ymm7,64(%rdi)
vmovdqa		%ymm6,96(%rdi)
vmovdqa		%ymm5,128(%rdi)
vmovdqa		%ymm4,160(%rdi)
vmovdqa		%ymm3,192(%rdi)
vmovdqa		%ymm11,224(%rdi)

add $256,%rdi
#load
vmovdqa		(%rdi),%ymm4
vmovdqa		32(%rdi),%ymm5
vmovdqa		64(%rdi),%ymm6
vmovdqa		96(%rdi),%ymm7
vmovdqa		128(%rdi),%ymm8
vmovdqa		160(%rdi),%ymm9
vmovdqa		192(%rdi),%ymm10
vmovdqa		224(%rdi),%ymm11

agg_avx2_shuffle8	4,8,3,8
agg_avx2_shuffle8	5,9,4,9
agg_avx2_shuffle8	6,10,5,10
agg_avx2_shuffle8	7,11,6,11

agg_avx2_shuffle4	3,5,7,5
agg_avx2_shuffle4	8,10,3,10
agg_avx2_shuffle4	4,6,8,6
agg_avx2_shuffle4	9,11,4,11

agg_avx2_shuffle2	7,8,9,8
agg_avx2_shuffle2	5,6,7,6
agg_avx2_shuffle2	3,4,5,4
agg_avx2_shuffle2	10,11,3,11

#store
vmovdqa		%ymm9,(%rdi)
vmovdqa		%ymm8,32(%rdi)
vmovdqa		%ymm7,64(%rdi)
vmovdqa		%ymm6,96(%rdi)
vmovdqa		%ymm5,128(%rdi)
vmovdqa		%ymm4,160(%rdi)
vmovdqa		%ymm3,192(%rdi)
vmovdqa		%ymm11,224(%rdi)

vzeroupper
ret

.macro agg_avx2_butterfly l,h,zl0=1,zl1=1,zh0=2,zh1=2
vpsubd		%ymm\l,%ymm\h,%ymm12
vpaddd		%ymm\h,%ymm\l,%ymm\l

vpmuldq		%ymm\zl0,%ymm12,%ymm13
vmovshdup	%ymm12,%ymm\h
vpmuldq		%ymm\zl1,%ymm\h,%ymm14

vpmuldq		%ymm\zh0,%ymm12,%ymm12
vpmuldq		%ymm\zh1,%ymm\h,%ymm\h

vpmuldq		%ymm0,%ymm13,%ymm13
vpmuldq		%ymm0,%ymm14,%ymm14

vpsubd		%ymm13,%ymm12,%ymm12
vpsubd		%ymm14,%ymm\h,%ymm\h

vmovshdup	%ymm12,%ymm12
vpblendd	$0xAA,%ymm\h,%ymm12,%ymm\h
.endm

.macro agg_avx2_levels0t5 off
vmovdqa		256*\off+  0(%rdi),%ymm4
vmovdqa		256*\off+ 32(%rdi),%ymm5
vmovdqa		256*\off+ 64(%rdi),%ymm6
vmovdqa	 	256*\off+ 96(%rdi),%ymm7
vmovdqa		256*\off+128(%rdi),%ymm8
vmovdqa		256*\off+160(%rdi),%ymm9
vmovdqa		256*\off+192(%rdi),%ymm10
vmovdqa	 	256*\off+224(%rdi),%ymm11

/* level 0 */
agg_avx2_invz	35-\off,3,1,15,2
agg_avx2_butterfly	4,5,1,3,2,15

agg_avx2_invz	31-\off,3,1,15,2
agg_avx2_butterfly	6,7,1,3,2,15

agg_avx2_invz	27-\off,3,1,15,2
agg_avx2_butterfly	8,9,1,3,2,15

agg_avx2_invz	23-\off,3,1,15,2
agg_avx2_butterfly	10,11,1,3,2,15

/* level 1 */
agg_avx2_invz	19-\off,3,1,15,2
agg_avx2_butterfly	4,6,1,3,2,15
agg_avx2_butterfly	5,7,1,3,2,15

agg_avx2_invz	15-\off,3,1,15,2
agg_avx2_butterfly	8,10,1,3,2,15
agg_avx2_butterfly	9,11,1,3,2,15

/* level 2 */
agg_avx2_invz	11-\off,3,1,15,2
agg_avx2_butterfly	4,8,1,3,2,15
agg_avx2_butterfly	5,9,1,3,2,15
agg_avx2_butterfly	6,10,1,3,2,15
agg_avx2_butterfly	7,11,1,3,2,15

/* level 3 */
agg_avx2_shuffle2	4,5,3,5
agg_avx2_shuffle2	6,7,4,7
agg_avx2_shuffle2	8,9,6,9
agg_avx2_shuffle2	10,11,8,11

vmovdqa		(agg_avx2__INV+32*(7-\off))*4(%rsi),%ymm1
vmovdqa		(agg_avx2__INV+32*(7-\off)+16)*4(%rsi),%ymm2
agg_avx2_butterfly	3,5
agg_avx2_butterfly	4,7
agg_avx2_butterfly	6,9
agg_avx2_butterfly	8,11

/* level 4 */
agg_avx2_shuffle4	3,4,10,4
agg_avx2_shuffle4	6,8,3,8
agg_avx2_shuffle4	5,7,6,7
agg_avx2_shuffle4	9,11,5,11

vmovdqa		(agg_avx2__INV+32*(3-\off))*4(%rsi),%ymm1
vmovdqa		(agg_avx2__INV+32*(3-\off)+16)*4(%rsi),%ymm2
agg_avx2_butterfly	10,4
agg_avx2_butterfly	3,8
agg_avx2_butterfly	6,7
agg_avx2_butterfly	5,11

/* level 5 */
agg_avx2_shuffle8	10,3,9,3
agg_avx2_shuffle8	6,5,10,5
agg_avx2_shuffle8	4,8,6,8
agg_avx2_shuffle8	7,11,4,11

vpbroadcastd	(agg_avx2__ZETAS_QINV+7-\off)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+7-\off)*4(%rsi),%ymm2
agg_avx2_butterfly	9,3
agg_avx2_butterfly	10,5
agg_avx2_butterfly	6,8
agg_avx2_butterfly	4,11

vmovdqa		%ymm9,256*\off+  0(%rdi)
vmovdqa		%ymm10,256*\off+ 32(%rdi)
vmovdqa		%ymm6,256*\off+ 64(%rdi)
vmovdqa		%ymm4,256*\off+ 96(%rdi)
vmovdqa		%ymm3,256*\off+128(%rdi)
vmovdqa		%ymm5,256*\off+160(%rdi)
vmovdqa		%ymm8,256*\off+192(%rdi)
vmovdqa		%ymm11,256*\off+224(%rdi)
.endm

/* Store a final value r in (-q,q): min_u32(r, r+q) is canonical, because a
   negative r is a huge unsigned value while r+q is in [0,q). Centered output
   then subtracts q above (q-1)/2, which levels6t7 keeps in ymm3. */
.macro agg_avx2_fstore r,dst,centered
vpaddd		%ymm0,%ymm\r,%ymm12
vpminud		%ymm12,%ymm\r,%ymm\r
.if \centered
vpcmpgtd	%ymm3,%ymm\r,%ymm12
vpand		%ymm0,%ymm12,%ymm12
vpsubd		%ymm12,%ymm\r,%ymm\r
.endif
vmovdqa		%ymm\r,\dst(%rdi)
.endm

.macro agg_avx2_levels6t7 off,centered
.if \centered
vpsrld		$1,%ymm0,%ymm3
.endif
vmovdqa		  0+32*\off(%rdi),%ymm4
vmovdqa		128+32*\off(%rdi),%ymm5
vmovdqa		256+32*\off(%rdi),%ymm6
vmovdqa		384+32*\off(%rdi),%ymm7
vmovdqa		512+32*\off(%rdi),%ymm8
vmovdqa		640+32*\off(%rdi),%ymm9
vmovdqa		768+32*\off(%rdi),%ymm10
vmovdqa		896+32*\off(%rdi),%ymm11

/* level 6 */
vpbroadcastd	(agg_avx2__ZETAS_QINV+3)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+3)*4(%rsi),%ymm2
agg_avx2_butterfly	4,6
agg_avx2_butterfly	5,7

vpbroadcastd	(agg_avx2__ZETAS_QINV+2)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+2)*4(%rsi),%ymm2
agg_avx2_butterfly	8,10
agg_avx2_butterfly	9,11

/* level 7 */
vpbroadcastd	(agg_avx2__ZETAS_QINV+0)*4(%rsi),%ymm1
vpbroadcastd	(agg_avx2__ZETAS+0)*4(%rsi),%ymm2

agg_avx2_butterfly	4,8
agg_avx2_butterfly	5,9
agg_avx2_butterfly	6,10
agg_avx2_butterfly	7,11

agg_avx2_fstore	8,512+32*\off,\centered
agg_avx2_fstore	9,640+32*\off,\centered
agg_avx2_fstore	10,768+32*\off,\centered
agg_avx2_fstore	11,896+32*\off,\centered

vmovdqa		(agg_avx2__8XDIV_QINV)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__8XDIV)*4(%rsi),%ymm2
vpmuldq		%ymm1,%ymm4,%ymm12
vpmuldq		%ymm1,%ymm5,%ymm13
vmovshdup	%ymm4,%ymm8
vmovshdup	%ymm5,%ymm9
vpmuldq		%ymm1,%ymm8,%ymm14
vpmuldq		%ymm1,%ymm9,%ymm15
vpmuldq		%ymm2,%ymm4,%ymm4
vpmuldq		%ymm2,%ymm5,%ymm5
vpmuldq		%ymm2,%ymm8,%ymm8
vpmuldq		%ymm2,%ymm9,%ymm9
vpmuldq		%ymm0,%ymm12,%ymm12
vpmuldq		%ymm0,%ymm13,%ymm13
vpmuldq		%ymm0,%ymm14,%ymm14
vpmuldq		%ymm0,%ymm15,%ymm15
vpsubd		%ymm12,%ymm4,%ymm4
vpsubd		%ymm13,%ymm5,%ymm5
vpsubd		%ymm14,%ymm8,%ymm8
vpsubd		%ymm15,%ymm9,%ymm9
vmovshdup	%ymm4,%ymm4
vmovshdup	%ymm5,%ymm5
vpblendd	$0xAA,%ymm8,%ymm4,%ymm4
vpblendd	$0xAA,%ymm9,%ymm5,%ymm5

vpmuldq		%ymm1,%ymm6,%ymm12
vpmuldq		%ymm1,%ymm7,%ymm13
vmovshdup	%ymm6,%ymm8
vmovshdup	%ymm7,%ymm9
vpmuldq		%ymm1,%ymm8,%ymm14
vpmuldq		%ymm1,%ymm9,%ymm15
vpmuldq		%ymm2,%ymm6,%ymm6
vpmuldq		%ymm2,%ymm7,%ymm7
vpmuldq		%ymm2,%ymm8,%ymm8
vpmuldq		%ymm2,%ymm9,%ymm9
vpmuldq		%ymm0,%ymm12,%ymm12
vpmuldq		%ymm0,%ymm13,%ymm13
vpmuldq		%ymm0,%ymm14,%ymm14
vpmuldq		%ymm0,%ymm15,%ymm15
vpsubd		%ymm12,%ymm6,%ymm6
vpsubd		%ymm13,%ymm7,%ymm7
vpsubd		%ymm14,%ymm8,%ymm8
vpsubd		%ymm15,%ymm9,%ymm9
vmovshdup	%ymm6,%ymm6
vmovshdup	%ymm7,%ymm7
vpblendd	$0xAA,%ymm8,%ymm6,%ymm6
vpblendd	$0xAA,%ymm9,%ymm7,%ymm7

agg_avx2_fstore	4,0+32*\off,\centered
agg_avx2_fstore	5,128+32*\off,\centered
agg_avx2_fstore	6,256+32*\off,\centered
agg_avx2_fstore	7,384+32*\off,\centered
.endm

.text
.global agg_mldsa_inverse_avx2
agg_mldsa_inverse_avx2:
vmovdqa		agg_avx2__8XQ*4(%rsi),%ymm0

agg_avx2_levels0t5	0
agg_avx2_levels0t5	1
agg_avx2_levels0t5	2
agg_avx2_levels0t5	3

test %edx,%edx
jnz 992f
agg_avx2_levels6t7 0,0
agg_avx2_levels6t7 1,0
agg_avx2_levels6t7 2,0
agg_avx2_levels6t7 3,0

vzeroupper
ret
992:
agg_avx2_levels6t7 0,1
agg_avx2_levels6t7 1,1
agg_avx2_levels6t7 2,1
agg_avx2_levels6t7 3,1
vzeroupper
ret


"#,
    options(att_syntax)
);
unsafe extern "sysv64" {
    pub(super) fn agg_mldsa_ntt_avx2(a: &mut Poly32, t: &tables::Table);
    pub(super) fn agg_mldsa_dot_avx2(
        out: &mut Poly32,
        a: &[Poly32; 4],
        b: &[Poly32; 4],
        t: &tables::Table,
        center: u32,
    );
    pub(super) fn agg_mldsa_order_avx2(a: &mut Poly32);
    pub(super) fn agg_mldsa_inverse_avx2(a: &mut Poly32, t: &tables::Table, center: u32);
}

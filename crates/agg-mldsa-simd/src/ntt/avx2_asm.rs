//! pq-crystals Dilithium AVX2 assembly port (CC0/Apache-2.0).
//! Kept for kernels whose LLVM register allocation misses the reference gate.
//! Forward butterflies keep |x| below 4.52q or 4.25P2; q inverse sums
//! stay below 256q. Four dot products accumulate in i64 below 2^58.
//! P2 dot outputs lie in (-p,p) before correction to the centered interval.
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

.macro agg_avx2_butterfly l,h,zl0=1,zl1=1,zh0=2,zh1=2
vpmuldq		%ymm\zl0,%ymm\h,%ymm13
vmovshdup	%ymm\h,%ymm12
vpmuldq		%ymm\zl1,%ymm12,%ymm14

vpmuldq		%ymm\zh0,%ymm\h,%ymm\h
vpmuldq		%ymm\zh1,%ymm12,%ymm12

vpmuldq		%ymm0,%ymm13,%ymm13
vpmuldq		%ymm0,%ymm14,%ymm14

vmovshdup	%ymm\h,%ymm\h
vpblendd	$0xAA,%ymm12,%ymm\h,%ymm\h

vpsubd		%ymm\h,%ymm\l,%ymm12
vpaddd		%ymm\h,%ymm\l,%ymm\l

vmovshdup	%ymm13,%ymm13
vpblendd	$0xAA,%ymm14,%ymm13,%ymm13

vpaddd		%ymm13,%ymm12,%ymm\h
vpsubd		%ymm13,%ymm\l,%ymm\l
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
vpsrlq		$32,%ymm1,%ymm10
vmovshdup	%ymm2,%ymm15

agg_avx2_butterfly	9,5,1,10,2,15
agg_avx2_butterfly	8,4,1,10,2,15
agg_avx2_butterfly	7,3,1,10,2,15
agg_avx2_butterfly	6,11,1,10,2,15

/* level 6 */
vmovdqa		(agg_avx2__ZETAS_QINV+104+8*\off)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+104+8*\off)*4(%rsi),%ymm2
vpsrlq		$32,%ymm1,%ymm10
vmovshdup	%ymm2,%ymm15
agg_avx2_butterfly	9,7,1,10,2,15
agg_avx2_butterfly	8,6,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+104+8*\off+32)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+104+8*\off+32)*4(%rsi),%ymm2
vpsrlq		$32,%ymm1,%ymm10
vmovshdup	%ymm2,%ymm15
agg_avx2_butterfly	5,3,1,10,2,15
agg_avx2_butterfly	4,11,1,10,2,15

/* level 7 */
vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off)*4(%rsi),%ymm2
vpsrlq		$32,%ymm1,%ymm10
vmovshdup	%ymm2,%ymm15
agg_avx2_butterfly	9,8,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off+32)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off+32)*4(%rsi),%ymm2
vpsrlq		$32,%ymm1,%ymm10
vmovshdup	%ymm2,%ymm15
agg_avx2_butterfly	7,6,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off+64)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off+64)*4(%rsi),%ymm2
vpsrlq		$32,%ymm1,%ymm10
vmovshdup	%ymm2,%ymm15
agg_avx2_butterfly	5,4,1,10,2,15

vmovdqa		(agg_avx2__ZETAS_QINV+168+8*\off+96)*4(%rsi),%ymm1
vmovdqa		(agg_avx2__ZETAS+168+8*\off+96)*4(%rsi),%ymm2
vpsrlq		$32,%ymm1,%ymm10
vmovshdup	%ymm2,%ymm15
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

ret

.section .note.GNU-stack,"",@progbits
.purgem agg_avx2_butterfly
.text
.macro agg_avx2_pointwise off
#load
vmovdqa		\off(%rsi),%ymm6
vmovdqa		\off+32(%rsi),%ymm8
vmovdqa		\off(%rdx),%ymm10
vmovdqa		\off+32(%rdx),%ymm12
vpsrlq		$32,%ymm6,%ymm7
vpsrlq		$32,%ymm8,%ymm9
vmovshdup	%ymm10,%ymm11
vmovshdup	%ymm12,%ymm13

#mul
vpmuldq		%ymm6,%ymm10,%ymm6
vpmuldq		%ymm7,%ymm11,%ymm7
vpmuldq		%ymm8,%ymm12,%ymm8
vpmuldq		%ymm9,%ymm13,%ymm9
.endm

.macro agg_avx2_acc
vpaddq		%ymm6,%ymm2,%ymm2
vpaddq		%ymm7,%ymm3,%ymm3
vpaddq		%ymm8,%ymm4,%ymm4
vpaddq		%ymm9,%ymm5,%ymm5
.endm

.macro agg_avx2_dotbody center
agg_avx2_pointwise	0

#mov
vmovdqa		%ymm6,%ymm2
vmovdqa		%ymm7,%ymm3
vmovdqa		%ymm8,%ymm4
vmovdqa		%ymm9,%ymm5

agg_avx2_pointwise	1024
agg_avx2_acc

agg_avx2_pointwise	2048
agg_avx2_acc

agg_avx2_pointwise	3072
agg_avx2_acc







#reduce
vpmuldq		%ymm0,%ymm2,%ymm6
vpmuldq		%ymm0,%ymm3,%ymm7
vpmuldq		%ymm0,%ymm4,%ymm8
vpmuldq		%ymm0,%ymm5,%ymm9
vpmuldq		%ymm1,%ymm6,%ymm6
vpmuldq		%ymm1,%ymm7,%ymm7
vpmuldq		%ymm1,%ymm8,%ymm8
vpmuldq		%ymm1,%ymm9,%ymm9
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
vpabsd %ymm2,%ymm6
vpcmpgtd %ymm14,%ymm6,%ymm6
vpand %ymm1,%ymm6,%ymm6
vpsignd %ymm2,%ymm6,%ymm6
vpsubd %ymm6,%ymm2,%ymm2
vpabsd %ymm4,%ymm6
vpcmpgtd %ymm14,%ymm6,%ymm6
vpand %ymm1,%ymm6,%ymm6
vpsignd %ymm4,%ymm6,%ymm6
vpsubd %ymm6,%ymm4,%ymm4
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
vmovdqa agg_avx2__8XQINV*4(%rcx),%ymm0
vmovdqa agg_avx2__8XQ*4(%rcx),%ymm1
xor %eax,%eax
test %r8d,%r8d
jnz 990f
.p2align 5
989:
agg_avx2_dotbody 0
add $1,%eax
cmp $16,%eax
jb 989b
ret
990:
vpsrld $1,%ymm1,%ymm14
.p2align 5
988:
.rept 4
agg_avx2_dotbody 1
.endr
add $1,%eax
cmp $4,%eax
jb 988b
ret
.section .note.GNU-stack,"",@progbits
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
vpermq		$0x1B,(agg_avx2__ZETAS_QINV+296-8*\off-8)*4(%rsi),%ymm3
vpermq		$0x1B,(agg_avx2__ZETAS+296-8*\off-8)*4(%rsi),%ymm15
vmovshdup	%ymm3,%ymm1
vmovshdup	%ymm15,%ymm2
agg_avx2_butterfly	4,5,1,3,2,15

vpermq		$0x1B,(agg_avx2__ZETAS_QINV+296-8*\off-40)*4(%rsi),%ymm3
vpermq		$0x1B,(agg_avx2__ZETAS+296-8*\off-40)*4(%rsi),%ymm15
vmovshdup	%ymm3,%ymm1
vmovshdup	%ymm15,%ymm2
agg_avx2_butterfly	6,7,1,3,2,15

vpermq		$0x1B,(agg_avx2__ZETAS_QINV+296-8*\off-72)*4(%rsi),%ymm3
vpermq		$0x1B,(agg_avx2__ZETAS+296-8*\off-72)*4(%rsi),%ymm15
vmovshdup	%ymm3,%ymm1
vmovshdup	%ymm15,%ymm2
agg_avx2_butterfly	8,9,1,3,2,15

vpermq		$0x1B,(agg_avx2__ZETAS_QINV+296-8*\off-104)*4(%rsi),%ymm3
vpermq		$0x1B,(agg_avx2__ZETAS+296-8*\off-104)*4(%rsi),%ymm15
vmovshdup	%ymm3,%ymm1
vmovshdup	%ymm15,%ymm2
agg_avx2_butterfly	10,11,1,3,2,15

/* level 1 */
vpermq		$0x1B,(agg_avx2__ZETAS_QINV+168-8*\off-8)*4(%rsi),%ymm3
vpermq		$0x1B,(agg_avx2__ZETAS+168-8*\off-8)*4(%rsi),%ymm15
vmovshdup	%ymm3,%ymm1
vmovshdup	%ymm15,%ymm2
agg_avx2_butterfly	4,6,1,3,2,15
agg_avx2_butterfly	5,7,1,3,2,15

vpermq		$0x1B,(agg_avx2__ZETAS_QINV+168-8*\off-40)*4(%rsi),%ymm3
vpermq		$0x1B,(agg_avx2__ZETAS+168-8*\off-40)*4(%rsi),%ymm15
vmovshdup	%ymm3,%ymm1
vmovshdup	%ymm15,%ymm2
agg_avx2_butterfly	8,10,1,3,2,15
agg_avx2_butterfly	9,11,1,3,2,15

/* level 2 */
vpermq		$0x1B,(agg_avx2__ZETAS_QINV+104-8*\off-8)*4(%rsi),%ymm3
vpermq		$0x1B,(agg_avx2__ZETAS+104-8*\off-8)*4(%rsi),%ymm15
vmovshdup	%ymm3,%ymm1
vmovshdup	%ymm15,%ymm2
agg_avx2_butterfly	4,8,1,3,2,15
agg_avx2_butterfly	5,9,1,3,2,15
agg_avx2_butterfly	6,10,1,3,2,15
agg_avx2_butterfly	7,11,1,3,2,15

/* level 3 */
agg_avx2_shuffle2	4,5,3,5
agg_avx2_shuffle2	6,7,4,7
agg_avx2_shuffle2	8,9,6,9
agg_avx2_shuffle2	10,11,8,11

vpermq		$0x1B,(agg_avx2__ZETAS_QINV+72-8*\off-8)*4(%rsi),%ymm1
vpermq		$0x1B,(agg_avx2__ZETAS+72-8*\off-8)*4(%rsi),%ymm2
agg_avx2_butterfly	3,5
agg_avx2_butterfly	4,7
agg_avx2_butterfly	6,9
agg_avx2_butterfly	8,11

/* level 4 */
agg_avx2_shuffle4	3,4,10,4
agg_avx2_shuffle4	6,8,3,8
agg_avx2_shuffle4	5,7,6,7
agg_avx2_shuffle4	9,11,5,11

vpermq		$0x1B,(agg_avx2__ZETAS_QINV+40-8*\off-8)*4(%rsi),%ymm1
vpermq		$0x1B,(agg_avx2__ZETAS+40-8*\off-8)*4(%rsi),%ymm2
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

.macro agg_avx2_levels6t7 off,centered
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

vpsrad $31,%ymm8,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm8,%ymm8
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm8,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm8,%ymm8
.endif
vmovdqa         %ymm8,512+32*\off(%rdi)
vpsrad $31,%ymm9,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm9,%ymm9
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm9,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm9,%ymm9
.endif
vmovdqa         %ymm9,640+32*\off(%rdi)
vpsrad $31,%ymm10,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm10,%ymm10
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm10,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm10,%ymm10
.endif
vmovdqa         %ymm10,768+32*\off(%rdi)
vpsrad $31,%ymm11,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm11,%ymm11
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm11,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm11,%ymm11
.endif
vmovdqa         %ymm11,896+32*\off(%rdi)

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

vpsrad $31,%ymm4,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm4,%ymm4
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm4,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm4,%ymm4
.endif
vmovdqa         %ymm4,  0+32*\off(%rdi)
vpsrad $31,%ymm5,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm5,%ymm5
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm5,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm5,%ymm5
.endif
vmovdqa         %ymm5,128+32*\off(%rdi)
vpsrad $31,%ymm6,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm6,%ymm6
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm6,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm6,%ymm6
.endif
vmovdqa         %ymm6,256+32*\off(%rdi)
vpsrad $31,%ymm7,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpaddd %ymm12,%ymm7,%ymm7
.if \centered
vpsrld $1,%ymm0,%ymm13
vpcmpgtd %ymm13,%ymm7,%ymm12
vpand %ymm0,%ymm12,%ymm12
vpsubd %ymm12,%ymm7,%ymm7
.endif
vmovdqa         %ymm7,384+32*\off(%rdi)
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

ret
992:
agg_avx2_levels6t7 0,1
agg_avx2_levels6t7 1,1
agg_avx2_levels6t7 2,1
agg_avx2_levels6t7 3,1
ret

.section .note.GNU-stack,"",@progbits
.section .note.GNU-stack,"",@progbits

"#,
    options(att_syntax)
);
unsafe extern "C" {
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

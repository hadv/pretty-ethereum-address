/**
 * AVX-512 8-way Parallel Field Multiplication
 * 
 * Uses AVX-512F for 8-way parallelism (512-bit registers = 8 × 64-bit)
 * Uses AVX-512IFMA for native 52×52→104 bit multiply-add (vpmadd52luq/vpmadd52huq)
 * 
 * Requires: AVX-512F, AVX-512IFMA, AVX-512VL
 */

#ifndef SECP256K1_AVX512_FIELD_MUL_H
#define SECP256K1_AVX512_FIELD_MUL_H

#ifdef __AVX512F__

#include <immintrin.h>
#include <stdint.h>

/**
 * 8-way parallel field elements in limb-sliced layout for AVX-512.
 * 
 * Memory layout (for 8 field elements A-H):
 *   limb[0] = [A.n[0], B.n[0], C.n[0], D.n[0], E.n[0], F.n[0], G.n[0], H.n[0]]
 *   limb[1] = [A.n[1], B.n[1], C.n[1], D.n[1], E.n[1], F.n[1], G.n[1], H.n[1]]
 *   ... etc for limb[2..4]
 */
typedef struct {
    uint64_t limb[5][8] __attribute__((aligned(64)));
} fe8_t;

/* AVX-512 constants */
static const uint64_t MASK52_8[8] __attribute__((aligned(64))) = {
    0xFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFULL
};

static const uint64_t R_8[8] __attribute__((aligned(64))) = {
    0x1000003D10ULL, 0x1000003D10ULL, 0x1000003D10ULL, 0x1000003D10ULL,
    0x1000003D10ULL, 0x1000003D10ULL, 0x1000003D10ULL, 0x1000003D10ULL
};

#ifdef __AVX512IFMA__
/**
 * 8-way 52×52→104 bit multiply using AVX-512 IFMA
 * 
 * vpmadd52luq: dst = dst + (a × b)[0:52]   (low 52 bits)
 * vpmadd52huq: dst = dst + (a × b)[52:104] (high 52 bits)
 * 
 * This is the IDEAL instruction for secp256k1's 5×52 representation!
 */
static inline void mul52x8_ifma(__m512i a, __m512i b, __m512i *lo, __m512i *hi) {
    __m512i zero = _mm512_setzero_si512();
    
    /* Low 52 bits of a×b */
    *lo = _mm512_madd52lo_epu64(zero, a, b);
    
    /* High 52 bits of a×b */
    *hi = _mm512_madd52hi_epu64(zero, a, b);
}

/**
 * 8-way parallel field multiplication using AVX-512 IFMA
 * 
 * This is the OPTIMAL implementation for CPUs with IFMA support
 * (Intel Ice Lake, Tiger Lake, Alder Lake, Sapphire Rapids, etc.)
 */
static inline void fe8_mul_ifma(fe8_t *r, const fe8_t *a, const fe8_t *b) {
    __m512i M = _mm512_load_si512(MASK52_8);
    __m512i R = _mm512_load_si512(R_8);
    
    /* Load all limbs */
    __m512i a0 = _mm512_load_si512(a->limb[0]);
    __m512i a1 = _mm512_load_si512(a->limb[1]);
    __m512i a2 = _mm512_load_si512(a->limb[2]);
    __m512i a3 = _mm512_load_si512(a->limb[3]);
    __m512i a4 = _mm512_load_si512(a->limb[4]);
    
    __m512i b0 = _mm512_load_si512(b->limb[0]);
    __m512i b1 = _mm512_load_si512(b->limb[1]);
    __m512i b2 = _mm512_load_si512(b->limb[2]);
    __m512i b3 = _mm512_load_si512(b->limb[3]);
    __m512i b4 = _mm512_load_si512(b->limb[4]);
    
    __m512i zero = _mm512_setzero_si512();
    __m512i r0, r1, r2, r3, r4;
    __m512i hi_acc;
    
    /* r0 = a0*b0 (low) */
    r0 = _mm512_madd52lo_epu64(zero, a0, b0);
    hi_acc = _mm512_madd52hi_epu64(zero, a0, b0);
    
    /* r1 = a0*b1 + a1*b0 (low) + carry from r0 */
    r1 = _mm512_madd52lo_epu64(hi_acc, a0, b1);
    r1 = _mm512_madd52lo_epu64(r1, a1, b0);
    hi_acc = _mm512_madd52hi_epu64(zero, a0, b1);
    hi_acc = _mm512_madd52hi_epu64(hi_acc, a1, b0);
    
    /* r2 = a0*b2 + a1*b1 + a2*b0 (low) + carry */
    r2 = _mm512_madd52lo_epu64(hi_acc, a0, b2);
    r2 = _mm512_madd52lo_epu64(r2, a1, b1);
    r2 = _mm512_madd52lo_epu64(r2, a2, b0);
    hi_acc = _mm512_madd52hi_epu64(zero, a0, b2);
    hi_acc = _mm512_madd52hi_epu64(hi_acc, a1, b1);
    hi_acc = _mm512_madd52hi_epu64(hi_acc, a2, b0);
    
    /* r3 = a0*b3 + a1*b2 + a2*b1 + a3*b0 + carry */
    r3 = _mm512_madd52lo_epu64(hi_acc, a0, b3);
    r3 = _mm512_madd52lo_epu64(r3, a1, b2);
    r3 = _mm512_madd52lo_epu64(r3, a2, b1);
    r3 = _mm512_madd52lo_epu64(r3, a3, b0);
    hi_acc = _mm512_madd52hi_epu64(zero, a0, b3);
    hi_acc = _mm512_madd52hi_epu64(hi_acc, a1, b2);
    hi_acc = _mm512_madd52hi_epu64(hi_acc, a2, b1);
    hi_acc = _mm512_madd52hi_epu64(hi_acc, a3, b0);
    
    /* r4 = a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 + carry */
    r4 = _mm512_madd52lo_epu64(hi_acc, a0, b4);
    r4 = _mm512_madd52lo_epu64(r4, a1, b3);
    r4 = _mm512_madd52lo_epu64(r4, a2, b2);
    r4 = _mm512_madd52lo_epu64(r4, a3, b1);
    r4 = _mm512_madd52lo_epu64(r4, a4, b0);
    
    /* High terms need reduction by R = 2^256 mod p */
    /* h5 = a1*b4 + a2*b3 + a3*b2 + a4*b1 → contributes R*h5 to r0 */
    __m512i h5 = _mm512_madd52lo_epu64(zero, a1, b4);
    h5 = _mm512_madd52lo_epu64(h5, a2, b3);
    h5 = _mm512_madd52lo_epu64(h5, a3, b2);
    h5 = _mm512_madd52lo_epu64(h5, a4, b1);
    r0 = _mm512_madd52lo_epu64(r0, h5, R);
    
    /* h6 = a2*b4 + a3*b3 + a4*b2 → contributes R*h6 to r1 */
    __m512i h6 = _mm512_madd52lo_epu64(zero, a2, b4);
    h6 = _mm512_madd52lo_epu64(h6, a3, b3);
    h6 = _mm512_madd52lo_epu64(h6, a4, b2);
    r1 = _mm512_madd52lo_epu64(r1, h6, R);
    
    /* h7 = a3*b4 + a4*b3 → contributes R*h7 to r2 */
    __m512i h7 = _mm512_madd52lo_epu64(zero, a3, b4);
    h7 = _mm512_madd52lo_epu64(h7, a4, b3);
    r2 = _mm512_madd52lo_epu64(r2, h7, R);
    
    /* h8 = a4*b4 → contributes R*h8 to r3 */
    __m512i h8 = _mm512_madd52lo_epu64(zero, a4, b4);
    r3 = _mm512_madd52lo_epu64(r3, h8, R);
    
    /* Carry propagation */
    __m512i c;
    c = _mm512_srli_epi64(r0, 52); r0 = _mm512_and_si512(r0, M);
    r1 = _mm512_add_epi64(r1, c);
    c = _mm512_srli_epi64(r1, 52); r1 = _mm512_and_si512(r1, M);
    r2 = _mm512_add_epi64(r2, c);
    c = _mm512_srli_epi64(r2, 52); r2 = _mm512_and_si512(r2, M);
    r3 = _mm512_add_epi64(r3, c);
    c = _mm512_srli_epi64(r3, 52); r3 = _mm512_and_si512(r3, M);
    r4 = _mm512_add_epi64(r4, c);
    
    /* Store results */
    _mm512_store_si512(r->limb[0], r0);
    _mm512_store_si512(r->limb[1], r1);
    _mm512_store_si512(r->limb[2], r2);
    _mm512_store_si512(r->limb[3], r3);
    _mm512_store_si512(r->limb[4], r4);
}

#endif /* __AVX512IFMA__ */

#endif /* __AVX512F__ */

#endif /* SECP256K1_AVX512_FIELD_MUL_H */


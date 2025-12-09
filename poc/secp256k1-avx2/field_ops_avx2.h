/**
 * secp256k1 Field Operations - AVX2 4-way parallel
 * 
 * Implements field addition and subtraction for 4 elements at once.
 */

#ifndef SECP256K1_AVX2_FIELD_OPS_H
#define SECP256K1_AVX2_FIELD_OPS_H

#include <immintrin.h>
#include "field.h"

/**
 * 4-way parallel field addition: r = a + b (mod p)
 * 
 * Note: This is a lazy reduction - results may exceed p but stay within
 * limb bounds. Full reduction happens during serialization.
 */
static inline void fe4_add(fe4_t *r, const fe4_t *a, const fe4_t *b) {
    for (int i = 0; i < 5; i++) {
        __m256i va = _mm256_load_si256((__m256i*)a->limb[i]);
        __m256i vb = _mm256_load_si256((__m256i*)b->limb[i]);
        __m256i vr = _mm256_add_epi64(va, vb);
        _mm256_store_si256((__m256i*)r->limb[i], vr);
    }
}

/**
 * 4-way parallel field subtraction: r = a - b (mod p)
 * 
 * To avoid underflow, we add 2*p before subtracting.
 * This keeps results positive while staying within limb bounds.
 */
static inline void fe4_sub(fe4_t *r, const fe4_t *a, const fe4_t *b) {
    /* 2*p in limb representation (allows subtraction without underflow) */
    static const uint64_t P2[5] = {
        0x1FFFFDFFFFF85EULL,  /* 2 * (2^52 - 0x1000003D1) */
        0x1FFFFFFFFFFFFEULL,
        0x1FFFFFFFFFFFFEULL,
        0x1FFFFFFFFFFFFEULL,
        0x1FFFFFFFFFFFFEULL
    };
    
    for (int i = 0; i < 5; i++) {
        __m256i va = _mm256_load_si256((__m256i*)a->limb[i]);
        __m256i vb = _mm256_load_si256((__m256i*)b->limb[i]);
        __m256i vp2 = _mm256_set1_epi64x(P2[i]);
        /* r = (a + 2*p) - b = a - b + 2*p */
        __m256i vr = _mm256_add_epi64(va, vp2);
        vr = _mm256_sub_epi64(vr, vb);
        _mm256_store_si256((__m256i*)r->limb[i], vr);
    }
}

/**
 * 4-way parallel field negation: r = -a (mod p) = p - a
 */
static inline void fe4_neg(fe4_t *r, const fe4_t *a) {
    static const uint64_t P2[5] = {
        0x1FFFFDFFFFF85EULL,
        0x1FFFFFFFFFFFFEULL,
        0x1FFFFFFFFFFFFEULL,
        0x1FFFFFFFFFFFFEULL,
        0x1FFFFFFFFFFFFEULL
    };
    
    for (int i = 0; i < 5; i++) {
        __m256i va = _mm256_load_si256((__m256i*)a->limb[i]);
        __m256i vp2 = _mm256_set1_epi64x(P2[i]);
        __m256i vr = _mm256_sub_epi64(vp2, va);
        _mm256_store_si256((__m256i*)r->limb[i], vr);
    }
}

/* Scalar versions are in field_mul.h - include that for fe_add, fe_sub */

#endif /* SECP256K1_AVX2_FIELD_OPS_H */


/**
 * Scalar Field Multiplication - Reference Implementation
 * 
 * Based on Bitcoin Core's libsecp256k1 (field_5x52_int128_impl.h)
 * Uses 128-bit integers for intermediate products.
 */

#ifndef SECP256K1_AVX2_FIELD_MUL_H
#define SECP256K1_AVX2_FIELD_MUL_H

#include "field.h"

/* 128-bit unsigned integer type */
typedef unsigned __int128 uint128_t;

/**
 * Scalar field multiplication: r = a * b mod p
 * 
 * Uses schoolbook multiplication with lazy reduction.
 * Reduction uses the fact that 2^256 ≡ 0x1000003D1 (mod p)
 */
static inline void fe_mul(fe_t *r, const fe_t *a, const fe_t *b) {
    uint128_t c, d;
    uint64_t t3, t4, tx, u0;
    uint64_t a0 = a->n[0], a1 = a->n[1], a2 = a->n[2], a3 = a->n[3], a4 = a->n[4];
    const uint64_t M = 0xFFFFFFFFFFFFFULL;  /* 52-bit mask */
    const uint64_t R = 0x1000003D10ULL;     /* 2^256 mod p (shifted by 4) */

    /* Compute p3 = a0*b3 + a1*b2 + a2*b1 + a3*b0 */
    d = (uint128_t)a0 * b->n[3];
    d += (uint128_t)a1 * b->n[2];
    d += (uint128_t)a2 * b->n[1];
    d += (uint128_t)a3 * b->n[0];

    /* Compute p8 = a4*b4, reduce to lower limbs */
    c = (uint128_t)a4 * b->n[4];
    d += (uint128_t)R * (uint64_t)c;
    c >>= 64;
    t3 = (uint64_t)d & M;
    d >>= 52;

    /* Compute p4 */
    d += (uint128_t)a0 * b->n[4];
    d += (uint128_t)a1 * b->n[3];
    d += (uint128_t)a2 * b->n[2];
    d += (uint128_t)a3 * b->n[1];
    d += (uint128_t)a4 * b->n[0];
    d += (uint128_t)(R << 12) * (uint64_t)c;
    t4 = (uint64_t)d & M;
    d >>= 52;
    tx = (t4 >> 48);
    t4 &= (M >> 4);

    /* Compute p0 */
    c = (uint128_t)a0 * b->n[0];
    
    /* Compute p5 */
    d += (uint128_t)a1 * b->n[4];
    d += (uint128_t)a2 * b->n[3];
    d += (uint128_t)a3 * b->n[2];
    d += (uint128_t)a4 * b->n[1];
    u0 = (uint64_t)d & M;
    d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (uint128_t)u0 * (R >> 4);
    r->n[0] = (uint64_t)c & M;
    c >>= 52;

    /* Compute p1 */
    c += (uint128_t)a0 * b->n[1];
    c += (uint128_t)a1 * b->n[0];
    
    /* Compute p6 */
    d += (uint128_t)a2 * b->n[4];
    d += (uint128_t)a3 * b->n[3];
    d += (uint128_t)a4 * b->n[2];
    c += (uint128_t)((uint64_t)d & M) * R;
    d >>= 52;
    r->n[1] = (uint64_t)c & M;
    c >>= 52;

    /* Compute p2 */
    c += (uint128_t)a0 * b->n[2];
    c += (uint128_t)a1 * b->n[1];
    c += (uint128_t)a2 * b->n[0];
    
    /* Compute p7 */
    d += (uint128_t)a3 * b->n[4];
    d += (uint128_t)a4 * b->n[3];
    c += (uint128_t)R * (uint64_t)d;
    d >>= 64;
    r->n[2] = (uint64_t)c & M;
    c >>= 52;

    /* Final assembly */
    c += (uint128_t)(R << 12) * (uint64_t)d;
    c += t3;
    r->n[3] = (uint64_t)c & M;
    c >>= 52;
    r->n[4] = (uint64_t)c + t4;
}

/**
 * Normalize field element (full reduction to [0, p))
 */
static inline void fe_normalize(fe_t *r) {
    uint64_t t0 = r->n[0], t1 = r->n[1], t2 = r->n[2], t3 = r->n[3], t4 = r->n[4];
    uint64_t m, x;
    const uint64_t M = 0xFFFFFFFFFFFFFULL;

    /* Reduce t4 */
    x = t4 >> 48;
    t4 &= 0x0FFFFFFFFFFFFULL;
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= M;
    t2 += (t1 >> 52); t1 &= M; m = t1;
    t3 += (t2 >> 52); t2 &= M; m &= t2;
    t4 += (t3 >> 52); t3 &= M; m &= t3;

    /* Check if >= p */
    x = (t4 >> 48) | ((t4 == 0x0FFFFFFFFFFFFULL) & (m == 0xFFFFFFFFFFFFFULL)
        & (t0 >= 0xFFFFEFFFFFC2FULL));

    /* Final reduction */
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= M;
    t2 += (t1 >> 52); t1 &= M;
    t3 += (t2 >> 52); t2 &= M;
    t4 += (t3 >> 52); t3 &= M;
    t4 &= 0x0FFFFFFFFFFFFULL;

    r->n[0] = t0; r->n[1] = t1; r->n[2] = t2; r->n[3] = t3; r->n[4] = t4;
}

/**
 * Field addition: r = a + b (without reduction)
 */
static inline void fe_add(fe_t *r, const fe_t *a, const fe_t *b) {
    r->n[0] = a->n[0] + b->n[0];
    r->n[1] = a->n[1] + b->n[1];
    r->n[2] = a->n[2] + b->n[2];
    r->n[3] = a->n[3] + b->n[3];
    r->n[4] = a->n[4] + b->n[4];
}

/**
 * Field subtraction: r = a - b + 2*p (to avoid underflow)
 */
static inline void fe_sub(fe_t *r, const fe_t *a, const fe_t *b) {
    /* Add 2*p to avoid underflow */
    r->n[0] = a->n[0] - b->n[0] + 0x1FFFFDFFFFF85EULL;
    r->n[1] = a->n[1] - b->n[1] + 0x1FFFFFFFFFFFFEULL;
    r->n[2] = a->n[2] - b->n[2] + 0x1FFFFFFFFFFFFEULL;
    r->n[3] = a->n[3] - b->n[3] + 0x1FFFFFFFFFFFFEULL;
    r->n[4] = a->n[4] - b->n[4] + 0x1FFFFFFFFFFFFEULL;
}

/**
 * Field negation: r = -a = 2*p - a
 */
static inline void fe_neg(fe_t *r, const fe_t *a) {
    r->n[0] = 0x1FFFFDFFFFF85EULL - a->n[0];
    r->n[1] = 0x1FFFFFFFFFFFFEULL - a->n[1];
    r->n[2] = 0x1FFFFFFFFFFFFEULL - a->n[2];
    r->n[3] = 0x1FFFFFFFFFFFFEULL - a->n[3];
    r->n[4] = 0x1FFFFFFFFFFFFEULL - a->n[4];
}

#endif /* SECP256K1_AVX2_FIELD_MUL_H */


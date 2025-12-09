/**
 * secp256k1 Field Element - 5x52-bit representation
 * 
 * Based on Bitcoin Core's libsecp256k1
 * Copyright (c) 2013, 2014 Pieter Wuille
 * MIT License
 */

#ifndef SECP256K1_AVX2_FIELD_H
#define SECP256K1_AVX2_FIELD_H

#include <stdint.h>
#include <string.h>

/**
 * Field element in 5x52-bit representation.
 * 
 * Represents: sum(i=0..4, n[i] << (i*52)) mod p
 * where p = 2^256 - 0x1000003D1 (secp256k1 field prime)
 * 
 * Limb constraints:
 *   n[0..3]: up to 52 bits (0xFFFFFFFFFFFFF)
 *   n[4]:    up to 48 bits (0x0FFFFFFFFFFFF)
 */
typedef struct {
    uint64_t n[5];
} fe_t;

/**
 * 4-way parallel field elements in limb-sliced layout for AVX2.
 * 
 * Memory layout (for 4 field elements A, B, C, D):
 *   limb[0] = [A.n[0], B.n[0], C.n[0], D.n[0]]  <- fits in one YMM register
 *   limb[1] = [A.n[1], B.n[1], C.n[1], D.n[1]]
 *   limb[2] = [A.n[2], B.n[2], C.n[2], D.n[2]]
 *   limb[3] = [A.n[3], B.n[3], C.n[3], D.n[3]]
 *   limb[4] = [A.n[4], B.n[4], C.n[4], D.n[4]]
 * 
 * This allows vpmuludq to multiply corresponding limbs of 4 elements at once.
 */
typedef struct {
    uint64_t limb[5][4] __attribute__((aligned(32)));
} fe4_t;

/* Constants */
#define FE_LIMB_MASK     0xFFFFFFFFFFFFFULL  /* 52 bits */
#define FE_LIMB4_MASK    0x0FFFFFFFFFFFFULL  /* 48 bits for limb[4] */
#define FE_R             0x1000003D10ULL     /* 2^256 mod p = 0x1000003D1 */

/* Field prime: p = 2^256 - 0x1000003D1 */
static const uint64_t FE_P[5] = {
    0xFFFFEFFFFFC2FULL,  /* 2^52 - 0x1000003D1 */
    0xFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFULL,
    0x0FFFFFFFFFFFFULL
};

/* Initialize field element from 32-byte big-endian */
static inline void fe_set_b32(fe_t *r, const uint8_t *a) {
    r->n[0] = (uint64_t)a[31]
            | ((uint64_t)a[30] << 8)
            | ((uint64_t)a[29] << 16)
            | ((uint64_t)a[28] << 24)
            | ((uint64_t)a[27] << 32)
            | ((uint64_t)a[26] << 40)
            | ((uint64_t)(a[25] & 0xF) << 48);
    r->n[1] = (uint64_t)((a[25] >> 4) & 0xF)
            | ((uint64_t)a[24] << 4)
            | ((uint64_t)a[23] << 12)
            | ((uint64_t)a[22] << 20)
            | ((uint64_t)a[21] << 28)
            | ((uint64_t)a[20] << 36)
            | ((uint64_t)a[19] << 44);
    r->n[2] = (uint64_t)a[18]
            | ((uint64_t)a[17] << 8)
            | ((uint64_t)a[16] << 16)
            | ((uint64_t)a[15] << 24)
            | ((uint64_t)a[14] << 32)
            | ((uint64_t)a[13] << 40)
            | ((uint64_t)(a[12] & 0xF) << 48);
    r->n[3] = (uint64_t)((a[12] >> 4) & 0xF)
            | ((uint64_t)a[11] << 4)
            | ((uint64_t)a[10] << 12)
            | ((uint64_t)a[9] << 20)
            | ((uint64_t)a[8] << 28)
            | ((uint64_t)a[7] << 36)
            | ((uint64_t)a[6] << 44);
    r->n[4] = (uint64_t)a[5]
            | ((uint64_t)a[4] << 8)
            | ((uint64_t)a[3] << 16)
            | ((uint64_t)a[2] << 24)
            | ((uint64_t)a[1] << 32)
            | ((uint64_t)a[0] << 40);
}

/* Convert field element to 32-byte big-endian */
static inline void fe_get_b32(uint8_t *r, const fe_t *a) {
    r[0] = (a->n[4] >> 40) & 0xFF;
    r[1] = (a->n[4] >> 32) & 0xFF;
    r[2] = (a->n[4] >> 24) & 0xFF;
    r[3] = (a->n[4] >> 16) & 0xFF;
    r[4] = (a->n[4] >> 8) & 0xFF;
    r[5] = a->n[4] & 0xFF;
    r[6] = (a->n[3] >> 44) & 0xFF;
    r[7] = (a->n[3] >> 36) & 0xFF;
    r[8] = (a->n[3] >> 28) & 0xFF;
    r[9] = (a->n[3] >> 20) & 0xFF;
    r[10] = (a->n[3] >> 12) & 0xFF;
    r[11] = (a->n[3] >> 4) & 0xFF;
    r[12] = ((a->n[2] >> 48) & 0xF) | ((a->n[3] & 0xF) << 4);
    r[13] = (a->n[2] >> 40) & 0xFF;
    r[14] = (a->n[2] >> 32) & 0xFF;
    r[15] = (a->n[2] >> 24) & 0xFF;
    r[16] = (a->n[2] >> 16) & 0xFF;
    r[17] = (a->n[2] >> 8) & 0xFF;
    r[18] = a->n[2] & 0xFF;
    r[19] = (a->n[1] >> 44) & 0xFF;
    r[20] = (a->n[1] >> 36) & 0xFF;
    r[21] = (a->n[1] >> 28) & 0xFF;
    r[22] = (a->n[1] >> 20) & 0xFF;
    r[23] = (a->n[1] >> 12) & 0xFF;
    r[24] = (a->n[1] >> 4) & 0xFF;
    r[25] = ((a->n[0] >> 48) & 0xF) | ((a->n[1] & 0xF) << 4);
    r[26] = (a->n[0] >> 40) & 0xFF;
    r[27] = (a->n[0] >> 32) & 0xFF;
    r[28] = (a->n[0] >> 24) & 0xFF;
    r[29] = (a->n[0] >> 16) & 0xFF;
    r[30] = (a->n[0] >> 8) & 0xFF;
    r[31] = a->n[0] & 0xFF;
}

/* Pack 4 scalar field elements into limb-sliced layout */
static inline void fe4_pack(fe4_t *r, const fe_t *a, const fe_t *b, const fe_t *c, const fe_t *d) {
    for (int i = 0; i < 5; i++) {
        r->limb[i][0] = a->n[i];
        r->limb[i][1] = b->n[i];
        r->limb[i][2] = c->n[i];
        r->limb[i][3] = d->n[i];
    }
}

/* Unpack limb-sliced layout to 4 scalar field elements */
static inline void fe4_unpack(fe_t *a, fe_t *b, fe_t *c, fe_t *d, const fe4_t *r) {
    for (int i = 0; i < 5; i++) {
        a->n[i] = r->limb[i][0];
        b->n[i] = r->limb[i][1];
        c->n[i] = r->limb[i][2];
        d->n[i] = r->limb[i][3];
    }
}

#endif /* SECP256K1_AVX2_FIELD_H */


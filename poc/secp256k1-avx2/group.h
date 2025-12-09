/**
 * secp256k1 Group (Elliptic Curve Point) Operations
 * 
 * Points are represented in Jacobian coordinates (X, Y, Z) where:
 *   affine (x, y) = (X/Z^2, Y/Z^3)
 * 
 * This allows point addition without expensive field inversion.
 */

#ifndef SECP256K1_AVX2_GROUP_H
#define SECP256K1_AVX2_GROUP_H

#include "field.h"

/**
 * Affine point (x, y) on the curve y^2 = x^3 + 7
 */
typedef struct {
    fe_t x;
    fe_t y;
    int infinity;  /* 1 if point at infinity, 0 otherwise */
} ge_t;

/**
 * Jacobian point (X, Y, Z) where affine (x, y) = (X/Z^2, Y/Z^3)
 */
typedef struct {
    fe_t x;
    fe_t y;
    fe_t z;
    int infinity;
} gej_t;

/**
 * 4-way parallel affine points (limb-sliced)
 */
typedef struct {
    fe4_t x;
    fe4_t y;
    int infinity[4];
} ge4_t;

/**
 * 4-way parallel Jacobian points (limb-sliced)
 */
typedef struct {
    fe4_t x;
    fe4_t y;
    fe4_t z;
    int infinity[4];
} gej4_t;

/* ========== secp256k1 Generator Point G ========== */

/*
 * G.x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
 * G.y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
 * 
 * In 5x52-bit representation:
 */
static const fe_t GEN_X = {{
    0x59F2815B16F817ULL,   /* limb 0 */
    0x029BFCDB2DCE28DULL,  /* limb 1 */
    0xCE870B07ULL,         /* limb 2 - needs recalc */
    0x55A06295ULL,         /* limb 3 - needs recalc */  
    0x79BE667EF9DCBBULL    /* limb 4 - needs recalc */
}};

static const fe_t GEN_Y = {{
    0x9C47D08FFB10D4ULL,   /* limb 0 */
    0xFD17B448A685541ULL,  /* limb 1 */
    0x0E1108A8ULL,         /* limb 2 - needs recalc */
    0x5DA4FBFCULL,         /* limb 3 - needs recalc */
    0x483ADA7726A3C4ULL    /* limb 4 - needs recalc */
}};

/**
 * Initialize generator point from hex constants
 * (Proper parsing - the above constants need to be recalculated)
 */
static inline void ge_set_generator(ge_t *g) {
    /* G.x in big-endian bytes */
    static const uint8_t GX[32] = {
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
        0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
        0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
        0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
    };
    /* G.y in big-endian bytes */
    static const uint8_t GY[32] = {
        0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65,
        0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
        0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19,
        0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8
    };
    
    fe_set_b32(&g->x, GX);
    fe_set_b32(&g->y, GY);
    g->infinity = 0;
}

/**
 * Set Jacobian point from affine point
 */
static inline void gej_set_ge(gej_t *r, const ge_t *a) {
    r->x = a->x;
    r->y = a->y;
    /* Z = 1 */
    r->z.n[0] = 1;
    r->z.n[1] = 0;
    r->z.n[2] = 0;
    r->z.n[3] = 0;
    r->z.n[4] = 0;
    r->infinity = a->infinity;
}

/**
 * Pack 4 Jacobian points into limb-sliced layout
 */
static inline void gej4_pack(gej4_t *r, const gej_t *a, const gej_t *b, 
                             const gej_t *c, const gej_t *d) {
    fe4_pack(&r->x, &a->x, &b->x, &c->x, &d->x);
    fe4_pack(&r->y, &a->y, &b->y, &c->y, &d->y);
    fe4_pack(&r->z, &a->z, &b->z, &c->z, &d->z);
    r->infinity[0] = a->infinity;
    r->infinity[1] = b->infinity;
    r->infinity[2] = c->infinity;
    r->infinity[3] = d->infinity;
}

/**
 * Unpack limb-sliced layout to 4 Jacobian points
 */
static inline void gej4_unpack(gej_t *a, gej_t *b, gej_t *c, gej_t *d,
                               const gej4_t *r) {
    fe4_unpack(&a->x, &b->x, &c->x, &d->x, &r->x);
    fe4_unpack(&a->y, &b->y, &c->y, &d->y, &r->y);
    fe4_unpack(&a->z, &b->z, &c->z, &d->z, &r->z);
    a->infinity = r->infinity[0];
    b->infinity = r->infinity[1];
    c->infinity = r->infinity[2];
    d->infinity = r->infinity[3];
}

#endif /* SECP256K1_AVX2_GROUP_H */


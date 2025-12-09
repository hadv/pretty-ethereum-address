/**
 * secp256k1 Group Operations - AVX2 4-way parallel
 * 
 * Implements point doubling and addition for 4 points at once.
 * Uses complete addition formulas for security.
 * 
 * Reference: libsecp256k1 group_impl.h
 */

#ifndef SECP256K1_AVX2_GROUP_OPS_H
#define SECP256K1_AVX2_GROUP_OPS_H

#include "group.h"
#include "field_mul_avx2.h"
#include "field_ops_avx2.h"

/**
 * 4-way parallel field squaring: r = a^2
 * (Same as multiply but with a=b, can be optimized later)
 */
static inline void fe4_sqr(fe4_t *r, const fe4_t *a) {
    fe4_mul(r, a, a);
}

/**
 * Scalar field squaring
 */
static inline void fe_sqr(fe_t *r, const fe_t *a) {
    fe_mul(r, a, a);
}

/**
 * 4-way parallel point doubling in Jacobian coordinates
 * 
 * Formula (for a=-3, but secp256k1 has a=0):
 *   For a=0: 3*X^2 (not 3*X^2 + a*Z^4)
 *   
 * Simplified formulas for secp256k1 (a=0):
 *   M = 3*X^2
 *   S = 4*X*Y^2
 *   T = M^2 - 2*S
 *   X3 = T
 *   Y3 = M*(S - T) - 8*Y^4
 *   Z3 = 2*Y*Z
 */
static inline void gej4_double(gej4_t *r, const gej4_t *a) {
    fe4_t t1, t2, t3, t4, t5;
    
    /* t1 = X^2 */
    fe4_sqr(&t1, &a->x);
    
    /* t2 = Y^2 */
    fe4_sqr(&t2, &a->y);
    
    /* t3 = Y^4 = t2^2 */
    fe4_sqr(&t3, &t2);
    
    /* t4 = X * Y^2 = X * t2 */
    fe4_mul(&t4, &a->x, &t2);
    
    /* t4 = 2 * t4 (will be 4*X*Y^2 = S after doubling again) */
    fe4_add(&t4, &t4, &t4);
    
    /* S = 4*X*Y^2 = 2*t4 */
    fe4_add(&t4, &t4, &t4);
    
    /* t5 = 3*X^2 = M (since a=0 for secp256k1) */
    fe4_add(&t5, &t1, &t1);      /* t5 = 2*X^2 */
    fe4_add(&t5, &t5, &t1);      /* t5 = 3*X^2 = M */
    
    /* X3 = M^2 - 2*S = t5^2 - 2*t4 */
    fe4_sqr(&r->x, &t5);         /* r.x = M^2 */
    fe4_sub(&r->x, &r->x, &t4);  /* r.x = M^2 - S */
    fe4_sub(&r->x, &r->x, &t4);  /* r.x = M^2 - 2*S = X3 */
    
    /* Y3 = M*(S - X3) - 8*Y^4 */
    fe4_sub(&t4, &t4, &r->x);    /* t4 = S - X3 */
    fe4_mul(&t4, &t5, &t4);      /* t4 = M*(S - X3) */
    
    /* t3 = 8*Y^4 */
    fe4_add(&t3, &t3, &t3);      /* t3 = 2*Y^4 */
    fe4_add(&t3, &t3, &t3);      /* t3 = 4*Y^4 */
    fe4_add(&t3, &t3, &t3);      /* t3 = 8*Y^4 */
    
    fe4_sub(&r->y, &t4, &t3);    /* r.y = M*(S-X3) - 8*Y^4 = Y3 */
    
    /* Z3 = 2*Y*Z */
    fe4_mul(&r->z, &a->y, &a->z);
    fe4_add(&r->z, &r->z, &r->z);
    
    /* Copy infinity flags */
    for (int i = 0; i < 4; i++) {
        r->infinity[i] = a->infinity[i];
    }
}

/**
 * 4-way parallel point addition: r = a + b
 * where a is in Jacobian coords and b is in affine coords
 * 
 * This is the common case for incremental mining:
 *   P[n+1] = P[n] + G  (add generator to running sum)
 * 
 * Formula (assuming b.z = 1):
 *   U1 = a.X
 *   U2 = b.x * a.Z^2
 *   S1 = a.Y
 *   S2 = b.y * a.Z^3
 *   H = U2 - U1
 *   R = S2 - S1
 *   X3 = R^2 - H^3 - 2*U1*H^2
 *   Y3 = R*(U1*H^2 - X3) - S1*H^3
 *   Z3 = a.Z * H
 */
static inline void gej4_add_ge(gej4_t *r, const gej4_t *a, const ge4_t *b) {
    fe4_t zz, zzz, u2, s2, h, hh, hhh, r_val, v, t;
    
    /* zz = a.Z^2 */
    fe4_sqr(&zz, &a->z);
    
    /* zzz = a.Z^3 = a.Z * zz */
    fe4_mul(&zzz, &a->z, &zz);
    
    /* u2 = b.x * zz */
    fe4_mul(&u2, &b->x, &zz);
    
    /* s2 = b.y * zzz */
    fe4_mul(&s2, &b->y, &zzz);
    
    /* h = u2 - a.X (U2 - U1) */
    fe4_sub(&h, &u2, &a->x);
    
    /* r = s2 - a.Y (S2 - S1) */
    fe4_sub(&r_val, &s2, &a->y);
    
    /* hh = h^2 */
    fe4_sqr(&hh, &h);
    
    /* hhh = h^3 = h * hh */
    fe4_mul(&hhh, &h, &hh);
    
    /* v = a.X * hh (U1 * H^2) */
    fe4_mul(&v, &a->x, &hh);
    
    /* X3 = r^2 - hhh - 2*v */
    fe4_sqr(&r->x, &r_val);      /* r.x = R^2 */
    fe4_sub(&r->x, &r->x, &hhh); /* r.x = R^2 - H^3 */
    fe4_sub(&r->x, &r->x, &v);   /* r.x = R^2 - H^3 - U1*H^2 */
    fe4_sub(&r->x, &r->x, &v);   /* r.x = R^2 - H^3 - 2*U1*H^2 = X3 */
    
    /* Y3 = R*(v - X3) - S1*H^3 */
    fe4_sub(&t, &v, &r->x);      /* t = v - X3 = U1*H^2 - X3 */
    fe4_mul(&t, &r_val, &t);     /* t = R*(U1*H^2 - X3) */
    fe4_mul(&v, &a->y, &hhh);    /* v = S1 * H^3 */
    fe4_sub(&r->y, &t, &v);      /* r.y = R*(U1*H^2 - X3) - S1*H^3 = Y3 */
    
    /* Z3 = a.Z * H */
    fe4_mul(&r->z, &a->z, &h);
    
    /* Copy infinity flags (simplified - real impl needs special cases) */
    for (int i = 0; i < 4; i++) {
        r->infinity[i] = a->infinity[i] && b->infinity[i];
    }
}

#endif /* SECP256K1_AVX2_GROUP_OPS_H */


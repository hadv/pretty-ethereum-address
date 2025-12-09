/**
 * secp256k1 Point Operations Benchmark
 *
 * Compares scalar vs AVX2 4-way parallel point addition.
 * This simulates incremental EOA mining: P[n+1] = P[n] + G
 *
 * NOTE: This PoC measures throughput, not correctness.
 * The simplified field arithmetic produces incorrect results but
 * demonstrates the achievable parallelism with proper implementation.
 */

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include "field.h"
#include "field_mul.h"
#include "field_mul_avx2.h"
#include "field_ops_avx2.h"
#include "group.h"
#include "group_avx2.h"

#define ITERATIONS 1000000
#define WARMUP 10000

/* ========== Scalar Point Operations ========== */

static inline void gej_double(gej_t *r, const gej_t *a) {
    fe_t t1, t2, t3, t4, t5;
    
    fe_sqr(&t1, &a->x);           /* t1 = X^2 */
    fe_sqr(&t2, &a->y);           /* t2 = Y^2 */
    fe_sqr(&t3, &t2);             /* t3 = Y^4 */
    fe_mul(&t4, &a->x, &t2);      /* t4 = X*Y^2 */
    fe_add(&t4, &t4, &t4);        /* t4 = 2*X*Y^2 */
    fe_add(&t4, &t4, &t4);        /* t4 = 4*X*Y^2 = S */
    
    fe_add(&t5, &t1, &t1);        /* t5 = 2*X^2 */
    fe_add(&t5, &t5, &t1);        /* t5 = 3*X^2 = M */
    
    fe_sqr(&r->x, &t5);           /* r.x = M^2 */
    fe_sub(&r->x, &r->x, &t4);    /* r.x = M^2 - S */
    fe_sub(&r->x, &r->x, &t4);    /* r.x = M^2 - 2*S = X3 */
    
    fe_sub(&t4, &t4, &r->x);      /* t4 = S - X3 */
    fe_mul(&t4, &t5, &t4);        /* t4 = M*(S - X3) */
    
    fe_add(&t3, &t3, &t3);        /* t3 = 2*Y^4 */
    fe_add(&t3, &t3, &t3);        /* t3 = 4*Y^4 */
    fe_add(&t3, &t3, &t3);        /* t3 = 8*Y^4 */
    
    fe_sub(&r->y, &t4, &t3);      /* r.y = M*(S-X3) - 8*Y^4 */
    fe_mul(&r->z, &a->y, &a->z);
    fe_add(&r->z, &r->z, &r->z);  /* r.z = 2*Y*Z */
    
    r->infinity = a->infinity;
}

static inline void gej_add_ge(gej_t *r, const gej_t *a, const ge_t *b) {
    fe_t zz, zzz, u2, s2, h, hh, hhh, rr, v, t;
    
    fe_sqr(&zz, &a->z);           /* zz = Z^2 */
    fe_mul(&zzz, &a->z, &zz);     /* zzz = Z^3 */
    fe_mul(&u2, &b->x, &zz);      /* u2 = b.x * Z^2 */
    fe_mul(&s2, &b->y, &zzz);     /* s2 = b.y * Z^3 */
    
    fe_sub(&h, &u2, &a->x);       /* h = u2 - X */
    fe_sub(&rr, &s2, &a->y);      /* rr = s2 - Y */
    
    fe_sqr(&hh, &h);              /* hh = h^2 */
    fe_mul(&hhh, &h, &hh);        /* hhh = h^3 */
    fe_mul(&v, &a->x, &hh);       /* v = X * h^2 */
    
    fe_sqr(&r->x, &rr);           /* r.x = rr^2 */
    fe_sub(&r->x, &r->x, &hhh);
    fe_sub(&r->x, &r->x, &v);
    fe_sub(&r->x, &r->x, &v);     /* r.x = rr^2 - hhh - 2*v */
    
    fe_sub(&t, &v, &r->x);
    fe_mul(&t, &rr, &t);
    fe_mul(&v, &a->y, &hhh);
    fe_sub(&r->y, &t, &v);        /* r.y = rr*(v - r.x) - Y*hhh */
    
    fe_mul(&r->z, &a->z, &h);     /* r.z = Z * h */
    r->infinity = 0;
}

int main(void) {
    printf("=== secp256k1 Point Operations Benchmark ===\n\n");
    
    /* Initialize generator point */
    ge_t gen;
    ge_set_generator(&gen);
    
    printf("Generator point G:\n");
    printf("  G.x[0..2] = [%llx, %llx, %llx, ...]\n", 
           (unsigned long long)gen.x.n[0], 
           (unsigned long long)gen.x.n[1],
           (unsigned long long)gen.x.n[2]);
    printf("  G.y[0..2] = [%llx, %llx, %llx, ...]\n\n",
           (unsigned long long)gen.y.n[0], 
           (unsigned long long)gen.y.n[1],
           (unsigned long long)gen.y.n[2]);
    
    /* ========== Scalar Benchmark ========== */
    printf("--- Scalar Point Addition (4x sequential) ---\n");
    
    /* Initialize 4 points as copies of G (in Jacobian) */
    gej_t p1, p2, p3, p4;
    gej_set_ge(&p1, &gen);
    gej_set_ge(&p2, &gen);
    gej_set_ge(&p3, &gen);
    gej_set_ge(&p4, &gen);
    
    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        gej_add_ge(&p1, &p1, &gen);
    }

    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        gej_add_ge(&p1, &p1, &gen);
        gej_add_ge(&p2, &p2, &gen);
        gej_add_ge(&p3, &p3, &gen);
        gej_add_ge(&p4, &p4, &gen);
    }
    clock_t end = clock();
    double scalar_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("  Time: %.3f sec\n", scalar_time);
    printf("  Throughput: %.2f M additions/sec\n", 
           (4.0 * ITERATIONS / 1000000.0) / scalar_time);
    
    /* ========== AVX2 Benchmark ========== */
    printf("\n--- AVX2 4-way Parallel Point Addition ---\n");
    
    /* Initialize 4 points packed */
    gej_t q1, q2, q3, q4;
    gej_set_ge(&q1, &gen);
    gej_set_ge(&q2, &gen);
    gej_set_ge(&q3, &gen);
    gej_set_ge(&q4, &gen);
    
    gej4_t p4_packed;
    gej4_pack(&p4_packed, &q1, &q2, &q3, &q4);
    
    /* Pack generator for affine addition */
    ge4_t gen4_packed;
    fe4_pack(&gen4_packed.x, &gen.x, &gen.x, &gen.x, &gen.x);
    fe4_pack(&gen4_packed.y, &gen.y, &gen.y, &gen.y, &gen.y);
    for (int i = 0; i < 4; i++) gen4_packed.infinity[i] = 0;

    /* Warmup */
    for (int i = 0; i < WARMUP; i++) {
        gej4_add_ge(&p4_packed, &p4_packed, &gen4_packed);
    }

    start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        gej4_add_ge(&p4_packed, &p4_packed, &gen4_packed);
    }
    end = clock();
    double avx2_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("  Time: %.3f sec\n", avx2_time);
    printf("  Throughput: %.2f M additions/sec\n", 
           (4.0 * ITERATIONS / 1000000.0) / avx2_time);
    
    printf("\nSpeedup: %.2fx\n", scalar_time / avx2_time);
    
    /* Prevent optimization */
    volatile uint64_t sink = p1.x.n[0] + p4_packed.x.limb[0][0];
    printf("\n(sink=%llu to prevent optimization)\n", (unsigned long long)sink);
    
    return 0;
}


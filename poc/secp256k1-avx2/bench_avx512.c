/**
 * AVX-512 IFMA Benchmark (requires AVX-512 IFMA support)
 * Compile: gcc -O3 -mavx512f -mavx512ifma -o bench_avx512 bench_avx512.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <cpuid.h>

#include "field.h"
#include "field_mul.h"

/* Check CPU features */
static int check_avx512_ifma(void) {
    unsigned int eax, ebx, ecx, edx;
    
    /* Check for AVX-512F (leaf 7, ebx bit 16) */
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return 0;
    
    int has_avx512f = (ebx >> 16) & 1;
    int has_avx512ifma = (ebx >> 21) & 1;
    
    printf("AVX-512F:    %s\n", has_avx512f ? "YES" : "NO");
    printf("AVX-512IFMA: %s\n", has_avx512ifma ? "YES" : "NO");
    
    return has_avx512f && has_avx512ifma;
}

#ifdef __AVX512F__
#ifdef __AVX512IFMA__
#include "field_mul_avx512.h"

#define ITERATIONS 10000000
#define WARMUP     1000000

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static void fe_random(fe_t *r, uint64_t seed) {
    r->n[0] = (seed * 0x123456789ABCDEFULL) & 0xFFFFFFFFFFFFFULL;
    r->n[1] = (seed * 0xFEDCBA987654321ULL) & 0xFFFFFFFFFFFFFULL;
    r->n[2] = (seed * 0xABCDEF0123456789ULL) & 0xFFFFFFFFFFFFFULL;
    r->n[3] = (seed * 0x9876543210FEDCBAULL) & 0xFFFFFFFFFFFFFULL;
    r->n[4] = (seed * 0x1234FEDC5678ABCDULL) & 0x0FFFFFFFFFFFFULL;
}

/* Pack 8 scalar field elements into fe8_t */
static void fe8_pack(fe8_t *r, const fe_t fe[8]) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 8; j++) {
            r->limb[i][j] = fe[j].n[i];
        }
    }
}

int main(void) {
    printf("=== secp256k1 AVX-512 IFMA Benchmark ===\n\n");
    
    if (!check_avx512_ifma()) {
        printf("\nERROR: This CPU does not support AVX-512 IFMA\n");
        printf("Run on Intel Ice Lake/Tiger Lake/Alder Lake or AMD Zen 4\n");
        return 1;
    }
    
    /* Initialize test data */
    fe_t a[8], b[8], r_scalar[8];
    fe8_t a8, b8, r8;
    
    for (int i = 0; i < 8; i++) {
        fe_random(&a[i], i + 1);
        fe_random(&b[i], i + 100);
    }
    
    fe8_pack(&a8, a);
    fe8_pack(&b8, b);
    
    printf("\n--- Benchmark (%d iterations) ---\n", ITERATIONS);
    
    uint64_t start, end;
    volatile uint64_t sink = 0;
    
    /* Warmup scalar */
    for (int iter = 0; iter < WARMUP; iter++) {
        for (int i = 0; i < 8; i++) {
            fe_mul(&r_scalar[i], &a[i], &b[i]);
            sink += r_scalar[i].n[0];
        }
    }
    
    /* Benchmark scalar (8 sequential) */
    start = get_time_ns();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < 8; i++) {
            fe_mul(&r_scalar[i], &a[i], &b[i]);
        }
        sink += r_scalar[0].n[0];
    }
    end = get_time_ns();
    
    double scalar_time = (double)(end - start) / 1e9;
    double scalar_ops = (double)ITERATIONS * 8 / scalar_time;
    printf("Scalar (8x sequential):   %.3f sec, %.2f M mul/sec\n", scalar_time, scalar_ops / 1e6);
    
    /* Warmup AVX-512 IFMA */
    for (int iter = 0; iter < WARMUP; iter++) {
        fe8_mul_ifma(&r8, &a8, &b8);
        sink += r8.limb[0][0];
    }
    
    /* Benchmark AVX-512 IFMA (1 call = 8 multiplications) */
    start = get_time_ns();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        fe8_mul_ifma(&r8, &a8, &b8);
        sink += r8.limb[0][0];
    }
    end = get_time_ns();
    
    double avx512_time = (double)(end - start) / 1e9;
    double avx512_ops = (double)ITERATIONS * 8 / avx512_time;
    printf("AVX-512 IFMA (8-way):     %.3f sec, %.2f M mul/sec\n", avx512_time, avx512_ops / 1e6);
    
    printf("\nSpeedup: %.2fx\n", avx512_ops / scalar_ops);
    printf("(sink=%llu)\n", (unsigned long long)sink);
    
    return 0;
}

#else
int main(void) {
    printf("Compiled without AVX-512 IFMA support.\n");
    printf("Recompile with: -mavx512f -mavx512ifma\n");
    check_avx512_ifma();
    return 1;
}
#endif
#else
int main(void) {
    printf("Compiled without AVX-512F support.\n");
    printf("Recompile with: -mavx512f -mavx512ifma\n");
    check_avx512_ifma();
    return 1;
}
#endif

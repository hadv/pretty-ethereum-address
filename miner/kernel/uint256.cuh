/*
 * 256-bit Unsigned Integer Arithmetic for CUDA
 * Optimized for secp256k1 operations
 *
 * Uses 4 x 64-bit limbs in little-endian order (limbs[0] is LSB)
 */

#ifndef UINT256_CUH
#define UINT256_CUH

#include <stdint.h>

typedef unsigned char uchar;
typedef uint64_t u64;
typedef unsigned __int128 u128;

// 256-bit unsigned integer
typedef struct {
    u64 limbs[4];  // Little-endian: limbs[0] is LSB
} uint256;

// secp256k1 prime: p = 2^256 - 2^32 - 977
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
__constant__ uint256 SECP256K1_P = {{
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
}};

// secp256k1 order: n
// n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
__constant__ uint256 SECP256K1_N = {{
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
}};

// Zero constant
__constant__ uint256 UINT256_ZERO = {{ 0ULL, 0ULL, 0ULL, 0ULL }};

// One constant
__constant__ uint256 UINT256_ONE = {{ 1ULL, 0ULL, 0ULL, 0ULL }};

// ============================================================================
// Comparison Operations
// ============================================================================

// Compare a and b: returns 1 if a > b, -1 if a < b, 0 if equal
__device__ __forceinline__ int uint256_cmp(const uint256* a, const uint256* b) {
    #pragma unroll
    for (int i = 3; i >= 0; i--) {
        if (a->limbs[i] > b->limbs[i]) return 1;
        if (a->limbs[i] < b->limbs[i]) return -1;
    }
    return 0;
}

// Check if a >= b
__device__ __forceinline__ int uint256_gte(const uint256* a, const uint256* b) {
    return uint256_cmp(a, b) >= 0;
}

// Check if a is zero
__device__ __forceinline__ int uint256_is_zero(const uint256* a) {
    return (a->limbs[0] | a->limbs[1] | a->limbs[2] | a->limbs[3]) == 0;
}

// ============================================================================
// Addition and Subtraction
// ============================================================================

// Add a + b, return carry (0 or 1)
__device__ __forceinline__ u64 uint256_add(uint256* r, const uint256* a, const uint256* b) {
    u128 sum;
    u64 carry = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        sum = (u128)a->limbs[i] + (u128)b->limbs[i] + carry;
        r->limbs[i] = (u64)sum;
        carry = (u64)(sum >> 64);
    }
    return carry;
}

// Subtract a - b, return borrow (0 or 1)
__device__ __forceinline__ u64 uint256_sub(uint256* r, const uint256* a, const uint256* b) {
    u64 borrow = 0;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        u64 diff = a->limbs[i] - b->limbs[i] - borrow;
        borrow = (a->limbs[i] < b->limbs[i] + borrow) ? 1 : 0;
        r->limbs[i] = diff;
    }
    return borrow;
}

// ============================================================================
// Modular Operations for secp256k1 prime p
// ============================================================================

// Reduce modulo p: r = a mod p
__device__ __forceinline__ void uint256_mod_p(uint256* r, const uint256* a) {
    uint256 tmp;
    if (uint256_gte(a, &SECP256K1_P)) {
        uint256_sub(&tmp, a, &SECP256K1_P);
        *r = tmp;
    } else {
        *r = *a;
    }
}

// Modular addition: r = (a + b) mod p
__device__ __forceinline__ void uint256_add_mod_p(uint256* r, const uint256* a, const uint256* b) {
    uint256 sum;
    u64 carry = uint256_add(&sum, a, b);

    // If carry or sum >= p, subtract p
    if (carry || uint256_gte(&sum, &SECP256K1_P)) {
        uint256_sub(r, &sum, &SECP256K1_P);
    } else {
        *r = sum;
    }
}

// Modular subtraction: r = (a - b) mod p
__device__ __forceinline__ void uint256_sub_mod_p(uint256* r, const uint256* a, const uint256* b) {
    uint256 diff;
    u64 borrow = uint256_sub(&diff, a, b);

    // If borrow, add p
    if (borrow) {
        uint256_add(r, &diff, &SECP256K1_P);
    } else {
        *r = diff;
    }
}

// Modular negation: r = (-a) mod p = (p - a) mod p
__device__ __forceinline__ void uint256_neg_mod_p(uint256* r, const uint256* a) {
    if (uint256_is_zero(a)) {
        *r = UINT256_ZERO;
    } else {
        uint256_sub(r, &SECP256K1_P, a);
    }
}

#endif // UINT256_CUH


/*
 * 256-bit Unsigned Integer Arithmetic for CUDA - Multiplication and Inversion
 * Optimized for secp256k1 operations
 */

#include "uint256.cuh"

// ============================================================================
// 512-bit intermediate for multiplication
// ============================================================================

typedef struct {
    u64 limbs[8];
} uint512;

// ============================================================================
// 256-bit Multiplication (produces 512-bit result)
// ============================================================================

// Multiply two 256-bit numbers to get 512-bit result
__device__ __noinline__ void uint256_mul_full(uint512* r, const uint256* a, const uint256* b) {
    u128 prod;
    u64 carry;

    // Initialize result to zero
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->limbs[i] = 0;
    }

    // Schoolbook multiplication with carry propagation
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            prod = (u128)a->limbs[i] * (u128)b->limbs[j] + r->limbs[i+j] + carry;
            r->limbs[i+j] = (u64)prod;
            carry = (u64)(prod >> 64);
        }
        r->limbs[i+4] = carry;
    }
}

// ============================================================================
// Fast Reduction for secp256k1 Prime
// ============================================================================

// Reduce 512-bit number modulo secp256k1 prime p
// p = 2^256 - 2^32 - 977
// Uses: a mod p = a_low + a_high * (2^32 + 977)
__device__ __noinline__ void uint512_reduce_p(uint256* r, const uint512* a) {
    // Split into low 256 bits and high 256 bits
    uint256 low, high;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        low.limbs[i] = a->limbs[i];
        high.limbs[i] = a->limbs[i + 4];
    }

    // If high is zero, just copy low and reduce if needed
    if (uint256_is_zero(&high)) {
        uint256_mod_p(r, &low);
        return;
    }

    // Multiply high by (2^32 + 977) and add to low
    // 2^32 + 977 = 0x100000000 + 0x3D1 = 0x1000003D1
    const u64 k = 0x1000003D1ULL;

    uint256 product;
    u128 prod;
    u64 carry = 0;

    // Multiply high by k
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        prod = (u128)high.limbs[i] * k + carry;
        product.limbs[i] = (u64)prod;
        carry = (u64)(prod >> 64);
    }

    // Add product to low
    uint256 sum;
    u64 c = uint256_add(&sum, &low, &product);

    // Handle remaining carry
    c += carry;

    // If there's still overflow, multiply carry by k and add again
    while (c > 0) {
        u64 extra = c * k;
        u128 s = (u128)sum.limbs[0] + extra;
        sum.limbs[0] = (u64)s;
        c = (u64)(s >> 64);
        
        #pragma unroll
        for (int i = 1; i < 4 && c > 0; i++) {
            s = (u128)sum.limbs[i] + c;
            sum.limbs[i] = (u64)s;
            c = (u64)(s >> 64);
        }
    }

    // Final reduction if sum >= p
    uint256_mod_p(r, &sum);
}

// ============================================================================
// Modular Multiplication
// ============================================================================

// Modular multiplication: r = (a * b) mod p
__device__ __noinline__ void uint256_mul_mod_p(uint256* r, const uint256* a, const uint256* b) {
    uint512 product;
    uint256_mul_full(&product, a, b);
    uint512_reduce_p(r, &product);
}

// Modular squaring: r = (a * a) mod p (slightly faster than mul)
__device__ __noinline__ void uint256_sqr_mod_p(uint256* r, const uint256* a) {
    uint256_mul_mod_p(r, a, a);
}

// ============================================================================
// Modular Inversion using Fermat's Little Theorem
// ============================================================================

// a^(-1) mod p = a^(p-2) mod p
// p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
__device__ __noinline__ void uint256_inv_mod_p(uint256* r, const uint256* a) {
    // Use square-and-multiply for a^(p-2)
    uint256 base = *a;
    uint256 result = UINT256_ONE;
    uint256 tmp;

    // p-2 in binary (process from LSB)
    const u64 exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };

    // NOTE: Do NOT use #pragma unroll here - 256 iterations would hang the compiler
    for (int w = 0; w < 4; w++) {
        u64 bits = exp[w];
        for (int i = 0; i < 64; i++) {
            if (bits & 1) {
                uint256_mul_mod_p(&tmp, &result, &base);
                result = tmp;
            }
            uint256_sqr_mod_p(&tmp, &base);
            base = tmp;
            bits >>= 1;
        }
    }

    *r = result;
}


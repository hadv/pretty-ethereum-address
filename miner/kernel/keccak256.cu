/*
 * CUDA Keccak256 Kernel for CREATE2 Vanity Address Mining
 * Optimized for NVIDIA GPUs (Compute Capability 5.0+)
 *
 * Keccak256 implementation based on kale-miner by Fred Kyung-jin Rezeau
 * Reference: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
 */

#include <stdint.h>

typedef unsigned char uchar;
typedef uint64_t u64;

typedef struct {
    uchar state[200];
    int offset;
} Keccak256Context;

__constant__ u64 roundConstants[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

__constant__ int rhoOffsets[24] = {
    1, 3, 6, 10, 15, 21, 28, 36,
    45, 55, 2, 14, 27, 41, 56,
    8, 25, 43, 62, 18, 39, 61,
    20, 44
};

__constant__ int piIndexes[24] = {
    10, 7, 11, 17, 18, 3, 5, 16,
    8, 21, 24, 4, 15, 23, 19, 13,
    12, 2, 20, 14, 22, 9, 6, 1
};

__device__ u64 rotl64(u64 x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ void keccakF1600(uchar* state) {
    u64 state64[25];

    // Load state - match OpenCL exactly (no unroll)
    for (int i = 0; i < 25; ++i) {
        state64[i] = 0;
        for (int j = 0; j < 8; ++j) {
            state64[i] |= ((u64)state[i * 8 + j]) << (8 * j);
        }
    }

    for (int round = 0; round < 24; ++round) {
        u64 C[5], D[5];

        for (int x = 0; x < 5; ++x)
            C[x] = state64[x] ^ state64[x + 5] ^ state64[x + 10] ^ state64[x + 15] ^ state64[x + 20];

        for (int x = 0; x < 5; ++x) {
            D[x] = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 25; y += 5)
                state64[y + x] ^= D[x];
        }

        u64 temp = state64[1];
        for (int i = 0; i < 24; ++i) {
            int index = piIndexes[i];
            u64 t = state64[index];
            state64[index] = rotl64(temp, rhoOffsets[i]);
            temp = t;
        }

        for (int y = 0; y < 25; y += 5) {
            u64 tempVars[5];
            for (int x = 0; x < 5; ++x)
                tempVars[x] = state64[y + x];
            for (int x = 0; x < 5; ++x)
                state64[y + x] = tempVars[x] ^ ((~tempVars[(x + 1) % 5]) & tempVars[(x + 2) % 5]);
        }

        state64[0] ^= roundConstants[round];
    }

    for (int i = 0; i < 25; ++i) {
        for (int j = 0; j < 8; ++j) {
            state[i * 8 + j] = (uchar)((state64[i] >> (8 * j)) & 0xFF);
        }
    }
}

__device__ void keccak256Reset(Keccak256Context* ctx) {
    for (int i = 0; i < 200; ++i) {
        ctx->state[i] = 0;
    }
    ctx->offset = 0;
}

__device__ void keccak256Update(Keccak256Context* ctx, const uchar* data, int len) {
    int rate = 136;
    while (len > 0) {
        int chunk = (len < rate - ctx->offset) ? len : rate - ctx->offset;
        for (int i = 0; i < chunk; ++i) {
            ctx->state[ctx->offset + i] ^= data[i];
        }
        ctx->offset += chunk;
        data += chunk;
        len -= chunk;
        if (ctx->offset == rate) {
            keccakF1600(ctx->state);
            ctx->offset = 0;
        }
    }
}

__device__ void keccak256Finalize(Keccak256Context* ctx, uchar* hash) {
    ctx->state[ctx->offset] ^= 0x01;
    ctx->state[135] ^= 0x80;
    keccakF1600(ctx->state);
    for (int i = 0; i < 32; ++i) {
        hash[i] = ctx->state[i];
    }
}

__device__ void keccak256(const uchar* input, int size, uchar* output) {
    Keccak256Context ctx;
    keccak256Reset(&ctx);
    keccak256Update(&ctx, input, size);
    keccak256Finalize(&ctx, output);
}

// Convert 64-bit nonce to bytes (big-endian for salt suffix)
__device__ void nonce_to_bytes(u64 nonce, uchar bytes[12]) {
    bytes[0] = 0;
    bytes[1] = 0;
    bytes[2] = 0;
    bytes[3] = 0;
    bytes[4] = (uchar)(nonce >> 56);
    bytes[5] = (uchar)(nonce >> 48);
    bytes[6] = (uchar)(nonce >> 40);
    bytes[7] = (uchar)(nonce >> 32);
    bytes[8] = (uchar)(nonce >> 24);
    bytes[9] = (uchar)(nonce >> 16);
    bytes[10] = (uchar)(nonce >> 8);
    bytes[11] = (uchar)(nonce);
}

/*
 * Main CREATE2 mining kernel
 *
 * Each thread processes one nonce value
 *
 * Parameters:
 *   data_template: 85-byte CREATE2 data template (0xff ++ deployer ++ salt_prefix ++ init_code_hash)
 *                  salt_prefix is 20 bytes, we fill in the remaining 12 bytes with nonce
 *   pattern: The address pattern to match (up to 20 bytes)
 *   pattern_length: Number of bytes in pattern to match
 *   start_nonce: Starting nonce for this batch
 *   result_salt: Output - the 12-byte salt suffix that produces matching address
 *   result_address: Output - the 20-byte address found
 *   found: Atomic flag - set to 1 when a match is found
 */
extern "C" __global__ void mine_create2(
    const uchar * __restrict__ data_template,
    const uchar * __restrict__ pattern,
    int pattern_length,
    u64 start_nonce,
    uchar * __restrict__ result_salt,
    uchar * __restrict__ result_address,
    int *found
) {
    // Check if another thread already found a result
    if (*found) {
        return;
    }

    // Calculate this thread's nonce
    u64 nonce = start_nonce + (u64)(blockIdx.x * blockDim.x + threadIdx.x);

    // Prepare the CREATE2 data buffer (85 bytes)
    uchar data[85];

    // Copy the template
    for (int i = 0; i < 85; i++) {
        data[i] = data_template[i];
    }

    // Fill in the salt suffix (bytes 41-52, which is salt bytes 20-31)
    uchar salt_suffix[12];
    nonce_to_bytes(nonce, salt_suffix);
    for (int i = 0; i < 12; i++) {
        data[41 + i] = salt_suffix[i];
    }

    // Calculate Keccak256 hash
    uchar hash[32];
    keccak256(data, 85, hash);

    // The address is the last 20 bytes of the hash (bytes 12-31)
    // Check if it matches the pattern
    bool match = true;
    for (int i = 0; i < pattern_length; i++) {
        if (hash[12 + i] != pattern[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        // Use atomicCAS for thread-safe first-match detection
        if (atomicCAS(found, 0, 1) == 0) {
            // We are the first to find a match - store the result
            for (int i = 0; i < 12; i++) {
                result_salt[i] = salt_suffix[i];
            }
            for (int i = 0; i < 20; i++) {
                result_address[i] = hash[12 + i];
            }
        }
    }
}


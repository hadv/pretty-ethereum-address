/*
 * CUDA Keccak256 Kernel for CREATE2 Vanity Address Mining
 * Optimized for NVIDIA GPUs (Compute Capability 5.0+)
 *
 * Implements Keccak-f[1600] with full loop unrolling and register-based state.
 */

#include <stdint.h>

typedef unsigned char uchar;
typedef uint64_t u64;

// Round constants
__constant__ u64 RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotate left (circular shift)
#define ROL64(a, offset) ((((u64)a) << offset) ^ (((u64)a) >> (64 - offset)))

// Keccak-f[1600] permutation
// Using __forceinline__ to ensure it's inlined into the kernel
__device__ __forceinline__ void keccak_f1600(u64 *st) {
    u64 bc[5];
    u64 t;

    #pragma unroll
    for (int round = 0; round < 24; ++round) {
        // Theta step
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        t = bc[4] ^ ROL64(bc[1], 1); st[0] ^= t; st[5] ^= t; st[10] ^= t; st[15] ^= t; st[20] ^= t;
        t = bc[0] ^ ROL64(bc[2], 1); st[1] ^= t; st[6] ^= t; st[11] ^= t; st[16] ^= t; st[21] ^= t;
        t = bc[1] ^ ROL64(bc[3], 1); st[2] ^= t; st[7] ^= t; st[12] ^= t; st[17] ^= t; st[22] ^= t;
        t = bc[2] ^ ROL64(bc[4], 1); st[3] ^= t; st[8] ^= t; st[13] ^= t; st[18] ^= t; st[23] ^= t;
        t = bc[3] ^ ROL64(bc[0], 1); st[4] ^= t; st[9] ^= t; st[14] ^= t; st[19] ^= t; st[24] ^= t;

        // Rho and Pi steps
        t = st[1];
        st[1] = ROL64(st[6], 44);
        st[6] = ROL64(st[9], 20);
        st[9] = ROL64(st[22], 61);
        st[22] = ROL64(st[14], 39);
        st[14] = ROL64(st[20], 18);
        st[20] = ROL64(st[2], 62);
        st[2] = ROL64(st[12], 43);
        st[12] = ROL64(st[13], 25);
        st[13] = ROL64(st[19], 8);
        st[19] = ROL64(st[23], 56);
        st[23] = ROL64(st[15], 41);
        st[15] = ROL64(st[4], 27);
        st[4] = ROL64(st[24], 14);
        st[24] = ROL64(st[21], 2);
        st[21] = ROL64(st[8], 55);
        st[8] = ROL64(st[16], 45);
        st[16] = ROL64(st[5], 36);
        st[5] = ROL64(st[3], 28);
        st[3] = ROL64(st[18], 21);
        st[18] = ROL64(st[17], 15);
        st[17] = ROL64(st[11], 10);
        st[11] = ROL64(st[7], 6);
        st[7] = ROL64(st[10], 3);
        st[10] = ROL64(t, 1);

        // Chi step
        bc[0] = st[0]; bc[1] = st[1]; bc[2] = st[2]; bc[3] = st[3]; bc[4] = st[4];
        st[0] = bc[0] ^ (~bc[1] & bc[2]); st[1] = bc[1] ^ (~bc[2] & bc[3]); st[2] = bc[2] ^ (~bc[3] & bc[4]); st[3] = bc[3] ^ (~bc[4] & bc[0]); st[4] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[5]; bc[1] = st[6]; bc[2] = st[7]; bc[3] = st[8]; bc[4] = st[9];
        st[5] = bc[0] ^ (~bc[1] & bc[2]); st[6] = bc[1] ^ (~bc[2] & bc[3]); st[7] = bc[2] ^ (~bc[3] & bc[4]); st[8] = bc[3] ^ (~bc[4] & bc[0]); st[9] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[10]; bc[1] = st[11]; bc[2] = st[12]; bc[3] = st[13]; bc[4] = st[14];
        st[10] = bc[0] ^ (~bc[1] & bc[2]); st[11] = bc[1] ^ (~bc[2] & bc[3]); st[12] = bc[2] ^ (~bc[3] & bc[4]); st[13] = bc[3] ^ (~bc[4] & bc[0]); st[14] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[15]; bc[1] = st[16]; bc[2] = st[17]; bc[3] = st[18]; bc[4] = st[19];
        st[15] = bc[0] ^ (~bc[1] & bc[2]); st[16] = bc[1] ^ (~bc[2] & bc[3]); st[17] = bc[2] ^ (~bc[3] & bc[4]); st[18] = bc[3] ^ (~bc[4] & bc[0]); st[19] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[20]; bc[1] = st[21]; bc[2] = st[22]; bc[3] = st[23]; bc[4] = st[24];
        st[20] = bc[0] ^ (~bc[1] & bc[2]); st[21] = bc[1] ^ (~bc[2] & bc[3]); st[22] = bc[2] ^ (~bc[3] & bc[4]); st[23] = bc[3] ^ (~bc[4] & bc[0]); st[24] = bc[4] ^ (~bc[0] & bc[1]);

        // Iota step
        st[0] ^= RC[round];
    }
}

// Convert 64-bit nonce to bytes (big-endian for salt suffix)
__device__ __forceinline__ void nonce_to_bytes(u64 nonce, uchar *bytes) {
    // We only need this for the result, not for the hash calculation
    // because we can inject the nonce directly into the u64 state
    bytes[0] = 0; bytes[1] = 0; bytes[2] = 0; bytes[3] = 0;
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
 * Optimized to load data directly into u64 state registers.
 * The input data is 85 bytes:
 *   0-20: 0xff ++ deployer (21 bytes)
 *   21-40: salt_prefix (20 bytes)
 *   41-52: salt_suffix (12 bytes) -> THIS CONTAINS THE NONCE
 *   53-84: init_code_hash (32 bytes)
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
    // Early exit if found
    if (*found) return;

    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    u64 nonce = start_nonce + idx;

    // Initialize state
    u64 st[25];

    // Load data_template into state.
    // The data is 85 bytes. We need to load it into st[0]...st[10] (first 11 words).
    // st[0]: bytes 0-7
    // st[1]: bytes 8-15
    // ...
    // st[5]: bytes 40-47. This word contains the transition from salt_prefix to salt_suffix.
    // st[6]: bytes 48-55. This word contains the rest of salt_suffix and start of init_code_hash.
    
    // We can load the constant parts from global memory (cached) or constant memory.
    // Since data_template is constant across threads, we can read it once.
    // However, to avoid unaligned reads and complex byte manipulation, we can just read as bytes and pack.
    // But for performance, we should pre-calculate the u64 words on the host or read them carefully.
    // Given the structure, let's construct the state words.

    // Note: Keccak uses little-endian words for the internal state, but the input is a byte stream.
    // So byte 0 goes to LSB of st[0].

    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;

    // We manually pack the 85 bytes into the state to avoid loops and local memory.
    // This is tedious but fastest.
    
    // Helper to load 8 bytes from template
    #define LD64(off) (*((u64*)(data_template + (off))))
    
    // We need to handle the nonce injection.
    // Nonce is at bytes 41-52 (12 bytes).
    // This affects st[5] (bytes 40-47) and st[6] (bytes 48-55).
    
    // Byte 40 is the last byte of salt_prefix.
    // Bytes 41-47 are the first 7 bytes of nonce (salt_suffix).
    // Bytes 48-52 are the last 5 bytes of nonce.
    // Bytes 53-55 are the first 3 bytes of init_code_hash.

    // Let's load the template as u64s, assuming the host provided a buffer we can read as u64s?
    // No, the host provides bytes. And they might not be aligned.
    // But we can just load bytes and shift.
    
    // Optimization: The compiler handles constant index byte loads well.
    
    // Word 0: bytes 0-7
    st[0] = ((u64)data_template[0]) | ((u64)data_template[1] << 8) | ((u64)data_template[2] << 16) | ((u64)data_template[3] << 24) |
            ((u64)data_template[4] << 32) | ((u64)data_template[5] << 40) | ((u64)data_template[6] << 48) | ((u64)data_template[7] << 56);
            
    // Word 1: bytes 8-15
    st[1] = ((u64)data_template[8]) | ((u64)data_template[9] << 8) | ((u64)data_template[10] << 16) | ((u64)data_template[11] << 24) |
            ((u64)data_template[12] << 32) | ((u64)data_template[13] << 40) | ((u64)data_template[14] << 48) | ((u64)data_template[15] << 56);

    // Word 2: bytes 16-23
    st[2] = ((u64)data_template[16]) | ((u64)data_template[17] << 8) | ((u64)data_template[18] << 16) | ((u64)data_template[19] << 24) |
            ((u64)data_template[20] << 32) | ((u64)data_template[21] << 40) | ((u64)data_template[22] << 48) | ((u64)data_template[23] << 56);

    // Word 3: bytes 24-31
    st[3] = ((u64)data_template[24]) | ((u64)data_template[25] << 8) | ((u64)data_template[26] << 16) | ((u64)data_template[27] << 24) |
            ((u64)data_template[28] << 32) | ((u64)data_template[29] << 40) | ((u64)data_template[30] << 48) | ((u64)data_template[31] << 56);

    // Word 4: bytes 32-39
    st[4] = ((u64)data_template[32]) | ((u64)data_template[33] << 8) | ((u64)data_template[34] << 16) | ((u64)data_template[35] << 24) |
            ((u64)data_template[36] << 32) | ((u64)data_template[37] << 40) | ((u64)data_template[38] << 48) | ((u64)data_template[39] << 56);

    // Word 5: bytes 40-47. 
    // Byte 40 is template[40].
    // Bytes 41-47 are nonce bytes 0-6 (of the 12-byte nonce).
    // The nonce is big-endian in the salt, so:
    // salt[0..3] = 0
    // salt[4] = nonce >> 56
    // ...
    // salt[11] = nonce & 0xFF
    // BUT, the nonce passed to us is a u64.
    // The salt suffix is 12 bytes. The first 4 are 0.
    // The next 8 are the u64 nonce in big-endian.
    // So salt_suffix[0..3] are 0.
    // salt_suffix[4] is MSB of nonce.
    // salt_suffix[11] is LSB of nonce.
    
    // Let's reconstruct the salt bytes from the u64 nonce.
    // salt_suffix bytes: 0, 0, 0, 0, N7, N6, N5, N4, N3, N2, N1, N0 (where N7 is MSB)
    
    // Bytes in data stream:
    // 41: 0
    // 42: 0
    // 43: 0
    // 44: 0
    // 45: N7 (nonce >> 56)
    // 46: N6 (nonce >> 48)
    // 47: N5 (nonce >> 40)
    // 48: N4 (nonce >> 32)
    // 49: N3 (nonce >> 24)
    // 50: N2 (nonce >> 16)
    // 51: N1 (nonce >> 8)
    // 52: N0 (nonce)
    
    // st[5] covers bytes 40-47.
    // Byte 40: template[40]
    // Byte 41: 0
    // Byte 42: 0
    // Byte 43: 0
    // Byte 44: 0
    // Byte 45: N7
    // Byte 46: N6
    // Byte 47: N5
    
    st[5] = ((u64)data_template[40]) |
            // Bytes 41-44 are 0, so we skip ORing them (they are 0)
            ((u64)(nonce >> 56) << 40) | // Byte 45
            ((u64)((nonce >> 48) & 0xFF) << 48) | // Byte 46
            ((u64)((nonce >> 40) & 0xFF) << 56);  // Byte 47

    // Word 6: bytes 48-55
    // Byte 48: N4
    // Byte 49: N3
    // Byte 50: N2
    // Byte 51: N1
    // Byte 52: N0
    // Byte 53: template[53]
    // Byte 54: template[54]
    // Byte 55: template[55]
    
    st[6] = ((u64)((nonce >> 32) & 0xFF)) |       // Byte 48
            ((u64)((nonce >> 24) & 0xFF) << 8) |  // Byte 49
            ((u64)((nonce >> 16) & 0xFF) << 16) | // Byte 50
            ((u64)((nonce >> 8) & 0xFF) << 24) |  // Byte 51
            ((u64)(nonce & 0xFF) << 32) |         // Byte 52
            ((u64)data_template[53] << 40) |
            ((u64)data_template[54] << 48) |
            ((u64)data_template[55] << 56);

    // Word 7: bytes 56-63
    st[7] = ((u64)data_template[56]) | ((u64)data_template[57] << 8) | ((u64)data_template[58] << 16) | ((u64)data_template[59] << 24) |
            ((u64)data_template[60] << 32) | ((u64)data_template[61] << 40) | ((u64)data_template[62] << 48) | ((u64)data_template[63] << 56);

    // Word 8: bytes 64-71
    st[8] = ((u64)data_template[64]) | ((u64)data_template[65] << 8) | ((u64)data_template[66] << 16) | ((u64)data_template[67] << 24) |
            ((u64)data_template[68] << 32) | ((u64)data_template[69] << 40) | ((u64)data_template[70] << 48) | ((u64)data_template[71] << 56);

    // Word 9: bytes 72-79
    st[9] = ((u64)data_template[72]) | ((u64)data_template[73] << 8) | ((u64)data_template[74] << 16) | ((u64)data_template[75] << 24) |
            ((u64)data_template[76] << 32) | ((u64)data_template[77] << 40) | ((u64)data_template[78] << 48) | ((u64)data_template[79] << 56);

    // Word 10: bytes 80-87
    // We have 85 bytes total (0-84).
    // Bytes 80-84 are from template.
    // Padding starts at byte 85.
    // Keccak padding: 0x01 at byte 85, 0x80 at byte 135 (end of block).
    // Since block size is 136 bytes, and we have 85 bytes, we are in the first block.
    // Byte 85 is 0x01.
    // Byte 86, 87 are 0.
    
    st[10] = ((u64)data_template[80]) | ((u64)data_template[81] << 8) | ((u64)data_template[82] << 16) | ((u64)data_template[83] << 24) |
             ((u64)data_template[84] << 32) |
             ((u64)0x01 << 40); // Padding start
             
    // Words 11-15 are 0.
    // Word 16: bytes 128-135.
    // Byte 135 must have 0x80.
    st[16] = 0x8000000000000000ULL;

    // Run Keccak permutation
    keccak_f1600(st);

    // Extract result (address is bytes 12-31 of the hash)
    // The hash is in st[0], st[1], st[2], st[3] (first 32 bytes).
    // st[0]: bytes 0-7
    // st[1]: bytes 8-15
    // st[2]: bytes 16-23
    // st[3]: bytes 24-31
    
    // Address starts at byte 12.
    // Byte 12 is in st[1] at offset 4 (bytes 8,9,10,11, 12...).
    // So address bytes 0-3 are st[1] >> 32.
    // Address bytes 4-11 are st[2].
    // Address bytes 12-19 are st[3].
    
    // Check pattern match
    // We check byte by byte or word by word.
    // Pattern is in global memory.
    
    bool match = true;
    
    // Check first 4 bytes (from st[1])
    if (pattern_length > 0) {
        u64 val = st[1] >> 32;
        for (int i = 0; i < 4 && i < pattern_length; i++) {
            if (((uchar)(val >> (i*8)) & 0xFF) != pattern[i]) {
                match = false;
                break;
            }
        }
    }
    
    // Check next 8 bytes (from st[2])
    if (match && pattern_length > 4) {
        u64 val = st[2];
        for (int i = 0; i < 8 && (i + 4) < pattern_length; i++) {
            if (((uchar)(val >> (i*8)) & 0xFF) != pattern[i+4]) {
                match = false;
                break;
            }
        }
    }
    
    // Check remaining bytes (from st[3])
    if (match && pattern_length > 12) {
        u64 val = st[3];
        for (int i = 0; i < 8 && (i + 12) < pattern_length; i++) {
            if (((uchar)(val >> (i*8)) & 0xFF) != pattern[i+12]) {
                match = false;
                break;
            }
        }
    }

    if (match) {
        if (atomicCAS(found, 0, 1) == 0) {
            // Found! Write result.
            uchar salt_suffix[12];
            nonce_to_bytes(nonce, salt_suffix);
            for (int i = 0; i < 12; i++) result_salt[i] = salt_suffix[i];
            
            // Write address
            // Re-extract from state to be sure
            uchar *hash_bytes = (uchar*)st;
            for (int i = 0; i < 20; i++) {
                result_address[i] = hash_bytes[12 + i];
            }
        }
    }
}

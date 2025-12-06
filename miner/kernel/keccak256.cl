/*
 * OpenCL Keccak256 Kernel for CREATE2 Vanity Address Mining
 * Optimized version using u64 state manipulation (like CUDA kernel)
 *
 * Keccak256 implementation optimized for mining workloads
 * Reference: https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf
 */

// Round constants for Keccak-f[1600]
__constant ulong RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotate left (circular shift) - optimized inline
inline ulong rotl64(ulong x, uint n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation - operates directly on u64 state
// This avoids expensive byte-to-u64 and u64-to-byte conversions
void keccak_f1600_u64(ulong *st) {
    ulong bc[5];
    ulong t;

    for (int round = 0; round < 24; ++round) {
        // Theta step
        bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
        bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
        bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
        bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
        bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

        t = bc[4] ^ rotl64(bc[1], 1); st[0] ^= t; st[5] ^= t; st[10] ^= t; st[15] ^= t; st[20] ^= t;
        t = bc[0] ^ rotl64(bc[2], 1); st[1] ^= t; st[6] ^= t; st[11] ^= t; st[16] ^= t; st[21] ^= t;
        t = bc[1] ^ rotl64(bc[3], 1); st[2] ^= t; st[7] ^= t; st[12] ^= t; st[17] ^= t; st[22] ^= t;
        t = bc[2] ^ rotl64(bc[4], 1); st[3] ^= t; st[8] ^= t; st[13] ^= t; st[18] ^= t; st[23] ^= t;
        t = bc[3] ^ rotl64(bc[0], 1); st[4] ^= t; st[9] ^= t; st[14] ^= t; st[19] ^= t; st[24] ^= t;

        // Rho and Pi steps - fully unrolled for performance
        t = st[1];
        st[1] = rotl64(st[6], 44);
        st[6] = rotl64(st[9], 20);
        st[9] = rotl64(st[22], 61);
        st[22] = rotl64(st[14], 39);
        st[14] = rotl64(st[20], 18);
        st[20] = rotl64(st[2], 62);
        st[2] = rotl64(st[12], 43);
        st[12] = rotl64(st[13], 25);
        st[13] = rotl64(st[19], 8);
        st[19] = rotl64(st[23], 56);
        st[23] = rotl64(st[15], 41);
        st[15] = rotl64(st[4], 27);
        st[4] = rotl64(st[24], 14);
        st[24] = rotl64(st[21], 2);
        st[21] = rotl64(st[8], 55);
        st[8] = rotl64(st[16], 45);
        st[16] = rotl64(st[5], 36);
        st[5] = rotl64(st[3], 28);
        st[3] = rotl64(st[18], 21);
        st[18] = rotl64(st[17], 15);
        st[17] = rotl64(st[11], 10);
        st[11] = rotl64(st[7], 6);
        st[7] = rotl64(st[10], 3);
        st[10] = rotl64(t, 1);

        // Chi step - unrolled for each row
        bc[0] = st[0]; bc[1] = st[1]; bc[2] = st[2]; bc[3] = st[3]; bc[4] = st[4];
        st[0] = bc[0] ^ (~bc[1] & bc[2]); st[1] = bc[1] ^ (~bc[2] & bc[3]);
        st[2] = bc[2] ^ (~bc[3] & bc[4]); st[3] = bc[3] ^ (~bc[4] & bc[0]);
        st[4] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[5]; bc[1] = st[6]; bc[2] = st[7]; bc[3] = st[8]; bc[4] = st[9];
        st[5] = bc[0] ^ (~bc[1] & bc[2]); st[6] = bc[1] ^ (~bc[2] & bc[3]);
        st[7] = bc[2] ^ (~bc[3] & bc[4]); st[8] = bc[3] ^ (~bc[4] & bc[0]);
        st[9] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[10]; bc[1] = st[11]; bc[2] = st[12]; bc[3] = st[13]; bc[4] = st[14];
        st[10] = bc[0] ^ (~bc[1] & bc[2]); st[11] = bc[1] ^ (~bc[2] & bc[3]);
        st[12] = bc[2] ^ (~bc[3] & bc[4]); st[13] = bc[3] ^ (~bc[4] & bc[0]);
        st[14] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[15]; bc[1] = st[16]; bc[2] = st[17]; bc[3] = st[18]; bc[4] = st[19];
        st[15] = bc[0] ^ (~bc[1] & bc[2]); st[16] = bc[1] ^ (~bc[2] & bc[3]);
        st[17] = bc[2] ^ (~bc[3] & bc[4]); st[18] = bc[3] ^ (~bc[4] & bc[0]);
        st[19] = bc[4] ^ (~bc[0] & bc[1]);

        bc[0] = st[20]; bc[1] = st[21]; bc[2] = st[22]; bc[3] = st[23]; bc[4] = st[24];
        st[20] = bc[0] ^ (~bc[1] & bc[2]); st[21] = bc[1] ^ (~bc[2] & bc[3]);
        st[22] = bc[2] ^ (~bc[3] & bc[4]); st[23] = bc[3] ^ (~bc[4] & bc[0]);
        st[24] = bc[4] ^ (~bc[0] & bc[1]);

        // Iota step
        st[0] ^= RC[round];
    }
}

// Convert 64-bit nonce to bytes (big-endian for salt suffix)
inline void nonce_to_bytes(ulong nonce, uchar bytes[12]) {
    // We only use 12 bytes for the salt suffix
    // Put zeros in first 4 bytes, then 8 bytes of nonce
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
 * Main CREATE2 mining kernel - OPTIMIZED VERSION
 *
 * Optimization: Load data directly into u64 state registers to avoid
 * byte-by-byte conversions. This is ~2-3x faster than the context-based approach.
 *
 * The input data is 85 bytes:
 *   0-20: 0xff ++ deployer (21 bytes)
 *   21-40: salt_prefix (20 bytes)
 *   41-52: salt_suffix (12 bytes) -> THIS CONTAINS THE NONCE
 *   53-84: init_code_hash (32 bytes)
 */
__kernel void mine_create2(
    __global const uchar *data_template,
    __global const uchar *pattern,
    const int pattern_length,
    const ulong start_nonce,
    __global uchar *result_salt,
    __global uchar *result_address,
    __global volatile int *found
) {
    // Early exit if found
    if (*found) return;

    ulong nonce = start_nonce + get_global_id(0);

    // Initialize state - directly load into u64 registers
    ulong st[25];

    // Zero initialize all state words
    for (int i = 0; i < 25; i++) st[i] = 0;

    // Load data_template directly into state words (little-endian)
    // Word 0: bytes 0-7
    st[0] = ((ulong)data_template[0]) | ((ulong)data_template[1] << 8) |
            ((ulong)data_template[2] << 16) | ((ulong)data_template[3] << 24) |
            ((ulong)data_template[4] << 32) | ((ulong)data_template[5] << 40) |
            ((ulong)data_template[6] << 48) | ((ulong)data_template[7] << 56);

    // Word 1: bytes 8-15
    st[1] = ((ulong)data_template[8]) | ((ulong)data_template[9] << 8) |
            ((ulong)data_template[10] << 16) | ((ulong)data_template[11] << 24) |
            ((ulong)data_template[12] << 32) | ((ulong)data_template[13] << 40) |
            ((ulong)data_template[14] << 48) | ((ulong)data_template[15] << 56);

    // Word 2: bytes 16-23
    st[2] = ((ulong)data_template[16]) | ((ulong)data_template[17] << 8) |
            ((ulong)data_template[18] << 16) | ((ulong)data_template[19] << 24) |
            ((ulong)data_template[20] << 32) | ((ulong)data_template[21] << 40) |
            ((ulong)data_template[22] << 48) | ((ulong)data_template[23] << 56);

    // Word 3: bytes 24-31
    st[3] = ((ulong)data_template[24]) | ((ulong)data_template[25] << 8) |
            ((ulong)data_template[26] << 16) | ((ulong)data_template[27] << 24) |
            ((ulong)data_template[28] << 32) | ((ulong)data_template[29] << 40) |
            ((ulong)data_template[30] << 48) | ((ulong)data_template[31] << 56);

    // Word 4: bytes 32-39
    st[4] = ((ulong)data_template[32]) | ((ulong)data_template[33] << 8) |
            ((ulong)data_template[34] << 16) | ((ulong)data_template[35] << 24) |
            ((ulong)data_template[36] << 32) | ((ulong)data_template[37] << 40) |
            ((ulong)data_template[38] << 48) | ((ulong)data_template[39] << 56);

    // Word 5: bytes 40-47 (contains transition from salt_prefix to salt_suffix)
    // Byte 40: template[40], Bytes 41-47: nonce bytes 0-6 (first 4 are zeros)
    // Nonce layout in salt_suffix (big-endian): [0,0,0,0, N7,N6,N5,N4, N3,N2,N1,N0]
    st[5] = ((ulong)data_template[40]) |
            // Bytes 41-44 are 0 (no contribution)
            ((ulong)(nonce >> 56) << 40) |         // Byte 45: N7
            ((ulong)((nonce >> 48) & 0xFF) << 48) | // Byte 46: N6
            ((ulong)((nonce >> 40) & 0xFF) << 56);  // Byte 47: N5

    // Word 6: bytes 48-55
    // Bytes 48-52: nonce bytes 7-11, Bytes 53-55: init_code_hash[0-2]
    st[6] = ((ulong)((nonce >> 32) & 0xFF)) |       // Byte 48: N4
            ((ulong)((nonce >> 24) & 0xFF) << 8) |  // Byte 49: N3
            ((ulong)((nonce >> 16) & 0xFF) << 16) | // Byte 50: N2
            ((ulong)((nonce >> 8) & 0xFF) << 24) |  // Byte 51: N1
            ((ulong)(nonce & 0xFF) << 32) |         // Byte 52: N0
            ((ulong)data_template[53] << 40) |
            ((ulong)data_template[54] << 48) |
            ((ulong)data_template[55] << 56);

    // Word 7: bytes 56-63
    st[7] = ((ulong)data_template[56]) | ((ulong)data_template[57] << 8) |
            ((ulong)data_template[58] << 16) | ((ulong)data_template[59] << 24) |
            ((ulong)data_template[60] << 32) | ((ulong)data_template[61] << 40) |
            ((ulong)data_template[62] << 48) | ((ulong)data_template[63] << 56);

    // Word 8: bytes 64-71
    st[8] = ((ulong)data_template[64]) | ((ulong)data_template[65] << 8) |
            ((ulong)data_template[66] << 16) | ((ulong)data_template[67] << 24) |
            ((ulong)data_template[68] << 32) | ((ulong)data_template[69] << 40) |
            ((ulong)data_template[70] << 48) | ((ulong)data_template[71] << 56);

    // Word 9: bytes 72-79
    st[9] = ((ulong)data_template[72]) | ((ulong)data_template[73] << 8) |
            ((ulong)data_template[74] << 16) | ((ulong)data_template[75] << 24) |
            ((ulong)data_template[76] << 32) | ((ulong)data_template[77] << 40) |
            ((ulong)data_template[78] << 48) | ((ulong)data_template[79] << 56);

    // Word 10: bytes 80-87 (data ends at 84, padding starts at 85)
    // Keccak padding: 0x01 at byte 85, 0x80 at byte 135 (end of block)
    st[10] = ((ulong)data_template[80]) | ((ulong)data_template[81] << 8) |
             ((ulong)data_template[82] << 16) | ((ulong)data_template[83] << 24) |
             ((ulong)data_template[84] << 32) |
             ((ulong)0x01 << 40); // Padding start at byte 85

    // Word 16: padding end (byte 135 must have 0x80)
    st[16] = 0x8000000000000000ULL;

    // Run Keccak-f[1600] permutation
    keccak_f1600_u64(st);

    // Extract and check pattern match
    // Address is bytes 12-31 of the hash (in st[1] bits 32-63, st[2], and st[3])
    bool match = true;

    // Check first 4 bytes (from st[1] >> 32)
    if (pattern_length > 0) {
        ulong val = st[1] >> 32;
        for (int i = 0; i < 4 && i < pattern_length; i++) {
            if (((uchar)(val >> (i*8)) & 0xFF) != pattern[i]) {
                match = false;
                break;
            }
        }
    }

    // Check next 8 bytes (from st[2])
    if (match && pattern_length > 4) {
        ulong val = st[2];
        for (int i = 0; i < 8 && (i + 4) < pattern_length; i++) {
            if (((uchar)(val >> (i*8)) & 0xFF) != pattern[i+4]) {
                match = false;
                break;
            }
        }
    }

    // Check remaining bytes (from st[3])
    if (match && pattern_length > 12) {
        ulong val = st[3];
        for (int i = 0; i < 8 && (i + 12) < pattern_length; i++) {
            if (((uchar)(val >> (i*8)) & 0xFF) != pattern[i+12]) {
                match = false;
                break;
            }
        }
    }

    if (match) {
        // Use atomic compare-and-swap for OpenCL 1.2 compatibility
        int expected = 0;
        if (atomic_cmpxchg(found, expected, 1) == 0) {
            // Write salt suffix
            uchar salt_suffix[12];
            nonce_to_bytes(nonce, salt_suffix);
            for (int i = 0; i < 12; i++) {
                result_salt[i] = salt_suffix[i];
            }

            // Write address from state
            uchar *hash_bytes = (uchar*)st;
            for (int i = 0; i < 20; i++) {
                result_address[i] = hash_bytes[12 + i];
            }
        }
    }
}


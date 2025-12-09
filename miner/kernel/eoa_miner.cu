/*
 * CUDA EOA Vanity Address Mining Kernel
 * Generates Ethereum EOA addresses with specified prefix patterns
 *
 * Pipeline per thread:
 *   Private Key -> secp256k1 scalar mul -> Public Key -> Keccak256 -> Address
 */

#include "secp256k1.cu"

// Keccak-f[1600] round constants (from keccak256.cu)
__constant__ u64 RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

#define ROL64(a, offset) ((((u64)a) << offset) ^ (((u64)a) >> (64 - offset)))

// Keccak-f[1600] permutation
__device__ __forceinline__ void keccak_f1600_eoa(u64 *st) {
    u64 bc[5], t;

    #pragma unroll
    for (int round = 0; round < 24; ++round) {
        // Theta
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

        // Rho and Pi
        t = st[1];
        st[1] = ROL64(st[6], 44);  st[6] = ROL64(st[9], 20);  st[9] = ROL64(st[22], 61);
        st[22] = ROL64(st[14], 39); st[14] = ROL64(st[20], 18); st[20] = ROL64(st[2], 62);
        st[2] = ROL64(st[12], 43);  st[12] = ROL64(st[13], 25); st[13] = ROL64(st[19], 8);
        st[19] = ROL64(st[23], 56); st[23] = ROL64(st[15], 41); st[15] = ROL64(st[4], 27);
        st[4] = ROL64(st[24], 14);  st[24] = ROL64(st[21], 2);  st[21] = ROL64(st[8], 55);
        st[8] = ROL64(st[16], 45);  st[16] = ROL64(st[5], 36);  st[5] = ROL64(st[3], 28);
        st[3] = ROL64(st[18], 21);  st[18] = ROL64(st[17], 15); st[17] = ROL64(st[11], 10);
        st[11] = ROL64(st[7], 6);   st[7] = ROL64(st[10], 3);   st[10] = ROL64(t, 1);

        // Chi
        bc[0] = st[0]; bc[1] = st[1]; bc[2] = st[2]; bc[3] = st[3]; bc[4] = st[4];
        st[0] ^= ~bc[1] & bc[2]; st[1] ^= ~bc[2] & bc[3]; st[2] ^= ~bc[3] & bc[4];
        st[3] ^= ~bc[4] & bc[0]; st[4] ^= ~bc[0] & bc[1];

        bc[0] = st[5]; bc[1] = st[6]; bc[2] = st[7]; bc[3] = st[8]; bc[4] = st[9];
        st[5] ^= ~bc[1] & bc[2]; st[6] ^= ~bc[2] & bc[3]; st[7] ^= ~bc[3] & bc[4];
        st[8] ^= ~bc[4] & bc[0]; st[9] ^= ~bc[0] & bc[1];

        bc[0] = st[10]; bc[1] = st[11]; bc[2] = st[12]; bc[3] = st[13]; bc[4] = st[14];
        st[10] ^= ~bc[1] & bc[2]; st[11] ^= ~bc[2] & bc[3]; st[12] ^= ~bc[3] & bc[4];
        st[13] ^= ~bc[4] & bc[0]; st[14] ^= ~bc[0] & bc[1];

        bc[0] = st[15]; bc[1] = st[16]; bc[2] = st[17]; bc[3] = st[18]; bc[4] = st[19];
        st[15] ^= ~bc[1] & bc[2]; st[16] ^= ~bc[2] & bc[3]; st[17] ^= ~bc[3] & bc[4];
        st[18] ^= ~bc[4] & bc[0]; st[19] ^= ~bc[0] & bc[1];

        bc[0] = st[20]; bc[1] = st[21]; bc[2] = st[22]; bc[3] = st[23]; bc[4] = st[24];
        st[20] ^= ~bc[1] & bc[2]; st[21] ^= ~bc[2] & bc[3]; st[22] ^= ~bc[3] & bc[4];
        st[23] ^= ~bc[4] & bc[0]; st[24] ^= ~bc[0] & bc[1];

        // Iota
        st[0] ^= RC[round];
    }
}

// Keccak256 for 64-byte public key (one block, no padding issues)
__device__ __forceinline__ void keccak256_64(const uchar* input, uchar* output) {
    u64 st[25];

    // Initialize state to zero
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;

    // Load 64 bytes into first 8 words (little-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        st[i] = ((u64)input[i*8]) | ((u64)input[i*8+1] << 8) |
                ((u64)input[i*8+2] << 16) | ((u64)input[i*8+3] << 24) |
                ((u64)input[i*8+4] << 32) | ((u64)input[i*8+5] << 40) |
                ((u64)input[i*8+6] << 48) | ((u64)input[i*8+7] << 56);
    }

    // Padding for 64-byte input with rate=136 (Keccak256)
    // Byte 64 gets 0x01, byte 135 gets 0x80
    st[8] ^= 0x01ULL;              // Byte 64
    st[16] ^= 0x8000000000000000ULL; // Byte 135

    // Apply Keccak-f[1600]
    keccak_f1600_eoa(st);

    // Extract first 32 bytes of output (little-endian)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        output[i*8] = (uchar)(st[i]); output[i*8+1] = (uchar)(st[i] >> 8);
        output[i*8+2] = (uchar)(st[i] >> 16); output[i*8+3] = (uchar)(st[i] >> 24);
        output[i*8+4] = (uchar)(st[i] >> 32); output[i*8+5] = (uchar)(st[i] >> 40);
        output[i*8+6] = (uchar)(st[i] >> 48); output[i*8+7] = (uchar)(st[i] >> 56);
    }
}

// ============================================================================
// EOA Mining Kernel
// ============================================================================

// Constant memory for input data (read-only, same for all threads)
// Constant memory for input data (read-only, same for all threads)
__constant__ uchar c_base_private_key[32];
__constant__ uchar c_pattern[20];
__constant__ int c_pattern_length;

// ============================================================================
// Generator Table Optimization
// ============================================================================

// Initialize generator table: table[i] = i * G
extern "C" __global__ void init_generator_table(AffinePoint* table, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // k = idx
    uint256 k;
    k.limbs[0] = idx;
    k.limbs[1] = 0; k.limbs[2] = 0; k.limbs[3] = 0;

    AffinePoint pubkey;
    if (idx == 0) {
        // 0 * G = Infinity (represented as 0,0 for affine storage?)
        // Actually, our scalar_mul handles 0 correctly returning infinity (0,1,0 Jacobian).
        // But for affine addition, we need a special format or valid point.
        // Let's store Infinity as x=0, y=0.
        // But better: store 0*G as a point at infinity marker if expected,
        // OR simply handle idx=0 specially in loop. 
        // scalar_mul with 0 returns Jacobian Infinity (0, 1, 0).
        // jacobian_to_affine with Infinity returns (0, 0).
        // point_add_mixed handles Infinity correctly.
        derive_public_key(&pubkey, &k);
    } else {
        derive_public_key(&pubkey, &k);
    }
    table[idx] = pubkey;
}

// ============================================================================
// Batched Inversion Logic
// ============================================================================

// Parallel Prefix Scan (Product) in Shared Memory
// Input: value in 'val'
// Output: returns prefix product (inclusive)
// buffer: shared memory array of size blockDim.x
__device__ uint256 block_scan_prefix_mul(uint256 val, uint256* buffer) {
    int tid = threadIdx.x;
    buffer[tid] = val;
    __syncthreads();

    // Hillis-Steele Scan
    #pragma unroll
    for (int stride = 1; stride < 256; stride *= 2) {
        uint256 neighbor;
        if (tid >= stride) {
             neighbor = buffer[tid - stride];
        } else {
             // Identity for multiplication is 1
             // We can optimize by branching, OR just load UINT256_ONE
             // Branching is fine here.
        }
        __syncthreads();
        
        if (tid >= stride) {
            uint256 current = buffer[tid];
            uint256_mul_mod_p(&buffer[tid], &current, &neighbor);
        }
        __syncthreads();
    }
    return buffer[tid];
}

// Parallel Suffix Scan (Product) in Shared Memory
// Input: value in 'val'
// Output: returns suffix product (inclusive)
__device__ uint256 block_scan_suffix_mul(uint256 val, uint256* buffer) {
    int tid = threadIdx.x;
    buffer[tid] = val;
    __syncthreads();

    #pragma unroll
    for (int stride = 1; stride < 256; stride *= 2) {
        uint256 neighbor;
        if (tid + stride < 256) {
             neighbor = buffer[tid + stride];
        }
        __syncthreads();
        
        if (tid + stride < 256) {
            uint256 current = buffer[tid];
            uint256_mul_mod_p(&buffer[tid], &current, &neighbor);
        }
        __syncthreads();
    }
    return buffer[tid];
}

extern "C" __global__ __launch_bounds__(256, 2) void mine_eoa_opt(
    u64 start_nonce,
    uint256 batch_base_pub_x,
    uint256 batch_base_pub_y, // Jacobian point (Z=1 assumed)
    AffinePoint* __restrict__ generator_table,
    uchar* __restrict__ result_private_key,
    uchar* __restrict__ result_address,
    int* found
) {
    if (*found) return;

    u64 idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // R = BatchBase + Table[idx]
    JacobianPoint R;
    R.X = batch_base_pub_x;
    R.Y = batch_base_pub_y;
    R.Z = UINT256_ONE;

    // Handle idx=0 specially: Table[0] = 0*G = infinity (stored as (0,0))
    // Adding infinity is identity, so for global idx=0 we skip the addition.
    // However, global idx=0 only happens for thread 0 of block 0.
    // For each block, we use `idx` which is the GLOBAL index.
    // So we only need to handle the case when idx==0 (first thread of first block).
    if (idx != 0) {
        AffinePoint G_i = generator_table[idx]; // Load precomputed i*G
        
        // R = R + G_i
        // Note: Must use temporary to avoid aliasing issue (r and p same memory)
        JacobianPoint result;
        point_add_mixed(&result, &R, &G_i);
        R = result;
    }
    // For idx==0: R is already batch_base_pub, which is correct.

    // --------------------------------------------------------
    // Convert to Affine (using standard per-thread inversion for debugging)
    // --------------------------------------------------------
    AffinePoint P;
    jacobian_to_affine(&P, &R);

    // Serialize
    uchar pubkey_bytes[64];
    serialize_public_key(pubkey_bytes, &P);

    // Hash
    uchar hash[32];
    keccak256_64(pubkey_bytes, hash);
    uchar* address = hash + 12;

    // Check match
    bool match = true;
    for (int i = 0; i < c_pattern_length && match; i++) {
        if (address[i] != c_pattern[i]) {
            match = false;
        }
    }

    if (match) {
        if (atomicCAS(found, 0, 1) == 0) {
            // Reconstruct private key: priv = base_priv + nonce
            uint256 priv;
            uint256_from_bytes_be(&priv, c_base_private_key);
            
            u64 nonce = start_nonce + idx;
            uint256_add_u64(&priv, &priv, nonce);
            
            uint256_to_bytes_be(result_private_key, &priv);
            
            #pragma unroll
            for (int i = 0; i < 20; i++) result_address[i] = address[i];
        }
    }
}


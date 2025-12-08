/*
 * secp256k1 Elliptic Curve Operations for CUDA
 * Implements point operations for EOA vanity address mining
 */

#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include "uint256.cuh"

// ============================================================================
// Point Structures
// ============================================================================

// Affine point (x, y) - used for storage and final output
typedef struct {
    uint256 x;
    uint256 y;
} AffinePoint;

// Jacobian point (X, Y, Z) where x = X/Z², y = Y/Z³
// Avoids expensive modular inversions during intermediate calculations
typedef struct {
    uint256 X;
    uint256 Y;
    uint256 Z;
} JacobianPoint;

// ============================================================================
// secp256k1 Generator Point G
// ============================================================================

// G.x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
__constant__ uint256 SECP256K1_GX = {{
    0x59F2815B16F81798ULL,
    0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL,
    0x79BE667EF9DCBBACULL
}};

// G.y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
__constant__ uint256 SECP256K1_GY = {{
    0x9C47D08FFB10D4B8ULL,
    0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL,
    0x483ADA7726A3C465ULL
}};

// ============================================================================
// Point Constants
// ============================================================================

// Point at infinity (identity element) - represented with Z = 0
#define JACOBIAN_INFINITY {UINT256_ZERO, UINT256_ONE, UINT256_ZERO}

// Check if point is at infinity
__device__ __forceinline__ int jacobian_is_infinity(const JacobianPoint* p) {
    return uint256_is_zero(&p->Z);
}

// ============================================================================
// Point Operations - Declarations
// ============================================================================

// Point doubling in Jacobian coordinates
// R = 2 * P
__device__ void point_double_jacobian(JacobianPoint* r, const JacobianPoint* p);

// Point addition in Jacobian coordinates
// R = P + Q
__device__ void point_add_jacobian(JacobianPoint* r, const JacobianPoint* p, const JacobianPoint* q);

// Mixed addition: Jacobian + Affine (faster than full Jacobian addition)
// R = P (Jacobian) + Q (Affine)
__device__ void point_add_mixed(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q);

// Scalar multiplication: R = k * P (double-and-add)
__device__ void scalar_mul(JacobianPoint* r, const uint256* k, const AffinePoint* p);

// Convert Jacobian to Affine: (X, Y, Z) -> (X/Z², Y/Z³)
__device__ void jacobian_to_affine(AffinePoint* r, const JacobianPoint* p);

// Convert Affine to Jacobian: (x, y) -> (x, y, 1)
__device__ __forceinline__ void affine_to_jacobian(JacobianPoint* r, const AffinePoint* p) {
    r->X = p->x;
    r->Y = p->y;
    r->Z = UINT256_ONE;
}

// ============================================================================
// Public Key Derivation
// ============================================================================

// Derive public key from private key: pubkey = privkey * G
__device__ void derive_public_key(AffinePoint* pubkey, const uint256* privkey);

// Serialize public key to 64 bytes (uncompressed, without 04 prefix)
__device__ void serialize_public_key(uchar* out, const AffinePoint* pubkey);

// ============================================================================
// Utility Functions
// ============================================================================

// Load uint256 from big-endian bytes
__device__ void uint256_from_bytes_be(uint256* r, const uchar* bytes);

// Store uint256 to big-endian bytes
__device__ void uint256_to_bytes_be(uchar* bytes, const uint256* a);

// Add scalar to uint256 (for private key increment)
__device__ void uint256_add_u64(uint256* r, const uint256* a, u64 b);

#endif // SECP256K1_CUH


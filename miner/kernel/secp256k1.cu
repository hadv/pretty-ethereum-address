/*
 * secp256k1 Elliptic Curve Operations for CUDA
 * Implements point operations for EOA vanity address mining
 */

#include "secp256k1.cuh"
#include "uint256.cu"

// ============================================================================
// Point Doubling: R = 2 * P
// Using standard Jacobian doubling formulas
// ============================================================================

__device__ void point_double_jacobian(JacobianPoint* r, const JacobianPoint* p) {
    // Handle point at infinity
    if (jacobian_is_infinity(p)) {
        *r = *p;
        return;
    }

    uint256 s, m, x3, y3, z3, tmp1, tmp2;

    // M = 3 * X^2 (secp256k1 has a = 0)
    uint256_sqr_mod_p(&tmp1, &p->X);           // tmp1 = X^2
    uint256_add_mod_p(&tmp2, &tmp1, &tmp1);    // tmp2 = 2*X^2
    uint256_add_mod_p(&m, &tmp2, &tmp1);       // M = 3*X^2

    // S = 4 * X * Y^2
    uint256_sqr_mod_p(&tmp1, &p->Y);           // tmp1 = Y^2
    uint256_mul_mod_p(&tmp2, &p->X, &tmp1);    // tmp2 = X*Y^2
    uint256_add_mod_p(&tmp1, &tmp2, &tmp2);    // tmp1 = 2*X*Y^2
    uint256_add_mod_p(&s, &tmp1, &tmp1);       // S = 4*X*Y^2

    // X3 = M^2 - 2*S
    uint256_sqr_mod_p(&tmp1, &m);              // tmp1 = M^2
    uint256_add_mod_p(&tmp2, &s, &s);          // tmp2 = 2*S
    uint256_sub_mod_p(&x3, &tmp1, &tmp2);      // X3 = M^2 - 2*S

    // Z3 = 2 * Y * Z
    uint256_mul_mod_p(&tmp1, &p->Y, &p->Z);    // tmp1 = Y*Z
    uint256_add_mod_p(&z3, &tmp1, &tmp1);      // Z3 = 2*Y*Z

    // Y3 = M * (S - X3) - 8 * Y^4
    uint256_sub_mod_p(&tmp1, &s, &x3);         // tmp1 = S - X3
    uint256_mul_mod_p(&tmp2, &m, &tmp1);       // tmp2 = M*(S-X3)
    
    uint256_sqr_mod_p(&tmp1, &p->Y);           // tmp1 = Y^2
    uint256_sqr_mod_p(&s, &tmp1);              // s = Y^4 (reuse s)
    uint256_add_mod_p(&tmp1, &s, &s);          // tmp1 = 2*Y^4
    uint256_add_mod_p(&s, &tmp1, &tmp1);       // s = 4*Y^4
    uint256_add_mod_p(&tmp1, &s, &s);          // tmp1 = 8*Y^4
    uint256_sub_mod_p(&y3, &tmp2, &tmp1);      // Y3 = M*(S-X3) - 8*Y^4

    r->X = x3;
    r->Y = y3;
    r->Z = z3;
}

// ============================================================================
// Mixed Addition: R = P (Jacobian) + Q (Affine)
// More efficient than full Jacobian addition
// ============================================================================

__device__ void point_add_mixed(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q) {
    // Handle special cases
    if (jacobian_is_infinity(p)) {
        r->X = q->x;
        r->Y = q->y;
        r->Z = UINT256_ONE;
        return;
    }

    uint256 z2, z3, u2, s2, h, i, j, rr, v, tmp;

    // Z1^2 and Z1^3
    uint256_sqr_mod_p(&z2, &p->Z);             // z2 = Z1^2
    uint256_mul_mod_p(&z3, &z2, &p->Z);        // z3 = Z1^3

    // U2 = X2 * Z1^2
    uint256_mul_mod_p(&u2, &q->x, &z2);        // u2 = X2*Z1^2

    // S2 = Y2 * Z1^3
    uint256_mul_mod_p(&s2, &q->y, &z3);        // s2 = Y2*Z1^3

    // H = U2 - X1
    uint256_sub_mod_p(&h, &u2, &p->X);         // h = U2 - X1

    // Check if points are the same (H == 0)
    if (uint256_is_zero(&h)) {
        // Check if Y values match for doubling
        uint256_sub_mod_p(&tmp, &s2, &p->Y);
        if (uint256_is_zero(&tmp)) {
            // P == Q, use doubling
            point_double_jacobian(r, p);
            return;
        } else {
            // P == -Q, result is infinity
            r->X = UINT256_ZERO;
            r->Y = UINT256_ONE;
            r->Z = UINT256_ZERO;
            return;
        }
    }

    // I = (2*H)^2
    uint256_add_mod_p(&tmp, &h, &h);           // tmp = 2*H
    uint256_sqr_mod_p(&i, &tmp);               // I = (2*H)^2

    // J = H * I
    uint256_mul_mod_p(&j, &h, &i);             // J = H*I

    // r = 2 * (S2 - Y1)
    uint256_sub_mod_p(&tmp, &s2, &p->Y);       // tmp = S2 - Y1
    uint256_add_mod_p(&rr, &tmp, &tmp);        // r = 2*(S2-Y1)

    // V = X1 * I
    uint256_mul_mod_p(&v, &p->X, &i);          // V = X1*I

    // X3 = r^2 - J - 2*V
    uint256_sqr_mod_p(&r->X, &rr);             // X3 = r^2
    uint256_sub_mod_p(&r->X, &r->X, &j);       // X3 = r^2 - J
    uint256_add_mod_p(&tmp, &v, &v);           // tmp = 2*V
    uint256_sub_mod_p(&r->X, &r->X, &tmp);     // X3 = r^2 - J - 2*V

    // Y3 = r * (V - X3) - 2 * Y1 * J
    uint256_sub_mod_p(&tmp, &v, &r->X);        // tmp = V - X3
    uint256_mul_mod_p(&r->Y, &rr, &tmp);       // Y3 = r*(V-X3)
    uint256_mul_mod_p(&tmp, &p->Y, &j);        // tmp = Y1*J
    uint256_add_mod_p(&tmp, &tmp, &tmp);       // tmp = 2*Y1*J
    uint256_sub_mod_p(&r->Y, &r->Y, &tmp);     // Y3 = r*(V-X3) - 2*Y1*J

    // Z3 = (Z1 + H)^2 - Z1^2 - H^2 = 2*Z1*H
    uint256_mul_mod_p(&tmp, &p->Z, &h);        // tmp = Z1*H
    uint256_add_mod_p(&r->Z, &tmp, &tmp);      // Z3 = 2*Z1*H
}

// ============================================================================
// Scalar Multiplication: R = k * P (double-and-add)
// ============================================================================

__device__ void scalar_mul(JacobianPoint* r, const uint256* k, const AffinePoint* p) {
    // Start with point at infinity
    r->X = UINT256_ZERO;
    r->Y = UINT256_ONE;
    r->Z = UINT256_ZERO;

    // Handle zero scalar
    if (uint256_is_zero(k)) {
        return;
    }

    JacobianPoint current;
    affine_to_jacobian(&current, p);

    JacobianPoint tmp;

    // Process from LSB to MSB
    // NOTE: Do NOT use #pragma unroll - 256 iterations would hang the compiler
    for (int w = 0; w < 4; w++) {
        u64 bits = k->limbs[w];
        for (int i = 0; i < 64; i++) {
            if (bits & 1) {
                if (jacobian_is_infinity(r)) {
                    *r = current;
                } else {
                    point_add_mixed(&tmp, r, p);
                    *r = tmp;
                }
            }
            point_double_jacobian(&tmp, &current);
            current = tmp;
            bits >>= 1;
        }
    }
}

// ============================================================================
// Jacobian to Affine Conversion
// ============================================================================

__device__ void jacobian_to_affine(AffinePoint* r, const JacobianPoint* p) {
    // Handle point at infinity
    if (jacobian_is_infinity(p)) {
        r->x = UINT256_ZERO;
        r->y = UINT256_ZERO;
        return;
    }

    uint256 z_inv, z_inv2, z_inv3;

    // Compute Z^(-1)
    uint256_inv_mod_p(&z_inv, &p->Z);

    // Z^(-2)
    uint256_sqr_mod_p(&z_inv2, &z_inv);

    // Z^(-3)
    uint256_mul_mod_p(&z_inv3, &z_inv2, &z_inv);

    // x = X * Z^(-2)
    uint256_mul_mod_p(&r->x, &p->X, &z_inv2);

    // y = Y * Z^(-3)
    uint256_mul_mod_p(&r->y, &p->Y, &z_inv3);
}

// ============================================================================
// Public Key Derivation
// ============================================================================

__device__ void derive_public_key(AffinePoint* pubkey, const uint256* privkey) {
    AffinePoint G;
    G.x = SECP256K1_GX;
    G.y = SECP256K1_GY;

    JacobianPoint result;
    scalar_mul(&result, privkey, &G);
    jacobian_to_affine(pubkey, &result);
}

// ============================================================================
// Byte Conversion Utilities
// ============================================================================

// Load uint256 from 32 big-endian bytes
__device__ void uint256_from_bytes_be(uint256* r, const uchar* bytes) {
    r->limbs[3] = ((u64)bytes[0] << 56) | ((u64)bytes[1] << 48) |
                  ((u64)bytes[2] << 40) | ((u64)bytes[3] << 32) |
                  ((u64)bytes[4] << 24) | ((u64)bytes[5] << 16) |
                  ((u64)bytes[6] << 8)  | (u64)bytes[7];
    r->limbs[2] = ((u64)bytes[8] << 56) | ((u64)bytes[9] << 48) |
                  ((u64)bytes[10] << 40) | ((u64)bytes[11] << 32) |
                  ((u64)bytes[12] << 24) | ((u64)bytes[13] << 16) |
                  ((u64)bytes[14] << 8) | (u64)bytes[15];
    r->limbs[1] = ((u64)bytes[16] << 56) | ((u64)bytes[17] << 48) |
                  ((u64)bytes[18] << 40) | ((u64)bytes[19] << 32) |
                  ((u64)bytes[20] << 24) | ((u64)bytes[21] << 16) |
                  ((u64)bytes[22] << 8) | (u64)bytes[23];
    r->limbs[0] = ((u64)bytes[24] << 56) | ((u64)bytes[25] << 48) |
                  ((u64)bytes[26] << 40) | ((u64)bytes[27] << 32) |
                  ((u64)bytes[28] << 24) | ((u64)bytes[29] << 16) |
                  ((u64)bytes[30] << 8) | (u64)bytes[31];
}

// Store uint256 to 32 big-endian bytes
__device__ void uint256_to_bytes_be(uchar* bytes, const uint256* a) {
    bytes[0] = (uchar)(a->limbs[3] >> 56); bytes[1] = (uchar)(a->limbs[3] >> 48);
    bytes[2] = (uchar)(a->limbs[3] >> 40); bytes[3] = (uchar)(a->limbs[3] >> 32);
    bytes[4] = (uchar)(a->limbs[3] >> 24); bytes[5] = (uchar)(a->limbs[3] >> 16);
    bytes[6] = (uchar)(a->limbs[3] >> 8);  bytes[7] = (uchar)(a->limbs[3]);
    bytes[8] = (uchar)(a->limbs[2] >> 56); bytes[9] = (uchar)(a->limbs[2] >> 48);
    bytes[10] = (uchar)(a->limbs[2] >> 40); bytes[11] = (uchar)(a->limbs[2] >> 32);
    bytes[12] = (uchar)(a->limbs[2] >> 24); bytes[13] = (uchar)(a->limbs[2] >> 16);
    bytes[14] = (uchar)(a->limbs[2] >> 8); bytes[15] = (uchar)(a->limbs[2]);
    bytes[16] = (uchar)(a->limbs[1] >> 56); bytes[17] = (uchar)(a->limbs[1] >> 48);
    bytes[18] = (uchar)(a->limbs[1] >> 40); bytes[19] = (uchar)(a->limbs[1] >> 32);
    bytes[20] = (uchar)(a->limbs[1] >> 24); bytes[21] = (uchar)(a->limbs[1] >> 16);
    bytes[22] = (uchar)(a->limbs[1] >> 8); bytes[23] = (uchar)(a->limbs[1]);
    bytes[24] = (uchar)(a->limbs[0] >> 56); bytes[25] = (uchar)(a->limbs[0] >> 48);
    bytes[26] = (uchar)(a->limbs[0] >> 40); bytes[27] = (uchar)(a->limbs[0] >> 32);
    bytes[28] = (uchar)(a->limbs[0] >> 24); bytes[29] = (uchar)(a->limbs[0] >> 16);
    bytes[30] = (uchar)(a->limbs[0] >> 8); bytes[31] = (uchar)(a->limbs[0]);
}

// Serialize public key to 64 bytes (X || Y, big-endian)
__device__ void serialize_public_key(uchar* out, const AffinePoint* pubkey) {
    uint256_to_bytes_be(out, &pubkey->x);
    uint256_to_bytes_be(out + 32, &pubkey->y);
}

// Add u64 to uint256
__device__ void uint256_add_u64(uint256* r, const uint256* a, u64 b) {
    u128 sum = (u128)a->limbs[0] + b;
    r->limbs[0] = (u64)sum;
    u64 carry = (u64)(sum >> 64);

    #pragma unroll
    for (int i = 1; i < 4; i++) {
        sum = (u128)a->limbs[i] + carry;
        r->limbs[i] = (u64)sum;
        carry = (u64)(sum >> 64);
    }
}


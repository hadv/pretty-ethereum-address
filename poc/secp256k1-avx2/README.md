# secp256k1-avx2/avx512 Proof of Concept

**Parallel secp256k1 field multiplication using AVX2/AVX-512 SIMD instructions.**

## Overview

This PoC demonstrates the "limb-slicing" technique for parallel elliptic curve operations:

| SIMD Level | Register Size | Parallel Elements | Key Feature |
|------------|---------------|-------------------|-------------|
| **AVX2** | 256-bit | 4-way | `vpmuludq` (32×32→64) |
| **AVX-512F** | 512-bit | 8-way | `vpmuludq` (32×32→64) |
| **AVX-512 IFMA** | 512-bit | 8-way | `vpmadd52` (52×52→104) ⭐ |

**AVX-512 IFMA is the killer feature** - it has native 52-bit multiply-add instructions designed for cryptography!

## Why AVX-512 IFMA is Perfect for secp256k1

secp256k1 uses **5×52-bit limb representation**. AVX-512 IFMA provides:

```c
// Native 52×52→104 bit multiply-add!
vpmadd52luq zmm_dst, zmm_a, zmm_b  // dst += (a × b)[0:52]   (low 52 bits)
vpmadd52huq zmm_dst, zmm_a, zmm_b  // dst += (a × b)[52:104] (high 52 bits)
```

**Available on**: Intel Ice Lake, Tiger Lake, Alder Lake, Sapphire Rapids, AMD Zen 4

## Files

- `field.h` - Field element representation (5×52-bit limbs, from libsecp256k1)
- `field_mul.h` - Scalar field multiplication (reference implementation)
- `field_mul_avx2.h` - **AVX2 4-way parallel multiplication**
- `field_mul_avx512.h` - **AVX-512 8-way parallel multiplication** (with IFMA support)
- `bench.c` - Benchmark comparing scalar vs SIMD performance

## Building

```bash
# Linux (with AVX2 support)
make
./bench

# macOS with Intel CPU
make
./bench

# macOS with Apple Silicon (M1/M2/M3)
# Compiles to x86_64, runs via Rosetta 2
make
./bench
```

## Expected Results

On a modern x86_64 CPU with AVX2:

```
=== secp256k1 Field Multiplication Benchmark ===

--- Benchmark (10000000 iterations) ---
Scalar (4x sequential): 1.234 sec, 32.41 M mul/sec
AVX2 (4-way parallel):  0.456 sec, 87.72 M mul/sec

Speedup: 2.71x
```

**Expected speedup: 2-3x** (not 4x due to overhead and memory bandwidth)

## How It Works

### Traditional (Scalar) Approach
```
for each field element:
    r = a * b mod p    // One at a time
```

### AVX2 Limb-Slicing Approach
```
Pack 4 elements' limbs into YMM registers:
    ymm0 = [a0.limb0, a1.limb0, a2.limb0, a3.limb0]
    ymm1 = [a0.limb1, a1.limb1, a2.limb1, a3.limb1]
    ...

Compute 4 multiplications simultaneously:
    vpmuludq ymm_r0, ymm_a0, ymm_b0  // 4 partial products at once
    ...

Unpack results back to 4 field elements
```

## Limitations

This PoC uses a **simplified reduction** that may not produce bit-exact results compared to the scalar version. A production implementation would need:

1. Full 128-bit intermediate products
2. Proper carry propagation across all limbs
3. Complete modular reduction

## Next Steps

1. Implement full `fe_mul_x4` with 128-bit intermediates
2. Add `point_add_x4` for parallel EC point addition
3. Add `batch_inv` for Montgomery's batch inversion
4. Integrate with Go via CGO bindings

## References

- [libsecp256k1](https://github.com/bitcoin-core/secp256k1) - Bitcoin Core's secp256k1 library
- [AVXECC](https://github.com/hchengv/avxecc) - AVX2 elliptic curve library (SAC 2020)


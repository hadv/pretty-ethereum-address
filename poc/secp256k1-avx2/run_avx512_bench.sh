#!/bin/bash
# Run AVX-512 IFMA benchmark on a cloud instance
#
# Usage:
#   1. Copy this directory to an AVX-512 capable machine:
#      scp -r poc/secp256k1-avx2 user@server:~/
#
#   2. SSH into the machine and run:
#      cd ~/secp256k1-avx2 && ./run_avx512_bench.sh
#
# Supported cloud instances:
#   AWS:   c6i.*, c7i.*, m6i.*, r6i.* (Ice Lake / Sapphire Rapids)
#   GCP:   n2-* with Intel Ice Lake
#   Azure: Dv5, Ev5 series

set -e

echo "=== secp256k1 AVX-512 IFMA Benchmark Setup ==="
echo

# Check for AVX-512 IFMA
if grep -q avx512ifma /proc/cpuinfo 2>/dev/null; then
    echo "✅ AVX-512 IFMA detected!"
else
    echo "❌ AVX-512 IFMA NOT detected"
    echo "CPU flags:"
    grep -o 'avx[^ ]*' /proc/cpuinfo 2>/dev/null | sort -u || echo "Unable to check"
    echo
    echo "This benchmark requires AVX-512 IFMA support."
    echo "Use Intel Ice Lake, Tiger Lake, Alder Lake, Sapphire Rapids, or AMD Zen 4"
    exit 1
fi

# Compile
echo
echo "Compiling with AVX-512 IFMA support..."
gcc -O3 -march=native -mavx512f -mavx512ifma -o bench_avx512 bench_avx512.c

# Also compile AVX2 version for comparison
echo "Compiling AVX2 version for comparison..."
gcc -O3 -march=native -mavx2 -o bench bench.c

echo
echo "=== Running AVX2 Benchmark (4-way) ==="
./bench

echo
echo "=== Running AVX-512 IFMA Benchmark (8-way) ==="
./bench_avx512

echo
echo "=== CPU Information ==="
lscpu | grep -E "Model name|CPU\(s\)|MHz|cache" | head -10


# secp256k1-avx2 Proof of Concept Makefile

CC = gcc
CFLAGS = -O3 -march=native -mavx2 -Wall -Wextra

# For macOS with Apple Silicon (cross-compile to x86_64)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
    CC = clang
    CFLAGS = -O3 -target x86_64-apple-macos10.15 -mavx2 -Wall -Wextra
    # Note: Running on ARM requires Rosetta 2
endif

# For Linux
ifeq ($(shell uname),Linux)
    CC = gcc
    CFLAGS = -O3 -march=native -mavx2 -Wall -Wextra
endif

HEADERS = field.h field_mul.h field_mul_avx2.h

all: bench

bench: bench.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ bench.c

run: bench
	./bench

clean:
	rm -f bench

# Debug build with symbols
debug: CFLAGS = -O0 -g -mavx2 -Wall -Wextra
debug: bench

# Check if AVX2 is supported
check-avx2:
	@echo "Checking AVX2 support..."
ifeq ($(UNAME_M),arm64)
	@echo "Running on ARM64 (Apple Silicon)"
	@echo "Cross-compiling to x86_64 - requires Rosetta 2 to run"
else
	@if grep -q avx2 /proc/cpuinfo 2>/dev/null || sysctl -a 2>/dev/null | grep -q AVX2; then \
		echo "AVX2 is supported"; \
	else \
		echo "WARNING: AVX2 may not be supported on this CPU"; \
	fi
endif

.PHONY: all run clean debug check-avx2


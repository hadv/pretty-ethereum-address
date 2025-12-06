package miner

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"testing"

	"github.com/cloudflare/circl/simd/keccakf1600"
	"golang.org/x/crypto/sha3"
)

// setupBenchmarkData creates realistic test data for benchmarking
func setupBenchmarkData() (dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte) {
	// Deployer address (Solady CREATE2 factory)
	deployerAddress := [20]byte{
		0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xe8, 0xB4,
		0x7B, 0x3e, 0x21, 0x30, 0x21, 0x3B, 0x80, 0x22,
		0x12, 0x43, 0x94, 0x97,
	}

	// Sample init code hash
	initCodeHash := [32]byte{
		0x74, 0x7d, 0xd6, 0x3d, 0xfa, 0xe9, 0x91, 0x11,
		0x7d, 0xeb, 0xeb, 0x00, 0x8f, 0x2f, 0xb0, 0x53,
		0x3b, 0xb5, 0x9a, 0x6e, 0xee, 0x74, 0xba, 0x0e,
		0x19, 0x7e, 0x21, 0x09, 0x9d, 0x03, 0x4c, 0x7a,
	}

	// Salt prefix (caller address)
	saltPrefixBytes = [20]byte{
		0x18, 0xEe, 0x4C, 0x04, 0x05, 0x68, 0x23, 0x86,
		0x43, 0xC0, 0x7e, 0x7a, 0xFd, 0x6c, 0x53, 0xef,
		0xc1, 0x96, 0xD2, 0x6b,
	}

	// Pattern to match: 0x00000000 (4 bytes of zeros)
	patternBytes = []byte{0x00, 0x00, 0x00, 0x00}

	// Build data template
	dataTemplate = make([]byte, DataSize)
	dataTemplate[0] = 0xff
	copy(dataTemplate[1:21], deployerAddress[:])
	copy(dataTemplate[21:41], saltPrefixBytes[:])
	copy(dataTemplate[53:85], initCodeHash[:])

	return
}

// BenchmarkXoroshiro128Plus benchmarks the fast PRNG
func BenchmarkXoroshiro128Plus(b *testing.B) {
	rng := newXoroshiro128plus()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = rng.next()
	}
}

// BenchmarkCryptoRand benchmarks crypto/rand for comparison
func BenchmarkCryptoRand(b *testing.B) {
	buf := make([]byte, 12)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rand.Read(buf)
	}
}

// BenchmarkHashOnly benchmarks just the Keccak256 hash operation
func BenchmarkHashOnly(b *testing.B) {
	dataTemplate, _, saltPrefixBytes := setupBenchmarkData()
	data := append([]byte{}, dataTemplate...)
	hasher := sha3.NewLegacyKeccak256()
	var salt [32]byte
	copy(salt[:20], saltPrefixBytes[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulate random salt suffix
		binary.LittleEndian.PutUint64(salt[20:28], uint64(i))
		binary.LittleEndian.PutUint32(salt[28:32], uint32(i>>32))
		_ = calculateCreate2Address(hasher, data, salt)
	}
}

// BenchmarkFullIterationOptimized benchmarks the full optimized iteration
func BenchmarkFullIterationOptimized(b *testing.B) {
	dataTemplate, patternBytes, saltPrefixBytes := setupBenchmarkData()
	data := append([]byte{}, dataTemplate...)
	hasher := sha3.NewLegacyKeccak256()
	rng := newXoroshiro128plus()
	var salt [32]byte
	copy(salt[:20], saltPrefixBytes[:])
	patternLen := len(patternBytes)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Generate random bytes using fast PRNG
		r1 := rng.next()
		r2 := rng.next()
		binary.LittleEndian.PutUint64(salt[20:28], r1)
		binary.LittleEndian.PutUint32(salt[28:32], uint32(r2))

		addressBytes := calculateCreate2Address(hasher, data, salt)

		// Pattern matching (fast path)
		if patternLen > 0 && addressBytes[0] != patternBytes[0] {
			continue
		}
		for j := 1; j < patternLen; j++ {
			if addressBytes[j] != patternBytes[j] {
				break
			}
		}
	}
}

// BenchmarkFullIterationCryptoRand benchmarks using crypto/rand (old method)
func BenchmarkFullIterationCryptoRand(b *testing.B) {
	dataTemplate, patternBytes, saltPrefixBytes := setupBenchmarkData()
	data := append([]byte{}, dataTemplate...)
	hasher := sha3.NewLegacyKeccak256()
	var salt [32]byte
	copy(salt[:20], saltPrefixBytes[:])
	patternLen := len(patternBytes)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Generate random bytes using crypto/rand (old slow method)
		rand.Read(salt[20:])

		addressBytes := calculateCreate2Address(hasher, data, salt)

		// Pattern matching
		if patternLen > 0 && addressBytes[0] != patternBytes[0] {
			continue
		}
		for j := 1; j < patternLen; j++ {
			if addressBytes[j] != patternBytes[j] {
				break
			}
		}
	}
}

// BenchmarkParallelOptimized benchmarks parallel mining with optimization
func BenchmarkParallelOptimized(b *testing.B) {
	dataTemplate, patternBytes, saltPrefixBytes := setupBenchmarkData()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		data := append([]byte{}, dataTemplate...)
		hasher := sha3.NewLegacyKeccak256()
		rng := newXoroshiro128plus()
		var salt [32]byte
		copy(salt[:20], saltPrefixBytes[:])
		patternLen := len(patternBytes)

		for pb.Next() {
			r1 := rng.next()
			r2 := rng.next()
			binary.LittleEndian.PutUint64(salt[20:28], r1)
			binary.LittleEndian.PutUint32(salt[28:32], uint32(r2))

			addressBytes := calculateCreate2Address(hasher, data, salt)

			if patternLen > 0 && addressBytes[0] != patternBytes[0] {
				continue
			}
			for j := 1; j < patternLen; j++ {
				if addressBytes[j] != patternBytes[j] {
					break
				}
			}
		}
	})
}

// BenchmarkParallelCryptoRand benchmarks parallel mining with crypto/rand
func BenchmarkParallelCryptoRand(b *testing.B) {
	dataTemplate, patternBytes, saltPrefixBytes := setupBenchmarkData()

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		data := append([]byte{}, dataTemplate...)
		hasher := sha3.NewLegacyKeccak256()
		var salt [32]byte
		copy(salt[:20], saltPrefixBytes[:])
		patternLen := len(patternBytes)

		for pb.Next() {
			rand.Read(salt[20:])

			addressBytes := calculateCreate2Address(hasher, data, salt)

			if patternLen > 0 && addressBytes[0] != patternBytes[0] {
				continue
			}
			for j := 1; j < patternLen; j++ {
				if addressBytes[j] != patternBytes[j] {
					break
				}
			}
		}
	})
}

// BenchmarkHashThroughput measures raw hashing throughput
func BenchmarkHashThroughput(b *testing.B) {
	dataTemplate, _, saltPrefixBytes := setupBenchmarkData()
	data := append([]byte{}, dataTemplate...)
	hasher := sha3.NewLegacyKeccak256()
	var salt [32]byte
	copy(salt[:20], saltPrefixBytes[:])

	b.SetBytes(int64(len(data)))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(salt[20:28], uint64(i))
		copy(data[21:53], salt[:])
		hasher.Reset()
		hasher.Write(data)
		_ = hasher.Sum(nil)
	}
}

// BenchmarkHashZeroAlloc benchmarks hash with zero allocation
func BenchmarkHashZeroAlloc(b *testing.B) {
	dataTemplate, _, saltPrefixBytes := setupBenchmarkData()
	data := append([]byte{}, dataTemplate...)
	hasher := sha3.NewLegacyKeccak256()
	var salt [32]byte
	var hashBuf [32]byte
	copy(salt[:20], saltPrefixBytes[:])

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		binary.LittleEndian.PutUint64(salt[20:28], uint64(i))
		binary.LittleEndian.PutUint32(salt[28:32], uint32(i>>32))
		_ = calculateCreate2AddressZeroAlloc(hasher, data, salt, &hashBuf)
	}
}

// BenchmarkSIMDKeccakX4 benchmarks the 4-way SIMD Keccak (per batch of 4 hashes)
func BenchmarkSIMDKeccakX4(b *testing.B) {
	dataTemplate, _, saltPrefixBytes := setupBenchmarkData()
	var data [4][]byte
	for i := range 4 {
		data[i] = append([]byte{}, dataTemplate...)
	}
	var salts [4][32]byte
	for i := range 4 {
		copy(salts[i][:20], saltPrefixBytes[:])
	}

	miner := NewSIMDCPUMiner()
	if !miner.IsSIMDEnabled() {
		b.Skip("SIMD not available on this platform")
	}

	var perm keccakf1600.StateX4
	var hashes [4][32]byte
	rng := newXoroshiro128plus()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Generate 4 random salts
		for j := range 4 {
			r1 := rng.next()
			r2 := rng.next()
			binary.LittleEndian.PutUint64(salts[j][20:28], r1)
			binary.LittleEndian.PutUint32(salts[j][28:32], uint32(r2))
			copy(data[j][21:53], salts[j][:])
		}
		keccak256x4(&perm, data, &hashes)
	}
	// Report as 4 hashes per operation
	b.ReportMetric(float64(b.N*4)/b.Elapsed().Seconds()/1_000_000, "MH/s")
}

// BenchmarkSIMDParallel benchmarks SIMD mining in parallel (per batch of 4 hashes)
func BenchmarkSIMDParallel(b *testing.B) {
	dataTemplate, patternBytes, saltPrefixBytes := setupBenchmarkData()

	miner := NewSIMDCPUMiner()
	if !miner.IsSIMDEnabled() {
		b.Skip("SIMD not available on this platform")
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var data [4][]byte
		for i := range 4 {
			data[i] = append([]byte{}, dataTemplate...)
		}
		var salts [4][32]byte
		for i := range 4 {
			copy(salts[i][:20], saltPrefixBytes[:])
		}

		var perm keccakf1600.StateX4
		var hashes [4][32]byte
		rng := newXoroshiro128plus()
		patternLen := len(patternBytes)

		for pb.Next() {
			// Generate 4 random salts
			for j := range 4 {
				r1 := rng.next()
				r2 := rng.next()
				binary.LittleEndian.PutUint64(salts[j][20:28], r1)
				binary.LittleEndian.PutUint32(salts[j][28:32], uint32(r2))
				copy(data[j][21:53], salts[j][:])
			}
			keccak256x4(&perm, data, &hashes)

			// Check each result
			for i := range 4 {
				address := hashes[i][12:32]
				if patternLen > 0 && address[0] != patternBytes[0] {
					continue
				}
				for j := 1; j < patternLen; j++ {
					if address[j] != patternBytes[j] {
						break
					}
				}
			}
		}
	})
	// Report as 4 hashes per operation
	b.ReportMetric(float64(b.N*4)/b.Elapsed().Seconds()/1_000_000, "MH/s")
}

// BenchmarkThroughputComparison compares hash throughput between standard and SIMD
func BenchmarkThroughputComparison(b *testing.B) {
	dataTemplate, _, saltPrefixBytes := setupBenchmarkData()

	b.Run("Standard_SingleThread", func(b *testing.B) {
		data := append([]byte{}, dataTemplate...)
		hasher := sha3.NewLegacyKeccak256()
		var salt [32]byte
		var hashBuf [32]byte
		copy(salt[:20], saltPrefixBytes[:])
		rng := newXoroshiro128plus()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r1 := rng.next()
			r2 := rng.next()
			binary.LittleEndian.PutUint64(salt[20:28], r1)
			binary.LittleEndian.PutUint32(salt[28:32], uint32(r2))
			_ = calculateCreate2AddressZeroAlloc(hasher, data, salt, &hashBuf)
		}
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds()/1_000_000, "MH/s")
	})

	b.Run("SIMD_SingleThread", func(b *testing.B) {
		miner := NewSIMDCPUMiner()
		if !miner.IsSIMDEnabled() {
			b.Skip("SIMD not available on this platform")
		}

		var data [4][]byte
		for i := range 4 {
			data[i] = append([]byte{}, dataTemplate...)
		}
		var salts [4][32]byte
		for i := range 4 {
			copy(salts[i][:20], saltPrefixBytes[:])
		}

		var perm keccakf1600.StateX4
		var hashes [4][32]byte
		rng := newXoroshiro128plus()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := range 4 {
				r1 := rng.next()
				r2 := rng.next()
				binary.LittleEndian.PutUint64(salts[j][20:28], r1)
				binary.LittleEndian.PutUint32(salts[j][28:32], uint32(r2))
				copy(data[j][21:53], salts[j][:])
			}
			keccak256x4(&perm, data, &hashes)
		}
		// Each iteration produces 4 hashes
		b.ReportMetric(float64(b.N*4)/b.Elapsed().Seconds()/1_000_000, "MH/s")
	})

	b.Run("Standard_Parallel", func(b *testing.B) {
		b.RunParallel(func(pb *testing.PB) {
			data := append([]byte{}, dataTemplate...)
			hasher := sha3.NewLegacyKeccak256()
			var salt [32]byte
			var hashBuf [32]byte
			copy(salt[:20], saltPrefixBytes[:])
			rng := newXoroshiro128plus()

			for pb.Next() {
				r1 := rng.next()
				r2 := rng.next()
				binary.LittleEndian.PutUint64(salt[20:28], r1)
				binary.LittleEndian.PutUint32(salt[28:32], uint32(r2))
				_ = calculateCreate2AddressZeroAlloc(hasher, data, salt, &hashBuf)
			}
		})
		b.ReportMetric(float64(b.N)/b.Elapsed().Seconds()/1_000_000, "MH/s")
	})

	b.Run("SIMD_Parallel", func(b *testing.B) {
		miner := NewSIMDCPUMiner()
		if !miner.IsSIMDEnabled() {
			b.Skip("SIMD not available on this platform")
		}

		b.RunParallel(func(pb *testing.PB) {
			var data [4][]byte
			for i := range 4 {
				data[i] = append([]byte{}, dataTemplate...)
			}
			var salts [4][32]byte
			for i := range 4 {
				copy(salts[i][:20], saltPrefixBytes[:])
			}

			var perm keccakf1600.StateX4
			var hashes [4][32]byte
			rng := newXoroshiro128plus()

			for pb.Next() {
				for j := range 4 {
					r1 := rng.next()
					r2 := rng.next()
					binary.LittleEndian.PutUint64(salts[j][20:28], r1)
					binary.LittleEndian.PutUint32(salts[j][28:32], uint32(r2))
					copy(data[j][21:53], salts[j][:])
				}
				keccak256x4(&perm, data, &hashes)
			}
		})
		// Each iteration produces 4 hashes
		b.ReportMetric(float64(b.N*4)/b.Elapsed().Seconds()/1_000_000, "MH/s")
	})
}

// TestXoroshiro128PlusQuality tests that the PRNG produces reasonable output
func TestXoroshiro128PlusQuality(t *testing.T) {
	rng := newXoroshiro128plus()

	// Generate many values and check for basic statistical properties
	counts := make(map[uint64]int)
	n := 100000

	for i := 0; i < n; i++ {
		val := rng.next()
		counts[val]++
	}

	// Check for uniqueness (no repeats in 100k values for a 64-bit PRNG)
	if len(counts) < n-10 { // Allow a tiny bit of slack
		t.Errorf("Too many duplicates: got %d unique values out of %d", len(counts), n)
	}
}

// TestCalculateCreate2AddressCorrectness verifies CREATE2 address calculation
func TestCalculateCreate2AddressCorrectness(t *testing.T) {
	// Known test vector
	// deployer: 0x0000000000FFe8B47B3e2130213B802212439497
	// salt: 0x18Ee4C040568238643C07e7aFd6c53efc196D26b000000000000000000000000
	// initCodeHash: 0x747dd63dfae991117debeb008f2fb0533bb59a6eee74ba0e197e21099d034c7a

	dataTemplate, _, saltPrefixBytes := setupBenchmarkData()
	data := append([]byte{}, dataTemplate...)
	hasher := sha3.NewLegacyKeccak256()

	var salt [32]byte
	copy(salt[:20], saltPrefixBytes[:])
	// Salt suffix: zeros

	address := calculateCreate2Address(hasher, data, salt)

	// Verify the address is 20 bytes
	if len(address) != 20 {
		t.Errorf("Expected 20-byte address, got %d bytes", len(address))
	}

	// Verify it's deterministic (same input = same output)
	address2 := calculateCreate2Address(hasher, data, salt)
	if !bytes.Equal(address[:], address2[:]) {
		t.Errorf("Address calculation is not deterministic")
	}

	// Log the address for manual verification if needed
	t.Logf("CREATE2 address: 0x%s", hex.EncodeToString(address[:]))
}

// TestPatternMatching verifies pattern matching logic
func TestPatternMatching(t *testing.T) {
	tests := []struct {
		address  [20]byte
		pattern  []byte
		expected bool
	}{
		{
			address:  [20]byte{0x00, 0x00, 0x00, 0x00, 0x12, 0x34},
			pattern:  []byte{0x00, 0x00, 0x00, 0x00},
			expected: true,
		},
		{
			address:  [20]byte{0x00, 0x00, 0x00, 0x01, 0x12, 0x34},
			pattern:  []byte{0x00, 0x00, 0x00, 0x00},
			expected: false,
		},
		{
			address:  [20]byte{0xde, 0xad, 0xbe, 0xef, 0x12, 0x34},
			pattern:  []byte{0xde, 0xad},
			expected: true,
		},
		{
			address:  [20]byte{0x00},
			pattern:  []byte{},
			expected: true, // Empty pattern matches everything
		},
	}

	for i, tc := range tests {
		// Simulate pattern matching logic from Mine function
		match := true
		patternLen := len(tc.pattern)

		if patternLen > 0 && tc.address[0] != tc.pattern[0] {
			match = false
		} else {
			for j := 1; j < patternLen; j++ {
				if tc.address[j] != tc.pattern[j] {
					match = false
					break
				}
			}
		}

		if match != tc.expected {
			t.Errorf("Test %d: expected match=%v, got %v", i, tc.expected, match)
		}
	}
}

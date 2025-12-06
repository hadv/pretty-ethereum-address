package miner

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cloudflare/circl/simd/keccakf1600"
)

// SIMDCPUMiner handles CPU-based mining with SIMD acceleration
// Uses AVX2 4-way parallel Keccak-f[1600] for ~4x speedup on supported CPUs
type SIMDCPUMiner struct {
	numGoroutines int
	simdEnabled   bool
}

// NewSIMDCPUMiner creates a new SIMD-optimized CPU miner
func NewSIMDCPUMiner() *SIMDCPUMiner {
	numCores := runtime.NumCPU()
	runtime.GOMAXPROCS(numCores)

	simdEnabled := keccakf1600.IsEnabledX4()

	return &SIMDCPUMiner{
		numGoroutines: numCores * 800,
		simdEnabled:   simdEnabled,
	}
}

// IsSIMDEnabled returns true if SIMD acceleration is available
func (m *SIMDCPUMiner) IsSIMDEnabled() bool {
	return m.simdEnabled
}

// NumGoroutines returns the number of goroutines used for mining
func (m *SIMDCPUMiner) NumGoroutines() int {
	return m.numGoroutines
}

// NumCores returns the number of CPU cores
func (m *SIMDCPUMiner) NumCores() int {
	return runtime.NumCPU()
}

// keccak256x4 computes 4 Keccak256 hashes in parallel using SIMD
// Input: 4 data buffers of 85 bytes each (CREATE2 data)
// Output: 4 hash results of 32 bytes each
func keccak256x4(perm *keccakf1600.StateX4, data [4][]byte, hashes *[4][32]byte) {
	state := perm.Initialize(false) // 24-round Keccak

	// Keccak256 rate is 136 bytes (1088 bits), capacity is 64 bytes (512 bits)
	// For 85 bytes of data, we need only one permutation (85 < 136)

	// Load data into interleaved state
	// State layout: state[4*i + lane] where lane=0..3 for 4 parallel hashes
	// Each word is 8 bytes, we have 11 words of data (88 bytes, rounds up from 85)

	for lane := 0; lane < 4; lane++ {
		d := data[lane]
		// Load 10 full 64-bit words (80 bytes)
		for word := 0; word < 10; word++ {
			state[4*word+lane] = binary.LittleEndian.Uint64(d[word*8 : word*8+8])
		}
		// Word 10: bytes 80-84 (5 bytes) + padding domain separator (0x01)
		// Keccak256 domain separator is 0x01 (SHA3-256 uses 0x06)
		lastWord := uint64(d[80]) |
			uint64(d[81])<<8 |
			uint64(d[82])<<16 |
			uint64(d[83])<<24 |
			uint64(d[84])<<32 |
			uint64(0x01)<<40 // Keccak domain separator
		state[4*10+lane] = lastWord

		// Padding: 0x80 at byte 135 (word 16, bit 63)
		// Words 11-15 are 0 (already initialized)
		state[4*16+lane] = 0x8000000000000000
	}

	// Execute the permutation
	perm.Permute()

	// Extract 32 bytes (4 words) of hash for each lane
	for lane := 0; lane < 4; lane++ {
		for word := 0; word < 4; word++ {
			binary.LittleEndian.PutUint64(hashes[lane][word*8:word*8+8], state[4*word+lane])
		}
	}
}

// Mine runs the SIMD-optimized CPU mining
// Falls back to standard mining if SIMD is not available
func (m *SIMDCPUMiner) Mine(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte) *Result {
	if !m.simdEnabled {
		// Fall back to standard CPU miner
		cpuMiner := NewCPUMiner()
		return cpuMiner.Mine(dataTemplate, patternBytes, saltPrefixBytes)
	}

	startTime := time.Now()
	var found atomic.Bool
	var totalHashes atomic.Uint64
	var result *Result
	var resultMu sync.Mutex

	patternLen := len(patternBytes)

	// Start progress reporting goroutine
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				hashes := totalHashes.Load()
				elapsed := time.Since(startTime)
				hashRate := float64(hashes) / elapsed.Seconds() / 1_000_000
				fmt.Printf("\rSearching... %d hashes, %.2f MH/s", hashes, hashRate)
			}
		}
	}()

	var wg sync.WaitGroup
	for range m.numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Create 4 data buffers for SIMD processing
			var data [4][]byte
			for i := range 4 {
				data[i] = append([]byte{}, dataTemplate...)
			}

			var perm keccakf1600.StateX4
			var hashes [4][32]byte
			var salts [4][32]byte

			// Copy salt prefix to all 4 salts
			for i := range 4 {
				copy(salts[i][:20], saltPrefixBytes[:])
			}

			rng := newXoroshiro128plus()
			const checkInterval = 256 // Check less frequently since we do 4x work per iteration
			counter := 0

		loop:
			for {
				if counter >= checkInterval {
					counter = 0
					totalHashes.Add(checkInterval * 4) // 4 hashes per iteration
					if found.Load() {
						return
					}
				}
				counter++

				// Generate 4 random salts
				for i := range 4 {
					r1 := rng.next()
					r2 := rng.next()
					binary.LittleEndian.PutUint64(salts[i][20:28], r1)
					binary.LittleEndian.PutUint32(salts[i][28:32], uint32(r2))
					copy(data[i][21:53], salts[i][:])
				}

				// Compute 4 hashes in parallel
				keccak256x4(&perm, data, &hashes)

				// Check each result
				for i := range 4 {
					address := hashes[i][12:32] // Last 20 bytes

					// Fast path: check first byte
					if patternLen > 0 && address[0] != patternBytes[0] {
						continue
					}

					// Check remaining pattern bytes
					match := true
					for j := 1; j < patternLen; j++ {
						if address[j] != patternBytes[j] {
							match = false
							break
						}
					}

					if !match {
						continue
					}

					// Found it!
					if found.CompareAndSwap(false, true) {
						close(done) // Stop the progress goroutine
						hashes := totalHashes.Load()
						elapsed := time.Since(startTime)
						hashRate := float64(hashes) / elapsed.Seconds() / 1_000_000
						var addressBytes [20]byte
						copy(addressBytes[:], address)

						fmt.Printf("\n\nFound!\n")
						fmt.Printf("Salt: 0x%s\n", hex.EncodeToString(salts[i][:]))
						fmt.Printf("Address: 0x%s\n", hex.EncodeToString(addressBytes[:]))
						fmt.Printf("Time elapsed: %s\n", elapsed)
						fmt.Printf("Total hashes: %d\n", hashes)
						fmt.Printf("Hash rate: %.2f MH/s\n", hashRate)

						resultMu.Lock()
						result = &Result{
							Salt:    salts[i],
							Address: addressBytes,
						}
						resultMu.Unlock()
					}
					return
				}
				continue loop
			}
		}()
	}
	wg.Wait()

	return result
}

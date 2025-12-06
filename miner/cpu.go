package miner

import (
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/crypto/sha3"
)

// xoroshiro128plus is a fast PRNG suitable for non-cryptographic purposes
// We use it for salt generation since we only need random-looking values
type xoroshiro128plus struct {
	s0, s1 uint64
}

// newXoroshiro128plus creates a new PRNG seeded from crypto/rand
func newXoroshiro128plus() *xoroshiro128plus {
	var seed [16]byte
	rand.Read(seed[:])
	return &xoroshiro128plus{
		s0: binary.LittleEndian.Uint64(seed[:8]),
		s1: binary.LittleEndian.Uint64(seed[8:]),
	}
}

func rotl(x uint64, k int) uint64 {
	return (x << k) | (x >> (64 - k))
}

func (r *xoroshiro128plus) next() uint64 {
	s0 := r.s0
	s1 := r.s1
	result := s0 + s1

	s1 ^= s0
	r.s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16)
	r.s1 = rotl(s1, 37)

	return result
}

// CPUMiner handles CPU-based mining operations
type CPUMiner struct {
	numGoroutines int
}

// NewCPUMiner creates a new CPU miner
func NewCPUMiner() *CPUMiner {
	numCores := runtime.NumCPU()
	runtime.GOMAXPROCS(numCores)
	return &CPUMiner{
		numGoroutines: numCores * 800,
	}
}

// NumGoroutines returns the number of goroutines used for mining
func (m *CPUMiner) NumGoroutines() int {
	return m.numGoroutines
}

// NumCores returns the number of CPU cores
func (m *CPUMiner) NumCores() int {
	return runtime.NumCPU()
}

// calculateCreate2Address computes the CREATE2 address
// address = keccak256(0xff ++ deployer_address ++ salt ++ keccak256(init_code))[12:]
// Returns the 20-byte address directly
// The data buffer is reused and only the salt portion (bytes 21-53) needs to be updated
// The hasher is reused to avoid allocations in the hot loop
func calculateCreate2Address(hasher hash.Hash, data []byte, salt [32]byte) [20]byte {
	// Update only the salt portion (bytes 21-53)
	copy(data[21:53], salt[:])

	// Hash using reusable hasher
	hasher.Reset()
	hasher.Write(data)
	hashBytes := hasher.Sum(nil)

	// Take last 20 bytes for address
	var result [20]byte
	copy(result[:], hashBytes[12:32])
	return result
}

// calculateCreate2AddressZeroAlloc computes the CREATE2 address with zero allocations
// Uses a pre-allocated hashBuf to avoid the Sum() allocation in the hot loop
func calculateCreate2AddressZeroAlloc(hasher hash.Hash, data []byte, salt [32]byte, hashBuf *[32]byte) [20]byte {
	// Update only the salt portion (bytes 21-53)
	copy(data[21:53], salt[:])

	// Hash using reusable hasher with pre-allocated buffer
	hasher.Reset()
	hasher.Write(data)
	hasher.Sum(hashBuf[:0]) // Append to our buffer, avoiding allocation

	// Take last 20 bytes for address
	var result [20]byte
	copy(result[:], hashBuf[12:32])
	return result
}

// Mine runs the CPU-based mining with goroutines
// Returns the result when found
func (m *CPUMiner) Mine(dataTemplate []byte, patternBytes []byte, saltPrefixBytes [20]byte) *Result {
	startTime := time.Now()
	var found atomic.Bool
	var totalHashes atomic.Uint64

	var result *Result
	var resultMu sync.Mutex

	// Pre-compute pattern length for optimization
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
			data := append([]byte{}, dataTemplate...)
			hasher := sha3.NewLegacyKeccak256()

			// Use fast PRNG instead of crypto/rand for ~10x speedup
			rng := newXoroshiro128plus()

			var salt [32]byte
			var hashBuf [32]byte // Pre-allocated hash buffer for zero-alloc hashing
			copy(salt[:20], saltPrefixBytes[:])

			// Counter for batch checking the found flag (reduces atomic operations)
			const checkInterval = 1024
			counter := 0

		loop:
			for {
				// Check found flag less frequently to reduce atomic contention
				if counter >= checkInterval {
					counter = 0
					totalHashes.Add(checkInterval)
					if found.Load() {
						return
					}
				}
				counter++

				// Generate 12 random bytes using fast PRNG (2 calls for 16 bytes, use 12)
				r1 := rng.next()
				r2 := rng.next()
				// Fill salt[20:32] with random bytes
				binary.LittleEndian.PutUint64(salt[20:28], r1)
				binary.LittleEndian.PutUint32(salt[28:32], uint32(r2))

				addressBytes := calculateCreate2AddressZeroAlloc(hasher, data, salt, &hashBuf)

				// Fast path: check first byte (most likely to fail)
				if patternLen > 0 && addressBytes[0] != patternBytes[0] {
					continue loop
				}

				// Check remaining bytes
				match := true
				for i := 1; i < patternLen; i++ {
					if addressBytes[i] != patternBytes[i] {
						match = false
						break
					}
				}

				if !match {
					continue loop
				}

				// Found it!
				if found.CompareAndSwap(false, true) {
					close(done) // Stop the progress goroutine
					hashes := totalHashes.Load()
					elapsed := time.Since(startTime)
					hashRate := float64(hashes) / elapsed.Seconds() / 1_000_000
					fmt.Printf("\n\nFound!\n")
					fmt.Printf("Salt: 0x%s\n", hex.EncodeToString(salt[:]))
					fmt.Printf("Address: 0x%s\n", hex.EncodeToString(addressBytes[:]))
					fmt.Printf("Time elapsed: %s\n", elapsed)
					fmt.Printf("Total hashes: %d\n", hashes)
					fmt.Printf("Hash rate: %.2f MH/s\n", hashRate)

					resultMu.Lock()
					result = &Result{
						Salt:    salt,
						Address: addressBytes,
					}
					resultMu.Unlock()
				}
				return
			}
		}()
	}
	wg.Wait()

	return result
}
